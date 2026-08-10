"""
train.py

Entry point for Phase 4: loads config, builds the env + model + risk
pipeline, runs training/ppo_hybrid.py's rollout/GAE/update loop, and
checkpoints periodically.

Metrics come in two flavors now, both through the same MetricsWriter/
metrics_path, distinguished by a "record_type" field:

    "tick"    -- one record per real env-step (one 5-min market bar
                 processed across all tickers). Written from inside
                 collect_rollout()'s loop via the tick_callback hook
                 training/ppo_hybrid.py exposes for exactly this purpose.
                 Carries live position/equity/drawdown/fills -- no PPO
                 training stats, since those genuinely don't exist at
                 tick resolution (policy_loss etc. are only computed once
                 per full rollout, after ppo_update() runs).
    "rollout" -- one record per cfg.ppo.rollout_length-step rollout (the
                 old, only, granularity before this revision). Carries
                 PPO stats (policy_loss, value_loss, approx_kl, ...) plus
                 the same net-worth/trade summary fields as before.

Three genuinely different counters, previously conflated (both "step" and
"episode" used to just be rollout_idx logged twice under different names):
    global_tick -- real env-steps processed, ever. Increments every tick.
    rollout_idx -- which PPO rollout (cfg.ppo.rollout_length-step batch)
                   we're on. Increments once per collect_rollout() call.
    episode_idx -- how many full passes through the training split have
                   completed (vec_trading_env.py's `done` firing). This is
                   what "episode" means in the normal RL sense; increments
                   far less often than rollout_idx.

This file NEVER imports monitoring.dashboard's Rich-dependent pieces --
only MetricsWriter, which is dependency-light and crash-isolated from
training by design (see monitoring/dashboard.py's module docstring).
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

from dataset import MultiTickerRolloutDataset  # noqa: E402
from paths import is_kaggle  # noqa: E402
from vec_trading_env import VecTradingEnv, StepResult  # noqa: E402

from training.config import TrainingConfig  # noqa: E402
from training.ppo_hybrid import HybridActorCritic, collect_rollout, compute_gae, ppo_update  # noqa: E402
from training.reward import DifferentialSharpeReward  # noqa: E402

from risk.kelly_sizing import KellySizer  # noqa: E402
from risk.kill_switch import KillSwitch  # noqa: E402
from risk.risk_manager import RiskLimits, RiskManager  # noqa: E402

from monitoring.dashboard import MetricsWriter  # noqa: E402


def build_risk_pipeline(cfg: TrainingConfig, n_envs: int, device: torch.device):
    """
    Shared construction, kept in one place because train.py, train_ddp.py,
    eval/backtest_report.py, and live/live_loop.py all build the exact same
    three objects from the exact same cfg.risk fields.
    """
    kelly_sizer = KellySizer(
        n_envs=n_envs,
        lookback_trades=cfg.risk.kelly_lookback_trades,
        min_trades_for_estimate=cfg.risk.kelly_min_trades_for_estimate,
        kelly_multiplier=cfg.risk.kelly_multiplier,
        kelly_cap=cfg.risk.kelly_cap,
        default_fraction=cfg.risk.kelly_default_fraction,
        device=str(device),
    )
    risk_manager = RiskManager(
        RiskLimits(
            max_position_frac=cfg.risk.max_position_frac,
            max_gross_exposure_frac=cfg.risk.max_gross_exposure_frac,
            max_ticker_concentration_frac=cfg.risk.max_ticker_concentration_frac,
            max_order_notional_frac=cfg.risk.max_order_notional_frac,
            drawdown_halt_frac=cfg.risk.drawdown_halt_frac,
        ),
        device=str(device),
    )
    kill_switch = KillSwitch(
        n_envs=n_envs,
        daily_loss_limit_frac=cfg.risk.daily_loss_limit_frac,
        broker_error_streak_limit=cfg.risk.broker_error_streak_limit,
        state_mismatch_tolerance=cfg.risk.state_mismatch_tolerance,
        device=str(device),
    )
    return kelly_sizer, risk_manager, kill_switch


@dataclass
class _TickState:
    """
    Mutable counters threaded into the tick_callback closure below. A plain
    class instead of nonlocal ints because tick_callback is defined once
    but rollout_idx/global_tick/episode_idx all change across calls, and
    Python closures can't reassign an outer int without nonlocal
    boilerplate for every single field -- attribute mutation on a shared
    object is simpler here.
    """

    n_envs: int
    global_tick: int = 0
    episode_idx: int = 0
    rollout_idx: int = 0
    total_trades_per_ticker: List[int] = field(default_factory=list)
    trades_this_rollout_per_ticker: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.total_trades_per_ticker:
            self.total_trades_per_ticker = [0] * self.n_envs
        if not self.trades_this_rollout_per_ticker:
            self.trades_this_rollout_per_ticker = [0] * self.n_envs

    def start_new_rollout(self) -> None:
        self.trades_this_rollout_per_ticker = [0] * self.n_envs


def make_tick_callback(env: VecTradingEnv, metrics_writer: MetricsWriter, state: _TickState):
    """
    Returns a closure matching training/ppo_hybrid.py's collect_rollout()
    tick_callback signature. Logs one "tick" record per real env-step --
    live position/equity/drawdown/fills, no PPO stats (those only exist at
    rollout granularity). fsync=False on every call: see MetricsWriter.log's
    docstring on why paying a disk-sync syscall 256x per rollout instead of
    once is worth avoiding for data this disposable.
    """

    def tick_callback(
        local_t: int,
        step_result: StepResult,
        final_direction: torch.Tensor,
        final_size: torch.Tensor,
        kill_switch: KillSwitch,
    ) -> None:
        state.global_tick += 1

        filled_qty = step_result.info["filled_qty"]
        filled_list = filled_qty.tolist()
        for i, qty in enumerate(filled_list):
            if qty != 0:
                state.trades_this_rollout_per_ticker[i] += 1
                state.total_trades_per_ticker[i] += 1

        equity_per_ticker = step_result.info["equity"]
        drawdown_per_ticker = step_result.info["drawdown"]
        position_per_ticker = step_result.info["position"]

        # unrealized PnL isn't in step_result.info (see vec_trading_env.py's
        # StepResult) -- cheap enough to recompute directly off the
        # portfolio's current mark, same accessor env.step() itself used.
        mid_price_now = env._current_prices()  # noqa: SLF001
        unrealized_per_ticker = env.portfolio.unrealized_pnl(mid_price_now.unsqueeze(1))

        metrics_writer.log(
            step=state.global_tick,
            rollout=state.rollout_idx,
            episode=state.episode_idx,
            record_type="tick",
            tickers=env.tickers,
            position=position_per_ticker.tolist(),
            net_worth=float(equity_per_ticker.sum().item()),
            net_worth_per_ticker=equity_per_ticker.tolist(),
            unrealized_pnl=unrealized_per_ticker.tolist(),
            drawdown=float(drawdown_per_ticker.mean().item()),
            drawdown_per_ticker=drawdown_per_ticker.tolist(),
            filled_qty_this_tick=filled_list,
            trades_this_rollout=int(sum(state.trades_this_rollout_per_ticker)),
            trades_per_ticker_this_rollout=list(state.trades_this_rollout_per_ticker),
            total_trades=int(sum(state.total_trades_per_ticker)),
            total_trades_per_ticker=list(state.total_trades_per_ticker),
            halted=kill_switch.is_halted().tolist(),
            fsync=False,
        )

    return tick_callback


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid PPO trading policy.")
    parser.add_argument("--kaggle", action="store_true", help="Force Kaggle-safe paths/checkpointing.")
    parser.add_argument("--local", action="store_true", help="Force local paths/checkpointing.")
    parser.add_argument("--total-rollouts", type=int, default=None, help="Override cfg.run.total_rollouts.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = TrainingConfig()

    if args.total_rollouts is not None:
        cfg.run.total_rollouts = args.total_rollouts

    on_kaggle = args.kaggle or (is_kaggle() and not args.local)
    if on_kaggle and cfg.run.checkpoint_dir == "checkpoints":
        cfg.run.checkpoint_dir = "/kaggle/working/checkpoints"

    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.run.seed)

    os.makedirs(cfg.run.checkpoint_dir, exist_ok=True)

    train_dataset = MultiTickerRolloutDataset(
        window_size=cfg.env.window_size,
        split="train",
        device=str(device),
    )
    env = VecTradingEnv(
        dataset=train_dataset,
        initial_cash=cfg.env.initial_cash,
        max_position_frac=cfg.env.max_position_frac,
        tick_size=cfg.env.tick_size,
        spread_bps=cfg.env.spread_bps,
        impact_coef=cfg.env.impact_coef,
        max_participation=cfg.env.max_participation,
        commission_per_share=cfg.env.commission_per_share,
        commission_bps=cfg.env.commission_bps,
        min_commission=cfg.env.min_commission,
        platform_fee_per_trade=cfg.env.platform_fee_per_trade,
        r_step_scale=cfg.env.r_step_scale,
        hold_loser_penalty=cfg.env.hold_loser_penalty,
        enable_mirroring=cfg.env.enable_mirroring,
        mirror_prob=cfg.env.mirror_prob,
        overtrade_window=cfg.env.overtrade_window,
        overtrade_free_trades=cfg.env.overtrade_free_trades,
        overtrade_surcharge_bps=cfg.env.overtrade_surcharge_bps,
        bias_window=cfg.env.bias_window,
        diversity_bonus_coef=cfg.env.diversity_bonus_coef,
        device=str(device),
    )

    actor_critic = HybridActorCritic(n_features=len(train_dataset.feature_names), cfg=cfg).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=cfg.ppo.learning_rate, eps=cfg.ppo.adam_eps)

    start_rollout = 0
    best_metric = float("-inf")
    ema_reward = None
    state = _TickState(n_envs=env.n_envs)

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        actor_critic.load_state_dict(checkpoint["actor_critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_rollout = checkpoint["rollout_idx"] + 1
        best_metric = checkpoint.get("best_metric", float("-inf"))
        ema_reward = checkpoint.get("ema_reward", None)
        state.global_tick = checkpoint.get("global_tick", 0)
        state.episode_idx = checkpoint.get("episode_idx", 0)
        saved_total = checkpoint.get("total_trades_per_ticker", None)
        if isinstance(saved_total, list) and len(saved_total) == env.n_envs:
            state.total_trades_per_ticker = saved_total
        print(f"[train] resumed from {args.resume} at rollout {start_rollout} "
              f"(global_tick={state.global_tick}, episode={state.episode_idx}, "
              f"best_metric so far: {best_metric if best_metric != float('-inf') else 'none yet'})")
    kelly_sizer, risk_manager, kill_switch = build_risk_pipeline(cfg, env.n_envs, device)
    reward_shaper = DifferentialSharpeReward(
        n_envs=env.n_envs,
        eta=cfg.reward.dsr_eta,
        eps=cfg.reward.dsr_eps,
        warmup_steps=cfg.reward.dsr_warmup_steps,
        clip=cfg.reward.dsr_clip,
        device=str(device),
    )

    metrics_writer = MetricsWriter(cfg.run.metrics_path)
    tick_callback = make_tick_callback(env, metrics_writer, state)

    # Created ONCE, outside the rollout loop -- see ppo_hybrid.py's
    # ppo_update() docstring on why a fresh scaler per call would defeat
    # its own loss-scale adaptation. enabled=False when cfg.run.use_amp is
    # off makes every scaler method a no-op, so this is safe to always
    # construct and pass regardless of the flag.
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.run.use_amp)
    if args.resume is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    obs = env.reset()
    hidden = actor_critic.init_hidden(env.n_envs, device)
    kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001

    for rollout_idx in range(start_rollout, cfg.run.total_rollouts):
        state.rollout_idx = rollout_idx
        state.start_new_rollout()

        buffer, obs, final_value, hidden = collect_rollout(
            env, actor_critic, kelly_sizer, risk_manager, kill_switch, reward_shaper, obs, hidden, cfg,
            tick_callback=tick_callback,
        )
        compute_gae(buffer, final_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        stats = ppo_update(actor_critic, optimizer, buffer, cfg, scaler=scaler)

        if buffer.done.any():
            # end of a full pass through the training split -- a real
            # episode boundary in the RL sense, not a rollout boundary.
            state.episode_idx += 1
            obs = env.reset()
            hidden = actor_critic.init_hidden(env.n_envs, device)
            kelly_sizer.reset()
            reward_shaper.reset()
            # KillSwitch.reset() clears any halt tripped during the pass
            # that just ended -- see this function's earlier version for
            # the full rationale (training-only departure from
            # kill_switch.py's live-trading "never auto-clear" semantics).
            kill_switch.reset()
            kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001

        rollout_reward_mean = buffer.reward.mean().item()
        alpha = cfg.run.best_metric_ema_alpha
        ema_reward = rollout_reward_mean if ema_reward is None else (
            alpha * rollout_reward_mean + (1 - alpha) * ema_reward
        )

        current_prices_unsq = env._current_prices().unsqueeze(1)  # noqa: SLF001
        equity_per_ticker = env.portfolio.equity(current_prices_unsq)
        unrealized_per_ticker = env.portfolio.unrealized_pnl(current_prices_unsq)
        peak = env.portfolio.peak_equity.clamp(min=1e-6)
        drawdown_per_ticker = (peak - equity_per_ticker).clamp(min=0.0) / peak
        net_worth = float(equity_per_ticker.sum().item())

        if rollout_idx % cfg.run.log_every_n_rollouts == 0:
            metrics_writer.log(
                step=state.global_tick,
                rollout=rollout_idx,
                episode=state.episode_idx,
                record_type="rollout",
                reward=rollout_reward_mean,
                reward_ema=ema_reward,
                sharpe=None,  # per-rollout Sharpe isn't meaningful over 256 steps; see eval/metrics.py for the real thing at eval time
                drawdown=float(drawdown_per_ticker.mean().item()),
                drawdown_per_ticker=drawdown_per_ticker.tolist(),
                net_worth=net_worth,
                net_worth_per_ticker=equity_per_ticker.tolist(),
                unrealized_pnl=unrealized_per_ticker.tolist(),
                trades_this_rollout=int(sum(state.trades_this_rollout_per_ticker)),
                trades_per_ticker_this_rollout=list(state.trades_this_rollout_per_ticker),
                total_trades=int(sum(state.total_trades_per_ticker)),
                total_trades_per_ticker=list(state.total_trades_per_ticker),
                tickers=env.tickers,
                position=env.portfolio.positions[:, 0].tolist(),
                **stats,
            )

        checkpoint_state = {
            "actor_critic": actor_critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rollout_idx": rollout_idx,
            "best_metric": best_metric,
            "ema_reward": ema_reward,
            "global_tick": state.global_tick,
            "episode_idx": state.episode_idx,
            "total_trades_per_ticker": state.total_trades_per_ticker,
            "scaler": scaler.state_dict(),
        }

        if rollout_idx % cfg.run.checkpoint_every_n_rollouts == 0:
            checkpoint_path = os.path.join(cfg.run.checkpoint_dir, f"checkpoint_{rollout_idx}.pt")
            torch.save(checkpoint_state, checkpoint_path)
            print(f"[train] rollout {rollout_idx}: checkpoint saved to {checkpoint_path}")

        if rollout_idx >= cfg.run.best_metric_warmup_rollouts and ema_reward > best_metric:
            best_metric = ema_reward
            checkpoint_state["best_metric"] = best_metric
            best_path = os.path.join(cfg.run.checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint_state, best_path)
            print(f"[train] rollout {rollout_idx}: new best (EMA reward {best_metric:.6f}) -> {best_path}")

    metrics_writer.close()


if __name__ == "__main__":
    main()