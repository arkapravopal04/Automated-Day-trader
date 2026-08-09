"""
train.py

Entry point for Phase 4: loads config, builds the env + model + risk
pipeline, runs training/ppo_hybrid.py's rollout/GAE/update loop, and
checkpoints periodically.

This file NEVER imports monitoring.dashboard's Rich-dependent pieces --
only MetricsWriter, which is dependency-light and crash-isolated from
training by design (see monitoring/dashboard.py's module docstring). If you
want a live view while training, run the dashboard against the same
metrics_path from a separate cell/process; that process is the one that
imports Rich/IPython.
"""

import argparse
import os
import sys
from typing import List, Optional

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

from dataset import MultiTickerRolloutDataset  # noqa: E402
from paths import is_kaggle  # noqa: E402
from vec_trading_env import VecTradingEnv  # noqa: E402

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
    three objects from the exact same cfg.risk fields -- this is the one
    spot to update if RiskConfig ever grows a field the others forget to
    thread through.
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
    total_trades = 0
    total_trades_per_ticker = [0] * env.n_envs
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        actor_critic.load_state_dict(checkpoint["actor_critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_rollout = checkpoint["rollout_idx"] + 1
        best_metric = checkpoint.get("best_metric", float("-inf"))
        ema_reward = checkpoint.get("ema_reward", None)
        total_trades = checkpoint.get("total_trades", 0)
        # .get() with a default + length guard so resuming from a checkpoint
        # saved before per-ticker trade counting existed (or against a
        # different ticker count) doesn't crash -- it just restarts
        # per-ticker tracking from zero rather than carrying over
        # mismatched-length state.
        saved_per_ticker = checkpoint.get("total_trades_per_ticker", None)
        if isinstance(saved_per_ticker, list) and len(saved_per_ticker) == env.n_envs:
            total_trades_per_ticker = saved_per_ticker
        print(f"[train] resumed from {args.resume} at rollout {start_rollout} "
              f"(best_metric so far: {best_metric if best_metric != float('-inf') else 'none yet'}, "
              f"total_trades so far: {total_trades})")

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

    obs = env.reset()
    hidden = actor_critic.init_hidden(env.n_envs, device)
    kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001

    for rollout_idx in range(start_rollout, cfg.run.total_rollouts):
        buffer, obs, final_value, hidden = collect_rollout(
            env, actor_critic, kelly_sizer, risk_manager, kill_switch, reward_shaper, obs, hidden, cfg
        )
        compute_gae(buffer, final_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        stats = ppo_update(actor_critic, optimizer, buffer, cfg)

        if buffer.done.any():
            obs = env.reset()
            hidden = actor_critic.init_hidden(env.n_envs, device)
            kelly_sizer.reset()
            reward_shaper.reset()
            # KillSwitch.reset() clears any halt tripped during the pass
            # that just ended -- a deliberate departure from
            # kill_switch.py's live-trading semantics (halt persists until
            # a human explicitly clears it). During TRAINING, an early halt
            # (near-inevitable: random-init policy + real transaction costs
            # vs a modest per-stream starting balance) would otherwise
            # silently zero out ALL real trading signal for the rest of the
            # run. Confirmed directly: without this, a halt at rollout 0
            # produced exactly 0.0 reward for 150+ subsequent rollouts with
            # zero real fills, while the policy still "trained" on that
            # empty signal.
            kill_switch.reset()
            kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001

        rollout_reward_mean = buffer.reward.mean().item()
        alpha = cfg.run.best_metric_ema_alpha
        ema_reward = rollout_reward_mean if ema_reward is None else (
            alpha * rollout_reward_mean + (1 - alpha) * ema_reward
        )

        # Per-ticker net worth / unrealized PnL / drawdown -- each stream
        # started this episode at cfg.env.initial_cash independently (NOT
        # one shared pool), so these are genuinely per-env numbers, not a
        # single account balance split up after the fact. net_worth (the
        # scalar) is just their sum, kept for the dashboard's header total.
        current_prices_unsq = env._current_prices().unsqueeze(1)  # noqa: SLF001
        equity_per_ticker = env.portfolio.equity(current_prices_unsq)              # [n_envs]
        unrealized_per_ticker = env.portfolio.unrealized_pnl(current_prices_unsq)  # [n_envs]
        # Read-only: portfolio.peak_equity was already advanced inside
        # env.step() via update_drawdown_tracking() -- do NOT call that
        # again here, it would double-advance the peak.
        peak = env.portfolio.peak_equity.clamp(min=1e-6)
        drawdown_per_ticker = (peak - equity_per_ticker).clamp(min=0.0) / peak
        net_worth = float(equity_per_ticker.sum().item())

        # Trade counts, both aggregate (unchanged) and per-ticker (new --
        # this is what lets the dashboard show each env's own trade count
        # updating independently instead of one system-wide number).
        # buffer.filled_qty is [T, n_envs], signed, 0 where nothing filled.
        trades_per_ticker_this_rollout = (buffer.filled_qty != 0).sum(dim=0).tolist()  # length n_envs
        trades_this_rollout = int(sum(trades_per_ticker_this_rollout))
        total_trades += trades_this_rollout
        total_trades_per_ticker = [
            total_trades_per_ticker[i] + trades_per_ticker_this_rollout[i] for i in range(env.n_envs)
        ]

        if rollout_idx % cfg.run.log_every_n_rollouts == 0:
            metrics_writer.log(
                step=rollout_idx,
                episode=rollout_idx,
                reward=rollout_reward_mean,
                reward_ema=ema_reward,
                sharpe=None,  # per-rollout Sharpe isn't meaningful over 256 steps; see eval/metrics.py for the real thing at eval time
                drawdown=float(drawdown_per_ticker.mean().item()),
                drawdown_per_ticker=drawdown_per_ticker.tolist(),
                net_worth=net_worth,
                net_worth_per_ticker=equity_per_ticker.tolist(),
                unrealized_pnl=unrealized_per_ticker.tolist(),
                trades_this_rollout=trades_this_rollout,
                total_trades=total_trades,
                trades_per_ticker_this_rollout=trades_per_ticker_this_rollout,
                total_trades_per_ticker=total_trades_per_ticker,
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
            "total_trades": total_trades,
            "total_trades_per_ticker": total_trades_per_ticker,
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