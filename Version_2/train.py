"""
train.py

Entry point for Phase 4: loads config, builds the env + model + risk
pipeline, runs training/ppo_hybrid.py's rollout/GAE/update loop, and
checkpoints periodically.

Assumption flagged up front: this file imports `MultiTickerRolloutDataset`
from `dataset.py`, which was never shared with the assistant that wrote
this. The constructor call below (`MultiTickerRolloutDataset(split=...,
tickers=..., window_size=..., device=...)`) is a best-guess against the
attributes every other file in this project already relies on
(window_size, n_envs, tickers, feature_names, aligned_dates, device,
__len__, __getitem__) -- adjust the call to match your actual signature if
it differs; nothing else here depends on the constructor's exact argument
names.

This file NEVER imports monitoring.dashboard's Rich-dependent pieces --
only MetricsWriter, which is dependency-light and crash-isolated from
training by design (see monitoring/dashboard.py's module docstring). If you
want a live view while training, run
`python main.py monitor --metrics-path <path>` in a second terminal/cell;
that process is the one that imports Rich.
"""

import argparse
import os
import sys
from typing import List, Optional

import torch

# Force line-buffered stdout. When this script runs as a subprocess (e.g. a
# Kaggle/Jupyter `!python train.py` cell), stdout is a pipe, not a real
# terminal -- CPython fully-buffers a piped stdout by default, so every
# print() below sits in an internal buffer and never reaches the notebook
# cell until that buffer fills (a few KB) or the process exits. That's what
# "no output on the terminal" during a long training run almost always is,
# not a hang. reconfigure(line_buffering=True) forces each print() to flush
# immediately instead, regardless of how this script is invoked. (Available
# on Python 3.7+; the hasattr guard is defensive, not because this project
# targets anything older.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# vec_trading_env.py / portfolio_state.py live in env/, not project root, and
# (unlike model/, risk/, training/, eval/, monitoring/) that folder isn't a
# dotted package import -- this project's convention treats it as a flat
# sibling-import directory instead (see vec_trading_env.py's own internal
# sys.path handling for paths.py/execution_sim.py/portfolio_state.py). That
# only works once env/ is actually on sys.path, which nothing does
# automatically for a plain `python train.py` invocation -- add it here,
# once, at the actual entrypoint.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

from dataset import MultiTickerRolloutDataset  # noqa: E402 -- see module docstring's assumption note
from paths import is_kaggle
from vec_trading_env import VecTradingEnv

from training.config import TrainingConfig
from training.ppo_hybrid import HybridActorCritic, collect_rollout, compute_gae, ppo_update
from training.reward import DifferentialSharpeReward

from risk.kelly_sizing import KellySizer
from risk.kill_switch import KillSwitch
from risk.risk_manager import RiskLimits, RiskManager

from monitoring.dashboard import MetricsWriter


def build_risk_pipeline(cfg: TrainingConfig, n_envs: int, device: torch.device):
    """
    Shared construction, kept in one place because train.py,
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
    """
    argv=None (the default) reads sys.argv[1:] as normal -- correct for a
    real `python train.py ...` shell invocation.

    Pass an explicit list (e.g. [] or ["--total-rollouts", "2000"]) when
    calling main() in-process from a notebook cell instead of via a shell
    command. Notebook kernels (Jupyter/Colab/Kaggle) run this Python
    process with their OWN launch arguments in sys.argv (typically
    `-f /path/to/kernel-xxxx.json`) -- if main() reads sys.argv by default
    in that context, argparse chokes on the kernel's own flags with
    "unrecognized arguments: -f ...". Passing argv explicitly sidesteps
    sys.argv entirely, which is the robust fix -- monkeypatching
    `sys.argv = [...]` before calling main() also works, but is fragile
    (easy to do in the wrong cell, or have it silently overwritten).
    """
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
        # keep the default relative path off Kaggle's read-only working dir
        # root when it wasn't explicitly overridden by the caller
        cfg.run.checkpoint_dir = "/kaggle/working/checkpoints"

    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.run.seed)

    os.makedirs(cfg.run.checkpoint_dir, exist_ok=True)

    # --- dataset + env -- see module docstring's assumption note. Actual
    # MultiTickerRolloutDataset signature (verified against dataset.py):
    # (window_size, split='train', device=None) -- it does NOT take a
    # `tickers` kwarg, since it auto-discovers tickers from whichever
    # *_features.parquet files exist alongside metadata.json (see
    # preprocess.py). cfg.env.tickers still drives fetch_alpaca.py's
    # TICKERS list upstream; it's just not a dataset constructor argument.
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

    # --- model + optimizer
    actor_critic = HybridActorCritic(n_features=len(train_dataset.feature_names), cfg=cfg).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=cfg.ppo.learning_rate, eps=cfg.ppo.adam_eps)

    start_rollout = 0
    best_metric = float("-inf")
    ema_reward = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        actor_critic.load_state_dict(checkpoint["actor_critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_rollout = checkpoint["rollout_idx"] + 1
        # .get() with a default so resuming from a checkpoint saved BEFORE
        # best-tracking existed doesn't crash -- it just restarts "best"
        # tracking from scratch instead of carrying over unknown state.
        best_metric = checkpoint.get("best_metric", float("-inf"))
        ema_reward = checkpoint.get("ema_reward", None)
        print(f"[train] resumed from {args.resume} at rollout {start_rollout} "
              f"(best_metric so far: {best_metric if best_metric != float('-inf') else 'none yet'})")

    # --- risk pipeline + reward shaper
    kelly_sizer, risk_manager, kill_switch = build_risk_pipeline(cfg, env.n_envs, device)
    reward_shaper = DifferentialSharpeReward(
        n_envs=env.n_envs,
        eta=cfg.reward.dsr_eta,
        eps=cfg.reward.dsr_eps,
        warmup_steps=cfg.reward.dsr_warmup_steps,
        clip=cfg.reward.dsr_clip,
        device=str(device),
    )

    # --- metrics (structured log only -- see module docstring)
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
            # end of a full pass through the training split -- start a fresh
            # pass (see training/ppo_hybrid.py's collect_rollout() docstring:
            # resetting is a session-boundary decision made HERE, not inside
            # the rollout collector).
            obs = env.reset()
            hidden = actor_critic.init_hidden(env.n_envs, device)
            kelly_sizer.reset()
            reward_shaper.reset()
            kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001

        rollout_reward_mean = buffer.reward.mean().item()
        alpha = cfg.run.best_metric_ema_alpha
        ema_reward = rollout_reward_mean if ema_reward is None else (
            alpha * rollout_reward_mean + (1 - alpha) * ema_reward
        )

        if rollout_idx % cfg.run.log_every_n_rollouts == 0:
            metrics_writer.log(
                step=rollout_idx,
                episode=rollout_idx,
                reward=rollout_reward_mean,
                reward_ema=ema_reward,
                sharpe=None,  # per-rollout Sharpe isn't meaningful over 256 steps; see eval/metrics.py for the real thing at eval time
                drawdown=None,
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
        }

        if rollout_idx % cfg.run.checkpoint_every_n_rollouts == 0:
            checkpoint_path = os.path.join(cfg.run.checkpoint_dir, f"checkpoint_{rollout_idx}.pt")
            torch.save(checkpoint_state, checkpoint_path)
            print(f"[train] rollout {rollout_idx}: checkpoint saved to {checkpoint_path}")

        if rollout_idx >= cfg.run.best_metric_warmup_rollouts and ema_reward > best_metric:
            best_metric = ema_reward
            checkpoint_state["best_metric"] = best_metric  # keep the saved dict's own record in sync
            best_path = os.path.join(cfg.run.checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint_state, best_path)
            print(f"[train] rollout {rollout_idx}: new best (EMA reward {best_metric:.6f}) -> {best_path}")

    metrics_writer.close()


if __name__ == "__main__":
    main()