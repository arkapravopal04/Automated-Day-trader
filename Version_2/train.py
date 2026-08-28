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
from typing import Any, Dict, List, Optional

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

from dataset import MultiTickerRolloutDataset  # noqa: E402
from paths import is_kaggle  # noqa: E402
from env.vec_trading_env import VecTradingEnv, StepResult  # noqa: E402

from training.config import TrainingConfig  # noqa: E402
from training.ppo_hybrid import (  # noqa: E402
    AdaptiveEntropyCoef, HybridActorCritic, collect_rollout, compute_gae,
    load_actor_critic_state, ppo_update,
)
from training.reward import DifferentialSharpeReward  # noqa: E402

from risk.kelly_sizing import KellyDiagnostics, KellySizer  # noqa: E402
from risk.kill_switch import KillSwitch  # noqa: E402
from risk.risk_manager import RiskLimits, RiskManager  # noqa: E402

from monitoring.dashboard import MetricsWriter  # noqa: E402

# Rewards are bounded by the DSR clip (|D| <= dsr_clip=10) plus the capped
# checkpoint bonus (<= ~4), so any EMA/best-metric value far above this is
# the pre-fix reward-explosion signature (observed ~830-1096). Used by the
# resume auto-heal in train.py and train_ddp.py.
_REWARD_SANITY_BOUND = 50.0


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
        # TRAINING-ONLY exploration floor -- see RiskConfig.kelly_min_fraction
        # and kelly_sizing.py's __init__ docstring. This function is called
        # only from train.py / train_ddp.py / hyperparam_sweep.py;
        # eval/backtest_report.py and live/live_loop.py construct their own
        # KellySizer without it and keep the strict zero-floor behavior.
        min_fraction=cfg.risk.kelly_min_fraction,
        edge_source=cfg.risk.kelly_edge_source,
        enabled=cfg.risk.kelly_enabled_in_training,
        device=str(device),
    )
    risk_manager = RiskManager(
        RiskLimits(
            max_position_frac=cfg.risk.max_position_frac,
            max_gross_exposure_frac=cfg.risk.max_gross_exposure_frac,
            max_ticker_concentration_frac=cfg.risk.max_ticker_concentration_frac,
            max_order_notional_frac=cfg.risk.max_order_notional_frac,
            drawdown_halt_frac=cfg.risk.drawdown_halt_frac,
            min_order_notional=cfg.risk.min_order_notional,
            min_order_equity_frac=cfg.risk.min_order_equity_frac,
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


def kelly_metrics_fields(diag: KellyDiagnostics) -> Dict[str, Any]:
    """
    Builds the Kelly-sizing diagnostic fields for one "rollout" metrics
    record, from a KellySizer.diagnostics() snapshot -- same _per_ticker-list
    + scalar-aggregate shape as the existing fields on that record (e.g.
    total_trades_per_ticker / total_trades). One shared function so
    train.py and train_ddp.py can't drift on what these fields mean (see
    train_ddp.py's "reuse, don't duplicate" imports from this module).

    win_rate/payoff_ratio/raw_kelly aggregates are averaged over is_warm
    streams only -- the estimate is meaningless pre-warm (see
    kelly_sizing.py's _edge_estimate()). fractional_kelly's aggregate and
    kelly_zero_count cover every stream regardless of warm status, since
    that's the value actually capping (or not capping) orders right now,
    including the kelly_default_fraction pre-warm streams get.

    Added to trace exactly which rollout each stream's fractional_kelly
    first lands on 0.0 -- see this project's diagnostic history: a stream
    sitting flat with fractional_kelly==0.0 has every opening order capped
    to 0 shares, which means it can never close a trade to update the PnL
    history that estimate is built from, so the 0.0 is permanent for the
    rest of the run once it happens. kelly_zero_count climbing rollout over
    rollout, in step with total_trades going flat, is the signature to
    watch for.
    """
    is_warm = diag.is_warm
    n_warm = int(is_warm.sum().item())

    def _warm_mean(x: torch.Tensor) -> Optional[float]:
        return float(x[is_warm].mean().item()) if n_warm > 0 else None

    return {
        "kelly_win_rate_per_ticker": diag.win_rate.tolist(),
        "kelly_payoff_ratio_per_ticker": diag.payoff_ratio.tolist(),
        "kelly_raw_per_ticker": diag.raw_kelly.tolist(),
        "kelly_fractional_per_ticker": diag.fractional_kelly.tolist(),
        "kelly_is_warm_per_ticker": is_warm.tolist(),
        "kelly_win_rate": _warm_mean(diag.win_rate),
        "kelly_payoff_ratio": _warm_mean(diag.payoff_ratio),
        "kelly_raw": _warm_mean(diag.raw_kelly),
        "kelly_fractional": float(diag.fractional_kelly.mean().item()),
        "kelly_warm_count": n_warm,
        "kelly_zero_count": int((diag.fractional_kelly == 0.0).sum().item()),
    }


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


def make_tick_callback(
    env: VecTradingEnv, metrics_writer: MetricsWriter, state: _TickState, log_every_n_ticks: int = 2
):
    """
    Returns a closure matching training/ppo_hybrid.py's ``collect_rollout``
    tick-callback signature.

    Every real environment step advances ``global_tick`` and trade counters.
    Only the JSONL write is throttled by ``log_every_n_ticks``; the
    ``record_type="tick"`` field lets ``MetricsWriter`` route the record to
    its bounded tick log. ``fsync=False`` is intentional because tick
    telemetry is disposable compared with rollout/checkpoint state.
    """

    if log_every_n_ticks < 0:
        raise ValueError("log_every_n_ticks must be >= 0")

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

        # Counters above always advance; the write below is throttled.
        # log_every_n_ticks <= 1 means "log every tick" (max resolution).
        if log_every_n_ticks > 1 and state.global_tick % log_every_n_ticks != 0:
            return

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
            price_per_ticker=mid_price_now.tolist(),
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
    parser.add_argument(
        "--pretrained-trunk", type=str, default=None,
        help="Checkpoint from training/pretrain_trunk.py. Loads the shared "
             "cnn/lstm/cross_attn/fusion stack and leaves the policy and critic "
             "heads randomly initialised. Ignored when --resume is given.")
    parser.add_argument("--kaggle", action="store_true", help="Force Kaggle-safe paths/checkpointing.")
    parser.add_argument("--local", action="store_true", help="Force local paths/checkpointing.")
    parser.add_argument("--total-rollouts", type=int, default=None, help="Override cfg.run.total_rollouts.")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint path to resume from, or 'latest' (highest checkpoint_N.pt) / "
             "'best' (checkpoint_best.pt). Mutually exclusive with --fresh.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Force a cold start from random weights: deletes any existing *.pt in the "
             "checkpoint dir first. Mutually exclusive with --resume. If NEITHER --resume "
             "nor --fresh is given and checkpoints exist, training refuses to start rather "
             "than guess (pass --resume latest to continue, or --fresh to reset).",
    )
    parser.add_argument(
        "--tick-log-every-n-ticks", type=int, default=None,
        help="Override cfg.run.tick_log_every_n_ticks. 1 = log every env-step (max dashboard resolution).",
    )
    return parser.parse_args(argv)


def _list_checkpoints(checkpoint_dir: str) -> List[str]:
    """*.pt files in checkpoint_dir, sorted by name (rollout-index order for checkpoint_N.pt)."""
    if not os.path.isdir(checkpoint_dir):
        return []
    return sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".pt"))


def _resolve_checkpoint_alias(alias: str, checkpoint_dir: str) -> Optional[str]:
    """Resolve 'latest' / 'best' to a concrete checkpoint path, or None."""
    if alias == "best":
        path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        return path if os.path.exists(path) else None
    # 'latest': highest rollout index among checkpoint_<N>.pt
    best_file, best_idx = None, -1
    for fname in _list_checkpoints(checkpoint_dir):
        if fname.startswith("checkpoint_") and fname.endswith(".pt"):
            try:
                idx = int(fname[len("checkpoint_"):-3])
            except ValueError:
                continue
            if idx > best_idx:
                best_idx, best_file = idx, fname
    return os.path.join(checkpoint_dir, best_file) if best_file is not None else None


def _clean_checkpoint_dir(checkpoint_dir: str) -> None:
    """Delete this run's own checkpoints -- used ONLY by an explicit --fresh.

    NARROWED to `checkpoint_*.pt`, which is every file train.py and
    train_ddp.py write (`checkpoint_<N>.pt` and `checkpoint_best.pt`) and
    nothing else. It used to delete every `*.pt` in the directory, which was
    fine while this run was the only thing that put a .pt there and became a
    trap the moment `training/pretrain_trunk.py` existed: a
    `--fresh --pretrained-trunk` invocation deleted the trunk during startup
    and then failed to load the file it had just removed. Artefacts that are
    INPUTS to a run must survive a reset of that run's own outputs.
    """
    removed = []
    for fname in _list_checkpoints(checkpoint_dir):
        if not fname.startswith("checkpoint_"):
            continue
        os.remove(os.path.join(checkpoint_dir, fname))
        removed.append(fname)
    if removed:
        print(f"[train] --fresh: deleted existing checkpoint(s): {', '.join(removed)}")


def resolve_resume_path(
    resume_arg: Optional[str],
    checkpoint_dir: str,
    fresh: bool,
) -> Optional[str]:
    """
    Decides, ONCE, whether this run continues from a checkpoint or starts
    fresh -- the user's explicit choice, never a silent default:

      --resume <path>     -> that exact checkpoint (must exist)
      --resume latest     -> highest checkpoint_<N>.pt in checkpoint_dir
      --resume best       -> checkpoint_best.pt in checkpoint_dir
      --fresh             -> cold reset (random weights); deletes existing *.pt
      neither, no ckpts   -> fresh start (first run -- fine)
      neither, ckpts exist -> RuntimeError: refuse to guess

    Returns the concrete path to load, or None for a fresh start.
    """
    if resume_arg is not None and fresh:
        raise RuntimeError("--resume and --fresh are mutually exclusive -- pick one.")
    if resume_arg is None:
        existing = _list_checkpoints(checkpoint_dir)
        if not existing:
            return None
        if fresh:
            _clean_checkpoint_dir(checkpoint_dir)
            return None
        raise RuntimeError(
            f"checkpoints found in {checkpoint_dir}: {', '.join(existing)}. "
            "Refusing to guess what you want: pass --resume <path|latest|best> to "
            "continue, or --fresh to reset to random weights."
        )
    if resume_arg in ("latest", "best"):
        resolved = _resolve_checkpoint_alias(resume_arg, checkpoint_dir)
        if resolved is None:
            raise RuntimeError(f"--resume {resume_arg}: no matching checkpoint in {checkpoint_dir}.")
        return resolved
    if not os.path.exists(resume_arg):
        raise RuntimeError(f"--resume: checkpoint not found: {resume_arg}")
    return resume_arg


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = TrainingConfig()

    if args.total_rollouts is not None:
        cfg.run.total_rollouts = args.total_rollouts
    if args.tick_log_every_n_ticks is not None:
        cfg.run.tick_log_every_n_ticks = args.tick_log_every_n_ticks

    on_kaggle = args.kaggle or (is_kaggle() and not args.local)
    if on_kaggle and cfg.run.checkpoint_dir == "checkpoints":
        cfg.run.checkpoint_dir = "/kaggle/working/checkpoints"
    # Same redirect for metrics: without it, metrics_path stays relative
    # ("logs/metrics.jsonl") and resolves against the CWD -- which on Kaggle
    # is the cloned repo dir, NOT /kaggle/working/logs as run_kaggle.py
    # documents and its summary checks. Files would still land under
    # /kaggle/working only as long as the repo happens to be cloned there,
    # and would crash outright if the repo were a read-only /kaggle/input
    # attachment. The `== default` guard keeps an explicit --metrics-path
    # override (or train_ddp.py's metrics_path arg) intact.
    if on_kaggle and cfg.run.metrics_path == "logs/metrics.jsonl":
        cfg.run.metrics_path = "/kaggle/working/logs/metrics.jsonl"

    device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.run.seed)

    os.makedirs(cfg.run.checkpoint_dir, exist_ok=True)

    # Explicit continue-vs-reset decision (--resume / --fresh / refuse-to-guess).
    resume_path = resolve_resume_path(args.resume, cfg.run.checkpoint_dir, args.fresh)
    if resume_path is not None:
        print(f"[train] resuming from {resume_path}")
    else:
        print("[train] fresh start (random weights)")

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
        execution_price_column=cfg.env.execution_price_column,
        r_step_scale=cfg.env.r_step_scale,
        hold_loser_penalty=cfg.env.hold_loser_penalty,
        enable_mirroring=cfg.env.enable_mirroring,
        mirror_prob=cfg.env.mirror_prob,
        overtrade_window=cfg.env.overtrade_window,
        overtrade_free_trades=cfg.env.overtrade_free_trades,
        overtrade_penalty_coef=cfg.reward.overtrade_penalty_coef,
        bias_window=cfg.env.bias_window,
        diversity_bonus_coef=cfg.env.diversity_bonus_coef,
        trade_cooldown_bars=cfg.env.trade_cooldown_bars,
        min_hold_bars=cfg.env.min_hold_bars,
        trading_window=cfg.env.trading_window,
        flatten_at_session_close=cfg.env.flatten_at_session_close,
        device=str(device),
    )

    # env.feature_names, not train_dataset.feature_names: VecTradingEnv appends
    # portfolio-state channels to every observation (see
    # _augment_obs_with_portfolio_state), so the dataset's own count is
    # narrower than what the network actually receives.
    actor_critic = HybridActorCritic(n_features=len(env.feature_names), cfg=cfg).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=cfg.ppo.learning_rate, eps=cfg.ppo.adam_eps)

    start_rollout = 0
    best_metric = float("-inf")
    ema_reward = None
    state = _TickState(n_envs=env.n_envs)

    if resume_path is None and args.pretrained_trunk:
        # P2 bullet 3: attach a freshly-initialised policy and critic to a trunk
        # that was already fitted, supervised, against the tradeable forward
        # return (`training/pretrain_trunk.py`). Only the shared feature stack
        # is loaded -- the policy and critic heads stay random on purpose, and
        # the edge head does not exist in this model at all.
        #
        # Deliberately mutually exclusive with --resume: a resumed checkpoint
        # already contains a trunk that PPO has been updating, and overwriting
        # it mid-run with the supervised one would silently discard however many
        # rollouts of learning while leaving the optimizer state that was fitted
        # to it in place.
        blob = torch.load(args.pretrained_trunk, map_location=device, weights_only=True)
        trunk_sd = dict(blob.get("trunk", blob))
        missing, unexpected = actor_critic.load_state_dict(trunk_sd, strict=False)
        if unexpected:
            raise SystemExit(
                f"[train] --pretrained-trunk {args.pretrained_trunk} carries keys this "
                f"model does not have: {sorted(unexpected)[:6]}. Was it written by a "
                "different model config?"
            )
        # Everything except the trunk is SUPPOSED to be missing: the policy and
        # critic are attached fresh, and the edge head is loaded separately
        # below because it is optional.
        fresh = ("policy_head.", "critic_head.", "edge_head.", "edge_scale_bps")
        unloaded = [k for k in missing if not k.startswith(fresh)]
        if unloaded:
            raise SystemExit(
                f"[train] --pretrained-trunk {args.pretrained_trunk} is missing trunk "
                f"weights this model needs: {sorted(unloaded)[:6]}"
            )

        # The edge head and its scale travel together or not at all -- a head
        # loaded without the sd it was standardised against emits a number in
        # arbitrary units that KellySizer would read as bps.
        if "edge_head" in blob and "target_sd_bps" in blob:
            actor_critic.edge_head.load_state_dict(blob["edge_head"])
            actor_critic.edge_scale_bps.fill_(float(blob["target_sd_bps"]))
            have_edge_head = True
        else:
            have_edge_head = False

        print(f"[train] loaded pre-trained trunk from {args.pretrained_trunk}: "
              f"{len(trunk_sd)} tensors (hold {blob.get('hold', '?')} bars, "
              f"val IC {blob.get('val_ic', float('nan')):+.5f}); "
              f"policy and critic heads are fresh"
              + ("; edge head loaded, scale "
                 f"{float(actor_critic.edge_scale_bps):.2f} bps" if have_edge_head
                 else "; NO edge head in this checkpoint"))
        if cfg.risk.kelly_edge_source == "model" and not have_edge_head:
            raise SystemExit(
                "[train] risk.kelly_edge_source == 'model' but the checkpoint carries "
                "no edge head. Sizing would fall back to a constant fraction while "
                "reporting itself as Kelly. Re-run training/pretrain_trunk.py, or set "
                "kelly_edge_source back to 'realized'."
            )
    elif cfg.risk.kelly_edge_source == "model" and resume_path is None:
        raise SystemExit(
            "[train] risk.kelly_edge_source == 'model' needs a supervised edge head. "
            "Pass --pretrained-trunk <checkpoint from training/pretrain_trunk.py>."
        )

    if resume_path is not None:
        # weights_only=True -- see main.py's matching comment (checkpoints
        # are pickle files; arbitrary-object unpickling is RCE).
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        resumed_edge_head = load_actor_critic_state(
            actor_critic, checkpoint["actor_critic"], resume_path
        )
        if cfg.risk.kelly_edge_source == "model" and not resumed_edge_head:
            raise SystemExit(
                "[train] risk.kelly_edge_source == 'model' but the resumed checkpoint "
                "carries no edge head, so the governor would size on a randomly "
                "initialised one. Set kelly_edge_source to 'realized', or start from "
                "--pretrained-trunk instead of --resume."
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_rollout = checkpoint["rollout_idx"] + 1
        best_metric = checkpoint.get("best_metric", float("-inf"))
        ema_reward = checkpoint.get("ema_reward", None)
        # AUTO-HEAL for the pre-fix reward explosion (observed EMA ~830-1096
        # vs. legit values bounded by the DSR clip, |reward| <= ~14). If a
        # checkpoint carries that signature, reset both the EMA and the
        # best-metric baseline so tracking works again; a poisoned baseline
        # would otherwise make checkpoint_best.pt unreachable forever.
        if best_metric > _REWARD_SANITY_BOUND:
            print(f"[train] WARNING: loaded best_metric {best_metric:.4f} is the pre-fix "
                  "reward-explosion signature -- resetting best-metric tracking")
            best_metric = float("-inf")
        if ema_reward is not None and ema_reward > _REWARD_SANITY_BOUND:
            print(f"[train] WARNING: loaded ema_reward {ema_reward:.4f} is the pre-fix "
                  "reward-explosion signature -- restarting the EMA from scratch")
            ema_reward = None
        state.global_tick = checkpoint.get("global_tick", 0)
        state.episode_idx = checkpoint.get("episode_idx", 0)
        saved_total = checkpoint.get("total_trades_per_ticker", None)
        if isinstance(saved_total, list) and len(saved_total) == env.n_envs:
            state.total_trades_per_ticker = saved_total
        print(f"[train] resumed from {resume_path} at rollout {start_rollout} "
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

    # MetricsWriter routes record_type="tick" records to its bounded tick log
    # automatically; rollout records stay in cfg.run.metrics_path. Keep the
    # construction here backwards-compatible and let the writer own routing
    # and rotation policy/defaults. tick_max_bytes wires up
    # cfg.run.max_tick_log_bytes (the tick-log rotation threshold) -- it was
    # previously dead config that MetricsWriter's own 8MB default ignored.
    metrics_writer = MetricsWriter(
        cfg.run.metrics_path,
        tick_max_bytes=cfg.run.max_tick_log_bytes,
        tick_backup_count=cfg.run.tick_backup_count,
    )
    tick_callback = make_tick_callback(env, metrics_writer, state, log_every_n_ticks=cfg.run.tick_log_every_n_ticks)

    # Created ONCE, outside the rollout loop -- see ppo_hybrid.py's
    # ppo_update() docstring on why a fresh scaler per call would defeat
    # its own loss-scale adaptation. enabled=False when cfg.run.use_amp is
    # off makes every scaler method a no-op, so this is safe to always
    # construct and pass regardless of the flag.
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.run.use_amp)
    if resume_path is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])


    # Entropy collapse guard. Created ONCE outside the loop and checkpointed:
    # KellySizer's un-checkpointed state is exactly what let the Session 1 lock
    # re-arm on every restart, and a controller silently resetting to its
    # initial coefficient on --resume would repeat that.
    entropy_ctl = None
    if cfg.ppo.target_entropy_discrete is not None:
        entropy_ctl = AdaptiveEntropyCoef(
            target=cfg.ppo.target_entropy_discrete,
            init_coef=cfg.ppo.entropy_coef_discrete,
            lr=cfg.ppo.entropy_coef_lr,
            min_coef=cfg.ppo.entropy_coef_min,
            max_coef=cfg.ppo.entropy_coef_max,
            warmup_rollouts=cfg.ppo.entropy_coef_warmup_rollouts,
            lr_up_mult=cfg.ppo.entropy_coef_lr_up_mult,
        )
        if resume_path is not None and "entropy_ctl" in checkpoint:
            entropy_ctl.load_state_dict(checkpoint["entropy_ctl"])
            print(f"[train] resumed entropy controller: coef={entropy_ctl.coef():.4f}")
    # Collapse detector: consecutive rollouts that are BOTH near-zero entropy
    # and near-zero trades. Both conditions, because low entropy alone is a
    # committed policy (fine) and low trades alone can be a cooldown artefact.
    collapse_streak = 0

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
        # Snapshot BEFORE the episode-boundary kelly_sizer.reset() below can
        # run, so this reflects what was actually used for this rollout's
        # own ticks -- not a freshly-reset state that belongs to the next
        # rollout. See risk/kelly_sizing.py's diagnostics() docstring.
        kelly_diag = kelly_sizer.diagnostics()
        compute_gae(buffer, final_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        stats = ppo_update(actor_critic, optimizer, buffer, cfg, scaler=scaler,
                           entropy_ctl=entropy_ctl, rollout_idx=rollout_idx)

        # DRAWDOWN-METRIC FIX: snapshot peak_equity BEFORE either reset
        # branch below can touch it. The periodic branch's
        # reset_peak_equity() overwrites peak_equity to the CURRENT equity,
        # and drawdown_per_ticker below was computed from
        # env.portfolio.peak_equity AFTER that reset already ran -- so it
        # compared equity against a peak just set equal to itself, logging
        # exactly 0.0 every rollout by construction (verified on a
        # 50-rollout run: 50/50 logged drawdown == 0.0 while the tick-level
        # log -- a separate path not subject to this reset -- showed real
        # intra-rollout drawdown up to 1.87%). This snapshot is the peak
        # this rollout's trading actually ran against.
        pre_reset_peak_equity = env.portfolio.peak_equity.clone()

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
        elif (rollout_idx + 1) % cfg.risk.kill_switch_reset_every_n_rollouts == 0:
            # The episode-boundary reset above fires far too rarely to be
            # the ONLY reset during training -- see
            # training/config.py's RiskConfig.kill_switch_reset_every_n_rollouts
            # docstring. `elif`, not a second independent `if`: no point
            # resetting twice in the same rollout when the episode boundary
            # already did it.
            kill_switch.reset()
            kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001
            # Same rationale, different module: risk_manager.py's
            # drawdown_halt_frac compares against portfolio.peak_equity,
            # which is otherwise a true all-time high that never resets on
            # its own -- see PortfolioState.reset_peak_equity()'s docstring
            # for the full story (this is what "stopped trading, episodes
            # just roll by, no errors" actually was).
            env.portfolio.reset_peak_equity(env._current_prices().unsqueeze(1))  # noqa: SLF001

        rollout_reward_mean = buffer.reward.mean().item()
        alpha = cfg.run.best_metric_ema_alpha
        ema_reward = rollout_reward_mean if ema_reward is None else (
            alpha * rollout_reward_mean + (1 - alpha) * ema_reward
        )

        current_prices_unsq = env._current_prices().unsqueeze(1)  # noqa: SLF001
        equity_per_ticker = env.portfolio.equity(current_prices_unsq)
        unrealized_per_ticker = env.portfolio.unrealized_pnl(current_prices_unsq)
        peak = pre_reset_peak_equity.clamp(min=1e-6)
        drawdown_per_ticker = (peak - equity_per_ticker).clamp(min=0.0) / peak
        net_worth = float(equity_per_ticker.sum().item())

        if rollout_idx % cfg.run.log_every_n_rollouts == 0:
            # THE NUMBERS TO WATCH, and they are deliberately logged beside
            # reward_ema rather than beside net_worth.
            #
            # Net worth is a product of edge, size and trade count, so it
            # moves for reasons that say nothing about whether the policy
            # knows anything: a bigger account, a longer rollout or a higher
            # trade rate all move it, and a policy with negative edge can post
            # a rising curve for a long time on a directional tape. These two
            # split the only question that matters into its independent
            # halves -- is there edge (alpha_per_turnover), and does it exceed
            # what collecting it costs (cost_per_turnover) -- denominated per
            # dollar traded, so they are invariant to account size and trade
            # count and directly comparable against each other.
            #
            # Popped inside the log block, not outside it: the accumulators
            # then cover exactly the interval since the previous record, so
            # raising log_every_n_rollouts widens the window rather than
            # discarding the rollouts in between.
            turnover_stats = env.pop_turnover_stats()
            metrics_writer.log(
                step=state.global_tick,
                rollout=rollout_idx,
                episode=state.episode_idx,
                record_type="rollout",
                reward=rollout_reward_mean,
                reward_ema=ema_reward,
                **turnover_stats,
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
                # Overnight carry telemetry. residual_overnight_count should
                # stay at 0 (or move only on zero-volume closing bars, where a
                # forced close genuinely cannot fill). Anything else means
                # positions are surviving the session close.
                forced_flatten_count=int(getattr(env, "forced_flatten_count", 0)),
                residual_overnight_count=int(getattr(env, "residual_overnight_count", 0)),
                tickers=env.tickers,
                position=env.portfolio.positions[:, 0].tolist(),
                **kelly_metrics_fields(kelly_diag),
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
            "entropy_ctl": entropy_ctl.state_dict() if entropy_ctl is not None else None,
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

        # --- collapse detector ------------------------------------------
        # Both conditions, not either: low entropy alone is a committed
        # policy, and low trade count alone can be a trade_cooldown_bars
        # artefact. Together they are the signature the last run held for its
        # final 65 rollouts -- entropy 0.000, 124 trades across all 100
        # streams, 43% of the compute confirming a conclusion already reached.
        if cfg.ppo.collapse_patience_rollouts > 0:
            _h = stats.get("entropy_discrete", float("inf"))
            _tr = int(sum(state.trades_this_rollout_per_ticker))
            if (_h < cfg.ppo.collapse_entropy_threshold
                    and _tr <= cfg.ppo.collapse_trades_threshold):
                collapse_streak += 1
            else:
                collapse_streak = 0
            if collapse_streak >= cfg.ppo.collapse_patience_rollouts:
                print(f"[train] COLLAPSE: entropy_discrete < "
                      f"{cfg.ppo.collapse_entropy_threshold} and <= "
                      f"{cfg.ppo.collapse_trades_threshold} trades for "
                      f"{collapse_streak} consecutive rollouts. Stopping at "
                      f"rollout {rollout_idx} of {cfg.run.total_rollouts}.")
                print("[train] This is a RESULT, not a crash: the policy found "
                      "that trading loses on average and stopped. Check the "
                      "alpha gate (eval/alpha_lab.py) before training again.")
                break

    metrics_writer.close()


if __name__ == "__main__":
    main()