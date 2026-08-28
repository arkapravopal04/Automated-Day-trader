"""
hyperparam_sweep.py

Local hyperparameter sweep runner -- NOT a Kaggle script, meant to run on
your own machine (CPU or a single local GPU; falls back to CPU
automatically the same way train.py already does).

WHAT THIS ACTUALLY DOES: for each combination of hyperparameter values in
the grid below, it runs the REAL, unmodified training loop -- literally
train.py's main() -- for a short number of rollouts, then reads back that
run's own metrics.jsonl to report whether the run looks healthy (is
entropy collapsing? are trades happening at all? is reward moving?). This
is not a simplified proxy or a separate re-implementation of training; it
drives train.py exactly the way your Kaggle cells do, just with different
config values injected and a short rollout budget so a sweep finishes in a
reasonable time on a laptop.

HOW A HYPERPARAMETER GETS OVERRIDDEN: train.py builds
`cfg = TrainingConfig()` by calling the class directly (imported via `from
training.config import TrainingConfig`). This script monkeypatches the
name `TrainingConfig` INSIDE train's own module namespace (`train.
TrainingConfig = lambda: cfg`) to a zero-arg callable that returns our
pre-built, overridden config -- the same trick your own Cell 4 notebook
snippet already uses for exactly this reason. train.py itself is never
modified.

SPEED: this revision parallelizes the sweep across processes (--parallel,
default = half your logical cores) and strips per-run overhead that a
short sweep doesn't need, without changing what gets measured:
  - tick-level metrics logging is throttled to one write per rollout
    (cfg.run.tick_log_every_n_ticks = total_rollouts). The sweep's
    comparison reads only the per-rollout records, which are unaffected;
    the tick records are still there, just 256x sparser.
  - checkpoint saving is disabled for sweep runs
    (checkpoint_every_n_rollouts and best_metric_warmup_rollouts both set
    to never-fire). No checkpoint, no torch.save serialization cost.
  - each run's stdout/stderr goes to <output_root>/run_<id>/stdout.log
    instead of the shared console, so N parallel runs can't garble each
    other; the parent prints one compact progress line per finished run.
  - each run writes its own <output_root>/run_<id>/result.json; a re-run
    of the sweep reuses any run whose result.json carries a matching grid
    fingerprint (same overrides + same rollout budget) instead of
    re-training it. --clean forces a full redo.
Training dynamics are byte-identical to what this script did before: same
seed, same rollout_length, same ppo_epochs, same 50-rollout budget --
only logging/checkpoint I/O and process topology changed.

PREREQUISITE: this needs REAL local data. Run these two scripts locally
first (same ones the Kaggle cells run) if you haven't already:
    python fetch_alpaca.py
    python preprocess.py
paths.py resolves to <this folder>/data/{parquet,processed} when NOT on
Kaggle -- this script checks that PROCESSED_DIR/metadata.json exists
before doing anything else, and tells you exactly what to run if it
doesn't, rather than failing confusingly deep inside dataset.py.

OUTPUT (see run_sweep()'s docstring for exact file layout):
    sweep_output/manifest.json       -- every swept param: its dotted config
                                         path, which file DEFINES it, which
                                         file(s) CONSUME it, and the values
                                         tried. Printed to console too.
    sweep_output/results.csv         -- one row per run: every overridden
                                         value + the outcome metrics pulled
                                         from that run's own metrics.jsonl.
    sweep_output/run_<id>/           -- that run's own metrics.jsonl,
                                         result.json, and stdout.log,
                                         untouched and inspectable
                                         individually.

SCOPE, stated plainly: a short local sweep (default 50 rollouts/run) is
for RELATIVE comparison between configs -- "did entropy collapse happen
faster or slower here, did trades stop happening here" -- not a substitute
for a full Kaggle training run. Don't pick a "winner" from this and assume
it'll hold at 2000+ rollouts; use it to rule OUT configs that show obvious
early pathology (entropy crashing to ~0 in the first handful of rollouts,
zero trades from rollout 1) before spending real Kaggle GPU time on them.
"""

import argparse
import contextlib
import copy
import csv
import hashlib
import itertools
import json
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

from paths import PROCESSED_DIR  # noqa: E402

import torch  # noqa: E402  (needed in workers to pin per-process thread count)


# --------------------------------------------------------------------------
# Hyperparameter registry -- the "output all the required params and from
# which script" requirement. Every entry here is BOTH the sweep grid AND a
# traceability record: where the field lives, what actually reads it.
# Add to this list to sweep something else; nothing else in this file needs
# to change to support a new param, as long as it's a real dotted path on
# TrainingConfig.
# --------------------------------------------------------------------------

@dataclass
class SweptParam:
    dotted_path: str            # e.g. "ppo.entropy_coef_discrete" -- cfg.ppo.entropy_coef_discrete
    defined_in: str              # file + class where the field is declared
    consumed_by: List[str]       # file(s)/function(s) that actually read this value
    description: str
    values: List[Any]            # the values this sweep will try


HYPERPARAM_REGISTRY: List[SweptParam] = [
    SweptParam(
        dotted_path="ppo.entropy_coef_discrete",
        defined_in="training/config.py -- PPOConfig.entropy_coef_discrete",
        consumed_by=["training/ppo_hybrid.py -- ppo_update() (entropy_bonus term)"],
        description="Entropy bonus weight on the discrete direction head (SHORT/FLAT/LONG). "
                     "Leading suspect for policy collapse to always-FLAT -- see this project's "
                     "diagnostic history: low values let PPO settle on 'never trade' as a locally "
                     "safe optimum against real transaction costs.",
        values=[0.01, 0.02, 0.05, 0.10],   # 0.02 is the current config.py default
    ),
    SweptParam(
        dotted_path="ppo.entropy_coef_continuous",
        defined_in="training/config.py -- PPOConfig.entropy_coef_continuous",
        consumed_by=["training/ppo_hybrid.py -- ppo_update() (entropy_bonus term)",
                      "model/hybrid_policy.py -- HybridPolicyHead (FLAT-masked, see its docstring point 2)"],
        description="Entropy bonus weight on the (FLAT-masked) size/limit_offset Beta heads.",
        values=[0.01, 0.03],   # 0.01 is the current config.py default
    ),
    SweptParam(
        dotted_path="risk.kelly_multiplier",
        defined_in="training/config.py -- RiskConfig.kelly_multiplier",
        consumed_by=["risk/kelly_sizing.py -- KellySizer.apply()"],
        description="Fractional-Kelly scaling (0.25 = quarter-Kelly). A rolling losing streak can "
                     "push the RAW Kelly fraction to exactly 0 (see kelly_sizing.py's "
                     "_edge_estimate() clamp), silently zeroing every exposure-increasing order "
                     "regardless of this multiplier -- included here to see whether a larger "
                     "multiplier changes how often that floor gets hit in practice.",
        values=[0.1, 0.25, 0.5, 1.0],   # 0.25 is the current config.py default
    ),
    SweptParam(
        dotted_path="ppo.learning_rate",
        defined_in="training/config.py -- PPOConfig.learning_rate",
        consumed_by=["train.py -- torch.optim.Adam(..., lr=cfg.ppo.learning_rate)"],
        description="Adam learning rate for the whole actor-critic.",
        values=[1e-4, 3e-4, 1e-3],   # 3e-4 is the current config.py default
    ),
    SweptParam(
        dotted_path="reward.raw_weight",
        defined_in="training/config.py -- RewardConfig.raw_weight",
        consumed_by=["training/ppo_hybrid.py -- shaped_reward = sharpe_weight*DSR + raw_weight*env_reward (line ~333)"],
        description="Weight on vec_trading_env.py's own shaped reward (r_step_scale*vol-normalized step pnl "
                     "- hold_loser_penalty + diversity_bonus + terminal_alpha -- see that file's _step_reward). "
                     "Default 0.0 means training relies ONLY on the Differential-Sharpe term; raising it brings the "
                     "env's hold-penalty and directional-diversity shaping into the signal for the first time. "
                     "COUPLED to env.r_step_scale: that knob scales only the vol-normalized-pnl sub-term, so it is a "
                     "dead knob while raw_weight==0 (it multiplies a term the blend then multiplies by 0). "
                     "PROBE-DRIVEN GRID: the 2026-08-17 probe (sweep_probe.log, 3 rollouts, default config) "
                     "measured mean|raw|=0.00038 vs mean|DSR|=0.0467 (ratio 0.008) -- the raw term is ~125x "
                     "SMALLER than DSR, so the original {0.01, 0.05, 0.2} grid was shifted UP two orders of "
                     "magnitude per the probe rule (ratio << 0.01 => raise). At {1.0, 5.0, 20.0} the raw term "
                     "contributes ~0.8% / 4% / 16% of the DSR magnitude at init-policy scales; if the "
                     "steady-state raw magnitude grows as the policy trades more, Tier 1 can re-probe before "
                     "the coupled r_step_scale arm.",
        values=[0.0, 1.0, 5.0, 20.0],   # 0.0 is the current config.py default; shifted up per probe ratio 0.008
    ),
    SweptParam(
        dotted_path="reward.sharpe_weight",
        defined_in="training/config.py -- RewardConfig.sharpe_weight",
        consumed_by=["training/ppo_hybrid.py -- shaped_reward blend (line ~333)"],
        description="Weight on the Differential-Sharpe term. Held at 1.0 for this sweep; the only non-default cell is "
                     "the pure-raw probe (sharpe_weight=0, raw_weight=1) that tests the opposite extreme.",
        values=[1.0],
    ),
    SweptParam(
        dotted_path="ppo.clip_range",
        defined_in="training/config.py -- PPOConfig.clip_range",
        consumed_by=["training/ppo_hybrid.py -- ppo_update() (ratio clipping)"],
        description="PPO ratio clip range. 0.2 is the standard sweet spot at this batch/epoch count; included for "
                     "completeness but the lowest-leverage knob -- first candidate to drop if the sweep budget shrinks.",
        values=[0.1, 0.2, 0.3],   # 0.2 is the current config.py default
    ),
    SweptParam(
        dotted_path="env.r_step_scale",
        defined_in="training/config.py -- EnvConfig.r_step_scale",
        consumed_by=["env/vec_trading_env.py -- _vol_normalized_step_reward() (scales ONLY the vol-normalized pnl "
                      "sub-term, NOT hold_loser_penalty / diversity_bonus, which are added unscaled)"],
        description="Scales the vol-normalized step-PnL inside the env reward. DEAD while reward.raw_weight==0 (it "
                     "scales a term the blend then multiplies by 0). Swept ONLY in the coupled raw_weight arm of "
                     "Tier 1, AFTER the best raw_weight is known -- deliberately excluded from the Tier 0 screening "
                     "grid to avoid byte-identical runs.",
        values=[0.25, 0.5, 1.0],   # 0.5 is the current config.py default
    ),
]


def _default_grid_dict() -> Dict[str, List[Any]]:
    return {p.dotted_path: p.values for p in HYPERPARAM_REGISTRY}


def _default_tier0_grid() -> List[Dict[str, Any]]:
    """The Tier-0 screening grid this script runs by default: a one-at-a-time
    list, NOT a Cartesian product. Each cell is the baseline plus a SINGLE
    knob moved off its default -- exactly the candidate set Tier 1 would run
    on GPU, here at a short rollout budget to rule out early pathology
    (entropy collapse / zero trades) before spending Kaggle GPU time.

    This matches the design's search strategy (staged one-at-a-time, not
    grid/random) and keeps every run attributable. Two deliberate exclusions:
      - env.r_step_scale is NOT swept here -- it is a dead knob at
        raw_weight==0 (see its SweptParam docstring) and is deferred to the
        coupled raw_weight arm in Tier 1, after the best raw_weight is known.
      - reward.sharpe_weight stays 1.0 everywhere except the pure-raw cell,
        which sets sharpe_weight=0 + raw_weight=1 to probe the opposite
        extreme of the blend.
    """
    return [
        {},                                             # baseline (all defaults)
        {"ppo.entropy_coef_discrete": 0.01},            # entropy_discrete (0.02 default)
        {"ppo.entropy_coef_discrete": 0.05},
        {"ppo.entropy_coef_discrete": 0.10},
        {"ppo.learning_rate": 1e-4},                    # learning_rate (3e-4 default)
        {"ppo.learning_rate": 1e-3},
        {"reward.raw_weight": 1.0},                     # raw_weight (0.0 default) -- probe ratio 0.008 => grid shifted UP 2 orders
        {"reward.raw_weight": 5.0},
        {"reward.raw_weight": 20.0},
        {"reward.raw_weight": 1.0, "reward.sharpe_weight": 0.0},   # pure-raw extreme
        {"risk.kelly_multiplier": 0.1},                 # kelly_multiplier (0.25 default)
        {"risk.kelly_multiplier": 0.5},
        {"risk.kelly_multiplier": 1.0},
        {"ppo.clip_range": 0.1},                        # clip_range (0.2 default)
        {"ppo.clip_range": 0.3},
    ]


# --------------------------------------------------------------------------
# Config override plumbing
# --------------------------------------------------------------------------

def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    """cfg, 'ppo.entropy_coef_discrete', 0.05 -> cfg.ppo.entropy_coef_discrete = 0.05"""
    parts = dotted_path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def build_cfg(overrides: Dict[str, Any]):
    """
    Builds a fresh TrainingConfig() (each call gets its own independent
    sub-dataclass instances, since TrainingConfig's fields use
    default_factory -- no aliasing risk between sweep runs) and applies the
    given {dotted_path: value} overrides on top of the real defaults.
    """
    from training.config import TrainingConfig  # local import -- see module docstring on why
    cfg = TrainingConfig()
    for dotted_path, value in overrides.items():
        _set_nested(cfg, dotted_path, value)
    return cfg


def _grid_fingerprint(overrides: Dict[str, Any], total_rollouts: int) -> str:
    """Hash of the overrides + rollout budget -- used to safely reuse a
    previous run's result.json on re-sweep (only when the grid cell is
    byte-identical; any grid change produces a different fingerprint)."""
    payload = json.dumps(
        {"overrides": {k: str(v) for k, v in sorted(overrides.items())},
         "total_rollouts": total_rollouts},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Running one config through the REAL training loop
# --------------------------------------------------------------------------

def run_one(
    run_id: str,
    overrides: Dict[str, Any],
    total_rollouts: int,
    output_root: str,
) -> Dict[str, Any]:
    """
    Runs train.py's actual main() with the given overrides applied, isolated
    to its own checkpoint_dir/metrics_path under output_root/run_id/, then
    reads that run's own metrics.jsonl back to summarize outcome metrics.

    Restores train.TrainingConfig to the real class in a finally block, so
    a crash in one run can't leave later runs (or anything else importing
    train.py afterward) permanently monkeypatched.

    Sweep-relevant overhead is stripped WITHOUT changing training dynamics
    (see module docstring "SPEED" section): tick logging throttled to once
    per rollout, checkpoint saving disabled. This run's stdout/stderr is
    captured to run_<id>/stdout.log and its summary is written to
    run_<id>/result.json (with a grid fingerprint) so the parent process
    can rebuild results.csv even if it dies mid-sweep.
    """
    import train  # imported here (not at module top) so the monkeypatch below is scoped per-call

    run_dir = os.path.join(output_root, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    stdout_path = os.path.join(run_dir, "stdout.log")

    cfg = build_cfg(overrides)
    cfg.run.total_rollouts = total_rollouts
    cfg.run.checkpoint_dir = checkpoint_dir
    cfg.run.metrics_path = metrics_path
    # Sweep-only I/O cuts -- logging/checkpoint throttles, NOT training
    # semantics: the sweep reads only per-rollout records, so writing tick
    # records once per rollout keeps them present for post-hoc inspection
    # at ~zero cost; disabling all checkpoint saves removes the per-rollout
    # torch.save serialization (rollout 0, and every best-metric improvement
    # after warmup) that a 24-run sweep would otherwise pay for 24x.
    cfg.run.tick_log_every_n_ticks = total_rollouts
    cfg.run.checkpoint_every_n_rollouts = 10**9
    cfg.run.best_metric_warmup_rollouts = 10**9

    original_training_config = train.TrainingConfig
    train.TrainingConfig = lambda: cfg  # see module docstring -- the actual override mechanism

    start = time.time()
    error: Optional[str] = None
    try:
        # stdout.log gets the full train.py console output (dataset loads,
        # per-rollout lines, any traceback) isolated per run -- with N runs
        # in parallel, shared-console output would be unreadable garble.
        with open(stdout_path, "w", buffering=1) as stdout_f:
            with contextlib.redirect_stdout(stdout_f), contextlib.redirect_stderr(stdout_f):
                train.main(argv=["--local"])  # --local: don't let is_kaggle() redirect checkpoint_dir on your machine
    except Exception as e:  # noqa: BLE001 -- a failed sweep run should be recorded, not crash the whole sweep
        error = f"{type(e).__name__}: {e}"
    finally:
        train.TrainingConfig = original_training_config

    wall_seconds = time.time() - start

    result: Dict[str, Any] = {
        "run_id": run_id,
        "wall_seconds": round(wall_seconds, 1),
        "error": error,
        **{f"param.{k}": v for k, v in overrides.items()},
    }
    result.update(_summarize_run_metrics(metrics_path))

    # Per-run result file (with fingerprint) -- lets the sweep parent
    # rebuild results.csv after a crash, and lets a re-run reuse finished
    # cells whose grid hasn't changed.
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(
            {"grid_fingerprint": _grid_fingerprint(overrides, total_rollouts),
             "result": result},
            f, indent=2, default=str,
        )
    return result


def _worker_init(threads_per_worker: int) -> None:
    """Pool initializer: pin this process's torch thread count so N parallel
    workers don't oversubscribe the machine's cores (each worker gets
    cpu_count // n_workers threads by default -- see parse_args)."""
    torch.set_num_threads(max(1, threads_per_worker))


def _worker_run(task: tuple) -> Dict[str, Any]:
    """Top-level (picklable under Windows spawn) wrapper: unpacks the task
    tuple and runs one sweep cell. Returns the result dict."""
    run_id, overrides, total_rollouts, output_root = task
    return run_one(run_id, overrides, total_rollouts, output_root)


def _summarize_run_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Reads back this run's own metrics.jsonl and extracts the fields that
    actually matter for spotting the failure modes this sweep exists to
    catch: entropy collapse and trading stopping. Returns None-filled
    fields (not zeros) if the file is missing/empty, so a crashed run is
    visibly distinguishable from a run that genuinely had zero trades.
    """
    empty = {
        "final_reward_ema": None, "final_entropy_discrete": None,
        "mean_entropy_discrete_last5": None, "final_total_trades": None,
        "trades_in_last_rollout": None, "final_net_worth": None,
    }
    if not os.path.exists(metrics_path):
        return empty

    rollout_records = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type", "rollout") != "tick":
                rollout_records.append(rec)

    if not rollout_records:
        return empty

    last = rollout_records[-1]
    entropy_series = [r.get("entropy_discrete") for r in rollout_records if r.get("entropy_discrete") is not None]
    recent_entropy = entropy_series[-5:] if entropy_series else []

    return {
        "final_reward_ema": last.get("reward_ema"),
        "final_entropy_discrete": last.get("entropy_discrete"),
        "mean_entropy_discrete_last5": (sum(recent_entropy) / len(recent_entropy)) if recent_entropy else None,
        "final_total_trades": last.get("total_trades"),
        "trades_in_last_rollout": last.get("trades_this_rollout"),
        "final_net_worth": last.get("net_worth"),
    }


def probe_reward_scale(n_rollouts: int = 2, output_dir: str = "sweep_output") -> Dict[str, float]:
    """
    Runs the REAL training loop for a few rollouts at the DEFAULT config and
    reports the native per-step magnitudes of the two reward terms that
    training/ppo_hybrid.py blends (shaped = sharpe_weight*DSR + raw_weight*env
    reward). This is the scale probe the sweep design calls for BEFORE
    committing a raw_weight grid: if mean|raw reward| and mean|DSR reward|
    are wildly mismatched (orders of magnitude apart), shift the raw_weight
    values accordingly so the raw term neither dominates nor vanishes. The
    env reward is hard-bounded to roughly [-r_step_scale, r_step_scale]
    (~[-0.5, 0.5]); DSR is clipped to +/-dsr_clip=10 but typically small per
    step. That a raw_weight of 0.1 can already dominate is the exact risk
    this probe quantifies.

    Instrumentation does not change training semantics: it only wraps
    reward_shaper.step to capture |DSR reward| and feeds a tick_callback that
    captures |env reward| from each StepResult. Reports means over all
    envs x steps.
    """
    import torch

    from dataset import MultiTickerRolloutDataset
    from env.vec_trading_env import VecTradingEnv
    from training.config import TrainingConfig
    from training.ppo_hybrid import HybridActorCritic, collect_rollout, compute_gae, ppo_update
    from training.reward import DifferentialSharpeReward
    from train import build_risk_pipeline

    cfg = TrainingConfig()
    cfg.run.total_rollouts = n_rollouts
    cfg.run.checkpoint_every_n_rollouts = 10**9
    cfg.run.best_metric_warmup_rollouts = 10**9
    cfg.run.tick_log_every_n_ticks = n_rollouts
    cfg.run.metrics_path = os.path.join(output_dir, "probe_metrics.jsonl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.run.seed)

    ds = MultiTickerRolloutDataset(window_size=cfg.env.window_size, split="train", device=str(device))
    env = VecTradingEnv(
        dataset=ds, initial_cash=cfg.env.initial_cash, max_position_frac=cfg.env.max_position_frac,
        tick_size=cfg.env.tick_size, spread_bps=cfg.env.spread_bps, impact_coef=cfg.env.impact_coef,
        max_participation=cfg.env.max_participation, commission_per_share=cfg.env.commission_per_share,
        commission_bps=cfg.env.commission_bps, min_commission=cfg.env.min_commission,
        platform_fee_per_trade=cfg.env.platform_fee_per_trade, r_step_scale=cfg.env.r_step_scale,
        hold_loser_penalty=cfg.env.hold_loser_penalty, enable_mirroring=cfg.env.enable_mirroring,
        mirror_prob=cfg.env.mirror_prob, overtrade_window=cfg.env.overtrade_window,
        overtrade_free_trades=cfg.env.overtrade_free_trades,
        overtrade_penalty_coef=cfg.reward.overtrade_penalty_coef,
        execution_price_column=cfg.env.execution_price_column,
        bias_window=cfg.env.bias_window, diversity_bonus_coef=cfg.env.diversity_bonus_coef, device=str(device),
    )
    print(f"[probe] n_envs={env.n_envs}, tickers={len(env.tickers)}, rollout_length={cfg.ppo.rollout_length} "
          f"(T*n_envs = {cfg.ppo.rollout_length * env.n_envs} transitions/rollout)")
    actor_critic = HybridActorCritic(n_features=len(env.feature_names), cfg=cfg).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=cfg.ppo.learning_rate, eps=cfg.ppo.adam_eps)
    kelly_sizer, risk_manager, kill_switch = build_risk_pipeline(cfg, env.n_envs, device)
    reward_shaper = DifferentialSharpeReward(
        n_envs=env.n_envs, eta=cfg.reward.dsr_eta, eps=cfg.reward.dsr_eps,
        warmup_steps=cfg.reward.dsr_warmup_steps, clip=cfg.reward.dsr_clip, device=str(device),
    )

    dsr_acc: List[float] = []
    raw_acc: List[float] = []

    _orig_step = reward_shaper.step

    def _step_wrapper(step_return: Any) -> Any:
        out = _orig_step(step_return)
        dsr_acc.append(out.abs().detach().mean().item())
        return out

    reward_shaper.step = _step_wrapper  # type: ignore[method-assign]

    def _tick_cb(local_t: int, step_result: Any, *args: Any, **kwargs: Any) -> None:
        raw_acc.append(step_result.reward.abs().detach().mean().item())

    obs = env.reset()
    hidden = actor_critic.init_hidden(env.n_envs, device)
    kill_switch.start_new_day(env.portfolio.equity(env._current_prices().unsqueeze(1)))  # noqa: SLF001
    for _ in range(n_rollouts):
        buffer, obs, final_value, hidden = collect_rollout(
            env, actor_critic, kelly_sizer, risk_manager, kill_switch, reward_shaper, obs, hidden, cfg,
            tick_callback=_tick_cb,
        )
        compute_gae(buffer, final_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        ppo_update(actor_critic, optimizer, buffer, cfg)
        if buffer.done.any():
            obs = env.reset()
            hidden = actor_critic.init_hidden(env.n_envs, device)
            kelly_sizer.reset()
            reward_shaper.reset()
            kill_switch.reset()

    mean_dsr = (sum(dsr_acc) / len(dsr_acc)) if dsr_acc else float("nan")
    mean_raw = (sum(raw_acc) / len(raw_acc)) if raw_acc else float("nan")
    ratio = (mean_raw / max(mean_dsr, 1e-12)) if mean_dsr == mean_dsr else float("nan")

    print("\n" + "=" * 68)
    print(f"REWARD-SCALE PROBE ({n_rollouts} rollout(s), default config, mean|.| over all envs x steps)")
    print(f"  mean|DSR reward| = {mean_dsr:.6f}")
    print(f"  mean|raw reward| = {mean_raw:.6f}")
    print(f"  ratio raw/DSR    = {ratio:.3f}")
    print("  -> raw_weight grid {0.01, 0.05, 0.2}: at these ratios the raw term stays a small-to-moderate "
          "fraction of DSR. If ratio >> 1, shrink raw_weight ~10x; if ratio << 0.01, raise it.")
    print("=" * 68 + "\n")
    return {"mean_dsr": mean_dsr, "mean_raw": mean_raw, "ratio": ratio}


# --------------------------------------------------------------------------
# Manifest -- the traceability report
# --------------------------------------------------------------------------

def print_and_save_manifest(output_root: str, run_ids: List[str], overrides_per_run: List[Dict[str, Any]]) -> None:
    manifest = {
        "swept_params": [
            {
                "dotted_path": p.dotted_path,
                "defined_in": p.defined_in,
                "consumed_by": p.consumed_by,
                "description": p.description,
                "values_tried": p.values,
            }
            for p in HYPERPARAM_REGISTRY
        ],
        "runs": [
            {"run_id": rid, "overrides": overrides}
            for rid, overrides in zip(run_ids, overrides_per_run)
        ],
    }

    print("=" * 78)
    print("HYPERPARAMETER SWEEP MANIFEST")
    print("=" * 78)
    for p in HYPERPARAM_REGISTRY:
        print(f"\n{p.dotted_path}")
        print(f"  defined in:   {p.defined_in}")
        print(f"  consumed by:  {', '.join(p.consumed_by)}")
        print(f"  description:  {p.description}")
        print(f"  values tried: {p.values}")
    print(f"\n{len(run_ids)} total run(s). "
          f"Registry lists every swept param's candidate values (traceability); "
          f"the actual per-run overrides are the one-at-a-time Tier-0 cells below "
          f"unless a Cartesian grid was passed explicitly.")
    print("=" * 78 + "\n")

    manifest_path = os.path.join(output_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Manifest written to {manifest_path}\n")


# --------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------

def run_sweep(
    grid: Optional[Dict[str, List[Any]]] = None,
    total_rollouts_per_run: int = 50,
    output_root: str = "sweep_output",
    clean: bool = False,
    parallel: int = 1,
    threads_per_worker: Optional[int] = None,
) -> None:
    if not os.path.exists(os.path.join(PROCESSED_DIR, "metadata.json")):
        print(
            f"No processed data found at {PROCESSED_DIR}/metadata.json.\n"
            "This sweep runs REAL training, so it needs real local data first. Run:\n"
            "    python fetch_alpaca.py\n"
            "    python preprocess.py\n"
            "then re-run this script."
        )
        sys.exit(1)

    grid = grid if grid is not None else _default_tier0_grid()

    if clean and os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    # grid is either a dict {dotted_path: [values]} (Cartesian product) or a
    # list of override-dicts (one cell per dict -- the one-at-a-time Tier 0
    # default). Normalize both into overrides_per_run.
    if isinstance(grid, dict):
        param_names = list(grid.keys())
        value_lists = [grid[name] for name in param_names]
        combos = list(itertools.product(*value_lists))
        overrides_per_run = [dict(zip(param_names, combo)) for combo in combos]
    else:  # list[Dict[str, Any]]
        overrides_per_run = [dict(cell) for cell in grid]

    run_ids = [f"{i:03d}" for i in range(len(overrides_per_run))]

    print_and_save_manifest(output_root, run_ids, overrides_per_run)

    if parallel < 1:
        raise ValueError(f"--parallel must be >= 1, got {parallel}")

    tasks = [
        (run_id, overrides, total_rollouts_per_run, output_root)
        for run_id, overrides in zip(run_ids, overrides_per_run)
    ]

    # Reuse finished cells from a previous sweep run when the grid cell is
    # byte-identical (same overrides + same rollout budget) -- see
    # _grid_fingerprint / run_one's result.json. --clean already wiped
    # everything, so in that case there is nothing to reuse.
    cached: Dict[str, Dict[str, Any]] = {}
    if not clean:
        for run_id, overrides in zip(run_ids, overrides_per_run):
            result_path = os.path.join(output_root, f"run_{run_id}", "result.json")
            if not os.path.exists(result_path):
                continue
            try:
                with open(result_path, "r") as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if payload.get("grid_fingerprint") == _grid_fingerprint(overrides, total_rollouts_per_run):
                cached[run_id] = payload["result"]

    todo = [t for t in tasks if t[0] not in cached]
    for run_id in cached:
        print(f"[sweep] run {run_id}: reused cached result "
              f"(wall_seconds={cached[run_id].get('wall_seconds')}, "
              f"trades_last_rollout={cached[run_id].get('trades_in_last_rollout')})")

    results: Dict[str, Dict[str, Any]] = dict(cached)
    if todo:
        print(f"[sweep] {len(todo)} run(s) to execute with {parallel} parallel worker(s), "
              f"{threads_per_worker or max(1, os.cpu_count() // parallel)} torch thread(s) each.\n")
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(
            processes=parallel,
            initializer=_worker_init,
            initargs=(threads_per_worker or max(1, os.cpu_count() // parallel),),
        )
        try:
            for result in pool.imap_unordered(_worker_run, todo):
                results[result["run_id"]] = result
                status = "ERROR" if result.get("error") else "ok"
                print(f"[sweep] run {result['run_id']}: {status}  "
                      f"reward_ema={result.get('final_reward_ema')}  "
                      f"entropy(last5 avg)={result.get('mean_entropy_discrete_last5')}  "
                      f"trades_last_rollout={result.get('trades_in_last_rollout')}  "
                      f"({result.get('wall_seconds')}s)")
        finally:
            pool.close()
            pool.join()

    ordered = [results[rid] for rid in run_ids if rid in results]
    results_path = os.path.join(output_root, "results.csv")
    if ordered:
        fieldnames = list(ordered[0].keys())
        # keep column order stable even if some runs errored before all keys existed
        for r in ordered:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered)
        print(f"\nResults written to {results_path}")

    _print_summary_table(ordered)


def _print_summary_table(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY (sorted by trades_in_last_rollout, descending -- runs where trading "
          "stopped will sort to the bottom)")
    print("=" * 78)
    rows = sorted(
        results,
        key=lambda r: (r.get("trades_in_last_rollout") is None, -(r.get("trades_in_last_rollout") or 0)),
    )
    for r in rows:
        param_str = ", ".join(f"{k.replace('param.', '')}={v}" for k, v in r.items() if k.startswith("param."))
        flag = "  <-- ERROR" if r.get("error") else ("  <-- trading stopped" if r.get("trades_in_last_rollout") == 0 else "")
        print(f"  run {r['run_id']}  [{param_str}]  "
              f"trades={r.get('trades_in_last_rollout')}  "
              f"entropy~{r.get('mean_entropy_discrete_last5')}  "
              f"reward_ema={r.get('final_reward_ema')}{flag}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local hyperparameter sweep, driving real train.py runs.")
    parser.add_argument("--total-rollouts-per-run", type=int, default=50,
                         help="Short by design -- see this file's module docstring on scope.")
    parser.add_argument("--output-dir", type=str, default="sweep_output")
    parser.add_argument("--grid-file", type=str, default=None,
                         help="Optional path to a JSON file of {dotted_path: [values]} to override the "
                              "built-in HYPERPARAM_REGISTRY grid entirely.")
    parser.add_argument("--clean", action="store_true", help="Delete --output-dir before starting.")
    parser.add_argument("--parallel", type=int, default=0,
                         help="Number of sweep runs to execute in parallel processes. Default: "
                              "cpu_count // 2 (6 on a 12-core machine). Each worker is pinned to "
                              "cpu_count // parallel torch threads so cores aren't oversubscribed.")
    parser.add_argument("--threads-per-worker", type=int, default=None,
                         help="torch threads per worker process (default: cpu_count // parallel). "
                              "Only tune if you know thread scaling on this machine.")
    parser.add_argument("--probe-scale", action="store_true",
                         help="Run the reward-scale probe (design step 2): one short real-training run at the "
                              "default config reporting mean|DSR reward| vs mean|raw reward| so the raw_weight "
                              "grid can be scaled correctly before committing it. Runs standalone; does not sweep.")
    parser.add_argument("--probe-rollouts", type=int, default=2,
                         help="Rollouts for --probe-scale (default 2).")
    args = parser.parse_args(argv)
    if args.parallel == 0:
        args.parallel = max(1, os.cpu_count() // 2)
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.probe_scale:
        probe_reward_scale(n_rollouts=args.probe_rollouts, output_dir=args.output_dir)
        sys.exit(0)
    grid_override = None
    if args.grid_file:
        with open(args.grid_file, "r") as f:
            grid_override = json.load(f)
    run_sweep(
        grid=grid_override,
        total_rollouts_per_run=args.total_rollouts_per_run,
        output_root=args.output_dir,
        clean=args.clean,
        parallel=args.parallel,
        threads_per_worker=args.threads_per_worker,
    )
