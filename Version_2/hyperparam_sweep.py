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
    sweep_output/run_<id>/           -- that run's own checkpoints + metrics.jsonl,
                                         untouched and inspectable individually.

SCOPE, stated plainly: a short local sweep (default 50 rollouts/run) is
for RELATIVE comparison between configs -- "did entropy collapse happen
faster or slower here, did trades stop happening here" -- not a substitute
for a full Kaggle training run. Don't pick a "winner" from this and assume
it'll hold at 2000+ rollouts; use it to rule OUT configs that show obvious
early pathology (entropy crashing to ~0 in the first handful of rollouts,
zero trades from rollout 1) before spending real Kaggle GPU time on them.
"""

import argparse
import copy
import csv
import itertools
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "env"))

from paths import PROCESSED_DIR  # noqa: E402


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
        values=[0.02, 0.05, 0.10],   # 0.02 is the current config.py default
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
        values=[0.25, 0.5],   # 0.25 is the current config.py default
    ),
    SweptParam(
        dotted_path="ppo.learning_rate",
        defined_in="training/config.py -- PPOConfig.learning_rate",
        consumed_by=["train.py -- torch.optim.Adam(..., lr=cfg.ppo.learning_rate)"],
        description="Adam learning rate for the whole actor-critic.",
        values=[3e-4, 1e-4],   # 3e-4 is the current config.py default
    ),
]


def _default_grid_dict() -> Dict[str, List[Any]]:
    return {p.dotted_path: p.values for p in HYPERPARAM_REGISTRY}


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
    """
    import train  # imported here (not at module top) so the monkeypatch below is scoped per-call

    run_dir = os.path.join(output_root, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    cfg = build_cfg(overrides)
    cfg.run.total_rollouts = total_rollouts
    cfg.run.checkpoint_dir = checkpoint_dir
    cfg.run.metrics_path = metrics_path
    # Keep checkpoint I/O out of the way during a short sweep run -- we
    # only need the metrics, not intermediate checkpoints, for comparison.
    cfg.run.checkpoint_every_n_rollouts = max(total_rollouts, 1)

    original_training_config = train.TrainingConfig
    train.TrainingConfig = lambda: cfg  # see module docstring -- the actual override mechanism

    start = time.time()
    error: Optional[str] = None
    try:
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
    return result


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
    print(f"\n{len(run_ids)} total run(s) (cartesian product of the values above).")
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

    grid = grid if grid is not None else _default_grid_dict()

    if clean and os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

    param_names = list(grid.keys())
    value_lists = [grid[name] for name in param_names]
    combos = list(itertools.product(*value_lists))

    run_ids = [f"{i:03d}" for i in range(len(combos))]
    overrides_per_run = [dict(zip(param_names, combo)) for combo in combos]

    print_and_save_manifest(output_root, run_ids, overrides_per_run)

    results: List[Dict[str, Any]] = []
    for run_id, overrides in zip(run_ids, overrides_per_run):
        print(f"[sweep] run {run_id}/{len(combos) - 1}: {overrides}")
        result = run_one(run_id, overrides, total_rollouts_per_run, output_root)
        results.append(result)
        status = "ERROR" if result.get("error") else "ok"
        print(f"[sweep]   -> {status}  "
              f"reward_ema={result.get('final_reward_ema')}  "
              f"entropy(last5 avg)={result.get('mean_entropy_discrete_last5')}  "
              f"trades_last_rollout={result.get('trades_in_last_rollout')}  "
              f"({result.get('wall_seconds')}s)")

    results_path = os.path.join(output_root, "results.csv")
    if results:
        fieldnames = list(results[0].keys())
        # keep column order stable even if some runs errored before all keys existed
        for r in results:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {results_path}")

    _print_summary_table(results)


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    grid_override = None
    if args.grid_file:
        with open(args.grid_file, "r") as f:
            grid_override = json.load(f)
    run_sweep(
        grid=grid_override,
        total_rollouts_per_run=args.total_rollouts_per_run,
        output_root=args.output_dir,
        clean=args.clean,
    )