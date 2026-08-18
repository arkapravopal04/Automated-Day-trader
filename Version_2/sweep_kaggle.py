"""
sweep_kaggle.py -- Tier-0 hyperparameter sweep, Kaggle-GPU edition.

WHY THIS FILE EXISTS: hyperparam_sweep.py is a LOCAL sweep runner -- it
parallelizes across CPU cores via a multiprocessing Pool and assumes a
12-core laptop, where per-rollout cost (~19 min/update on CPU) makes even a
50-rollout cell expensive. On Kaggle the same training loop costs ~38
seconds per rollout on a T4 (measured 2026-08 from kaggle_data/metrics
(4).jsonl: 100 tickers, rollout_length=256, fp32, world_size=2 DDP -- the
per-GPU work is identical for a single-GPU cell). That changes the whole
budget math: 15 cells x 50 rollouts = ~8h serial on one GPU, which is why
this runner exists.

WHAT THIS DOES: runs the REAL Tier-0 screening grid (the same cells
hyperparam_sweep.py runs -- its `_default_tier0_grid()`, or a custom
--grid-file) as one training run per cell, but:

  - ONE CELL PER GPU, all GPUs in the session at once. Each child
    subprocess is pinned with CUDA_VISIBLE_DEVICES=<i> and runs its share
    of cells sequentially. This is NOT DDP -- the two T4s run two
    independent screening runs in parallel, which is exactly what a
    screening sweep needs (cells are independent by construction). No
    cross-GPU allreduce, no DDP wrapper, no sync barrier between cells.
  - Each cell is a subprocess, so a crashed cell (OOM, NaN, whatever)
    cannot take down the session; the orchestrator aggregates whatever
    finished and reports what didn't.
  - Per-cell rollout budget defaults to 15 instead of 50. The sweep's
    stated purpose is RULE-OUT -- entropy collapse and zero-trade
    pathology show up in the first handful of rollouts (see
    hyperparam_sweep.py's module docstring) -- so 15 rollouts keeps every
    failure mode the sweep exists to catch visible while cutting 70% of
    the compute. Nothing about training dynamics is changed: same
    rollout_length, same ppo_epochs, same seed, same data, same I/O
    throttles run_one() already applies (tick logging once per rollout,
    no checkpoints).
  - Cells reuse the same fingerprint scheme as hyperparam_sweep.py: a
    finished run_<id>/result.json with a matching grid fingerprint is
    skipped, so re-running with a shorter --cells list or after a crash
    only trains what's actually missing.
  - A `--merge-only` mode aggregates an existing sweep_output tree into
    results.csv + summary. That is how you combine TWO parallel Kaggle
    sessions (see the two-session plan below): pull both sessions'
    sweep_output dirs together on one machine, run --merge-only, done.

TIME BUDGET (measured per-rollout cost 38s on T4, ~90s fixed overhead per
cell for dataset load + model init):

  cells x rollouts  | 1 session, 2 GPUs | 2 sessions, 4 GPUs
  15 x 15           |      ~88 min      |     ~44 min
  13 x 15 (drop     |      ~77 min      |     ~44 min
    clip_range)     |                   |
  15 x 20           |     ~113 min      |     ~57 min

  Plus ~5-10 min of session/import boot. "About an hour" therefore means:
    (a) one session, full 15-cell grid, 15 rollouts      -> ~1.5h
    (b) one session, 13 cells (drop clip_range 0.1/0.3),
        15 rollouts                                      -> ~77 min
    (c) two sessions in parallel (4 GPUs), full grid,
        15-20 rollouts                                   -> ~45-60 min  <-- recommended
  clip_range is the docstring-designated first knob to drop when the
  budget shrinks (see hyperparam_sweep.py's SweptParam for it).

  Kaggle GPU quota note: a dual-GPU session burns 2 GPU-hours per wall
  hour. Plan (c) costs ~4 GPU-hours of the ~30h weekly quota; the full
  post-sweep training run (1000 rollouts DDP) costs ~21. Plan for it.

USAGE -- one session, all 15 cells (or a --cells subset):

    !git clone <repo>   # into /kaggle/working, as usual
    !python Version_2/sweep_kaggle.py --rollouts 15        # ~86 min on 2x T4
    !python Version_2/sweep_kaggle.py --cells 0-7          # one GPU gets cells 0-7...
    # ...and in a SECOND parallel session, the other half:
    !python Version_2/sweep_kaggle.py --cells 8-14
    # ...then, after pulling both sessions' sweep_output trees together:
    !python Version_2/sweep_kaggle.py --merge-only --output-dir sweep_output

Prereq: processed data must exist where paths.py resolves on Kaggle
(/kaggle/working/data/processed/metadata.json) -- run
`python Version_2/run_kaggle.py --preprocess` once, or attach your cached
data Dataset. The script refuses to start with a pointer to that command.

OUTPUT (same layout as hyperparam_sweep.py):
    sweep_output/manifest.json          -- swept params + per-run overrides
    sweep_output/results.csv            -- one row per finished cell
    sweep_output/run_<id>/              -- per-cell metrics.jsonl, result.json,
                                           stdout.log (written by run_one)
    sweep_output/gpu_<gpu>.log          -- this run's child-process console
    The summary (printed at the end) also lists, per knob, the winner cell
    by the sweep's own rule-out criteria -- that is the direct input to
    the "change the values in training/config.py" step.

NOTE ON GRID: the default grid is hyperparam_sweep._default_tier0_grid()
(15 one-at-a-time cells -- baseline, entropy_discrete x3, lr x2,
raw_weight x3 + pure-raw probe, kelly_multiplier x3, clip_range x2). Pass
--grid-file <json> ({dotted_path: [values]}) for a Cartesian product grid
instead -- same semantics as hyperparam_sweep.py.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Measured on Kaggle 2x T4 (DDP, world_size=2), 100 tickers,
# rollout_length=256, fp32 -- see module docstring. Per-GPU work for a
# single-GPU cell is the same, so this is the right per-rollout estimate.
SECONDS_PER_ROLLOUT = 38.0
# Fixed per-cell overhead: dataset load + model/env init + imports.
SECONDS_PER_CELL_OVERHEAD = 90.0


# ---------------------------------------------------------------------------
# Grid handling (mirrors hyperparam_sweep.run_sweep's normalization)
# ---------------------------------------------------------------------------

def normalize_grid(grid):
    """grid: list of override-dicts (one cell per dict) OR {dotted_path:
    [values]} (Cartesian product). Returns list[Dict[str, Any]]."""
    if isinstance(grid, dict):
        import itertools
        param_names = list(grid.keys())
        value_lists = [grid[name] for name in param_names]
        combos = list(itertools.product(*value_lists))
        return [dict(zip(param_names, combo)) for combo in combos]
    return [dict(cell) for cell in grid]


def parse_cells(spec):
    """'0-7,9' -> [0,1,2,3,4,5,6,7,9]. None -> None (all cells)."""
    if spec is None:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def load_grid(grid_file):
    if grid_file:
        with open(grid_file) as f:
            return normalize_grid(json.load(f))
    from hyperparam_sweep import _default_tier0_grid
    return normalize_grid(_default_tier0_grid())


# ---------------------------------------------------------------------------
# Child worker: run a subset of cells sequentially on one pinned GPU
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicate writes to several streams (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:  # noqa: BLE001 -- logging must never crash training
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass


def worker_main(args) -> int:
    """Runs the cells assigned to this GPU in-process (no Pool -- this
    process IS the worker). Returns 0 if every assigned cell finished or
    was already cached, 1 otherwise."""
    # Pin threads BEFORE torch import (torch reads these at import time).
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    import torch
    torch.set_num_threads(2)

    from hyperparam_sweep import _grid_fingerprint, run_one

    grid = load_grid(args.grid_file)
    if args.cells is None:
        cells = list(range(len(grid)))
    else:
        cells = parse_cells(args.cells)
    bad = [c for c in cells if c >= len(grid)]
    if bad:
        print(f"[gpu {args.gpu_index}] WARNING: --cells out of range (grid has "
              f"{len(grid)} cells): {bad} -- skipping")
        cells = [c for c in cells if c < len(grid)]
    if not cells:
        print(f"[gpu {args.gpu_index}] no valid cells assigned -- nothing to do.")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"gpu_{args.gpu_index}.log")
    log_fh = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = sys.stdout

    print(f"[gpu {args.gpu_index}] {len(cells)} cell(s): {cells}  "
          f"rollouts/cell={args.rollouts}  cuda:0 = physical GPU {args.gpu_index}")

    failed = 0
    for cell_idx in cells:
        run_id = f"{cell_idx:03d}"
        overrides = grid[cell_idx]
        result_path = os.path.join(args.output_dir, f"run_{run_id}", "result.json")

        cached = False
        if os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    payload = json.load(f)
                cached = payload.get("grid_fingerprint") == _grid_fingerprint(overrides, args.rollouts)
            except (json.JSONDecodeError, OSError):
                cached = False
        if cached:
            print(f"[gpu {args.gpu_index}] run {run_id}: reused cached result "
                  f"({payload['result'].get('wall_seconds')}s wall, "
                  f"trades_last_rollout={payload['result'].get('trades_in_last_rollout')})")
            continue

        t0 = time.time()
        result = run_one(run_id, overrides, args.rollouts, args.output_dir)
        status = "ERROR" if result.get("error") else "ok"
        if result.get("error"):
            failed += 1
        print(f"[gpu {args.gpu_index}] run {run_id}: {status}  "
              f"reward_ema={result.get('final_reward_ema')}  "
              f"entropy(last5)={result.get('mean_entropy_discrete_last5')}  "
              f"trades_last_rollout={result.get('trades_in_last_rollout')}  "
              f"({time.time() - t0:.0f}s wall)")

    log_fh.close()
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Aggregation: rebuild results.csv + summary from run_*/result.json files
# ---------------------------------------------------------------------------

def _read_results(output_dir, cells, grid, rollouts):
    """Returns (results dict keyed by run_id, missing list, stale list)."""
    from hyperparam_sweep import _grid_fingerprint
    results, missing, stale = {}, [], []
    for cell_idx in cells:
        run_id = f"{cell_idx:03d}"
        result_path = os.path.join(output_dir, f"run_{run_id}", "result.json")
        if not os.path.exists(result_path):
            missing.append(run_id)
            continue
        try:
            with open(result_path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            missing.append(run_id)
            continue
        if payload.get("grid_fingerprint") != _grid_fingerprint(grid[cell_idx], rollouts):
            stale.append(run_id)
            continue
        results[run_id] = payload["result"]
    return results, missing, stale


def _write_csv(output_root, ordered):
    import csv
    results_path = os.path.join(output_root, "results.csv")
    if not ordered:
        print(f"[sweep_kaggle] no finished runs -- no {results_path} written.")
        return
    fieldnames = list(ordered[0].keys())
    for r in ordered:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"[sweep_kaggle] results written to {results_path}")


def _print_winners(results, grid, cells):
    """Per-knob winner by the sweep's rule-out criteria: must still be
    trading in the last rollout AND have non-collapsed entropy; among
    those, highest final reward_ema. Direct input to the config.py edit.

    Single-knob grids (the Tier-0 default) get one winner per knob.
    Multi-knob cells (Cartesian grids like Tier 1's raw_weight x
    r_step_scale) have no single attribution -- for those, print the
    top-5 cells by the same rule-out score instead."""
    from collections import defaultdict

    def health(result):
        """Rule-out score: trading and non-collapsed entropy gate, then
        reward_ema. Higher is better."""
        if result is None or result.get("error"):
            return float("-inf")
        if (result.get("trades_in_last_rollout") or 0) <= 0:
            return float("-inf")
        entropy = result.get("mean_entropy_discrete_last5")
        if entropy is None or entropy <= 0.05:
            return float("-inf")
        ema = result.get("final_reward_ema")
        return ema if ema is not None else float("-inf")

    groups = defaultdict(list)  # dotted_path -> [(value, result)]
    multi = []
    for cell_idx in cells:
        run_id = f"{cell_idx:03d}"
        result = results.get(run_id)
        if result is None or result.get("error"):
            continue
        overrides = grid[cell_idx]
        if len(overrides) == 1:
            path, value = next(iter(overrides.items()))
            groups[path].append((value, result))
        else:
            multi.append((cell_idx, overrides, result))

    print("\n" + "=" * 78)
    print("PER-KNOB WINNERS (rule-out criteria: trading in last rollout + entropy not")
    print("collapsed; tie-break: highest final reward_ema)")
    print("=" * 78)
    if not groups and not multi:
        print("  (no finished cells with results -- check for crashes above)")
        return
    for path in sorted(groups):
        entries = groups[path]
        healthy = [e for e in entries if health(e[1]) != float("-inf")]
        pool = healthy or entries
        if not pool:
            continue
        best_value, best_result = max(pool, key=lambda e: health(e[1]))
        tag = "" if healthy else "  <-- NO healthy cell; least-bad pick"
        print(f"  {path:<28} -> {str(best_value):<8} (run {best_result['run_id']}, "
              f"reward_ema={best_result.get('final_reward_ema')}, "
              f"trades_last={best_result.get('trades_in_last_rollout')}){tag}")

    if multi:
        ranked = sorted(multi, key=lambda m: health(m[2]), reverse=True)
        print("\n  TOP CELLS (multi-knob / Cartesian grid -- no single-knob attribution):")
        for cell_idx, overrides, result in ranked[:5]:
            ov_str = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
            score = health(result)
            if score == float("-inf"):
                print(f"    run {result['run_id']:>3}  [{ov_str}]  "
                      f"trades_last={result.get('trades_in_last_rollout')}  "
                      f"entropy={result.get('mean_entropy_discrete_last5')}  <-- unhealthy")
            else:
                print(f"    run {result['run_id']:>3}  [{ov_str}]  "
                      f"reward_ema={result.get('final_reward_ema'):.6g}  "
                      f"trades_last={result.get('trades_in_last_rollout')}")
    print("\nNext step: edit training/config.py defaults to the winners above, then run")
    print("the full training + backtest (run_kaggle.py --train --fresh --total-rollouts ...).")


def aggregate(output_root, cells, grid, rollouts):
    from hyperparam_sweep import _print_summary_table
    results, missing, stale = _read_results(output_root, cells, grid, rollouts)
    ordered = [results[rid] for rid in (f"{c:03d}" for c in cells) if rid in results]
    _write_csv(output_root, ordered)
    _print_summary_table(ordered)
    _print_winners(results, grid, cells)
    if missing:
        print(f"\n[sweep_kaggle] MISSING runs (never finished): {', '.join(missing)} -- "
              f"re-run with --cells for those indices to fill them in.")
    if stale:
        print(f"[sweep_kaggle] STALE runs (fingerprint mismatch -- grid or rollouts changed): "
              f"{', '.join(stale)} -- delete their run dirs and re-run to force retraining.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _time_estimate(n_cells, n_gpus, rollouts):
    per_cell_s = rollouts * SECONDS_PER_ROLLOUT + SECONDS_PER_CELL_OVERHEAD
    wall_min = math.ceil(n_cells / max(1, n_gpus)) * per_cell_s / 60.0
    return wall_min, per_cell_s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Tier-0 hyperparameter sweep on Kaggle GPUs: one cell per GPU, "
                    "all session GPUs in parallel. See module docstring for the "
                    "time budget and the two-session plan.")
    parser.add_argument("--cells", type=str, default=None,
                        help="Cell indices to run, e.g. '0-7' or '0,2,4'. Default: all.")
    parser.add_argument("--rollouts", type=int, default=15,
                        help="Rollouts per cell (default 15; 50 was the local-CPU "
                             "budget -- screening only needs the first handful, "
                             "see module docstring).")
    parser.add_argument("--gpus", type=int, default=0,
                        help="GPUs to use (default: all CUDA devices in the session).")
    parser.add_argument("--grid-file", type=str, default=None,
                        help="JSON {dotted_path: [values]} grid (Cartesian). "
                             "Default: hyperparam_sweep's Tier-0 list.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where run_*/ + results.csv go. Default: "
                             "/kaggle/working/sweep_output on Kaggle, else ./sweep_output.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete --output-dir before starting.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Don't train: aggregate an existing sweep_output tree "
                             "(e.g. after combining two sessions' outputs) into "
                             "results.csv + summary.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and time estimate, train nothing.")
    parser.add_argument("--as-worker", action="store_true",
                        help=argparse.SUPPRESS)  # internal: child process entry
    parser.add_argument("--gpu-index", type=int, default=0,
                        help=argparse.SUPPRESS)  # internal: which GPU this child is pinned to
    args = parser.parse_args(argv)

    os.chdir(SCRIPT_DIR)  # repo root as CWD, same as run_kaggle.py

    from paths import PROCESSED_DIR, is_kaggle

    if args.output_dir is None:
        args.output_dir = "/kaggle/working/sweep_output" if is_kaggle() else "sweep_output"

    if args.as_worker:
        return worker_main(args)

    grid = load_grid(args.grid_file)
    cells = parse_cells(args.cells)
    if cells is None:
        cells = list(range(len(grid)))
    if not cells:
        print("[sweep_kaggle] no cells selected -- nothing to do.")
        return 0
    bad = [c for c in cells if c >= len(grid)]
    if bad:
        print(f"[sweep_kaggle] --cells out of range (grid has {len(grid)} cells): {bad}")
        return 2

    if args.merge_only:
        aggregate(args.output_dir, cells, grid, args.rollouts)
        return 0

    if not os.path.exists(os.path.join(PROCESSED_DIR, "metadata.json")):
        print(f"[sweep_kaggle] no processed data at {PROCESSED_DIR}/metadata.json.\n"
              "This sweep runs REAL training, so it needs the feature cache first. On "
              "Kaggle run:\n"
              "    !python Version_2/run_kaggle.py --preprocess\n"
              "(or attach your cached data Dataset). Then re-run this script.")
        return 1

    import torch
    available = torch.cuda.device_count()
    if available == 0:
        print("[sweep_kaggle] WARNING: no CUDA. This script exists for Kaggle GPU "
              "sessions; local CPU sweeps are not viable (see memory: ~19 min/update). "
              "Running anyway with 1 'GPU' slot.")
        n_gpus = 1
    else:
        n_gpus = args.gpus if args.gpus > 0 else available
        if n_gpus > available:
            print(f"[sweep_kaggle] WARNING: requested {n_gpus} GPUs but only "
                  f"{available} available -- using what's here.")
            n_gpus = available

    wall_min, per_cell_s = _time_estimate(len(cells), n_gpus, args.rollouts)
    print(f"[sweep_kaggle] plan: {len(cells)} cell(s) x {args.rollouts} rollouts "
          f"across {n_gpus} GPU(s).")
    print(f"[sweep_kaggle] per-cell estimate ~{per_cell_s:.0f}s "
          f"({args.rollouts} x {SECONDS_PER_ROLLOUT:.0f}s/rollout measured + "
          f"{SECONDS_PER_CELL_OVERHEAD:.0f}s overhead) -> wall-clock ~{wall_min:.0f} min. "
          f"Run `--dry-run` to skip training and just print this.")

    if args.dry_run:
        return 0

    if args.clean and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Manifest: same traceability record hyperparam_sweep.py writes.
    from hyperparam_sweep import print_and_save_manifest
    run_ids = [f"{c:03d}" for c in cells]
    print_and_save_manifest(args.output_dir, run_ids, [grid[c] for c in cells])

    # Round-robin cells across GPUs: balances load when one GPU is slower,
    # and a GPU dying mid-sweep only loses its own share.
    per_gpu = {g: [] for g in range(n_gpus)}
    for i, cell_idx in enumerate(cells):
        per_gpu[i % n_gpus].append(cell_idx)

    children = []
    for g in range(n_gpus):
        if not per_gpu[g]:
            continue
        cells_arg = ",".join(str(c) for c in per_gpu[g])
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--as-worker", "--gpu-index", str(g),
               "--cells", cells_arg,
               "--rollouts", str(args.rollouts),
               "--output-dir", args.output_dir]
        if args.grid_file:
            cmd += ["--grid-file", args.grid_file]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        print(f"[sweep_kaggle] spawning GPU {g}: cells {per_gpu[g]} "
              f"(CUDA_VISIBLE_DEVICES={g}, log -> {args.output_dir}/gpu_{g}.log)")
        children.append(subprocess.Popen(cmd, env=env))

    codes = [p.wait() for p in children]
    for g, code in zip(range(n_gpus), codes):
        print(f"[sweep_kaggle] GPU {g} child exited with code {code}")

    aggregate(args.output_dir, cells, grid, args.rollouts)
    return 1 if any(code != 0 for code in codes) else 0


if __name__ == "__main__":
    sys.exit(main())
