"""
run_kaggle.py -- one-shot runner for the Version_2 pipeline on Kaggle.

Drop this file in the repo, clone the repo into a Kaggle notebook
(Internet: ON), and run the cells you need:

    !python Version_2/run_kaggle.py --quick
    # == same as: --diagnostics --train --total-rollouts 100 (data must exist)

or stage-by-stage (each flag is independent and re-runnable):

    !python Version_2/run_kaggle.py --fetch             # 1. download 5-min bars
    !python Version_2/run_kaggle.py --preprocess       # 2. features + metadata.json
    !python Version_2/run_kaggle.py --diagnostics      # 3. data/env/GPU sanity checks
    !python Version_2/run_kaggle.py --book             # 4. P2 book + lambda sweep (minutes)
    !python Version_2/run_kaggle.py --pretrain         # 5. supervised trunk (GPU)
    !python Version_2/run_kaggle.py --train --total-rollouts 200   # 6. PPO training

Secrets (Add-ons -> Secrets): reads ALPACA_API_KEY / ALPACA_SECRET_KEY and
exports them as env vars, plus TRADING_ALPACA_PAPER_KEY/SECRET if you
defined those too. fetch_alpaca.py auto-detects Kaggle Secrets as a
fallback, so this is belt-and-braces. No secret is ever printed or written
to disk by this script.

Outputs (attach /kaggle/working as a Dataset output to persist them):
    /kaggle/working/data/parquet/...            raw 5-min bars
    /kaggle/working/data/processed/...          features + metadata.json
    /kaggle/working/logs/metrics.jsonl          rollout metrics
    /kaggle/working/logs/metrics.jsonl.ticks.jsonl   per-tick metrics
    /kaggle/working/checkpoints/checkpoint_*.pt + checkpoint_best.pt
    /kaggle/working/pretrained/trunk_pretrained.pt   supervised trunk (--pretrain)
    /kaggle/working/logs/xsec_book.json              the P2 book sweep (--book)
    /kaggle/working/logs/trunk_edge.npz              the learned edge, timestamped
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Stage 0: secrets
# ---------------------------------------------------------------------------

def setup_secrets() -> None:
    """Export Alpaca credentials from Kaggle Secrets into env vars."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        print("[secrets] kaggle_secrets not available (not on Kaggle?) -- relying on .env / env vars.")
        return

    client = UserSecretsClient()
    mapping = {
        "ALPACA_API_KEY": "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY": "ALPACA_SECRET_KEY",
        "TRADING_ALPACA_PAPER_KEY": "TRADING_ALPACA_PAPER_KEY",
        "TRADING_ALPACA_PAPER_SECRET": "TRADING_ALPACA_PAPER_SECRET",
    }
    found = []
    for secret_name, env_name in mapping.items():
        if os.environ.get(env_name):
            found.append(env_name)
            continue
        try:
            value = client.get_secret(secret_name)
        except Exception:
            continue  # secret not defined -- fine, some stages don't need it
        if value:
            os.environ[env_name] = value
            found.append(env_name)

    if found:
        print(f"[secrets] exported: {', '.join(found)}")
    else:
        print("[secrets] no Alpaca secrets found -- fetch will skip network and use cached data only.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_python(script: str, *args: str, check: bool = True) -> int:
    """Run one of the repo's scripts with the same interpreter that launched us."""
    print(f"\n{'=' * 70}\n>>> python {script} {' '.join(args)}\n{'=' * 70}")
    cmd = [sys.executable, script, *args]
    proc = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if check and proc.returncode != 0:
        raise SystemExit(f"[run_kaggle] {script} failed with exit code {proc.returncode}")
    return proc.returncode


def ensure_deps(no_install: bool) -> None:
    if no_install:
        return
    req = os.path.join(SCRIPT_DIR, "requirements.txt")
    print("[deps] installing missing packages from requirements.txt ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", req],
        cwd=SCRIPT_DIR,
    )


def out_dir(kind: str) -> str:
    """Where a stage's artefacts go: "logs" or "checkpoints".

    Matches train.py's rule exactly (`--kaggle` / is_kaggle() -> /kaggle/working
    /...). This is not cosmetic. SCRIPT_DIR on Kaggle is the CLONED REPO, so
    writing there puts the book's JSON and the pre-trained trunk somewhere the
    training run does not look, somewhere the notebook's own metrics path does
    not point, and somewhere a re-clone silently discards.
    """
    sys.path.insert(0, SCRIPT_DIR)
    from paths import is_kaggle  # noqa: E402

    path = f"/kaggle/working/{kind}" if is_kaggle() else os.path.join(SCRIPT_DIR, kind)
    os.makedirs(path, exist_ok=True)
    return path


def cache_ready() -> bool:
    sys.path.insert(0, SCRIPT_DIR)
    from paths import is_cache_ready  # noqa: E402
    return is_cache_ready()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_fetch() -> None:
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_SECRET_KEY"):
        print("[fetch] no credentials -- skipping network fetch; cached data (if any) will be used as-is.")
        return
    run_python("fetch_alpaca.py")


def stage_preprocess() -> None:
    run_python("preprocess.py")


def stage_diagnostics() -> None:
    run_python("data_digonastics.py", check=False)
    run_python("env_digonastics.py", check=False)
    run_python("diagnostics_gpu_and_learning.py", check=False)


def stage_book(holds, lams, variant) -> None:
    """P2's analytic baseline on the CORRECTED panel.

    It has to run here rather than locally: the local parquet cache is stale
    and split-contaminated, and `eval/xsec_book.py` reads raw closes and volume
    directly, so a contaminated cache fabricates both the returns it scores and
    the ADV it prices impact against.

    Costs minutes, not GPU-hours, and it decides whether a training run is
    worth starting -- so run it before --train, not after.
    """
    args = ["--json", os.path.join(out_dir("logs"), "xsec_book.json"),
            "--variant", variant]
    for h in holds:
        args += ["--hold", str(h)]
    for l in lams:
        args += ["--lam", str(l)]
    run_python(os.path.join("eval", "xsec_book.py"), *args, check=False)


def stage_pretrain(hold: int, epochs: int, dump_edge: bool) -> None:
    """P2 bullet 3: fit the shared trunk against the tradeable forward return.

    Runs before --train, not instead of it. It answers "can these features
    predict the return the env can actually capture" in a form a regression can
    be held to, and writes pretrained/trunk_pretrained.pt for
    `--train --pretrained-trunk`. With --book-after-pretrain the resulting edge
    is fed straight back through eval/xsec_book.py, which is the only test that
    matters: an IC that does not clear the cost hurdle is not tradeable however
    significant it is.
    """
    logs = out_dir("logs")
    # Its OWN directory, not the checkpoint dir. The trunk is an INPUT to a
    # training run, and `--fresh` resets that run's outputs -- keeping the two
    # apart is what makes `--fresh --pretrained-trunk` a coherent thing to ask
    # for. (`_clean_checkpoint_dir` is also narrowed to `checkpoint_*.pt` now,
    # so this is belt and braces rather than the only guard.)
    out = os.path.join(out_dir("pretrained"), "trunk_pretrained.pt")
    npz = os.path.join(logs, "trunk_edge.npz")
    args = ["--hold", str(hold), "--epochs", str(epochs), "--out", out]
    if dump_edge:
        args += ["--dump-edge", npz]
    run_python(os.path.join("training", "pretrain_trunk.py"), *args, check=False)

    if dump_edge and os.path.exists(npz):
        run_python(
            os.path.join("eval", "xsec_book.py"),
            "--edge", "npz", "--edge-npz", npz, "--hold", str(hold),
            "--json", os.path.join(logs, "xsec_book_trunk.json"),
            check=False,
        )
    print()
    print(f"[pretrain] trunk -> {out}")
    print(f"[pretrain] feed it to training with:  "
          f"python run_kaggle.py --train --total-rollouts 151 --fresh "
          f"--pretrained-trunk {out}")


def stage_train(total_rollouts: int, resume: str, fresh: bool, force_single: bool,
                pretrained_trunk: str = None) -> None:
    import torch

    use_ddp = (not force_single) and torch.cuda.is_available() and torch.cuda.device_count() >= 2
    script = "train_ddp.py" if use_ddp else "train.py"
    args = ["--total-rollouts", str(total_rollouts)]
    if resume:
        args += ["--resume", resume]
    if fresh:
        args += ["--fresh"]
    if pretrained_trunk:
        if resume:
            # Both entrypoints ignore the trunk when resuming; say so here
            # rather than letting the flag look like it took effect.
            print("[run_kaggle] --resume given, so --pretrained-trunk is IGNORED: "
                  "the checkpoint already carries a trunk PPO has been updating.")
        elif not os.path.exists(pretrained_trunk):
            raise SystemExit(
                f"[run_kaggle] --pretrained-trunk {pretrained_trunk} does not exist. "
                "Run `python run_kaggle.py --pretrain` first."
            )
        args += ["--pretrained-trunk", pretrained_trunk]
    run_python(script, *args)


def print_summary(trained: bool = True) -> None:
    """Summarise what this invocation actually did.

    ``trained`` gates the metrics/checkpoint report. It used to print
    unconditionally, so `--fetch` and `--preprocess` each ended with a
    full training report read off a PREVIOUS run's metrics.jsonl -- a
    fetch cell signing off with 'rollout 79 ... net_worth 690630 ...
    total_trades 244835' from some earlier session. That output lands
    directly above the Phase 0/1 data gates, which is the worst possible
    place for a stale number that reads like a fresh result.
    """
    print("\n" + "=" * 70)
    print("RUN SUMMARY -- what to look for")
    print("=" * 70)

    if not trained:
        print("[stage] no training stage ran in this invocation.")
        print("        metrics.jsonl / checkpoints deliberately NOT reported --")
        print("        any numbers there belong to an earlier run, not this one.")
        return

    metrics = os.path.join("/kaggle/working", "logs", "metrics.jsonl")
    if os.path.exists(metrics):
        rollouts = []
        with open(metrics) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("record_type", "rollout") != "tick":
                    rollouts.append(rec)
        if rollouts:
            last = rollouts[-1]
            print(f"[metrics] {len(rollouts)} rollout record(s); latest:")
            for key in ("rollout", "reward", "reward_ema",
                        "alpha_per_turnover", "cost_per_turnover", "net_per_turnover",
                        "policy_loss", "value_loss",
                        "entropy_discrete", "approx_kl", "total_trades", "net_worth"):
                if key in last:
                    print(f"    {key:<18} {last[key]}")
        else:
            print("[metrics] file exists but no rollout records yet.")
    else:
        print("[metrics] no metrics file yet (nothing trained?).")

    ckpt_dir = os.path.join("/kaggle/working", "checkpoints")
    if os.path.isdir(ckpt_dir):
        files = sorted(os.listdir(ckpt_dir))
        print(f"[checkpoints] {len(files)} file(s): {', '.join(files[:6])}{' ...' if len(files) > 6 else ''}")

    print("""
EXPECTED ON A HEALTHY QUICK RUN (100 rollouts):
  * diagnostics: all [PASS] lines; GPU script: parameters on cuda, grad_norm > 0
  * train: entropy_discrete starts ~0.6-1.1 and trends down WITHOUT collapsing
    to ~0 in the first few rollouts (collapse = policy settling on never-trade)
  * total_trades climbs every rollout; halted stays all-False
  * policy_loss/value_loss finite (no NaN/Inf); grad_norm ~0.1-1.0
  * reward_ema drifts (up or down) -- it is a BLEND, not dollar PnL, and
    NOT a pure differential-Sharpe signal: RewardConfig.raw_weight=80
    (not 0 -- that note predated the measured 79x scale gap between the
    DSR term and the env's own StepResult.reward) puts the mix near
    DSR 50% / step-PnL 28% / hold-penalty 11% / diversity 10%
  * alpha_per_turnover / cost_per_turnover ARE THE NUMBERS TO WATCH, in bps
    per dollar traded. cost_per_turnover should sit near 1 bps and be almost
    flat -- it is the venue, not the policy, and a drifting one means the
    policy is churning into thinner bars. alpha_per_turnover is the actual
    question: it starts near 0 and only progress moves it. net_per_turnover
    is their difference and must be positive for any of this to be worth
    doing. All three are invariant to account size and trade count, which is
    exactly what net_worth is not.
  * net_worth is the SUM across ~100 streams, each with $10k -> baseline ~$1M.
    It is now a secondary number: it is a product of edge, size and trade
    count, so it moves for reasons unrelated to whether the policy knows
    anything, and a negative-edge policy can post a rising curve for a long
    time on a directional tape. Watch its DIRECTION, not the absolute number,
    and believe the per-turnover pair over it when they disagree
  * checkpoints: checkpoint_0.pt ... checkpoint_75.pt (every 25) +
    checkpoint_best.pt once EMA reward beats the warmup baseline
  * the live-path fixes (H1-H6) don't change training behavior -- they only
    matter when you point live/live_loop.py at a broker account

RED FLAGS:
  * any [FAIL] or AssertionError in diagnostics
  * entropy_discrete ~0 from rollout 1-2 (collapse), or NaN anywhere
  * total_trades == 0 in every rollout (env/action wiring or all-halted)
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Kaggle one-shot runner for Version_2.")
    p.add_argument("--fetch", action="store_true", help="fetch_alpaca.py (needs ALPACA secrets)")
    p.add_argument("--preprocess", action="store_true", help="preprocess.py (features + metadata)")
    p.add_argument("--diagnostics", action="store_true", help="run data/env/GPU diagnostics")
    p.add_argument("--train", action="store_true", help="run train.py (or train_ddp.py on 2+ GPUs)")
    p.add_argument("--book", action="store_true",
                   help="eval/xsec_book.py -- the P2 cost-aware cross-sectional "
                        "book and its lambda sweep, on the corrected panel")
    p.add_argument("--book-holds", type=int, nargs="+", default=[12, 24, 48],
                   help="holds in bars for --book (default 12 24 48)")
    p.add_argument("--book-lams", type=float, nargs="+", default=None,
                   help="lambdas for --book (default: the script's own grid)")
    p.add_argument("--book-variant", choices=("full", "nosize"), default="full",
                   help="'nosize' ablates the / cost sizing term")
    p.add_argument("--pretrain", action="store_true",
                   help="training/pretrain_trunk.py -- supervised fit of the "
                        "shared trunk against the tradeable forward return")
    p.add_argument("--pretrain-hold", type=int, default=24)
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument("--pretrained-trunk", type=str, default=None,
                   help="with --train: start PPO from this supervised trunk "
                        "(written by --pretrain). Ignored with --resume.")
    p.add_argument("--no-book-after-pretrain", dest="book_after_pretrain",
                   action="store_false", default=True,
                   help="skip scoring the pre-trained edge through the book")
    p.add_argument("--quick", action="store_true",
                   help="diagnostics + train with --total-rollouts 100 (data must already exist)")
    p.add_argument("--total-rollouts", type=int, default=100)
    p.add_argument(
        "--resume", type=str, default=None,
        help="checkpoint path to resume from, or 'latest' / 'best' (see train.py). "
             "Mutually exclusive with --fresh.",
    )
    p.add_argument(
        "--fresh", action="store_true",
        help="cold reset: start from random weights and delete existing checkpoints. "
             "Mutually exclusive with --resume. Without either, training refuses to "
             "start if checkpoints already exist.",
    )
    p.add_argument("--no-install", action="store_true", help="skip pip install of requirements.txt")
    p.add_argument("--no-ddp", action="store_true", help="force single-GPU train.py even on 2 GPUs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    os.chdir(SCRIPT_DIR)
    setup_secrets()

    if args.quick:
        if not cache_ready():
            raise SystemExit(
                "[run_kaggle] no processed data cached -- run `python run_kaggle.py --fetch --preprocess` "
                "once (or attach a cached dataset) before --quick."
            )
        args.diagnostics = True
        args.train = True

    any_stage = any([args.fetch, args.preprocess, args.diagnostics, args.train,
                     args.book, args.pretrain])
    if not any_stage:
        print(__doc__)
        return

    ensure_deps(args.no_install)

    if args.fetch:
        stage_fetch()
    if args.preprocess:
        stage_preprocess()
    if args.diagnostics:
        stage_diagnostics()
    if args.book:
        stage_book(args.book_holds, args.book_lams or [], args.book_variant)
    if args.pretrain:
        stage_pretrain(args.pretrain_hold, args.pretrain_epochs,
                       args.book_after_pretrain)
    if args.train:
        stage_train(args.total_rollouts, args.resume, args.fresh, args.no_ddp,
                    args.pretrained_trunk)

    print_summary(trained=bool(args.train))


if __name__ == "__main__":
    main()
