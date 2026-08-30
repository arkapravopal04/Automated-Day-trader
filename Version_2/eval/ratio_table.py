"""
eval/ratio_table.py -- section 03's ratio table, rebuilt on the richer panel.

    ALPHA/TURN = gross return per bar / one-way turnover per bar
    COST/TURN  = one-way cost in bps for the names actually traded
    RATIO      = ALPHA/TURN / COST/TURN   -> must exceed 1.0

Section 03's claim is that ALPHA/TURN is close to INVARIANT across the things
you would normally tune -- 9 features or 63, four ways of tiering the universe,
selectivity from 100% to 10%, holds from 1 bar to 12 -- moving only between
0.15 and 0.28 bps. That stability is what makes it a target rather than a
coincidence, and it is the number P3 exists to move.

This script runs section 03's six configurations and prints its columns. It
contains no book logic of its own: every row is `xsec_book.main()` with
different flags, so the cost model, the execution frame, the session cap and
the train-only selection are the ones already audited there, not a second
implementation that could drift from them.

ONE DIFFERENCE FROM THE PUBLISHED TABLE, AND IT MATTERS FOR ONE COLUMN.
Section 03 priced cost as "per-name half-tick + 0.05 bps fees". `xsec_book.py`
charges `EnvConfig`'s model instead -- half-spread floored at a half tick,
PLUS a half-tick adverse snap, plus commission, plus impact -- which is
roughly double. So COST/TURN and RATIO here are strictly harsher than the
published ones and should not be compared cell to cell.

ALPHA/TURN is unaffected: it is gross alpha over turnover and no cost term
enters it. That is the column P3 is judged on, and it is directly comparable
to the published 0.15-0.28 bps range.

Usage
-----
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min \\
        python eval/ratio_table.py
    python eval/ratio_table.py --json logs/ratio_table.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval.xsec_book as xsec_book  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


# (label, extra argv). Every row fixes lambda explicitly, because this table is
# NOT a lambda sweep -- each row is one named configuration, and letting
# xsec_book pick lambda per row would silently make the rows incomparable.
#
#   --variant nosize   w proportional to excess edge          "edge-weighted"
#   --variant full     w proportional to excess edge / cost   "sized edge / cost"
#   --lam 0            no hurdle: trade the whole cross-section
#   --lam 1            hurdle: an edge must pay for its own round trip once
CONFIGURATIONS = [
    ("all names, edge-weighted, hold 1",        ["--hold", "1",  "--lam", "0", "--variant", "nosize"]),
    ("all names, edge-weighted, hold 6",        ["--hold", "6",  "--lam", "0", "--variant", "nosize"]),
    ("tick < 0.6 bps, hold 6",                  ["--hold", "6",  "--lam", "0", "--variant", "nosize",
                                                 "--tick-max-bps", "0.6"]),
    ("tick < 0.45 bps, hold 12",                ["--hold", "12", "--lam", "0", "--variant", "nosize",
                                                 "--tick-max-bps", "0.45"]),
    ("all names, sized edge / cost, hold 6",    ["--hold", "6",  "--lam", "0", "--variant", "full"]),
    ("all names, cost hurdle, hold 24",         ["--hold", "24", "--lam", "1", "--variant", "full"]),
]


def run_row(extra_argv, verbose=False):
    """One configuration -> its validation summary dict, via xsec_book itself."""
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        argv = list(extra_argv) + ["--json", tmp]
        sink = io.StringIO()
        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
        # A row can be legitimately unsatisfiable -- `--tick-max-bps 0.45` on a
        # panel of cheap names raises SystemExit from xsec_book's own guard.
        # Letting that propagate would discard the five rows that did run, and
        # on the full panel those are ~100 minutes of compute. Report the row
        # as unavailable and keep going.
        try:
            with ctx:
                xsec_book.main(argv)
            with open(tmp) as fh:
                out = json.load(fh)
            # One hold, one lambda per row, so there is exactly one cell to read.
            hold = next(iter(out["holds"].values()))
            return hold["rows"][0]["val"], hold["rows"][0]["train"], None
        except SystemExit as exc:
            return None, None, str(exc)
        except Exception as exc:
            return None, None, f"{type(exc).__name__}: {exc}"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Section 03's ratio table.")
    ap.add_argument("--tickers", type=int, default=None, help="cap universe size")
    ap.add_argument("--configs", type=str, default=None,
                    help="comma-separated 1-based row numbers to run (default all). "
                         "Row 6 (the lambda=1 cost hurdle) is the degenerate corner "
                         "-- it stands flat almost always, so on a short fold its "
                         "ALPHA/TURN is computed on a handful of bets and is noise")
    ap.add_argument("--ridge-alpha", default=None,
                    help="forwarded to xsec_book; 'auto' selects the penalty on a "
                         "train-only inner holdout, per configuration")
    ap.add_argument("--verbose", action="store_true",
                    help="let each xsec_book run print its own output too")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    common = ["--tickers", str(args.tickers)] if args.tickers else []
    if args.ridge_alpha:
        common += ["--ridge-alpha", args.ridge_alpha]

    hdr = (f"{'Configuration':<38}{'Names':>7}{'Turn/bar':>10}{'Alpha/turn':>12}"
           f"{'Cost/turn':>11}{'Ratio':>8}{'Net SR':>9}")
    print()
    print("VALIDATION -- ridge signal fit on TRAIN only, test never read")
    print(hdr)
    print("-" * len(hdr))

    chosen = CONFIGURATIONS
    if args.configs:
        want = {int(x) for x in args.configs.split(",")}
        chosen = [c for i, c in enumerate(CONFIGURATIONS, 1) if i in want]

    rows = []
    for i, (label, extra) in enumerate(chosen, 1):
        # Announced BEFORE the run, not after: each row is ~20 minutes on the
        # full panel and xsec_book's own output is suppressed, so without this
        # the table is indistinguishable from a hang for its first third.
        print(f"  ... running {i}/{len(chosen)}: {label}", flush=True)
        val, tr, err = run_row(common + extra, verbose=args.verbose)
        if err is not None:
            print(f"{label:<38}{'--':>7}{'--':>10}{'--':>12}{'--':>11}{'--':>8}{'--':>9}"
                  f"   UNAVAILABLE: {err}")
            rows.append({"config": label, "argv": extra, "error": err})
            continue
        print(f"{label:<38}{val['mean_names']:>7.1f}{val['turnover_per_bar']:>10.3f}"
              f"{val['alpha_per_turnover']:>12.3f}{val['cost_per_turnover']:>11.3f}"
              f"{val['ratio']:>8.2f}{val['sharpe']:>9.2f}")
        rows.append({"config": label, "argv": extra, "val": val, "train": tr})

    print()
    print("ALPHA/TURN is the P3 number. Section 03 measured it at 0.15-0.28 bps and")
    print("found it near-invariant to feature count, tiering, selectivity and hold.")
    print("It carries no cost term, so it is comparable to the published table.")
    print("COST/TURN and RATIO are NOT: this charges EnvConfig's model (half-spread")
    print("floored at a half tick + half-tick adverse snap + commission + impact),")
    print("roughly double section 03's half-tick-plus-fees. Both are harsher here.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"raw_dir": xsec_book.RAW_DIR, "rows": rows}, fh, indent=2, default=float)
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
