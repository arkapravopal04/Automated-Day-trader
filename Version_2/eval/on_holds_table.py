"""
eval/on_holds_table.py -- Study B, step 11. The holding-period curve.

Declared in eval/PREREG_step10_14.md before it was run. NOTHING IS SELECTED
HERE. The reference stays at hold 1; picking the best cell of a five-point grid
on five folds is exactly what the walk-forward protocol exists to prevent, so
the grid is reported whole and the reader is handed the sample size beside every
number.

WHAT THE CURVE IS MEASURING. Hold h cuts turnover per session to 2/h BY
CONSTRUCTION -- an overnight book at hold 1 does a full round trip every
session, which is why COST/TURN ~1.96 bps consumes about half of gross. If the
reversal has any persistence past one night, the ratio improves mechanically.
What it costs is h-1 DAY sessions of exposure per period that the reversal edge
does not forecast at all.

THE SAMPLE SHRINKS AS FAST AS THE TURNOVER. A validation window that holds 149
periods at hold 1 holds 29 at hold 5. Sharpe on 29 periods has a standard error
near +/-2.5 before any of this starts, so the right-hand end of this curve is
not a measurement. `periods` is printed on its own line for that reason, and it
is the first row to read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

FOLDS = (1, 2, 3, 4, 5)


def load_ref(fmt, ref):
    """-> {hold: {fold: (lam, val, train, ic)}} over whatever is on disk."""
    out = {}
    for k in FOLDS:
        p = fmt.format(ref=ref, k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for hold, h in d["holds"].items():
            lam = h["chosen_lambda"]
            rec = out.setdefault(int(hold), {})
            if lam is None:
                rec[k] = (None, {}, {}, h.get("val_ic"))
                continue
            row = next(r for r in h["rows"] if r["lam"] == lam)
            rec[k] = (lam, row["val"], row["train"], h.get("val_ic"))
    return out


def mean_se(vals):
    v = np.asarray([x if x is not None else np.nan for x in vals], dtype=float)
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return float("nan"), float("nan")
    m = float(ok.mean())
    se = float(ok.std(ddof=1) / math.sqrt(ok.size)) if ok.size > 1 else float("nan")
    return m, se


def block(title, byhold, getter, fmt="{:>9.2f}", note=None):
    print()
    print(title)
    if note:
        print(f"  {note}")
    hdr = "".join(f"{'f' + str(k):>9}" for k in FOLDS)
    print(f"  {'hold':<10}{hdr}{'mean':>10}{'se':>9}")
    for h in sorted(byhold):
        vals = [getter(byhold[h].get(k)) for k in FOLDS]
        m, se = mean_se(vals)
        cells = "".join(
            (fmt.format(v) if v is not None and np.isfinite(v) else f"{'-':>9}")
            for v in vals
        )
        lab = f"{h} night" + ("s" if h != 1 else "")
        print(f"  {lab:<10}{cells}{m:>10.2f}{se:>9.4f}")


def val(rec, key):
    if not rec or not rec[1]:
        return None
    x = rec[1].get(key)
    return float(x) if x is not None else None


def trn(rec, key):
    if not rec or not rec[2]:
        return None
    x = rec[2].get(key)
    return float(x) if x is not None else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Study B: the hold curve.")
    ap.add_argument("--fmt", default="logs/p3/on/holds/{ref}_f{k}.json")
    ap.add_argument("--refs", default="amendA,freeze3",
                    help="comma-separated; the first is the one the conclusion "
                         "is drawn from, the rest are the pre-registered "
                         "cross-checks")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    loaded = {r: load_ref(args.fmt, r) for r in refs}
    if not any(loaded.values()):
        raise SystemExit(f"no hold jsons at {args.fmt} -- run run_studyB_holds.sh")

    out = {}
    for ref in refs:
        byhold = loaded[ref]
        if not byhold:
            print(f"\n[skip] {ref}: nothing on disk at {args.fmt.format(ref=ref, k='*')}")
            continue
        print()
        print("=" * 108)
        print(f"STUDY B -- HOLDING PERIOD, VAL, reference `{ref}`")
        print("=" * 108)
        print("rules: eval/PREREG_step10_14.md -- MEASUREMENT ONLY, no hold is "
              "selected here")
        print(f"source: {args.fmt.format(ref=ref, k='{k}')}")

        block("PERIODS IN THE VALIDATION WINDOW  <-- read this row first", byhold,
              lambda r: val(r, "periods"), "{:>9.0f}",
              note="the denominator of every number below it. At hold 5 a fold "
                   "carries ~29 periods,\n  and a Sharpe on 29 periods is not a "
                   "measurement of anything.")
        block("turnover per BAR (the book's own units)", byhold,
              lambda r: val(r, "turnover_per_bar"), "{:>9.5f}",
              note="this is the quantity hold h is bought with: it should fall "
                   "as ~1/h, by construction.")
        block("lambda", byhold, lambda r: (r[0] if r else None))
        block("ALPHA/TURN  bps", byhold, lambda r: val(r, "alpha_per_turnover"))
        block("COST/TURN   bps", byhold, lambda r: val(r, "cost_per_turnover"),
              note="falls with h only through the carry term; the per-round-trip "
                   "spread is unchanged.\n  What falls with h is turnover, not cost "
                   "per unit of it.")
        block("ratio  (alpha/turn over cost/turn)", byhold,
              lambda r: val(r, "ratio"))
        block("net Sharpe (annual)", byhold, lambda r: val(r, "sharpe"))
        block("net bps / PERIOD", byhold, lambda r: val(r, "net_bps"),
              note="per period, not per night. At hold h a period spans h nights, "
                   "so these are not\n  comparable down the column without dividing "
                   "by h.")
        block("gross bps / PERIOD", byhold, lambda r: val(r, "gross_bps"))
        block("cost  bps / PERIOD", byhold, lambda r: val(r, "cost_bps"))
        block("names in book", byhold, lambda r: val(r, "mean_names"), "{:>9.1f}")
        block("hit rate % (active)", byhold, lambda r: val(r, "hit_rate"), "{:>9.1f}")
        block("val block IC (h-night target)", byhold,
              lambda r: (r[3] if r else None), "{:>9.4f}",
              note="the edge is REFITTED on the h-night target at every hold, so "
                   "this is the IC of a\n  different estimator at every row, not the "
                   "same one decaying.")
        block("TRAIN ratio", byhold, lambda r: trn(r, "ratio"))

        # The one arithmetic the study exists to check.
        print()
        print("-" * 108)
        print("DID THE TURNOVER ACTUALLY FALL AS 1/h, AND DID ALPHA/TURN SURVIVE IT?")
        print("-" * 108)
        base = byhold.get(1)
        if base:
            b_turn, _ = mean_se([val(base.get(k), "turnover_per_bar") for k in FOLDS])
            b_alpha, _ = mean_se([val(base.get(k), "alpha_per_turnover") for k in FOLDS])
            b_ratio, _ = mean_se([val(base.get(k), "ratio") for k in FOLDS])
            print(f"  {'hold':<8}{'turn/bar':>12}{'x hold-1':>11}{'ideal 1/h':>11}"
                  f"{'ALPHA/TURN':>13}{'x hold-1':>11}{'ratio':>9}{'x hold-1':>11}")
            for h in sorted(byhold):
                t, _ = mean_se([val(byhold[h].get(k), "turnover_per_bar") for k in FOLDS])
                a, _ = mean_se([val(byhold[h].get(k), "alpha_per_turnover") for k in FOLDS])
                r, _ = mean_se([val(byhold[h].get(k), "ratio") for k in FOLDS])
                print(f"  {h:<8}{t:>12.5f}{t / b_turn:>11.3f}{1.0 / h:>11.3f}"
                      f"{a:>13.3f}{(a / b_alpha if b_alpha else float('nan')):>11.3f}"
                      f"{r:>9.3f}{(r / b_ratio if b_ratio else float('nan')):>11.3f}")
            print()
            print("  `x hold-1` against `ideal 1/h` is the only part of this table "
                  "that is not a")
            print("  measurement -- the turnover ratio is mechanical and a deviation "
                  "from 1/h means the")
            print("  schedule is not doing what it says. ALPHA/TURN is where the "
                  "question actually lives:")
            print("  it holds up only if the reversal survives past one night.")
        out[ref] = {str(h): {str(k): {"lambda": byhold[h][k][0],
                                      "val": byhold[h][k][1],
                                      "train": byhold[h][k][2],
                                      "val_ic": byhold[h][k][3]}
                             for k in sorted(byhold[h])}
                    for h in sorted(byhold)}

    print()
    print("-" * 108)
    print("READ BEFORE QUOTING")
    print("-" * 108)
    print("  1. NO HOLD IS SELECTED. The reference stays at hold 1 unless a separate")
    print("     pre-registration adopts another.")
    print("  2. The sample shrinks as 1/h. The right-hand end of the curve has a")
    print("     standard error larger than any effect it could show.")
    print("  3. Hold h carries the book through h-1 DAY sessions the reversal edge")
    print("     does not forecast. Lower turnover is not free.")
    print("  4. TEST HAS NEVER BEEN READ.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"source": args.fmt, "prereg": "eval/PREREG_step10_14.md",
                   "refs": out}, open(args.json, "w"), indent=1, default=float)
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
