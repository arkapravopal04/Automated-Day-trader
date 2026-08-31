"""
eval/on_auction_curve.py -- Study C, step 10. The auction-cost sensitivity.

Declared in eval/PREREG_step10_14.md before it was run, and the declaration that
matters most is this one:

    NO AUCTION COST IS CHOSEN HERE AND THE FROZEN REFERENCE DOES NOT MOVE ON
    THIS OUTPUT.

`ratio` is alpha-per-turnover DIVIDED BY cost-per-turnover, so a low enough
assumption about what an auction cross costs clears the brief's bar by
arithmetic with no change whatever to the signal. There is no auction imbalance
data in this project, so there is nothing to declare a value from. The parameter
is swept and the whole curve is printed, together with the BREAKEVEN -- how
cheap the auction would have to be for the book to clear its bars on this
correction alone. That number is the honest form of the result: it converts
"could this pass?" into "what would have to be true for it to pass?", which is
answerable without a measurement the project does not have.

Cells, all pre-declared:

    (reference)  the uncorrected book -- quoted half-spread + adverse tick snap
    mocmoo       C1 alone: entry moved from the 15:55 quote to the 16:00 cross
    phi100/050/025/000
                 C2 alone: auction cost = phi x that leg's MEASURED half-spread
                 (entry 0.262, exit 1.464). phi = 0 is the pure-impact FLOOR,
                 a lower bound and not an estimate.
    both100/000  C1 and C2 together -- the book the strategy actually describes,
                 MOC in and MOO out, bracketed.

phi = 1.00 is still not the reference: it drops the adverse half-tick snap,
which a cross does not pay either. The reference is quoted beside every row so
the two are never confused.
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
CELLS = ["mocmoo", "phi100", "phi050", "phi025", "phi000", "both100", "both000",
         "exit100", "exit050", "exit025", "exit000"]
PHI = {"phi100": 1.00, "phi050": 0.50, "phi025": 0.25, "phi000": 0.00,
       "both100": 1.00, "both000": 0.00,
       "exit100": 1.00, "exit050": 0.50, "exit025": 0.25, "exit000": 0.00}
DESC = {
    "mocmoo":  "C1 only: entry at the 16:00 cross, quoted costs",
    "phi100":  "C2 phi 1.00  (entry 0.262 / exit 1.464 bps)",
    "phi050":  "C2 phi 0.50  (entry 0.131 / exit 0.732 bps)",
    "phi025":  "C2 phi 0.25  (entry 0.066 / exit 0.366 bps)",
    "phi000":  "C2 phi 0     pure impact -- LOWER BOUND",
    "both100": "C1 + C2 phi 1.00",
    "both000": "C1 + C2 phi 0        -- LOWER BOUND",
    # Appendix A. The entry leg is a QUOTED 15:55 fill and is left alone; only
    # the 09:30 cross is re-priced. This is the only correction in Study C that
    # costs no information -- MOO entry closes at 09:28, not 15:50.
    "exit100": "App A: EXIT-only cross, phi 1.00 (1.464 bps)",
    "exit050": "App A: EXIT-only cross, phi 0.50 (0.732 bps)",
    "exit025": "App A: EXIT-only cross, phi 0.25 (0.366 bps)",
    "exit000": "App A: EXIT-only cross, phi 0    -- LOWER BOUND",
}

RATIO_BAR = 2.0
SHARPE_BAR = 1.5


def load(fmt, ref, cell):
    out = {}
    for k in FOLDS:
        p = fmt.format(ref=ref, cell=cell, k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        h = d["holds"][next(iter(d["holds"]))]
        lam = h["chosen_lambda"]
        if lam is None:
            out[k] = (None, {}, {})
            continue
        row = next(r for r in h["rows"] if r["lam"] == lam)
        out[k] = (lam, row["val"], row["train"])
    return out


def load_ref(fmt_ref):
    out = {}
    for k in FOLDS:
        p = fmt_ref.format(k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        h = d["holds"][next(iter(d["holds"]))]
        lam = h["chosen_lambda"]
        row = next(r for r in h["rows"] if r["lam"] == lam)
        out[k] = (lam, row["val"], row["train"])
    return out


def m_se(vals):
    v = np.asarray([x if x is not None else np.nan for x in vals], dtype=float)
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return float("nan"), float("nan")
    return (float(ok.mean()),
            float(ok.std(ddof=1) / math.sqrt(ok.size)) if ok.size > 1
            else float("nan"))


def g(rec, key):
    if not rec or not rec[1]:
        return None
    x = rec[1].get(key)
    return float(x) if x is not None else None


def pooled_ratio(cells_data):
    """Sum of gross over sum of cost, weighted by turnover -- the pooled ratio.

    Not the mean of the five fold ratios. A fold with a tiny turnover
    denominator can post an enormous ratio and drag a mean; pooling weights each
    fold by the trading it actually did, which is the quantity the ratio is
    per-unit-of.
    """
    num = den = 0.0
    for k in FOLDS:
        rec = cells_data.get(k)
        a, c = g(rec, "alpha_per_turnover"), g(rec, "cost_per_turnover")
        t = g(rec, "turnover_per_bar")
        if None in (a, c, t) or not np.isfinite(a * c * t):
            continue
        num += a * t
        den += c * t
    return num / den if den > 0 else float("nan")


def breakeven(xs, ys, bar):
    """The x at which y first reaches `bar`, by linear interpolation.

    Reported as a bracket when the curve does not reach the bar inside the swept
    range: "below 0" is a real answer and it means the correction cannot get
    there on its own however cheap the cross is assumed to be.
    """
    pts = sorted((x, y) for x, y in zip(xs, ys)
                 if x is not None and y is not None
                 and np.isfinite(x) and np.isfinite(y))
    if len(pts) < 2:
        return None, "not enough points"
    # Ordered EXPENSIVE first: y is expected to rise as phi falls, so the walk
    # goes from the dearest assumption toward the cheapest and the first
    # crossing is the breakeven.
    pts = pts[::-1]
    if pts[0][1] >= bar:
        return pts[0][0], (f"already at or above the bar at the DEAREST point "
                           f"swept, phi {pts[0][0]:.2f}")
    if pts[-1][1] < bar:
        return None, (f"NEVER reached: even at phi {pts[-1][0]:.2f}, the cheapest "
                      f"point that could be evaluated, the curve only reaches "
                      f"{max(y for _, y in pts):.3f}")
    # y RISES as the walk proceeds (phi is falling), so the crossing is the
    # first segment whose left end is below the bar and right end at or above it.
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 < bar <= y1:
            f = (bar - y0) / (y1 - y0) if y1 != y0 else 0.0
            return x0 + f * (x1 - x0), "interpolated"
    return None, ("no crossing found -- the curve is NOT MONOTONE in phi, so a "
                  "single breakeven does not describe it")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Study C: the auction sensitivity curve.")
    ap.add_argument("--fmt", default="logs/p3/on/auction/{ref}_{cell}_f{k}.json")
    ap.add_argument("--refs", default="amendA,freeze3")
    ap.add_argument("--ref-json", default=None,
                    help="path template for the UNCORRECTED reference of the "
                         "first --refs entry, e.g. "
                         "logs/p3/on/freeze4/freeze4_f{k}.json")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    refs = [r.strip() for r in args.refs.split(",") if r.strip()]
    default_refjson = {"amendA": "logs/p3/on/freeze4/freeze4_f{k}.json",
                       "freeze3": "logs/p3/on/freeze3/freeze3_f{k}.json"}
    out = {}

    for ri, ref in enumerate(refs):
        data = {c: load(args.fmt, ref, c) for c in CELLS}
        if not any(data.values()):
            print(f"\n[skip] {ref}: nothing on disk")
            continue
        rj = (args.ref_json if (args.ref_json and ri == 0)
              else default_refjson.get(ref))
        base = load_ref(rj) if rj else {}

        print()
        print("=" * 112)
        print(f"STUDY C -- AUCTION PRICING, VAL, reference `{ref}`")
        print("=" * 112)
        print("rules: eval/PREREG_step10_14.md -- SENSITIVITY ONLY. No auction cost "
              "is chosen and the")
        print("       frozen reference does not move on this table.")
        print(f"source: {args.fmt.format(ref=ref, cell='{cell}', k='{k}')}")
        if base:
            print(f"uncorrected reference: {rj}")

        rows = [("(reference)", base, "uncorrected: quoted half-spread + "
                                      "adverse tick snap")] if base else []
        rows += [(c, data[c], DESC[c]) for c in CELLS if data[c]]

        print()
        print(f"  {'cell':<10}{'description':<46}"
              f"{'C/T':>8}{'A/T':>8}{'ratio':>8}{'+/-':>7}"
              f"{'pooled':>9}{'Sharpe':>9}{'+/-':>7}{'names':>8}")
        curve = {}
        for name, d, desc in rows:
            ct, _ = m_se([g(d.get(k), "cost_per_turnover") for k in FOLDS])
            at, _ = m_se([g(d.get(k), "alpha_per_turnover") for k in FOLDS])
            rt, rse = m_se([g(d.get(k), "ratio") for k in FOLDS])
            sh, sse = m_se([g(d.get(k), "sharpe") for k in FOLDS])
            nm, _ = m_se([g(d.get(k), "mean_names") for k in FOLDS])
            pr = pooled_ratio(d)
            print(f"  {name:<10}{desc:<46}{ct:>8.3f}{at:>8.3f}{rt:>8.3f}{rse:>7.2f}"
                  f"{pr:>9.3f}{sh:>9.3f}{sse:>7.2f}{nm:>8.1f}")
            curve[name] = dict(cost_per_turnover=ct, alpha_per_turnover=at,
                               ratio=rt, ratio_se=rse, pooled_ratio=pr,
                               sharpe=sh, sharpe_se=sse, mean_names=nm)

        # ---------------- the breakeven ----------------
        print()
        print("-" * 112)
        print("BREAKEVEN -- how cheap the cross would have to be, on this "
              "correction alone")
        print("-" * 112)
        for family, cells in (("C2 alone (15:55 entry)",
                               ["phi100", "phi050", "phi025", "phi000"]),
                              ("C1 + C2 (MOC in, MOO out)",
                               ["both100", "both000"]),
                              ("App A: EXIT-ONLY cross, quoted 15:55 entry",
                               ["exit100", "exit050", "exit025", "exit000"])):
            have = [c for c in cells if data.get(c)]
            if len(have) < 2:
                continue
            xs = [PHI[c] for c in have]
            pr = [pooled_ratio(data[c]) for c in have]
            sh = [m_se([g(data[c].get(k), "sharpe") for k in FOLDS])[0] for c in have]
            x_r, note_r = breakeven(xs, pr, RATIO_BAR)
            x_s, note_s = breakeven(xs, sh, SHARPE_BAR)
            print(f"  {family}")
            print(f"    pooled ratio >= {RATIO_BAR:.1f}: "
                  + (f"phi = {x_r:.3f}  ({note_r})" if x_r is not None
                     else f"{note_r}"))
            print(f"    net Sharpe   >= {SHARPE_BAR:.1f}: "
                  + (f"phi = {x_s:.3f}  ({note_s})" if x_s is not None
                     else f"{note_s}"))
            if x_r is not None:
                print(f"    -> the 09:30 cross would have to cost "
                      f"{x_r * 1.464:.3f} bps a side against the {1.464:.3f} bps "
                      f"quoted half-spread measured there,")
                if not family.startswith("App A"):
                    # The exit-only family leaves the 15:55 entry as a quoted
                    # fill on purpose, so quoting a required entry-cross price
                    # for it would describe a trade it does not make.
                    print(f"       and the 16:00 cross {x_r * 0.262:.3f} bps "
                          f"against {0.262:.3f}.")
                print(f"       Whether that is true is not knowable from "
                      f"anything in this project.")

        # ---------------- the second-order effect ----------------
        print()
        print("-" * 112)
        print("THE PART THAT IS NOT ARITHMETIC")
        print("-" * 112)
        print("  Cheaper cost does not only divide the ratio -- it lets more names "
              "clear the hurdle,")
        print("  so the book WIDENS and holds different things. That is a change in "
              "the strategy, not")
        print("  in its accounting, and it is the only part of this table that "
              "could carry information.")
        print()
        print(f"  {'cell':<10}{'names':>9}{'x reference':>13}{'flat %':>9}"
              f"{'max share':>11}{'val IC-ish (hit rate %)':>26}")
        base_nm = m_se([g(base.get(k), "mean_names") for k in FOLDS])[0] if base else float("nan")
        for name, d, _ in rows:
            nm, _ = m_se([g(d.get(k), "mean_names") for k in FOLDS])
            fl, _ = m_se([g(d.get(k), "flat_pct") for k in FOLDS])
            ms, _ = m_se([g(d.get(k), "max_share_max") for k in FOLDS])
            hr, _ = m_se([g(d.get(k), "hit_rate") for k in FOLDS])
            print(f"  {name:<10}{nm:>9.1f}{nm / base_nm if np.isfinite(base_nm) and base_nm else float('nan'):>13.2f}"
                  f"{fl:>9.1f}{ms:>11.3f}{hr:>26.1f}")

        out[ref] = {"curve": curve,
                    "cells": {c: {str(k): {"lambda": data[c][k][0],
                                           "val": data[c][k][1]}
                                  for k in sorted(data[c])}
                              for c in CELLS if data[c]}}

    print()
    print("-" * 112)
    print("READ BEFORE QUOTING")
    print("-" * 112)
    print("  1. A BAR CLEARED AT ANY phi < 1 IS A STATEMENT ABOUT THE ASSUMPTION,")
    print("     NOT ABOUT THE STRATEGY. Quote it with its phi and with the")
    print("     uncorrected reference beside it, always.")
    print("  2. phi = 0 is a floor, not an estimate. It says the cross is free")
    print("     apart from the size being crossed, which nothing here establishes.")
    print("  3. The frozen reference does not move on this table. It moves when an")
    print("     auction cost is MEASURED, and that needs cross and imbalance data")
    print("     this project does not have.")
    print("  4. `mocmoo` at quoted costs is the one cell that is a correction to")
    print("     the model rather than an assumption about a price. It is a")
    print("     candidate for adoption on its own; that is a later decision.")
    print("  5. TEST HAS NEVER BEEN READ.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"source": args.fmt, "prereg": "eval/PREREG_step10_14.md",
                   "ratio_bar": RATIO_BAR, "sharpe_bar": SHARPE_BAR,
                   "refs": out}, open(args.json, "w"), indent=1, default=float)
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
