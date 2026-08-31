"""
eval/on_fold_diff.py -- WHICH CELLS a fold's risk-control change came out of.

`on_fold_decomp.py` describes one book. The quantity in question here is a
DIFFERENCE between two -- fold 1 loses 0.68 of net Sharpe between the `base` and
`both` arms of step 5 -- and a difference cannot be read off two independent
decompositions, because the two books hold different names on the same night and
the per-name and per-month margins both hide that.

So this builds all four arms IN ONE PROCESS, on one panel, one edge, one causal
vol, one schedule, one cost model, AT A MATCHED LAMBDA, and keeps every
[period, name] grid:

    A  base   raw edge,      no cap
    E  earn   excluded edge, no cap
    C  cap    raw edge,      capped
    B  both   excluded edge, capped

MATCHED LAMBDA IS THE POINT. Each arm of step 5 re-selects lambda on train, and
on fold 1 that moved base from 0.75 to 1.00. A diff of the two arms as they were
run is therefore the controls AND the lambda move added together, and neither
term is recoverable from it. Here lambda is held fixed across all four arms, so
the difference is the controls and nothing else; the lambda move is a separate
run of this script at the other lambda.

The gap is reported two ways, because the two answer different questions and
only one of them can be taken down to individual cells.

AT ARM LEVEL, symmetric -- what each control does on its own:

    B - A  =  (E - A)  +  (C - A)  +  (B - E - C + A)
              calendar    cap        interaction

AT CELL LEVEL, sequential -- calendar first, then the cap on the book the
calendar left behind:

    B - A  =  (E - A)  +  (B - E)
              calendar    cap GIVEN calendar

There is no interaction row in the second: the cap term carries it, because the
cap's effect genuinely is not the same before and after the calendar. The
calendar narrows the book, and KAPPA/n_selected LOOSENS as it does -- which is
the defect Appendix A of the pre-registration exists to close. The gap between
the two cap numbers is exactly that loosening, and it is worth reading.

Cells of the calendar term are tagged EXCLUDED (the calendar forbade a position
A held) or reallocation; cells of the cap term are tagged CLIPPED (|w| sits at
the bar's threshold) or reallocation.

THE QUESTION THIS ANSWERS. "The controls are correct but expensive" and "the
controls are removing signal" are different findings with different remedies,
and a Sharpe delta alone does not distinguish them. The discriminator is the
COUNTERFACTUAL VALUE of what each control took away: what base earned on the
cells it was forbidden to hold, and what the shaved weight would have earned.
Negative, the control removed losses and what it cost is reallocation and
turnover. Positive, it removed profit -- BUT a sum that is one or two enormous
cells with both signs present is a lottery ticket, not signal, and which way it
landed in one 148-night window says nothing about its expectation. The
dispersion is printed next to the sum for exactly that reason.

NOTHING HERE SELECTS ANYTHING. Lambda is passed in, the split fractions come
from the environment exactly as the run script set them, test is never touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import TRAIN_FRAC, VAL_FRAC  # noqa: E402
from eval.alpha_lab import (  # noqa: E402
    BARS_PER_DAY,
    TRADING_DAYS,
    forward_return_bps,
    load_panel,
    overnight_decision_bars,
)
from eval.on_fold_decomp import run_book_detail  # noqa: E402
from eval.xsec_book import (  # noqa: E402
    env_cost_constants,
    execution_frame,
    measure_liquidity,
    overnight_schedule,
    reversal_edge,
    trailing_overnight_vol,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def score(NET, per_year):
    """(net bps/period, annualised Sharpe) of a [period, name] net grid."""
    s = NET.sum(axis=1)
    sd = float(s.std(ddof=1))
    sh = (float(s.mean()) / sd) * math.sqrt(per_year) if sd > 0 else float("nan")
    return float(s.mean()), sh


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Diff two risk-control arms of one fold, cell by cell.")
    ap.add_argument("--lam", type=float, required=True,
                    help="the MATCHED lambda all four arms are built at")
    ap.add_argument("--earnings-calendar", type=str, required=True)
    ap.add_argument("--max-weight-mult", type=float, default=3.0)
    ap.add_argument("--max-weight-frac", type=float, default=None,
                    help="absolute per-name cap, share of gross (Appendix A)")
    ap.add_argument("--open-spread-bps", type=float, default=1.464)
    ap.add_argument("--close-spread-bps", type=float, default=0.262)
    ap.add_argument("--carry-bps", type=float, default=0.20)
    ap.add_argument("--risk-scale", choices=("none", "vol"), default="vol")
    ap.add_argument("--risk-window", type=int, default=60)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--min-names", type=int, default=2)
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--reconcile-base", type=str, default=None,
                    help="the arm's own json, to prove arm A is the book on record")
    ap.add_argument("--reconcile-both", type=str, default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    cap_desc = f"{args.max_weight_mult:g}x equal weight"
    if args.max_weight_frac is not None:
        cap_desc = (f"min({args.max_weight_mult:g}x equal weight, "
                    f"{args.max_weight_frac:g} of gross)")

    print("=" * 104)
    print(f"FOLD DIFF -- four arms at MATCHED lambda {args.lam}, split {args.split}")
    print("=" * 104)
    print(f"[cap]      {cap_desc}")
    print(f"[calendar] {args.earnings_calendar}")

    k = env_cost_constants()
    panel = load_panel(None)
    P, tickers = panel["P"], panel["tickers"]
    day_id, sli, index = panel["day_id"], panel["session_last_idx"], panel["index"]
    T, N = P.shape

    i_train = int(T * TRAIN_FRAC)
    i_val = int(T * (TRAIN_FRAC + VAL_FRAC))
    print(f"[split]    train 0:{i_train}  val {i_train}:{i_val}  "
          f"test {i_val}:{T} (untouched)")

    Px, _ = execution_frame(index, tickers, sli, column="open")
    adv, sigma = measure_liquidity(index, tickers, day_id, i_train, P)

    fwd = forward_return_bps(Px, 1, None)
    keep = np.zeros(fwd.shape[0], dtype=bool)
    keep[overnight_decision_bars(day_id, sli, T) - 1] = True
    fwd = np.where(keep[:, None], fwd, np.nan).astype(np.float32)

    edge, edge_desc = reversal_edge(P, fwd, i_train, day_id)
    print(f"[edge]     {edge_desc}")

    from eval.earnings import (apply_to_edge, assert_mapping, exclusion_mask,
                               load_calendar)
    cal = load_calendar(args.earnings_calendar)
    assert_mapping(index, tickers, day_id, cal)
    emask, estats = exclusion_mask(index, tickers, day_id, cal, report=False)
    edge_x = apply_to_edge(edge, emask)

    risk = None
    if args.risk_scale == "vol":
        risk = trailing_overnight_vol(fwd, day_id, sli, T, window=args.risk_window)

    t0, t1 = (i_train, i_val) if args.split == "val" else (0, i_train)
    sched = overnight_schedule(t0, t1, day_id, sli, T)
    n_bars = t1 - t0
    win = index[[sched[0][0], sched[-1][1]]].tz_convert("America/New_York")
    print(f"[schedule] {len(sched)} {args.split} gap periods, "
          f"{str(win[0])[:10]} -> {str(win[1])[:10]}")

    def build(e, capped):
        return run_book_detail(
            e, fwd, Px, adv, sigma, k, sched, day_id, args.lam, args.capital,
            args.min_names, risk=risk,
            spread_entry=args.close_spread_bps,
            spread_exit=args.open_spread_bps,
            carry_bps=args.carry_bps,
            max_weight_mult=(args.max_weight_mult if capped else None),
            max_weight_frac=(args.max_weight_frac if capped else None))

    A = build(edge, False)      # base
    E = build(edge_x, False)    # earnings only
    C = build(edge, True)       # cap only
    B = build(edge_x, True)     # both

    ent = A["entries"]
    n_per = len(ent)
    years = n_bars / (BARS_PER_DAY * TRADING_DAYS)
    per_year = n_per / years

    NETS = {n: (d["G"] - d["C"]) for n, d in (("A", A), ("E", E), ("C", C), ("B", B))}

    # ---------------- reconciliation ----------------
    print()
    print("-" * 104)
    print("RECONCILIATION -- arms A and B against the jsons they are supposed to be")
    print("-" * 104)
    ok_all = True
    for tag, det, path in (("A base", A, args.reconcile_base),
                           ("B both", B, args.reconcile_both)):
        g = float(det["G"].sum(axis=1).mean())
        c = float(det["C"].sum(axis=1).mean())
        nb, sh = score(NETS[tag[0]], per_year)
        line = (f"  {tag:<8} here  gross {g:+8.4f}  cost {c:+7.4f}  net {nb:+8.4f} "
                f"bps/period  Sharpe {sh:+6.3f}")
        if not path or not os.path.exists(path):
            print(line + "   [no json given -- NOT reconciled]")
            continue
        h = json.load(open(path))["holds"]["1"]
        r = next((x for x in h["rows"] if abs(x["lam"] - args.lam) < 1e-9), None)
        if r is None:
            print(line + f"   [lambda {args.lam} not swept in {path}]")
            continue
        v = r[args.split]
        d = max(abs(g - v["gross_bps"]), abs(c - v["cost_bps"]))
        ok_all &= bool(d < 1e-6)
        print(line)
        print(f"  {'':<8} json  gross {v['gross_bps']:+8.4f}  cost "
              f"{v['cost_bps']:+7.4f}  net {v['net_bps']:+8.4f} bps/period  "
              f"Sharpe {v['sharpe']:+6.3f}   -> "
              f"{'OK' if d < 1e-6 else f'MISMATCH {d:.2e}'}")
    if not ok_all:
        print("  [reconcile] an arm does not reproduce its json. The diff below is "
              "of a different\n              book and nothing in it is admissible.")

    # ---------------- the gap, decomposed ----------------
    print()
    print("=" * 104)
    print(f"THE GAP AT MATCHED LAMBDA {args.lam}")
    print("=" * 104)
    print(f"  {'arm':<28}{'gross':>10}{'cost':>10}{'net':>10}{'Sharpe':>10}"
          f"{'names':>9}")
    stats = {}
    for tag, det, label in (("A", A, "base   raw edge, no cap"),
                            ("E", E, "earn   excluded, no cap"),
                            ("C", C, "cap    raw edge, capped"),
                            ("B", B, "both   excluded, capped")):
        nb, sh = score(NETS[tag], per_year)
        g = float(det["G"].sum(axis=1).mean())
        c = float(det["C"].sum(axis=1).mean())
        nm = float(det["selected"].mean())
        stats[tag] = dict(gross=g, cost=c, net=nb, sharpe=sh, names=nm)
        print(f"  {label:<28}{g:>10.3f}{c:>10.3f}{nb:>10.3f}{sh:>10.3f}{nm:>9.1f}")

    print()
    print(f"  {'term':<28}{'d gross':>10}{'d cost':>10}{'d net':>10}{'d Sharpe':>10}")
    for label, x, y in (("calendar   (E - A)", "E", "A"),
                        ("cap        (C - A)", "C", "A")):
        print(f"  {label:<28}{stats[x]['gross'] - stats[y]['gross']:>10.3f}"
              f"{stats[x]['cost'] - stats[y]['cost']:>10.3f}"
              f"{stats[x]['net'] - stats[y]['net']:>10.3f}"
              f"{stats[x]['sharpe'] - stats[y]['sharpe']:>10.3f}")
    inter = {m: stats["B"][m] - stats["E"][m] - stats["C"][m] + stats["A"][m]
             for m in ("gross", "cost", "net", "sharpe")}
    print(f"  {'interaction':<28}{inter['gross']:>10.3f}{inter['cost']:>10.3f}"
          f"{inter['net']:>10.3f}{inter['sharpe']:>10.3f}")
    print(f"  {'-' * 66}")
    print(f"  {'TOTAL      (B - A)':<28}"
          f"{stats['B']['gross'] - stats['A']['gross']:>10.3f}"
          f"{stats['B']['cost'] - stats['A']['cost']:>10.3f}"
          f"{stats['B']['net'] - stats['A']['net']:>10.3f}"
          f"{stats['B']['sharpe'] - stats['A']['sharpe']:>10.3f}")
    print()
    print("  The Sharpe column does NOT add -- Sharpe is not linear in the net "
          "series. The\n  gross/cost/net columns do, exactly, and they are the "
          "ones the attribution below\n  is stated in.")

    # ---------------- cell attribution ----------------
    #
    # EACH CONTROL IS MEASURED AGAINST THE RIGHT COUNTERFACTUAL. Tagging cells
    # of (B - A) directly does not work: "clipped" is a statement about the cap,
    # defined against the same-edge uncapped book E, but (B - A) also contains
    # the calendar's effect on that cell, so a cap cell would be credited with
    # a calendar change. So the two controls are attributed on the two diffs
    # that isolate them --
    #
    #     calendar             (E - A)   same cap setting (off), edge differs
    #     cap GIVEN calendar   (B - E)   same edge (excluded), cap differs
    #
    # THIS IS A SEQUENTIAL DECOMPOSITION AND THE ORDER IS PART OF IT. The two
    # terms sum to (B - A) identically -- there is no interaction row, because
    # the second term already carries it. That is deliberate: the cap's effect
    # is not the same before and after the calendar, since the calendar narrows
    # the book and KAPPA/n_selected loosens as it does. The arm-level table
    # above reports the symmetric split, cap-ALONE (C - A) against the
    # interaction, and the difference between the two cap numbers is exactly
    # that loosening. Neither is wrong; they answer different questions, and
    # mixing them is what would be wrong.
    dNET = NETS["B"] - NETS["A"]
    dCAL = NETS["E"] - NETS["A"]
    dCAP = NETS["B"] - NETS["E"]
    WA, WB, WE = A["W"], B["W"], E["W"]
    selB = np.maximum(B["selected"], 1)
    thr = args.max_weight_mult / selB
    if args.max_weight_frac is not None:
        thr = np.minimum(thr, args.max_weight_frac)                 # [periods]
    EM = emask[ent]                                                 # [periods, N]

    held_A = np.abs(WA) > 0
    at_cap = np.abs(WB) >= thr[:, None] * (1.0 - 1e-6)
    would_clip = np.abs(WE) > thr[:, None] * (1.0 + 1e-6)

    tag_excl = EM & held_A                      # calendar forbade a held position
    tag_clip = at_cap & would_clip              # cap bound on this name

    tot = float(dNET.sum())
    print()
    print("=" * 104)
    print("WHERE THE CHANGE CAME FROM -- sequential: calendar first, then cap")
    print("=" * 104)
    print("  The cap term is the cap GIVEN the calendar, (both - earn). It carries")
    print("  the interaction, because the cap's effect genuinely depends on the")
    print("  calendar having narrowed the book first. See the note in the source.")
    print()
    print(f"  {'term':<26}{'cells':>9}{'d net bps':>12}{'% of change':>13}"
          f"  what it is")
    rows = []

    def line(label, grid, mask, what, key):
        s = float(grid[mask].sum())
        share = 100.0 * s / tot if abs(tot) > 1e-12 else float("nan")
        print(f"  {label:<26}{int(mask.sum()):>9}{s:>12.1f}{share:>12.1f}%  {what}")
        rows.append(dict(tag=key, cells=int(mask.sum()), d_net=s, share=share))
        return s

    c_direct = line("calendar, EXCLUDED", dCAL, tag_excl,
                    "the position the calendar forbade", "calendar_excluded")
    c_rest = line("calendar, reallocation", dCAL, ~tag_excl,
                  "book re-normalised around it", "calendar_realloc")
    p_direct = line("cap|calendar, CLIPPED", dCAP, tag_clip,
                    "|w| driven down to the bar's threshold", "cap_clipped")
    p_rest = line("cap|calendar, realloc", dCAP, ~tag_clip,
                  "book re-normalised around it", "cap_realloc")
    print(f"  {'-' * 62}")
    print(f"  {'TOTAL  (both - base)':<26}{'':>9}{tot:>12.1f}{100.0:>12.1f}%")
    ident = abs((c_direct + c_rest + p_direct + p_rest) - tot)
    print(f"  [identity] the four terms sum to (both - base) to {ident:.2e} bps")

    # ---------------- the discriminator ----------------
    print()
    print("=" * 104)
    print("CORRECT-BUT-EXPENSIVE, OR REMOVING SIGNAL?")
    print("=" * 104)
    print("  The value of what each control actually took away, measured on the")
    print("  book it took it from. Negative = it removed losses. Positive = it")
    print("  removed profit -- but read the DISPERSION beside it before calling")
    print("  that signal: a removed bet that is one or two enormous cells with")
    print("  both signs present is a lottery ticket, and which way it landed in")
    print("  148 nights is not evidence about its expectation.")
    print()
    NA = NETS["A"]
    counter = {}

    def removed(label, per_cell, mask, key, gap_mask=None):
        v = per_cell[mask]
        n = int(mask.sum())
        s = float(v.sum())
        pos = int((v > 0).sum())
        neg = int((v < 0).sum())
        top = float(v[np.argmax(np.abs(v))]) if n else float("nan")
        sd = float(v.std(ddof=1)) if n > 1 else float("nan")
        t = (float(v.mean()) / (sd / math.sqrt(n))) if (n > 1 and sd > 0) else float("nan")
        counter[key] = dict(cells=n, removed_net=s, n_positive=pos, n_negative=neg,
                            largest_cell=top, t=t)
        print(f"  {label}")
        print(f"    {n:>5} cells    net removed {s:+9.1f} bps    "
              f"{pos} positive / {neg} negative")
        print(f"    largest single cell {top:+9.1f} bps = "
              f"{100.0 * top / s if abs(s) > 1e-9 else float('nan'):5.1f}% of it;   "
              f"t across cells {t:+6.2f}")
        if gap_mask is not None:
            rr = fwd[ent][gap_mask]
            rr = rr[np.isfinite(rr)]
            if rr.size:
                print(f"    realised |gap| on those cells: mean "
                      f"{np.abs(rr).mean():.0f} bps, max {np.abs(rr).max():.0f} bps")
        if s > 0 and (abs(top) > 0.5 * abs(s) or not np.isfinite(t) or abs(t) < 2.0):
            print("    -> REMOVED A LOTTERY. Net-positive in this window, but the "
                  "sum is not\n       distinguishable from zero across its own "
                  "cells. Not evidence of signal.")
        elif s > 0:
            print("    -> REMOVED PROFIT, and the cells agree with each other. "
                  "This control is\n       taking signal out.")
        else:
            print("    -> REMOVED LOSSES. Correct; what it cost is reallocation "
                  "and turnover.")
        print()

    # The calendar removes whole positions, so what it took away is exactly what
    # base earned on those cells.
    removed("CALENDAR -- positions base held into a scheduled print",
            NA, tag_excl, "excluded", gap_mask=tag_excl)
    # The cap removes only the EXCESS weight, not the position, so what it took
    # away is the return on the weight difference, priced on the same edge (E).
    RET = np.nan_to_num(fwd[ent], nan=0.0)
    shaved = (WE - WB) * RET
    removed("CAP -- the weight shaved off names at the threshold",
            shaved, tag_clip, "clipped", gap_mask=tag_clip)

    # ---------------- the biggest individual cells ----------------
    order = np.argsort(dNET.ravel())
    half = max(args.top // 2, 1)
    show = np.concatenate([order[:half], order[::-1][:half]])
    print("-" * 104)
    print(f"LARGEST INDIVIDUAL (name, session) CHANGES -- worst {half}, "
          f"then best {half}")
    print("-" * 104)
    print(f"  {'name':<8}{'session':<12}{'tag':<10}{'d net':>9}{'w base':>10}"
          f"{'w both':>10}{'cap thr':>9}{'ret bps':>9}")
    for c in show:
        t_i, n_i = divmod(int(c), N)
        if abs(dNET[t_i, n_i]) < 1e-9:
            continue
        tg = ("EXCLUDED" if tag_excl[t_i, n_i] else
              "CLIPPED" if tag_clip[t_i, n_i] else "REALLOC")
        r = float(np.nan_to_num(fwd[ent[t_i], n_i], nan=0.0))
        print(f"  {tickers[n_i]:<8}{str(index[ent[t_i]])[:10]:<12}{tg:<10}"
              f"{dNET[t_i, n_i]:>9.1f}{WA[t_i, n_i]:>+10.4f}{WB[t_i, n_i]:>+10.4f}"
              f"{thr[t_i]:>9.3f}{r:>9.0f}")

    # ---------------- per name ----------------
    #
    # The cell list above is dominated by whichever single nights were largest,
    # and a name can be the biggest contributor without owning the biggest cell.
    # This is the same difference summed per name, so a REALLOCATION that
    # quietly walked the book into one collapsing ticker is visible as such.
    print()
    print("-" * 104)
    print("PER NAME -- the change summed over the window, worst and best")
    print("-" * 104)
    dname = dNET.sum(axis=0)
    nm_order = np.argsort(dname)
    print(f"  {'name':<8}{'d net':>10}{'% of change':>13}{'d cal':>10}{'d cap':>10}"
          f"{'excl':>7}{'clip':>7}{'held A':>8}{'held B':>8}")
    for i in np.concatenate([nm_order[:8], nm_order[::-1][:5]]):
        if abs(dname[i]) < 1e-9:
            continue
        print(f"  {tickers[i]:<8}{dname[i]:>10.1f}"
              f"{100.0 * dname[i] / tot if abs(tot) > 1e-12 else float('nan'):>12.1f}%"
              f"{dCAL[:, i].sum():>10.1f}{dCAP[:, i].sum():>10.1f}"
              f"{int(tag_excl[:, i].sum()):>7}{int(tag_clip[:, i].sum()):>7}"
              f"{int((np.abs(WA[:, i]) > 0).sum()):>8}"
              f"{int((np.abs(WB[:, i]) > 0).sum()):>8}")
    worst_nm = int(nm_order[0])
    print(f"  [names] the single worst name is {tickers[worst_nm]} at "
          f"{100.0 * dname[worst_nm] / tot:.1f}% of the entire change, over "
          f"{int((np.abs(dNET[:, worst_nm]) > 0).sum())} sessions")

    # ---------------- breadth, which is what the cap is applied to ----------------
    print()
    print("-" * 104)
    print("BREADTH AND WHAT THE CAP COULD DO WITH IT")
    print("-" * 104)
    nb_A, nb_B = A["selected"], B["selected"]
    print(f"  names selected, base : median {np.median(nb_A):>5.1f}  "
          f"p10 {np.percentile(nb_A, 10):>5.1f}  min {nb_A.min():>3d}")
    print(f"  names selected, both : median {np.median(nb_B):>5.1f}  "
          f"p10 {np.percentile(nb_B, 10):>5.1f}  min {nb_B.min():>3d}")
    peak_A = np.abs(WA).max(axis=1)
    peak_B = np.abs(WB).max(axis=1)
    print(f"  max name share, base : median {np.median(peak_A):.3f}  "
          f"p99 {np.percentile(peak_A, 99):.3f}  max {peak_A.max():.3f}")
    print(f"  max name share, both : median {np.median(peak_B):.3f}  "
          f"p99 {np.percentile(peak_B, 99):.3f}  max {peak_B.max():.3f}")
    infeas_n = 0
    if args.max_weight_frac is not None:
        infeas = (args.max_weight_frac * selB) < 1.0 + 1e-3
        infeas_n = int(infeas.sum())
        print(f"  bars too narrow for the absolute cap "
              f"(n < {math.ceil(1.0 / args.max_weight_frac)}): "
              f"{infeas_n} of {n_per} ({100.0 * infeas.mean():.1f}%)")

    # ---------------- per month, both arms side by side ----------------
    ny = index[ent].tz_convert("America/New_York")
    month = pd.PeriodIndex(ny, freq="M")
    print()
    print("-" * 104)
    print("PER MONTH -- where in the window the change landed")
    print("-" * 104)
    print(f"  {'month':<9}{'sess':>6}{'base net':>10}{'both net':>10}{'d net':>9}"
          f"{'d cal':>9}{'d cap':>9}{'':>9}{'names A':>9}{'names B':>9}")
    months = []
    sA = NETS["A"].sum(axis=1)
    sB = NETS["B"].sum(axis=1)
    for m in month.unique():
        sel = np.asarray(month == m)
        d_c = float(dCAL[sel].sum())
        d_p = float(dCAP[sel].sum())
        d_i = 0.0
        print(f"  {str(m):<9}{int(sel.sum()):>6}{sA[sel].mean():>10.2f}"
              f"{sB[sel].mean():>10.2f}{(sB[sel] - sA[sel]).mean():>9.2f}"
              f"{d_c:>9.1f}{d_p:>9.1f}{d_i:>9.1f}"
              f"{nb_A[sel].mean():>9.1f}{nb_B[sel].mean():>9.1f}")
        months.append(dict(month=str(m), sessions=int(sel.sum()),
                           base_net=float(sA[sel].mean()),
                           both_net=float(sB[sel].mean()),
                           d_calendar=d_c, d_cap=d_p, d_interaction=d_i))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(dict(
            lam=args.lam, split=args.split, n_periods=n_per,
            cap_mult=args.max_weight_mult, cap_frac=args.max_weight_frac,
            arms=stats, interaction=inter, attribution=rows,
            counterfactual=counter, infeasible_bars=infeas_n,
            months=months,
            earnings_stats={kk: vv for kk, vv in estats.items()
                            if kk != "names_with_no_events"},
        ), open(args.json, "w"), indent=1, default=float)
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
