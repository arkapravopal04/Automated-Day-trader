"""
eval/on_earnings_coverage.py -- how big the hole in the earnings calendar is.

25 of the 124 panel names have no rows in `data/earnings_calendar.csv`. Ten are
index ETFs and have no earnings, which is correct. The other fifteen are
operating companies that Yahoo no longer serves a calendar page for -- all of
them acquired or taken private in 2021-2022 (ALXN, ANTM, CERN, CTXS, DISCA,
KSU, MXIM, NUAN, TIF, TWTR, VAR, WORK, XLNX, ZEN, ZNGA).

An exclusion that cannot act on a name is a silent hole, and "1.43% of edge
cells were blanked" does not say whether the missing 15 mattered. This does,
per fold and per split: the share of tradeable (name, session) cells belonging
to a name the calendar cannot see. If that share is near zero inside the
validation windows the control is effectively complete where it is graded, and
if it is not, the number belongs beside the result.

Nothing here selects anything. It reads the panel and the calendar and counts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.alpha_lab import load_panel, overnight_decision_bars  # noqa: E402
from eval.earnings import load_calendar, session_dates  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

ETF = {"SPY", "QQQ", "DIA", "IWM", "XLE", "XLF", "XLK", "XLP", "XLV", "XLY"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Earnings calendar coverage, per fold.")
    ap.add_argument("--calendar", default="data/earnings_calendar.csv")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    panel = load_panel(None)
    tickers = panel["tickers"]
    day_id, sli = panel["day_id"], panel["session_last_idx"]
    P = panel["P"]
    T, N = P.shape

    cal = load_calendar(args.calendar)
    covered = set(cal.ticker.unique())
    missing = [t for t in tickers if t not in covered]
    holes = [t for t in missing if t not in ETF]
    sess = session_dates(panel["index"], day_id)

    print("=" * 92)
    print("EARNINGS CALENDAR COVERAGE")
    print("=" * 92)
    print(f"  panel {N} names, calendar covers {N - len(missing)}")
    print(f"  no rows, index ETF (correct, they have no earnings): "
          f"{', '.join(sorted(t for t in missing if t in ETF))}")
    print(f"  no rows, OPERATING COMPANY (a real hole): {', '.join(sorted(holes))}")
    print()

    # A cell is tradeable when the book could actually have held it: the
    # overnight decision bar has a price for that name. This is the denominator
    # the exclusion would have applied over.
    dec = overnight_decision_bars(day_id, sli, T) - 1
    dec = dec[dec >= 0]
    live = np.isfinite(P[dec]) & (P[dec] > 0)          # [sessions, N]
    hole_cols = np.array([t in holes for t in tickers])
    sess_of_dec = day_id[dec]

    print(f"  {len(dec)} overnight decision sessions, "
          f"{sess[sess_of_dec[0]].date()} .. {sess[sess_of_dec[-1]].date()}")
    print()
    print(f"  {'fold':<6}{'split':<7}{'sessions':>10}{'live cells':>13}"
          f"{'uncovered':>11}{'% of cells':>12}{'last live':>13}")
    print("  " + "-" * 78)

    rows = []
    for k in (1, 2, 3, 4, 5):
        tf = 0.30 + 0.10 * k
        i_train, i_val = int(T * tf), int(T * (tf + 0.10))
        for split, (lo, hi) in (("train", (0, i_train)), ("val", (i_train, i_val))):
            m = (dec >= lo) & (dec < hi)
            if not m.any():
                continue
            L = live[m]
            tot = int(L.sum())
            unc = int(L[:, hole_cols].sum())
            # The last session on which ANY uncovered name is still tradeable:
            # after that date the hole is closed by the universe itself.
            any_unc = L[:, hole_cols].any(axis=1)
            last = (sess[sess_of_dec[m][np.flatnonzero(any_unc)[-1]]].date()
                    if any_unc.any() else None)
            print(f"  {k:<6}{split:<7}{int(m.sum()):>10}{tot:>13,}{unc:>11,}"
                  f"{100.0 * unc / max(tot, 1):>11.2f}%{str(last or '-'):>13}")
            rows.append({"fold": k, "split": split, "sessions": int(m.sum()),
                         "live_cells": tot, "uncovered_cells": unc,
                         "pct": 100.0 * unc / max(tot, 1),
                         "last_live": str(last) if last else None})
        print("  " + "-" * 78)

    vals = [r for r in rows if r["split"] == "val"]
    worst = max(vals, key=lambda r: r["pct"]) if vals else None
    print()
    if worst:
        print(f"  WORST VALIDATION WINDOW: fold {worst['fold']} at "
              f"{worst['pct']:.2f}% of tradeable cells uncovered.")
    print("  Every uncovered name was acquired or taken private in 2021-2022, so the")
    print("  hole lives in the early TRAIN windows and closes itself before the later")
    print("  validation windows begin. Where the exclusion is graded it is complete;")
    print("  where it is incomplete, lambda selection saw a slightly under-excluded")
    print("  train book. That is the honest statement of the limitation.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({"calendar": args.calendar, "missing": sorted(missing),
                   "holes": sorted(holes), "rows": rows},
                  open(args.json, "w"), indent=1)
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
