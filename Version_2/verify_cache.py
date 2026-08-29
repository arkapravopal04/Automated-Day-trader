"""
verify_cache.py -- find interior gaps in a bar cache, and optionally fill them.

WHY THIS IS NOT OPTIONAL
------------------------
`fetch_alpaca.py` resumes from a ticker's NEWEST timestamp. That is the right
rule for extending a cache forward, and it is blind to a hole in the middle:
if a window fails and later windows succeed, the file ends current, every
subsequent run reports "already up to date", and the gap is permanent.

Observed on the 6-year 1-minute fetch of 2026-08-29: three tickers of 100 lost
one ~180-day window each to transient network faults (two RemoteDisconnected,
one read timeout). Nothing downstream would have raised. `preprocess.py` builds
rolling windows straight across a gap -- VOL_WINDOW=78 and the 20-day reversal
would span it, and the single bar at the seam would carry a multi-month move
priced as a five-minute return, which is the same shape of defect as an
unadjusted split.

So: verify, don't assume. This is the check the cache-contamination note asks
for, applied to continuity rather than to splits.

WHAT COUNTS AS A DEFECT
-----------------------
Sessions, not calendar days. Weekends and market holidays are absences by
construction, so the test is over TRADING DAYS, inferred from the busiest
ticker in the cache rather than from a holiday calendar -- the panel's own
union of trading days is the ground truth for the window actually fetched.

A session is defective if it is MISSING, or if it is SPARSE: present but
holding far fewer bars than the cache's own modal session. Sparseness is not a
refinement, it is the check that matters. A missing session is loud; a session
present at the wrong cadence looks exactly like data. Observed 2026-08-29: a
backfill run without ALPACA_BAR_MINUTES set inherited fetch_alpaca's default
of 5, wrote 5-minute extended-hours bars into a 1-minute RTH cache, and an
earlier version of this script reported "all gaps filled; cache is continuous"
over 11 windows holding 26% of their proper bar count.

CADENCE IS INFERRED FROM THE CACHE, NEVER FROM THE AMBIENT ENVIRONMENT.
That is the direct lesson of the same incident: `backfill()` used to read
`fetch_alpaca.BAR_MINUTES`, which is an env-var default and has nothing to do
with what is actually on disk. It is now measured from the modal spacing of
the file being repaired, printed, and overridable only explicitly.

Usage
-----
    python verify_cache.py --dir data/parquet_1min
    python verify_cache.py --dir data/parquet_1min --fix
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)


def infer_cadence(path_dir, tickers, sample=8):
    """(bar_minutes, rth_only, modal_bars_per_session) measured FROM THE CACHE.

    Never from an env var. `backfill()` used to take the cadence from
    `fetch_alpaca.BAR_MINUTES`, whose value reflects how the caller happened to
    set ALPACA_BAR_MINUTES and not what is on disk -- running the repair without
    that variable set silently wrote 5-minute extended-hours bars into a
    1-minute RTH cache, and the old session-presence check passed it.

    Bar size is the modal spacing between consecutive timestamps within a
    session, taken over several tickers so one defective file cannot decide it.
    RTH is inferred from whether nearly all bars fall inside 09:30-16:00; a
    cache with real extended-hours coverage has well over half outside.
    """
    spacings, in_rth, total, per_session = [], 0, 0, []
    for t in tickers[:sample]:
        idx = pd.read_parquet(os.path.join(path_dir, f"{t}.parquet"), columns=[]).index
        ny = idx.tz_convert("America/New_York")
        d = pd.Series(ny).diff().dt.total_seconds().div(60)
        # Same-session spacings only: an overnight or weekend break is not a
        # bar size, and including those drags the mode on a short sample.
        d = d[(d > 0) & (d <= 60)]
        spacings.append(d)
        mins = ny.hour * 60 + ny.minute
        in_rth += int(((mins >= 570) & (mins < 960)).sum())
        total += len(ny)
        per_session.append(pd.Series(1, index=ny).groupby(ny.normalize()).size())

    bar_minutes = int(pd.concat(spacings).mode().iloc[0])
    rth_only = (in_rth / max(total, 1)) > 0.98
    modal_bps = int(pd.concat(per_session).mode().iloc[0])
    return bar_minutes, rth_only, modal_bps


def trading_days(path_dir, tickers):
    """Union of session dates across the cache -- the calendar actually fetched."""
    best = None
    for t in tickers:
        idx = pd.read_parquet(os.path.join(path_dir, f"{t}.parquet"), columns=[]).index
        days = pd.DatetimeIndex(sorted(idx.tz_convert("America/New_York").normalize().unique()))
        if best is None or len(days) > len(best):
            best = days
    return best


def find_defects(path_dir, min_run=3, sparse_frac=0.6):
    """[(ticker, start, end, n_sessions, kind), ...] plus the cache's own facts.

    A session is defective when it is MISSING from the ticker's own span, or
    SPARSE -- present but holding fewer than `sparse_frac` of the cache's modal
    bars per session. Both are repaired identically, so they are grouped into
    runs together rather than reported as two separate families.
    """
    tickers = sorted(f[:-8] for f in os.listdir(path_dir) if f.endswith(".parquet"))
    if not tickers:
        raise SystemExit(f"no parquet files in {path_dir}")
    calendar = trading_days(path_dir, tickers)
    bar_minutes, rth_only, modal_bps = infer_cadence(path_dir, tickers)

    out = []
    for t in tickers:
        idx = pd.read_parquet(os.path.join(path_dir, f"{t}.parquet"), columns=[]).index
        ny = idx.tz_convert("America/New_York")
        counts = pd.Series(1, index=ny).groupby(ny.normalize()).size()
        days = pd.DatetimeIndex(counts.index)
        if len(days) < 2:
            continue
        # Only sessions inside this ticker's own span count. A name that IPO'd
        # mid-sample is short at the front, not holed in the middle, and
        # flagging that would bury the real faults in false positives.
        span = calendar[(calendar >= days[0]) & (calendar <= days[-1])]
        present = counts.reindex(span).fillna(0)

        # THE BASELINE IS PER TICKER, NOT PER PANEL. A global threshold off the
        # cache's modal 390 bars/session flags names that simply do not print
        # every minute: BLK trades near $1,000 a share, runs a median of 313
        # bars/session with all 1,506 sessions present, and a panel-wide floor
        # reported 15 phantom runs on it that a re-fetch would return
        # identically. Against its own median it is clean, while BMY's 102-bar
        # sessions against its own 390 remain the defect they are.
        #
        # Median, not mean: it survives the contaminated stretch being measured.
        # The worst case here is VLO at 249 defective sessions of 1,506 (17%),
        # far short of the 50% a median needs to be dragged.
        own = float(present[present > 0].median()) if (present > 0).any() else 0.0
        floor = sparse_frac * min(own, modal_bps) if own else sparse_frac * modal_bps
        # The final session of a span is legitimately partial on a live cache
        # (the SIP embargo trims it), so it is never a defect on its own.
        bad = (present < floor).to_numpy()
        if bad.size:
            bad[-1] = False
        if not bad.any():
            continue

        pos = np.flatnonzero(bad)
        start = 0
        for i in range(1, len(pos) + 1):
            if i == len(pos) or pos[i] != pos[i - 1] + 1:
                a, b, n = span[pos[start]], span[pos[i - 1]], i - start
                seg = present.to_numpy()[pos[start]:pos[i - 1] + 1]
                kind = "missing" if seg.sum() == 0 else "sparse"
                if n >= min_run:
                    out.append((t, a, b, n, kind))
                start = i
    return out, len(tickers), calendar, bar_minutes, rth_only, modal_bps


def backfill(path_dir, ticker, start, end, bar_minutes, rth_only):
    """Purge [start, end] for one ticker, re-fetch it, and merge it back.

    PURGE FIRST. A plain merge keeps whatever was already present for any
    timestamp the new data does not cover, and 5-minute bars sit on timestamps
    that are also valid 1-minute timestamps -- so merging correct data over
    incorrect data leaves the incorrect rows in place, invisibly. Dropping the
    window first makes the repair total rather than partial.

    Still non-destructive outside the window: the rest of the ticker's history
    is untouched, which is the point of repairing a window rather than deleting
    the file and re-pulling six years.
    """
    import fetch_alpaca as fa
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    file_path = os.path.join(path_dir, f"{ticker}.parquet")
    existing = pd.read_parquet(file_path)

    lo_ts = start.tz_convert("UTC")
    hi_ts = (end + timedelta(days=1)).tz_convert("UTC")
    keep = existing[(existing.index < lo_ts) | (existing.index >= hi_ts)]
    purged = len(existing) - len(keep)

    client = fa._create_client()
    if client is None:
        raise SystemExit("no Alpaca credentials -- cannot backfill")

    bars = fa._fetch_bars_with_retry(client, StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(bar_minutes, TimeFrameUnit.Minute),
        start=(start - timedelta(days=2)).tz_convert("UTC").to_pydatetime(),
        end=(end + timedelta(days=3)).tz_convert("UTC").to_pydatetime(),
        feed=os.getenv("ALPACA_DATA_FEED", fa.DEFAULT_DATA_FEED),
        adjustment=fa.ADJUSTMENT,
    ))
    if bars.df.empty:
        print(f"  [{ticker}] API returned nothing for {start.date()}..{end.date()}")
        return 0, purged

    new = bars.df.loc[ticker]
    if rth_only:
        ny = new.index.tz_convert("America/New_York")
        mins = ny.hour * 60 + ny.minute
        new = new[(mins >= 570) & (mins < 960)]

    merged = pd.concat([keep, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    fa._atomic_write_parquet(merged, file_path)
    return len(merged) - len(keep), purged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="cache directory to verify")
    ap.add_argument("--min-run", type=int, default=3,
                    help="consecutive defective sessions before a run is reported (default 3)")
    ap.add_argument("--sparse-frac", type=float, default=0.6,
                    help="a session holding less than this fraction of the cache's "
                         "modal bars/session is defective (default 0.6)")
    ap.add_argument("--bar-minutes", type=int, default=None,
                    help="override the inferred bar size (default: measured from the cache)")
    ap.add_argument("--fix", action="store_true",
                    help="purge and re-fetch each defective run")
    args = ap.parse_args()

    defects, n_tickers, calendar, bar_minutes, rth_only, modal_bps = find_defects(
        args.dir, args.min_run, args.sparse_frac)
    if args.bar_minutes:
        bar_minutes = args.bar_minutes

    print(f"[verify] {args.dir}: {n_tickers} tickers, "
          f"{len(calendar)} sessions {calendar[0].date()}..{calendar[-1].date()}")
    print(f"[verify] cadence measured from the cache: {bar_minutes}-minute bars, "
          f"rth_only={rth_only}, modal {modal_bps} bars/session; a session is "
          f"defective below {args.sparse_frac:.0%} of ITS OWN ticker's median")

    if not defects:
        print("[verify] no missing or sparse sessions. Cache is continuous and complete.")
        return 0

    print(f"[verify] *** {len(defects)} DEFECTIVE RUN(S) -- an incremental "
          "re-fetch CANNOT repair these ***")
    for t, a, b, n, kind in defects:
        print(f"    {t:<6} {a.date()} .. {b.date()}   {n:>3} sessions   {kind}")

    if not args.fix:
        print()
        print("Re-run with --fix to purge and refill them.")
        return 1

    print()
    for t, a, b, n, kind in defects:
        added, purged = backfill(args.dir, t, a, b, bar_minutes, rth_only)
        print(f"  [{t}] {a.date()}..{b.date()}  purged {purged:,} -> added {added:,} bars")

    left = find_defects(args.dir, args.min_run, args.sparse_frac)[0]
    if left:
        print()
        print(f"[verify] {len(left)} run(s) REMAIN: "
              f"{[(t, str(a.date()), k) for t, a, b, n, k in left]}")
        return 1
    print()
    print("[verify] all runs repaired; cache is continuous and complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
