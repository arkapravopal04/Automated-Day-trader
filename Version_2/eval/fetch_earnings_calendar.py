"""
eval/fetch_earnings_calendar.py -- the scheduled-earnings calendar the overnight
book excludes on.

There is no earnings data in this project. Alpaca's corporate-actions endpoint
carries splits and dividends and nothing else, so the calendar is fetched from
Yahoo Finance and cached to disk ONCE. Every downstream run reads the csv; none
of them touches the network.

WHAT IS FETCHED AND WHY IT IS ENOUGH

Yahoo exposes the same table two ways. The JSON visualization endpoint stopped
being updated in mid-2025 -- verified here: PANW's last row on that path is
2025-05-20, which would leave folds 4 and 5 uncovered. The HTML calendar page
returns 2001 through 2026-10 in one request, so THAT is the source, and the
JSON path is used only as a fallback for symbols whose calendar page is gone.

THE TIMING FLAG IS THE POINT OF THE FILE.

An overnight book is flat or not flat across one specific window, so a date
without a BMO/AMC flag is not usable. Yahoo publishes a release time; this
script keeps it and classifies:

    <= 09:30 ET   BMO      the print gaps that morning's open
    >= 16:00 ET   AMC      the print gaps the next morning's open
    otherwise     unknown  including the 00:00 and 12:00 placeholders Yahoo
                           emits when it has no time -- NOT silently treated as
                           either, because guessing wrong here excludes the safe
                           session and holds the dangerous one, and the control
                           would then do nothing while appearing to work.

Verified against known releases before use: PANW and GOOGL come back at 16:00 ET
(both report after the close), MS, JPM and KO at 06:00-08:00 ET (all three
report before it).

Index ETFs in the panel -- SPY, QQQ, DIA, IWM, XL* -- have no earnings and are
expected to come back empty. The coverage report names every symbol with zero
rows rather than letting an operating company hide among them.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

warnings.filterwarnings("ignore")

# Yahoo's placeholder times. 00:00 is "no time recorded"; 12:00 is the midday
# marker the visualization endpoint emits for the same thing. Neither is a real
# 12pm release, and both must classify as unknown.
PLACEHOLDER = {(0, 0), (12, 0)}


def classify(ts):
    """ET release timestamp -> 'bmo' | 'amc' | 'unknown'."""
    if ts is None or pd.isna(ts):
        return "unknown"
    hm = (int(ts.hour), int(ts.minute))
    if hm in PLACEHOLDER:
        return "unknown"
    minutes = hm[0] * 60 + hm[1]
    if minutes <= 9 * 60 + 30:
        return "bmo"
    if minutes >= 16 * 60:
        return "amc"
    return "unknown"          # 09:31-15:59: a release inside the session


def to_et(idx):
    """Index -> tz-aware America/New_York, whatever Yahoo handed back."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize("America/New_York")
    return idx.tz_convert("America/New_York")


def fetch_one(ticker, yf):
    """[(ts_et, timing, source)] for one symbol; HTML first, JSON as fallback."""
    rows = []
    t = yf.Ticker(ticker)

    sources = (
        ("html", lambda: t._get_earnings_dates_using_scrape(limit=100, offset=0)),
        ("json", lambda: t._get_earnings_dates_using_screener(limit=60)),
    )
    for source, call in sources:
        try:
            df = call()
        except Exception as e:                      # noqa: BLE001 -- report, continue
            print(f"    [{ticker}] {source}: {type(e).__name__}: {str(e)[:90]}")
            continue
        if df is None or not len(df):
            continue
        # The JSON path tags stockholders' meetings and earnings CALLS alongside
        # the report itself. Only the report gaps the stock.
        if "Event Type" in df.columns:
            df = df[df["Event Type"].astype(str).str.lower().isin(("earnings", "2"))]
            if not len(df):
                continue
        for ts in to_et(df.index):
            rows.append((ts, classify(ts), source))
        # The HTML path is authoritative and complete; only ask the stale JSON
        # endpoint when it returned nothing.
        if rows:
            break
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cache the scheduled-earnings calendar.")
    ap.add_argument("--processed-dir", default="data/processed_1min_ib_du",
                    help="directory whose *_features.parquet names define the universe")
    ap.add_argument("--tickers", default=None,
                    help="comma-separated override for the universe")
    ap.add_argument("--start", default="2020-07-01",
                    help="the panel starts 2020-09-29; a month of margin either "
                         "side so no fold and not the sealed test has an edge effect")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--out", default="data/earnings_calendar.csv")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="seconds between symbols; this is one polite pass over a "
                         "public endpoint, not a scraper")
    args = ap.parse_args(argv)

    import yfinance as yf

    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = sorted(os.path.basename(p).split("_features")[0]
                         for p in glob.glob(os.path.join(args.processed_dir,
                                                         "*_features.parquet")))
    if not tickers:
        raise SystemExit(f"no *_features.parquet under {args.processed_dir}")

    lo = pd.Timestamp(args.start, tz="America/New_York")
    hi = pd.Timestamp(args.end, tz="America/New_York") + pd.Timedelta(days=1)
    print(f"[fetch] {len(tickers)} symbols, window {args.start} .. {args.end}")

    out, empty, failed = [], [], []
    for i, tk in enumerate(tickers, 1):
        # Yahoo uses '-' where the panel uses '.' (BRK.B -> BRK-B).
        rows = fetch_one(tk.replace(".", "-"), yf)
        if not rows:
            failed.append(tk)
            print(f"  [{i:>3}/{len(tickers)}] {tk:<6} NO DATA")
            time.sleep(args.sleep)
            continue
        kept = [(tk, ts, tm, src) for ts, tm, src in rows if lo <= ts < hi]
        if not kept:
            empty.append(tk)
        out.extend(kept)
        print(f"  [{i:>3}/{len(tickers)}] {tk:<6} {len(rows):>3} rows, "
              f"{len(kept):>3} in window")
        time.sleep(args.sleep)

    if not out:
        raise SystemExit("nothing fetched")

    df = pd.DataFrame(out, columns=["ticker", "ts_et", "timing", "source"])
    df["date"] = df["ts_et"].map(lambda x: x.date().isoformat())
    df = (df.drop_duplicates(subset=["ticker", "date"], keep="first")
            .sort_values(["ticker", "date"])
            .reset_index(drop=True))
    df["ts_et"] = df["ts_et"].map(lambda x: x.isoformat())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df[["ticker", "date", "timing", "ts_et", "source"]].to_csv(args.out, index=False)

    print()
    print("=" * 78)
    print(f"COVERAGE -- {args.out}")
    print("=" * 78)
    print(f"  {len(df):,} events, {df.ticker.nunique()} symbols, "
          f"{df.date.min()} .. {df.date.max()}")
    vc = df.timing.value_counts()
    for tm in ("bmo", "amc", "unknown"):
        n = int(vc.get(tm, 0))
        print(f"  {tm:<8} {n:>5}  ({100.0 * n / len(df):5.1f}%)")
    per = df.groupby("ticker").size()
    print(f"  events per symbol: min {per.min()}  median {int(per.median())}  "
          f"max {per.max()}")
    thin = sorted(per[per < 12].index)              # < 3 years of quarters
    if thin:
        print(f"  THIN (<12 events over {args.start[:4]}-{args.end[:4]}): "
              f"{', '.join(thin)}")
    if empty:
        print(f"  IN UNIVERSE, ZERO EVENTS IN WINDOW: {', '.join(sorted(empty))}")
    if failed:
        print(f"  NO DATA AT ALL: {', '.join(sorted(failed))}")
    print("  ETFs (SPY QQQ DIA IWM XL*) are expected in the two lists above; "
          "an operating")
    print("  company appearing there is a coverage hole and the exclusion "
          "cannot act on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
