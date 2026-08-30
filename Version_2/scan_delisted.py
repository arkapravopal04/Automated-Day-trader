"""Find the large-cap names that DIED during the sample, so the universe stops lying.

WHY THIS EXISTS
---------------
`fetch_alpaca.TICKERS` is 100 mega-caps hand-picked in 2026. Every one of them
survived and stayed liquid across 2020-2026 by construction. For the intraday
work that bias is mild and it cannot explain a negative result. For the
OVERNIGHT work it is a live alternative explanation for a positive one:
cross-sectional reversal buys yesterday's losers, and in a survivor-only
universe yesterday's loser is a name already known to have recovered.

The names this excludes are not marginal. SIVB and FRC -- Silicon Valley Bank
and First Republic -- both collapsed inside the sample, and both are exactly
the catastrophic overnight gap that a reversal book would have bought into.

WHY THE OBVIOUS METHOD DOES NOT WORK
-----------------------------------
Alpaca's assets endpoint cannot enumerate delistings. Measured 2026-08-30:
SIVB, FRC, TWTR, ATVI, PXD, HES and ABMD are ABSENT FROM THE ASSET LIST
ENTIRELY -- every one of them a name this study specifically needs -- while
LHX, COR, MTCH, IR and CZR appear as INACTIVE duplicates despite still
trading today. Screening `status=INACTIVE` returned 15 names, of which 5 were
live and the most important delistings were missing.

WHAT IT DOES INSTEAD
--------------------
The DATA api is the source of truth: it serves full history for delisted
symbols, terminating at the delisting date (TWTR ends 2022-10-27, SIVB
2023-03-09, ATVI 2023-10-13). So a curated candidate list is proposed and
every name is VERIFIED against the data:

  * median daily dollar volume >= the floor, over its own active life; and
  * its history ENDS before the panel does -- which is what makes it a
    delisting rather than a name that simply is not in TICKERS.

A candidate that fails either test drops out automatically, so errors of
INCLUSION are self-correcting. Errors of OMISSION are not: this is a curated
list, not a point-in-time index membership, and it can still be missing names.
That limitation is real and is recorded rather than papered over.

The floor is $150M median daily dollar volume against the incumbent
universe's observed minimum of $192M (GD). Deliberately BELOW it: the error
that matters here is excluding a name that belongs, because that is the error
which preserves the bias this script exists to remove.

    python scan_delisted.py --out logs/p3/delisted_candidates.json
"""
import argparse, json, os, sys
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fetch_alpaca import TICKERS

START = datetime(2020, 8, 31)
END = datetime(2026, 8, 28)


# Large/mid-cap US names that were acquired, merged away, or collapsed between
# 2020-09 and 2026-08. Proposed from the record and then VERIFIED against the
# data api -- nothing here is trusted on the strength of the list alone.
#
# The collapses matter more than the acquisitions for this study. SIVB, FRC and
# SBNY are the catastrophic overnight gaps that a survivor-only universe erases,
# and a cross-sectional reversal book would have been buying all three on the
# way down.
CANDIDATES = [
    # --- bank failures / collapses, March-May 2023 ---
    "SIVB", "FRC", "SBNY",
    # --- large acquisitions ---
    "ATVI", "TWTR", "XLNX", "ALXN", "MXIM", "CERN", "NUAN", "ZNGA", "ABMD",
    "KSU", "WORK", "TIF", "VAR", "PXD", "HES", "SGEN", "HZNP", "SPLK", "VMW",
    "CTXS", "ZEN", "INFO", "CXO", "WPX", "XEC", "MRO", "CTLT", "JNPR", "ANSS",
    "VER", "PSXP", "DISCA", "DISCK", "TWNK", "RE", "PEAK", "FLIR", "MGLN",
    "CONE", "QTS", "CDAY", "FISV", "ANTM", "NLSN", "PNM", "STOR", "ATC",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dollar-vol", type=float, default=150e6)
    ap.add_argument("--min-sessions", type=int, default=60)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out", default="logs/p3/delisted_candidates.json")
    args = ap.parse_args()

    key, sec = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    tc = TradingClient(key, sec, paper=True)
    dc = StockHistoricalDataClient(key, sec)

    cand = sorted(set(CANDIDATES) - set(TICKERS))
    print(f"[scan] {len(cand)} curated candidates to verify against the data api")
    panel_end = pd.Timestamp(END).tz_localize("UTC") - pd.Timedelta(days=20)

    keep, rejected = [], []
    for i in range(0, len(cand), args.batch):
        chunk = cand[i:i + args.batch]
        try:
            df = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                start=START, end=END, adjustment=Adjustment.ALL, feed="sip")).df
        except Exception as e:
            print(f"[scan] batch {i // args.batch}: {type(e).__name__} {str(e)[:90]}")
            continue
        if df is None or len(df) == 0:
            continue
        df = df.reset_index()
        for sym, g in df.groupby("symbol"):
            last = g["timestamp"].max()
            reasons = []
            if len(g) < args.min_sessions:
                reasons.append(f"only {len(g)} sessions")
            dv = float((g["close"] * g["volume"]).median())
            if dv < args.min_dollar_vol:
                reasons.append(f"${dv/1e6:,.0f}M/day below floor")
            # STILL TRADING is a disqualifier, not a pass: a name whose history
            # runs to the panel end did not delist, it is simply not in TICKERS,
            # and adding it would change the universe for a different reason
            # than the one under test.
            if last >= panel_end:
                reasons.append(f"still trading ({str(last)[:10]})")
            if reasons:
                rejected.append({"symbol": sym, "why": "; ".join(reasons)})
                continue
            keep.append({"symbol": sym, "median_dollar_vol": dv, "sessions": int(len(g)),
                         "first": str(g["timestamp"].min())[:10],
                         "last": str(last)[:10]})
        print(f"[scan] {i + len(chunk):>5}/{len(cand)}  kept so far {len(keep)}")

    keep.sort(key=lambda r: -r["median_dollar_vol"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"min_dollar_vol": args.min_dollar_vol,
                   "min_sessions": args.min_sessions,
                   "incumbent_floor_note": "incumbent universe min is GD at $192M/day",
                   "candidates": keep}, fh, indent=2)
    print(f"\n[scan] {len(keep)} names clear ${args.min_dollar_vol/1e6:,.0f}M/day "
          f"and {args.min_sessions} sessions")
    for r in keep[:40]:
        print(f"   {r['symbol']:<7} ${r['median_dollar_vol']/1e6:>7,.0f}M  "
              f"{r['sessions']:>5} sess  {r['first']} -> {r['last']}")
    print(f"[scan] -> {args.out}")


if __name__ == "__main__":
    main()
