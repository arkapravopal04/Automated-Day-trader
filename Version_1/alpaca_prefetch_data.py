"""
alpaca_prefetch_data.py — run once before overnight training.

Fetches full 6-year history for all tickers and saves to
./alpaca_cache/ as pickle (.pkl) files. After this, train.py reads
purely from cache — token/API not needed overnight.

Run order:
    1. python alpaca_prefetch_data.py   (~3-5 min first time)
    2. python train.py                  (leave overnight)
"""

import sys
from alpaca_data import load_data, transform_data, build_windows, WINDOW_SIZE

# TICKERS = ["SPY", "QQQ", "IWM", "XLE", "XBI", "GLD", "USO", "ARKK", "AAPL", "NVDA"]
TICKERS = ['NFLX']


def prefetch():
    print(f"\nPrefetching {len(TICKERS)} tickers — cached to ./alpaca_cache/")
    print("Subsequent runs only fetch the gap since last time.\n")

    success, failed = [], []

    for ticker in TICKERS:
        print(f"── {ticker} {'─' * (20 - len(ticker))}")
        try:
            raw          = load_data(ticker)
            transformed  = transform_data(raw)
            X, y, prices = build_windows(transformed, WINDOW_SIZE, raw_data=raw)
            print(f"   {ticker}: {X.shape[0]:,} windows ready\n")
            success.append(ticker)
        except Exception as e:
            print(f"   {ticker}: FAILED — {e}\n")
            failed.append(ticker)

    print("─" * 40)
    print(f"Done.  {len(success)} tickers cached.")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    else:
        print("All tickers ready — run train.py overnight.")


if __name__ == "__main__":
    prefetch()