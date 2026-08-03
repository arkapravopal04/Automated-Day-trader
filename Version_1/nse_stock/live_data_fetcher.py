"""
must be used everytime i want to fetch the data from the upstox api and save it to the cache folder

What it does:
    Fetches full 5m history (Jan 2022 → today) for every ticker
    in your TICKERS list and saves them to ./upstox_cache/ as
    parquet files.

    After this, train.py reads purely from cache and never
    touches the Upstox API — so the token expiring at midnight
    doesn't matter at all.

Run order each day you want to train overnight:
    1. python get_token.py        ← refresh token (morning)
    2. python prefetch_data.py    ← pull and cache data (~2 min)
    3. python train.py            ← leave running overnight

On subsequent nights, step 2 only fetches the gap since last run
(a few days of candles) so it finishes in seconds.
"""

import sys
from live_data import load_data, transform_data, build_windows, WINDOW_SIZE

# ── same ticker list as train.py ──────────────────────────────────────────────
TICKERS = ["RELIANCE", "TCS", "INFY", "ITC", "ICICIBANK", "ADANIPORTS"]

# ─────────────────────────────────────────────────────────────────────────────

def prefetch():
    print(f"\nPrefetching {len(TICKERS)} tickers — this caches everything to disk.")
    print("Subsequent runs only fetch the gap since last time (very fast).\n")

    success = []
    failed  = []

    for ticker in TICKERS:
        print(f"── {ticker} {'─' * (20 - len(ticker))}")
        try:
            raw         = load_data(ticker)
            transformed = transform_data(raw)
            X, y, prices= build_windows(transformed, WINDOW_SIZE, raw_data=raw)
            print(f"   {ticker}: {X.shape[0]:,} windows ready\n")
            success.append(ticker)
        except Exception as e:
            print(f"   {ticker}: FAILED — {e}\n")
            failed.append(ticker)

    print("─" * 40)
    print(f"Done.  {len(success)} tickers cached successfully.")
    if failed:
        print(f"Failed: {failed}")
        print("Check your token (run get_token.py) and try again.")
        sys.exit(1)
    else:
        print("All tickers ready — you can now run train.py overnight.")

if __name__ == "__main__":
    prefetch()