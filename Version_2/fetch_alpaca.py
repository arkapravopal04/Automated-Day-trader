"""
Incremental Alpaca 5-minute OHLCV data fetcher.

The module downloads 5-minute stock bars for ``TICKERS`` and stores them as
per-ticker Parquet files. Existing caches are extended incrementally, while
new tickers receive a configurable historical warm-up period.

Credentials are resolved from environment variables, local ``.env`` files,
or Kaggle Secrets, in that order of preference.

Kaggle cached-input behavior: if a Kaggle input Dataset with a "data/parquet"
folder is attached (see paths.py's module docstring), paths.py bootstraps it
into RAW_DIR automatically before this script ever runs. Once that cache is
present, this script defaults to SKIPPING the network fetch step entirely
for every already-cached ticker (see _default_skip_fetch()) -- otherwise
every run would still make a slow incremental "any new bars?" API call per
ticker, and any ticker missing from the cache would silently trigger a full
multi-year fetch that can hang a session. Tickers with no cache at all are
still fetched in full by default (see SKIP_MISSING_TICKERS below).

On a Kaggle session with no Alpaca credentials configured, this script also
no longer treats that as fatal -- any ticker with existing cached data is
left as-is (a warning is printed instead of raising), so a cache-only
Kaggle run completes instead of crashing partway through.

Env var overrides:
    TRADING_SKIP_FETCH=0|1   -> force-disable/force-enable skipping the
                                 network step entirely, overriding the
                                 Kaggle-cache-based auto-default below.
    TRADING_SKIP_MISSING=1   -> when SKIP_FETCH is active, also skip tickers
                                 that have NO cached data at all (strict
                                 "use only what's cached" mode) instead of
                                 fetching them in full.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Ensure this module's directory is importable when executed from a notebook
# or another working directory (notably on Kaggle).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from paths import RAW_DIR as DATA_DIR, is_kaggle


TICKERS = [
    # Broad ETFs (10)
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",

    # Technology & Software (18)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD",
    "QCOM", "INTC", "MU", "TXN", "ORCL", "CRM", "ADBE", "NOW", "PANW",

    # Financials & FinTech (15)
    "JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "SCHW", "V", "MA",
    "AXP", "PYPL", "SQ", "COIN", "BRK.B",

    # Healthcare & Biotechnology (14)
    "JNJ", "PFE", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "VRTX",

    # Consumer Discretionary & Staples (15)
    "WMT", "COST", "PG", "KO", "PEP", "NKE", "HD", "MCD", "SBUX",
    "TGT", "LOW", "PM", "MO", "CL", "MDLZ",

    # Energy & Utilities (10)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "NEE", "DUK",

    # Industrials, Aerospace & Defense (12)
    "CAT", "GE", "BA", "LMT", "RTX", "HON", "DE", "UNP", "UPS",
    "FDX", "MMM", "GD",

    # Communications & Entertainment (6)
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
]

HISTORY_YEARS = int(os.getenv("ALPACA_HISTORY_YEARS", "6"))
DEFAULT_DATA_FEED = "iex"


def _default_skip_fetch() -> bool:
    """
    Decides whether to skip the network fetch step entirely for
    already-cached tickers:
      - Explicit TRADING_SKIP_FETCH=0/1 always wins if set.
      - Otherwise, on Kaggle with ANY cached parquet already present in
        DATA_DIR (typically bootstrapped from an attached input dataset --
        see paths.py), default to skipping. This is what makes "attach a
        cached data/ folder" behave like an actual cache instead of
        triggering a per-ticker incremental API round-trip (slow) and a
        full multi-year fetch for any ticker not already cached (very
        slow, can hang a session).
      - Off Kaggle, or on Kaggle with no cache at all yet, default to
        fetching as before (this script's original behavior).
    """
    explicit = os.getenv("TRADING_SKIP_FETCH")
    if explicit is not None:
        return explicit == "1"
    if not is_kaggle():
        return False
    has_any_cache = os.path.isdir(DATA_DIR) and any(f.endswith(".parquet") for f in os.listdir(DATA_DIR))
    return has_any_cache


SKIP_FETCH = _default_skip_fetch()

# When SKIP_FETCH is active: tickers with NO cached parquet at all are still
# fetched in full by default (a partially-uploaded cache shouldn't silently
# leave permanent gaps in the ticker universe). Set TRADING_SKIP_MISSING=1
# for strict "use only what's already cached, fetch nothing" mode.
SKIP_MISSING_TICKERS = os.getenv("TRADING_SKIP_MISSING", "0") == "1"

API_KEY, SECRET_KEY = None, None


def get_alpaca_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Return Alpaca API credentials from local, environment, or Kaggle sources.

    Resolution order:
      1. ``.env`` in the current working directory, then this module's folder.
      2. ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY`` environment variables.
      3. Kaggle Secrets when running in an environment that provides them.

    Returns:
        A ``(api_key, secret_key)`` tuple. Missing credentials are returned as
        ``None`` rather than raising.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            "Warning: python-dotenv is not installed, so .env files are ignored. "
            "Run `pip install python-dotenv` or set ALPACA_API_KEY / ALPACA_SECRET_KEY "
            "as real environment variables instead."
        )
    else:
        candidate_paths = (
            os.path.join(os.getcwd(), ".env"),
            os.path.join(SCRIPT_DIR, ".env"),
        )
        for env_path in candidate_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path, override=False)
                if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
                    break

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError:
            pass
        else:
            try:
                secrets = UserSecretsClient()
                api_key = api_key or secrets.get_secret("ALPACA_API_KEY")
                secret_key = secret_key or secrets.get_secret("ALPACA_SECRET_KEY")
                print("API Credentials loaded from Kaggle Secrets.")
            except Exception:
                # Kaggle Secrets may be unavailable outside Kaggle notebooks.
                pass

    return api_key, secret_key


def _create_client() -> Optional[StockHistoricalDataClient]:
    """Create an Alpaca historical-data client when credentials are available."""
    if not API_KEY or not SECRET_KEY:
        return None
    return StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)


API_KEY, SECRET_KEY = get_alpaca_credentials()
client = _create_client()

if client is None:
    print(
        "Warning: Alpaca API credentials not found. "
        "Any ticker without existing cached data will be skipped; "
        "cached tickers will be used as-is without incremental updates."
    )


def fetch_incremental_data(ticker: str, end_date: datetime) -> None:
    """Fetch and cache 5-minute bars for one ticker up to ``end_date``.

    Existing Parquet data is extended from one minute after its latest
    timestamp. A missing cache is initialized with ``HISTORY_YEARS`` of
    historical data. New records are merged, duplicate timestamps are removed
    in favor of the newest record, and the result is sorted chronologically.

    Args:
        ticker: Stock symbol to fetch.
        end_date: Timezone-aware upper bound for the Alpaca request.

    Note:
        If Alpaca credentials are unavailable, a ticker with NO existing
        cache is skipped with a warning (nothing to fall back on). A ticker
        WITH existing cache (e.g. bootstrapped from a Kaggle input dataset --
        see paths.py) is also skipped with a warning rather than raising --
        this lets a credential-less, cache-only Kaggle session complete
        using stale-but-present data instead of crashing the whole pipeline.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")

    if os.path.exists(file_path):
        print(f"\nLoading cached 5-min candles for {ticker} from local Parquet...")
        existing_df = pd.read_parquet(file_path)
        latest_timestamp = existing_df.index.max()
        print(
            f"  └─ Imported {len(existing_df)} cached 5-min candles "
            f"(Last timestamp: {latest_timestamp})"
        )

        if pd.isna(latest_timestamp):
            start_date = end_date - timedelta(days=365 * HISTORY_YEARS)
        else:
            start_date = latest_timestamp + timedelta(minutes=1)

        print(
            f"Fetching new 5-min candles for {ticker} from Alpaca API "
            f"since {start_date}..."
        )
    else:
        existing_df = None
        start_date = end_date - timedelta(days=365 * HISTORY_YEARS)
        print(
            f"\nFetching full {HISTORY_YEARS}-year history of 5-min candles for "
            f"{ticker} from Alpaca API since {start_date}..."
        )

    if start_date >= end_date:
        print(f"  └─ [{ticker}] Data is already up to date.")
        return

    if client is None:
        if existing_df is not None:
            print(
                f"  └─ [{ticker}] No Alpaca credentials -- using existing cached data "
                f"as-is (last timestamp: {latest_timestamp}, not refreshed)."
            )
        else:
            print(f"  └─ [{ticker}] No Alpaca credentials and no existing cache -- skipping this ticker.")
        return

    feed = os.getenv("ALPACA_DATA_FEED", DEFAULT_DATA_FEED)
    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_date,
        end=end_date,
        feed=feed,
    )

    try:
        try:
            bars = client.get_stock_bars(request_params)
        except Exception as exc:
            if feed == "sip" and "subscription does not permit" in str(exc):
                print(
                    f"  └─ [{ticker}] SIP feed not available on this account, "
                    "retrying with IEX feed..."
                )
                request_params.feed = "iex"
                bars = client.get_stock_bars(request_params)
            else:
                raise

        if bars.df.empty:
            print(f"  └─ [{ticker}] No new bars returned by API.")
            return

        new_df = bars.df.loc[ticker]
        print(
            f"  └─ Successfully imported {len(new_df)} new 5-min candles "
            f"for {ticker} from API."
        )

        if existing_df is not None:
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        else:
            combined_df = new_df

        combined_df = combined_df.sort_index()
        combined_df.to_parquet(file_path, engine="pyarrow")
        print(
            f"  └─ Saved updated total of {len(combined_df)} 5-min candles "
            f"to {file_path}"
        )

    except Exception as exc:
        print(f"[{ticker}] Error fetching 5-min candles: {exc}")


if __name__ == "__main__":
    current_time = datetime.now(timezone.utc)

    print("=" * 60)
    print(f"STARTING 5-MIN CANDLE FETCH PIPELINE FOR {len(TICKERS)} TICKERS")
    print(
        f"Environment: {'Kaggle' if is_kaggle() else 'Local'} | "
        f"Cache dir: {DATA_DIR}"
    )
    print("=" * 60)

    if SKIP_FETCH:
        missing = [t for t in TICKERS if not os.path.exists(os.path.join(DATA_DIR, f"{t}.parquet"))]
        print(
            f"SKIP_FETCH active (cached data detected in {DATA_DIR}) -- "
            "no incremental API calls will be made for already-cached tickers."
        )
        if missing:
            if SKIP_MISSING_TICKERS:
                print(
                    f"  └─ {len(missing)} ticker(s) have no cache and TRADING_SKIP_MISSING=1: "
                    f"{missing} -- leaving them unfetched."
                )
            else:
                print(f"  └─ {len(missing)} ticker(s) have no cache and will still be fetched in full: {missing}")
                for ticker in missing:
                    fetch_incremental_data(ticker, current_time)
        else:
            print("  └─ Every configured ticker already has cached data. Nothing to do.")
    else:
        for ticker in TICKERS:
            fetch_incremental_data(ticker, current_time)