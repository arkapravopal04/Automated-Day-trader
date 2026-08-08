"""
Alpaca Data Fetcher
This module connects to the Alpaca API to download 5-minute OHLCV (Open, High, Low, Close, Volume)
bar data for a predefined set of tickers. It implements an incremental fetching strategy to 
only download new data since the last cache update, saving bandwidth and API calls.
It seamlessly handles credentials locally via .env files or remotely via Kaggle Secrets.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Make sure this script's own folder is importable (needed on Kaggle where
# the working directory during notebook execution isn't always the script dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from paths import RAW_DIR as DATA_DIR, is_kaggle

TICKERS= [
    # Broad ETFs (10)
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
    
    # # Technology & Software (18)
    # "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "AMD", 
    # "QCOM", "INTC", "MU", "TXN", "ORCL", "CRM", "ADBE", "NOW", "PANW",
    
    # # Financials & FinTech (15)
    # "JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "SCHW", "V", "MA", 
    # "AXP", "PYPL", "SQ", "COIN", "BRK.B",
    
    # # Healthcare & Biotechnology (14)
    # "JNJ", "PFE", "UNH", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", 
    # "BMY", "AMGN", "GILD", "ISRG", "VRTX",
    
    # # Consumer Discretionary & Staples (15)
    # "WMT", "COST", "PG", "KO", "PEP", "NKE", "HD", "MCD", "SBUX", 
    # "TGT", "LOW", "PM", "MO", "CL", "MDLZ",
    
    # # Energy & Utilities (10)
    # "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "NEE", "DUK",
    
    # # Industrials, Aerospace & Defense (12)
    # "CAT", "GE", "BA", "LMT", "RTX", "HON", "DE", "UNP", "UPS", 
    # "FDX", "MMM", "GD",
    
    # # Communications & Entertainment (6)
    # "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"
]

def get_alpaca_credentials():
    """
    Retrieves Alpaca API credentials using a tiered fallback approach:
    1. Local .env file (using python-dotenv if available) — checked explicitly
       in the current working directory, then this script's own folder. We do
       NOT use python-dotenv's upward parent-directory search, because a stray
       unrelated .env higher up the tree (e.g. in a parent repo folder) would
       get picked up first and silently block the real one from loading.
    2. Environment variables (os.getenv)
    3. Kaggle Secrets (if running in a Kaggle environment)
    
    Returns:
        tuple: (API_KEY, SECRET_KEY) or (None, None) if not found.
    """
    try:
        from dotenv import load_dotenv

        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        candidate_paths = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(script_dir, ".env"),
        ]

        for env_path in candidate_paths:
            if os.path.exists(env_path):
                # override=False: don't clobber real env vars the user already set
                load_dotenv(env_path, override=False)
                if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
                    break
    except ImportError:
        print("Warning: python-dotenv is not installed, so .env files are ignored. "
              "Run `pip install python-dotenv` or set ALPACA_API_KEY / ALPACA_SECRET_KEY "
              "as real environment variables instead.")

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            api_key = api_key or user_secrets.get_secret("ALPACA_API_KEY")
            secret_key = secret_key or user_secrets.get_secret("ALPACA_SECRET_KEY")
            print("API Credentials loaded from Kaggle Secrets.")
        except Exception:
            pass

    return api_key, secret_key

# Initialize client
API_KEY, SECRET_KEY = get_alpaca_credentials()
if API_KEY and SECRET_KEY:
    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
else:
    client = None
    print("Warning: Alpaca API credentials not found. API fetching will fail unless keys are provided.")


def fetch_incremental_data(ticker: str, end_date: datetime):
    """
    Fetches 5-minute bars for a specific ticker up to the `end_date`.
    If a local Parquet file exists, it only fetches data from the last recorded
    timestamp + 1 minute. Otherwise, it pulls a 3-year historical warm-up.

    Args:
        ticker (str): The stock symbol to fetch (e.g., 'AAPL').
        end_date (datetime): The upper bound datetime (timezone-aware) for the fetch request.
    """
    file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    
    # 1. Determine the start date based on existing cached data
    HISTORY_YEARS = int(os.getenv("ALPACA_HISTORY_YEARS", "6"))
    if os.path.exists(file_path):
        print(f"\n📂 Loading cached 5-min candles for {ticker} from local Parquet...")
        existing_df = pd.read_parquet(file_path)
        print(f"  └─ Imported {len(existing_df)} cached 5-min candles (Last timestamp: {existing_df.index.max()})")
        
        # Add 1 minute to avoid fetching the exact same minute twice
        start_date = existing_df.index.max() + timedelta(minutes=1)
        print(f"Fetching new 5-min candles for {ticker} from Alpaca API since {start_date}...")
    else:
        # Default warm-up period if no data exists
        start_date = end_date - timedelta(days=365 * HISTORY_YEARS)
        existing_df = None
        print(f"\nFetching full {HISTORY_YEARS}-year history of 5-min candles for {ticker} from Alpaca API since {start_date}...")

    if start_date >= end_date:
        print(f"  └─ [{ticker}] Data is already up to date.")
        return

    if client is None:
        raise ValueError(f"Cannot fetch data for {ticker}. Missing Alpaca API credentials.")

    # 2. Setup the Alpaca API Request
    # "sip" (all US exchanges) requires a paid Alpaca subscription; free accounts
    # only get "iex". Default to iex so this works out of the box, but let it be
    # overridden via ALPACA_DATA_FEED=sip if you have a paid plan.
    feed = os.getenv("ALPACA_DATA_FEED", "iex")
    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_date,
        end=end_date,
        feed=feed
    )

    # 3. Execute the API Request
    try:
        try:
            bars = client.get_stock_bars(request_params)
        except Exception as e:
            # Auto-fallback: if a paid "sip" feed was requested but the account
            # doesn't have access, retry once on the free "iex" feed instead of
            # failing the whole run.
            if feed == "sip" and "subscription does not permit" in str(e):
                print(f"  └─ [{ticker}] SIP feed not available on this account, retrying with IEX feed...")
                request_params.feed = "iex"
                bars = client.get_stock_bars(request_params)
            else:
                raise

        if bars.df.empty:
            print(f"  └─ [{ticker}] No new bars returned by API.")
            return
            
        new_df = bars.df.loc[ticker] 
        print(f"  └─ Successfully imported {len(new_df)} new 5-min candles for {ticker} from API.")
        
        # 4. Merge new data with existing data and save to Parquet
        if existing_df is not None:
            combined_df = pd.concat([existing_df, new_df])
            # Ensure no overlapping duplicates exist
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        else:
            combined_df = new_df
            
        # Sort chronologically just to be safe before saving
        combined_df.sort_index(inplace=True)
        combined_df.to_parquet(file_path, engine="pyarrow")
        print(f"  └─ Saved updated total of {len(combined_df)} 5-min candles to {file_path}")
        
    except Exception as e:
        print(f"[{ticker}] Error fetching 5-min candles: {e}")


if __name__ == "__main__":
    # Ensure current time is timezone-aware (UTC) for Alpaca filtering
    current_time = datetime.now(timezone.utc)
    print("=" * 60)
    print(f"STARTING 5-MIN CANDLE FETCH PIPELINE FOR {len(TICKERS)} TICKERS")
    print(f"Environment: {'Kaggle' if is_kaggle() else 'Local'} | Cache dir: {DATA_DIR}")
    print("=" * 60)
    for ticker in TICKERS:
        fetch_incremental_data(ticker, current_time)