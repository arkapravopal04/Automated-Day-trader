"""
Data Preprocessing & Feature Engineering
Transforms raw OHLCV Parquet files into standardized feature tensors.
Generates technical and structural features (log returns, realized volatility,
volume z-scores, session time embeddings). Critically, it computes normalisation 
statistics (mean/std) strictly on the training dataset to prevent lookahead bias,
then applies these scaling factors across the entire dataset.

Kaggle cached-input behavior: if a Kaggle input Dataset with a "data/processed"
folder is attached (see paths.py's module docstring), paths.py bootstraps it
into PROCESSED_DIR automatically before this script runs. In that case
generate_features_and_metadata() detects the already-complete cache via
paths.is_cache_ready() and skips reprocessing entirely -- this is what makes
attaching that input "just work" instead of silently reprocessing 6 years of
5-min bars again every session. Set TRADING_FORCE_PREPROCESS=1 to bypass this
and force a full reprocess regardless of what's already in PROCESSED_DIR.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from paths import RAW_DIR, PROCESSED_DIR, TRAIN_FRAC, is_kaggle, is_cache_ready

# Configuration Parameters
HORIZONS = [3, 6, 12] # Lag steps representing 15m, 30m, and 1h past returns
RV_WINDOW = 12 # Realized volatility rolling window (1 hour)
VOL_WINDOW = 78 # Volume z-score rolling window (1 full trading day)

# Set to force a full reprocess even if PROCESSED_DIR already looks complete
# (e.g. bootstrapped from a Kaggle input cache dataset -- see paths.py).
FORCE_PREPROCESS = os.getenv("TRADING_FORCE_PREPROCESS", "0") == "1"

def process_ticker(ticker: str) -> pd.DataFrame:
    """
    Loads raw ticker data and engineers machine-learning ready features.
    
    Args:
        ticker (str): The stock symbol being processed.
        
    Returns:
        pd.DataFrame: A dataframe containing the engineered features, with NaN values dropped.
    """
    file_path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    
    print(f"[PREPROCESS] Importing 5-min candles for {ticker} from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"Successfully loaded {len(df)} raw 5-min candles for {ticker}.")

    # Standardize the datetime index to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    
    # 1. Immediate Log Returns (t vs t-1)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Multi-Horizon Log Returns (Captures short-term momentum)
    for h in HORIZONS:
        df[f'log_ret_{h}'] = np.log(df['close'] / df['close'].shift(h))
        
    # 3. Annualized Realized Volatility
    # Calculation: standard deviation of recent returns scaled by annualizing factor
    # (78 5-min bars/day * 252 trading days/year)
    df['rv'] = df['log_ret'].rolling(window=RV_WINDOW).std() * np.sqrt(252 * 78)
    
    # 4. Volume Z-Score
    # Measures how abnormal current trading volume is compared to the recent rolling window
    vol_mean = df['volume'].rolling(window=VOL_WINDOW).mean()
    vol_std = df['volume'].rolling(window=VOL_WINDOW).std()
    df['vol_z'] = (df['volume'] - vol_mean) / (vol_std + 1e-8) # +1e-8 prevents division by zero
    
    # 5. Time-of-Day / Session Encoding
    # Converts New York time to a continuous cyclical feature (sine/cosine)
    # Market open (9:30 AM) to close (4:00 PM) is 390 minutes.
    ny_time = df.index.tz_convert("America/New_York")
    minutes_since_open = (ny_time.hour - 9) * 60 + (ny_time.minute - 30)
    day_fraction = np.clip(minutes_since_open / 390.0, 0, 1)
    df['time_sin'] = np.sin(day_fraction * 2 * np.pi)
    df['time_cos'] = np.cos(day_fraction * 2 * np.pi)
    
    # Remove leading rows that contain NaNs due to rolling window calculations
    df.dropna(inplace=True)
    return df

def generate_features_and_metadata():
    """
    Iterates over all raw Parquet files, extracts features, calculates normalization
    constants based strictly on the training set limit, and saves normalized features
    and metadata for the PyTorch DataLoader.

    Skips entirely if PROCESSED_DIR already contains a complete cache (metadata.json
    plus every *_features.parquet file -- see paths.is_cache_ready()), e.g. bootstrapped
    from an attached Kaggle input dataset. Set TRADING_FORCE_PREPROCESS=1 to override.
    """
    if not FORCE_PREPROCESS and is_cache_ready():
        print(
            f"[PREPROCESS] PROCESSED_DIR ({PROCESSED_DIR}) already has a complete "
            "processed cache (metadata.json + feature files) -- skipping preprocessing. "
            "Set TRADING_FORCE_PREPROCESS=1 to force a rebuild."
        )
        return

    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"Directory {RAW_DIR} not found. Run fetch_alpaca.py first.")

    tickers = [f[:-len(".parquet")] for f in os.listdir(RAW_DIR) if f.endswith(".parquet")]
    tickers.sort()
    
    print("=" * 60)
    print(f"STARTING PREPROCESSING FOR {len(tickers)} TICKERS")
    print(f"Environment: {'Kaggle' if is_kaggle() else 'Local'} | Raw: {RAW_DIR} | Processed: {PROCESSED_DIR}")
    print("=" * 60)

    # Initialize Metadata Registry
    metadata = {
        "features": ['log_ret', 'rv', 'vol_z', 'time_sin', 'time_cos'] + [f'log_ret_{h}' for h in HORIZONS],
        "norm_constants": {},
        "tick_sizes": {ticker: 0.01 for ticker in tickers} # Assuming $0.01 standard US equity tick size
    }
    
    for ticker in tickers:
        df = process_ticker(ticker)
        
        # Determine the training set mask as the first TRAIN_FRAC of this
        # ticker's history (row-position based, not a fixed calendar date) so
        # it scales automatically with however much history was fetched, and
        # stays consistent with the split dataset.py uses at training time.
        train_cutoff_idx = int(len(df) * TRAIN_FRAC)
        train_df = df.iloc[:train_cutoff_idx][metadata["features"]]
        
        # Compute mean and standard deviation ONLY on training data
        mean = train_df.mean()
        std = train_df.std() + 1e-8
        
        metadata["norm_constants"][ticker] = {
            "mean": mean.to_dict(),
            "std": std.to_dict()
        }
        
        # Normalize the full dataset using the pre-computed training statistics
        df[metadata["features"]] = (df[metadata["features"]] - mean) / std
        
        # Save processed features
        out_path = os.path.join(PROCESSED_DIR, f"{ticker}_features.parquet")
        df[metadata["features"]].to_parquet(out_path, engine="pyarrow")
        print(f"  └─ Feature tensor exported: {df.shape[0]} rows, {df.shape[1]} columns -> {out_path}\n")

    # Persist normalization and metadata for the environment/model layers to use
    meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Preprocessing complete. Metadata saved to {meta_path}")

if __name__ == "__main__":
    generate_features_and_metadata()