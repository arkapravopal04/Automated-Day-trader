"""
Data Preprocessing & Feature Engineering
Transforms raw OHLCV Parquet files into standardized feature tensors.
Generates technical and structural features (log returns, realized volatility,
volume z-scores, session time embeddings). Critically, it computes normalisation 
statistics (mean/std) strictly on the training dataset to prevent lookahead bias,
then applies these scaling factors across the entire dataset.

This script always reprocesses raw data end-to-end -- it does not trust or
reuse any pre-existing contents of PROCESSED_DIR (e.g. from an attached
Kaggle input Dataset). Every run regenerates *_features.parquet and
metadata.json from RAW_DIR from scratch, so there is no stale-cache class of
bug to worry about. Writes are atomic (temp file + os.replace) so a crash or
interrupt mid-run can never leave a half-written or corrupted output file.
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from paths import RAW_DIR, PROCESSED_DIR, TRAIN_FRAC, is_kaggle

# Configuration Parameters
HORIZONS = [3, 6, 12] # Lag steps representing 15m, 30m, and 1h past returns
RV_WINDOW = 12 # Realized volatility rolling window (1 hour)
VOL_WINDOW = 78 # Volume z-score rolling window (1 full trading day)

if len(HORIZONS) != len(set(HORIZONS)):
    raise ValueError(f"HORIZONS contains duplicate values: {HORIZONS} -- would create duplicate feature columns")

REQUIRED_RAW_COLUMNS = {"open", "high", "low", "close", "volume"}
# Longest rolling window used anywhere below -- used to sanity-check that a
# ticker has enough history to produce any non-NaN feature rows.
MIN_ROWS_REQUIRED = max(VOL_WINDOW, RV_WINDOW, max(HORIZONS)) + 1


def _atomic_write_parquet(df: pd.DataFrame, out_path: str) -> None:
    """Write a parquet file atomically: write to a temp file in the same
    directory, then os.replace() it into place. Guarantees readers never see
    a partially-written file, and a crash mid-write leaves only a stray temp
    file, never a corrupted out_path."""
    out_dir = os.path.dirname(out_path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=out_dir)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, engine="pyarrow")
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _atomic_write_json(obj: dict, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=4)
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def process_ticker(ticker: str) -> pd.DataFrame:
    """
    Loads raw ticker data and engineers machine-learning ready features.
    
    Args:
        ticker (str): The stock symbol being processed.
        
    Returns:
        pd.DataFrame: A dataframe containing the engineered features, with NaN values dropped.

    Raises:
        FileNotFoundError: if the raw parquet file is missing.
        ValueError: if the raw data is empty, missing required columns,
            contains non-positive close prices (breaks log returns), or
            has too few rows to produce any valid feature rows.
    """
    file_path = os.path.join(RAW_DIR, f"{ticker}.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found for {ticker}: {file_path}")

    print(f"[PREPROCESS] Importing 5-min candles for {ticker} from {file_path}...")
    df = pd.read_parquet(file_path)

    if df.empty:
        raise ValueError(f"{ticker}: raw parquet file is empty ({file_path})")

    missing_cols = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"{ticker}: raw data missing required columns {sorted(missing_cols)}")

    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"{ticker}: only {len(df)} raw rows, need at least {MIN_ROWS_REQUIRED} "
            "to compute rolling features"
        )

    if df.columns.duplicated().any():
        dupe_cols = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"{ticker}: raw data has duplicate column names {dupe_cols}")

    print(f"Successfully loaded {len(df)} raw 5-min candles for {ticker}.")

    # Standardize the datetime index to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    # Drop exact duplicate timestamps (keep first) -- duplicated bars would
    # silently corrupt the rolling-window calculations below.
    if df.index.duplicated().any():
        n_dupes = int(df.index.duplicated().sum())
        print(f"  [WARN] {ticker}: dropping {n_dupes} duplicate-timestamp rows")
        df = df[~df.index.duplicated(keep="first")]

    # Drop individual bars with non-positive OHLC values -- these are
    # zero/garbage-filled placeholder rows (e.g. a no-trade gap the vendor
    # zero-filled rather than omitted), not real prices. A handful of these
    # shouldn't sink an entire ticker; only failing to compute log returns
    # would (a single close<=0 row poisons that row AND, via shift(), the
    # next H rows too, which is exactly why we drop the row itself here
    # rather than letting log() emit -inf/NaN and relying on later dropna).
    bad_price_mask = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad_price_mask.any():
        n_bad = int(bad_price_mask.sum())
        bad_frac = n_bad / len(df)
        print(f"  [WARN] {ticker}: dropping {n_bad} row(s) ({bad_frac:.2%}) with non-positive OHLC")
        if bad_frac > 0.05:
            raise ValueError(
                f"{ticker}: {n_bad}/{len(df)} rows ({bad_frac:.2%}) have non-positive OHLC -- "
                "this looks like a data quality problem, not occasional bad ticks; refusing to "
                "silently drop that much data"
            )
        df = df[~bad_price_mask]

    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"{ticker}: only {len(df)} rows remain after dropping bad-price rows, need at least "
            f"{MIN_ROWS_REQUIRED} to compute rolling features"
        )

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

    if df.empty:
        raise ValueError(
            f"{ticker}: no rows survived feature engineering (all NaN after rolling windows) "
            "-- raw history is too short or too gappy"
        )

    # Any remaining non-finite values (e.g. inf from a pathological std==0
    # edge case) would silently poison training -- fail loudly instead.
    feature_cols = ['log_ret', 'rv', 'vol_z', 'time_sin', 'time_cos'] + [f'log_ret_{h}' for h in HORIZONS]
    if not np.isfinite(df[feature_cols].to_numpy()).all():
        raise ValueError(f"{ticker}: non-finite (inf/NaN) values remain in engineered features")

    return df

def generate_features_and_metadata():
    """
    Iterates over all raw Parquet files, extracts features, calculates normalization
    constants based strictly on the training set limit, and saves normalized features
    and metadata for the PyTorch DataLoader.

    Always does a full reprocess from RAW_DIR -- any existing contents of
    PROCESSED_DIR (including a Kaggle input-dataset bootstrap) are treated as
    stale and unconditionally overwritten. This removes an entire class of
    "stale/mismatched cache" bugs (e.g. an attached cache built with different
    HORIZONS/windows/TRAIN_FRAC silently being reused). Writes are atomic per
    file, so a failure partway through leaves already-written tickers intact
    and never leaves a half-written file on disk; metadata.json is written
    last, only after every ticker succeeds, so its presence is a reliable
    signal that the whole cache is complete and consistent.
    """
    if not os.path.exists(RAW_DIR):
        raise FileNotFoundError(f"Directory {RAW_DIR} not found. Run fetch_alpaca.py first.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".parquet")]
    tickers = [f[:-len(".parquet")] for f in raw_files]

    if not tickers:
        raise FileNotFoundError(f"No .parquet files found in {RAW_DIR}. Run fetch_alpaca.py first.")

    # Guard against two raw files mapping to the same ticker key (e.g. a
    # case-collision like AAPL.parquet / aapl.parquet on a case-insensitive
    # filesystem) -- silently processing both would make the second one
    # overwrite the first's entry in norm_constants/tick_sizes with no
    # indication anything was lost.
    if len(tickers) != len(set(tickers)):
        seen, dupes = set(), set()
        for t in tickers:
            (dupes if t in seen else seen).add(t)
        raise ValueError(
            f"Duplicate ticker key(s) derived from {RAW_DIR}: {sorted(dupes)} "
            "-- check for case-collisions or stray duplicate files"
        )

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

    failures = {}

    for ticker in tickers:
        try:
            df = process_ticker(ticker)

            # Determine the training set mask as the first TRAIN_FRAC of this
            # ticker's history (row-position based, not a fixed calendar date) so
            # it scales automatically with however much history was fetched, and
            # stays consistent with the split dataset.py uses at training time.
            train_cutoff_idx = int(len(df) * TRAIN_FRAC)
            if train_cutoff_idx < 2:
                raise ValueError(
                    f"{ticker}: training split only has {train_cutoff_idx} rows "
                    f"(TRAIN_FRAC={TRAIN_FRAC}, total rows={len(df)}) -- too few to "
                    "compute stable normalization statistics"
                )
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

            if not np.isfinite(df[metadata["features"]].to_numpy()).all():
                raise ValueError(f"{ticker}: non-finite values after normalization")

            # Save processed features atomically
            out_path = os.path.join(PROCESSED_DIR, f"{ticker}_features.parquet")
            _atomic_write_parquet(df[metadata["features"]], out_path)
            print(f"  └─ Feature tensor exported: {df.shape[0]} rows, {df.shape[1]} columns -> {out_path}\n")

        except Exception as e:
            failures[ticker] = str(e)
            print(f"  └─ [FAILED] {ticker}: {e}\n")

    if failures:
        # Don't write a metadata.json that references tickers whose feature
        # files don't actually exist -- that would silently corrupt
        # downstream training with a cache that looks complete but isn't.
        summary = "\n".join(f"  - {t}: {msg}" for t, msg in failures.items())
        raise RuntimeError(
            f"Preprocessing failed for {len(failures)}/{len(tickers)} ticker(s); "
            f"metadata.json NOT written.\n{summary}"
        )

    # Persist normalization and metadata for the environment/model layers to use.
    # Written last and atomically, so its existence + completeness is a
    # reliable signal that every ticker's feature file is present and valid.
    meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
    _atomic_write_json(metadata, meta_path)
    print(f"Preprocessing complete. Metadata saved to {meta_path}")

if __name__ == "__main__":
    generate_features_and_metadata()