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

# Progress lines below use box-drawing characters. Kaggle's stdout is UTF-8 so
# they render there, but a Windows console defaults to cp1252 and raises
# UnicodeEncodeError on the FIRST such print -- which lands inside pass 3's
# per-ticker try/except, whose handler then prints the same characters again
# and takes the whole run down with a secondary exception. Net effect: a full,
# successful preprocess of all 100 tickers reported as a crash. Re-encode
# instead of de-fanging the messages.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

# Configuration Parameters
HORIZONS = [3, 6, 12] # Lag steps representing 15m, 30m, and 1h past returns
RV_WINDOW = 12 # Realized volatility rolling window (1 hour)
BARS_PER_SESSION = 78 # RTH 5-min bars in one session (09:30-16:00)
VOL_WINDOW = BARS_PER_SESSION # Volume z-score rolling window (1 full trading day)

# Multi-day short-term reversal, in bars. These are NOT part of HORIZONS and
# are deliberately computed differently: HORIZONS features are session-reset
# cumulative sums that must never straddle the overnight gap, whereas reversal
# is a close-to-close effect that only exists ACROSS sessions. Running these
# through the HORIZONS loop would silently reduce them to "return since this
# morning's open" -- for any h > 78 the session reset makes the two identical.
#
# Added after the alpha gate found them to be the only signal in the panel that
# clears cost: walk-forward over 5 folds spanning 2022-10 to 2026-01 gives net
# Sharpe positive in every fold (min 0.28, mean 0.80-0.97 per name) with the
# train-split sign consistently NEGATIVE in all folds, i.e. genuine reversal.
# Everything shorter fails: the intraday reversal in the close ramp reaches
# t=6.2 and still prices at Sharpe -10 because the hold is too short to
# amortise the 9.1 bps round trip.
REVERSAL_HORIZONS = {
    'log_ret_5d': 5 * BARS_PER_SESSION,
    'log_ret_20d': 20 * BARS_PER_SESSION,
}

# Feature columns, in the order they appear in the tensor. Defined once here
# so process_ticker()'s finite-check and generate_features_and_metadata()'s
# metadata registry can never drift apart.
#
# Direction-sensitive features (sign-flipped for mirrored streams by
# VecTradingEnv) must contain one of env.return_feature_keywords in their
# name: 'log_ret*', 'overnight_ret' and 'xs_resid' match "ret"/"resid",
# 'vwap_dev' matches "vwap", 'intrabar_pres' matches "pres". Magnitude-only
# features ('rv', 'vol_z', 'time_*', 'is_overnight') must NOT match any
# keyword -- note 'is_overnight' does not contain "ret" but 'overnight_ret'
# does, which is the intended split. Renaming a column here without checking
# that tuple silently breaks mirroring: mirrored streams get flipped prices
# against unflipped signals.
#
# 'log_ret_5d'/'log_ret_20d' match "ret" and so are sign-flipped, which is
# correct: they are directional return features like every other log_ret_*.
FEATURE_COLUMNS = (
    ['log_ret', 'overnight_ret', 'is_overnight', 'rv', 'vol_z', 'time_sin', 'time_cos']
    + [f'log_ret_{h}' for h in HORIZONS]
    + list(REVERSAL_HORIZONS)
    + ['vwap_dev', 'intrabar_pres', 'xs_resid']
)

if set(REVERSAL_HORIZONS) & {f'log_ret_{h}' for h in HORIZONS}:
    raise ValueError("REVERSAL_HORIZONS collides with a HORIZONS column name")

if len(HORIZONS) != len(set(HORIZONS)):
    raise ValueError(f"HORIZONS contains duplicate values: {HORIZONS} -- would create duplicate feature columns")

REQUIRED_RAW_COLUMNS = {"open", "high", "low", "close", "volume", "vwap"}

# Regular US equity trading hours, as minutes past midnight America/New_York:
# 09:30 (570) inclusive to 16:00 (960) exclusive == 78 five-minute bars.
#
# The SIP feed returns extended-hours bars too -- measured on AAPL, 184.6
# bars/day of which only 78.0 are RTH (IEX by contrast was 95.6% RTH, so
# this barely mattered before). Every window constant in this file assumes
# the RTH cadence: VOL_WINDOW=78 means "one trading day", rv annualises by
# sqrt(252*78), and day_fraction below clips minutes_since_open/390 to
# [0,1] -- which silently encodes every pre-market bar identically to the
# open and every post-market bar identically to the close. Filtering here
# rather than at fetch keeps the raw cache complete (overnight-gap features
# may want it later); load_aligned_close_prices() reindexes raw onto the
# feature index, so the dropped bars fall out of the price path too.
RTH_START_MIN = 9 * 60 + 30
RTH_END_MIN = 16 * 60
# Longest rolling window used anywhere below -- used to sanity-check that a
# ticker has enough history to produce any non-NaN feature rows.
MIN_ROWS_REQUIRED = max(
    VOL_WINDOW, RV_WINDOW, max(HORIZONS), max(REVERSAL_HORIZONS.values())
) + 1


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
    bad_price_mask = (df[["open", "high", "low", "close", "vwap"]] <= 0).any(axis=1)
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

    # Restrict to regular trading hours -- see RTH_START_MIN. Done before any
    # feature computation so rolling windows never span an EXTENDED-HOURS gap
    # in units they don't expect.
    #
    # It does NOT remove the overnight gap -- it concentrates it. After this
    # filter the 09:30 row directly follows the previous session's 15:55 row,
    # so a naive close/close.shift(1) there is a ~17.5-hour return. The
    # session-boundary block further down is what actually handles that; read
    # it before adding any new rolling feature here.
    ny_index = df.index.tz_convert("America/New_York")
    minutes_of_day = ny_index.hour * 60 + ny_index.minute
    rth_mask = (minutes_of_day >= RTH_START_MIN) & (minutes_of_day < RTH_END_MIN)
    n_ext = int((~rth_mask).sum())
    if n_ext:
        print(f"  [INFO] {ticker}: dropping {n_ext} extended-hours bar(s) "
              f"({n_ext / len(df):.1%}), keeping {int(rth_mask.sum())} RTH bars")
    df = df[rth_mask]

    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"{ticker}: only {len(df)} regular-hours rows remain, need at least "
            f"{MIN_ROWS_REQUIRED} to compute rolling features"
        )

    # --- Session boundaries -------------------------------------------------
    # The RTH filter above removes 09:30-16:00's complement, which means
    # consecutive rows now straddle the overnight gap: the row stamped 09:30
    # follows the row stamped 15:55 of the PREVIOUS session. Before this
    # block existed, close/close.shift(1) at that row was a ~17.5-hour return
    # living in a column the model reads as a 5-minute one -- one bar per day
    # per ticker, on all 100 tickers.
    #
    # Measured on the marking price path, that bar's median |move| is
    # 75.8 bps against 9.8 bps for every other bar of the day: a 7.7x ratio
    # that holds on 100/100 tickers (range 4.6x-9.9x). It is 8.1% of a day's
    # total absolute movement compressed into 1 of 78 slots.
    #
    # `open` is in the raw parquet, so the two components separate exactly
    # rather than approximately:
    #     overnight_ret = log(open_t / close_{t-1})   the true close-to-open gap
    #     log_ret       = log(close_t / open_t)       the true first 5 minutes
    # Every value in 'log_ret' is therefore a genuine intra-session 5-minute
    # return, which is what makes the rv annualisation and the multi-horizon
    # sums below dimensionally correct.
    session_date = pd.Series(
        df.index.tz_convert("America/New_York").normalize(), index=df.index
    )
    is_session_start = session_date != session_date.shift(1)
    prev_close = df['close'].shift(1)

    # 1a. Overnight (close-to-open) return -- zero on every intra-session bar.
    # Direction-sensitive; the name contains "ret" so VecTradingEnv sign-flips
    # it for mirrored streams along with the other return columns.
    df['overnight_ret'] = np.where(
        is_session_start, np.log(df['open'] / prev_close), 0.0
    )

    # 1b. Session-start indicator. Redundant with time_sin/time_cos in
    # principle (both are deterministic functions of minute-of-day) but it
    # makes "this row carries a gap" a single channel the policy can gate on
    # instead of a trigonometric coincidence it has to infer. Magnitude-only:
    # the name must NOT match any of env.return_feature_keywords, or mirrored
    # streams would see it negated.
    df['is_overnight'] = is_session_start.astype(np.float64)

    # 1. Immediate intra-session log return (t vs t-1, or open->close on the
    # session's first bar). Never spans the overnight gap.
    df['log_ret'] = np.where(
        is_session_start,
        np.log(df['close'] / df['open']),
        np.log(df['close'] / prev_close),
    )

    # 2. Multi-Horizon Log Returns (Captures short-term momentum)
    # Cumulative sum of intra-session returns over the trailing h bars, reset
    # at each session open. The previous close/close.shift(h) form silently
    # straddled the boundary for the first h bars of every session -- 12 of
    # 78 bars/day at the longest horizon -- folding the overnight gap into
    # what the model reads as intraday momentum. Where fewer than h bars have
    # elapsed this is the return since the open, which is the honest answer
    # rather than a NaN that dropna() would punch a hole with.
    cum = df['log_ret'].groupby(session_date).cumsum()
    for h in HORIZONS:
        prior = cum.shift(h).where(session_date == session_date.shift(h), 0.0)
        df[f'log_ret_{h}'] = cum - prior

    # 2b. Multi-day short-term reversal (see REVERSAL_HORIZONS).
    # Plain close-to-close over h bars, deliberately crossing session
    # boundaries -- the effect lives in the multi-day drift, so the session
    # reset applied above would destroy it rather than protect it. Leading
    # rows are NaN and are removed by the dropna() below, costing one warmup
    # window per ticker (20 sessions at the longest horizon).
    log_close = np.log(df['close'])
    for name, h in REVERSAL_HORIZONS.items():
        df[name] = log_close - log_close.shift(h)

    # 3. Annualized Realized Volatility
    # Calculation: standard deviation of recent returns scaled by annualizing factor
    # (78 5-min bars/day * 252 trading days/year). Valid only because 'log_ret'
    # is now purely intra-session -- the window may span a session boundary,
    # but every element in it is a true 5-minute return.
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

    # 6. VWAP deviation -- where the bar closed relative to its own
    # volume-weighted average price. Strongest single signal measured
    # (IC -0.0044, t=-6.0) vs the close-derived family's IC -0.0010 (t=-1.3,
    # indistinguishable from noise). Direction-sensitive.
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']

    # 7. Intrabar pressure -- close position within the bar's range, as
    # (distance off the low) minus (distance off the high). +1 = closed on
    # the high, -1 = closed on the low. Direction-sensitive.
    #
    # high == low on a flat/no-trade bar, which would divide by zero. Those
    # are 1.3%-11.7% of bars depending on ticker (AMGN 11.7%, AVGO 10.1%),
    # far too many to drop -- doing so would punch scattered holes through
    # the middle of every series, not just trim the leading rolling-window
    # rows. A zero-range bar carries no directional pressure information, so
    # 0.0 is the honest encoding rather than a missing value.
    rng = df['high'] - df['low']
    df['intrabar_pres'] = (
        ((df['close'] - df['low']) - (df['high'] - df['close'])) / rng.where(rng > 0)
    ).fillna(0.0)

    
    # Remove leading rows that contain NaNs due to rolling window calculations
    df.dropna(inplace=True)

    # 8. Cross-sectional residual return -- placeholder, created AFTER the
    # dropna above (an all-NaN column would otherwise delete every row).
    # Requires the whole universe on a common index, which this per-ticker
    # function cannot see; generate_features_and_metadata() fills it in a
    # second pass and finite-checks it before normalising.
    df['xs_resid'] = np.nan

    if df.empty:
        raise ValueError(
            f"{ticker}: no rows survived feature engineering (all NaN after rolling windows) "
            "-- raw history is too short or too gappy"
        )

    # Any remaining non-finite values (e.g. inf from a pathological std==0
    # edge case) would silently poison training -- fail loudly instead.
    # 'xs_resid' is still all-NaN here by construction (filled in pass 2), so
    # it is excluded -- generate_features_and_metadata() finite-checks the
    # full set, this one included, after the cross-sectional pass.
    finite_cols = [c for c in FEATURE_COLUMNS if c != 'xs_resid']
    if not np.isfinite(df[finite_cols].to_numpy()).all():
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
        "features": list(FEATURE_COLUMNS),
        "norm_constants": {},
        "tick_sizes": {ticker: 0.01 for ticker in tickers} # Assuming $0.01 standard US equity tick size
    }

    failures = {}

    # ---- Pass 1: per-ticker features -------------------------------------
    # Held in memory rather than written straight out, because 'xs_resid'
    # needs the whole universe on a common index and can only be computed
    # once every ticker is available. Slimmed to FEATURE_COLUMNS immediately
    # (the raw OHLCV columns are unused from here on) and cast to float32 --
    # ~100 tickers x ~115k rows x 11 cols is about 0.5 GB, which fits
    # comfortably in a Kaggle session.
    frames = {}
    for ticker in tickers:
        try:
            frames[ticker] = process_ticker(ticker)[list(FEATURE_COLUMNS)].astype(np.float32)
        except Exception as e:
            failures[ticker] = str(e)
            print(f"  [FAILED] {ticker}: {e}")

    if failures:
        summary = chr(10).join(f"  - {t}: {msg}" for t, msg in failures.items())
        raise RuntimeError(
            f"Feature generation failed for {len(failures)}/{len(tickers)} ticker(s); "
            f"metadata.json NOT written.{chr(10)}{summary}"
        )

    # ---- Pass 2: cross-sectional residual --------------------------------
    # resid = own return - equal-weighted universe return, on the aligned
    # union index. Measured IC -0.0029 (t=-4.0) against next-bar return,
    # roughly 3x the own-return family's -0.0010 (t=-1.3, i.e. noise).
    #
    # Computed here rather than in VecTradingEnv at runtime: the env's
    # streams are a MIX of mirrored and unmirrored tickers, so a
    # cross-sectional mean taken there would not be the real market return.
    # Computed on raw (pre-normalisation) log returns, the only scale on
    # which an equal-weighted mean means anything.
    #
    # Mirroring does NOT sign-flip 'xs_resid'. A residual is
    # own_return - market_return; mirroring inverts a stream's own path but
    # the market term is a fact about the real universe and does not invert
    # with it, so flipping introduces an error of 2x market (measured at 1.30
    # residual-sigmas, larger than the signal). VecTradingEnv's
    # cross_sectional_feature_keywords zeroes the channel for mirrored
    # streams instead.
    print("Computing cross-sectional residual returns across the universe...")
    market = pd.concat({t: f['log_ret'] for t, f in frames.items()}, axis=1)
    market_ret = market.mean(axis=1, skipna=True)
    coverage = market.notna().sum(axis=1)
    print(f"  aligned union index: {len(market_ret)} timestamps, median "
          f"{int(coverage.median())} of {len(frames)} tickers present per bar")

    for ticker, f in frames.items():
        f['xs_resid'] = (f['log_ret'] - market_ret.reindex(f.index)).astype(np.float32)
        if not np.isfinite(f['xs_resid'].to_numpy()).all():
            # Unreachable while every row of f has a finite log_ret and
            # market_ret is a skipna mean over a superset index -- assert it
            # rather than discover NaN in the observation tensor later.
            raise ValueError(f"{ticker}: non-finite values in cross-sectional residual")

    # ---- Pass 3: normalise and write -------------------------------------
    for ticker in tickers:
        try:
            df = frames[ticker]

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
            train_df = df.iloc[:train_cutoff_idx]

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
            f"Normalisation failed for {len(failures)}/{len(tickers)} ticker(s); "
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