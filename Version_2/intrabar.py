"""
intrabar.py -- collapse a 1-minute cache onto the 5-minute decision grid,
keeping what the aggregation would otherwise throw away.

WHY THIS EXISTS
---------------
Five-minute OHLCV is a lagged, aggregated, lossy view of order flow, and the
P2 study established that the resulting panel holds one weak factor against a
~0.78 bps break-even. Capacity cannot manufacture information the inputs do
not carry. The same Alpaca endpoint serves 1-minute bars for the same price,
so the cheapest available upgrade is five observations per decision instead
of one.

The DECISION GRID DOES NOT CHANGE. This module does not make the system trade
five times as often -- everything downstream (BARS_PER_DAY=78, the session
cap, the holds in bars, the cost model) keeps its meaning. What changes is
that each 5-minute row now also carries the shape of the five minutes inside
it: where in the bar the volume landed, how much of the move was one print,
whether the path trended or chopped, and a genuine 5-observation realized
volatility instead of a 12-bar rolling proxy.

OUTPUT
------
One parquet per ticker containing three groups of columns:

  1. The 5-minute OHLCV bar itself (`open`, `high`, `low`, `close`, `volume`,
     `vwap`, `trade_count`) -- byte-compatible with what `fetch_alpaca.py`
     writes at `ALPACA_BAR_MINUTES=5`, so `preprocess.py` reads this directory
     as a drop-in replacement for `data/parquet/` and every existing feature
     is reproduced unchanged. `verify_against_5min()` checks that claim
     against the API rather than asserting it.

     ONE COLUMN IS NOT EXACT: `vwap`. Alpaca computes a bar's VWAP from every
     print in the window; the aggregate rebuilds it as the volume-weighted
     mean of the minutes' own VWAPs. Measured on AAPL, 6,475 bars, 2026-04-30
     to 2026-08-28: mean relative error 1.5e-5, and the feature that reads it
     (`vwap_dev`, the strongest single signal in the panel at IC -0.0044,
     t=-6.0) correlates 0.9978 with the same feature built on native bars --
     the error is 6.6% of the feature's own standard deviation, which
     attenuates a measured IC by about 0.2%. OHLC, volume and trade_count are
     EXACT to the last bit. If that 6.6% ever needs removing, the fix is a
     separate native 5-min fetch used for the `vwap` column alone; it was not
     judged worth a second pass over 100 tickers.

  2. `ib_*` -- the intrabar features, the actual point of the exercise.

  3. `x_*` -- intra-window EXECUTION MARKS, not features. These exist so the
     convention table can price a fill somewhere other than `open[t+1]`:
     with 1-minute resolution the book can enter at the close of the first
     minute, at that minute's VWAP, or across the first two minutes, instead
     of accepting the window's opening print as given. `preprocess.py` never
     sees these -- it selects FEATURE_COLUMNS -- so they ride along in the
     raw frame for `eval/` to read directly.

NAMING IS MIRROR-SENSITIVE, READ THIS BEFORE ADDING A COLUMN
------------------------------------------------------------
`VecTradingEnv` decides which features to sign-flip for a mirrored stream by
substring match against `return_feature_keywords = ("ret", "mom", "vwap",
"pres", "resid")`. So a DIRECTION-SENSITIVE `ib_*` column MUST contain one of
those substrings and a MAGNITUDE-ONLY one MUST NOT. That is why the signed
columns here are named `..._ret_..` / `.._pres` and the unsigned ones avoid
those letters entirely. Getting it backwards does not raise -- it feeds
mirrored streams flipped prices against unflipped signals, which is the
Session 3 defect that made the cross-section fictitious.

  direction-sensitive : ib_ret_first, ib_ret_last, ib_ret_skew,
                        ib_flow_pres, ib_clv_pres
  magnitude-only      : ib_rv, ib_jump, ib_eff, ib_vol_center, ib_vol_hhi,
                        ib_range_ov, ib_tsize

Usage
-----
    python intrabar.py                       # data/parquet_1min -> data/parquet_agg5
    python intrabar.py --grid-minutes 5
    python intrabar.py --tickers AAPL MSFT
    python intrabar.py --verify AAPL         # aggregated bars vs the API's own 5-min
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from paths import BASE_DIR

# Source and destination default to siblings of the 5-min cache. Neither is
# ever the 5-min cache itself: `fetch_alpaca.py` merges on timestamp, so a
# directory holding two cadences is one incoherent series and nothing raises.
DEFAULT_SRC = os.path.join(BASE_DIR, "parquet_1min")
DEFAULT_DST = os.path.join(BASE_DIR, "parquet_agg5")

# Columns Alpaca returns for a bar request. `trade_count` is not in
# preprocess.py's REQUIRED_RAW_COLUMNS but it is the one order-flow
# observable that comes free with the bar, so it is carried through and
# `ib_tsize` is built from it.
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]

IB_DIRECTIONAL = ["ib_ret_first", "ib_ret_last", "ib_ret_skew", "ib_flow_pres", "ib_clv_pres"]
IB_MAGNITUDE = ["ib_rv", "ib_jump", "ib_eff", "ib_vol_center", "ib_vol_hhi", "ib_range_ov", "ib_tsize"]
IB_COLUMNS = IB_DIRECTIONAL + IB_MAGNITUDE

# Execution marks. `x_open` duplicates `open` on purpose: the convention table
# reads this group alone, and having the incumbent convention sit beside the
# alternatives keeps the comparison in one frame.
X_COLUMNS = ["x_open", "x_close_m1", "x_vwap_m1", "x_vwap_m12", "x_vwap_full"]

EPS = 1e-12


def _atomic_write_parquet(df: pd.DataFrame, out_path: str) -> None:
    """Temp file + os.replace, matching preprocess.py and fetch_alpaca.py. A
    crash mid-write leaves a stray temp file, never a corrupted cache."""
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=out_dir)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, engine="pyarrow")
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _group_bounds(codes: np.ndarray) -> np.ndarray:
    """Start index of each run in a sorted, non-decreasing code array."""
    if codes.size == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(np.diff(codes)) + 1
    return np.concatenate(([0], starts)).astype(np.int64)


def aggregate(df_1min: pd.DataFrame, grid_minutes: int = 5) -> pd.DataFrame:
    """Collapse 1-minute bars onto a ``grid_minutes`` grid, with intrabar columns.

    Args:
        df_1min: 1-minute bars, tz-aware UTC index, Alpaca's column set.
        grid_minutes: width of the decision bar. 5 keeps every downstream
            constant (BARS_PER_DAY, holds, the session cap) meaning what it
            already means.

    Returns:
        One row per non-empty grid bin, indexed by the bin's left edge --
        the same labelling Alpaca uses for a native bar of that width.

    Bins are formed by flooring the timestamp, NOT by ``resample``: resample
    materialises every empty bin between gaps, and a 6-year RTH cache has an
    empty bin for every overnight and every holiday. Flooring plus factorize
    touches only bins that contain data.

    A bin with fewer than ``grid_minutes`` rows is kept, not dropped. Missing
    minutes are real (illiquid names genuinely do not print every minute),
    and dropping them would punch holes through the middle of the series
    rather than trim its edges. Quantities that need at least two
    observations degrade to their defined value on one (``ib_jump`` to 0,
    ``ib_rv`` to the single return's magnitude) rather than to NaN.
    """
    if df_1min.empty:
        return pd.DataFrame(columns=BAR_COLUMNS + IB_COLUMNS + X_COLUMNS)

    df = df_1min.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    bin_edges = df.index.floor(f"{grid_minutes}min")
    codes, uniq = pd.factorize(bin_edges)
    codes = codes.astype(np.int64)
    n_bins = len(uniq)

    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64)
    vw = df["vwap"].to_numpy(np.float64) if "vwap" in df.columns else c
    tc = df["trade_count"].to_numpy(np.float64) if "trade_count" in df.columns else np.zeros_like(v)

    starts = _group_bounds(codes)
    ends = np.append(starts[1:], len(codes)) - 1          # last row index per bin
    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)

    # ---- the grid bar itself -------------------------------------------
    bar_open = o[starts]
    bar_close = c[ends]
    bar_high = np.maximum.reduceat(h, starts)
    bar_low = np.minimum.reduceat(l, starts)
    bar_vol = np.bincount(codes, weights=v, minlength=n_bins)
    bar_tc = np.bincount(codes, weights=tc, minlength=n_bins)
    # VWAP of a union of intervals is the volume-weighted mean of their VWAPs.
    # Zero-volume minutes contribute nothing and must not drag the mean, so
    # the fallback for a fully zero-volume bin is the close, not a 0/0 nan.
    notional = np.bincount(codes, weights=vw * v, minlength=n_bins)
    bar_vwap = np.where(bar_vol > 0, notional / np.where(bar_vol > 0, bar_vol, 1.0), bar_close)

    # ---- within-bin returns --------------------------------------------
    # r_j = log(close_j / close_{j-1}) inside the bin; the first minute uses
    # its own open, so the five returns tile open->close continuously and
    # sum to the bar's own log return (up to the negligible close-to-open
    # seams between consecutive minutes).
    is_start = np.zeros(len(codes), dtype=bool)
    is_start[starts] = True
    prev_c = np.empty_like(c)
    prev_c[1:] = c[:-1]
    prev_c[0] = o[0]
    base = np.where(is_start, o, prev_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.where((c > 0) & (base > 0), c / np.where(base > 0, base, 1.0), 1.0))
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    sum_r2 = np.bincount(codes, weights=r * r, minlength=n_bins)
    sum_abs_r = np.bincount(codes, weights=np.abs(r), minlength=n_bins)
    pos_r2 = np.bincount(codes, weights=np.where(r > 0, r * r, 0.0), minlength=n_bins)
    neg_r2 = np.bincount(codes, weights=np.where(r < 0, r * r, 0.0), minlength=n_bins)

    # Bipower variation: (pi/2) * sum |r_j||r_{j-1}| over adjacent pairs
    # INSIDE a bin. Cross-bin pairs are excluded by zeroing the term at each
    # bin start -- including them would mix an overnight gap into a jump
    # estimate. BV estimates the diffusive part only, so 1 - BV/RV is the
    # share of the bar's variance that arrived as a discontinuity.
    abs_r = np.abs(r)
    prev_abs = np.empty_like(abs_r)
    prev_abs[1:] = abs_r[:-1]
    prev_abs[0] = 0.0
    pair = np.where(is_start, 0.0, abs_r * prev_abs)
    sum_pair = np.bincount(codes, weights=pair, minlength=n_bins)
    n_pairs = np.maximum(counts - 1.0, 1.0)
    bpv = (np.pi / 2.0) * sum_pair * (counts / n_pairs)
    ib_jump = np.clip(1.0 - bpv / np.maximum(sum_r2, EPS), 0.0, 1.0)
    ib_jump = np.where(counts >= 2, ib_jump, 0.0)

    # ---- volume placement ----------------------------------------------
    # Position of each minute inside its bin, mapped to (0, 1) as
    # (j + 0.5) / n_bin. Midpoints rather than j/(n-1) so a single-minute bin
    # is 0.5 (the honest "no timing information") instead of a divide by zero.
    pos_in_bin = np.arange(len(codes), dtype=np.float64) - starts[codes].astype(np.float64)
    frac = (pos_in_bin + 0.5) / counts[codes]
    vol_safe = np.where(bar_vol > 0, bar_vol, 1.0)
    ib_vol_center = np.where(
        bar_vol > 0,
        np.bincount(codes, weights=v * frac, minlength=n_bins) / vol_safe,
        0.5,
    )
    share = v / vol_safe[codes]
    ib_vol_hhi = np.where(bar_vol > 0, np.bincount(codes, weights=share * share, minlength=n_bins), 1.0)

    # ---- signed flow ----------------------------------------------------
    # Tick rule at 1-minute resolution: volume signed by the sign of the
    # minute's return, normalised by bar volume. This is a PROXY for
    # Lee-Ready signed order flow -- it signs a minute's whole volume by that
    # minute's net direction, where Lee-Ready signs each print against the
    # prevailing quote. It is the best available from bars, and it is exactly
    # the quantity trade/quote data would replace with the real thing.
    ib_flow_pres = np.where(
        bar_vol > 0,
        np.bincount(codes, weights=np.sign(r) * v, minlength=n_bins) / vol_safe,
        0.0,
    )

    # Close-location value per minute, volume-weighted across the bin.
    # high == low is a real, common state on a quiet minute (a flat bar
    # carries no directional pressure), so 0.0 is the honest encoding --
    # the same choice preprocess.py makes for `intrabar_pres`.
    rng_1m = h - l
    clv = np.where(rng_1m > 0, ((c - l) - (h - c)) / np.where(rng_1m > 0, rng_1m, 1.0), 0.0)
    ib_clv_pres = np.where(
        bar_vol > 0,
        np.bincount(codes, weights=clv * v, minlength=n_bins) / vol_safe,
        0.0,
    )

    # ---- path shape ------------------------------------------------------
    net_move = np.abs(np.log(np.maximum(bar_close, EPS) / np.maximum(bar_open, EPS)))
    ib_eff = np.where(sum_abs_r > EPS, net_move / np.maximum(sum_abs_r, EPS), 0.0)
    ib_eff = np.clip(ib_eff, 0.0, 1.0)

    sum_rng_1m = np.bincount(codes, weights=rng_1m, minlength=n_bins)
    ib_range_ov = np.where(sum_rng_1m > 0, (bar_high - bar_low) / np.maximum(sum_rng_1m, EPS), 0.0)
    ib_range_ov = np.clip(ib_range_ov, 0.0, 1.0)

    # Volatility and average trade size are both strongly right-skewed and
    # both get per-ticker z-scored downstream. Logging them here keeps the
    # resulting z from reaching the three-digit range a raw heavy tail
    # produces -- the failure mode report.md 11.2 documents for split jumps.
    ib_rv = np.log(np.sqrt(sum_r2) + 1e-6)
    ib_tsize = np.log1p(np.where(bar_tc > 0, bar_vol / np.maximum(bar_tc, 1.0), 0.0))

    ib_ret_skew = (pos_r2 - neg_r2) / np.maximum(sum_r2, EPS)
    ib_ret_first = r[starts]
    ib_ret_last = r[ends]

    # ---- execution marks -------------------------------------------------
    # The first minute of a bar, and the first two. `open[t+1]` is what the
    # execution frame currently assumes; these are the alternatives 1-minute
    # resolution makes available. Second-minute rows exist only where the bin
    # actually has two minutes, so the two-minute VWAP falls back to the
    # one-minute one rather than silently reaching into the next bar.
    second = np.minimum(starts + 1, ends)
    has_second = (ends > starts)
    v1, v2 = v[starts], np.where(has_second, v[second], 0.0)
    vw1, vw2 = vw[starts], np.where(has_second, vw[second], 0.0)
    v12 = v1 + v2
    x_vwap_m12 = np.where(v12 > 0, (vw1 * v1 + vw2 * v2) / np.where(v12 > 0, v12, 1.0), c[starts])

    out = pd.DataFrame(
        {
            "open": bar_open, "high": bar_high, "low": bar_low, "close": bar_close,
            "volume": bar_vol, "vwap": bar_vwap, "trade_count": bar_tc,
            "ib_ret_first": ib_ret_first, "ib_ret_last": ib_ret_last,
            "ib_ret_skew": ib_ret_skew, "ib_flow_pres": ib_flow_pres,
            "ib_clv_pres": ib_clv_pres, "ib_rv": ib_rv, "ib_jump": ib_jump,
            "ib_eff": ib_eff, "ib_vol_center": ib_vol_center, "ib_vol_hhi": ib_vol_hhi,
            "ib_range_ov": ib_range_ov, "ib_tsize": ib_tsize,
            "x_open": bar_open,
            "x_close_m1": c[starts],
            "x_vwap_m1": np.where(v1 > 0, vw1, c[starts]),
            "x_vwap_m12": x_vwap_m12,
            "x_vwap_full": bar_vwap,
            "n_minutes": counts,
        },
        index=pd.DatetimeIndex(uniq, name=df.index.name or "timestamp"),
    )

    bad = out[IB_COLUMNS].to_numpy()
    if not np.isfinite(bad).all():
        n_bad = int((~np.isfinite(bad)).sum())
        raise ValueError(f"aggregate(): {n_bad} non-finite values in ib_* columns")
    return out


def build(src_dir: str, dst_dir: str, tickers: Optional[List[str]] = None,
          grid_minutes: int = 5) -> Dict[str, int]:
    """Aggregate every ticker parquet in ``src_dir`` into ``dst_dir``."""
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"1-minute cache not found at {src_dir}. Build it with:\n"
            f"  ALPACA_BAR_MINUTES=1 ALPACA_RTH_ONLY=1 ALPACA_CHUNK_DAYS=180 "
            f"ALPACA_FETCH_WORKERS=6 python fetch_alpaca.py"
        )
    os.makedirs(dst_dir, exist_ok=True)

    if tickers is None:
        tickers = sorted(f[:-8] for f in os.listdir(src_dir) if f.endswith(".parquet"))

    written: Dict[str, int] = {}
    for i, ticker in enumerate(tickers, 1):
        src = os.path.join(src_dir, f"{ticker}.parquet")
        if not os.path.exists(src):
            print(f"[{i}/{len(tickers)}] {ticker}: no 1-min parquet, skipping")
            continue
        df = pd.read_parquet(src)
        agg = aggregate(df, grid_minutes=grid_minutes)
        _atomic_write_parquet(agg, os.path.join(dst_dir, f"{ticker}.parquet"))
        written[ticker] = len(agg)
        short = int((agg["n_minutes"] < grid_minutes).sum())
        print(
            f"[{i}/{len(tickers)}] {ticker}: {len(df):>8,} 1-min -> {len(agg):>7,} "
            f"{grid_minutes}-min bars ({short / max(len(agg), 1):.1%} incomplete)"
        )
    return written


def verify_against_5min(ticker: str, src_dir: str = DEFAULT_SRC,
                        days: int = 30, grid_minutes: int = 5) -> None:
    """Compare aggregated bars against the API's own bars of the same width.

    The whole drop-in claim rests on this: if `preprocess.py` is to read the
    aggregated directory and reproduce the existing features, an aggregated
    bar has to BE the native bar. Rather than assert that, fetch both for a
    recent window and print the disagreement per column.

    VWAP is expected to differ slightly and is reported separately -- Alpaca
    computes it from every print in the window, while the aggregate rebuilds
    it as a volume-weighted mean of the minutes' own VWAPs. OHLC and volume
    are expected to match to floating-point noise.
    """
    from datetime import datetime, timedelta, timezone
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.enums import Adjustment
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    import fetch_alpaca

    key, secret = fetch_alpaca.get_alpaca_credentials()
    if not key:
        raise RuntimeError("no Alpaca credentials -- cannot verify against the API")
    client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    ours = aggregate(pd.read_parquet(os.path.join(src_dir, f"{ticker}.parquet")),
                     grid_minutes=grid_minutes)
    end = ours.index.max()
    start = end - timedelta(days=days)

    native = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(grid_minutes, TimeFrameUnit.Minute),
        start=start.to_pydatetime(), end=end.to_pydatetime(),
        feed=os.getenv("ALPACA_DATA_FEED", "sip"), adjustment=Adjustment.ALL,
    )).df.loc[ticker]

    common = ours.index.intersection(native.index)
    print(f"[verify] {ticker}: {len(common)} overlapping {grid_minutes}-min bars "
          f"({start.date()}..{end.date()})")
    print(f"[verify] bars only in aggregate: {len(ours.index.difference(native.index))}  "
          f"only in API: {len(native.index.difference(ours.index))}")
    if len(common) == 0:
        return
    a, b = ours.loc[common], native.loc[common]
    for col in ["open", "high", "low", "close", "volume", "vwap", "trade_count"]:
        if col not in b.columns:
            continue
        x, y = a[col].to_numpy(np.float64), b[col].to_numpy(np.float64)
        denom = np.maximum(np.abs(y), EPS)
        rel = np.abs(x - y) / denom
        print(f"  {col:<12} max_rel={rel.max():.3e}  mean_rel={rel.mean():.3e}  "
              f"exact={np.isclose(x, y, rtol=1e-9, atol=0).mean():.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help=f"1-minute cache (default {DEFAULT_SRC})")
    ap.add_argument("--dst", default=DEFAULT_DST, help=f"output directory (default {DEFAULT_DST})")
    ap.add_argument("--tickers", nargs="*", default=None, help="subset of tickers (default: all in --src)")
    ap.add_argument("--grid-minutes", type=int, default=5,
                    help="decision-bar width in minutes (default 5 -- keeps BARS_PER_DAY=78)")
    ap.add_argument("--verify", metavar="TICKER", default=None,
                    help="compare aggregated bars against the API's native bars and exit")
    args = ap.parse_args()

    if args.verify:
        verify_against_5min(args.verify, src_dir=args.src, grid_minutes=args.grid_minutes)
        return

    written = build(args.src, args.dst, tickers=args.tickers, grid_minutes=args.grid_minutes)
    total = sum(written.values())
    print(f"\n{len(written)} tickers, {total:,} {args.grid_minutes}-min bars -> {args.dst}")
    print(f"intrabar features: {len(IB_COLUMNS)} ({len(IB_DIRECTIONAL)} directional, "
          f"{len(IB_MAGNITUDE)} magnitude-only)")


if __name__ == "__main__":
    main()
