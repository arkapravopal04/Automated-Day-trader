"""
alpaca_data.py — replaces data.py for Alpaca Markets.

Fetches up to 6 years of 5m historical bars for US stocks via
Alpaca REST API, caches to disk, and streams live 5m bars via
WebSocket for paper trading.

No KYC required — just sign up at alpaca.markets and get your
API key from the dashboard. Paper trading is completely free.

Install:
    pip install alpaca-py pyarrow

Get your keys:
    https://app.alpaca.markets/paper/dashboard/overview
    → click "API Keys" → generate → paste below

Note:
    Free tier uses IEX feed (real time but single exchange).
    Data is still excellent for training — 6 years of 5m bars.
"""

import os
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed


from dotenv import find_dotenv , load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# ── Auth — paste from https://app.alpaca.markets/paper/dashboard/overview ─────
API_KEY    = os.environ.get("API_KEY")
API_SECRET = os.environ.get("API_SECRET")

# ── Tickers ───────────────────────────────────────────────────────────────────
# Liquid US stocks — good analogues to the Indian large caps you were training on
# RELIANCE → XOM  |  TCS → INFY (US listed)  |  ICICIBANK → JPM
# or just use the most liquid names:
TICKERS = ["SPY", "QQQ", "IWM", "XLE", "XBI", "GLD", "USO", "ARKK", "AAPL", "NVDA"]
# ── Config ────────────────────────────────────────────────────────────────────
TIMEFRAME     = TimeFrame(5, TimeFrameUnit.Minute)
WINDOW_SIZE   = 48           # 48 × 5m = 240 minutes of context
FEATURES      = ["open", "high", "low", "close", "volume"]

# Alpaca Free Tier has a strict 6-year historical data limit.
# We calculate the start dynamically to avoid 403 blocks for dates > 6 years old.
_six_years_ago = datetime.now(timezone.utc) - timedelta(days=6 * 365)
HISTORY_START   = _six_years_ago.strftime("%Y-%m-%d")

CHUNK_DAYS    = 30           # Alpaca handles large requests fine but 30d is safe
REQUEST_DELAY = 0.3          # seconds between paginated requests
ET            = ZoneInfo("America/New_York")

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = "./alpaca_cache"
USE_CACHE = True             # set False to force full re-fetch


# ─────────────────────────────────────────────────────────────────────────────
# REST client
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> StockHistoricalDataClient:
    return StockHistoricalDataClient(API_KEY, API_SECRET)


# ─────────────────────────────────────────────────────────────────────────────
# cache helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker}_5m.pkl")

def _load_cache(ticker: str) -> pd.DataFrame:
    path = _cache_path(ticker)
    if os.path.exists(path):
        df = pd.read_pickle(path)
        print(f"[alpaca_data] cache hit — {ticker}: {len(df):,} candles from disk")
        return df
    return pd.DataFrame()

def _save_cache(ticker: str, df: pd.DataFrame):
    df.to_pickle(_cache_path(ticker))
    print(f"[alpaca_data] cached → {_cache_path(ticker)}  ({len(df):,} candles)")

def _cache_is_fresh(df: pd.DataFrame) -> bool:
    """Fresh if cache already has data up to yesterday."""
    if df.empty:
        return False
    latest    = df.index.max()
    # make timezone-aware if needed
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    return latest >= yesterday


# ─────────────────────────────────────────────────────────────────────────────
# REST: single chunk fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_chunk(ticker: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    client = _get_client()
    req = StockBarsRequest(
        symbol_or_symbols = ticker,
        timeframe         = TIMEFRAME,
        start             = from_dt,
        end               = to_dt,
        feed              = DataFeed.IEX,   # free tier — no subscription needed
    )
    for attempt in range(3):
        try:
            bars = client.get_stock_bars(req)
            break
        except Exception as e:
            print(f"\n[alpaca_data] attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    else:
        return pd.DataFrame()

    df = bars.df
    if df.empty:
        return df

    # alpaca returns MultiIndex (symbol, timestamp) — drop symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df[FEATURES].copy()
    df.columns = FEATURES
    return df.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# REST: paginated full-history fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_range(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    start = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(to_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    chunks      = []
    chunk_start = start
    total_days  = max((end - start).days, 1)
    done_days   = 0

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        df_chunk  = _fetch_chunk(ticker, chunk_start, chunk_end)

        if not df_chunk.empty:
            chunks.append(df_chunk)

        done_days += (chunk_end - chunk_start).days
        pct        = done_days / total_days * 100
        n          = len(df_chunk) if not df_chunk.empty else 0
        print(f"[alpaca_data] {chunk_start.date()} → {chunk_end.date()}"
              f"  {n:5d} candles  ({pct:.0f}% done)", end="\r")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    print()

    if not chunks:
        raise ValueError(f"No candles returned for {ticker} ({from_date} → {to_date})")

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# public API — mirrors data.py interface exactly
# ─────────────────────────────────────────────────────────────────────────────

def load_data(ticker: str,
              start_date: str = None,
              end_date: str = None) -> pd.DataFrame:
    """
    Mirrors data.load_data().

    First run  : fetches full 6-year history, saves to cache (~3-5 min).
    Later runs : loads cache, tops up only missing days (seconds).

    Returns raw OHLCV DataFrame (prices, not pct-change).
    """

    today     = datetime.now(timezone.utc).date()
    to_date   = end_date   or str(today)
    from_date = start_date or HISTORY_START

    # try cache
    cached = pd.DataFrame()
    if USE_CACHE:
        cached = _load_cache(ticker)

    if not cached.empty and _cache_is_fresh(cached):
        df = cached

    elif not cached.empty:
        latest = str((cached.index.max().date() + timedelta(days=1)))
        print(f"[alpaca_data] topping up {ticker} from {latest} → {to_date}")
        try:
            new_data = _fetch_range(ticker, latest, to_date)
            df = pd.concat([cached, new_data]).sort_index()
            df = df[~df.index.duplicated(keep="first")]
            if USE_CACHE:
                _save_cache(ticker, df)
        except ValueError:
            # no new candles — weekend/holiday, cache is fine
            print(f"[alpaca_data] no new candles (weekend/holiday) — using cache as-is")
            df = cached

    else:
        # full fetch
        print(f"[alpaca_data] full fetch for {ticker} "
              f"({from_date} → {to_date}) — first time only, cached after")
        df = _fetch_range(ticker, from_date, to_date)
        if USE_CACHE:
            _save_cache(ticker, df)

    # Convert DataFrame index to timezone-aware UTC for accurate slicing
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Slice the historical dataframe to the requested start and end dates
    from_dt = pd.to_datetime(from_date, utc=True)
    to_dt   = pd.to_datetime(to_date, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df = df.loc[from_dt:to_dt]

    # US market hours only — 9:30 to 16:00 ET
    df_et = df.copy()
    if not df_et.empty:
        df_et.index = df_et.index.tz_convert(ET)
        df_et = df_et.between_time("09:30", "16:00")
        # convert back to UTC for consistency
        df_et.index = df_et.index.tz_convert("UTC")

    if len(df_et) > 0:
        print(f"[alpaca_data] {ticker}: {len(df_et):,} candles ready  "
              f"({df_et.index.min().date()} → {df_et.index.max().date()})")
    else:
        print(f"[alpaca_data] {ticker}: 0 candles ready within requested range ({from_date} → {to_date})")
        
    return df_et


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    pct = df.pct_change()

    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        gaps     = df.index.to_series().diff()
        boundary = gaps > pd.Timedelta(minutes=10)
        pct[boundary] = np.nan   # entire row → removed cleanly by dropna below

    return pct.dropna()


def build_windows(transformed: pd.DataFrame, window_size: int,
                  raw_data: pd.DataFrame = None):
    X, y, prices = [], [], []
    close_col = transformed.columns.get_loc("close")

    for i in range(len(transformed) - window_size):
        window = transformed.iloc[i: i + window_size].values
        if not np.isfinite(window).all():
            continue
        window = np.clip(window, -0.5, 0.5)

        label = 1 if transformed.iloc[i + window_size, close_col] > 0 else 0
        if raw_data is not None:
            idx = i + window_size + 1
            if idx >= len(raw_data):
                continue   # no valid execution price — skip this window
            exec_price = float(raw_data.iloc[idx]["open"])
        else:
            exec_price = 0.0

        X.append(window)
        y.append(label)
        prices.append(exec_price)

    return (
        np.array(X,      dtype=np.float64),
        np.array(y,      dtype=np.int32),
        np.array(prices, dtype=np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: live 5m bar stream
# ─────────────────────────────────────────────────────────────────────────────

class LiveCandleStream:

    def __init__(self, ticker: str,
                 buffer_size: int = WINDOW_SIZE + 10,
                 bar_minutes: int = 5):
        self.ticker       = ticker
        self.buffer_size  = buffer_size
        self.bar_minutes  = bar_minutes

        self._raw_buffer: list = []
        self._lock        = threading.Lock()
        self._stream      = None
        self._thread      = None
        self._running     = False

        # current 5m bar being built from 1m bars
        self._bar_open    = None
        self._bar_high    = None
        self._bar_low     = None
        self._bar_last    = None
        self._bar_volume  = 0.0
        self._bar_count   = 0   # 1m bars accumulated in current 5m bar

        self._seed_from_rest()

    def _seed_from_rest(self):
        today     = datetime.now(timezone.utc).date()
        from_date = str(today - timedelta(days=5))
        to_date   = str(today)
        try:
            df = _fetch_range(self.ticker, from_date, to_date)
            df_et = df.copy()
            df_et.index = df_et.index.tz_convert(ET)
            df_et = df_et.between_time("09:30", "16:00")
            for _, row in df_et.iterrows():
                self._raw_buffer.append(row[FEATURES].values.astype(np.float64))
            self._raw_buffer = self._raw_buffer[-self.buffer_size:]
            print(f"[LiveCandleStream] seeded {len(self._raw_buffer)} bars for {self.ticker}")
        except Exception as e:
            print(f"[LiveCandleStream] seed failed ({e}) — relying on live data")

    async def _handle_bar(self, bar):
        """
        Alpaca streams 1m bars — aggregate 5 of them into one 5m bar.
        """
        ltp = float(bar.close)
        vol = float(bar.volume)

        if self._bar_open is None:
            self._bar_open = float(bar.open)

        self._bar_high    = max(self._bar_high or ltp, float(bar.high))
        self._bar_low     = min(self._bar_low  or ltp, float(bar.low))
        self._bar_last    = ltp
        self._bar_volume += vol
        self._bar_count  += 1

        if self._bar_count >= self.bar_minutes:
            completed = np.array([
                self._bar_open, self._bar_high, self._bar_low,
                self._bar_last, self._bar_volume
            ], dtype=np.float64)
            with self._lock:
                self._raw_buffer.append(completed)
                if len(self._raw_buffer) > self.buffer_size + 5:
                    self._raw_buffer.pop(0)

            print(f"[LiveCandleStream] 5m bar closed  "
                  f"close={self._bar_last:.4f}  "
                  f"buffer={len(self._raw_buffer)}")

            # reset
            self._bar_open   = None
            self._bar_high   = None
            self._bar_low    = None
            self._bar_last   = None
            self._bar_volume = 0.0
            self._bar_count  = 0

    def _run_stream(self):
        import asyncio
        self._stream = StockDataStream(API_KEY, API_SECRET, feed=DataFeed.IEX)
        self._stream.subscribe_bars(self._handle_bar, self.ticker)
        self._stream.run()

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run_stream, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()

    def get_window(self, window_size: int):
        """
        Returns (window, price):
          window : (window_size, 5) float64 pct-change
          price  : float — raw close of the newest completed 5m bar
        Returns (None, None) if not enough data yet.
        """
        with self._lock:
            buf = list(self._raw_buffer)

        if len(buf) < window_size + 1:
            return None, None

        raw   = np.array(buf[-(window_size + 1):], dtype=np.float64)
        pct   = np.diff(raw, axis=0) / (raw[:-1] + 1e-8)
        price = float(buf[-1][3])
        pct   = np.clip(pct, -0.5, 0.5)
        return pct.astype(np.float64), price