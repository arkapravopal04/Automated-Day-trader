"""
use this everytime

Fetches historical 5m candles via Upstox REST for training,
and streams live OHLC candles via MarketDataStreamerV3 for live trading.

Install:
    pip install upstox-python-sdk pyarrow

Auth:
    Upstox uses OAuth2 — fresh access_token required every day.
    Paste it into ACCESS_TOKEN below, or run the token helper script
    in upstox_train_patch.py each morning before training.

Instrument keys:
    NSE stocks use "NSE_EQ|<ISIN>" format.
    Common ones are listed in INSTRUMENT_KEYS below.
    Full list: https://upstox.com/developer/api-documentation/instruments
"""

import os
import time
import threading
import numpy as np
import pandas as pd
import upstox_client
from upstox_client.rest import ApiException
from datetime import datetime, timedelta, timezone

# ── Auth ──────────────────────────────────────────────────────────────────────
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"

# ── Instrument keys for common NSE stocks ─────────────────────────────────────
INSTRUMENT_KEYS = {
    "RELIANCE"   : "NSE_EQ|INE002A01018",
    "TCS"        : "NSE_EQ|INE467B01029",
    "INFY"       : "NSE_EQ|INE009A01021",
    "ITC"        : "NSE_EQ|INE154A01025",
    "ICICIBANK"  : "NSE_EQ|INE090A01021",
    "ADANIPORTS" : "NSE_EQ|INE742F01042",
    "HDFCBANK"   : "NSE_EQ|INE040A01034",
    "WIPRO"      : "NSE_EQ|INE075A01022",
    "SBIN"       : "NSE_EQ|INE062A01020",
    "BAJFINANCE" : "NSE_EQ|INE296A01024",
}

INTERVAL      = "5minute"
WINDOW_SIZE   = 20           # 20 × 5m = 100 minutes of context
FEATURES      = ["open", "high", "low", "close", "volume"]

# Upstox V3 has 5m data from Jan 2022 onwards
HISTORY_START = "2022-01-03"
CHUNK_DAYS    = 90           # 90-day pages — well within per-request limits
REQUEST_DELAY = 0.5          # seconds between requests (rate limit headroom)

# ── Cache ─────────────────────────────────────────────────────────────────────
# Fetching 3 years of 5m data takes ~2 minutes per ticker.
# Cached parquet files make subsequent training runs instant.
# Set USE_CACHE = False to force a full re-fetch (e.g. after a long gap).
CACHE_DIR = "./upstox_cache"
USE_CACHE = True


# ─────────────────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_config() -> upstox_client.Configuration:
    config = upstox_client.Configuration()
    config.access_token = ACCESS_TOKEN
    return config


def _ticker_to_key(ticker: str) -> str:
    if ticker in INSTRUMENT_KEYS:
        return INSTRUMENT_KEYS[ticker]
    if ticker.endswith(".NS"):
        short = ticker.replace(".NS", "")
        if short in INSTRUMENT_KEYS:
            return INSTRUMENT_KEYS[short]
    if "|" in ticker:
        return ticker
    raise ValueError(
        f"Unknown ticker '{ticker}'. Add it to INSTRUMENT_KEYS "
        f"or pass the full key e.g. 'NSE_EQ|INE002A01018'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{ticker}_5m.parquet")


def _load_cache(ticker: str) -> pd.DataFrame:
    path = _cache_path(ticker)
    if os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"[upstox_data] cache hit — {ticker}: {len(df):,} candles loaded from disk")
        return df
    return pd.DataFrame()


def _save_cache(ticker: str, df: pd.DataFrame):
    path = _cache_path(ticker)
    df.to_parquet(path)
    print(f"[upstox_data] cache saved → {path}  ({len(df):,} candles)")


def _cache_is_fresh(df: pd.DataFrame) -> bool:
    """
    Cache is considered fresh if it already contains data up to yesterday.
    If there is a gap (e.g. a few trading days missing), we top it up
    rather than re-fetching everything.
    """
    if df.empty:
        return False
    latest     = df.index.max().date()
    yesterday  = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    return latest >= yesterday


# ─────────────────────────────────────────────────────────────────────────────
# REST: single chunk fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_chunk(instrument_key: str,
                 from_date: str,
                 to_date: str) -> pd.DataFrame:
    """One paginated REST request — returns DataFrame or empty on failure."""
    config     = _get_config()
    api_client = upstox_client.ApiClient(config)
    api        = upstox_client.HistoryV3Api(api_client)

    for attempt in range(3):
        try:
            resp = api.get_historical_candle_data1(
                instrument_key, INTERVAL, to_date, from_date
            )
            break
        except ApiException as e:
            print(f"\n[upstox_data] attempt {attempt+1} failed "
                  f"({from_date}→{to_date}): {e}")
            time.sleep(2 ** attempt)
    else:
        print(f"\n[upstox_data] skipping {from_date}→{to_date} after 3 failures")
        return pd.DataFrame()

    candles = resp.data.candles
    if not candles:
        return pd.DataFrame()

    rows = [
        {
            "datetime": pd.to_datetime(c[0]),
            "open"    : float(c[1]),
            "high"    : float(c[2]),
            "low"     : float(c[3]),
            "close"   : float(c[4]),
            "volume"  : float(c[5]),
        }
        for c in candles
    ]
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    return df[FEATURES]


# ─────────────────────────────────────────────────────────────────────────────
# REST: paginated full-history fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_range(instrument_key: str,
                 from_date: str,
                 to_date: str) -> pd.DataFrame:
    """
    Splits date range into CHUNK_DAYS pages and stitches results.
    Prints a live progress line as chunks arrive.
    """
    start = datetime.strptime(from_date, "%Y-%m-%d").date()
    end   = datetime.strptime(to_date,   "%Y-%m-%d").date()

    chunks      = []
    chunk_start = start
    total_days  = max((end - start).days, 1)
    done_days   = 0

    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        df_chunk  = _fetch_chunk(instrument_key, str(chunk_start), str(chunk_end))

        if not df_chunk.empty:
            chunks.append(df_chunk)

        done_days += (chunk_end - chunk_start).days
        pct        = done_days / total_days * 100
        n          = len(df_chunk) if not df_chunk.empty else 0
        print(f"[upstox_data] {chunk_start} → {chunk_end}  "
              f"{n:5d} candles  ({pct:.0f}% done)", end="\r")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    print()   # newline after progress line

    if not chunks:
        raise ValueError(
            f"No candles returned for {instrument_key} "
            f"between {from_date} and {to_date}"
        )

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────────────

def load_data(ticker: str,
              start_date: str = None,
              end_date: str = None) -> pd.DataFrame:
    """
    Mirrors data.load_data().

    First run  : fetches full history (Jan 2022 → today), saves to cache.
    Later runs : loads from cache, then tops up only the missing days —
                 so retraining is nearly instant.

    Set USE_CACHE = False to force a full re-fetch.

    Returns raw OHLCV DataFrame (prices, not pct-change).
    """
    today     = datetime.now(timezone.utc).date()
    to_date   = end_date   or str(today)
    from_date = start_date or HISTORY_START

    # ── try cache first ───────────────────────────────────────────────────────
    cached = pd.DataFrame()
    if USE_CACHE:
        cached = _load_cache(ticker)

    if not cached.empty and _cache_is_fresh(cached):
        # cache is up to date — nothing to fetch
        df = cached
    elif not cached.empty:
        # cache exists but has a gap — top up from latest cached date
        latest_cached = str(cached.index.max().date() + timedelta(days=1))
        print(f"[upstox_data] topping up {ticker} from {latest_cached} → {to_date}")
        instrument_key = _ticker_to_key(ticker)
        new_data = _fetch_range(instrument_key, latest_cached, to_date)
        df = pd.concat([cached, new_data]).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if USE_CACHE:
            _save_cache(ticker, df)
    else:
        # no cache — fetch everything
        print(f"[upstox_data] fetching full history for {ticker} "
              f"({from_date} → {to_date}) — this takes ~2 min, cached after")
        instrument_key = _ticker_to_key(ticker)
        df = _fetch_range(instrument_key, from_date, to_date)
        if USE_CACHE:
            _save_cache(ticker, df)

    # market hours only — 09:15 to 15:30 IST
    df = df.between_time("09:15", "15:30")

    print(f"[upstox_data] {ticker}: {len(df):,} candles ready  "
          f"({df.index.min().date()} → {df.index.max().date()})")
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors data.transform_data() — pct-change, drop first NaN row."""
    return df.pct_change().dropna()


def build_windows(transformed: pd.DataFrame, window_size: int,
                  raw_data: pd.DataFrame = None):
    """
    Mirrors data.build_windows().
    Returns (X, y, prices) — same dtypes as the yfinance version.
    """
    X, y, prices = [], [], []
    close_col = transformed.columns.get_loc("close")

    for i in range(len(transformed) - window_size):
        window = transformed.iloc[i: i + window_size].values
        label  = 1 if transformed.iloc[i + window_size, close_col] > 0 else 0
        X.append(window)
        y.append(label)
        prices.append(
            float(raw_data.iloc[i + window_size]["close"])
            if raw_data is not None else 0.0
        )

    return (
        np.array(X,      dtype=np.float64),
        np.array(y,      dtype=np.int32),
        np.array(prices, dtype=np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket: live 5m bar stream via MarketDataStreamerV3
# ─────────────────────────────────────────────────────────────────────────────

class LiveCandleStream:
    """
    Subscribes to Upstox MarketDataStreamerV3 (full mode).
    Aggregates live ticks into completed 5m bars.
    Seeds the buffer from REST on startup so data is ready immediately.

    Usage
    -----
        stream = LiveCandleStream("RELIANCE", buffer_size=WINDOW_SIZE + 10)
        stream.start()
        window, price = stream.get_window(WINDOW_SIZE)
    """

    def __init__(self, ticker: str,
                 buffer_size: int = WINDOW_SIZE + 10,
                 bar_minutes: int = 5):
        self.instrument_key = _ticker_to_key(ticker)
        self.ticker         = ticker
        self.buffer_size    = buffer_size
        self.bar_minutes    = bar_minutes

        self._raw_buffer: list = []
        self._lock             = threading.Lock()
        self._streamer         = None
        self._running          = False

        self._bar_open   = None
        self._bar_high   = None
        self._bar_low    = None
        self._bar_last   = None
        self._bar_volume = 0.0
        self._bar_start  = None

        self._seed_from_rest()

    def _seed_from_rest(self):
        today     = datetime.now(timezone.utc).date()
        from_date = str(today - timedelta(days=5))
        to_date   = str(today)
        try:
            df = _fetch_range(self.instrument_key, from_date, to_date)
            df = df.between_time("09:15", "15:30")
            for _, row in df.iterrows():
                self._raw_buffer.append(row[FEATURES].values.astype(np.float64))
            self._raw_buffer = self._raw_buffer[-self.buffer_size:]
            print(f"[LiveCandleStream] seeded {len(self._raw_buffer)} bars for {self.ticker}")
        except Exception as e:
            print(f"[LiveCandleStream] seed failed ({e}) — relying on live data")

    def _on_message(self, message):
        try:
            feeds = message.get("feeds", {})
            feed  = feeds.get(self.instrument_key, {})
            ltpc  = feed.get("ff", {}).get("marketFF", {}).get("ltpc", {})
            ltp   = float(ltpc.get("ltp", 0))
            vol   = float(ltpc.get("ltq", 0))

            if ltp <= 0:
                return

            now       = datetime.now()
            bar_start = now.replace(
                minute=(now.minute // self.bar_minutes) * self.bar_minutes,
                second=0, microsecond=0
            )

            if self._bar_start is None:
                self._bar_start = bar_start

            if bar_start > self._bar_start:
                if self._bar_open is not None:
                    completed = np.array([
                        self._bar_open, self._bar_high, self._bar_low,
                        self._bar_last, self._bar_volume
                    ], dtype=np.float64)
                    with self._lock:
                        self._raw_buffer.append(completed)
                        if len(self._raw_buffer) > self.buffer_size + 5:
                            self._raw_buffer.pop(0)
                    print(f"[LiveCandleStream] bar closed  "
                          f"close={self._bar_last:.2f}  "
                          f"buffer={len(self._raw_buffer)}")
                self._bar_start  = bar_start
                self._bar_open   = ltp
                self._bar_high   = ltp
                self._bar_low    = ltp
                self._bar_last   = ltp
                self._bar_volume = vol
            else:
                if self._bar_open is None:
                    self._bar_open = ltp
                self._bar_high    = max(self._bar_high or ltp, ltp)
                self._bar_low     = min(self._bar_low  or ltp, ltp)
                self._bar_last    = ltp
                self._bar_volume += vol

        except Exception as e:
            print(f"[LiveCandleStream] message error: {e}")

    def _on_error(self, message):
        print(f"[LiveCandleStream] error: {message}")

    def _on_close(self, message):
        print("[LiveCandleStream] closed")
        if self._running:
            print("[LiveCandleStream] reconnecting in 5s …")
            time.sleep(5)
            self._connect()

    def _on_open(self, message):
        print(f"[LiveCandleStream] connected — {self.ticker}")

    def _connect(self):
        config     = _get_config()
        api_client = upstox_client.ApiClient(config)
        self._streamer = upstox_client.MarketDataStreamerV3(
            api_client, [self.instrument_key], "full"
        )
        self._streamer.on("message", self._on_message)
        self._streamer.on("error",   self._on_error)
        self._streamer.on("close",   self._on_close)
        self._streamer.on("open",    self._on_open)
        self._streamer.auto_reconnect(enable=True, interval=5, retryCount=10)
        self._streamer.connect()

    def start(self):
        self._running = True
        threading.Thread(target=self._connect, daemon=True).start()

    def stop(self):
        self._running = False
        if self._streamer:
            self._streamer.disconnect()

    def get_window(self, window_size: int):
        """
        Returns (window, price):
          window : (window_size, 5) float64 pct-change array
          price  : float — raw close of the newest completed bar
        Returns (None, None) if buffer not full yet.
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