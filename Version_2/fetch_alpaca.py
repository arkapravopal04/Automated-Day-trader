"""
Incremental Alpaca OHLCV bar fetcher (5-minute by default; see BAR_MINUTES).

The module downloads stock bars for ``TICKERS`` at the ``ALPACA_BAR_MINUTES``
cadence and stores them as per-ticker Parquet files. A non-default cadence
gets its own cache directory -- mixing cadences in one file is silent
corruption, see BAR_MINUTES. Existing caches are extended incrementally, while
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
    ALPACA_BAR_MINUTES=N     -> bar size in minutes (default 5). Anything
                                 other than 5 redirects the cache to
                                 <RAW_DIR>_Nmin.
    ALPACA_RTH_ONLY=1        -> drop pre/post-market bars at write time.
    ALPACA_CHUNK_DAYS=N      -> fetch each ticker as N-day windows, saving
                                 after each. Makes long runs resumable.
    ALPACA_FETCH_WORKERS=N   -> fetch N tickers concurrently.
    ALPACA_READ_TIMEOUT=S    -> per-request read timeout in seconds
                                 (default 120). The SDK sets none, and
                                 without one a dead socket hangs a worker
                                 permanently -- see REQUEST_TIMEOUT.
"""

import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import Adjustment
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

# Universe override. `TRADING_TICKERS` takes a comma-separated list and
# replaces TICKERS entirely; `TRADING_TICKERS_EXTRA` appends to it.
#
# Added for the delisting-inclusive universe (see scan_delisted.py). The
# hardcoded TICKERS list above is 100 mega-caps picked in 2026 and therefore
# contains only names that survived -- a bias that is mild for the intraday
# work and is a live alternative explanation for any positive OVERNIGHT
# result, because cross-sectional reversal buys yesterday's losers and in a
# survivor-only universe those are names already known to have recovered.
_ov = os.getenv("TRADING_TICKERS", "").strip()
if _ov:
    TICKERS = [t.strip().upper() for t in _ov.split(",") if t.strip()]
_extra = os.getenv("TRADING_TICKERS_EXTRA", "").strip()
if _extra:
    TICKERS = TICKERS + [t.strip().upper() for t in _extra.split(",")
                         if t.strip() and t.strip().upper() not in set(TICKERS)]

HISTORY_YEARS = int(os.getenv("ALPACA_HISTORY_YEARS", "6"))

# Split/dividend adjustment applied by the Alpaca API. Override with
# ALPACA_ADJUSTMENT=raw|split|dividend|all. Anything other than "all" will
# reintroduce unadjusted price jumps -- see the note in the request below.
ADJUSTMENT = Adjustment(os.getenv("ALPACA_ADJUSTMENT", "all").lower())
DEFAULT_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "sip")

# Alpaca's free tier serves full historical SIP but blocks the most recent
# ~15 minutes ("subscription does not permit querying recent SIP data").
# Verified 2026-08-19: SIP at -24h -> 200, SIP at -10min -> 403.
#
# This matters because the 403 is caught below and silently retried on IEX.
# Since end_date is "now", EVERY ticker's request would span the blocked
# window, 403, and fall back -- handing back IEX bars for all 100 tickers
# while the caller believes it has SIP. IEX volume is a small fraction of
# consolidated volume (77k vs 5.65M on one sampled AAPL bar), so that would
# understate volume ~70x, inflate participation, and overcharge sqrt-impact:
# exactly the Session 1 bug, minus the VOLUME_SCALE fudge that was
# compensating for it.
#
# So on SIP we simply never ask for the last SIP_LAG_MINUTES. Losing the
# most recent quarter-hour of a six-year history costs nothing for training.
SIP_LAG_MINUTES = int(os.getenv("ALPACA_SIP_LAG_MINUTES", "20"))

# ---------------------------------------------------------------------------
# Bar timeframe. The same endpoint serves 1-minute bars for the same price:
# five observations per five-minute decision instead of one (intrabar shape,
# where in the bar the volume landed, and the option to time entry inside the
# window rather than accepting open[t+1] as given).
#
# A 1-min cache MUST NOT land in the 5-min cache directory. This fetch is
# incremental and merges on timestamp, so two cadences in one file produce a
# single incoherent series with no error raised anywhere -- the identical
# failure mode the RAW/adjusted note in the request below describes. So a
# non-default timeframe is redirected to its own directory unless the caller
# has named one explicitly via TRADING_RAW_DIR.
BAR_MINUTES = int(os.getenv("ALPACA_BAR_MINUTES", "5"))
if BAR_MINUTES < 1:
    raise ValueError(f"ALPACA_BAR_MINUTES must be >= 1, got {BAR_MINUTES}")

if BAR_MINUTES != 5 and not os.environ.get("TRADING_RAW_DIR"):
    DATA_DIR = f"{DATA_DIR}_{BAR_MINUTES}min"
    os.makedirs(DATA_DIR, exist_ok=True)

# Regular US trading hours, minutes past midnight America/New_York: 09:30
# inclusive to 16:00 exclusive. preprocess.py applies exactly this filter and
# deliberately does it there, so the raw cache stays complete. At 1-minute
# cadence "complete" costs 2.3x the disk for bars nothing downstream reads:
# SIP returns ~184.6 bars/day against 78 RTH at 5-min, and the same ratio
# holds at 1-min. Filtering at write keeps a 6-year 100-name 1-min cache near
# 2 GB rather than 4.8 GB.
#
# This does NOT foreclose the overnight regime: an overnight gap is the last
# RTH close of day d to the first RTH open of day d+1, and both survive the
# filter. It forecloses only the pre/post-market bars themselves.
RTH_ONLY = os.getenv("ALPACA_RTH_ONLY", "0") == "1"
RTH_START_MIN = 9 * 60 + 30
RTH_END_MIN = 16 * 60

# A six-year 1-minute history is ~1.2M bars per ticker, which the SDK pages
# 10k at a time. Measured 2026-08-29: an uninterrupted 6-year request runs
# ~10 minutes, and a single mid-pagination ConnectionReset discards all of it
# because _fetch_bars_with_retry restarts the whole request. Chunking bounds
# that loss to one window and makes the run resumable -- each window is merged
# and written before the next is requested, so a killed process picks up from
# the last saved timestamp. 0 disables chunking (the historical behaviour,
# and what the 5-min path has always used).
CHUNK_DAYS = int(os.getenv("ALPACA_CHUNK_DAYS", "0"))

# Threads for the ticker loop. requests.Session is not thread-safe, so each
# worker builds its own client (see _fetch_all). Alpaca rate-limits per
# account rather than per connection, so this trades headroom for wall-clock;
# 4 measured comfortably inside the limit on a 200 req/min plan.
FETCH_WORKERS = int(os.getenv("ALPACA_FETCH_WORKERS", "1"))


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


# (connect, read) seconds applied to every HTTP call. The SDK calls
# `self._session.request(method, url, **opts)` and never puts a `timeout` in
# `opts`, so requests blocks FOREVER on a half-open socket -- one of the
# observed failures literally read "Read timed out. (read timeout=None)".
#
# That is not a slow request, it is a dead worker: no exception is ever raised,
# so _fetch_bars_with_retry never sees anything to retry and the thread parks
# permanently. Measured 2026-08-29: three tickers that had already fetched
# their full history sat wedged for three hours on a request that would never
# return, and the run could not exit. A read timeout converts that silent hang
# into an ordinary transient error, which the retry loop now handles.
#
# 120s is generous for one 10k-row page -- the slowest observed full page was
# well under 30s -- and the cost of it being too tight is a retry, while the
# cost of no timeout at all is an unbounded stall.
REQUEST_TIMEOUT = (
    float(os.getenv("ALPACA_CONNECT_TIMEOUT", "15")),
    float(os.getenv("ALPACA_READ_TIMEOUT", "120")),
)


def _create_client() -> Optional[StockHistoricalDataClient]:
    """Create an Alpaca historical-data client when credentials are available.

    The client's underlying `requests.Session` is wrapped so every call carries
    REQUEST_TIMEOUT -- see the note above it. Wrapped rather than configured
    because the SDK exposes no timeout parameter at any level.
    """
    if not API_KEY or not SECRET_KEY:
        return None
    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

    session = getattr(client, "_session", None)
    if session is None:
        print("[fetch] WARNING: client exposes no _session -- requests will run "
              "WITHOUT a timeout and a dead socket can hang a worker forever.")
        return client

    original_request = session.request

    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", REQUEST_TIMEOUT)
        return original_request(method, url, **kwargs)

    session.request = request_with_timeout
    return client


API_KEY, SECRET_KEY = get_alpaca_credentials()
client = _create_client()

if client is None:
    print(
        "Warning: Alpaca API credentials not found. "
        "Any ticker without existing cached data will be skipped; "
        "cached tickers will be used as-is without incremental updates."
    )


def _atomic_write_parquet(df: pd.DataFrame, out_path: str) -> None:
    """Write a parquet file atomically (temp file + os.replace) so a crash
    mid-write can never leave a corrupted cache -- same pattern
    preprocess.py's _atomic_write_parquet uses. A stray temp file on
    failure is harmless; a half-written out_path is not."""
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


def _fetch_bars_with_retry(client, request_params, retries: int = 3, base_delay: float = 1.0):
    """
    Retries TRANSIENT failures (network blips, 429 rate limits, 5xx) with
    exponential backoff so a single hiccup doesn't silently leave a
    permanent gap in the cache. Non-transient errors (bad symbol, auth)
    propagate immediately -- retrying those cannot help.
    """
    # WHAT COUNTS AS TRANSIENT, AND WHY THE OBVIOUS TEST IS WRONG.
    # This used to read `isinstance(exc, (ConnectionError, TimeoutError))`,
    # naming the BUILTINS. The Alpaca SDK talks through `requests`, and none of
    # the exceptions it actually raises on a network fault are subclasses of
    # either builtin -- `requests.exceptions.ConnectionError`,
    # `ChunkedEncodingError` and `ReadTimeout` all descend from RequestException
    # -> IOError, and `urllib3.exceptions.ProtocolError` from neither. Verified
    # 2026-08-29. So the retry loop re-raised on the FIRST network fault and
    # only ever retried HTTP 429/5xx, which arrive with a status_code.
    #
    # It went unnoticed while each ticker was one request on a 5-minute
    # timeframe. On the 6-year 1-minute fetch -- 13 windows per ticker, 12
    # concurrent workers, hours of wall clock -- it cost 9 windows across 9
    # tickers to RemoteDisconnected/ConnectionAborted/ReadTimeout, each one an
    # interior hole that the incremental resume cannot see. See verify_cache.py.
    try:
        from requests.exceptions import RequestException
    except ImportError:
        RequestException = ()
    try:
        from urllib3.exceptions import HTTPError as _Urllib3Error
    except ImportError:
        _Urllib3Error = ()
    transient_types = (ConnectionError, TimeoutError, OSError, RequestException, _Urllib3Error)

    last_exc = None
    for attempt in range(retries):
        try:
            return client.get_stock_bars(request_params)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            # An auth failure or a bad symbol arrives as an APIError with a 4xx
            # status; retrying those cannot help and must still propagate fast.
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            transient = (
                isinstance(exc, transient_types)
                or (status is not None and (status == 429 or status >= 500))
            )
            if not transient:
                raise
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


def _apply_rth_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pre/post-market bars when ALPACA_RTH_ONLY=1.

    Mirrors preprocess.py's filter term for term (09:30 inclusive to 16:00
    exclusive, America/New_York) so a pre-filtered cache and a complete one
    produce the identical feature panel -- preprocess re-applies the same
    mask, and on a filtered cache it is simply a no-op.
    """
    if not RTH_ONLY or df.empty:
        return df
    ny_index = df.index.tz_convert("America/New_York")
    minutes_of_day = ny_index.hour * 60 + ny_index.minute
    mask = (minutes_of_day >= RTH_START_MIN) & (minutes_of_day < RTH_END_MIN)
    return df[mask]


def _chunk_ranges(start: datetime, end: datetime):
    """Split ``[start, end)`` into CHUNK_DAYS-wide windows (one window if
    chunking is disabled). See CHUNK_DAYS for why this exists."""
    if CHUNK_DAYS <= 0:
        yield (start, end)
        return
    cursor = start
    step = timedelta(days=CHUNK_DAYS)
    while cursor < end:
        stop = min(cursor + step, end)
        yield (cursor, stop)
        cursor = stop


# Windows that failed after their retries, collected across threads. A gap in
# the middle of a cache is invisible to the incremental resume (which reads
# only the newest timestamp), so it has to be surfaced at the end of the run
# rather than left in the scrollback.
_FAILED_WINDOWS: list = []


def fetch_incremental_data(ticker: str, end_date: datetime, api_client=None) -> None:
    """Fetch and cache ``BAR_MINUTES``-minute bars for one ticker up to ``end_date``.

    Existing Parquet data is extended from one minute after its latest
    timestamp. A missing cache is initialized with ``HISTORY_YEARS`` of
    historical data. New records are merged, duplicate timestamps are removed
    in favor of the newest record, and the result is sorted chronologically.

    When ``CHUNK_DAYS`` is set the span is fetched as a sequence of windows,
    each merged and written before the next is requested. That makes a long
    run resumable: a killed or crashed process leaves a valid cache, and the
    next invocation restarts from the last saved timestamp rather than from
    the beginning.

    Args:
        ticker: Stock symbol to fetch.
        end_date: Timezone-aware upper bound for the Alpaca request.
        api_client: Client to use. Defaults to the module-level singleton.
            Worker threads MUST pass their own -- ``requests.Session`` is not
            thread-safe, so sharing one client across threads interleaves
            responses under concurrency.

    Note:
        If Alpaca credentials are unavailable, a ticker with NO existing
        cache is skipped with a warning (nothing to fall back on). A ticker
        WITH existing cache (e.g. bootstrapped from a Kaggle input dataset --
        see paths.py) is also skipped with a warning rather than raising --
        this lets a credential-less, cache-only Kaggle session complete
        using stale-but-present data instead of crashing the whole pipeline.
    """
    if api_client is None:
        api_client = client

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    tf_label = f"{BAR_MINUTES}-min"

    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        latest_timestamp = existing_df.index.max()
        print(
            f"[{ticker}] cached {len(existing_df)} {tf_label} candles "
            f"(last: {latest_timestamp})"
        )

        if pd.isna(latest_timestamp):
            start_date = end_date - timedelta(days=365 * HISTORY_YEARS)
        else:
            start_date = latest_timestamp + timedelta(minutes=1)
    else:
        existing_df = None
        latest_timestamp = None
        start_date = end_date - timedelta(days=365 * HISTORY_YEARS)
        print(
            f"[{ticker}] no cache -- fetching {HISTORY_YEARS}y of {tf_label} "
            f"candles since {start_date.date()}"
        )

    if start_date >= end_date:
        print(f"[{ticker}] already up to date.")
        return

    if api_client is None:
        if existing_df is not None:
            print(
                f"[{ticker}] No Alpaca credentials -- using existing cached data "
                f"as-is (last timestamp: {latest_timestamp}, not refreshed)."
            )
        else:
            print(f"[{ticker}] No Alpaca credentials and no existing cache -- skipping this ticker.")
        return

    feed = os.getenv("ALPACA_DATA_FEED", DEFAULT_DATA_FEED)

    # Hold the request end off the SIP embargo window -- see SIP_LAG_MINUTES.
    if feed == "sip":
        capped_end = min(end_date, datetime.now(timezone.utc) - timedelta(minutes=SIP_LAG_MINUTES))
        if capped_end <= start_date:
            print(
                f"[{ticker}] Already current to within the {SIP_LAG_MINUTES}-min SIP "
                "embargo window -- nothing to fetch."
            )
            return
        end_date = capped_end

    combined_df = existing_df
    total_new = 0

    for chunk_start, chunk_end in _chunk_ranges(start_date, end_date):
        # Alpaca defaults to Adjustment.RAW -- unadjusted for splits. Raw closes
        # feed BOTH the feature pipeline (one ~-90% log return per split; observed
        # z-scores up to 770) and position marking in load_aligned_close_prices()
        # (a stream holding GE through its 1-for-8 reverse split books an instant
        # +700%). ALL covers splits and dividends; SPLIT alone leaves ex-div gaps.
        # NOTE: the fetch is INCREMENTAL. Any RAW parquet already on disk would be
        # concatenated with adjusted bars into one incoherent series, so the cache
        # must be wiped before the first adjusted fetch, not extended. The same
        # argument applies to BAR_MINUTES, which is why a non-default timeframe
        # gets its own directory -- see the DATA_DIR redirect above.
        request_params = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
            start=chunk_start,
            end=chunk_end,
            feed=feed,
            adjustment=ADJUSTMENT,
        )

        try:
            try:
                bars = _fetch_bars_with_retry(api_client, request_params)
            except Exception as exc:
                if feed == "sip" and "subscription does not permit" in str(exc):
                    print(
                        f"[{ticker}] *** SIP REJECTED, FALLING BACK TO IEX *** -- this "
                        f"ticker's volume will be ~70x lower than SIP tickers in the same cache. "
                        f"A mixed-feed cache silently corrupts participation and impact costs; "
                        f"re-fetch this ticker on one feed before training. Error: {exc}"
                    )
                    request_params.feed = "iex"
                    bars = _fetch_bars_with_retry(api_client, request_params)
                else:
                    raise

            if bars.df.empty:
                continue

            new_df = _apply_rth_filter(bars.df.loc[ticker])
            if new_df.empty:
                continue
            total_new += len(new_df)

            if combined_df is not None:
                combined_df = pd.concat([combined_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
            else:
                combined_df = new_df

            combined_df = combined_df.sort_index()
            # Atomic write -- a crash mid-`to_parquet` must never corrupt the cache.
            # Written per chunk rather than once at the end, so the run resumes.
            _atomic_write_parquet(combined_df, file_path)

        except Exception as exc:
            # Keep going: a failed window leaves a gap the next incremental run
            # cannot see (resume reads only the LAST timestamp), so failures are
            # recorded and reported as a block at the end of the run.
            print(
                f"[{ticker}] Error fetching {tf_label} candles for "
                f"{chunk_start.date()}..{chunk_end.date()}: {exc}"
            )
            _FAILED_WINDOWS.append((ticker, chunk_start, chunk_end, str(exc)))

    if total_new:
        print(
            f"[{ticker}] +{total_new} new {tf_label} candles -> "
            f"{len(combined_df)} total in {os.path.basename(file_path)}"
        )
    else:
        print(f"[{ticker}] no new bars returned.")


def _fetch_all(tickers, current_time) -> None:
    """Fetch ``tickers``, with FETCH_WORKERS threads and one client per thread."""
    if FETCH_WORKERS <= 1:
        for ticker in tickers:
            fetch_incremental_data(ticker, current_time)
        return

    import threading
    from concurrent.futures import ThreadPoolExecutor

    local = threading.local()

    def worker(ticker: str) -> None:
        if not hasattr(local, "api_client"):
            local.api_client = _create_client()
        fetch_incremental_data(ticker, current_time, api_client=local.api_client)

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        list(pool.map(worker, tickers))


if __name__ == "__main__":
    current_time = datetime.now(timezone.utc)

    print("=" * 60)
    print(f"STARTING {BAR_MINUTES}-MIN CANDLE FETCH PIPELINE FOR {len(TICKERS)} TICKERS")
    print(
        f"Environment: {'Kaggle' if is_kaggle() else 'Local'} | "
        f"Cache dir: {DATA_DIR}"
    )
    print(
        f"feed={os.getenv('ALPACA_DATA_FEED', DEFAULT_DATA_FEED)} "
        f"adjustment={ADJUSTMENT.value} history={HISTORY_YEARS}y "
        f"rth_only={int(RTH_ONLY)} chunk_days={CHUNK_DAYS} workers={FETCH_WORKERS}"
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
                    f"  - {len(missing)} ticker(s) have no cache and TRADING_SKIP_MISSING=1: "
                    f"{missing} -- leaving them unfetched."
                )
            else:
                print(f"  - {len(missing)} ticker(s) have no cache and will still be fetched in full: {missing}")
                _fetch_all(missing, current_time)
        else:
            print("  - Every configured ticker already has cached data. Nothing to do.")
    else:
        _fetch_all(TICKERS, current_time)

    if _FAILED_WINDOWS:
        print("=" * 60)
        print(f"*** {len(_FAILED_WINDOWS)} WINDOW(S) FAILED -- THESE ARE HOLES IN THE CACHE ***")
        print("An incremental re-run will NOT repair them: resume reads the newest")
        print("timestamp only, so an interior gap stays a gap. Delete the affected")
        print("ticker parquet(s) and re-fetch, or fetch those windows explicitly.")
        for tkr, cs, ce, err in _FAILED_WINDOWS:
            print(f"  {tkr} {cs.date()}..{ce.date()}: {err[:160]}")
        print("=" * 60)
        sys.exit(1)
