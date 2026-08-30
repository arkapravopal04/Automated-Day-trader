"""P3 bullet 2: real NBBO and trade prints, aggregated to the 5-minute grid.

THE QUESTION THIS EXISTS TO ANSWER
----------------------------------
The brief says trades and quotes are "where genuine five-minute alpha lives".
The panel already carries CRUDE PROXIES for the same quantities, built from
1-minute OHLCV by intrabar.py:

    ib_flow_pres   sign-weighted volume        -> proxy for signed order flow
    ib_clv_pres    close-location value        -> proxy for buy/sell pressure
    ib_vol_center  where in the bar the volume landed
    ib_tsize       volume / trade_count        -> proxy for trade size

Those proxies produced NO establishable lift: ALPHA/TURN 0.136-0.148 against a
0.5-0.6 target, t 0.57-0.73, with the inner-holdout IC pointing the other way.
So the specific question is NOT "do NBBO features predict returns" -- it is:

    DO REAL NBBO AND LEE-READY FLOW BEAT THEIR OWN LOSSY PROXIES?

If real quote data does not clearly beat a sign-weighted-volume proxy on 20
names, it will not on 100, and P3 closes negative having tested its own central
claim rather than having run out of patience.

WHY THIS IS A PROBE AND NOT A PANEL
-----------------------------------
Measured 2026-08-30: AAPL's 09:30-10:30 hour alone is 294,700 quotes and
328,089 trades. Quote volume runs 86x between AAPL and GD, so cost is dominated
by a handful of names. The universe here is 20 names STRATIFIED across the
liquidity range (logs/p3/tq_universe.json), not the top 20 -- a top-20 sample
would cost roughly 10x more and would answer the question only for the most
liquid corner of a universe the book trades all of.

Raw ticks are NEVER STORED. A name-day is fetched, reduced to 78 rows of
features, and discarded. Storing the raw would be ~0.8 TB of prints and ~8.5 TB
of NBBO for the full universe; the features are a few MB.

FEATURES
--------
From quotes, TIME-WEIGHTED (quotes arrive irregularly and in bursts; a simple
mean would weight a quote that stood for 2ms the same as one that stood for
4 seconds):

    nb_spread_bps  the actual quoted spread -- what the tick-grid proxy and the
                   half-tick cost floor have been standing in for
    nb_imb         (bid_size - ask_size) / (bid_size + ask_size)
    nb_micro_dev   (microprice - mid) / mid in bps, microprice being
                   (bid*ask_size + ask*bid_size) / (bid_size + ask_size):
                   where the next trade is likelier to print

From trades:

    tq_flow        Lee-Ready signed volume / total volume -- the real version
                   of ib_flow_pres
    tq_flow_n      Lee-Ready signed COUNT / trade count -- unweighted by size,
                   so one block cannot carry a bin
    tq_tsize_z     log mean trade size, the real version of ib_tsize
    tq_large_frac  share of volume printing above 4x the name's own median
                   trade size -- institutional participation

LEE-READY. A trade above the prevailing midpoint is a buy, below is a sell, and
one AT the midpoint falls back to the tick test against the previous trade
price. The quote must be PREVAILING -- merge_asof direction="backward" --
because using the quote that FOLLOWED the trade would let the trade's own
impact classify it, manufacturing exactly the signal being measured.

    python tickquote.py --start 2025-01-02 --end 2025-06-30 --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest, StockTradesRequest

NB_COLUMNS = ["nb_spread_bps", "nb_imb", "nb_micro_dev"]
TQ_COLUMNS = ["tq_flow", "tq_flow_n", "tq_tsize_z", "tq_large_frac"]
TICKQUOTE_COLUMNS = NB_COLUMNS + TQ_COLUMNS

TZ = "America/New_York"
EPS = 1e-12


_local = threading.local()


def _client():
    """One client per THREAD, reused across name-days.

    A fresh client per unit of work would build a new requests.Session for each
    of ~2,500 name-days, throwing away connection pooling and TLS reuse on a
    workload that is entirely dominated by paginated HTTPS round trips. Per
    thread rather than global because the SDK's session is not documented as
    thread-safe.
    """
    c = getattr(_local, "client", None)
    if c is None:
        c = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"),
                                      os.getenv("ALPACA_SECRET_KEY"))
        _local.client = c
    return c


def _rth(df):
    """09:30 to 15:59:59.999, matching ALPACA_RTH_ONLY and preprocess.py."""
    if df is None or len(df) == 0:
        return df
    idx = df.index.tz_convert(TZ)
    mins = idx.hour * 60 + idx.minute
    return df[(mins >= 570) & (mins < 960)]


def quote_features(q):
    """Time-weighted NBBO features per 5-minute bin, indexed by the left edge."""
    empty = pd.DataFrame(columns=NB_COLUMNS)
    if q is None or len(q) == 0:
        return empty
    q = q[(q["bid_price"] > 0) & (q["ask_price"] > 0)
          & (q["ask_price"] >= q["bid_price"])].sort_index()
    if len(q) == 0:
        return empty

    bp = q["bid_price"].to_numpy(float)
    ap = q["ask_price"].to_numpy(float)
    bs = q["bid_size"].to_numpy(float)
    asz = q["ask_size"].to_numpy(float)
    mid = 0.5 * (bp + ap)
    safe_mid = np.where(mid > 0, mid, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        spread_bps = 1e4 * (ap - bp) / safe_mid
        tot = bs + asz
        safe_tot = np.where(tot > 0, tot, np.nan)
        imb = (bs - asz) / safe_tot
        micro = np.where(tot > 0, (bp * asz + ap * bs) / safe_tot, mid)
        micro_dev = 1e4 * (micro - mid) / safe_mid

    # DWELL TIME, not row count. NBBO updates are extremely bursty, and a quote
    # that stood for four seconds describes the book far better than one
    # replaced two milliseconds later.
    #
    # Dwell runs to the next quote OR THE END OF THE BIN, whichever comes
    # first, and both halves of that matter. Without the bin clamp a quote at
    # 09:34:59 would carry its dwell into the 09:35 bin while being counted in
    # the 09:30 one. Without the run-to-bin-end the LAST quote of a bin would
    # get an arbitrary sliver of weight, when in fact it is the quote that
    # stood for most of the interval -- on a quiet name that is nearly the
    # whole bin, and it is the observation that most describes the book.
    ts = q.index.view("int64").astype(np.int64)
    bin_start = q.index.floor("5min")
    bin_end = (bin_start + pd.Timedelta(minutes=5)).view("int64").astype(np.int64)
    nxt = np.empty_like(ts)
    nxt[:-1] = ts[1:]
    nxt[-1] = np.iinfo(np.int64).max
    dt = (np.minimum(nxt, bin_end) - ts).astype(float)
    dt = np.clip(dt, 0.0, 5 * 60e9)
    g = pd.Series(bin_start, index=q.index)

    def tw(x):
        num = pd.Series(np.where(np.isfinite(x), x, 0.0) * dt, index=q.index).groupby(g).sum()
        den = pd.Series(np.where(np.isfinite(x), dt, 0.0), index=q.index).groupby(g).sum()
        return num / den.replace(0.0, np.nan)

    out = pd.DataFrame({"nb_spread_bps": tw(spread_bps),
                        "nb_imb": tw(imb),
                        "nb_micro_dev": tw(micro_dev)})
    out.index.name = None
    return out


def lee_ready_sign(t, q):
    """+1 buy, -1 sell, per trade. Prevailing-quote rule with a tick-test fallback."""
    p = t["price"].to_numpy(float)
    sign = np.zeros(len(t))
    if q is not None and len(q):
        qq = q[(q["bid_price"] > 0) & (q["ask_price"] > 0)].sort_index()
        if len(qq):
            mid = (0.5 * (qq["bid_price"] + qq["ask_price"])).rename("mid").to_frame()
            m = pd.merge_asof(t[["price"]], mid, left_index=True, right_index=True,
                              direction="backward")
            md = m["mid"].to_numpy(float)
            sign = np.where(p > md, 1.0, np.where(p < md, -1.0, 0.0))
            sign = np.where(np.isfinite(md), sign, 0.0)

    # Tick test for prints AT the midpoint, and for any trade with no prevailing
    # quote. Zero ticks carry the last non-zero direction forward, which is the
    # rule as published -- a flat sequence is not information.
    dp = np.diff(p, prepend=p[0])
    tick = np.sign(dp)
    nz = tick != 0
    if nz.any():
        idx = np.where(nz, np.arange(len(tick)), 0)
        np.maximum.accumulate(idx, out=idx)
        tick = tick[idx]
    return np.where(sign != 0, sign, tick)


def trade_features(t, q):
    """Signed flow and trade-size shape per 5-minute bin."""
    empty = pd.DataFrame(columns=TQ_COLUMNS)
    if t is None or len(t) == 0:
        return empty
    t = t[(t["price"] > 0) & (t["size"] > 0)].sort_index()
    if len(t) == 0:
        return empty

    sign = lee_ready_sign(t, q)
    v = t["size"].to_numpy(float)
    g = pd.Series(t.index.floor("5min"), index=t.index)

    sv = pd.Series(sign * v, index=t.index).groupby(g).sum()
    av = pd.Series(v, index=t.index).groupby(g).sum()
    sn = pd.Series(sign, index=t.index).groupby(g).sum()
    cn = pd.Series(1.0, index=t.index).groupby(g).sum()

    # "Large" is relative to the NAME'S OWN median print for the day. An
    # absolute share threshold would call every SPY print small and every
    # BLK print large, which measures price, not participation.
    med = float(np.median(v))
    big = pd.Series(np.where(v > 4.0 * med, v, 0.0), index=t.index).groupby(g).sum()
    msize = av / cn.replace(0.0, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "tq_flow": sv / av.replace(0.0, np.nan),
            "tq_flow_n": sn / cn.replace(0.0, np.nan),
            "tq_tsize_z": np.log(msize.replace(0.0, np.nan)),
            "tq_large_frac": big / av.replace(0.0, np.nan),
        })
    out.index.name = None
    return out


def fetch_day(client, sym, day):
    """One name-day -> at most 78 rows of features. Raw ticks die here."""
    start = pd.Timestamp(f"{day} 09:30", tz=TZ).tz_convert("UTC").to_pydatetime()
    end = pd.Timestamp(f"{day} 16:00", tz=TZ).tz_convert("UTC").to_pydatetime()

    q = client.get_stock_quotes(StockQuotesRequest(
        symbol_or_symbols=sym, start=start, end=end, feed="sip")).df
    t = client.get_stock_trades(StockTradesRequest(
        symbol_or_symbols=sym, start=start, end=end, feed="sip")).df
    for d in (q, t):
        if d is not None and len(d) and isinstance(d.index, pd.MultiIndex):
            d.index = d.index.droplevel(0)
    q, t = _rth(q), _rth(t)

    feats = quote_features(q).join(trade_features(t, q), how="outer")
    n_q = 0 if q is None else len(q)
    n_t = 0 if t is None else len(t)
    return feats, n_q, n_t


def _atomic_write_parquet(df, out_path):
    """Temp file + os.replace, matching intrabar.py and fetch_alpaca.py."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet",
                               dir=os.path.dirname(out_path))
    os.close(fd)
    try:
        df.to_parquet(tmp, engine="pyarrow")
        os.replace(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def part_path(out_dir, sym, day):
    return os.path.join(out_dir, "parts", sym, f"{day}.parquet")


def fetch_one(sym, day, out_dir, log):
    """One (symbol, day) unit of work, written to its own part file.

    THE UNIT OF WORK IS THE NAME-DAY, NOT THE NAME. Measured cost per session:
    MO 37s, JPM 45s, SPY 752s -- SPY alone is 5.85M quotes in one session, 17x
    JPM. Parallelising per SYMBOL would make SPY a 26-hour critical path while
    other workers sat idle; per name-day the same total work finishes in about
    a twelfth of that.

    One part file per name-day also makes the probe resumable for free: an
    interrupted run skips what exists rather than restarting, and no two
    workers ever touch the same file, so there is no lock and no partial merge.
    """
    p = part_path(out_dir, sym, day)
    if os.path.exists(p):
        return 0, 0, 0
    client = _client()

    # BACKOFF SIZED TO THE ACTUAL FAULT. The first version retried after 2s and
    # 4s, three attempts. Against "too many requests" that is not a retry, it is
    # three more requests into a limit that is already saturated -- measured: it
    # lost a SPY name-day on the first pass. This tier allows ~200 requests per
    # minute and a single heavy session is hundreds of paginated requests, so a
    # rate-limit fault needs to wait on the order of the limit's own window.
    # Jitter prevents ten workers from retrying in lockstep and re-colliding.
    delays = [5.0, 20.0, 60.0, 150.0, 300.0]
    for attempt in range(len(delays) + 1):
        try:
            f, nq, nt = fetch_day(client, sym, day)
            # An empty frame is still written, so a genuinely quiet day is not
            # refetched forever on every resume.
            _atomic_write_parquet(f if len(f) else pd.DataFrame(columns=TICKQUOTE_COLUMNS), p)
            return len(f), nq, nt
        except Exception as exc:
            msg = str(exc).lower()
            throttled = "too many requests" in msg or "429" in msg
            if attempt >= len(delays):
                log(f"[{sym} {day}] FAILED after {attempt + 1} tries: "
                    f"{type(exc).__name__} {exc}")
                return 0, 0, 0
            d = delays[attempt] * (2.0 if throttled else 1.0)
            time.sleep(d * (0.75 + 0.5 * random.random()))
    return 0, 0, 0


def consolidate(out_dir, syms, log):
    """Merge each symbol's part files into one parquet on the 5-minute grid."""
    total = 0
    for sym in syms:
        d = os.path.join(out_dir, "parts", sym)
        if not os.path.isdir(d):
            continue
        frames = []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".parquet"):
                continue
            df = pd.read_parquet(os.path.join(d, fn))
            if len(df):
                frames.append(df)
        if not frames:
            continue
        got = pd.concat(frames).sort_index()
        got = got[~got.index.duplicated(keep="last")]
        _atomic_write_parquet(got, os.path.join(out_dir, f"{sym}.parquet"))
        total += len(got)
        log(f"[{sym}] {len(got):,} bins -> {out_dir}/{sym}.parquet")
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="logs/p3/tq_universe.json")
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--out", default="data/tickquote")
    ap.add_argument("--workers", type=int, default=6,
                    help="More workers cannot beat the API rate limit -- ~200 "
                         "requests/min, and one heavy session is hundreds of "
                         "paginated requests. Past saturation extra workers "
                         "only convert throughput into 429s.")
    ap.add_argument("--tickers", nargs="*", default=None)
    args = ap.parse_args(argv)

    syms = args.tickers or json.load(open(args.universe))
    cal = pd.bdate_range(args.start, args.end)
    days = [d.strftime("%Y-%m-%d") for d in cal]
    os.makedirs(args.out, exist_ok=True)

    print(f"[tq] {len(syms)} symbols x {len(days)} business days -> {args.out}")
    print(f"[tq] {args.workers} workers; raw ticks are discarded after aggregation")

    # DAY-MAJOR, NOT SYMBOL-MAJOR. This job runs for many hours and will be
    # stopped early -- by choice, a rate limit, or a laptop lid. What it has
    # produced at the moment it stops must be USABLE, and the thing being built
    # is a CROSS-SECTION.
    #
    # Symbol-major ordering fails that badly: measured on the first pass, XLP
    # reached 128/128 sessions while AMZN sat at 13, so stopping there would
    # have bought complete histories for a few names and nothing for the rest --
    # zero usable cross-sectional bars. Day-major fills every name for day 1,
    # then day 2, so stopping at any point yields all 20 names over a
    # contiguous, shorter window, which is exactly the dataset the IC
    # comparison needs.
    #
    # Within a day the heaviest names go first, so one slow name cannot become
    # the tail of that day's batch.
    order = {s: i for i, s in enumerate(syms)}
    dpos = {d: i for i, d in enumerate(days)}
    work = [(s, d) for d in days for s in syms]
    work.sort(key=lambda sd: (dpos[sd[1]], order[sd[0]]))

    todo = [(s, d) for (s, d) in work if not os.path.exists(part_path(args.out, s, d))]
    print(f"[tq] {len(work) - len(todo):,} name-days already cached, {len(todo):,} to fetch")

    t0 = time.time()
    done = nq_tot = nt_tot = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, s, d, args.out, print): (s, d) for (s, d) in todo}
        for fu in as_completed(futs):
            s, d = futs[fu]
            try:
                _, nq, nt = fu.result()
                nq_tot += nq
                nt_tot += nt
            except Exception as exc:
                print(f"[{s} {d}] WORKER FAILED: {type(exc).__name__} {exc}")
            done += 1
            if done % 25 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(todo) - done) / max(rate, 1e-9) / 60.0
                print(f"[tq] {done:,}/{len(todo):,} name-days  {el/60:.1f} min elapsed  "
                      f"ETA {eta:.0f} min  ({nq_tot/1e6:.1f}M quotes, {nt_tot/1e6:.1f}M trades)")

    print(f"[tq] fetch complete in {(time.time() - t0) / 60:.1f} min; consolidating")
    total = consolidate(args.out, syms, print)
    print(f"[tq] {total:,} feature bins across {len(syms)} symbols")


if __name__ == "__main__":
    main()
