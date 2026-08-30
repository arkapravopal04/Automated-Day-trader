"""
Alpha lab -- the gate that belongs in front of every training run.

Three sessions of this project have answered "is there any predictive signal?"
by spending a multi-hour 151-rollout PPO run and reading the equity curve. That
is the most expensive possible way to ask the question, and it conflates four
things at once: whether a signal exists, whether the cost model permits trading
it, whether PPO can find it, and whether the execution layer is honest.

This script answers only the first two, supervised, on data already on disk, in
minutes:

    for each (horizon x regime x signal):
        fit on the TRAIN split, score on VAL, and convert the result into an
        estimated annual Sharpe net of the env's own friction model.

Nothing here reads the TEST split.

Why the headline number is Sharpe, not "x break-even"
-----------------------------------------------------
Break-even IC falls with horizon (you pay the round trip once however long you
hold), so ranking cells by ic/breakeven makes weekly holding look spectacular
and hides that it only gives you ~50 bets a year. Net annual Sharpe prices both
halves:

    edge_bps  = ic * SELECTIVITY_K * sigma_h
    net_bps   = edge_bps - round_trip_cost
    sharpe    = (net_bps / sigma_h) * sqrt(bets_per_year * SELECTIVITY_FRACTION)

SELECTIVITY_K and SELECTIVITY_FRACTION are one assumption, not two: a trader
who only takes the top decile gets 1.75x the edge AND one tenth of the bets.
Taking the first without the second overstated every cell here by 3.16x.

`bets_per_year` respects the fact that a position held h bars cannot be
re-entered every bar. This is per name and assumes no diversification -- a
cross-sectional book of N weakly-correlated names multiplies it by up to
sqrt(N), which is exactly why the cross-sectional rows matter.

Three traps this script is built to avoid
-----------------------------------------
1. SIGN PICKED ON VAL. A univariate signal's sign is taken from TRAIN and
   applied to val. Reporting |IC| on val quietly grants one bit of lookahead
   per cell, and across a few hundred cells that alone manufactures winners.

2. OVERLAPPING RETURNS. A 1-week forward return shares 389 of its 390 bars with
   the next bar's. Clustering t-stats by calendar day does not fix that, so
   clusters are BLOCKS of ceil(h / bars_per_day) + 1 days.

3. MULTIPLE TESTING. Several hundred cells are scanned, so |t| > 2 somewhere is
   guaranteed by chance. Benjamini-Hochberg at q = 0.10 runs across every cell
   tested, and only BH-surviving cells can pass the gate.

A cell that clears all three and still beats the always-long benchmarks is a
real finding. Anything else is the scan finding itself.

Usage
-----
    python eval/alpha_lab.py                     # full universe, all cells
    python eval/alpha_lab.py --tickers 40        # lighter, for a laptop
    python eval/alpha_lab.py --json out.json     # machine-readable result
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import PROCESSED_DIR, RAW_DIR, TRAIN_FRAC, VAL_FRAC  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


BARS_PER_DAY = 78
TRADING_DAYS = 252

# E[x | x in the top decile] / sigma for a standard normal = phi(z_0.9) / 0.1.
# The policy need not trade every bar; this is the concentration a top-decile
# selective trader gets on the same signal.
SELECTIVITY_K = 1.75

# The fraction of opportunities that same top-decile trader actually takes.
# SELECTIVITY_K is the edge concentration selectivity buys; this is what it
# costs in bet count. They are two halves of one assumption and net_sharpe()
# must apply both -- applying only the first overstated every cell in this
# gate by sqrt(1/0.10) = 3.16x. Change them together or not at all.
SELECTIVITY_FRACTION = 0.10

# Order participation assumed for the impact term. NOT zero: at zero the
# impact term vanishes and the round trip prices at 3.64 bps, while the two
# most recent completed runs actually paid a fitted 4.20 bps + $0.0048/share,
# i.e. 9.0 bps round trip at the median price. 0.00032 is the participation
# that reconciles the two, so the gate screens against the friction the system
# demonstrably pays rather than a frictionless ideal. Running at zero made
# every break-even IC in the opportunity table 2.6x too easy.
DEFAULT_PARTICIPATION = 0.00032

# Gate thresholds. A cell must clear ALL of them.
MIN_SHARPE = 0.50          # net of cost, per name, no diversification credit
MIN_ABS_T = 2.0            # block-clustered
BH_Q = 0.10                # Benjamini-Hochberg false-discovery rate

HORIZONS = {
    "5min": 1,
    "30min": 6,
    "1hr": 12,
    "1day": BARS_PER_DAY,
    "1week": 5 * BARS_PER_DAY,
}

# Regimes are ENTRY bars, indexed from the session open (bar 0 == 09:30).
# Measured on the last run's marking price path, median |move| by regime was
# overnight 75.8 bps, open hour 18.6, close ramp 11.9, midday 9.6 -- a 7.7x
# spread that the uniform-across-the-day policy spent its cost budget ignoring.
REGIMES = {
    "all": list(range(BARS_PER_DAY - 1)),
    "open_hour": list(range(0, 12)),
    "midday": list(range(12, 72)),
    # Bar 77 is excluded everywhere below: with flatten_at_session_close the
    # env force-closes there and refuses to open, so no entry exists on it.
    "close_ramp": list(range(74, BARS_PER_DAY - 1)),
}

# The overnight bar is not a regime here at all: its exit IS the session close,
# so a session-capped forward return is empty by construction. It appears once
# in the benchmark block below, uncapped and labelled not tradable, so the size
# of what flatten_at_session_close declines stays on the page.


# ---------------------------------------------------------------------------
# Cost model -- derived from the env, not hardcoded
# ---------------------------------------------------------------------------

def build_cost_model(participation: float = 0.0):
    """Return (fn(price) -> bps per side, description).

    Mirrors ExecutionSimulator._compute_fill_price term by term so the gate
    moves when friction moves. The two 1/price terms are what made cheap names
    5-6x more expensive per trade than expensive ones in the last run
    (spearman(price, cost per trade) = -0.552, p = 2.7e-09).
    """
    try:
        from training.config import EnvConfig

        cfg = EnvConfig()
        spread_bps = float(cfg.spread_bps)
        commission_bps = float(cfg.commission_bps)
        impact_coef = float(cfg.impact_coef)
        platform_fee = float(cfg.platform_fee_per_trade)
        tick = float(getattr(cfg, "tick_size", 0.01))
        source = "training.config.EnvConfig"
    except Exception as exc:  # pragma: no cover
        spread_bps, commission_bps, impact_coef = 1.0, 0.5, 0.015
        platform_fee, tick = 0.0, 0.01
        source = f"fallback defaults ({type(exc).__name__})"

    impact_bps = impact_coef * math.sqrt(max(participation, 0.0)) * 1e4

    def cost_bps(price):
        price = np.asarray(price, dtype=np.float64)
        half_tick_bps = 1e4 * (tick / 2.0) / price
        # Half-spread floored at half a tick: a 1 bps proportional spread on a
        # $10 stock is unphysical when one tick is already 10 bps.
        half_spread = np.maximum(spread_bps, half_tick_bps)
        # snap_to_tick_adverse ceils buys and floors sells, so a fill gives up
        # a further half tick in expectation.
        return half_spread + commission_bps + impact_bps + half_tick_bps

    desc = (
        f"spread={spread_bps} bps (floored at half a {tick:.2f} tick), "
        f"commission={commission_bps} bps, impact={impact_bps:.3f} bps "
        f"@participation={participation}, adverse tick snap=half a tick, "
        f"platform_fee=${platform_fee:.2f}/trade  [{source}]"
    )
    return cost_bps, desc


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def load_panel(max_tickers=None):
    """Load normalised features + raw close onto one aligned timeline.

    Features come from PROCESSED_DIR (already z-scored on train-split
    statistics by preprocess.py, so they are comparable across tickers). Close
    comes from RAW_DIR reindexed onto the feature timeline, the same way
    load_aligned_close_prices() does it for the env.
    """
    meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"No metadata.json at {meta_path}. Run preprocess first.")
    meta = json.load(open(meta_path))
    features = list(meta["features"])

    tickers = sorted(meta.get("tickers") or [
        f[: -len("_features.parquet")]
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith("_features.parquet")
    ])
    if max_tickers:
        tickers = tickers[:max_tickers]

    frames, closes = {}, {}
    for t in tickers:
        fp = os.path.join(PROCESSED_DIR, f"{t}_features.parquet")
        rp = os.path.join(RAW_DIR, f"{t}.parquet")
        if not (os.path.exists(fp) and os.path.exists(rp)):
            continue
        f = pd.read_parquet(fp)
        c = pd.read_parquet(rp)["close"]
        frames[t] = f
        closes[t] = c[c > 0].reindex(f.index).ffill()

    if not frames:
        raise SystemExit("No ticker had both a feature file and a raw file.")

    index = None
    for f in frames.values():
        index = f.index if index is None else index.union(f.index)
    index = index.sort_values()

    names = sorted(frames)
    T, N = len(index), len(names)
    print(f"[panel] {N} tickers x {T} timestamps ({T * N / 1e6:.1f}M cells)")

    X = np.full((T, N, len(features)), np.nan, dtype=np.float32)
    P = np.full((T, N), np.nan, dtype=np.float32)
    for j, t in enumerate(names):
        f = frames[t].reindex(index)
        X[:, j, :] = f[features].to_numpy(dtype=np.float32)
        P[:, j] = closes[t].reindex(index).to_numpy(dtype=np.float32)

    # Phase 0 guard. Splits fabricate a -90% (or, on GE, a +700%) single-bar
    # return that lands in overnight_ret and in the forward returns this
    # script scores, so an unadjusted cache produces confident nonsense.
    # fetch_alpaca.py passes adjustment=Adjustment.ALL, but a stale local
    # cache or a re-seeded Kaggle input dataset silently predates that.
    # A DELISTING-INCLUSIVE UNIVERSE TRIPS THIS GUARD LEGITIMATELY. The test
    # "single-bar ratio outside [0.6, 1.6]" cannot tell an unadjusted split
    # from a name that actually collapsed, and a universe that includes the
    # names which died contains real -65% bars: FRC gaps 81.75 -> 28.56 from
    # 2023-03-10 15:55 to 2023-03-13 09:30, the SVB weekend. So the guard now
    # reports WHEN the jump happened and whether the ticker's history ends
    # before the panel's, which is what separates the two cases -- a split is
    # an isolated jump in a name that keeps trading, a collapse clusters and
    # the name stops.
    jumpy = []
    for j, t in enumerate(names):
        fin = np.isfinite(P[:, j]) & (P[:, j] > 0)
        c = P[fin, j]
        if c.size < 2:
            continue
        r = c[1:] / c[:-1]
        hit = np.where((r < 0.6) | (r > 1.6))[0]
        if hit.size:
            rows = np.where(fin)[0]
            when = index[rows[hit[0] + 1]]
            ends = index[rows[-1]]
            jumpy.append((t, int(hit.size), str(when)[:16], str(ends)[:10],
                          float(r[hit[0]]), bool(rows[-1] < len(index) - 1)))
    if jumpy:
        print()
        print("!" * 78)
        print(f"!! {len(jumpy)} ticker(s) carry a single-bar close ratio outside "
              f"[0.6, 1.6]:")
        for t, n, when, ends, ratio, delisted in jumpy[:16]:
            delisted_tag = "series ENDS here -- collapse/delisting, not a split"
            split_tag = "name keeps trading -- CHECK FOR AN UNADJUSTED SPLIT"
            tag = delisted_tag if delisted else split_tag
            print(f"!!   {t:<7} {n} jump(s), first {when} ratio {ratio:.3f}, "
                  f"last bar {ends}  [{tag}]")
        print("!! An unadjusted split looks like this. So does a real collapse. Read "
              "the tag on each line;")
        print("!! only the CHECK FOR AN UNADJUSTED SPLIT rows question the cache.")
        if any(not d for *_, d in jumpy):
            print("!! At least one is a LIVE name. That is the unadjusted-split")
            print("!! signature and those numbers are contaminated. Re-fetch.")
        else:
            print("!! All of them END at the jump, so these are the universe's real")
            print("!! collapses -- SIVB, FRC and the rest -- not a stale cache.")
        print("!" * 78)
        print()

    ny = index.tz_convert("America/New_York")
    # Bar-of-day from the clock, not a row counter: one missing bar would
    # otherwise shift every later regime label for that session.
    bar_of_day = np.asarray(
        ((ny.hour * 60 + ny.minute) - (9 * 60 + 30)) // 5, dtype=np.int16
    )
    day_id = pd.factorize(pd.Series(ny.normalize()))[0].astype(np.int64)

    # For every bar, the index of the LAST bar of its own session. The env
    # force-closes there (EnvConfig.flatten_at_session_close), so no hold may
    # run past it and forward returns have to be capped the same way.
    n = len(day_id)
    session_last_idx = np.empty(n, dtype=np.int64)
    cur = n - 1
    for k in range(n - 1, -1, -1):
        if k == n - 1 or day_id[k] != day_id[k + 1]:
            cur = k
        session_last_idx[k] = cur

    return dict(X=X, P=P, features=features, tickers=names,
                day_id=day_id, bar_of_day=bar_of_day,
                session_last_idx=session_last_idx,
                # The aligned timeline itself. Nothing in this script needs it,
                # but eval/xsec_book.py reindexes raw volume onto it to measure
                # per-ticker ADV, and reconstructing it from day_id/bar_of_day
                # would not survive a missing bar.
                index=index)


def overnight_decision_bars(day_id, session_last_idx, T):
    """Bars at which an overnight trade is DECIDED, one per session.

    Returns L, the last-bar index of every session that has a bar before it and
    a session after it. The decision bar is L-1, the entry bar L, the exit bar
    L+1 -- which is the first bar of the next session, because the panel is
    RTH-only and contiguous, so the overnight hold is the single transition
    L -> L+1.

    Shared by convention_table (which measures its IC) and xsec_book (which
    builds the book), so the two cannot drift apart on which bars are eligible.
    """
    L = np.unique(session_last_idx)
    L = L[(L - 1 >= 0) & (L + 1 < T)]
    # A single-bar session has no decision bar inside itself, and entering on
    # the strength of the PREVIOUS session's close is a different trade.
    return L[day_id[L - 1] == day_id[L]]


def exit_index(T, h, session_last_idx=None):
    """The bar a position opened at t is actually closed on, for horizon h.

    Factored out of forward_return_bps because the session cap makes several
    nominal horizons the SAME trade, and the only exact way to detect that is
    to compare the exit bars themselves.
    """
    t = np.arange(T)
    if session_last_idx is None:
        exit_idx = np.minimum(t + h, T - 1)
    else:
        exit_idx = np.minimum(t + h, session_last_idx)
    return np.minimum(exit_idx, T - 1)


def distinct_horizons(T, horizons, entry_mask, session_last_idx):
    """Drop horizons that are a relabelling of a shorter one on these entries.

    The session cap means a hold cannot run past the close, so on entries in
    the close ramp a nominal 30min, 1hr, 1day and 1week horizon all describe
    the same two-or-three-bar trade, and on any regime every horizon >= one
    session collapses onto 1day. Scoring them separately was not harmless:
    the duplicates were credited FEWER bets per year on an identical return
    series, so they entered the table as strictly dominated ghost rows, and
    they inflated the Benjamini-Hochberg denominator -- 300 cells corrected as
    300 when only 210 carried independent information, making the correction
    about 30% stricter than the evidence warranted.

    Returns [(name, h, aliases)], shortest first, keeping the shortest name of
    each equivalence class so bets_per_year is computed from the hold that is
    really being taken.
    """
    kept, seen = [], {}
    for name, h in sorted(horizons.items(), key=lambda kv: kv[1]):
        sig = exit_index(T, h, session_last_idx)[entry_mask].tobytes()
        if sig in seen:
            kept[seen[sig]][2].append(name)
            continue
        seen[sig] = len(kept)
        kept.append((name, h, []))
    return kept


def forward_return_bps(P, h, session_last_idx=None):
    """log(close[t+h] / close[t]) in bps, capped at the session close.

    With EnvConfig.flatten_at_session_close the env liquidates on the last bar
    of every session, so a position opened at bar b cannot be held for more
    than (77 - b) bars however long the nominal horizon is. Scoring an
    uncapped 78- or 390-bar return would measure a trade the system is
    incapable of placing -- and those were precisely the horizons with
    reachable break-even ICs, so an uncapped lab would hand back a PASS on a
    cell that cannot be built. Entries with no room left to trade (already at
    the close) return NaN and drop out.
    """
    T = P.shape[0]
    t = np.arange(T)
    exit_idx = exit_index(T, h, session_last_idx)
    out = np.full_like(P, np.nan, dtype=np.float32)
    ok = exit_idx > t
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = np.log(P[exit_idx[ok]] / P[t[ok]]) * 1e4
    return out



def cross_sectional_demean(a):
    """Subtract the per-timestamp cross-sectional mean. Shape (T, N[, F])."""
    import warnings
    with warnings.catch_warnings():
        # An all-NaN timestamp (every ticker missing that bar) is normal on a
        # union index and nanmean is right to return NaN for it.
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(a, axis=1, keepdims=True)
    return a - np.nan_to_num(m, nan=0.0)


# ---------------------------------------------------------------------------
# Volatility-conditioned feature augmentation
# ---------------------------------------------------------------------------

# Directional features worth conditioning on volatility. Deliberately only the
# return/reversal family: these are the ones with a sign that means "up" or
# "down", so scaling or regime-splitting them is meaningful. Conditioning
# is_overnight or time_sin on volatility would just manufacture noise columns
# and inflate the Benjamini-Hochberg denominator for nothing.
AUGMENT_BASE = ("log_ret", "log_ret_3", "log_ret_6", "log_ret_12",
                "log_ret_5d", "log_ret_20d", "vwap_dev", "xs_resid")

# Feature carrying the volatility read. First match wins.
AUGMENT_VOL = ("vol_z", "rv")


def _vol_rank(vol):
    """Per-bar cross-sectional rank of `vol` (T, N) mapped to (0, 1].

    A RANK, not the raw value, because preprocess.py z-scores every feature on
    train-split statistics: `rv`/`vol_z` arrive centred near zero and can be
    negative, so dividing a signal by them directly would flip its sign on
    roughly half the panel and explode near zero. The cross-sectional rank is
    strictly positive, bounded, and scale-free, so it survives whatever
    normalisation upstream applied. NaN volatility ranks to the middle (0.5)
    rather than dropping the observation.
    """
    T, N = vol.shape
    out = np.full((T, N), 0.5, dtype=np.float32)
    for t in range(T):
        v = vol[t]
        ok = np.isfinite(v)
        n = int(ok.sum())
        if n < 2:
            continue
        order = np.argsort(np.argsort(v[ok]))
        out[t, ok] = (order + 1.0) / n
    return out


def augment_features(X, features):
    """Append volatility-conditioned variants of the directional features.

    Three variants per base feature, addressing the two things asked of them:

      vsc:<f>  volatility-SCALED and cross-sectionally CENTRED.
               f / (0.5 + vol_rank) down-weights high-volatility names so a
               few noisy ones cannot dominate, then the per-bar cross-
               sectional mean is removed. Centring is what makes this a
               directional-bias fix: a signal with zero cross-sectional mean
               every bar cannot express a systematic long or short tilt, so
               the long/short imbalance is removed by construction rather
               than penalised after the fact the way the env's
               _diversity_bonus() does it.

      vch:<f>  the signal, live only in the HIGH-volatility half of the
      vcl:<f>  cross-section (vol_rank > 0.5) / the LOW-volatility half.
               Splitting rather than interacting (f * vol_z) keeps each
               column on the same scale as its parent and makes an
               asymmetric result readable: if reversal only pays when
               volatility is high, vch scores and vcl does not.

    Returns (X_aug, features_aug). A no-op returning the inputs unchanged if
    no volatility feature is present, so this cannot silently half-apply.
    """
    vol_name = next((v for v in AUGMENT_VOL if v in features), None)
    if vol_name is None:
        print("[augment] no volatility feature "
              f"({'/'.join(AUGMENT_VOL)}) in panel -- skipped")
        return X, features

    base = [f for f in AUGMENT_BASE if f in features]
    if not base:
        print("[augment] no directional base features in panel -- skipped")
        return X, features

    vr = _vol_rank(X[:, :, features.index(vol_name)])
    scale = (0.5 + vr)[:, :, None]          # in [0.5, 1.5], never zero
    hi = (vr > 0.5)[:, :, None]

    idx = [features.index(f) for f in base]
    B = X[:, :, idx]

    vsc = cross_sectional_demean(B / scale)
    vch = np.where(hi, B, 0.0).astype(np.float32)
    vcl = np.where(hi, 0.0, B).astype(np.float32)

    X_aug = np.concatenate([X, vsc, vch, vcl], axis=2).astype(np.float32)
    features_aug = (list(features)
                    + [f"vsc:{f}" for f in base]
                    + [f"vch:{f}" for f in base]
                    + [f"vcl:{f}" for f in base])
    print(f"[augment] vol feature '{vol_name}', {len(base)} base "
          f"({', '.join(base)}) -> +{3 * len(base)} columns, "
          f"{len(features)} -> {len(features_aug)} features")
    return X_aug, features_aug


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _demean_one(v, g):
    """Subtract each group's mean, vectorised.

    The obvious loop -- `for k in np.unique(g): out[g == k] -= ...` -- is O(groups
    x n) and measured 12.0 s for one 9M-point array at 77 bars x 100 tickers.
    Called twice per side, ~15 candidates per cell, ~20 cells, that is roughly
    two hours of a Kaggle session spent on arithmetic that bincount does in one
    pass. Groups with fewer than two finite values are left alone, matching the
    loop's behaviour.
    """
    out = np.asarray(v, dtype=np.float64).copy()
    ok = np.isfinite(out)
    if not ok.any():
        return out
    gi = np.asarray(g, dtype=np.int64)
    size = int(gi.max()) + 1 if gi.size else 0
    cnt = np.bincount(gi[ok], minlength=size)
    tot = np.bincount(gi[ok], weights=out[ok], minlength=size)
    mean = np.where(cnt >= 2, tot / np.maximum(cnt, 1), 0.0)
    out[ok] -= mean[gi[ok]]
    return out



def control_fixed_effects(v, bar, ticker, passes=2):
    """Sweep out entry-bar and ticker means -- a two-way fixed effect.

    Both are confounds that manufacture passing cells out of nothing:

    ENTRY BAR. A regime spanning several bars rewards any signal that merely
    identifies WHICH bar it is on, because the forward return is not
    homogeneous across them. close_ramp is the sharp case: at bar 77 the
    1-bar forward return IS the overnight gap (median 57 bps) while at bar 75
    it is an ordinary intraday move (median 16 bps). A first pass had
    time_sin, time_cos and vol_z all "passing" there at Sharpe 1.7-2.8, every
    one a clock proxy collecting the overnight risk premium the benchmark row
    already prices. Time-of-day is a lever this project handles with an
    env-level trading window; the alpha gate must not hand out credit for it.

    TICKER. preprocess z-scores every feature against PER-TICKER statistics,
    so two names at the same bar carry different values for even a pure clock
    feature. Removing only the bar mean therefore leaves a per-name constant,
    and correlating a per-name constant with a per-name forward return just
    reports which tickers happened to rise during val. That is name selection
    after the fact, not a tradable signal -- time_sin survived the bar control
    alone at Sharpe 1.46 on exactly this mechanism.

    Two passes because the two demeanings are not orthogonal on an unbalanced
    panel; it converges fast at this many groups.
    """
    out = np.asarray(v, dtype=np.float64).copy()
    for _ in range(passes):
        out = _demean_one(out, bar)
        out = _demean_one(out, ticker)
    return out


def block_ic(pred, actual, block_id, min_obs=20, min_blocks=8):
    """Mean per-block IC and its t-statistic across blocks, vectorised.

    Blocks, not days: with an h-bar forward return, observations inside h bars
    of each other share almost all of their outcome, so day-level clustering
    still counts the same move many times. The caller sizes the block to the
    horizon.

    Per-block Pearson correlation comes from grouped sums rather than a Python
    loop over blocks -- same reason as _demean_one, the loop measured 5.7 s per
    call and there are several hundred calls.
    """
    ok = np.isfinite(pred) & np.isfinite(actual)
    if ok.sum() < 200:
        return np.nan, np.nan, 0, 0
    x = np.asarray(pred, dtype=np.float64)[ok]
    y = np.asarray(actual, dtype=np.float64)[ok]
    g = np.asarray(block_id, dtype=np.int64)[ok]

    size = int(g.max()) + 1
    n = np.bincount(g, minlength=size).astype(np.float64)
    sx = np.bincount(g, weights=x, minlength=size)
    sy = np.bincount(g, weights=y, minlength=size)
    sxx = np.bincount(g, weights=x * x, minlength=size)
    syy = np.bincount(g, weights=y * y, minlength=size)
    sxy = np.bincount(g, weights=x * y, minlength=size)

    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        ics = cov / np.sqrt(vx * vy)

    keep = (n >= min_obs) & (vx > 0) & (vy > 0) & np.isfinite(ics)
    ics = ics[keep]
    if ics.size < min_blocks:
        return np.nan, np.nan, int(ics.size), int(ok.sum())
    se = ics.std(ddof=1) / math.sqrt(ics.size)
    t = float(ics.mean() / se) if se > 0 else np.nan
    return float(ics.mean()), t, int(ics.size), int(ok.sum())



def ridge_fit(X, y, alpha=10.0):
    """Closed-form ridge on the (small) feature set. Returns coefficients."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 1000:
        return None
    Xs = np.c_[X[ok].astype(np.float64), np.ones(int(ok.sum()))]
    ys = y[ok].astype(np.float64)
    A = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    A[-1, -1] -= alpha  # do not penalise the intercept
    try:
        return np.linalg.solve(A, Xs.T @ ys)
    except np.linalg.LinAlgError:
        return None


def bets_per_year(regime_bars, h):
    """Entries per name per year, respecting that one position blocks re-entry."""
    per_day = min(len(regime_bars), BARS_PER_DAY / h)
    return per_day * TRADING_DAYS


def net_sharpe(ic, sigma_bps, rt_cost_bps, n_bets):
    """Annual Sharpe per name, net of round-trip cost, no diversification.

    SELECTIVITY AND BET COUNT MUST AGREE. The earlier form took the top-decile
    edge concentration (SELECTIVITY_K = 1.75) and then multiplied by
    sqrt(n_bets) for EVERY opportunity -- claiming both the selectivity of a
    trader who sits out 90% of bars and the bet count of one who never does.
    That inflated every cell by sqrt(1 / SELECTIVITY_FRACTION) = 3.16x, which
    is the whole gap between the gate reporting Sharpe 0.96 on
    all/next_close/uni:log_ret_1560 and a portfolio backtest of the same signal
    returning ~0.3. Both consistent readings agree at ~0.30:

        trade everything   edge = ic * 1.00 * sigma, bets = n_bets
        top-decile         edge = ic * 1.75 * sigma, bets = n_bets * 0.10

    The selective form is kept because it is the higher of the two and the
    honest ceiling, but the bet count is now scaled to match it.
    """
    if not np.isfinite(ic) or sigma_bps <= 0 or n_bets <= 0:
        return np.nan
    edge = ic * SELECTIVITY_K * sigma_bps
    effective_bets = n_bets * SELECTIVITY_FRACTION
    if effective_bets <= 0:
        return np.nan
    return ((edge - rt_cost_bps) / sigma_bps) * math.sqrt(effective_bets)


def breadth_factor(signal, n_names):
    """Diversification credit for a cross-sectional book.

    A directional signal is traded one name at a time and those names move
    together, so the per-name Sharpe is what you get -- factor 1.0.

    A cross-sectionally demeaned signal is traded as a dollar-neutral book of
    N names at once. The market factor has been swept out of both sides, so
    what remains is residual and genuinely closer to independent; the
    fundamental law credits sqrt(breadth). Residual returns are still
    correlated through sectors, so the effective breadth is well under N --
    N/4 is the conservative convention used here, i.e. a 5x credit at 100
    names rather than the 10x full independence would imply.

    IMPORTANT: this scales the ECONOMIC magnitude only. The statistical bar --
    |t| >= 2 on block-clustered errors, and BH survival across every cell
    tested -- is untouched by breadth, so a cross-sectional cell still has to
    be significant on its own before this credit means anything. Fixed here
    before any real result was seen, so it cannot become a way to talk a
    marginal cell into passing.
    """
    if "xsectional" not in signal:
        return 1.0
    return math.sqrt(max(n_names / 4.0, 1.0))


def benjamini_hochberg(pvals, q):
    """Return a boolean mask of discoveries at false-discovery rate q."""
    p = np.asarray(pvals, dtype=float)
    keep = np.isfinite(p)
    out = np.zeros(len(p), dtype=bool)
    if not keep.any():
        return out
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * (np.arange(1, m + 1) / m)
    below = p[order] <= thresh
    if not below.any():
        return out
    cut = np.flatnonzero(below)[-1]
    out[order[: cut + 1]] = True
    return out


def two_sided_p(t, dof):
    """Normal approximation is fine at the block counts here; guard small dof."""
    if not np.isfinite(t) or dof < 3:
        return np.nan
    from math import erfc, sqrt
    return float(erfc(abs(t) / sqrt(2.0)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Supervised alpha gate.")
    ap.add_argument("--tickers", type=int, default=None, help="cap universe size")
    ap.add_argument("--participation", type=float, default=DEFAULT_PARTICIPATION,
                    help="assumed order participation for the impact term "
                         f"(default {DEFAULT_PARTICIPATION}, the value that "
                         "reproduces the round trip the live runs paid)")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--json", type=str, default=None, help="write results here")
    ap.add_argument("--augment", dest="augment", action="store_true", default=True,
                    help="add volatility-scaled/centred (vsc:) and "
                         "vol-regime-split (vch:/vcl:) variants of the "
                         "directional features (default on)")
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="score only the panel's own features, as before")
    args = ap.parse_args(argv)

    cost_bps, cost_desc = build_cost_model(args.participation)
    print("=" * 92)
    print("ALPHA LAB")
    print("=" * 92)
    print(f"[cost] {cost_desc}")

    panel = load_panel(args.tickers)
    X, P = panel["X"], panel["P"]
    features, tickers = panel["features"], panel["tickers"]
    if args.augment:
        # Before any split index is taken: the augmentation is a per-bar
        # cross-sectional transform, so it never reads across time and cannot
        # leak validation information backwards into train.
        X, features = augment_features(X, features)
    day_id, bod = panel["day_id"], panel["bar_of_day"]
    T, N = P.shape

    i_train = int(T * TRAIN_FRAC)
    i_val = int(T * (TRAIN_FRAC + VAL_FRAC))
    print(f"[split] train 0:{i_train}  val {i_train}:{i_val}  "
          f"test {i_val}:{T} (untouched)")

    median_price = float(np.nanmedian(P))
    rt_cost = 2.0 * float(cost_bps(median_price))
    print(f"[cost] median price ${median_price:,.2f} -> round trip {rt_cost:.2f} bps")
    print()

    rows = np.arange(T)
    sli = panel["session_last_idx"]
    fwd_cache = {h: forward_return_bps(P, h, sli) for h in HORIZONS.values()}

    # Horizons the session cap collapses onto a shorter one, per regime. Done
    # once here so the opportunity table and the scan below agree on which
    # cells actually exist.
    horizons_for = {
        rname: distinct_horizons(T, HORIZONS, np.isin(bod, bars), sli)
        for rname, bars in REGIMES.items()
    }

    # --- opportunity side --------------------------------------------------
    print("REALISED MOVE AND BREAK-EVEN IC (train split)")
    hdr = f"{'regime':<12}" + "".join(f"{h:>15}" for h in HORIZONS)
    print(hdr)
    sigma_tab = {}
    aliased = []
    for rname, bars in REGIMES.items():
        entry = np.isin(bod, bars)
        line = f"{rname:<12}"
        keep = {name: al for name, _, al in horizons_for[rname]}
        for hname, h in HORIZONS.items():
            if hname not in keep:
                # A relabelling of a shorter horizon; marked, not scored.
                line += f"{'=':>15}"
                continue
            sel = fwd_cache[h][entry & (rows < i_train)]
            sel = sel[np.isfinite(sel)]
            if sel.size < 500:
                sigma_tab[(rname, hname)] = np.nan
                line += f"{'-':>15}"
                continue
            sigma = float(sel.std())
            sigma_tab[(rname, hname)] = sigma
            med = float(np.median(np.abs(sel)))
            line += f"{med:>8.1f}/{rt_cost / (sigma * SELECTIVITY_K):>6.4f}"
        print(line)
        for name, _, al in horizons_for[rname]:
            if al:
                aliased.append(f"{rname}: {', '.join(al)} == {name}")
    print("  (median |move| bps / break-even IC;  '=' the session cap makes "
          "this the same trade as a shorter horizon)")
    for a in aliased:
        print(f"  {a}")
    print()

    # --- benchmarks --------------------------------------------------------
    print("BENCHMARKS -- always long, no model, val split")
    print(f"{'cell':<24}{'mean bps':>12}{'net of cost':>14}{'bets/yr':>10}{'sharpe':>10}")
    bench = {}
    for rname, hname in [("all", "5min"), ("all", "1day")]:
        h = HORIZONS[hname]
        entry = np.isin(bod, REGIMES[rname])
        sel = entry & (rows >= i_train) & (rows < i_val)
        y = fwd_cache[h][sel].ravel()
        y = y[np.isfinite(y)]
        if y.size < 100:
            continue
        nb = bets_per_year(REGIMES[rname], h)
        sh = ((y.mean() - rt_cost) / y.std()) * math.sqrt(nb) if y.std() > 0 else np.nan
        bench[f"{rname}/{hname}"] = sh
        print(f"{rname + ' / ' + hname:<24}{y.mean():>12.3f}"
              f"{y.mean() - rt_cost:>14.3f}{nb:>10.0f}{sh:>10.2f}")
    # Uncapped, entry on the session's last bar: the trade the env will not
    # place. Shown so the cost of flatten_at_session_close is a number rather
    # than an assumption.
    on_entry = np.isin(bod, [BARS_PER_DAY - 1])
    on_sel = on_entry & (rows >= i_train) & (rows < i_val)
    on_y = forward_return_bps(P, 1, None)[on_sel].ravel()
    on_y = on_y[np.isfinite(on_y)]
    if on_y.size > 100:
        nb = TRADING_DAYS
        sh = ((on_y.mean() - rt_cost) / on_y.std()) * math.sqrt(nb)
        print(f"{'overnight (NOT tradable)':<24}{on_y.mean():>12.3f}"
              f"{on_y.mean() - rt_cost:>14.3f}{nb:>10.0f}{sh:>10.2f}")
    print("  A model must beat these. They are risk premia, not skill.")
    print("  The overnight row is what flatten_at_session_close gives up; the")
    print("  env cannot place it, so no cell above may be judged against it.")
    print()

    # --- the scan ----------------------------------------------------------
    results = []
    for rname, bars in REGIMES.items():
        entry = np.isin(bod, bars)
        tr = entry & (rows < i_train)
        va = entry & (rows >= i_train) & (rows < i_val)
        if tr.sum() < 200 or va.sum() < 50:
            continue

        for hname, h, _ in horizons_for[rname]:
            sigma = sigma_tab.get((rname, hname))
            if not sigma or not np.isfinite(sigma):
                continue
            fwd = fwd_cache[h]
            y_tr, y_va = fwd[tr].ravel(), fwd[va].ravel()

            # Block width scaled to the horizon so overlapping outcomes never
            # land in different clusters.
            block_days = int(math.ceil(h / BARS_PER_DAY)) + 1
            blk_va = np.repeat(day_id[va] // block_days, N)
            # Entry-bar label per flattened observation, for the control below.
            bar_va = np.repeat(bod[va], N)
            tkr_va = np.tile(np.arange(N), int(va.sum()))

            nb = bets_per_year(bars, h)
            cand = []

            # 1. univariate -- sign taken from TRAIN, applied to val
            for k, fname in enumerate(features):
                s_tr = X[tr, :, k].ravel()
                ok = np.isfinite(s_tr) & np.isfinite(y_tr)
                if ok.sum() < 1000:
                    continue
                c = np.corrcoef(s_tr[ok], y_tr[ok])[0, 1]
                if not np.isfinite(c) or c == 0:
                    continue
                sign = math.copysign(1.0, c)
                cand.append((f"uni:{fname}", sign * X[va, :, k].ravel(), y_va))

            # 2. ridge over the full feature set, fit on train only
            beta = ridge_fit(X[tr].reshape(-1, len(features)), y_tr)
            if beta is not None:
                pred = X[va].reshape(-1, len(features)) @ beta[:-1] + beta[-1]
                cand.append(("ridge:directional", pred, y_va))

            # 3. cross-sectional: demean both sides, so the market factor (and
            #    the equity/overnight risk premium riding on it) cannot score.
            Xcs_tr = cross_sectional_demean(X[tr]).reshape(-1, len(features))
            Xcs_va = cross_sectional_demean(X[va]).reshape(-1, len(features))
            ycs_tr = cross_sectional_demean(fwd)[tr].ravel()
            ycs_va = cross_sectional_demean(fwd)[va].ravel()
            beta_cs = ridge_fit(Xcs_tr, ycs_tr)
            if beta_cs is not None:
                cand.append(("ridge:xsectional",
                             Xcs_va @ beta_cs[:-1] + beta_cs[-1], ycs_va))

            # The controlled target depends only on the cell, not on the
            # candidate, and there are just two variants (directional and
            # cross-sectional) against ~15 candidates. Cache it.
            _tgt_cache = {}

            for signal, pred, target in cand:
                # Sweep entry-bar and ticker fixed effects out of BOTH sides
                # before scoring, so a signal cannot earn IC by identifying
                # the bar or the name rather than predicting the move.
                pred = control_fixed_effects(pred, bar_va, tkr_va)
                _key = id(target)
                if _key not in _tgt_cache:
                    _tgt_cache[_key] = control_fixed_effects(target, bar_va, tkr_va)
                target = _tgt_cache[_key]
                ic, t, nblocks, n = block_ic(pred, target, blk_va)
                if not np.isfinite(ic):
                    continue
                per_name = net_sharpe(ic, sigma, rt_cost, nb)
                bf = breadth_factor(signal, N)
                results.append(dict(
                    regime=rname, horizon=hname, signal=signal, ic=ic, t=t,
                    breakeven=rt_cost / (sigma * SELECTIVITY_K),
                    sharpe_per_name=per_name,
                    breadth=bf,
                    sharpe=per_name * bf if np.isfinite(per_name) else np.nan,
                    bets_per_year=nb, n_blocks=nblocks, n_obs=n,
                    p=two_sided_p(t, nblocks),
                ))

    if not results:
        print("No cell produced a usable estimate. Check the panel and splits.")
        return 1

    # --- multiple-testing correction across EVERY cell tested --------------
    disc = benjamini_hochberg([r["p"] for r in results], BH_Q)
    for r, d in zip(results, disc):
        r["bh"] = bool(d)

    results.sort(key=lambda r: -(r["sharpe"] if np.isfinite(r["sharpe"]) else -9e9))
    passing = [r for r in results
               if np.isfinite(r["sharpe"]) and r["sharpe"] >= MIN_SHARPE
               and abs(r["t"]) >= MIN_ABS_T and r["bh"]
               and r["sharpe"] > max(bench.values(), default=0.0)]

    print(f"MEASURED ON VAL -- {len(results)} cells tested, "
          f"BH q={BH_Q} across all of them")
    print(f"{'regime':<11}{'horizon':<8}{'signal':<21}{'ic':>8}{'t':>7}"
          f"{'blocks':>7}{'bets/yr':>8}{'/name':>7}{'xN':>5}{'book':>7}{'BH':>4}  verdict")
    for r in results[: args.top]:
        verdict = "PASS" if r in passing else ""
        print(f"{r['regime']:<11}{r['horizon']:<8}{r['signal']:<21}"
              f"{r['ic']:>8.4f}{r['t']:>7.2f}{r['n_blocks']:>7d}"
              f"{r['bets_per_year']:>8.0f}{r['sharpe_per_name']:>7.2f}"
              f"{r['breadth']:>5.1f}{r['sharpe']:>7.2f}"
              f"{'y' if r['bh'] else 'n':>4}  {verdict}")
    if len(results) > args.top:
        print(f"... {len(results) - args.top} further cell(s) with lower Sharpe")
    print()

    print("=" * 92)
    if passing:
        print(f"GATE PASS -- {len(passing)} cell(s) clear sharpe >= {MIN_SHARPE}, "
              f"|t| >= {MIN_ABS_T}, BH q={BH_Q}, and beat every benchmark.")
        for r in passing[:5]:
            print(f"    {r['regime']}/{r['horizon']}/{r['signal']}: "
                  f"ic={r['ic']:.4f} t={r['t']:.2f} sharpe={r['sharpe']:.2f}")
        print("Build the route these name: they fix the horizon, the regime, and")
        print("whether the edge is directional or cross-sectional.")
    else:
        best = results[0]
        print(f"GATE FAIL -- no cell clears sharpe >= {MIN_SHARPE} with |t| >= "
              f"{MIN_ABS_T}, BH survival, and a benchmark beat.")
        print(f"Best by Sharpe: {best['regime']}/{best['horizon']}/{best['signal']} "
              f"ic={best['ic']:.4f} t={best['t']:.2f} sharpe={best['sharpe']:.2f} "
              f"BH={'y' if best['bh'] else 'n'}")
        print("Do NOT start a training run. The blocker is the feature set, not")
        print("the policy, the reward, or the execution model.")
    print("=" * 92)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(cost_desc=cost_desc, round_trip_bps=rt_cost,
                           min_sharpe=MIN_SHARPE, bh_q=BH_Q,
                           n_tickers=len(tickers), n_cells=len(results),
                           benchmarks=bench, passing=len(passing),
                           results=results), fh, indent=2, default=float)
        print(f"[json] wrote {args.json}")

    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
