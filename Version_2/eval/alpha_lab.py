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
    sharpe    = (net_bps / sigma_h) * sqrt(bets_per_year)

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
    jumpy = []
    for j, t in enumerate(names):
        c = P[:, j]
        c = c[np.isfinite(c) & (c > 0)]
        if c.size < 2:
            continue
        r = c[1:] / c[:-1]
        if np.any((r < 0.6) | (r > 1.6)):
            jumpy.append(t)
    if jumpy:
        print()
        print("!" * 78)
        print(f"!! {len(jumpy)} ticker(s) carry a single-bar close ratio outside "
              f"[0.6, 1.6]:")
        print(f"!!   {", ".join(jumpy[:16])}{" ..." if len(jumpy) > 16 else ""}")
        print("!! That is the unadjusted-split signature. This cache predates "
              "adjustment=Adjustment.ALL.")
        print("!! Every number below is contaminated. Re-fetch before believing "
              "any of it.")
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
                session_last_idx=session_last_idx)


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
    if session_last_idx is None:
        exit_idx = np.minimum(t + h, T - 1)
    else:
        exit_idx = np.minimum(t + h, session_last_idx)
    exit_idx = np.minimum(exit_idx, T - 1)
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
    """Annual Sharpe per name, net of round-trip cost, no diversification."""
    if not np.isfinite(ic) or sigma_bps <= 0 or n_bets <= 0:
        return np.nan
    edge = ic * SELECTIVITY_K * sigma_bps
    return ((edge - rt_cost_bps) / sigma_bps) * math.sqrt(n_bets)


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
    ap.add_argument("--participation", type=float, default=0.0,
                    help="assumed order participation for the impact term")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--json", type=str, default=None, help="write results here")
    args = ap.parse_args(argv)

    cost_bps, cost_desc = build_cost_model(args.participation)
    print("=" * 92)
    print("ALPHA LAB")
    print("=" * 92)
    print(f"[cost] {cost_desc}")

    panel = load_panel(args.tickers)
    X, P = panel["X"], panel["P"]
    features, tickers = panel["features"], panel["tickers"]
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

    # --- opportunity side --------------------------------------------------
    print("REALISED MOVE AND BREAK-EVEN IC (train split)")
    hdr = f"{'regime':<12}" + "".join(f"{h:>15}" for h in HORIZONS)
    print(hdr)
    sigma_tab = {}
    for rname, bars in REGIMES.items():
        entry = np.isin(bod, bars)
        line = f"{rname:<12}"
        for hname, h in HORIZONS.items():
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
    print("  (median |move| bps / break-even IC)")
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

        for hname, h in HORIZONS.items():
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
