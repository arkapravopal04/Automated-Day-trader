"""
eval/xsec_book.py -- the cost-aware cross-sectional book, as an analytic baseline.

P2's first claim is that the system's problem is not the signal but the
STRUCTURE it is traded in. Today the env runs 100 independent single-name
accounts (`PortfolioState(n_envs=100, n_tickers=1)`), each taking a directional
bet, each paying its own round trip, each exposed to the market factor. P1
closed with alpha 0.006 bps against cost 0.776 bps on exactly that structure.

This script prices the alternative WITHOUT training anything:

    one book, dollar-neutral, gross 1, with per-name cost as a first-class
    input to each name's weight:

        w_i  proportional to  (|edge_i| - lambda * cost_i)+ * sign(edge_i) / cost_i
        w    <- w - mean(w over the SELECTED names)      (dollar-neutral)
        w    <- w / sum|w|                                (gross-normalised)

Read the formula as three separate statements, because they are separable and
each is doing different work:

  * (|edge| - lambda*cost)+     a HURDLE. A name whose predicted edge does not
                                clear lambda times its own round trip is not
                                traded at all. lambda = 0 trades everything with
                                a sign; lambda = 1 demands the edge pay for the
                                trade once before it is taken.
  * ... * sign(edge)            direction.
  * ... / cost                  SIZING. Of the names that clear the hurdle, the
                                cheap-to-trade ones get more capital. This is the
                                half the 100-account structure cannot express at
                                all: there, cost enters only as a deduction after
                                the size has already been chosen.

The division is not a heuristic. With edge_i and cost_i in the same units,
(edge - lambda*cost)/cost is net edge per dollar of friction, and allocating a
fixed gross budget in proportion to it is the ratio-maximising split when the
names are weakly correlated.

WHY DOLLAR-NEUTRALITY IS NOT A DETAIL. Session 3 measured pairwise rho across
the panel at 0.13-0.26. At rho = 0.2, 100 directional names carry the
diversification of about 5 independent ones -- the book is one leveraged bet on
the market wearing a hundred tickers. Demeaning per bar removes that factor and
leaves the part the features were built to predict (`xs_resid` is in the panel
for this reason). It also removes the documented failure mode: a policy cannot
collapse to 96% long occupancy inside a book that sums to zero.

WHAT THIS SCRIPT IS NOT
-----------------------
It is not the env, and it does not replace `eval/backtest_report.py`. It holds
weights, not share counts; it has no order sizing, no partial fills, no
participation cap, no min-notional, no Kelly. Every one of those subtracts from
the number this prints, so read it as a CEILING on what the same signal does in
the env, and the gap between the two as implementation loss.

What it does share with the env, deliberately:
  * the cost model, rebuilt from `training.config.EnvConfig` term for term
    against `ExecutionSimulator._compute_fill_price` -- so the half-tick floor
    makes cheap names expensive here exactly as it does there. NOT alpha_lab's
    `build_cost_model`, which has been stale since P1; see the note above
    `env_cost_constants()`;
  * the session cap -- `flatten_at_session_close` means no hold survives the
    close, so the forward returns AND the rebalance schedule are both cut at
    `session_last_idx`, and the book is flat overnight;
  * the train/val/test split from `paths.py`. TEST IS NEVER READ.

lambda IS SELECTED ON TRAIN. The val column is printed for every lambda, but
selection reads the train column only -- otherwise "best lambda" is one bit of
val lookahead per sweep, which is the trap alpha_lab's docstring names.

Usage
-----
    python eval/xsec_book.py                        # ridge edge, hold 24, lambda sweep
    python eval/xsec_book.py --hold 12 --hold 24 --hold 48
    python eval/xsec_book.py --edge oracle          # machinery check, see build_edge
    python eval/xsec_book.py --variant nosize       # ablate the / cost sizing
    python eval/xsec_book.py --frame close          # the frame the env cannot trade
    python eval/xsec_book.py --edge npz --edge-npz logs/trunk_edge.npz
    python eval/xsec_book.py --json out.json

READ THE [hurdle] LINE FIRST. It prints the edge distribution against the
round-trip cost distribution and names the largest lambda that can admit
anything at all. If that number is below 1, every row at or above lambda = 1 in
the sweep below it describes a book that never trades, and its Sharpe is
computed on an empty sample.
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

from paths import RAW_DIR, TRAIN_FRAC, VAL_FRAC  # noqa: E402
from eval.alpha_lab import (
    overnight_decision_bars,  # noqa: E402
    BARS_PER_DAY,
    TRADING_DAYS,
    block_ic,
    cross_sectional_demean,
    forward_return_bps,
    load_panel,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
DEFAULT_HOLD = 24
RIDGE_ALPHA = 10.0

# Candidate penalties for `--ridge-alpha auto`. 10.0 stays the DEFAULT so every
# number measured before this existed remains reproducible; changing the default
# silently would make the P3 tables incomparable with the ones already recorded
# in AGENTS.md.
#
# Why it needs to be selectable at all: 10.0 was set when the design was ~15
# features wide. P3's panel is 27, nearly double, and more collinear -- the
# 12 `ib_*` columns are all functions of the same 5 one-minute bars. A ridge
# penalty is not scale-free in the feature count, so holding it fixed while the
# design widens confounds "did the features help" with "was the penalty right
# for this width", and biases against the wider panel.
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


def select_ridge_alpha(Xcs, target, i_train, F, alphas=RIDGE_ALPHAS, inner_frac=0.8):
    """Pick a ridge penalty on TRAIN ONLY. -> (alpha, [(alpha, inner_ic), ...])

    The selection uses an inner holdout carved from the END of train, never the
    validation split. Tuning a hyperparameter on validation and then reporting
    validation is the same defect as fitting there: it turns an out-of-sample
    number into an in-sample one, and this project has already retracted one
    result to that family of mistake.

    The inner split is TIME-ORDERED, not random. Shuffling rows of a panel puts
    bars from the same session on both sides of the split, and adjacent 5-minute
    bars share most of their information -- a shuffled holdout would score a
    penalty on data it effectively saw, and select too little regularisation.

    Ranking is by pooled IC on the inner holdout. Block-clustering matters for
    the STANDARD ERROR of an IC, not for its point estimate, and only the
    ranking is used here.
    """
    i_inner = int(i_train * inner_frac)
    if i_inner < 100 or i_train - i_inner < 100:
        return RIDGE_ALPHA, []

    Xf = Xcs[:i_inner].reshape(-1, F)
    yf = target[:i_inner].ravel()
    Xh = Xcs[i_inner:i_train].reshape(-1, F)
    yh = target[i_inner:i_train].ravel()
    ok = np.isfinite(yh) & np.isfinite(Xh).all(axis=1)
    Xh, yh = Xh[ok], yh[ok]
    if Xh.shape[0] < 1000:
        return RIDGE_ALPHA, []

    scored = []
    for a in alphas:
        beta, _ = ridge_fit_chunked(Xf, yf, alpha=a)
        if beta is None:
            continue
        pred = Xh @ beta[:-1] + beta[-1]
        ic = (float(np.corrcoef(pred, yh)[0, 1])
              if np.isfinite(pred).all() and pred.std() > 0 else float("nan"))
        scored.append((float(a), ic))

    finite = [(a, ic) for a, ic in scored if np.isfinite(ic)]
    if not finite:
        return RIDGE_ALPHA, scored
    return max(finite, key=lambda t: t[1])[0], scored

# Deployable capital the book runs on. 100 streams x EnvConfig.initial_cash is
# what the env models today, so gross 1 here is the same dollar amount the env
# has at risk -- which is what makes the impact term comparable between them.
DEFAULT_CAPITAL = 1_000_000.0


# ---------------------------------------------------------------------------
# Cost model -- ExecutionSimulator's, post-P1, term for term
# ---------------------------------------------------------------------------
#
# NOT alpha_lab's `build_cost_model`. That function computes the impact term as
#
#       impact_bps = impact_coef * sqrt(participation) * 1e4
#
# which was right when `impact_coef` was 0.015 -- a constant that had sigma_daily
# folded into it and ran against a per-BAR volume denominator. P1 redefined it as
# the dimensionless Y of
#
#       impact / price = Y * sigma_daily * sqrt(Q / ADV)
#
# and set it to 0.5, with sigma_daily and ADV supplied per ticker. alpha_lab
# still multiplies by Y alone, so with the current config it prices impact at
# 0.5 * sqrt(0.00032) * 1e4 = 89.4 bps per side instead of ~1.3 -- a factor of
# sigma_daily (~0.015) out. Anything that gate has scored since P1 has been
# screened against a ~180 bps round trip. Flagged, not fixed here: alpha_lab's
# published cells were produced with it and changing it silently would make
# them incomparable.


def env_cost_constants():
    """The friction constants, read from the env's own config."""
    from training.config import EnvConfig

    cfg = EnvConfig()
    return dict(
        spread_bps=float(cfg.spread_bps),
        commission_bps=float(cfg.commission_bps),
        impact_coef=float(cfg.impact_coef),
        tick=float(getattr(cfg, "tick_size", 0.01)),
    )


def side_cost_bps(price, shares, adv, sigma_daily, k, spread_bps=None):
    """Per-side friction in bps of notional. Mirrors `_compute_fill_price`.

        half-spread   max(spread_bps, half a tick) -- the minimum quotable US
                      equity spread is one tick, so a proportional spread alone
                      is unphysical at low prices.
        tick snap     `snap_to_tick_adverse` ceils buys and floors sells, so a
                      fill gives up a further half tick in expectation.
        impact        Y * sigma_daily * sqrt(Q / ADV), per ticker.
        commission    0.0 on Alpaca US equities.

    Both tick terms are 1/price, and that is the entire reason this book has
    anything to size on: a fixed half tick is 0.04 bps on the dearest name in
    the panel and 2.5 bps on the cheapest. Cost varies ~60x across the
    cross-section while the edge does not.
    """
    price = np.asarray(price, dtype=np.float64)
    half_tick_bps = 1e4 * (k["tick"] / 2.0) / price
    # `spread_bps` overrides EnvConfig's when the caller has measured the
    # spread for the specific moment being traded. The overnight book needs
    # this: its exit leg is the 09:30 print, where the measured effective
    # spread is 2.93 bps against 0.068 midday -- 43x -- and charging it the
    # midday figure is the single largest understatement in its cost.
    sp = k["spread_bps"] if spread_bps is None else spread_bps
    half_spread = np.maximum(sp, half_tick_bps)
    part = np.clip(np.asarray(shares, dtype=np.float64) / adv, 0.0, 1.0)
    impact_bps = 1e4 * k["impact_coef"] * sigma_daily * np.sqrt(part)
    return half_spread + half_tick_bps + k["commission_bps"] + impact_bps


# The convention table's rows. Every entry other than "close" fills and marks
# somewhere inside bar t+1; "close" scores close-to-close, which the env
# cannot trade and which is carried only as the comparison that shows how much
# of a result is convention rather than signal.
#
# The `x_*` rows require a raw directory built by intrabar.py from a 1-minute
# cache. They are the reason 1-minute bars were bought: at 5-minute resolution
# `open[t+1]` is the only price inside the fill bar that exists, so the
# execution frame had no alternative to accept. With the minutes present, the
# decision at t can be filled at the first minute's close or VWAP, or worked
# across the first two minutes, and the difference between those rows is a
# direct measurement of how much of the edge is being handed to whoever is on
# the other side of the opening print.
FRAME_COLUMNS = {
    "exec":     ("open",        "fills and marks at bar t+1 open (P1's frame)"),
    "close":    (None,          "close-to-close -- the env cannot trade this; comparison only"),
    "m1_close": ("x_close_m1",  "fills at the close of t+1's FIRST MINUTE"),
    "m1_vwap":  ("x_vwap_m1",   "fills at the VWAP of t+1's first minute"),
    "m12_vwap": ("x_vwap_m12",  "fills at the VWAP of t+1's first TWO minutes"),
    "bar_vwap": ("x_vwap_full", "fills at t+1's full-bar VWAP (an upper bound on "
                                "patient execution, not reachable in real time)"),
}


def execution_frame(index, tickers, session_last_idx, column="open"):
    """(exec_price [T, N], exec_session_last [T]) -- the P1 execution frame.

    The observation ends at bar t; the order it produces fills AND marks at bar
    t+1 (`EnvConfig.execution_price_column`, `open` by default). So the return a
    decision at t can actually capture is measured between bar t+1's open and
    the open of the bar it exits on -- not close-to-close, which is what
    alpha_lab's panel carries and what the first version of this script scored.

    Session cap, transposed into the same frame. `session_last_idx[t+1]` is the
    raw index of the last bar of the session that the FILL lands in; the exec
    index that reads that bar's price is one lower, hence the `- 1`. A decision
    whose fill lands on the session's last bar therefore gets an exit index
    equal to its own, and `forward_return_bps` returns NaN for it -- correct:
    it fills into a bar that `flatten_at_session_close` immediately liquidates.

    The final row of any shifted array is the degenerate one -- there is no
    t+1 at the end of the split -- and it is repeated rather than extrapolated,
    the same choice `_to_execution_frame()` makes, for the same reason.
    """
    import pandas as _pd

    T, N = len(index), len(tickers)
    Px = np.full((T, N), np.nan, dtype=np.float32)
    for j, t in enumerate(tickers):
        rp = os.path.join(RAW_DIR, f"{t}.parquet")
        if not os.path.exists(rp):
            continue
        raw = _pd.read_parquet(rp)
        if column not in raw.columns:
            raise KeyError(
                f"{t}: raw parquet has no '{column}' column. The intra-window "
                f"execution marks (x_*) exist only in an intrabar.py output "
                f"directory; point TRADING_RAW_DIR at one (e.g. data/parquet_agg5) "
                f"or use --frame exec. Present: {sorted(raw.columns)[:12]}"
            )
        col = raw[column].mask(raw[column] <= 0)
        # A ticker that DELISTED mid-sample has no price afterwards, and one
        # that listed late has none before. ffill().bfill() across the union
        # index invents both: it would hand a delisted name a frozen price for
        # every remaining year, which reads as a real quote with exactly zero
        # return and zero risk -- the most attractive thing a cost-aware book
        # can be offered. Fill only INSIDE the ticker's own life; outside it
        # the price is NaN, and NaN propagates to the return and drops the name
        # from the cross-section, which is what actually happened.
        #
        # This mattered the moment the universe stopped being 100 survivors:
        # see scan_delisted.py.
        first, last = col.first_valid_index(), col.last_valid_index()
        col = col.reindex(index).ffill()
        if first is not None:
            col = col.where((col.index >= first) & (col.index <= last))
        Px[:, j] = col.to_numpy(dtype=np.float32)

    shift = lambda a: np.concatenate([a[1:], a[-1:]], axis=0)
    return shift(Px), np.maximum(shift(session_last_idx) - 1, 0)


def measure_liquidity(index, tickers, day_id, i_train, P):
    """Per-ticker (ADV in shares, daily sigma), measured on TRAIN ONLY.

    The same two constants `VecTradingEnv._measure_liquidity_constants()`
    installs, measured the same way, so the impact charged here and the impact
    charged in a rollout are the same number. Train-only because they size
    every cost in the val column and measuring them across the whole panel
    would leak val liquidity backwards into the weights.
    """
    N = len(tickers)
    adv = np.full(N, np.nan)
    tr = np.arange(len(index)) < i_train
    d_tr = day_id[tr]
    n_days = len(np.unique(d_tr))

    for j, t in enumerate(tickers):
        rp = os.path.join(RAW_DIR, f"{t}.parquet")
        if not os.path.exists(rp):
            continue
        v = pd.read_parquet(rp)["volume"].reindex(index).fillna(0.0).to_numpy()
        adv[j] = float(v[tr].sum()) / max(n_days, 1)

    # Daily sigma from the last bar of each train session -- close-to-close,
    # which is what "daily" means in the impact law.
    last_of_day = np.flatnonzero(np.r_[np.diff(day_id) != 0, True])
    last_of_day = last_of_day[last_of_day < i_train]
    sigma = np.full(N, np.nan)
    if last_of_day.size > 3:
        dc = P[last_of_day]                       # [days, N]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(np.log(dc), axis=0)
        sigma = np.nanstd(r, axis=0)

    # The env clamps both; an unmeasurable name must not divide by zero or get
    # free impact. Medians, so one thin ticker cannot distort the panel.
    adv = np.where(np.isfinite(adv) & (adv > 0), adv, np.nanmedian(adv))
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nanmedian(sigma))
    return np.maximum(adv, 1.0), np.maximum(sigma, 1e-6)


# ---------------------------------------------------------------------------
# Edge estimation
# ---------------------------------------------------------------------------

def ridge_fit_chunked(X, y, alpha=RIDGE_ALPHA, chunk=200_000):
    """Ridge coefficients, accumulating the normal equations in chunks.

    Same closed form as alpha_lab.ridge_fit, but it never materialises a
    float64 copy of the design matrix. The train split is ~2.4M rows on the
    full panel; the copy is survivable at 13 features and fatal if the feature
    set grows. The intercept column is not penalised.
    """
    F = X.shape[1]
    A = np.zeros((F + 1, F + 1), dtype=np.float64)
    b = np.zeros(F + 1, dtype=np.float64)
    n_used = 0
    for s in range(0, X.shape[0], chunk):
        Xc = X[s:s + chunk]
        yc = y[s:s + chunk]
        ok = np.isfinite(yc) & np.isfinite(Xc).all(axis=1)
        if not ok.any():
            continue
        Xs = np.c_[Xc[ok].astype(np.float64), np.ones(int(ok.sum()))]
        A += Xs.T @ Xs
        b += Xs.T @ yc[ok].astype(np.float64)
        n_used += int(ok.sum())
    if n_used < 1000:
        return None, n_used
    A[np.arange(F), np.arange(F)] += alpha  # intercept row/col untouched
    try:
        return np.linalg.solve(A, b), n_used
    except np.linalg.LinAlgError:
        return None, n_used


def load_edge_npz(path, index, tickers):
    """A trunk's dumped edge, reindexed onto this panel. -> [T, N] bps.

    Keyed by timestamp and ticker NAME on both sides, never by row index: the
    two panels are built by different code (dataset.py's outer join vs
    alpha_lab's union) and agreeing today is not the same as agreeing after a
    re-fetch. Coverage is printed because a silent 3% overlap would otherwise
    look like a weak signal rather than a broken join.
    """
    import pandas as _pd

    blob = np.load(path, allow_pickle=False)
    stamps = _pd.DatetimeIndex(blob["timestamps"], tz="UTC")
    src_tickers = [str(t) for t in blob["tickers"]]
    edge_src = blob["edge_bps"]

    df = _pd.DataFrame(edge_src, index=stamps, columns=src_tickers)
    df = df[~df.index.duplicated(keep="last")]
    out = df.reindex(index=index.tz_convert("UTC"), columns=list(tickers))

    cov = float(np.isfinite(out.to_numpy()).mean())
    print(f"[edge] npz: {edge_src.shape[0]:,} dumped bars, "
          f"{len(set(src_tickers) & set(tickers))}/{len(tickers)} names matched, "
          f"{100 * cov:.1f}% of panel cells covered")
    if cov < 0.05:
        raise SystemExit(
            f"[edge] only {100 * cov:.1f}% of the panel is covered by {path} -- "
            "the timestamps or the tickers do not line up. Refusing to score a "
            "book on a broken join."
        )
    return out.to_numpy(dtype=np.float32), f"trunk edge from {os.path.basename(path)}"


def reversal_edge(P, fwd, i_train, day_id, lookback=1):
    """The target-free reversal signal, rescaled to bps by a TRAIN-ONLY fit.

    Why this exists. The overnight walk-forward measured mean IC +0.0353
    (t 2.68) for the target-free reversal against +0.0254 for the 27-feature
    ridge and +0.0280 for the 15-feature one: the features do not beat the raw
    signal at this horizon. Running the book only on `--edge ridge` would price
    the WEAKER of the two and risk calling the overnight hold dead on a signal
    that is not the one carrying the effect -- the mirror of the error the
    walk-forward protocol exists to prevent.

    THE RESCALING IS NOT COSMETIC. `build_edge`'s contract is that edge[t, i]
    is in BPS, because the weight formula compares |edge| against lambda * cost
    directly; a signal on any other scale silently rescales lambda and makes
    the sweep meaningless. A trailing return in bps is not a forecast in bps,
    so it is projected onto the realised target by a univariate least-squares
    slope fit on TRAIN ONLY. No intercept: both sides are cross-sectionally
    demeaned, so the intercept is zero by construction.

    The slope is one number estimated on train, which is one fitted parameter
    against the ridge's F+1 -- so this is a strictly more constrained
    estimator, not a freer one.
    """
    T, N = P.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        trail = np.full((T, N), np.nan, dtype=np.float32)
        trail[lookback:] = np.log(P[lookback:] / P[:-lookback]) * 1e4

    # The trailing window must not straddle the overnight gap: on the first
    # `lookback` bars of a session it spans a 17-hour move rather than a
    # 5-minute one. Same mask convention_table.build_signal applies, for the
    # same reason.
    idx = np.arange(T)
    first_of_day = np.r_[True, np.diff(day_id) != 0]
    day_start = np.maximum.accumulate(np.where(first_of_day, idx, 0))
    trail[(idx - day_start) < lookback] = np.nan

    sig = cross_sectional_demean(-trail)
    tgt = cross_sectional_demean(fwd)
    x, y = sig[:i_train].ravel(), tgt[:i_train].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    denom = float(np.dot(x[ok], x[ok]))
    if not np.isfinite(denom) or denom <= 0:
        raise SystemExit("reversal edge: degenerate train design")
    b = float(np.dot(x[ok], y[ok]) / denom)
    return (sig * b).astype(np.float32), (
        f"reversal ({lookback}-bar negated trailing return, demeaned, session-gap "
        f"masked), rescaled to bps by a train-only slope {b:+.5f} on "
        f"{int(ok.sum()):,} rows -- target-free signal, ONE fitted parameter")


def build_edge(kind, X, fwd, i_train, features, edge_npz=None, index=None,
               tickers=None, ridge_alpha=RIDGE_ALPHA):
    """Return (edge_bps [T, N], description).

    edge_bps[t, i] estimates, in bps, what name i earns AGAINST THE CROSS-
    SECTION over the hold starting at bar t. The units matter more than usual:
    the weight formula compares |edge| against lambda * cost directly, so an
    edge on any other scale silently rescales lambda and makes the sweep
    meaningless. Regressing onto the realised bps target returns a conditional
    mean in bps, which is the right scale by construction -- a z-scored or
    rank-transformed signal is not, however well it correlates.

    'ridge'   cross-sectional ridge over the panel's features, FIT ON TRAIN
              ONLY, both sides per-bar demeaned so the market factor cannot
              score. Same estimator as alpha_lab's 'ridge:xsectional' row.

    'oracle'  the realised demeaned forward return itself. NOT a strategy --
              it is the machinery check. With a perfect edge the book must post
              an enormous ratio; if it does not, the bug is in the weighting,
              the schedule or the accounting rather than in the signal.
    """
    if kind == "npz":
        if not edge_npz:
            raise SystemExit("--edge npz requires --edge-npz PATH")
        edge, desc = load_edge_npz(edge_npz, index, tickers)
        # Demeaned again here rather than trusting the producer: the book is
        # dollar-neutral and the hurdle is stated against |edge|, so a common
        # offset would tilt every name's selection the same way.
        return cross_sectional_demean(edge).astype(np.float32), desc

    Xcs = cross_sectional_demean(X)
    target = cross_sectional_demean(fwd)

    if kind == "oracle":
        return (target.astype(np.float32),
                "oracle (realised demeaned forward return -- NOT tradeable)")

    F = len(features)

    picked = ""
    if ridge_alpha == "auto":
        ridge_alpha, scored = select_ridge_alpha(Xcs, target, i_train, F)
        if scored:
            grid = "  ".join(f"{a:g}:{ic:+.5f}" for a, ic in scored)
            print(f"[ridge] inner-holdout IC by alpha (TRAIN only) -- {grid}")
        picked = " [selected on a train-only inner holdout]"

    beta, n_used = ridge_fit_chunked(
        Xcs[:i_train].reshape(-1, F), target[:i_train].ravel(), alpha=ridge_alpha
    )
    if beta is None:
        raise SystemExit("ridge fit failed -- not enough finite rows in train")
    T, N = fwd.shape
    edge = (Xcs.reshape(-1, F) @ beta[:-1] + beta[-1]).reshape(T, N).astype(np.float32)
    desc = (f"ridge:xsectional over {F} features, fit on {n_used:,} train rows "
            f"(ridge alpha={ridge_alpha:g}{picked})")
    return edge, desc


# ---------------------------------------------------------------------------
# The book
# ---------------------------------------------------------------------------

def trailing_overnight_vol(fwd, day_id, session_last_idx, T, window=60, min_obs=20):
    """[T, N] causal per-name volatility of the OVERNIGHT return, in bps.

    WHY THE BOOK NEEDS THIS. `book_weights` has no risk term at all: weights go
    as (|edge| - lambda*cost)/cost, so two names with the same edge and cost are
    held identically however differently they move. The reversal edge is
    proportional to the trailing move, and the trailing move is proportional to
    volatility, so the book systematically loads its largest positions onto its
    most volatile names. That is the mechanism behind `Sharpe ex top-5`
    collapsing: three of five overnight folds flip negative when the five best
    sessions are removed.

    Dividing by this turns equal-DOLLAR sizing into equal-RISK sizing.

    CAUSALITY IS THE WHOLE POINT. The window is the `window` sessions STRICTLY
    BEFORE the one being sized -- `[:-1]` before the rolling call, not after --
    so a session's own gap can never inform the position taken into it. Getting
    that backwards would be a look-ahead that makes the tail vanish by
    construction, which is exactly the flattery this is meant to remove.

    Names with fewer than `min_obs` prior gaps fall back to the cross-sectional
    median of that session, so a late IPO is sized like a typical name rather
    than dropped or given a degenerate zero-vol weight.
    """
    L = overnight_decision_bars(day_id, session_last_idx, T)
    dec = L - 1
    R = fwd[dec]                                  # [S, N] realised gaps, session-ordered
    S, N = R.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    if S < 2:
        return out

    df = pd.DataFrame(R.astype(np.float64)).shift(1)
    # shift(1) BEFORE rolling: row s sees sessions [s-window, s-1] and never s.
    V = df.rolling(window=window, min_periods=min_obs).std().to_numpy(dtype=np.float64)
    # Early sessions have no `min_obs` window yet. Fall back to an EXPANDING std
    # rather than leaving them unscaled: an unscaled session inside a risk-scaled
    # run is two sizing regimes in one number, which is worse than a noisier
    # estimate. Sessions 0-1 have no prior at all and stay NaN by design.
    exp = df.expanding(min_periods=2).std().to_numpy(dtype=np.float64)
    V = np.where(np.isfinite(V), V, exp)

    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(np.isfinite(V) & (V > 0), V, np.nan), axis=1)
    med = np.where(np.isfinite(med) & (med > 0), med, np.nan)
    fill = np.repeat(med[:, None], N, axis=1)
    V = np.where(np.isfinite(V) & (V > 0), V, fill)
    out[dec] = V.astype(np.float32)
    return out


def book_weights(edge_row, rt_cost_row, lam, min_names=2, size_by_cost=True,
                 risk_row=None):
    """One bar's weights, gross-normalised and dollar-neutral. (w [N], n_selected).

    `size_by_cost=False` is the ablation that isolates what the `/ cost` term
    contributes: same hurdle, same names, same signs, but the surviving names
    are held in proportion to their excess edge alone. Any difference between
    the two is the cost-aware SIZING, separated from the cost-aware SELECTION
    that the hurdle already performed.

    Dollar-neutrality is imposed over the SELECTED names only, not the whole
    universe. Demeaning across all N would hand a nonzero weight to every name
    the hurdle just rejected -- reintroducing, as the offsetting leg, exactly
    the positions whose cost the hurdle decided was not worth paying.

    A book of one name cannot be dollar-neutral (demeaning a single value gives
    zero), so `min_names` names must clear the hurdle or the book stands flat.
    That is a property of the strategy, not a guard: at high lambda it stands
    flat often, and that is what the `flat%` column reports.
    """
    N = edge_row.shape[0]
    w = np.zeros(N, dtype=np.float64)
    ok = np.isfinite(edge_row) & np.isfinite(rt_cost_row) & (rt_cost_row > 0)
    if not ok.any():
        return w, 0
    excess = np.abs(edge_row) - lam * rt_cost_row
    act = ok & (excess > 0)
    n_sel = int(act.sum())
    if n_sel < min_names:
        return w, n_sel
    w[act] = excess[act] * np.sign(edge_row[act])
    if size_by_cost:
        w[act] /= rt_cost_row[act]
    if risk_row is not None:
        # Equal-RISK rather than equal-dollar. Divide by the name's causal
        # trailing volatility so a name that moves twice as much is held half
        # as large. Non-finite or non-positive vol falls back to the selected
        # names' median rather than to 1.0: a hardcoded 1.0 is a silent
        # assertion that the name has unit volatility, which would make it the
        # LARGEST position in a book measured in bps.
        r = np.asarray(risk_row, dtype=np.float64)[act]
        good = np.isfinite(r) & (r > 0)
        if not good.any():
            # No usable vol for any selected name. Stand FLAT rather than fall
            # through unscaled: an unscaled bar inside a risk-scaled run mixes
            # two sizing rules into one reported number. Only the first two
            # sessions of train can reach this.
            return np.zeros(N, dtype=np.float64), n_sel
        r = np.where(good, r, np.median(r[good]))
        w[act] /= r
    w[act] -= w[act].mean()              # dollar-neutral
    gross = np.abs(w).sum()
    if gross <= 0:
        return np.zeros(N, dtype=np.float64), n_sel
    return w / gross, n_sel               # gross-normalised


def solve_weights(edge_row, price_row, adv, sigma, lam, capital, k, min_names=2,
                  iters=2, size_by_cost=True, risk_row=None,
                  spread_entry=None, spread_exit=None):
    """Weights and the per-side cost they were solved against. (w, n_sel, cost_side).

    There is a circularity to resolve: impact depends on the share count, the
    share count depends on the weight, and the weight depends on cost. It is a
    weak dependency -- impact is the smaller half of the cost at these sizes and
    enters under a square root -- so two passes from a zero-impact start are
    enough to converge it. Running one pass instead would price every name as
    if the book were infinitesimal, which is precisely the assumption that makes
    a backtest look tradeable and a live book not.
    """
    shares = np.zeros_like(price_row, dtype=np.float64)
    w, n_sel, cost_side = None, 0, None
    for _ in range(max(iters, 1)):
        cost_side = side_cost_bps(price_row, shares, adv, sigma, k, spread_entry)
        # THE HURDLE MUST USE THE ROUND TRIP THAT WILL ACTUALLY BE PAID. When
        # the two legs are struck at different times their spreads differ, and
        # doubling the CHEAP leg would admit names that cannot pay for their own
        # exit -- the exact error the round-trip hurdle exists to prevent. The
        # exit leg is priced at this bar's price and ADV: overnight the price
        # moves ~1% and ADV is a per-name constant, so the spread is the term
        # that actually differs.
        if spread_exit is None:
            rt = 2.0 * cost_side
        else:
            rt = cost_side + side_cost_bps(price_row, shares, adv, sigma, k, spread_exit)
        w, n_sel = book_weights(edge_row, rt, lam, min_names, size_by_cost,
                                risk_row=risk_row)
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = np.abs(w) * capital / np.where(price_row > 0, price_row, np.nan)
        shares = np.nan_to_num(shares, nan=0.0)
    return w, n_sel, cost_side


def rebalance_schedule(t0, t1, h, session_last_idx):
    """[(entry, exit)] in [t0, t1), holding h bars but never through a close.

    Non-overlapping: a position opened at t is held to min(t+h, session_last)
    and the next decision is taken there. The short stub at the end of each
    session is KEPT rather than skipped -- `flatten_at_session_close` forces
    that trade to exist, and dropping it would credit the book with a schedule
    the env cannot run. Sessions that straddle the split boundary are skipped
    whole, so no period is scored partly on the wrong split.
    """
    out = []
    t = int(t0)
    t1 = int(t1)
    while t < t1 - 1:
        sl = int(session_last_idx[t])
        if sl >= t1 or t >= sl:
            t = sl + 1
            continue
        e = min(t + h, sl)
        out.append((t, e))
        t = e
    return out


def overnight_schedule(t0, t1, day_id, session_last_idx, T):
    """[(entry, exit)] for the overnight hold: one period per session.

    `rebalance_schedule` cannot express this. It caps every hold at
    `session_last_idx` because `flatten_at_session_close` liquidates there, and
    at the last decision bar of a session its `t >= sl` branch fires and emits
    NOTHING -- which is precisely the trade being priced here.

    In the execution frame a period (t, e) fills at Px[t] and exits at Px[e],
    and Px[t] is raw open[t+1]. So the gap trade is (L-1, L): fill at open[L],
    exit at open[L+1]. One bet per name per session.

    A trade is emitted only when BOTH legs fall inside [t0, t1), so no period
    is scored partly on the wrong split -- the same rule rebalance_schedule
    applies to sessions straddling the boundary.
    """
    L = overnight_decision_bars(day_id, session_last_idx, T)
    return [(int(l - 1), int(l)) for l in L if l - 1 >= t0 and l < t1]


def run_book(edge, ret, P, adv, sigma, k, schedule, day_id, n_bars, lam,
             capital, min_names=2, size_by_cost=True, risk=None,
             spread_entry=None, spread_exit=None, carry_bps=0.0):
    """Walk the schedule once and return the period ledger plus totals.

    Friction is charged on TURNOVER, one side per unit traded, at the bar where
    the trade is placed -- the same accounting `pop_turnover_stats()` uses, so
    `alpha_per_turnover` and `cost_per_turnover` here mean what they mean in a
    training run. Entering a period pays the entry side; the exit side is paid
    by the next rebalance, or by the session-close flatten. Every round trip is
    therefore charged exactly once, and the final open book is liquidated so
    none is left unpaid.

    The HURDLE is stated against the ROUND TRIP (2 x per-side cost) because
    `edge` is what the name earns over a whole hold. A hurdle against one side
    would admit trades that cannot pay for their own exit.
    """
    N = edge.shape[1]
    prev_w = np.zeros(N, dtype=np.float64)
    prev_exit = None
    prev_day = None

    gross_s, cost_s, sel_s = [], [], []
    tot_turn = 0.0

    def liquidation_cost(w, bar):
        """Cost of closing `w` at `bar`, impact priced off the size being closed."""
        px = P[bar].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = np.nan_to_num(np.abs(w) * capital / np.where(px > 0, px, np.nan), nan=0.0)
        cs = np.nan_to_num(side_cost_bps(px, sh, adv, sigma, k, spread_exit), nan=0.0)
        return float(np.abs(w) @ cs)

    for (t, e) in schedule:
        if prev_day is not None and day_id[t] != prev_day and np.abs(prev_w).sum() > 0:
            # The env flattened at the previous session's close. Charge that
            # liquidation at the closing bar's prices, then start flat.
            cost_s[-1] += liquidation_cost(prev_w, prev_exit)
            tot_turn += float(np.abs(prev_w).sum())
            prev_w = np.zeros(N, dtype=np.float64)

        w, n_sel, cost_side = solve_weights(
            edge[t], P[t].astype(np.float64), adv, sigma, lam, capital, k,
            min_names, size_by_cost=size_by_cost,
            risk_row=(None if risk is None else risk[t]),
            spread_entry=spread_entry, spread_exit=spread_exit
        )
        dw = w - prev_w
        tot_turn += float(np.abs(dw).sum())

        r = np.nan_to_num(ret[t], nan=0.0)
        gross_s.append(float(w @ r))
        # Carry: borrow on the short leg and financing on the long, charged on
        # GROSS exposure for every night the position is held. Not a trading
        # cost -- it is paid for holding, which is exactly what this book newly
        # does and the intraday book never did.
        cost_s.append(float(np.abs(dw) @ np.nan_to_num(cost_side, nan=0.0))
                      + carry_bps * float(np.abs(w).sum()))
        sel_s.append(n_sel)

        # Drift: a +50 bps name is 0.5% more of the book at the exit than at
        # entry. Second-order at these magnitudes, carried anyway so turnover
        # is the number a rebalancer would actually place.
        prev_w = w * (1.0 + r / 1e4)
        prev_exit, prev_day = e, day_id[t]

    if prev_exit is not None and np.abs(prev_w).sum() > 0:
        cost_s[-1] += liquidation_cost(prev_w, prev_exit)
        tot_turn += float(np.abs(prev_w).sum())

    return dict(gross=np.asarray(gross_s), cost=np.asarray(cost_s),
                selected=np.asarray(sel_s), turnover=tot_turn, n_bars=int(n_bars))


def summarise(ledger, h):
    """Ledger -> the numbers P2 is judged on.

    `ratio` is alpha_per_turnover / cost_per_turnover: bps earned per dollar
    traded over bps paid per dollar traded. It is the one metric here that is
    directly comparable against a training run's `pop_turnover_stats()`, and
    it is scale-free -- it does not move when gross, account size or trade
    count move, which is exactly why P1 chose that pair over net worth.

    Sharpe is annualised from the PERIOD series, not per bar: the book takes
    one decision per period, so the period is the unit of independent risk.
    Overlapping holds would break that and there are none here by construction.
    """
    g, c, n_bars = ledger["gross"], ledger["cost"], ledger["n_bars"]
    net = g - c
    n_per = len(net)
    if n_per == 0:
        return {}
    years = n_bars / (BARS_PER_DAY * TRADING_DAYS)
    per_year = n_per / years if years > 0 else float("nan")
    turn = ledger["turnover"]

    active = ledger["selected"] >= 2
    active_n = int(active.sum())

    sd = float(net.std(ddof=1)) if n_per > 1 else 0.0
    sharpe = (float(net.mean()) / sd) * math.sqrt(per_year) if sd > 0 else float("nan")

    # THE ERROR BAR ON THAT SHARPE. Lo's standard error for an annualised
    # Sharpe over `years` of data: se ~ sqrt((1 + SR^2/2) / years). On a val
    # split under a year this lands near 1.1, which is larger than any Sharpe
    # this book has produced -- so the number is reported WITH it, because
    # quoting 0.62 without the +/-1.1 beside it is the single easiest way to
    # mistake a noise draw for a result.
    se = (math.sqrt((1.0 + 0.5 * sharpe ** 2) / years)
          if years > 0 and np.isfinite(sharpe) else float("nan"))

    # CONCENTRATION. A book whose whole year comes from five bars has not found
    # an edge, it has found five bars, and the distinction is invisible in
    # Sharpe, ratio and annual return alike. `top5_share` is those five periods'
    # contribution as a fraction of total net PnL; `sharpe_ex_top5` re-scores
    # the series with them removed. If the second number collapses, so does the
    # claim.
    top5_share, sharpe_ex5 = float("nan"), float("nan")
    if n_per > 5:
        order = np.argsort(net)[::-1]
        top5 = float(net[order[:5]].sum())
        total = float(net.sum())
        top5_share = top5 / total if abs(total) > 1e-12 else float("nan")
        rest = np.delete(net, order[:5])
        sd_r = float(rest.std(ddof=1))
        if sd_r > 0:
            sharpe_ex5 = (float(rest.mean()) / sd_r) * math.sqrt(per_year)

    alpha_pt = float(g.sum()) / turn if turn > 0 else float("nan")
    cost_pt = float(c.sum()) / turn if turn > 0 else float("nan")

    return dict(
        periods=n_per,
        periods_per_year=per_year,
        gross_bps=float(g.mean()),
        cost_bps=float(c.mean()),
        net_bps=float(net.mean()),
        sharpe=sharpe,
        sharpe_se=se,
        sharpe_t=(sharpe / se) if (se and np.isfinite(se) and se > 0) else float("nan"),
        sharpe_ex_top5=sharpe_ex5,
        top5_share=top5_share,
        years=years,
        annual_return=float(net.mean()) * per_year / 1e4,
        alpha_per_turnover=alpha_pt,
        cost_per_turnover=cost_pt,
        ratio=(alpha_pt / cost_pt) if (cost_pt and np.isfinite(cost_pt) and cost_pt > 0) else float("nan"),
        turnover_per_bar=turn / n_bars if n_bars else float("nan"),
        mean_names=float(ledger["selected"].mean()),
        flat_pct=100.0 * (1.0 - active_n / n_per),
        # Conditioned on the book being ON. A book that stands flat 99% of the
        # time has a 1% unconditional hit rate however good its trades are, and
        # reading that as "it is wrong 99 times in 100" is the obvious misread.
        mean_names_active=float(ledger["selected"][active].mean()) if active_n else float("nan"),
        hit_rate=100.0 * float((net[active] > 0).mean()) if active_n else float("nan"),
        active_periods=int(active_n),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

ROW = ("{lam:>6} {t_sharpe:>9} {t_ratio:>8} {t_ann:>8}   "
       "{v_sharpe:>9} {v_ratio:>8} {v_ann:>8} {v_names:>7} {v_flat:>7} {v_turn:>8}")


def _fmt(x, spec=".2f"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return format(x, spec)


def print_sweep(hold, rows, chosen):
    print()
    print(f"HOLD {hold} BARS ({hold * 5} min)")
    print("-" * 100)
    print(ROW.format(lam="lambda", t_sharpe="tr sharpe", t_ratio="tr ratio",
                     t_ann="tr ann%", v_sharpe="va sharpe", v_ratio="va ratio",
                     v_ann="va ann%", v_names="names", v_flat="flat%",
                     v_turn="turn/bar"))
    for r in rows:
        tr, va = r["train"], r["val"]
        mark = " <-- selected on train" if r["lam"] == chosen else ""
        print(ROW.format(
            lam=_fmt(r["lam"], ".2f"),
            t_sharpe=_fmt(tr.get("sharpe")),
            t_ratio=_fmt(tr.get("ratio")),
            t_ann=_fmt(100 * tr["annual_return"]) if tr.get("annual_return") is not None else "-",
            v_sharpe=_fmt(va.get("sharpe")),
            v_ratio=_fmt(va.get("ratio")),
            v_ann=_fmt(100 * va["annual_return"]) if va.get("annual_return") is not None else "-",
            v_names=_fmt(va.get("mean_names"), ".1f"),
            v_flat=_fmt(va.get("flat_pct"), ".1f"),
            v_turn=_fmt(va.get("turnover_per_bar"), ".4f"),
        ) + mark)


def print_detail(hold, lam, res):
    tr, va = res["train"], res["val"]
    print()
    print(f"SELECTED CELL -- hold {hold}, lambda {lam} (chosen on TRAIN)")
    print("-" * 100)
    print(f"{'':<26}{'train':>14}{'val':>14}")
    for label, key, spec in (
        ("gross bps / period", "gross_bps", ".3f"),
        ("cost bps / period", "cost_bps", ".3f"),
        ("net bps / period", "net_bps", ".3f"),
        ("alpha_per_turnover bps", "alpha_per_turnover", ".3f"),
        ("cost_per_turnover bps", "cost_per_turnover", ".3f"),
        ("ratio", "ratio", ".2f"),
        ("net Sharpe (annual)", "sharpe", ".2f"),
        ("  +/- standard error", "sharpe_se", ".2f"),
        ("  t (sharpe / se)", "sharpe_t", ".2f"),
        ("  Sharpe ex top-5 periods", "sharpe_ex_top5", ".2f"),
        ("  top-5 share of net PnL", "top5_share", ".2f"),
        ("years of data", "years", ".2f"),
        ("annual return @ gross 1", "annual_return", ".4f"),
        ("turnover / bar", "turnover_per_bar", ".4f"),
        ("names in book (all)", "mean_names", ".1f"),
        ("names in book (active)", "mean_names_active", ".1f"),
        ("flat %", "flat_pct", ".1f"),
        ("hit rate % (active only)", "hit_rate", ".1f"),
        ("active periods", "active_periods", ".0f"),
        ("periods", "periods", ".0f"),
    ):
        print(f"{label:<26}{_fmt(tr.get(key), spec):>14}{_fmt(va.get(key), spec):>14}")


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Cost-aware cross-sectional book.")
    ap.add_argument("--tickers", type=int, default=None, help="cap universe size")
    ap.add_argument("--tick-max-bps", type=float, default=None,
                    help="keep only names whose tick is below this many bps of "
                         "price, measured as the median over TRAIN. The liquid "
                         "tier: a $0.01 tick is 0.04 bps on the dearest name in "
                         "the panel and 2.5 bps on the cheapest, so this selects "
                         "on the dominant term of the cost model")
    ap.add_argument("--hold", type=int, action="append", default=None,
                    help="hold in bars; repeatable (default 24)")
    ap.add_argument("--lam", type=float, action="append", default=None,
                    help="lambda to sweep; repeatable (default 0..3)")
    ap.add_argument("--edge", choices=("ridge", "oracle", "npz", "reversal"), default="ridge")
    ap.add_argument("--edge-npz", type=str, default=None,
                    help="with --edge npz: the file written by "
                         "training/pretrain_trunk.py --dump-edge")
    ap.add_argument("--frame", choices=tuple(FRAME_COLUMNS), default="exec",
                    help="execution convention the book is filled and marked on; "
                         "see FRAME_COLUMNS. 'exec' is P1's frame (bar t+1 open); "
                         "'close' is close-to-close, which the env cannot trade; "
                         "the m1_*/m12_*/bar_vwap rows fill inside bar t+1 and "
                         "need an intrabar.py raw directory")
    ap.add_argument("--ridge-alpha", default=str(RIDGE_ALPHA),
                    help="ridge penalty for --edge ridge. A number, or 'auto' to "
                         "select one on a train-only inner holdout (see "
                         "select_ridge_alpha). Default 10.0 -- unchanged, so "
                         "previously recorded results stay reproducible")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                    help="dollars the book runs at gross 1 (default 1e6, the "
                         "100 x initial_cash the env models)")
    ap.add_argument("--min-names", type=int, default=2,
                    help="names that must clear the hurdle before the book trades")
    ap.add_argument("--variant", choices=("full", "nosize"), default="full",
                    help="'nosize' drops the / cost sizing term and keeps the "
                         "hurdle -- the ablation that says which half is working")
    ap.add_argument("--min-names-frac", type=float, default=0.0,
                    help="lambda selection floor on BREADTH: the train book must "
                         "average at least this fraction of the universe. The "
                         "existing --min-active-frac floor only asks whether the "
                         "book is ON, not whether it holds anything, so it admits "
                         "a book that trades every session and holds 1.7 names.")
    ap.add_argument("--min-active-frac", type=float, default=0.10,
                    help="a lambda whose TRAIN book stands flat more than "
                         "(1 - this) of the time is not a selection candidate; "
                         "see the note in main()")
    ap.add_argument("--open-spread-bps", type=float, default=None,
                    help="HALF-spread in bps charged on the leg struck at the 09:30 "
                         "print. Measured by eval/measure_open_spread.py "
                         "(Corwin-Schultz): full spread 2.93 bps at the open against "
                         "0.068 midday, so 1.46 here. Overnight mode only.")
    ap.add_argument("--close-spread-bps", type=float, default=None,
                    help="HALF-spread in bps on the leg struck at the 15:55 bar "
                         "(measured full spread 0.52 bps, so 0.26).")
    ap.add_argument("--carry-bps", type=float, default=0.0,
                    help="borrow + financing in bps of GROSS per night held. Not "
                         "measured -- there is no borrow data in this project -- so "
                         "it is an assumption and the sweep below reports its "
                         "breakeven. 0.20 corresponds to ~50 bps/yr on the short leg.")
    ap.add_argument("--risk-scale", choices=("none", "vol"), default="none",
                    help="vol: size by edge/(cost*trailing overnight vol) instead of "
                         "edge/cost, i.e. equal-risk rather than equal-dollar. "
                         "Overnight mode only; the vol is causal.")
    ap.add_argument("--risk-window", type=int, default=60,
                    help="sessions of trailing overnight vol for --risk-scale vol")
    ap.add_argument("--overnight", action="store_true",
                    help="price the OVERNIGHT hold instead of intraday: lift the "
                         "flatten_at_session_close cap and trade the single "
                         "session-close -> next-session-open transition, one bet "
                         "per name per session. Forces hold=1.")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    holds = args.hold or [DEFAULT_HOLD]
    if args.overnight:
        if args.hold:
            print("[overnight] --hold is ignored: the gap trade is one period, "
                  "session close to next session open, by definition.")
        holds = [1]
    lams = args.lam or list(DEFAULT_LAMBDAS)

    print("=" * 100)
    print("CROSS-SECTIONAL BOOK -- cost-aware sizing")
    print("=" * 100)

    k = env_cost_constants()
    cost_desc = (f"spread={k['spread_bps']} bps floored at half a {k['tick']:.2f} tick, "
                 f"+ half-tick adverse snap, commission={k['commission_bps']} bps, "
                 f"impact Y={k['impact_coef']} x sigma_daily x sqrt(Q/ADV)  "
                 f"[training.config.EnvConfig]")
    print(f"[cost] {cost_desc}")

    panel = load_panel(args.tickers)
    X, P = panel["X"], panel["P"]
    features, tickers = panel["features"], panel["tickers"]
    day_id, sli = panel["day_id"], panel["session_last_idx"]
    T, N = P.shape

    i_train = int(T * TRAIN_FRAC)
    i_val = int(T * (TRAIN_FRAC + VAL_FRAC))
    print(f"[split] train 0:{i_train}  val {i_train}:{i_val}  test {i_val}:{T} (untouched)")

    # Tier the universe on tick-in-bps, measured on TRAIN only -- selecting on
    # a median that includes validation prices would be one bit of lookahead
    # per name, and a name that got dearer over the sample would be admitted
    # on information the book does not have at decision time.
    if args.tick_max_bps is not None:
        tick = env_cost_constants()["tick"]
        with np.errstate(divide="ignore", invalid="ignore"):
            tick_bps = 1e4 * tick / np.nanmedian(np.where(P[:i_train] > 0, P[:i_train], np.nan), axis=0)
        keep = np.isfinite(tick_bps) & (tick_bps < args.tick_max_bps)
        if keep.sum() < 2:
            raise SystemExit(f"--tick-max-bps {args.tick_max_bps} leaves {int(keep.sum())} "
                             "name(s); a dollar-neutral book needs at least 2")
        X, P = X[:, keep, :], P[:, keep]
        tickers = [t for t, k in zip(tickers, keep) if k]
        N = P.shape[1]
        print(f"[tier] tick < {args.tick_max_bps} bps on train: {N} of {len(keep)} names "
              f"(median tick {np.nanmedian(tick_bps[keep]):.3f} bps)")

    # THE PRICES THE BOOK IS SCORED AND FILLED ON. `P` (close) still feeds the
    # liquidity measurement and the panel's own guards; `Px` is what returns and
    # costs are computed against.
    frame_col, frame_desc = FRAME_COLUMNS[args.frame]
    if frame_col is None:
        Px, sli_x = P, sli
    else:
        Px, sli_x = execution_frame(panel["index"], tickers, sli, column=frame_col)
    print(f"[frame] {args.frame}: {frame_desc}")

    adv, sigma = measure_liquidity(panel["index"], tickers, day_id, i_train, P)
    print(f"[liq] train-measured: median ADV {np.median(adv):,.0f} shares, "
          f"median sigma_daily {np.median(sigma) * 100:.2f}%")

    # Per-name, per-bar cost at an equal-weight reference size. This is the
    # whole point of the exercise: cost is a [T, N] object, never a scalar.
    ref_sh = (args.capital / max(N, 1)) / np.where(Px > 0, Px, np.nan)
    cs_ref = side_cost_bps(Px, np.nan_to_num(ref_sh, nan=0.0), adv, sigma, k)
    fin = np.isfinite(cs_ref)
    lo, hi = np.percentile(cs_ref[fin], 1), np.percentile(cs_ref[fin], 99)
    print(f"[cost] per-side bps at equal weight on ${args.capital:,.0f}: "
          f"p10 {np.percentile(cs_ref[fin], 10):.3f}  median {np.median(cs_ref[fin]):.3f}  "
          f"p90 {np.percentile(cs_ref[fin], 90):.3f}  "
          f"(p99/p1 = {hi / max(lo, 1e-9):.1f}x -- the dispersion the sizing trades on)")
    print(f"[edge] estimator: {args.edge}    [variant] {args.variant}"
          + ("  (no / cost sizing -- ablation)" if args.variant == "nosize" else ""))

    out = {"cost_desc": cost_desc, "edge_kind": args.edge, "n_tickers": N,
           "capital": args.capital, "variant": args.variant,
           "frame": args.frame, "holds": {}}

    for h in holds:
        if args.overnight:
            # NO SESSION CAP -- the cap is exactly what is being lifted. The
            # uncapped 1-bar exec return at index L-1 is log(open[L+1]/open[L]),
            # the gap itself.
            #
            # Then MASK to the decision bars. Leaving the other 77 bars in would
            # hand build_edge a regression whose target is a 5-minute return on
            # 98.7% of its rows and a 17-hour one on the rest, and the fit would
            # be dominated by the horizon that is not being traded. The plan is
            # explicit: fit the edge on the OVERNIGHT target specifically,
            # because a different horizon has a different conditional mean.
            fwd = forward_return_bps(Px, 1, None)
            keep = np.zeros(fwd.shape[0], dtype=bool)
            keep[overnight_decision_bars(day_id, sli, T) - 1] = True
            fwd = np.where(keep[:, None], fwd, np.nan).astype(np.float32)
        else:
            fwd = forward_return_bps(Px, h, sli_x)
        ridge_alpha = (args.ridge_alpha if args.ridge_alpha == "auto"
                       else float(args.ridge_alpha))
        if args.edge == "reversal":
            edge, edge_desc = reversal_edge(P, fwd, i_train, day_id)
        else:
            edge, edge_desc = build_edge(args.edge, X, fwd, i_train, features,
                                         ridge_alpha=ridge_alpha,
                                         edge_npz=args.edge_npz,
                                         index=panel["index"], tickers=tickers)
        print()
        print(f"[edge] hold {h}: {edge_desc}")

        # Is the edge real at all? Block-clustered IC on val, blocks sized to
        # the horizon so overlapping outcomes cannot land in different
        # clusters. Printed, never used for selection.
        tgt = cross_sectional_demean(fwd)
        block_days = int(math.ceil(h / BARS_PER_DAY)) + 1
        va_rows = np.arange(i_train, i_val)
        blk = np.repeat(day_id[va_rows] // block_days, N)
        ic, tstat, nblk, nobs = block_ic(edge[va_rows].ravel(), tgt[va_rows].ravel(), blk)
        print(f"[edge] val block IC {ic:+.5f}  t {tstat:+.2f}  over {nblk} blocks, {nobs:,} obs")

        # THE FEASIBILITY LINE. Read this before the sweep table below it.
        #
        # The hurdle admits a name when |edge| > lambda * round trip, so the
        # largest lambda that can EVER admit anything is max|edge| divided by
        # the cheapest round trip in the panel. If that number is below 1, then
        # "lambda = 1" does not describe a selective book -- it describes a book
        # that never trades, and every Sharpe printed at or above it is computed
        # on an empty sample. No amount of sweeping recovers from an edge
        # distribution whose tail sits under the cost distribution's middle.
        ae = np.abs(edge[np.isfinite(edge)])
        rt = 2.0 * cs_ref[fin]
        # Stated at the 99.9th percentile of |edge| against the MEDIAN round
        # trip, not at the maxima: a handful of extreme ridge predictions
        # (max 219 bps against a p99.9 of 3.3 here) would otherwise report a
        # feasible lambda of 351 on an edge that in practice clears nothing.
        lam_tail = float(np.percentile(ae, 99.9) / max(np.median(rt), 1e-9))
        print(f"[hurdle] |edge| p50 {np.percentile(ae, 50):.3f}  p90 {np.percentile(ae, 90):.3f}  "
              f"p99.9 {np.percentile(ae, 99.9):.3f} bps")
        print(f"[hurdle] round trip p10 {np.percentile(rt, 10):.3f}  median {np.median(rt):.3f} bps"
              f"  ->  even the top 0.1% of names stop clearing a median-cost "
              f"round trip at lambda {lam_tail:.2f}"
              )
        if lam_tail < 1.0:
            print("[hurdle] lambda >= 1 IS UNREACHABLE ON THIS EDGE -- every row at "
                  "or above it is an empty book, not a selective one.")

        # Per-leg spreads. The overnight round trip is struck at two different
        # moments with very different liquidity, and the book had been charging
        # the cheaper one twice.
        sp_entry = args.close_spread_bps
        sp_exit = args.open_spread_bps
        if (sp_entry is not None or sp_exit is not None) and not args.overnight:
            raise SystemExit("--open-spread-bps/--close-spread-bps describe the "
                             "overnight round trip's two legs; they are meaningless "
                             "without --overnight.")
        if args.overnight and (sp_entry is not None or sp_exit is not None
                               or args.carry_bps):
            print(f"[cost] overnight legs: entry half-spread "
                  f"{'env default' if sp_entry is None else f'{sp_entry:.3f} bps'}, "
                  f"exit half-spread "
                  f"{'env default' if sp_exit is None else f'{sp_exit:.3f} bps'}, "
                  f"carry {args.carry_bps:.3f} bps of gross/night")

        risk = None
        if args.risk_scale == "vol":
            if not args.overnight:
                raise SystemExit("--risk-scale vol is defined on the overnight "
                                 "schedule only; the intraday book would need a "
                                 "5-minute vol estimate, which is a different object.")
            risk = trailing_overnight_vol(fwd, day_id, sli, T, window=args.risk_window)
            fin = risk[np.isfinite(risk)]
            print(f"[risk] equal-risk sizing on {args.risk_window}-session trailing "
                  f"overnight vol (causal): median {np.median(fin):.0f} bps, "
                  f"p10 {np.percentile(fin, 10):.0f}  p90 {np.percentile(fin, 90):.0f} "
                  f"({np.percentile(fin, 90) / max(np.percentile(fin, 10), 1e-9):.1f}x "
                  f"spread -- the dispersion the scaling acts on)")

        if args.overnight:
            sched_tr = overnight_schedule(0, i_train, day_id, sli, T)
            sched_va = overnight_schedule(i_train, i_val, day_id, sli, T)
            print(f"[overnight] {len(sched_tr)} train / {len(sched_va)} val gap "
                  f"periods -- one per session, flatten_at_session_close lifted")
        else:
            sched_tr = rebalance_schedule(0, i_train, h, sli_x)
            sched_va = rebalance_schedule(i_train, i_val, h, sli_x)

        by_cost = args.variant == "full"
        rows = []
        for lam in lams:
            tr = summarise(run_book(edge, fwd, Px, adv, sigma, k, sched_tr, day_id,
                                    i_train, lam, args.capital, args.min_names,
                                    size_by_cost=by_cost, risk=risk,
                                    spread_entry=sp_entry, spread_exit=sp_exit,
                                    carry_bps=args.carry_bps), h)
            va = summarise(run_book(edge, fwd, Px, adv, sigma, k, sched_va, day_id,
                                    i_val - i_train, lam, args.capital, args.min_names,
                                    size_by_cost=by_cost, risk=risk,
                                    spread_entry=sp_entry, spread_exit=sp_exit,
                                    carry_bps=args.carry_bps), h)
            rows.append({"lam": lam, "train": tr, "val": va})

        # SELECTION IS ON TRAIN, AND IT NEEDS A FLOOR.
        #
        # Maximising train Sharpe over lambda has a degenerate corner: as lambda
        # rises the book trades less, and a book that is flat 99.9% of the time
        # has a return series that is almost all exact zeros with a handful of
        # nonzero periods. Its standard deviation collapses faster than its mean
        # does, so Sharpe RISES as the strategy disappears -- measured here, an
        # unfloored sweep picked lambda = 3.0 at hold 6 with turnover 0.0000 and
        # a 100% flat book, and reported Sharpe 0.64 for it. That is not a
        # strategy, it is an empty sample with a small denominator.
        #
        # The floor requires the train book to be ON for at least
        # `min_active_frac` of its periods before its Sharpe is allowed to
        # compete. This is a constraint on the SEARCH, not on the strategy, and
        # it is set on train only, so it grants no val information.
        min_active = args.min_active_frac
        # BREADTH FLOOR. `min_active_frac` asks whether the book is ON; it does
        # not ask whether it holds anything. Measured on the overnight book with
        # all three flatteries corrected: the train-selected lambda produced 1.7
        # names at 71% flat on fold 1 and 2.3 names on fold 2, and those two
        # folds are what lifted the mean ratio from 0.69 (full breadth) to 2.40.
        # A 1.7-name dollar-neutral book is not a selective strategy, it is a
        # handful of bets, and this project has already been burned twice by
        # exactly that corner -- see the lambda=1 row in AGENTS.md that swung
        # from -0.329 to +0.245 between two runs of identical code.
        min_names_n = args.min_names_frac * N
        viable = [r for r in rows
                  if r["train"] and np.isfinite(r["train"].get("sharpe", np.nan))
                  and r["train"].get("turnover_per_bar", 0) > 0
                  and (100.0 - r["train"].get("flat_pct", 100.0)) / 100.0 >= min_active
                  and r["train"].get("mean_names", 0.0) >= min_names_n]
        chosen = max(viable, key=lambda r: r["train"]["sharpe"])["lam"] if viable else None
        if chosen is None:
            print(f"  [select] no lambda kept the train book active for "
                  f"{min_active:.0%} of its periods and holding "
                  f"{min_names_n:.0f}+ names -- nothing selected.")

        print_sweep(h, rows, chosen)
        if chosen is not None:
            sel = next(r for r in rows if r["lam"] == chosen)
            print_detail(h, chosen, sel)

            # THE DEGENERATE CORNER, MADE LOUD.
            #
            # `--min-names-frac` defaults to 0 so that every number already in
            # the record reproduces. That leaves the corner reachable, and it is
            # not a hypothetical: on the corrected overnight book the
            # train-selected lambda produced a 1.7-name dollar-neutral book,
            # flat 71% of the time, whose top five sessions carried 555% of its
            # net PnL -- and it posted ratio 2.93, the best number in the study.
            # Every breadth floor from 5% up puts the same configuration at
            # 0.96-1.53. So this warning is not fastidiousness; it separates the
            # only "pass" this project has produced from the truth.
            tr_names = sel["train"].get("mean_names", 0.0)
            va_names = sel["val"].get("mean_names", 0.0)
            t5 = abs(sel["val"].get("top5_share", 0.0))
            thin = tr_names < max(0.05 * N, 5.0)
            if thin or t5 > 1.0:
                print()
                print("*" * 92)
                if thin:
                    print(f"** DEGENERATE BOOK: the selected lambda holds {tr_names:.1f} "
                          f"train / {va_names:.1f} val names out of {N}.")
                    print("** A book this thin is a handful of bets, not a selective "
                          "strategy, and its Sharpe")
                    print("** and ratio are computed on an almost-empty sample. Re-run "
                          "with --min-names-frac.")
                if t5 > 1.0:
                    print(f"** TAIL-CARRIED: the top 5 periods account for "
                          f"{t5:.0%} of net PnL on val.")
                    print("** Above 100% the rest of the sample is a net LOSS and the "
                          "headline is five sessions.")
                print("*" * 92)
        out["holds"][str(h)] = {"edge_desc": edge_desc, "val_ic": ic, "val_t": tstat,
                                "chosen_lambda": chosen, "rows": rows}

    print()
    print("=" * 100)
    print("Read `ratio` first: it is alpha_per_turnover / cost_per_turnover, the same")
    print("pair a training run logs. Above 1 the book pays for itself; the P2 target is 3.")
    print("Everything here is a CEILING -- no order sizing, no partial fills, no Kelly.")
    print("=" * 100)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
