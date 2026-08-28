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
from eval.alpha_lab import (  # noqa: E402
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


def side_cost_bps(price, shares, adv, sigma_daily, k):
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
    half_spread = np.maximum(k["spread_bps"], half_tick_bps)
    part = np.clip(np.asarray(shares, dtype=np.float64) / adv, 0.0, 1.0)
    impact_bps = 1e4 * k["impact_coef"] * sigma_daily * np.sqrt(part)
    return half_spread + half_tick_bps + k["commission_bps"] + impact_bps


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
        col = _pd.read_parquet(rp)[column]
        col = col.mask(col <= 0).reindex(index).ffill().bfill()
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


def build_edge(kind, X, fwd, i_train, features, edge_npz=None, index=None, tickers=None):
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
    beta, n_used = ridge_fit_chunked(
        Xcs[:i_train].reshape(-1, F), target[:i_train].ravel()
    )
    if beta is None:
        raise SystemExit("ridge fit failed -- not enough finite rows in train")
    T, N = fwd.shape
    edge = (Xcs.reshape(-1, F) @ beta[:-1] + beta[-1]).reshape(T, N).astype(np.float32)
    desc = (f"ridge:xsectional over {F} features, fit on {n_used:,} train rows "
            f"(ridge alpha={RIDGE_ALPHA})")
    return edge, desc


# ---------------------------------------------------------------------------
# The book
# ---------------------------------------------------------------------------

def book_weights(edge_row, rt_cost_row, lam, min_names=2, size_by_cost=True):
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
    w[act] -= w[act].mean()              # dollar-neutral
    gross = np.abs(w).sum()
    if gross <= 0:
        return np.zeros(N, dtype=np.float64), n_sel
    return w / gross, n_sel               # gross-normalised


def solve_weights(edge_row, price_row, adv, sigma, lam, capital, k, min_names=2,
                  iters=2, size_by_cost=True):
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
        cost_side = side_cost_bps(price_row, shares, adv, sigma, k)
        w, n_sel = book_weights(edge_row, 2.0 * cost_side, lam, min_names, size_by_cost)
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


def run_book(edge, ret, P, adv, sigma, k, schedule, day_id, n_bars, lam,
             capital, min_names=2, size_by_cost=True):
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
        cs = np.nan_to_num(side_cost_bps(px, sh, adv, sigma, k), nan=0.0)
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
            min_names, size_by_cost=size_by_cost
        )
        dw = w - prev_w
        tot_turn += float(np.abs(dw).sum())

        r = np.nan_to_num(ret[t], nan=0.0)
        gross_s.append(float(w @ r))
        cost_s.append(float(np.abs(dw) @ np.nan_to_num(cost_side, nan=0.0)))
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

    alpha_pt = float(g.sum()) / turn if turn > 0 else float("nan")
    cost_pt = float(c.sum()) / turn if turn > 0 else float("nan")

    return dict(
        periods=n_per,
        periods_per_year=per_year,
        gross_bps=float(g.mean()),
        cost_bps=float(c.mean()),
        net_bps=float(net.mean()),
        sharpe=sharpe,
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
    ap.add_argument("--hold", type=int, action="append", default=None,
                    help="hold in bars; repeatable (default 24)")
    ap.add_argument("--lam", type=float, action="append", default=None,
                    help="lambda to sweep; repeatable (default 0..3)")
    ap.add_argument("--edge", choices=("ridge", "oracle", "npz"), default="ridge")
    ap.add_argument("--edge-npz", type=str, default=None,
                    help="with --edge npz: the file written by "
                         "training/pretrain_trunk.py --dump-edge")
    ap.add_argument("--frame", choices=("exec", "close"), default="exec",
                    help="'exec' scores the book on the prices the env fills at "
                         "(bar t+1 open, P1's execution frame); 'close' scores "
                         "close-to-close, which the env cannot trade")
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL,
                    help="dollars the book runs at gross 1 (default 1e6, the "
                         "100 x initial_cash the env models)")
    ap.add_argument("--min-names", type=int, default=2,
                    help="names that must clear the hurdle before the book trades")
    ap.add_argument("--variant", choices=("full", "nosize"), default="full",
                    help="'nosize' drops the / cost sizing term and keeps the "
                         "hurdle -- the ablation that says which half is working")
    ap.add_argument("--min-active-frac", type=float, default=0.10,
                    help="a lambda whose TRAIN book stands flat more than "
                         "(1 - this) of the time is not a selection candidate; "
                         "see the note in main()")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    holds = args.hold or [DEFAULT_HOLD]
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

    # THE PRICES THE BOOK IS SCORED AND FILLED ON. `P` (close) still feeds the
    # liquidity measurement and the panel's own guards; `Px` is what returns and
    # costs are computed against.
    if args.frame == "exec":
        Px, sli_x = execution_frame(panel["index"], tickers, sli)
        print("[frame] execution frame: fills and marks at bar t+1 open (P1)")
    else:
        Px, sli_x = P, sli
        print("[frame] CLOSE frame -- the env cannot trade this; comparison only")

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
        fwd = forward_return_bps(Px, h, sli_x)
        edge, edge_desc = build_edge(args.edge, X, fwd, i_train, features,
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

        sched_tr = rebalance_schedule(0, i_train, h, sli_x)
        sched_va = rebalance_schedule(i_train, i_val, h, sli_x)

        by_cost = args.variant == "full"
        rows = []
        for lam in lams:
            tr = summarise(run_book(edge, fwd, Px, adv, sigma, k, sched_tr, day_id,
                                    i_train, lam, args.capital, args.min_names,
                                    size_by_cost=by_cost), h)
            va = summarise(run_book(edge, fwd, Px, adv, sigma, k, sched_va, day_id,
                                    i_val - i_train, lam, args.capital, args.min_names,
                                    size_by_cost=by_cost), h)
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
        viable = [r for r in rows
                  if r["train"] and np.isfinite(r["train"].get("sharpe", np.nan))
                  and r["train"].get("turnover_per_bar", 0) > 0
                  and (100.0 - r["train"].get("flat_pct", 100.0)) / 100.0 >= min_active]
        chosen = max(viable, key=lambda r: r["train"]["sharpe"])["lam"] if viable else None
        if chosen is None:
            print(f"  [select] no lambda kept the train book active for "
                  f"{min_active:.0%} of its periods -- nothing selected.")

        print_sweep(h, rows, chosen)
        if chosen is not None:
            sel = next(r for r in rows if r["lam"] == chosen)
            print_detail(h, chosen, sel)
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
