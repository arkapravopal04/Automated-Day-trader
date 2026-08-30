"""
eval/on_regime.py -- is the overnight book's fold-to-fold spread a REGIME?

Short-horizon cross-sectional reversal is supposed to lose when the
cross-section trends and persists, and fold 2's val window (2023-09 -> 2024-04)
spans exactly such a stretch. This script tests that on its own terms.

WHAT IT PRODUCES, AND WHAT IT DOES NOT
--------------------------------------
It produces a CHARACTERISED EXPOSURE: how the book's session P&L covaries with
a market state that is knowable before the session opens. It does NOT produce a
filter, a threshold, or a rule, and nothing in it selects anything. With five
folds and a few hundred sessions per marker, fitting a regime gate is the single
most efficient way to overfit this project, and the value of this measurement is
precisely that it is not one.

THE MARKERS -- all causal, none fitted
--------------------------------------
Every marker at session s reads sessions [s-W, s-1] and never s. The window is
W = 60 sessions, which is NOT chosen here: it is `--risk-window`, the constant
the book's own vol scaling already runs on. Picking W to separate the folds
would be selection on the outcome, and the whole point is to avoid that.

  xs_ac1   Lag-1 CROSS-SECTIONAL autocorrelation of overnight gaps: per session,
           corr across names between the demeaned gap and the previous session's
           demeaned gap, averaged over the trailing window. NEGATIVE means the
           cross-section reverses (reversal's home turf); POSITIVE means it
           persists. This is a property of the market, not of the signal, so it
           is not circular.

  trend    Trailing cumulative equal-weight market return over W sessions,
           divided by its own trailing vol -- a signed trend strength, in
           units of standard deviations of the mean daily move.

  disp     Trailing mean cross-sectional dispersion of overnight gaps. Reversal
           needs something to revert; dispersion is how much there is.

  ic_trail Trailing realised IC of the book's OWN reversal signal. This one IS
           circular -- it is the strategy's recent performance wearing a
           different name -- and it is reported for exactly one question:
           whether the edge's failure is autocorrelated enough that it could be
           detected in time. A NO here is the strongest argument against every
           regime gate anyone might propose, so it is worth measuring even
           though it can never be a feature.

TWO TESTS, ONE OF WHICH HAS NO POWER
------------------------------------
  (A) EX ANTE, ACROSS FOLDS. The marker at each fold's last TRAIN session --
      the last value knowable before its val window opens -- against that
      fold's realised val Sharpe. n = 5. Five points cannot establish anything;
      it is printed because "does it separate the folds ex ante" is the question
      that was asked, and the honest answer to it is a rank correlation with no
      confidence interval.

  (B) POOLED, WITHIN VAL. Every val session of every fold, pooled, with each
      session's net P&L against the marker known the evening before. The five
      val windows are disjoint and contiguous by construction (TRAIN_FRAC
      0.40..0.80, VAL_FRAC 0.10), so this is ~740 non-overlapping sessions and
      no session is counted twice. This is the test with power, and it is the
      one the exposure claim rests on.
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

from eval.alpha_lab import (  # noqa: E402
    cross_sectional_demean,
    forward_return_bps,
    load_panel,
    overnight_decision_bars,
)
from eval.xsec_book import execution_frame, reversal_edge  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def trailing(x, w):
    """Mean of the w values STRICTLY BEFORE each position. shift(1) then roll."""
    s = pd.Series(x, dtype="float64").shift(1)
    return s.rolling(window=w, min_periods=max(w // 3, 5)).mean().to_numpy()


def spearman(a, b):
    """Rank correlation on the pairwise-finite rows. -> (rho, n)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Regime exposure of the overnight book.")
    ap.add_argument("--window", type=int, default=60,
                    help="trailing sessions per marker; 60 = the book's own risk window")
    ap.add_argument("--decomp-dir", type=str, default="logs/p3/on/decomp",
                    help="where on_fold_decomp.py wrote fN_du.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    W = args.window

    print("=" * 100)
    print(f"OVERNIGHT REGIME EXPOSURE -- trailing window {W} sessions, causal, unfitted")
    print("=" * 100)

    panel = load_panel(None)
    P, tickers = panel["P"], panel["tickers"]
    day_id, sli, index = panel["day_id"], panel["session_last_idx"], panel["index"]
    T, N = P.shape

    Px, _ = execution_frame(index, tickers, sli, column="open")
    fwd = forward_return_bps(Px, 1, None)
    L = overnight_decision_bars(day_id, sli, T)
    dec = L - 1
    keep = np.zeros(T, dtype=bool)
    keep[dec] = True
    fwd_m = np.where(keep[:, None], fwd, np.nan).astype(np.float32)

    # The gap matrix, one row per session. Everything below is built from it.
    R = fwd[dec]                                          # [S, N] bps
    S = R.shape[0]
    sess_date = pd.DatetimeIndex(index[dec]).tz_convert("America/New_York").normalize()
    print(f"[sessions] {S} overnight gaps, {str(sess_date[0])[:10]} -> {str(sess_date[-1])[:10]}")

    Rd = R - np.nanmean(R, axis=1, keepdims=True)         # cross-sectionally demeaned

    # --- xs_ac1: per-session cross-sectional lag-1 autocorrelation -----------
    ac = np.full(S, np.nan)
    for s in range(1, S):
        a, b = Rd[s], Rd[s - 1]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 20 and a[ok].std() > 0 and b[ok].std() > 0:
            ac[s] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    xs_ac1 = trailing(ac, W)

    # --- trend: equal-weight market, close to close, in vol units -----------
    C = P[L]                                              # session close prices
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = np.log(C[1:] / C[:-1]) * 1e4
    mkt = np.r_[np.nan, np.nanmean(dr, axis=1)]           # [S] daily market return, bps
    ms = pd.Series(mkt).shift(1)
    mu = ms.rolling(W, min_periods=max(W // 3, 5)).mean().to_numpy()
    sg = ms.rolling(W, min_periods=max(W // 3, 5)).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        trend = np.where(sg > 0, mu * math.sqrt(W) / sg, np.nan)

    # --- disp: trailing cross-sectional dispersion of the gaps --------------
    disp = trailing(np.nanstd(Rd, axis=1), W)

    # --- ic_trail: trailing realised IC of the book's own signal ------------
    # Circular by construction. Fitted on the FULL sample only because it is
    # never used as an input to anything -- see the module docstring.
    edge, _ = reversal_edge(P, fwd_m, T, day_id)
    E = edge[dec]
    tgt = cross_sectional_demean(fwd_m)[dec]
    ic_s = np.full(S, np.nan)
    for s in range(S):
        a, b = E[s], tgt[s]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 20 and a[ok].std() > 0 and b[ok].std() > 0:
            ic_s[s] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    ic_trail = trailing(ic_s, W)

    markers = {"xs_ac1": xs_ac1, "trend": trend, "disp": disp, "ic_trail": ic_trail}
    date_to_row = {str(d)[:10]: i for i, d in enumerate(sess_date)}

    print()
    print("[markers] full-sample distribution")
    print(f"{'marker':<10}{'p10':>10}{'median':>10}{'p90':>10}{'finite':>9}")
    for nm, v in markers.items():
        f = v[np.isfinite(v)]
        print(f"{nm:<10}{np.percentile(f, 10):>10.4f}{np.median(f):>10.4f}"
              f"{np.percentile(f, 90):>10.4f}{f.size:>9}")

    # ---------------- load the folds ----------------
    folds = []
    for k in range(1, args.folds + 1):
        p = os.path.join(args.decomp_dir, f"f{k}_du.json")
        if not os.path.exists(p):
            print(f"[warn] missing {p} -- fold {k} skipped")
            continue
        d = json.load(open(p))
        rows = [date_to_row.get(x) for x in d["session_dates"]]
        folds.append(dict(k=k, sharpe=d["sharpe"], net=np.asarray(d["net_per_session"]),
                          rows=np.asarray([-1 if r is None else r for r in rows]),
                          dates=d["session_dates"]))
    if not folds:
        raise SystemExit("no decomposition jsons found; run run_overnight_decomp.sh first")

    # ---------------- (A) ex ante, across folds ----------------
    print()
    print("-" * 100)
    print("(A) EX ANTE ACROSS FOLDS -- marker at the last TRAIN session, before val opens")
    print("-" * 100)
    print(f"{'fold':<6}{'val window':<26}{'val Sharpe':>12}"
          + "".join(f"{nm:>11}" for nm in markers))
    ex_ante = {nm: [] for nm in markers}
    sharpes = []
    for f in folds:
        r0 = int(f["rows"][f["rows"] >= 0][0])
        last_train = max(r0 - 1, 0)
        sharpes.append(f["sharpe"])
        vals = []
        for nm, v in markers.items():
            x = float(v[last_train])
            ex_ante[nm].append(x)
            vals.append(x)
        print(f"{f['k']:<6}{f['dates'][0] + ' -> ' + f['dates'][-1]:<26}{f['sharpe']:>12.2f}"
              + "".join(f"{x:>11.4f}" for x in vals))
    print()
    print("rank correlation against val Sharpe (n=5 -- DESCRIPTIVE, no power, no CI):")
    for nm in markers:
        rho, n = spearman(ex_ante[nm], sharpes)
        print(f"  {nm:<10} rho {rho:+.3f}  (n={n})")

    # ---------------- (B) pooled, within val ----------------
    all_net, all_rows, all_fold = [], [], []
    for f in folds:
        ok = f["rows"] >= 0
        all_net.append(f["net"][ok])
        all_rows.append(f["rows"][ok])
        all_fold.append(np.full(int(ok.sum()), f["k"]))
    net = np.concatenate(all_net)
    rows = np.concatenate(all_rows)
    fold_id = np.concatenate(all_fold)
    assert len(np.unique(rows)) == len(rows), "val windows overlap -- pooling would double-count"

    print()
    print("-" * 100)
    print(f"(B) POOLED WITHIN VAL -- {len(net)} disjoint sessions across {len(folds)} folds")
    print("-" * 100)
    print("net P&L of the session against the marker known the evening before:")
    print(f"{'marker':<10}{'spearman':>10}{'n':>7}   terciles of the marker -> mean net bps")
    out_rows = {}
    for nm, v in markers.items():
        x = v[rows]
        rho, n = spearman(x, net)
        ok = np.isfinite(x)
        q = np.quantile(x[ok], [1 / 3, 2 / 3])
        buckets = []
        for lo, hi in ((-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)):
            m = ok & (x > lo) & (x <= hi)
            buckets.append(float(net[m].mean()) if m.any() else float("nan"))
        print(f"{nm:<10}{rho:>+10.3f}{n:>7}   low {buckets[0]:+8.2f}   "
              f"mid {buckets[1]:+8.2f}   high {buckets[2]:+8.2f}")
        out_rows[nm] = dict(spearman=rho, n=n, terciles=buckets)

    # ---------------- (B2) pooled, at MONTH scale ----------------
    #
    # (B) asks whether the marker predicts the NEXT SESSION. The regime claim is
    # about a seven-month stretch, and a slow exposure can be real at that
    # timescale while invisible session by session. So the same pooled test is
    # repeated on calendar months: ~37 non-overlapping months, which is more
    # than five folds and fewer than 743 sessions -- the timescale the
    # hypothesis is actually stated at.
    #
    # This is a change of TIMESCALE, not a second attempt at the same test with
    # a freer parameter. The marker window stays 60.
    m_key = pd.PeriodIndex(sess_date[rows], freq="M")
    print()
    print("-" * 100)
    print(f"(B2) POOLED AT MONTH SCALE -- {len(m_key.unique())} months, the timescale the claim is stated at")
    print("-" * 100)
    mo_net, mo_mark = [], {nm: [] for nm in markers}
    for m in m_key.unique():
        sel = np.asarray(m_key == m)
        mo_net.append(float(net[sel].mean()))
        for nm, v in markers.items():
            mo_mark[nm].append(float(np.nanmean(v[rows[sel]])))
    mo_net = np.asarray(mo_net)
    print(f"{'marker':<10}{'spearman':>10}{'n':>7}   terciles of the marker -> mean net bps")
    for nm in markers:
        x = np.asarray(mo_mark[nm])
        rho, n = spearman(x, mo_net)
        ok = np.isfinite(x)
        q = np.quantile(x[ok], [1 / 3, 2 / 3])
        b = []
        for lo, hi in ((-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)):
            m2 = ok & (x > lo) & (x <= hi)
            b.append(float(mo_net[m2].mean()) if m2.any() else float("nan"))
        print(f"{nm:<10}{rho:>+10.3f}{n:>7}   low {b[0]:+8.2f}   mid {b[1]:+8.2f}   "
              f"high {b[2]:+8.2f}")
        out_rows[nm]["month_spearman"] = rho
        out_rows[nm]["month_terciles"] = b
        out_rows[nm]["month_n"] = n

    # The one number that decides whether ANY regime gate is buildable here: if
    # the trailing realised IC does not predict the next session's IC, then the
    # edge's failure is not detectable in time and a gate is fitting noise.
    print()
    print("-" * 100)
    print("IS A GATE EVEN BUILDABLE? persistence of the edge's own IC")
    print("-" * 100)
    ok = np.isfinite(ic_trail) & np.isfinite(ic_s)
    rho_ic, n_ic = spearman(ic_trail[ok], ic_s[ok])
    print(f"  trailing-{W} IC  ->  next session's IC:   spearman {rho_ic:+.3f}  (n={n_ic})")
    lag1 = np.r_[np.nan, ic_s[:-1]]
    rho_1, n_1 = spearman(lag1, ic_s)
    print(f"  yesterday's IC  ->  today's IC:           spearman {rho_1:+.3f}  (n={n_1})")
    print("  A rho near zero means the edge's good and bad stretches are not")
    print("  separable in advance, and every threshold on a trailing statistic")
    print("  is fitting the sample it was read off.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(dict(
            window=W,
            ex_ante={nm: dict(values=ex_ante[nm],
                              spearman_vs_val_sharpe=spearman(ex_ante[nm], sharpes)[0])
                     for nm in markers},
            fold_sharpes=sharpes,
            pooled=out_rows,
            ic_persistence=dict(trailing=rho_ic, lag1=rho_1),
        ), open(args.json, "w"), indent=1)
        print(f"\n[json] wrote {args.json}")


if __name__ == "__main__":
    main()
