"""
eval/on_independence.py -- the denominator under every claim this project makes.

THE PROBLEM. The walk-forward is an EXPANDING window: fold k trains on
[0, 0.30 + 0.10k) and validates on the 0.10 that follows. Two things follow from
that arithmetic, and neither is visible in the freeze table:

  1. THE TRAINING SETS ARE NESTED. train_1 subset train_2 subset ... subset
     train_5. Folds 4 and 5 share 7/8 of their training data, so the lambda each
     selects is very nearly the same decision made twice.
  2. THE VALIDATION WINDOWS TILE ONE CONTIGUOUS SPAN. val_1 ends the session
     before val_2 begins, and so on through val_5. They are not five samples
     from a population; they are five consecutive slices of a single
     out-of-sample record, and the slice boundaries carry no information.

`t(n=5) = mean / (sd / sqrt(5))`, quoted in the freeze table and in every arm
table since, treats those five slices as five independent observations. They are
not. f4 and f5 in particular are one observation dressed as two.

HOW BIG THE ERROR IS, THOUGH, IS AN EMPIRICAL QUESTION AND THIS SCRIPT ANSWERS
IT RATHER THAN ASSUMING IT. The structural objection is valid on its own terms
regardless of which way the number moves; if the corrected statistic comes back
close to the naive one, that is a fact to report, not a reason to drop the
correction. What is NOT acceptable is quoting n=5 without having looked.

WHAT THIS SCRIPT DOES INSTEAD. Because the five validation windows are disjoint
AND contiguous, they can be concatenated into ONE walk-forward out-of-sample
series, and every session in it was graded under a lambda selected only on data
that preceded it. That series is the real out-of-sample record, and it is the
thing to compute a statistic on:

  * pooled Sharpe over all val sessions, with n = SESSIONS, not folds
  * an IID standard error, which is still wrong, but wrong by less
  * a NEWEY-WEST standard error, which handles the serial dependence the
    contiguity creates
  * a CIRCULAR BLOCK BOOTSTRAP interval, which makes no normality assumption
  * the f4+f5-merged four-observation version, reported because it is the
    minimum honest correction to the existing table

and, alongside them, the naive n=5 t, so the size of the overstatement is
readable rather than argued about.

NOTHING HERE SELECTS ANYTHING. Each fold's lambda and risk controls are read
from that fold's own json. Test is never touched.
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
    BARS_PER_DAY,
    TRADING_DAYS,
    forward_return_bps,
    load_panel,
    overnight_decision_bars,
)
from eval.on_fold_decomp import run_book_detail  # noqa: E402
from eval.xsec_book import (  # noqa: E402
    env_cost_constants,
    execution_frame,
    measure_liquidity,
    overnight_schedule,
    reversal_edge,
    trailing_overnight_vol,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

FOLDS = (1, 2, 3, 4, 5)


def train_frac(k):
    """The expanding-window schedule the run scripts use, stated once."""
    return 0.30 + 0.10 * k


def newey_west_se(x, lag=None):
    """HAC standard error of the MEAN of x. (se, lag_used).

    Bartlett kernel, Newey-West. The lag is the standard automatic choice
    floor(4 * (n/100)^(2/9)) unless one is given -- fixed by a rule rather than
    picked, because a bandwidth chosen after seeing the t is a bandwidth chosen
    to get a t.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan"), 0
    if lag is None:
        lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(int(lag), 0)
    e = x - x.mean()
    g0 = float(e @ e) / n
    var = g0
    for j in range(1, lag + 1):
        gj = float(e[j:] @ e[:-j]) / n
        var += 2.0 * (1.0 - j / (lag + 1.0)) * gj
    var = max(var, 1e-300)
    return math.sqrt(var / n), lag


def block_bootstrap_sharpe(x, per_year, block, n_boot, seed=0):
    """Circular block bootstrap of the annualised Sharpe. -> (lo, hi, sd).

    Resamples CONTIGUOUS blocks so the serial dependence that the contiguity of
    the validation windows creates survives into each resample. A plain iid
    bootstrap would destroy exactly the structure being corrected for.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < block * 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n_blocks = int(math.ceil(n / block))
    xx = np.concatenate([x, x[: block]])          # circular
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.randint(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        s = xx[idx % xx.size]
        sd = s.std(ddof=1)
        out[b] = (s.mean() / sd) * math.sqrt(per_year) if sd > 0 else np.nan
    out = out[np.isfinite(out)]
    if out.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            float(out.std(ddof=1)))


def sharpe_stats(x, per_year):
    """Annualised Sharpe plus IID and Newey-West standard errors of it.

    The SE of an annualised Sharpe is taken as SE(mean)/sd * sqrt(per_year),
    i.e. the sampling error of the numerator only. The denominator is estimated
    too and that is ignored here; it is the standard approximation and it makes
    these intervals slightly NARROWER than the truth, which is the conservative
    direction for an argument that the quoted t is already too generous.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    sd = float(x.std(ddof=1))
    if sd <= 0 or n < 2:
        return dict(n=n, sharpe=float("nan"))
    sh = (float(x.mean()) / sd) * math.sqrt(per_year)
    se_iid = (sd / math.sqrt(n)) / sd * math.sqrt(per_year)
    se_nw, lag = newey_west_se(x)
    se_nw = se_nw / sd * math.sqrt(per_year)
    return dict(n=n, sharpe=sh, se_iid=se_iid, se_nw=se_nw, nw_lag=lag,
                t_iid=sh / se_iid if se_iid > 0 else float("nan"),
                t_nw=sh / se_nw if se_nw > 0 else float("nan"))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Effective sample size of the walk-forward.")
    ap.add_argument("--fmt", default="logs/p3/on/cap2/both2_f{k}.json",
                    help="the arm whose folds are being pooled")
    ap.add_argument("--open-spread-bps", type=float, default=1.464)
    ap.add_argument("--close-spread-bps", type=float, default=0.262)
    ap.add_argument("--carry-bps", type=float, default=0.20)
    ap.add_argument("--risk-scale", choices=("none", "vol"), default="vol")
    ap.add_argument("--risk-window", type=int, default=60)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--min-names", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--block", type=int, default=21,
                    help="bootstrap block length in sessions (one trading month)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    print("=" * 104)
    print("EFFECTIVE SAMPLE SIZE OF THE WALK-FORWARD")
    print("=" * 104)
    print(f"source: {args.fmt}")

    # ---------------- read what each fold was actually run at ----------------
    cfg = {}
    for k in FOLDS:
        p = args.fmt.format(k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        h = d["holds"]["1"]
        cfg[k] = dict(lam=h["chosen_lambda"],
                      cap_mult=d.get("max_weight_mult"),
                      cap_frac=d.get("max_weight_frac"),
                      earn=d.get("earnings_calendar"))
    if not cfg:
        raise SystemExit(f"no fold jsons at {args.fmt}")
    ctl = {(c["cap_mult"], c["cap_frac"], bool(c["earn"])) for c in cfg.values()}
    if len(ctl) > 1:
        raise SystemExit(f"folds disagree on the risk controls: {sorted(ctl)}")
    mult, frac, has_earn = ctl.pop()
    print(f"controls: cap_mult={mult} cap_frac={frac} earnings={bool(has_earn)}")
    print(f"lambdas:  " + "  ".join(f"f{k}={cfg[k]['lam']}" for k in sorted(cfg)))

    # ---------------- panel, once ----------------
    k_cost = env_cost_constants()
    panel = load_panel(None)
    P, tickers = panel["P"], panel["tickers"]
    day_id, sli, index = panel["day_id"], panel["session_last_idx"], panel["index"]
    T, N = P.shape

    Px, _ = execution_frame(index, tickers, sli, column="open")
    fwd_all = forward_return_bps(Px, 1, None)
    keep = np.zeros(fwd_all.shape[0], dtype=bool)
    keep[overnight_decision_bars(day_id, sli, T) - 1] = True
    fwd_all = np.where(keep[:, None], fwd_all, np.nan).astype(np.float32)

    cal = None
    if has_earn:
        from eval.earnings import (apply_to_edge, assert_mapping, exclusion_mask,
                                   load_calendar)
        cal = load_calendar(cfg[sorted(cfg)[0]]["earn"])
        assert_mapping(index, tickers, day_id, cal)
        emask, _ = exclusion_mask(index, tickers, day_id, cal, report=False)

    # ---------------- the fold structure, stated ----------------
    print()
    print("-" * 104)
    print("FOLD STRUCTURE -- the arithmetic the expanding window actually implies")
    print("-" * 104)
    print(f"  {'fold':<6}{'train bars':>16}{'val bars':>16}{'val sessions':>14}"
          f"  {'val window':<26}")
    per_fold, spans = {}, {}
    for k in sorted(cfg):
        tf = train_frac(k)
        i_tr = int(T * tf)
        i_va = int(T * (tf + args.val_frac))
        sched = overnight_schedule(i_tr, i_va, day_id, sli, T)
        d0 = str(index[sched[0][0]].tz_convert("America/New_York"))[:10]
        d1 = str(index[sched[-1][1]].tz_convert("America/New_York"))[:10]
        spans[k] = (i_tr, i_va, sched, d0, d1)
        print(f"  {k:<6}{f'0:{i_tr}':>16}{f'{i_tr}:{i_va}':>16}"
              f"{len(sched):>14}  {d0} -> {d1}")

    ks = sorted(spans)
    print()
    print("  ADJACENT FOLDS, what they share:")
    print(f"  {'pair':<10}{'train overlap':>16}{'val gap (sessions)':>22}")
    overlaps = []
    for a, b in zip(ks, ks[1:]):
        ta, tb = spans[a][0], spans[b][0]
        ov = min(ta, tb) / max(ta, tb)
        # Measured entry-to-entry: the last night fold a decides on, against
        # the first night fold b decides on. Adjacent windows differ by one
        # session; a gap of 1 means nothing sits between them.
        end_a = spans[a][2][-1][0]
        start_b = spans[b][2][0][0]
        gap = int(day_id[start_b]) - int(day_id[end_a])
        overlaps.append(ov)
        print(f"  f{a}-f{b}{'':<5}{100.0 * ov:>15.1f}%{gap:>22}")
    print()
    print("  Train overlap is |train_a| / |train_b| for the nested windows: the")
    print("  smaller training set is entirely contained in the larger one, so this")
    print("  is the fraction of the larger fold's training data the smaller one")
    print("  already saw. A val gap of 1 session means the two windows are")
    print("  ADJACENT -- val_b starts the trading day after val_a ends, with")
    print("  nothing in between.")
    gaps = [int(day_id[spans[b][2][0][0]]) - int(day_id[spans[a][2][-1][0]])
            for a, b in zip(ks, ks[1:])]
    if gaps and all(g <= 1 for g in gaps):
        print()
        print("  ==> THE FIVE VALIDATION WINDOWS TILE ONE CONTIGUOUS SPAN.")
        print("      They are five consecutive slices of a single out-of-sample")
        print("      record, and the slice boundaries carry no information. A t")
        print("      computed across them with n=5 is counting cut points as data.")

    # ---------------- rebuild each fold's val session series ----------------
    print()
    print("-" * 104)
    print("REBUILDING each fold's val session series at its own chosen lambda")
    print("-" * 104)
    series, dates, fold_id = [], [], []
    g_series, c_series, fold_gc = [], [], {}
    fold_sharpe, fold_net = {}, {}
    for k in ks:
        i_tr, i_va, sched, d0, d1 = spans[k]
        adv, sigma = measure_liquidity(index, tickers, day_id, i_tr, P)
        edge, _ = reversal_edge(P, fwd_all, i_tr, day_id)
        if has_earn:
            edge = apply_to_edge(edge, emask)
        risk = None
        if args.risk_scale == "vol":
            risk = trailing_overnight_vol(fwd_all, day_id, sli, T,
                                          window=args.risk_window)
        det = run_book_detail(
            edge, fwd_all, Px, adv, sigma, k_cost, sched, day_id,
            float(cfg[k]["lam"]), args.capital, args.min_names, risk=risk,
            spread_entry=args.close_spread_bps, spread_exit=args.open_spread_bps,
            carry_bps=args.carry_bps,
            max_weight_mult=cfg[k]["cap_mult"], max_weight_frac=cfg[k]["cap_frac"])
        g_s = det["G"].sum(axis=1)
        c_s = det["C"].sum(axis=1)
        net = g_s - c_s
        n_bars = i_va - i_tr
        years = n_bars / (BARS_PER_DAY * TRADING_DAYS)
        py = len(net) / years
        sd = float(net.std(ddof=1))
        sh = (float(net.mean()) / sd) * math.sqrt(py) if sd > 0 else float("nan")
        fold_sharpe[k] = sh
        fold_net[k] = float(net.mean())
        series.append(net)
        g_series.append(g_s)
        c_series.append(c_s)
        fold_gc[k] = (float(g_s.sum()), float(c_s.sum()))
        dates.extend([str(index[t])[:10] for t in det["entries"]])
        fold_id.extend([k] * len(net))
        # Cross-check against the fold's own json, so a rebuilt series that does
        # not reproduce the arm cannot silently become the pooled record.
        h = json.load(open(args.fmt.format(k=k)))["holds"]["1"]
        r = next(x for x in h["rows"] if x["lam"] == cfg[k]["lam"])
        drift = abs(float(net.mean()) - r["val"]["net_bps"])
        print(f"  f{k}  lam {cfg[k]['lam']:<5} sessions {len(net):>4}  "
              f"net {net.mean():+7.3f}  Sharpe {sh:+6.3f}   "
              f"json net {r['val']['net_bps']:+7.3f}  "
              f"-> {'OK' if drift < 1e-6 else f'MISMATCH {drift:.2e}'}")

    pooled = np.concatenate(series)
    fold_id = np.asarray(fold_id)
    n_tot = pooled.size
    # Annualisation for the pooled series: one period per session, so the
    # per-year count is the trading-day count, not a bar count.
    per_year = float(TRADING_DAYS)

    # ---------------- the naive statistic, and the honest ones ----------------
    fs = np.array([fold_sharpe[k] for k in ks], dtype=float)
    m, sd5 = float(fs.mean()), float(fs.std(ddof=1))
    se5 = sd5 / math.sqrt(fs.size)
    t5 = m / se5 if se5 > 0 else float("nan")

    print()
    print("=" * 104)
    print("THE STATISTIC, FOUR WAYS")
    print("=" * 104)
    print(f"  {'basis':<44}{'n':>7}{'Sharpe':>10}{'se':>10}{'t':>9}")
    print(f"  {'-' * 80}")
    print(f"  {'across folds (WHAT THE TABLES QUOTE)':<44}{fs.size:>7}"
          f"{m:>10.3f}{se5:>10.3f}{t5:>9.2f}")

    # f4+f5 merged: the minimum honest correction.
    merged = []
    for k in ks:
        if k in (4, 5):
            continue
        merged.append(fold_sharpe[k])
    if 4 in fold_sharpe and 5 in fold_sharpe:
        pair = pooled[(fold_id == 4) | (fold_id == 5)]
        sdp = float(pair.std(ddof=1))
        merged.append((float(pair.mean()) / sdp) * math.sqrt(per_year)
                      if sdp > 0 else float("nan"))
    mg = np.array(merged, dtype=float)
    mg = mg[np.isfinite(mg)]
    if mg.size > 1:
        mm, sdm = float(mg.mean()), float(mg.std(ddof=1))
        sem = sdm / math.sqrt(mg.size)
        print(f"  {'across folds, f4+f5 merged as one period':<44}{mg.size:>7}"
              f"{mm:>10.3f}{sem:>10.3f}{mm / sem if sem > 0 else float('nan'):>9.2f}")

    st = sharpe_stats(pooled, per_year)
    print(f"  {'pooled walk-forward sessions, IID se':<44}{st['n']:>7}"
          f"{st['sharpe']:>10.3f}{st['se_iid']:>10.3f}{st['t_iid']:>9.2f}")
    nw_label = f"pooled walk-forward sessions, NW se (lag {st['nw_lag']})"
    print(f"  {nw_label:<44}{st['n']:>7}{st['sharpe']:>10.3f}"
          f"{st['se_nw']:>10.3f}{st['t_nw']:>9.2f}")

    lo, hi, sdb = block_bootstrap_sharpe(pooled, per_year, args.block,
                                         args.n_boot)
    print()
    print(f"  circular block bootstrap, {args.n_boot:,} resamples, block "
          f"{args.block} sessions:")
    print(f"    annualised Sharpe {st['sharpe']:+.3f}   95% CI "
          f"[{lo:+.3f}, {hi:+.3f}]   bootstrap sd {sdb:.3f}")
    print(f"    {'-> the interval EXCLUDES zero' if lo > 0 or hi < 0 else '-> the interval INCLUDES zero'}")

    # ---------------- what the pooled series looks like ----------------
    # ---------------- the ratio, which is the bar the project quotes ----------
    #
    # ratio = alpha_per_turnover / cost_per_turnover, and both divide by the
    # same turnover, so it is exactly sum(gross) / sum(cost) over the window.
    # That makes it poolable across the five contiguous val windows in the same
    # way the net series is, and bootstrappable on the same blocks.
    gc = np.stack([np.concatenate(g_series), np.concatenate(c_series)], axis=1)
    pooled_ratio = float(gc[:, 0].sum() / gc[:, 1].sum())
    rng = np.random.RandomState(1)
    nb = int(math.ceil(n_tot / args.block))
    gg = np.concatenate([gc, gc[: args.block]], axis=0)
    boot = np.empty(args.n_boot, dtype=float)
    for b in range(args.n_boot):
        st_ = rng.randint(0, n_tot, size=nb)
        idx = (st_[:, None] + np.arange(args.block)[None, :]).ravel()[:n_tot]
        sm = gg[idx % gg.shape[0]]
        boot[b] = sm[:, 0].sum() / sm[:, 1].sum() if sm[:, 1].sum() != 0 else np.nan
    boot = boot[np.isfinite(boot)]
    r_lo, r_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    fold_ratio = np.array([fold_gc[k][0] / fold_gc[k][1] for k in ks])
    r_m, r_sd = float(fold_ratio.mean()), float(fold_ratio.std(ddof=1))
    r_se = r_sd / math.sqrt(fold_ratio.size)

    print()
    print("-" * 104)
    print("THE RATIO -- the bar the project is actually stated against (> 2)")
    print("-" * 104)
    print(f"  across folds (n=5)      mean {r_m:+.3f}  se {r_se:.3f}  "
          f"t {r_m / r_se if r_se > 0 else float('nan'):+.2f}")
    print(f"  pooled over {n_tot} sessions   {pooled_ratio:+.3f}   "
          f"95% block-bootstrap CI [{r_lo:+.3f}, {r_hi:+.3f}]")
    print(f"  clears 2.0? {'YES' if r_lo > 2.0 else 'NO -- the interval does not reach 2'}")

    print()
    print("-" * 104)
    print("SERIAL DEPENDENCE IN THE POOLED SERIES")
    print("-" * 104)
    e = pooled - pooled.mean()
    g0 = float(e @ e) / n_tot
    acf = [float(e[j:] @ e[:-j]) / n_tot / g0 for j in range(1, 6)]
    print("  autocorrelation of the nightly net series: "
          + "  ".join(f"lag{j + 1} {a:+.3f}" for j, a in enumerate(acf)))
    infl = st["se_nw"] / st["se_iid"] if st["se_iid"] > 0 else float("nan")
    print(f"  Newey-West / IID standard error ratio: {infl:.3f}")
    print(f"  implied effective sessions: {n_tot / infl ** 2:,.0f} of {n_tot:,}")

    # THE NUMBER THE ARGUMENT IS ABOUT. What n would the across-fold t need in
    # order to agree with the pooled Newey-West t? Stated as a ratio, because
    # the two statistics are not the same estimator and the comparison is an
    # order-of-magnitude one, not an identity.
    print()
    print("-" * 104)
    print("THE NAIVE t AGAINST THE POOLED t")
    print("-" * 104)
    print(f"  across-fold Sharpe t (n=5)   {t5:+.2f}")
    print(f"  pooled Newey-West Sharpe t   {st['t_nw']:+.2f}")
    if np.isfinite(t5) and np.isfinite(st["t_nw"]) and abs(st["t_nw"]) > 1e-9:
        print(f"  ratio                        {t5 / st['t_nw']:.2f}x")
    print()
    print("  These are different estimators and the ratio is not a correction")
    print("  factor to divide by. It is the size of the disagreement between the")
    print("  number the tables quote and the number the same data supports when")
    print("  the contiguity is not thrown away.")
    print()
    print("  READ THIS WITH THE TWO BLOCKS ABOVE, NOT INSTEAD OF THEM. If the")
    print("  session-level autocorrelation is near zero the two t's will agree,")
    print("  and that is a real finding -- but it does NOT rescue the n=5 t. The")
    print("  damage from treating contiguous slices as observations shows up in")
    print("  the RE-BUCKETING (f4+f5 merged) and in the width of the bootstrap")
    print("  interval, both of which are computed above and neither of which is")
    print("  visible in a t across five fold summaries.")

    print()
    print("  PER-FOLD Sharpes, for reference: "
          + "  ".join(f"f{k} {fold_sharpe[k]:+.2f}" for k in ks))
    if 4 in fold_sharpe and 5 in fold_sharpe and mg.size:
        print(f"  f4 {fold_sharpe[4]:+.2f} and f5 {fold_sharpe[5]:+.2f} are "
              f"ONE contiguous period, and as one it is {merged[-1]:+.2f}.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(dict(
            source=args.fmt,
            controls=dict(cap_mult=mult, cap_frac=frac, earnings=bool(has_earn)),
            lambdas={str(k): cfg[k]["lam"] for k in ks},
            fold_sharpe={str(k): fold_sharpe[k] for k in ks},
            fold_net_bps={str(k): fold_net[k] for k in ks},
            train_overlap=overlaps,
            across_folds=dict(n=int(fs.size), mean=m, se=se5, t=t5),
            pooled=st,
            bootstrap=dict(lo=lo, hi=hi, sd=sdb, block=args.block,
                           n_boot=args.n_boot),
            ratio=dict(pooled=pooled_ratio, ci_lo=r_lo, ci_hi=r_hi,
                       across_folds_mean=r_m, across_folds_se=r_se),
            acf=acf, n_sessions=int(n_tot),
            val_windows={str(k): [spans[k][3], spans[k][4]] for k in ks},
        ), open(args.json, "w"), indent=1, default=float)
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
