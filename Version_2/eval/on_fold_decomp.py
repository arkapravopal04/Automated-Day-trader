"""
eval/on_fold_decomp.py -- WHERE a walk-forward fold's overnight loss comes from.

Fold 2 of the final overnight walk-forward posts val Sharpe -2.42 with
`sharpe_ex_top5` at -3.75. Removing the five largest contributors makes the
fold WORSE, which means those five were the least-bad sessions and the loss is
not carried by a handful of them. This script tests that properly rather than
inferring it from one summary statistic.

It rebuilds the fold exactly as `eval/xsec_book.py --overnight` does -- same
panel, same edge, same causal vol, same schedule, same cost model, same lambda
-- and then, instead of collapsing the book to a period series, keeps the
[period, name] grid so the loss can be split three ways:

  * PER NAME    sum_t w[t,i]*r[t,i] against sum_t cost[t,i]. Answers "is this
                a few broken tickers?" A data artefact concentrates; a regime
                does not.
  * PER MONTH   gross / cost / net and a block IC inside the window. Answers
                "is the damage spread evenly, or is it the Nov-2023 rally?"
  * PER SESSION the net series itself, so the concentration statistics the
                summary reports can be read against the calendar.

NOTHING HERE SELECTS ANYTHING. Lambda is passed in (or read from the fold's
json), the split fractions come from the environment exactly as the run script
set them, and test is never touched. This is a decomposition of a result that
already exists, not a new search.

The first thing it prints is a RECONCILIATION against the fold's own json. If
gross and cost do not reproduce to floating-point noise, the decomposition is
describing a different book and nothing below it is admissible.
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

from paths import TRAIN_FRAC, VAL_FRAC  # noqa: E402
from eval.alpha_lab import (  # noqa: E402
    BARS_PER_DAY,
    TRADING_DAYS,
    block_ic,
    cross_sectional_demean,
    forward_return_bps,
    load_panel,
    overnight_decision_bars,
)
from eval.xsec_book import (  # noqa: E402
    env_cost_constants,
    execution_frame,
    measure_liquidity,
    overnight_schedule,
    reversal_edge,
    side_cost_bps,
    solve_weights,
    trailing_overnight_vol,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def run_book_detail(edge, ret, P, adv, sigma, k, schedule, day_id, lam, capital,
                    min_names=2, risk=None, spread_entry=None, spread_exit=None,
                    carry_bps=0.0, max_weight_mult=None, max_weight_frac=None):
    """`xsec_book.run_book`, but keeping the [period, name] grid.

    Every line mirrors run_book term for term -- the day-change liquidation,
    the drift applied to prev_w, the carry charged on gross exposure -- because
    a decomposition that does not sum back to the headline is not a
    decomposition of it. The caller asserts that reconciliation.
    """
    N = edge.shape[1]
    prev_w = np.zeros(N, dtype=np.float64)
    prev_exit, prev_day = None, None

    G, C, W = [], [], []          # [periods, N] gross / cost / weight
    entries, sel = [], []
    tot_turn = 0.0

    def liquidation_vec(w, bar):
        px = P[bar].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = np.nan_to_num(np.abs(w) * capital / np.where(px > 0, px, np.nan), nan=0.0)
        cs = np.nan_to_num(side_cost_bps(px, sh, adv, sigma, k, spread_exit), nan=0.0)
        return np.abs(w) * cs

    for (t, e) in schedule:
        if prev_day is not None and day_id[t] != prev_day and np.abs(prev_w).sum() > 0:
            C[-1] += liquidation_vec(prev_w, prev_exit)
            tot_turn += float(np.abs(prev_w).sum())
            prev_w = np.zeros(N, dtype=np.float64)

        w, n_sel, cost_side = solve_weights(
            edge[t], P[t].astype(np.float64), adv, sigma, lam, capital, k,
            min_names, size_by_cost=True,
            risk_row=(None if risk is None else risk[t]),
            spread_entry=spread_entry, spread_exit=spread_exit,
            max_weight_mult=max_weight_mult, max_weight_frac=max_weight_frac,
        )
        dw = w - prev_w
        tot_turn += float(np.abs(dw).sum())

        r = np.nan_to_num(ret[t], nan=0.0)
        G.append(w * r)
        C.append(np.abs(dw) * np.nan_to_num(cost_side, nan=0.0) + carry_bps * np.abs(w))
        W.append(w.copy())
        sel.append(n_sel)
        entries.append(t)

        prev_w = w * (1.0 + r / 1e4)
        prev_exit, prev_day = e, day_id[t]

    if prev_exit is not None and np.abs(prev_w).sum() > 0:
        C[-1] += liquidation_vec(prev_w, prev_exit)
        tot_turn += float(np.abs(prev_w).sum())

    return dict(G=np.asarray(G), C=np.asarray(C), W=np.asarray(W),
                entries=np.asarray(entries), selected=np.asarray(sel),
                turnover=tot_turn)


def herfindahl(x):
    """Concentration of a contribution vector. 1/n = perfectly even."""
    s = float(np.abs(x).sum())
    if s <= 0:
        return float("nan")
    p = np.abs(x) / s
    return float((p ** 2).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decompose one overnight fold's loss.")
    ap.add_argument("--lam", type=float, default=None,
                    help="lambda to decompose (default: chosen_lambda from --from-json)")
    ap.add_argument("--from-json", type=str, default=None,
                    help="the fold's xsec_book json, for chosen_lambda and reconciliation")
    ap.add_argument("--open-spread-bps", type=float, default=1.464)
    ap.add_argument("--close-spread-bps", type=float, default=0.262)
    ap.add_argument("--carry-bps", type=float, default=0.20)
    ap.add_argument("--risk-scale", choices=("none", "vol"), default="vol")
    ap.add_argument("--risk-window", type=int, default=60)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--min-names", type=int, default=2)
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--max-weight-mult", type=float, default=None,
                    help="override the per-name cap; default is whatever "
                         "--from-json was run under")
    ap.add_argument("--max-weight-frac", type=float, default=None,
                    help="override the ABSOLUTE per-name cap (share of gross); "
                         "default is whatever --from-json was run under")
    ap.add_argument("--earnings-calendar", type=str, default=None,
                    help="override the earnings exclusion; default is whatever "
                         "--from-json was run under")
    ap.add_argument("--top", type=int, default=15, help="names printed at each tail")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    lam, ref = args.lam, None
    # THE RISK CONTROLS ARE INHERITED FROM THE JSON, NOT RE-SPECIFIED.
    #
    # `--from-json` already supplies lambda, because a decomposition has to
    # describe the book that exists rather than search for a new one. The
    # per-name cap and the earnings exclusion are part of that book in exactly
    # the same way, so they are read from the same file. Passing them by hand
    # would make it possible to decompose the frozen reference under settings it
    # was never run at, and the reconciliation gate below would then be
    # comparing two different books and failing for the wrong reason.
    cap_mult, cap_frac = args.max_weight_mult, args.max_weight_frac
    earn_cal = args.earnings_calendar
    if args.from_json:
        d = json.load(open(args.from_json))
        h = d["holds"]["1"]
        if lam is None:
            lam = float(h["chosen_lambda"])
        ref = next((r for r in h["rows"] if abs(r["lam"] - lam) < 1e-9), None)
        if cap_mult is None:
            cap_mult = d.get("max_weight_mult")
        if cap_frac is None:
            cap_frac = d.get("max_weight_frac")
        if earn_cal is None:
            earn_cal = d.get("earnings_calendar")
    if lam is None:
        raise SystemExit("--lam or --from-json required")

    print("=" * 100)
    print(f"OVERNIGHT FOLD DECOMPOSITION -- lambda {lam}, split {args.split}")
    print("=" * 100)
    _cap_bits = []
    if cap_mult is not None:
        _cap_bits.append(f"{cap_mult:g}x equal weight")
    if cap_frac is not None:
        _cap_bits.append(f"{cap_frac:g} of gross")
    print(f"[controls] per-name cap "
          f"{'off' if not _cap_bits else ' and '.join(_cap_bits)}; "
          f"earnings exclusion {'off' if not earn_cal else earn_cal}")

    k = env_cost_constants()
    panel = load_panel(None)
    P, tickers = panel["P"], panel["tickers"]
    day_id, sli, index = panel["day_id"], panel["session_last_idx"], panel["index"]
    T, N = P.shape

    i_train = int(T * TRAIN_FRAC)
    i_val = int(T * (TRAIN_FRAC + VAL_FRAC))
    print(f"[split] train 0:{i_train}  val {i_train}:{i_val}  test {i_val}:{T} (untouched)")

    Px, _ = execution_frame(index, tickers, sli, column="open")
    adv, sigma = measure_liquidity(index, tickers, day_id, i_train, P)

    # The overnight target: uncapped 1-bar exec return, masked to decision bars.
    fwd = forward_return_bps(Px, 1, None)
    keep = np.zeros(fwd.shape[0], dtype=bool)
    keep[overnight_decision_bars(day_id, sli, T) - 1] = True
    fwd = np.where(keep[:, None], fwd, np.nan).astype(np.float32)

    edge, edge_desc = reversal_edge(P, fwd, i_train, day_id)
    print(f"[edge] {edge_desc}")

    if earn_cal:
        from eval.earnings import (apply_to_edge, assert_mapping, exclusion_mask,
                                   load_calendar)
        cal = load_calendar(earn_cal)
        assert_mapping(index, tickers, day_id, cal)
        emask, _ = exclusion_mask(index, tickers, day_id, cal)
        edge = apply_to_edge(edge, emask)

    risk = None
    if args.risk_scale == "vol":
        risk = trailing_overnight_vol(fwd, day_id, sli, T, window=args.risk_window)

    t0, t1 = (i_train, i_val) if args.split == "val" else (0, i_train)
    sched = overnight_schedule(t0, t1, day_id, sli, T)
    n_bars = t1 - t0
    win = index[[sched[0][0], sched[-1][1]]].tz_convert("America/New_York")
    print(f"[schedule] {len(sched)} {args.split} gap periods, "
          f"{str(win[0])[:10]} -> {str(win[1])[:10]}")

    det = run_book_detail(edge, fwd, Px, adv, sigma, k, sched, day_id, lam,
                          args.capital, args.min_names, risk=risk,
                          spread_entry=args.close_spread_bps,
                          spread_exit=args.open_spread_bps,
                          carry_bps=args.carry_bps,
                          max_weight_mult=cap_mult, max_weight_frac=cap_frac)

    G, C, W = det["G"], det["C"], det["W"]
    g_per, c_per = G.sum(axis=1), C.sum(axis=1)
    net_per = g_per - c_per
    n_per = len(net_per)
    years = n_bars / (BARS_PER_DAY * TRADING_DAYS)
    per_year = n_per / years

    sd = float(net_per.std(ddof=1))
    sharpe = (float(net_per.mean()) / sd) * math.sqrt(per_year) if sd > 0 else float("nan")
    print()
    print(f"[reconcile] here  gross {g_per.mean():+.4f}  cost {c_per.mean():+.4f}  "
          f"net {net_per.mean():+.4f} bps/period   Sharpe {sharpe:+.3f}")
    if ref is not None:
        r = ref[args.split]
        print(f"[reconcile] json  gross {r['gross_bps']:+.4f}  cost {r['cost_bps']:+.4f}  "
              f"net {r['net_bps']:+.4f} bps/period   Sharpe {r['sharpe']:+.3f}")
        drift = max(abs(g_per.mean() - r["gross_bps"]), abs(c_per.mean() - r["cost_bps"]))
        print(f"[reconcile] {'OK' if drift < 1e-6 else f'MISMATCH {drift:.2e}'}")

    # ---------------- per name ----------------
    g_i, c_i = G.sum(axis=0), C.sum(axis=0)
    net_i = g_i - c_i
    held = (np.abs(W) > 0).sum(axis=0)
    tot_net = float(net_i.sum())

    order = np.argsort(net_i)
    n_pos = int((net_i > 0).sum())
    n_neg = int((net_i < 0).sum())
    n_held = int((held > 0).sum())

    print()
    print("-" * 100)
    print("PER-NAME NET CONTRIBUTION (bps of book, summed over the window)")
    print("-" * 100)
    print(f"names ever held        {n_held} of {N}")
    print(f"positive contributors  {n_pos}    negative {n_neg}    "
          f"({100.0 * n_neg / max(n_held, 1):.1f}% of held names lose money)")
    print(f"total net              {tot_net:+.1f} bps")
    for label, cut in (("worst 5", 5), ("worst 10", 10), ("worst 20", 20)):
        s = float(net_i[order[:cut]].sum())
        share = 100.0 * s / tot_net if abs(tot_net) > 1e-12 else float("nan")
        print(f"{label:<22} {s:+9.1f} bps = {share:6.1f}% of the total")
    print(f"Herfindahl of |net_i|  {herfindahl(net_i):.4f}  "
          f"(1/{n_held} = {1.0 / max(n_held, 1):.4f} if perfectly even)")

    # DROP-ONE-TAIL. The honest test of "a few broken tickers": rebuild the
    # period series with the k worst names removed and re-score it. Selecting
    # the names to drop ON the same window is in-sample by construction, which
    # is why this is an UPPER bound on any per-name repair -- if the fold is
    # not rescued even here, no ex-ante screen rescues it either.
    print()
    print("DROP-WORST-k (in-sample, so an upper bound on any per-name repair)")
    for cut in (0, 1, 3, 5, 10, 20):
        keep_mask = np.ones(N, dtype=bool)
        keep_mask[order[:cut]] = False
        nn = G[:, keep_mask].sum(axis=1) - C[:, keep_mask].sum(axis=1)
        s = float(nn.std(ddof=1))
        sh = (float(nn.mean()) / s) * math.sqrt(per_year) if s > 0 else float("nan")
        print(f"  drop worst {cut:>2} names -> net {nn.mean():+7.3f} bps/period, "
              f"Sharpe {sh:+6.2f}")

    losers, winners = order[:args.top], order[::-1][:args.top]
    print()
    print(f"{'WORST':<10}{'net':>9}{'gross':>9}{'cost':>8}{'held':>6}"
          f"   |  {'BEST':<10}{'net':>9}{'gross':>9}{'cost':>8}{'held':>6}")
    for a, b in zip(losers, winners):
        print(f"{tickers[a]:<10}{net_i[a]:>9.1f}{g_i[a]:>9.1f}{c_i[a]:>8.1f}{held[a]:>6}"
              f"   |  {tickers[b]:<10}{net_i[b]:>9.1f}{g_i[b]:>9.1f}{c_i[b]:>8.1f}{held[b]:>6}")

    # ---------------- single (name, session) cells ----------------
    #
    # The per-name and per-session views each marginalise the other away, and a
    # loss can be concentrated in ONE CELL while looking moderate in both
    # margins. This is the join: the largest individual bets, named and dated,
    # so an overnight earnings gap cannot hide inside a ticker's annual total.
    ent = det["entries"]
    NET = G - C
    flat = NET.ravel()
    cell_order = np.argsort(flat)
    print()
    print("-" * 100)
    print("LARGEST SINGLE (name, session) CELLS")
    print("-" * 100)
    print(f"{'name':<8}{'session':<12}{'net':>10}{'gross':>10}{'weight':>9}"
          f"{'ret bps':>10}{'% of total':>12}")
    for c in cell_order[:10]:
        t_i, n_i = divmod(int(c), N)
        r = float(np.nan_to_num(fwd[ent[t_i], n_i], nan=0.0))
        share = 100.0 * flat[c] / tot_net if abs(tot_net) > 1e-12 else float("nan")
        print(f"{tickers[n_i]:<8}{str(index[ent[t_i]])[:10]:<12}{flat[c]:>10.1f}"
              f"{G[t_i, n_i]:>10.1f}{W[t_i, n_i]:>+9.4f}{r:>10.0f}{share:>11.1f}%")
    worst_cell = float(flat[cell_order[0]])
    print(f"[cells] the single worst cell is {100.0 * worst_cell / tot_net:.1f}% "
          f"of the window's entire net loss")

    # CONCENTRATION. `book_weights` normalises by gross and imposes no per-name
    # cap, so nothing bounds how much of the book one name may become. This
    # prints what that permitted in practice, against the equal weight the same
    # book would have held.
    aw = np.abs(W)
    peak = aw.max(axis=1)
    eq = 1.0 / np.maximum(det["selected"], 1)
    print(f"[concentration] largest single position per session: median "
          f"{np.median(peak):.3f}, p90 {np.percentile(peak, 90):.3f}, "
          f"max {peak.max():.3f} of gross")
    print(f"[concentration] that max is {peak.max() / eq[int(np.argmax(peak))]:.1f}x "
          f"the equal weight of its own session's book "
          f"({det['selected'][int(np.argmax(peak))]} names)")

    # ---------------- the tail, in sigma units ----------------
    #
    # The cells above are dated to earnings prints. That is a DIFFERENT
    # distribution, not a worse draw from the same one: a scheduled
    # announcement turns a ~1% overnight gap into a 10-30% one, and the book
    # carries no calendar, so it holds straight through it.
    #
    # Sized in units of the SAME causal trailing vol the risk-scaling divides
    # by, so these are buckets the sizing rule itself could have seen. That
    # framing exposes a perverse incidence: equal-risk sizing gives the largest
    # weight to the name that has been QUIETEST, which is exactly the state a
    # name is in on the eve of its print.
    #
    # Zeroing a bucket is a MEASUREMENT, not a proposed rule -- |gap| is not
    # knowable at decision time. An earnings calendar IS, which is why the
    # number is worth having as a bound on what such a calendar could buy.
    if risk is not None:
        RS = risk[ent]                                  # [periods, N] causal vol, bps
        RET = np.where(np.isfinite(fwd[ent]), fwd[ent], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.abs(RET) / np.where(np.isfinite(RS) & (RS > 0), RS, np.nan)
        Z = np.nan_to_num(Z, nan=0.0)
        traded = np.abs(W) > 0
        n_tr = max(int(traded.sum()), 1)
        print()
        print("-" * 100)
        print("HELD CELLS BY GAP SIZE, in causal trailing-vol sigma")
        print("-" * 100)
        print(f"{'|gap|':<14}{'cells':>8}{'% cells':>9}{'net bps':>11}"
              f"{'% of total':>12}{'mean |ret| bps':>16}")
        for lo_z, hi_z in ((0, 2), (2, 4), (4, 6), (6, 1e9)):
            m = traded & (Z >= lo_z) & (Z < hi_z)
            s = float(NET[m].sum())
            lab = f"{lo_z}-{hi_z} sigma" if hi_z < 1e9 else f">{lo_z} sigma"
            mr = float(np.abs(RET[m]).mean()) if m.any() else float("nan")
            print(f"{lab:<14}{int(m.sum()):>8}{100.0 * m.sum() / n_tr:>8.1f}%"
                  f"{s:>11.1f}{100.0 * s / tot_net:>11.1f}%{mr:>16.0f}")
        for cut in (4, 6, 8):
            m = Z >= cut
            nn = np.where(m, 0.0, NET).sum(axis=1)
            sdc = float(nn.std(ddof=1))
            shc = (float(nn.mean()) / sdc) * math.sqrt(per_year) if sdc > 0 else float("nan")
            print(f"  zero every held cell beyond {cut} sigma "
                  f"({int((m & traded).sum())} cells) -> net {nn.mean():+7.3f} "
                  f"bps/period, Sharpe {shc:+6.2f}")

    # ---------------- per month ----------------
    ny = index[ent].tz_convert("America/New_York")
    month = pd.PeriodIndex(ny, freq="M")
    tgt = cross_sectional_demean(fwd)

    print()
    print("-" * 100)
    print("PER MONTH inside the window")
    print("-" * 100)
    print(f"{'month':<9}{'sess':>6}{'gross':>9}{'cost':>8}{'net':>9}{'cum net':>10}"
          f"{'hit%':>7}{'IC':>10}{'t':>7}{'names':>7}")
    months, cum = [], 0.0
    for m in month.unique():
        sel = np.asarray(month == m)
        rows = ent[sel]
        g, c = g_per[sel].mean(), c_per[sel].mean()
        nn = net_per[sel]
        cum += float(nn.sum())
        act = det["selected"][sel] >= 2
        hit = 100.0 * float((nn[act] > 0).mean()) if act.any() else float("nan")
        blk = np.repeat(day_id[rows], N)
        ic, tt, nb, nobs = block_ic(edge[rows].ravel(), tgt[rows].ravel(), blk)
        nm = float(det["selected"][sel].mean())
        print(f"{str(m):<9}{int(sel.sum()):>6}{g:>9.2f}{c:>8.2f}{nn.mean():>9.2f}"
              f"{cum:>10.1f}{hit:>7.1f}{ic:>+10.4f}{tt:>+7.2f}{nm:>7.1f}")
        months.append(dict(month=str(m), sessions=int(sel.sum()),
                           gross_bps=float(g), cost_bps=float(c),
                           net_bps=float(nn.mean()), cum_net=float(cum),
                           hit_rate=float(hit), ic=float(ic), ic_t=float(tt),
                           mean_names=nm))

    # A regime story predicts a RUN of negative months; an artefact predicts one.
    mn = np.array([m["net_bps"] for m in months])
    mic = np.array([m["ic"] for m in months])
    print()
    print(f"[months] {int((mn < 0).sum())} of {len(mn)} negative, worst {mn.min():+.2f}, "
          f"best {mn.max():+.2f}, median {np.median(mn):+.2f} bps")
    print(f"[months] IC negative in {int((mic < 0).sum())} of {len(mic)} months "
          f"(mean {np.nanmean(mic):+.4f}, median {np.nanmedian(mic):+.4f})")

    # ---------------- session concentration ----------------
    o = np.argsort(net_per)[::-1]
    top5 = float(net_per[o[:5]].sum())
    share5 = top5 / tot_net if abs(tot_net) > 1e-12 else float("nan")
    print()
    print(f"[sessions] top-5 share of net PnL {share5:+.3f}")
    print(f"[sessions] {int((net_per < 0).sum())} of {n_per} sessions negative "
          f"(hit rate {100.0 * (net_per > 0).mean():.1f}%)")
    print("[sessions] five worst:")
    for w in np.argsort(net_per)[:5]:
        print(f"    {str(index[ent[w]])[:10]}  net {net_per[w]:+8.2f} bps  "
              f"(gross {g_per[w]:+7.2f}, cost {c_per[w]:.2f})")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump(dict(
            lam=lam, split=args.split, n_periods=n_per, sharpe=sharpe,
            gross_bps=float(g_per.mean()), cost_bps=float(c_per.mean()),
            net_bps=float(net_per.mean()),
            per_name={tickers[i]: dict(net=float(net_i[i]), gross=float(g_i[i]),
                                       cost=float(c_i[i]), held=int(held[i]))
                      for i in range(N) if held[i] > 0},
            months=months, n_names_held=n_held, n_negative=n_neg,
            herfindahl=herfindahl(net_i),
            net_per_session=[float(x) for x in net_per],
            session_dates=[str(index[t])[:10] for t in ent],
        ), open(args.json, "w"), indent=1)
        print(f"\n[json] wrote {args.json}")


if __name__ == "__main__":
    main()
