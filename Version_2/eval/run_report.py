"""
Post-mortem for a training run: reads metrics.jsonl (+ every tick segment) and
prints the checks that have actually caught bugs in this project.

Written because the same analysis has been redone by hand after every run, and
because two of the three findings it encodes were missed for a whole session
each. Each section below exists for a specific historical failure:

  ECONOMICS        loss per trade, split by phase. The last run improved this
                   by 5.5% across 75 rollouts (0.2904 -> 0.2744) while cutting
                   trade count 90,277 -> 28. That is "trade less", not "trade
                   better", and a rising equity curve hides the difference.

  PER-SHARE COST   fits cost_bps = a/price + b across streams. Session 2's
                   +486% run was three per-share SUBSIDIES, caught by
                   rank corr(price, equity) = -0.921. The last run showed the
                   same signature with the sign flipped (-0.552 on cost per
                   trade): the half-tick spread floor makes cheap names 5-6x
                   more expensive. Either sign means the cross-section is
                   measuring price level, not alpha.

  KELLY BINDING    fraction of (rollout, ticker) cells pinned to the floor.
                   Last run: kelly_raw == 0 in 93.8% of cells and
                   kelly_fractional at the floor in 98.4%, with every observed
                   fill at exactly 0.0799 x equity. The sizer was decorative
                   and nothing said so.

  LEARNING         entropy trajectory, collapse point, and how much of one
                   epoch the run actually covered. The last run collapsed at
                   rollout 81 having seen 22% of the training split, then spent
                   43% of its compute unable to explore.

  TICK COVERAGE    which rollouts the tick log actually spans. Twice now the
                   post-mortem has had only the live segment: rollouts 126-150,
                   28 fills out of 257,137 trades, none of the active phase.

Usage
-----
    python eval/run_report.py                                  # default paths
    python eval/run_report.py --metrics /kaggle/working/logs/metrics.jsonl
    python eval/run_report.py --ticks "logs/metrics.ticks.jsonl*"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

BARS_PER_DAY = 78


def _load_rollouts(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type", "rollout") != "tick":
                rows.append(rec)
    rows.sort(key=lambda r: r.get("rollout", 0))
    return rows


def _load_ticks(pattern):
    """Load every tick segment, newest last.

    Rotated segments are metrics.ticks.jsonl.1 .. .N with .1 the MOST recent
    backup, so ordering is: highest-numbered backup first, live segment last.
    """
    paths = sorted(glob.glob(pattern))
    live = [p for p in paths if p.endswith(".jsonl")]
    rotated = sorted((p for p in paths if p not in live),
                     key=lambda p: -int(p.rsplit(".", 1)[-1]))
    ordered = rotated + live
    recs = []
    for p in ordered:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("record_type") == "tick":
                    recs.append(rec)
    return ordered, recs


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def report_economics(rows, initial_cash):
    section("ECONOMICS")
    nw = np.array([r.get("net_worth", np.nan) for r in rows], float)
    tr = np.array([r.get("trades_this_rollout", 0) for r in rows], float)
    n_streams = len(rows[-1].get("net_worth_per_ticker") or []) or 1
    start = initial_cash * n_streams

    total_trades = rows[-1].get("total_trades", int(tr.sum()))
    print(f"rollouts        {rows[0].get('rollout')} -> {rows[-1].get('rollout')} "
          f"({len(rows)} records, {n_streams} streams)")
    print(f"net worth       {start:,.0f} -> {nw[-1]:,.0f} "
          f"({(nw[-1] / start - 1) * 100:+.2f}%)")
    print(f"total trades    {total_trades:,}")
    if total_trades:
        print(f"per trade       ${(start - nw[-1]) / total_trades:+.4f}")

    print()
    print("by phase -- watch whether $/trade IMPROVES or only the trade count falls")
    print(f"{'rollouts':<14}{'d net worth':>14}{'trades':>11}{'$/trade':>11}")
    edges = np.linspace(0, len(rows) - 1, 7).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        d = nw[b] - nw[a]
        t = tr[a + 1:b + 1].sum()
        per = (-d / t) if t > 0 else float("nan")
        print(f"{str(rows[a]['rollout']) + '-' + str(rows[b]['rollout']):<14}"
              f"{d:>14,.0f}{t:>11,.0f}{per:>11.4f}")


def report_per_share_cost(rows, ticks, initial_cash):
    section("PER-SHARE COST SIGNATURE")
    last = rows[-1]
    eq = np.array(last.get("net_worth_per_ticker") or [], float)
    tt = np.array(last.get("total_trades_per_ticker") or [], float)
    names = last.get("tickers") or []
    if eq.size == 0 or not ticks:
        print("no per-ticker equity or no tick data -- skipped")
        return

    px = np.median(np.array([t["price_per_ticker"] for t in ticks], float), axis=0)
    if px.size != eq.size:
        print("price and equity vectors disagree in length -- skipped")
        return

    pnl = eq - initial_cash
    print(f"rank corr(price, final PnL)        {_spearman(px, pnl):+.3f}")
    print("   Session 2's subsidy signature was -0.921. Either sign, strongly")
    print("   nonzero, means the cross-section is sorted by price, not by alpha.")

    with np.errstate(divide="ignore", invalid="ignore"):
        cost_per_trade = -pnl / np.where(tt > 0, tt, np.nan)
    print(f"rank corr(price, cost per trade)   {_spearman(px, cost_per_trade):+.3f}")

    # Median order notional, needed to read the fit in the right units. The
    # regression is in DOLLARS per trade, so its 1/price coefficient carries
    # a factor of notional:
    #     cost_$ = notional * per_share_$ / price + notional * prop_bps / 1e4
    # Dividing it out is what turns the number into $/share, comparable to a
    # tick. Without that step the coefficient reads ~700x too large and looks
    # like a catastrophic per-share charge rather than half a cent.
    notionals = []
    for t in ticks:
        q = t.get("filled_qty_this_tick") or []
        pr = t.get("price_per_ticker") or []
        for x, pp in zip(q, pr):
            if x:
                notionals.append(abs(x) * pp)
    median_notional = float(np.median(notionals)) if notionals else float("nan")

    ok = np.isfinite(cost_per_trade) & np.isfinite(px) & (px > 0)
    if ok.sum() >= 10:
        A = np.c_[1.0 / px[ok], np.ones(int(ok.sum()))]
        coef, *_ = np.linalg.lstsq(A, cost_per_trade[ok], rcond=None)
        resid = cost_per_trade[ok] - A @ coef
        denom = ((cost_per_trade[ok] - cost_per_trade[ok].mean()) ** 2).sum()
        ss = 1 - (resid ** 2).sum() / denom if denom > 0 else float("nan")
        print()
        print(f"fit  cost_per_trade($) = {coef[0]:.4f}/price + {coef[1]:.4f}   "
              f"R2={ss:.3f}")
        if np.isfinite(median_notional) and median_notional > 0:
            per_share = coef[0] / median_notional
            prop_bps = coef[1] / median_notional * 1e4
            rt = 2 * (prop_bps + 1e4 * per_share / float(np.median(px)))
            print(f"   median fill notional ${median_notional:,.0f} "
                  f"(from {len(notionals):,} logged fill(s))")
            print(f"   per-share term  ${per_share:.5f}/share   "
                  f"(half of a $0.01 tick is $0.00500)")
            print(f"   proportional    {prop_bps:.2f} bps/trade")
            print(f"   -> round trip   {rt:.2f} bps at the median price")
            if per_share > 0.001:
                print("   -> a per-share PENALTY dominates the cross-section.")
                print("      Cheap names pay multiples of what expensive ones do.")
            elif per_share < -0.001:
                print("   -> a per-share SUBSIDY. This is the Session 2 bug family.")
        else:
            print("   no fills in the tick log, so the coefficient cannot be put")
            print("   into $/share -- it still carries the order notional.")

    order = np.argsort(cost_per_trade)
    fin = [i for i in order if np.isfinite(cost_per_trade[i])]
    if fin and names:
        print()
        print(f"{'cheapest per trade':<28}{'dearest per trade':<28}")
        for lo, hi in zip(fin[:5], fin[::-1][:5]):
            print(f"  {names[lo]:<6} ${cost_per_trade[lo]:>7.3f} @ ${px[lo]:>8.2f}    "
                  f"  {names[hi]:<6} ${cost_per_trade[hi]:>7.3f} @ ${px[hi]:>8.2f}")


def report_kelly(rows):
    section("KELLY BINDING")
    kf = [r.get("kelly_fractional_per_ticker") for r in rows]
    kr = [r.get("kelly_raw_per_ticker") for r in rows]
    kf = np.array([x for x in kf if x], float)
    kr = np.array([x for x in kr if x], float)
    if kf.size == 0:
        print("no per-ticker Kelly data -- skipped")
        return
    floor = float(np.min(kf))
    at_floor = float(np.mean(np.abs(kf - floor) < 1e-6))
    print(f"observed floor                     {floor:.4f}")
    print(f"cells at the floor                 {at_floor * 100:.2f}%")
    print(f"cells with kelly_raw == 0          {float(np.mean(kr == 0)) * 100:.2f}%")
    print(f"kelly_raw  median / p95            {np.median(kr):.4f} / "
          f"{np.percentile(kr, 95):.4f}")
    if at_floor > 0.90:
        print()
        print("   -> the sizer is DECORATIVE: the floor is doing the sizing, and the")
        print("      measured-edge pathway has no effect on order size. Either let")
        print("      kelly_raw size (and finally exercise min_order_equity_frac) or")
        print("      remove it and set sizing explicitly.")


def report_learning(rows, bars_in_train_split=None):
    section("LEARNING DYNAMICS")
    ent = np.array([r.get("entropy_discrete", np.nan) for r in rows], float)
    tr = np.array([r.get("trades_this_rollout", 0) for r in rows], float)
    coef = [r.get("entropy_coef_discrete") for r in rows]
    steps = np.array([r.get("step", np.nan) for r in rows], float)

    def first_below(a, th):
        idx = np.flatnonzero(a < th)
        return int(idx[0]) if idx.size else None

    e50, e05 = first_below(ent, 0.5), first_below(ent, 0.05)
    print(f"entropy_discrete   {ent[0]:.3f} -> {ent[-1]:.3f}")
    print(f"   first < 0.50 at rollout {rows[e50]['rollout'] if e50 is not None else '-'}")
    print(f"   first < 0.05 at rollout {rows[e05]['rollout'] if e05 is not None else '-'}")

    if any(c is not None for c in coef):
        cc = np.array([c for c in coef if c is not None], float)
        print(f"entropy_coef       {cc[0]:.4f} -> {cc[-1]:.4f} "
              f"(min {cc.min():.4f}, max {cc.max():.4f})")
        if cc.max() > cc[0] * 1.05:
            print("   controller pushed back against a falling entropy -- working")
    else:
        print("entropy_coef       not logged (controller off, or a pre-guard run)")

    dead = int(np.sum((ent < 0.05) & (tr <= 10)))
    if dead:
        print(f"rollouts with entropy < 0.05 AND <= 10 trades: {dead} "
              f"({dead / len(rows) * 100:.0f}% of the run)")

    if np.isfinite(steps[-1]):
        print()
        print(f"steps covered      {int(steps[-1]):,}")
        if bars_in_train_split:
            frac = steps[-1] / bars_in_train_split
            print(f"train split        {bars_in_train_split:,} bars -> "
                  f"{frac * 100:.0f}% of ONE epoch")
            if frac < 1.0:
                need = math.ceil(bars_in_train_split / (steps[-1] / len(rows)))
                print(f"   a full epoch needs about {need} rollouts at this "
                      f"rollout length")
    print()
    ff = rows[-1].get("forced_flatten_count")
    ro = rows[-1].get("residual_overnight_count")
    if ff is not None or ro is not None:
        print()
        print(f"forced flattens    {ff:,}" if ff is not None else "")
        print(f"residual overnight {ro:,}" if ro is not None else "")
        if ro:
            print("   -> positions survived a session close. Expected only on")
            print("      zero-volume closing bars, where a forced close cannot")
            print("      fill; anything more is a leak, not a market limit.")
    else:
        print()
        print("overnight carry    not logged (pre-flatten run)")
    print(f"episodes completed {rows[-1].get('episode', 0)}")
    if rows[-1].get("episode", 0) == 0:
        print("   -> no episode boundary was ever reached, so any episode-terminal")
        print("      reward term (terminal_alpha) contributed exactly zero.")
    if all(r.get("sharpe") is None for r in rows):
        print("sharpe             None on every rollout -- never computed")


def report_ticks(rows, seg_paths, ticks):
    section("TICK LOG COVERAGE")
    if not ticks:
        print("no tick records found -- per-trade attribution is impossible")
        return
    ro = np.array([t.get("rollout", -1) for t in ticks])
    fills = sum(1 for t in ticks
                if any(abs(x) > 0 for x in t.get("filled_qty_this_tick", [])))
    total_trades = rows[-1].get("total_trades", 0)
    span_lo, span_hi = int(ro.min()), int(ro.max())
    run_lo, run_hi = rows[0].get("rollout", 0), rows[-1].get("rollout", 0)

    print(f"segments          {len(seg_paths)}")
    for p in seg_paths:
        print(f"   {os.path.basename(p):34s} {os.path.getsize(p) / 1e6:>8,.1f} MB")
    print(f"tick records      {len(ticks):,}")
    print(f"rollouts spanned  {span_lo}-{span_hi}  (run was {run_lo}-{run_hi})")
    print(f"ticks with a fill {fills:,} ({fills / len(ticks) * 100:.2f}%)")

    covered = (span_hi - span_lo + 1) / max(run_hi - run_lo + 1, 1)
    if covered < 0.95:
        print()
        print(f"   !! only {covered * 100:.0f}% of the run is covered. Segments were")
        print("      rotated away or not collected. Raise RunConfig.tick_backup_count,")
        print("      and download EVERY metrics.ticks.jsonl* off Kaggle -- this has")
        print("      blocked the post-mortem twice already.")
    if total_trades and fills < total_trades * 0.5:
        print(f"   !! {fills:,} fills logged against {total_trades:,} trades in the")
        print("      run: per-trade attribution covers a fraction of the activity.")


def report_regimes(ticks):
    section("BAR-OF-DAY REGIME PROFILE")
    if len(ticks) < 2 * BARS_PER_DAY:
        print("too few tick records to profile a session -- skipped")
        return
    P = np.array([t["price_per_ticker"] for t in ticks], float)
    st = np.array([t.get("step", i) for i, t in enumerate(ticks)])
    if np.any(np.diff(st) != 1):
        print("tick steps are not contiguous (segments missing?) -- profile is")
        print("computed on the contiguous run only")
        cut = int(np.flatnonzero(np.diff(st) != 1)[-1]) + 1
        P, st = P[cut:], st[cut:]
        if len(P) < 2 * BARS_PER_DAY:
            print("   not enough contiguous data -- skipped")
            return

    a = np.abs(np.diff(np.log(np.maximum(P, 1e-9)), axis=0)) * 1e4
    b = st[1:] % BARS_PER_DAY
    prof = np.array([np.median(a[b == k]) if np.any(b == k) else np.nan
                     for k in range(BARS_PER_DAY)])
    if not np.isfinite(prof).all():
        print("incomplete bar-of-day coverage -- skipped")
        return

    overall = float(np.median(a))
    peak = int(np.nanargmax(prof))
    print(f"median |move| over all bars        {overall:.2f} bps")
    print(f"richest bar-of-day index           {peak} at {prof[peak]:.2f} bps "
          f"({prof[peak] / overall:.1f}x)")
    print()
    print("   Index 0 here is an arbitrary offset, not 09:30. A single bar many")
    print("   times richer than the rest IS the overnight gap: after the RTH")
    print("   filter, one row per session carries a ~17.5-hour return. Whether")
    print("   the policy is allowed to trade it is a scope decision, not an")
    print("   accident of indexing.")
    print()
    print(f"{'bars':<10}{'median |move| bps':>20}")
    for i in range(0, BARS_PER_DAY, 6):
        g = prof[i:i + 6]
        print(f"{str(i) + '-' + str(min(i + 5, BARS_PER_DAY - 1)):<10}"
              f"{np.mean(g):>20.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Training-run post-mortem.")
    ap.add_argument("--metrics", default="/kaggle/working/logs/metrics.jsonl")
    ap.add_argument("--ticks", default=None,
                    help="glob for tick segments; defaults next to --metrics")
    ap.add_argument("--initial-cash", type=float, default=None,
                    help="per-stream starting cash (default: read from config)")
    ap.add_argument("--train-bars", type=int, default=None,
                    help="bars in the train split, for epoch coverage")
    args = ap.parse_args(argv)

    if not os.path.exists(args.metrics):
        raise SystemExit(f"No metrics file at {args.metrics}")
    rows = _load_rollouts(args.metrics)
    if not rows:
        raise SystemExit("metrics file has no rollout records")

    pattern = args.ticks or (args.metrics[:-len(".jsonl")] + ".ticks.jsonl*"
                             if args.metrics.endswith(".jsonl")
                             else args.metrics + ".ticks.jsonl*")
    seg_paths, ticks = _load_ticks(pattern)

    initial_cash = args.initial_cash
    if initial_cash is None:
        try:
            from training.config import TrainingConfig
            initial_cash = float(TrainingConfig().env.initial_cash)
        except Exception:
            initial_cash = 10_000.0

    print(f"metrics : {args.metrics}")
    print(f"ticks   : {pattern}  ({len(seg_paths)} segment(s))")
    print(f"initial_cash per stream: ${initial_cash:,.0f}")

    report_economics(rows, initial_cash)
    report_per_share_cost(rows, ticks, initial_cash)
    report_kelly(rows)
    report_learning(rows, args.train_bars)
    report_ticks(rows, seg_paths, ticks)
    report_regimes(ticks)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
