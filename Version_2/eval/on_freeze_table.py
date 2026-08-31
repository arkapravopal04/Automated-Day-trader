"""
eval/on_freeze_table.py -- the FROZEN REFERENCE, printed from the run artefacts.

P4 is judged against these numbers, so they are generated from the jsons rather
than transcribed by hand, and the script prints its own provenance: which files
each column came from, and how far the freeze run drifted from the run it
reproduces. A reference that cannot be regenerated from disk is a number
somebody remembers, not a baseline.

Three blocks:

  OVERNIGHT   the cost-aware cross-sectional book on the overnight gap, all
              three flatteries corrected plus the breadth floor. This is the
              best thing measured in this project and it is the one that
              matters for P4.
  INTRADAY    the same book on the 5-minute schedule, for the contrast the
              overnight verdict rests on -- ALPHA/TURN is sixteen times larger
              overnight, and the ratio still fails, because cost per unit
              turnover rises with it.
  DRIFT       freeze run against the run it reproduces, cell by cell. Anything
              non-zero here means the cache moved under the result.

`t(n=5)` is the across-fold t of the mean: mean / (sd / sqrt 5). Five folds is
a small sample and the t is quoted so nobody has to reconstruct how small.

AND FIVE IS NOT THE RIGHT DENOMINATOR. The walk-forward expands, so the five
training sets are NESTED (adjacent folds share 80-88% of their training data)
and the five validation windows TILE ONE CONTIGUOUS SPAN with no gap between
them -- val_2 begins the trading day after val_1 ends, and so on. The folds are
five consecutive slices of a single out-of-sample record, not five draws, and
the slice boundaries carry no information. `eval/on_independence.py` pools the
five windows into the one session series they actually are and reports the
Sharpe and the ratio with a Newey-West standard error and a block-bootstrap
interval, alongside the f4+f5-merged version. EVERY t IN THIS TABLE IS TO BE
READ AGAINST THAT OUTPUT. On the step-4 reference the merged version moves the
mean Sharpe from +0.33 to -0.04, and the ratio's bootstrap interval is
[+0.22, +2.13] against a bar of 2 -- neither of which is visible here.
"""

from __future__ import annotations

import argparse
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

FOLDS = (1, 2, 3, 4, 5)


def load_overnight(path_fmt):
    """-> [(fold, lam, val_dict, train_dict)] for folds that exist on disk."""
    out = []
    for k in FOLDS:
        p = path_fmt.format(k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        h = d["holds"]["1"]
        lam = h["chosen_lambda"]
        row = next(r for r in h["rows"] if r["lam"] == lam)
        out.append((k, lam, row["val"], row["train"], h.get("val_ic"), h.get("val_t")))
    return out


def load_config(path_fmt):
    """The risk-control settings the freeze jsons were actually written under.

    Read off disk rather than typed into the header, because a header that is
    maintained by hand is the one thing in this script that can disagree with
    the numbers beneath it. Returns a description and raises if the folds do not
    agree with each other -- a five-fold table whose folds ran under different
    controls is not a table.
    """
    seen = set()
    for k in FOLDS:
        p = path_fmt.format(k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        seen.add((d.get("max_weight_mult"), bool(d.get("earnings_calendar"))))
    if not seen:
        return "unknown"
    if len(seen) > 1:
        raise SystemExit(f"folds disagree on the risk controls: {sorted(seen)} -- "
                         "these are not five folds of one configuration")
    mult, earn = seen.pop()
    parts = []
    parts.append(f"--max-weight-mult {mult:g} (per-name cap at {mult:g}x the bar's "
                 f"equal weight)" if mult is not None else "NO per-name cap")
    parts.append("--earnings-calendar (flat into scheduled prints, train and val)"
                 if earn else "NO earnings exclusion")
    return "; ".join(parts)


def stat_line(label, vals, fmt="{:>9.2f}"):
    v = np.asarray([x for x in vals], dtype=float)
    ok = v[np.isfinite(v)]
    mean = float(ok.mean()) if ok.size else float("nan")
    se = float(ok.std(ddof=1) / math.sqrt(ok.size)) if ok.size > 1 else float("nan")
    # A t on five folds that agree to floating-point noise is not a strong
    # result, it is a constant. The capped rows are exactly that -- the cap
    # binds to 3.00 in every fold by construction -- and printing t = 9.1e6
    # there would be a spurious significance claim sitting in a frozen
    # reference table, on top of overflowing its own column.
    degenerate = not (se and np.isfinite(se) and se > 1e-3 * max(abs(mean), 1e-12))
    t = float("nan") if degenerate else mean / se
    cells = "".join(fmt.format(x) for x in v)
    t_cell = f"{'const':>8}" if degenerate and ok.size > 1 else f"{t:>8.2f}"
    return f"{label:<24}{cells}{mean:>10.2f}{se:>9.4f}{t_cell}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the frozen walk-forward reference.")
    ap.add_argument("--freeze", default="logs/p3/on/freeze/freeze_f{k}.json")
    ap.add_argument("--against", default="logs/p3/on/final/final_f{k}.json",
                    help="the earlier run the freeze reproduces")
    ap.add_argument("--intraday", default="logs/p3/wf/ratio_f{k}_ib.json")
    ap.add_argument("--intraday-config", default="all names, sized edge / cost, hold 6")
    ap.add_argument("--delta-label", default=None,
                    help="what the bottom block IS. Default reads as reproduction "
                         "drift, which is only true when --against is the same "
                         "configuration. Point --against at a DIFFERENT variant and "
                         "the deltas are a deliberate change, not drift, and this "
                         "argument must say so -- a block labelled 'drift' that is "
                         "showing an intended difference trains the reader to "
                         "ignore the one check that catches a moving cache.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    fr = load_overnight(args.freeze)
    if not fr:
        raise SystemExit(f"no freeze jsons at {args.freeze} -- run run_overnight_freeze_wf.sh")
    ag = load_overnight(args.against)

    hdr = "".join(f"{'f' + str(k):>9}" for k, *_ in fr)
    print("=" * 100)
    print("FROZEN REFERENCE -- overnight cross-sectional book, walk-forward, VAL")
    print("=" * 100)
    print(f"source: {args.freeze}")
    print(f"config: --overnight --edge reversal --risk-scale vol "
          f"--open-spread-bps 1.464 --close-spread-bps 0.262")
    print(f"        --carry-bps 0.20 --min-names-frac 0.20, 124-name panel, "
          f"lambda selected on TRAIN, test never read")
    print(f"controls: {load_config(args.freeze)}")
    print()
    print(f"{'':<24}{hdr}{'mean':>10}{'se':>9}{'t':>8}")
    print("-" * 100)
    print(stat_line("lambda (train-chosen)", [lam for _, lam, *_ in fr]))
    print(stat_line("ALPHA/TURN  bps", [v["alpha_per_turnover"] for _, _, v, _, _, _ in fr]))
    print(stat_line("COST/TURN   bps", [v["cost_per_turnover"] for _, _, v, _, _, _ in fr]))
    print(stat_line("ratio", [v["ratio"] for _, _, v, _, _, _ in fr]))
    print(stat_line("net Sharpe (annual)", [v["sharpe"] for _, _, v, _, _, _ in fr]))
    print(stat_line("  Sharpe ex-top5", [v["sharpe_ex_top5"] for _, _, v, _, _, _ in fr]))
    print(stat_line("gross bps / night", [v["gross_bps"] for _, _, v, _, _, _ in fr]))
    print(stat_line("cost  bps / night", [v["cost_bps"] for _, _, v, _, _, _ in fr]))
    print(stat_line("net   bps / night", [v["net_bps"] for _, _, v, _, _, _ in fr]))
    print(stat_line("names in book", [v["mean_names"] for _, _, v, _, _, _ in fr]))
    print(stat_line("flat %", [v["flat_pct"] for _, _, v, _, _, _ in fr]))
    print(stat_line("hit rate % (active)", [v["hit_rate"] for _, _, v, _, _, _ in fr]))
    print(stat_line("val IC", [a for *_, a, _ in fr], fmt="{:>9.4f}"))
    print()
    print(stat_line("TRAIN ratio", [t["ratio"] for _, _, _, t, _, _ in fr]))
    print(stat_line("TRAIN net Sharpe", [t["sharpe"] for _, _, _, t, _, _ in fr]))

    # SINGLE-NAME CONCENTRATION, IN THE TABLE ITSELF.
    #
    # The step-4 reference was frozen without it, and the largest thing the
    # fold-2 investigation turned up was a 37%-of-gross single name that no
    # column on that table would have shown. It is quoted here in multiples of
    # the bar's own equal weight so the cap value reads directly against it.
    if any("max_mult_p50" in v for _, _, v, _, _, _ in fr):
        print()
        print(stat_line("max name x eq wt, p50", [v.get("max_mult_p50", float("nan"))
                                                  for _, _, v, _, _, _ in fr]))
        print(stat_line("max name x eq wt, p99", [v.get("max_mult_p99", float("nan"))
                                                  for _, _, v, _, _, _ in fr]))
        print(stat_line("max name x eq wt, max", [v.get("max_mult_max", float("nan"))
                                                  for _, _, v, _, _, _ in fr]))
        print(stat_line("max name, share of gross", [v.get("max_share_max", float("nan"))
                                                     for _, _, v, _, _, _ in fr],
                        fmt="{:>9.3f}"))

    print()
    print("AGAINST THE BRIEF'S BARS")
    rr = np.array([v["ratio"] for _, _, v, _, _, _ in fr])
    ss = np.array([v["sharpe"] for _, _, v, _, _, _ in fr])
    se_r = rr.std(ddof=1) / math.sqrt(rr.size)
    se_s = ss.std(ddof=1) / math.sqrt(ss.size)
    print(f"  ratio > 2          mean {rr.mean():.2f} +/- {se_r:.2f} -> clears in "
          f"{(rr > 2).sum()} of {rr.size} folds. "
          f"{'PASS' if rr.mean() - se_r > 2 else 'FAIL'}")
    print(f"  net Sharpe >= 1.5  mean {ss.mean():.2f} +/- {se_s:.2f} -> clears in "
          f"{(ss >= 1.5).sum()} of {ss.size} folds. "
          f"{'PASS' if ss.mean() - se_s >= 1.5 else 'FAIL'}")
    print()
    print("  The +/- above is sd/sqrt(5) across folds that are CONTIGUOUS SLICES")
    print("  of one out-of-sample record, not five independent draws. Run")
    print("  eval/on_independence.py for the pooled session-level interval before")
    print("  quoting either line as a pass or a fail.")

    # ---------------- drift ----------------
    if ag:
        same_config = load_config(args.freeze) == load_config(args.against)
        label = args.delta_label or ("REPRODUCTION DRIFT" if same_config else
                                     "DELTA AGAINST A DIFFERENT CONFIGURATION "
                                     "-- NOT A DRIFT CHECK")
        print()
        print("-" * 100)
        print(f"{label} -- freeze against {args.against}")
        if not same_config:
            print(f"  freeze  : {load_config(args.freeze)}")
            print(f"  against : {load_config(args.against)}")
            print("  These are two different books. The numbers below are what the "
                  "controls did, and\n  they say NOTHING about whether the cache "
                  "moved -- that check needs an --against\n  run under the same "
                  "configuration.")
        print("-" * 100)
        worst = 0.0
        print(f"{'fold':<6}{'lambda':>9}{'d ALPHA/TURN':>15}{'d COST/TURN':>14}"
              f"{'d ratio':>10}{'d Sharpe':>11}")
        for (k, lam, v, _, _, _), (k2, lam2, v2, _, _, _) in zip(fr, ag):
            same_lam = "" if lam == lam2 else f"  <-- was {lam2}"
            ds = [v[key] - v2[key] for key in
                  ("alpha_per_turnover", "cost_per_turnover", "ratio", "sharpe")]
            worst = max(worst, max(abs(x) for x in ds))
            print(f"{k:<6}{lam:>9}{ds[0]:>15.2e}{ds[1]:>14.2e}{ds[2]:>10.2e}"
                  f"{ds[3]:>11.2e}{same_lam}")
        if same_config:
            verdict = ("EXACT -- the cache and the code are unchanged" if worst < 1e-9
                       else f"DRIFT {worst:.2e} -- something moved; do not freeze "
                            "until explained")
            print(f"[drift] {verdict}")
        else:
            print(f"[delta] largest single cell change {worst:.3f} -- this is the "
                  f"effect of the controls,\n        not a reproduction check.")

    # ---------------- intraday contrast ----------------
    rows = []
    for k in FOLDS:
        p = args.intraday.format(k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        r = next((x for x in d["rows"] if x["config"] == args.intraday_config), None)
        if r:
            rows.append((k, r["val"]))
    if rows:
        print()
        print("-" * 100)
        print(f"INTRADAY CONTRAST -- {args.intraday_config}, same folds, VAL")
        print("-" * 100)
        hdr2 = "".join(f"{'f' + str(k):>9}" for k, _ in rows)
        print(f"{'':<24}{hdr2}{'mean':>10}{'se':>9}{'t':>8}")
        print(stat_line("ALPHA/TURN  bps", [v["alpha_per_turnover"] for _, v in rows]))
        print(stat_line("COST/TURN   bps", [v["cost_per_turnover"] for _, v in rows]))
        print(stat_line("ratio", [v["ratio"] for _, v in rows]))
        print(stat_line("net Sharpe (annual)", [v["sharpe"] for _, v in rows]))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({
            "source": args.freeze,
            "folds": [{"fold": k, "lambda": lam, "val": v, "train": t,
                       "val_ic": a, "val_ic_t": b} for k, lam, v, t, a, b in fr],
            "mean": {key: float(np.mean([v[key] for _, _, v, _, _, _ in fr]))
                     for key in ("alpha_per_turnover", "cost_per_turnover",
                                 "ratio", "sharpe", "gross_bps", "cost_bps",
                                 "net_bps", "sharpe_ex_top5")},
        }, open(args.json, "w"), indent=1)
        print(f"\n[json] wrote {args.json}")


if __name__ == "__main__":
    main()
