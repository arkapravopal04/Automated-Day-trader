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


def stat_line(label, vals, fmt="{:>9.2f}"):
    v = np.asarray([x for x in vals], dtype=float)
    ok = v[np.isfinite(v)]
    mean = float(ok.mean()) if ok.size else float("nan")
    se = float(ok.std(ddof=1) / math.sqrt(ok.size)) if ok.size > 1 else float("nan")
    t = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
    cells = "".join(fmt.format(x) for x in v)
    return f"{label:<24}{cells}{mean:>10.2f}{se:>9.2f}{t:>8.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the frozen walk-forward reference.")
    ap.add_argument("--freeze", default="logs/p3/on/freeze/freeze_f{k}.json")
    ap.add_argument("--against", default="logs/p3/on/final/final_f{k}.json",
                    help="the earlier run the freeze reproduces")
    ap.add_argument("--intraday", default="logs/p3/wf/ratio_f{k}_ib.json")
    ap.add_argument("--intraday-config", default="all names, sized edge / cost, hold 6")
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

    # ---------------- drift ----------------
    if ag:
        print()
        print("-" * 100)
        print(f"REPRODUCTION DRIFT -- freeze against {args.against}")
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
        verdict = ("EXACT -- the cache and the code are unchanged" if worst < 1e-9
                   else f"DRIFT {worst:.2e} -- something moved; do not freeze until explained")
        print(f"[drift] {verdict}")

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
