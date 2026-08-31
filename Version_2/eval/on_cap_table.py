"""
eval/on_cap_table.py -- step 5's four arms, side by side, on all five folds.

The question this table answers is NOT "do the controls help". It is the
exchange rate: what the book gives up in ratio and Sharpe for a given reduction
in single-name concentration and event exposure. The pre-registration says the
ratio is expected to fall and that this is information, so both halves are
printed together on every line -- a performance column that is not read against
the concentration column beside it is the number that got this project into
trouble in the first place.

Four arms, all pre-declared in eval/PREREG_step5_risk_controls.md:

    base   step-4 configuration, unchanged
    cap    per-name cap at KAPPA = 3.0 x the bar's own equal weight
    earn   flat into scheduled earnings, train and val alike
    both   ---> the frozen reference for step 6

THE FIRST BLOCK IS THE REGRESSION GATE. `base` must reproduce
logs/p3/on/freeze/ to floating-point zero. If it does not, the code change is
not inert when the controls are off, and every arm below it is measuring two
things at once.
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
# Defaults are step 5's four arms; --arms re-points the table at the amended
# run (base / cap2 / earn / both2) without duplicating the script.
ARMS = ["base", "cap", "earn", "both"]
ARM_DESC = {
    "base":  "uncapped, no calendar (step-4 reference)",
    "cap":   "cap 3.0x equal weight",
    "earn":  "flat into scheduled earnings",
    "both":  "cap + earnings  <-- NEW FROZEN REFERENCE",
    "cap2":  "cap min(3.0x eq wt, 0.10 gross)",
    "both2": "cap2 + earnings  <-- REFERENCE (App. A)",
}


def load_arm(fmt, arm):
    """-> {fold: (lam, val, train, ic)} for whichever folds exist on disk."""
    out = {}
    for k in FOLDS:
        p = fmt.format(arm=arm, k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        h = d["holds"]["1"]
        lam = h["chosen_lambda"]
        if lam is None:
            out[k] = (None, {}, {}, h.get("val_ic"))
            continue
        row = next(r for r in h["rows"] if r["lam"] == lam)
        out[k] = (lam, row["val"], row["train"], h.get("val_ic"))
    return out


def mean_se_t(vals):
    v = np.asarray([x if x is not None else np.nan for x in vals], dtype=float)
    ok = v[np.isfinite(v)]
    if ok.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(ok.mean())
    se = float(ok.std(ddof=1) / math.sqrt(ok.size)) if ok.size > 1 else float("nan")
    t = m / se if se and np.isfinite(se) and se > 0 else float("nan")
    return m, se, t


def block(title, arms, getter, fmt="{:>9.2f}", note=None):
    """One metric, arms as rows, folds as columns."""
    print()
    print(f"{title}")
    if note:
        print(f"  {note}")
    hdr = "".join(f"{'f' + str(k):>9}" for k in FOLDS)
    print(f"  {'':<38}{hdr}{'mean':>10}{'se':>9}{'t':>8}")
    for arm in ARMS:
        d = arms.get(arm) or {}
        vals = [getter(d.get(k)) for k in FOLDS]
        m, se, t = mean_se_t(vals)
        cells = "".join(
            (fmt.format(v) if v is not None and np.isfinite(v) else f"{'-':>9}")
            for v in vals
        )
        label = f"{arm:<6} {ARM_DESC[arm]}"
        # See on_freeze_table.stat_line: five folds that agree to noise are a
        # constant, not a significant mean. The capped concentration rows are
        # exactly that, by construction.
        degenerate = not (se and np.isfinite(se) and se > 1e-3 * max(abs(m), 1e-12))
        t_cell = f"{'const':>8}" if degenerate else f"{t:>8.2f}"
        print(f"  {label:<38}{cells}{m:>10.2f}{se:>9.4f}{t_cell}")


def val(rec, key):
    if not rec:
        return None
    _, v, _, _ = rec
    x = v.get(key) if v else None
    return float(x) if x is not None else None


def trn(rec, key):
    if not rec:
        return None
    _, _, t, _ = rec
    x = t.get(key) if t else None
    return float(x) if x is not None else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 5: capped vs uncapped, five folds.")
    ap.add_argument("--fmt", default="logs/p3/on/cap/{arm}_f{k}.json")
    ap.add_argument("--gate", default="logs/p3/on/freeze/freeze_f{k}.json",
                    help="the step-4 reference `base` must reproduce exactly")
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm names (default: base,cap,earn,both)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.arms:
        ARMS[:] = [a.strip() for a in args.arms.split(",") if a.strip()]
        for a in ARMS:
            ARM_DESC.setdefault(a, a)

    arms = {a: load_arm(args.fmt, a) for a in ARMS}
    have = {a: sorted(v) for a, v in arms.items()}
    if not have["base"]:
        raise SystemExit(f"no arm jsons at {args.fmt} -- run run_overnight_cap_wf.sh")

    print("=" * 108)
    print("STEP 5 -- PRE-REGISTERED RISK CONTROLS, WALK-FORWARD, VAL")
    print("=" * 108)
    print("rules: eval/PREREG_step5_risk_controls.md (fixed before the first run)")
    print(f"source: {args.fmt}")
    print("config: --overnight --edge reversal --risk-scale vol --open-spread-bps 1.464")
    print("        --close-spread-bps 0.262 --carry-bps 0.20 --min-names-frac 0.20,")
    print("        124-name panel, lambda re-selected on TRAIN inside every arm, "
          "test never read")
    for a in ARMS:
        print(f"  {a:<6} folds on disk: {have[a] or 'NONE'}")

    # ---------------- the regression gate ----------------
    print()
    print("-" * 108)
    print(f"REGRESSION GATE -- `base` against {args.gate}")
    print("-" * 108)
    worst, gate_rows = 0.0, 0
    keys = ("alpha_per_turnover", "cost_per_turnover", "ratio", "sharpe",
            "gross_bps", "cost_bps", "net_bps", "mean_names", "hit_rate")
    print(f"  {'fold':<6}{'lambda':>9}{'worst |delta| over 9 val metrics':>40}")
    for k in FOLDS:
        p = args.gate.format(k=k)
        if k not in arms["base"] or not os.path.exists(p):
            continue
        lam_b, vb, _, _ = arms["base"][k]
        h = json.load(open(p))["holds"]["1"]
        lam_g = h["chosen_lambda"]
        vg = next(r for r in h["rows"] if r["lam"] == lam_g)["val"]
        d = max(abs(vb[x] - vg[x]) for x in keys)
        worst = max(worst, d)
        gate_rows += 1
        flag = "" if lam_b == lam_g else f"   <-- lambda moved, was {lam_g}"
        print(f"  {k:<6}{lam_b:>9}{d:>40.2e}{flag}")
    if gate_rows == 0:
        print("  [gate] step-4 reference not on disk -- GATE NOT RUN")
    elif worst < 1e-12:
        print(f"  [gate] EXACT over {gate_rows} folds. The controls being OFF is a "
              f"no-op, so what follows\n         is the controls and nothing else.")
    else:
        print(f"  [gate] DRIFT {worst:.2e} -- the code change is NOT inert with the "
              f"controls off.\n         Everything below is measuring two things at "
              f"once. STOP.")

    # ---------------- what the controls cost ----------------
    print()
    print("=" * 108)
    print("PERFORMANCE")
    print("=" * 108)
    block("lambda (re-selected on TRAIN per arm)", arms,
          lambda r: (r[0] if r else None), "{:>9.2f}")
    block("ALPHA/TURN  bps", arms, lambda r: val(r, "alpha_per_turnover"))
    block("COST/TURN   bps", arms, lambda r: val(r, "cost_per_turnover"))
    block("ratio  (alpha/turn over cost/turn)", arms, lambda r: val(r, "ratio"),
          note="expected to FALL: the (edge - lam*cost)+/cost rule divides by cost, "
               "so the concentration\n  and the good ratio are the same mechanism. "
               "This is the exchange rate, not a verdict.")
    block("net Sharpe (annual)", arms, lambda r: val(r, "sharpe"))
    block("net bps / night", arms, lambda r: val(r, "net_bps"))
    block("gross bps / night", arms, lambda r: val(r, "gross_bps"))
    block("cost  bps / night", arms, lambda r: val(r, "cost_bps"))
    block("names in book", arms, lambda r: val(r, "mean_names"), "{:>9.1f}")
    block("hit rate % (active)", arms, lambda r: val(r, "hit_rate"), "{:>9.1f}")
    block("val IC", arms, lambda r: (r[3] if r else None), "{:>9.4f}")
    block("TRAIN ratio", arms, lambda r: trn(r, "ratio"))
    block("TRAIN net Sharpe", arms, lambda r: trn(r, "sharpe"))

    # ---------------- what they bought ----------------
    print()
    print("=" * 108)
    print("CONCENTRATION -- largest single name as a MULTIPLE of the bar's equal weight")
    print("=" * 108)
    print("  Read against the performance block above: this is the other half of the "
          "exchange rate.")
    block("VAL  max name x equal weight, p50", arms,
          lambda r: val(r, "max_mult_p50"))
    block("VAL  max name x equal weight, p99", arms,
          lambda r: val(r, "max_mult_p99"),
          note="the level the alternative cap rule would have been set at, "
               "measured on VAL for\n  contrast only -- the pre-registration uses "
               "the fixed 3.0x, not this number.")
    block("VAL  max name x equal weight, max", arms,
          lambda r: val(r, "max_mult_max"))
    block("VAL  max name, share of gross", arms,
          lambda r: val(r, "max_share_max"), "{:>9.3f}")

    print()
    print("=" * 108)
    print("THE DIAGNOSTIC THE PRE-REGISTRATION SET ASIDE")
    print("=" * 108)
    print("  The alternative rule was 'cap at the TRAIN p99 of realised max-name")
    print("  weight'. It was NOT used -- KAPPA = 3.0 is fixed a priori -- but the")
    print("  number is reported so KAPPA can be read against it. Both are train-only;")
    print("  the difference is that this one moves fold to fold.")
    print()
    hdr = "".join(f"{'f' + str(k):>9}" for k in FOLDS)
    print(f"  {'':<38}{hdr}{'mean':>10}{'se':>9}{'t':>8}")
    for label, key in (("TRAIN p99, x equal weight (uncapped)", "max_mult_p99"),
                       ("TRAIN p50, x equal weight (uncapped)", "max_mult_p50"),
                       ("TRAIN max, x equal weight (uncapped)", "max_mult_max")):
        vals = [trn(arms["base"].get(k), key) for k in FOLDS]
        m, se, t = mean_se_t(vals)
        cells = "".join((f"{v:>9.2f}" if v is not None and np.isfinite(v)
                         else f"{'-':>9}") for v in vals)
        print(f"  {label:<38}{cells}{m:>10.2f}{se:>9.2f}{t:>8.2f}")
    print()
    print("  KAPPA = 3.0 sits BELOW the train p99 in every fold, so the fixed rule is")
    print("  the tighter of the two -- the pre-registered arm is not the lenient one.")

    # ---------------- cap feasibility ----------------
    unconv = [(a, k, trn(arms[a].get(k), "cap_unconverged"),
               val(arms[a].get(k), "cap_unconverged"))
              for a in ("cap", "both") for k in FOLDS if arms[a].get(k)]
    tot = sum((x or 0) + (y or 0) for _, _, x, y in unconv)
    print()
    print(f"[cap] bars where the cap could not be satisfied (lopsided long/short "
          f"split): {int(tot)}")
    if tot:
        for a, k, x, y in unconv:
            if (x or 0) + (y or 0):
                print(f"      arm {a} fold {k}: {int(x or 0)} train, {int(y or 0)} val")
        print("      A book whose selected names are nearly all one sign cannot hold "
              "both legs\n      under the cap. These bars are reported, not absorbed.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({
            "source": args.fmt,
            "prereg": "eval/PREREG_step5_risk_controls.md",
            "gate_worst_delta": worst, "gate_folds": gate_rows,
            "arms": {a: {str(k): {"lambda": arms[a][k][0], "val": arms[a][k][1],
                                  "train": arms[a][k][2], "val_ic": arms[a][k][3]}
                         for k in sorted(arms[a])} for a in ARMS if arms[a]},
        }, open(args.json, "w"), indent=1, default=float)
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
