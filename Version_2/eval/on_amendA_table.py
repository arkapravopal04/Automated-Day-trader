"""
eval/on_amendA_table.py -- Amendment A's seven arms, side by side, five folds.

Steps 12, 13 and 14 of eval/PREREG_step10_14.md, which fixed the lambda value,
the reallocation mode, the breadth-floor test and the seven arms before any of
them was run, and declared arm `A` the new reference whatever the columns turn
out to say.

    base     train-max lambda, gross realloc, no floor   (= freeze3)
    lam      lambda FIXED at 1.0                          (clause 1)
    realloc  --cap-realloc edge                           (clause 2)
    floor    --cap-flat-if-infeasible                     (clause 3)
    A        all three                        <-- THE NEW FROZEN REFERENCE
    A1se     1-SE lambda instead of fixed     (clause 1's declared alternative)
    Anone    --cap-realloc none instead of edge (clause 2's stricter reading)

THE FIRST BLOCK IS THE REGRESSION GATE. `base` has all three options off, so it
must reproduce logs/p3/on/freeze3/ to floating-point zero. If it does not, the
code change is not inert when the amendment is off, and every arm below it is
measuring two things at once.

The three single-clause arms DECOMPOSE the change. They are not candidates to be
selected between, and neither are A1se and Anone: the reference was declared in
advance and is not chosen by reading this table.

Also usable for Study C's cells, which are arms in the same shape:

    python eval/on_amendA_table.py \\
      --fmt logs/p3/on/auction/amendA_{arm}_f{k}.json \\
      --gate logs/p3/on/freeze4/freeze4_f{k}.json \\
      --arms mocmoo,phi100,phi050,phi025,phi000,both100,both000 \\
      --title "STUDY C -- AUCTION PRICING" --no-gate-required
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
ARMS = ["base", "lam", "realloc", "floor", "A", "A1se", "Anone"]
ARM_DESC = {
    "base":    "train-max lam, gross realloc, no floor (= freeze3)",
    "lam":     "clause 1 only: lambda FIXED 1.0",
    "realloc": "clause 2 only: cap-realloc edge",
    "floor":   "clause 3 only: flat if cap infeasible",
    "A":       "all three  <-- NEW FROZEN REFERENCE",
    "A1se":    "A, but 1-SE lambda instead of fixed",
    "Anone":   "A, but cap-realloc none instead of edge",
    "mocmoo":  "C1: entry moved to the 16:00 cross",
    "phi100":  "C2: auction cost = 1.00 x measured half-spread",
    "phi050":  "C2: auction cost = 0.50 x measured half-spread",
    "phi025":  "C2: auction cost = 0.25 x measured half-spread",
    "phi000":  "C2: auction cost = 0 (pure impact -- LOWER BOUND)",
    "both100": "C1 + C2 at phi 1.00",
    "both000": "C1 + C2 at phi 0    (LOWER BOUND)",
}

# The nine validation metrics the gate is checked on. Same list on_cap_table
# uses, so a gate that passed there and fails here is a real difference and not
# a difference in what was compared.
GATE_KEYS = ("alpha_per_turnover", "cost_per_turnover", "ratio", "sharpe",
             "gross_bps", "cost_bps", "net_bps", "mean_names", "hit_rate")


def load_arm(fmt, arm, hold="1"):
    """-> {fold: (lam, val, train, ic, meta)} for whichever folds are on disk."""
    out = {}
    for k in FOLDS:
        p = fmt.format(arm=arm, k=k)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        h = d["holds"][hold if hold in d["holds"] else next(iter(d["holds"]))]
        meta = {x: d.get(x) for x in
                ("lam_select", "lam_fixed", "cap_realloc",
                 "cap_flat_if_infeasible", "exec_legs",
                 "entry_auction_bps", "exit_auction_bps")}
        lam = h["chosen_lambda"]
        if lam is None:
            out[k] = (None, {}, {}, h.get("val_ic"), meta)
            continue
        row = next(r for r in h["rows"] if r["lam"] == lam)
        out[k] = (lam, row["val"], row["train"], h.get("val_ic"), meta)
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
    print()
    print(f"{title}")
    if note:
        print(f"  {note}")
    hdr = "".join(f"{'f' + str(k):>9}" for k in FOLDS)
    print(f"  {'':<46}{hdr}{'mean':>10}{'se':>9}{'t':>8}")
    for arm in ARMS:
        d = arms.get(arm) or {}
        vals = [getter(d.get(k)) for k in FOLDS]
        m, se, t = mean_se_t(vals)
        cells = "".join(
            (fmt.format(v) if v is not None and np.isfinite(v) else f"{'-':>9}")
            for v in vals
        )
        label = f"{arm:<8} {ARM_DESC.get(arm, arm)}"
        # Five folds that agree to noise are a CONSTANT, not a significant mean.
        # Several rows here are constants by construction -- a fixed lambda, for
        # one -- and printing a t of 10^7 for them would be an artefact.
        degenerate = not (se and np.isfinite(se) and se > 1e-3 * max(abs(m), 1e-12))
        t_cell = f"{'const':>8}" if degenerate else f"{t:>8.2f}"
        print(f"  {label:<46}{cells}{m:>10.2f}{se:>9.4f}{t_cell}")


def val(rec, key):
    if not rec or not rec[1]:
        return None
    x = rec[1].get(key)
    return float(x) if x is not None else None


def trn(rec, key):
    if not rec or not rec[2]:
        return None
    x = rec[2].get(key)
    return float(x) if x is not None else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Amendment A: seven arms, five folds.")
    ap.add_argument("--fmt", default="logs/p3/on/amendA/{arm}_f{k}.json")
    ap.add_argument("--gate", default="logs/p3/on/freeze3/freeze3_f{k}.json",
                    help="the reference the FIRST arm must reproduce exactly")
    ap.add_argument("--gate-arm", default=None,
                    help="which arm the gate is checked on (default: the first)")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--hold", default="1")
    ap.add_argument("--title", default="AMENDMENT A -- STEPS 12/13/14, "
                                       "WALK-FORWARD, VAL")
    ap.add_argument("--prereg", default="eval/PREREG_step10_14.md")
    ap.add_argument("--no-gate-required", action="store_true",
                    help="report the gate delta without treating a nonzero one "
                         "as a stop. Use for Study B/C tables, where the cells "
                         "are DELIBERATELY not reproductions of the reference.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.arms:
        ARMS[:] = [a.strip() for a in args.arms.split(",") if a.strip()]
        for a in ARMS:
            ARM_DESC.setdefault(a, a)
    gate_arm = args.gate_arm or ARMS[0]

    arms = {a: load_arm(args.fmt, a, args.hold) for a in ARMS}
    have = {a: sorted(v) for a, v in arms.items()}
    if not any(have.values()):
        raise SystemExit(f"no arm jsons at {args.fmt} -- run the study's script first")

    print("=" * 116)
    print(args.title)
    print("=" * 116)
    print(f"rules:  {args.prereg} (fixed before the first run)")
    print(f"source: {args.fmt}")
    for a in ARMS:
        m = (arms[a].get(have[a][0])[4] if have[a] else {}) or {}
        cfg = " ".join(f"{x}={m[x]}" for x in
                       ("lam_select", "lam_fixed", "cap_realloc",
                        "cap_flat_if_infeasible", "exec_legs",
                        "entry_auction_bps", "exit_auction_bps")
                       if m.get(x) not in (None, False))
        print(f"  {a:<8} folds {have[a] or 'NONE'}   {cfg}")

    # ---------------- the regression gate ----------------
    print()
    print("-" * 116)
    print(f"REGRESSION GATE -- `{gate_arm}` against {args.gate}")
    print("-" * 116)
    worst, gate_rows = 0.0, 0
    print(f"  {'fold':<6}{'lambda':>9}{'worst |delta| over 9 val metrics':>42}")
    for k in FOLDS:
        p = args.gate.format(k=k)
        if k not in arms.get(gate_arm, {}) or not os.path.exists(p):
            continue
        lam_b, vb, _, _, _ = arms[gate_arm][k]
        h = json.load(open(p))["holds"]["1"]
        lam_g = h["chosen_lambda"]
        vg = next(r for r in h["rows"] if r["lam"] == lam_g)["val"]
        d = max(abs(vb[x] - vg[x]) for x in GATE_KEYS if x in vb and x in vg)
        worst = max(worst, d)
        gate_rows += 1
        flag = "" if lam_b == lam_g else f"   <-- lambda moved, was {lam_g}"
        print(f"  {k:<6}{str(lam_b):>9}{d:>42.2e}{flag}")
    if gate_rows == 0:
        print("  [gate] reference not on disk -- GATE NOT RUN")
    elif worst < 1e-12:
        print(f"  [gate] EXACT over {gate_rows} folds. The amendment being OFF is a "
              f"no-op, so what\n         follows is the amendment and nothing else.")
    elif args.no_gate_required:
        print(f"  [gate] delta {worst:.2e} -- EXPECTED for this table; these cells "
              f"are not\n         reproductions of the reference. Reported for scale "
              f"only.")
    else:
        print(f"  [gate] DRIFT {worst:.2e} -- the code change is NOT inert with the "
              f"amendment off.\n         Everything below is measuring two things at "
              f"once. STOP.")

    # ---------------- performance ----------------
    print()
    print("=" * 116)
    print("PERFORMANCE")
    print("=" * 116)
    block("lambda", arms, lambda r: (r[0] if r else None), "{:>9.2f}",
          note="constant down a column means the treatment is the same strategy "
               "on every fold.\n  That is the point of clause 1, and it is not a "
               "bug in the table.")
    block("ALPHA/TURN  bps", arms, lambda r: val(r, "alpha_per_turnover"))
    block("COST/TURN   bps", arms, lambda r: val(r, "cost_per_turnover"))
    block("ratio  (alpha/turn over cost/turn)", arms, lambda r: val(r, "ratio"),
          note="expected to FALL under every clause: (edge - lam*cost)+/cost "
               "divides by cost, so\n  the concentration and the good ratio are the "
               "same mechanism. Exchange rate, not verdict.")
    block("net Sharpe (annual)", arms, lambda r: val(r, "sharpe"))
    block("net bps / night", arms, lambda r: val(r, "net_bps"))
    block("gross bps / night", arms, lambda r: val(r, "gross_bps"))
    block("cost  bps / night", arms, lambda r: val(r, "cost_bps"))
    block("names in book", arms, lambda r: val(r, "mean_names"), "{:>9.1f}")
    block("periods the book stood FLAT, %", arms, lambda r: val(r, "flat_pct"),
          "{:>9.1f}",
          note="clause 3 removes periods rather than resizing them, so this is "
               "where its cost shows.")
    block("hit rate % (active)", arms, lambda r: val(r, "hit_rate"), "{:>9.1f}")
    block("val IC", arms, lambda r: (r[3] if r else None), "{:>9.4f}")
    block("TRAIN ratio", arms, lambda r: trn(r, "ratio"))
    block("TRAIN net Sharpe", arms, lambda r: trn(r, "sharpe"))

    # ---------------- what the amendment bought ----------------
    print()
    print("=" * 116)
    print("WHAT THE AMENDMENT BOUGHT")
    print("=" * 116)
    print("  Read against the performance block above. A performance column that is "
          "not read against\n  these is the number that got this project into trouble "
          "in the first place.")
    block("VAL  max name, share of gross (max)", arms,
          lambda r: val(r, "max_share_max"), "{:>9.3f}",
          note="the cap is A = 0.10. Above it means the bar admitted no book below "
               "the cap --\n  under clause 3 those bars are gone, so this should read "
               "0.100 flat.")
    block("VAL  max name, share of gross (p99)", arms,
          lambda r: val(r, "max_share_p99"), "{:>9.3f}")
    block("VAL  max name x equal weight (max)", arms,
          lambda r: val(r, "max_mult_max"))
    block("VAL  gross actually deployed, mean", arms,
          lambda r: val(r, "gross_deployed_mean"), "{:>9.3f}",
          note="1.000 under `gross` realloc by construction. Below 1 is clause 2 "
               "DECLINING to fund a\n  clip out of the other leg -- the risk it "
               "removed instead of transferring.")
    block("VAL  gross deployed, p10", arms,
          lambda r: val(r, "gross_deployed_p10"), "{:>9.3f}")
    block("VAL  bars stood flat: cap infeasible", arms,
          lambda r: val(r, "bar_flat_infeasible"), "{:>9.0f}",
          note="clause 3's count. Zero in every arm that does not carry it.")
    block("VAL  cap bars, breadth-infeasible", arms,
          lambda r: val(r, "cap_infeasible"), "{:>9.0f}",
          note="Appendix A's counter, breadth-only, kept unchanged so the recorded "
               "counts reproduce.")
    block("VAL  cap bars, lopsided (unconverged)", arms,
          lambda r: val(r, "cap_unconverged"), "{:>9.0f}")

    print()
    print("-" * 116)
    print("READ BEFORE QUOTING")
    print("-" * 116)
    print("  1. se is across FIVE FOLDS. The per-fold Lo standard error on a val")
    print("     Sharpe under a year is about +/-2.5, larger than any single cell.")
    print("  2. The five folds are not five independent draws of the strategy --")
    print("     they are five consecutive stretches of one market, and f4/f5 are")
    print("     adjacent.")
    print("  3. The reference was declared before the run. This table does not")
    print("     select it and must not be read as if it did.")
    print("  4. TEST HAS NEVER BEEN READ.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        json.dump({
            "source": args.fmt, "prereg": args.prereg, "title": args.title,
            "gate": args.gate, "gate_arm": gate_arm,
            "gate_worst_delta": worst, "gate_folds": gate_rows,
            "arms": {a: {str(k): {"lambda": arms[a][k][0], "val": arms[a][k][1],
                                  "train": arms[a][k][2], "val_ic": arms[a][k][3]}
                         for k in sorted(arms[a])} for a in ARMS if arms[a]},
        }, open(args.json, "w"), indent=1, default=float)
        print(f"\n[json] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
