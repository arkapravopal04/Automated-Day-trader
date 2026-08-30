"""Aggregate the overnight walk-forward: five folds, the mean, and a t.

Step 0 of the P3 close plan makes this mandatory. Fold 1 of the overnight run
gives IC +0.044 (t 2.5) and fold 2 gives -0.014 (t -0.9) on identical code,
panel and signal -- so any single fold, quoted alone, is a number about WHEN IT
WAS MEASURED. Only the across-fold distribution is a claim about the effect.

The per-fold t printed by convention_table is a WITHIN-fold statistic over
session blocks; it says the IC differs from zero inside that window. The t
computed HERE is ACROSS folds, n=5, and it is the one that speaks to whether
the effect exists at all. At n=5, |t| > 2.78 is the 5% two-sided threshold --
the same bar AGENTS.md applies to the intrabar lift.
"""
import glob, json, math, os, sys
import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else "logs/p3/on/wf/on_rev_f*.json"
files = sorted(glob.glob(pat))
if not files:
    raise SystemExit(f"no folds matched {pat}")

folds = []
for f in files:
    d = json.load(open(f))
    folds.append((os.path.basename(f), {r["convention"]: r for r in d["rows"]}))

convs = [r for r in folds[0][1]]
print(f"{len(folds)} folds from {pat}\n")
hdr = f"{'convention':<34}" + "".join(f"{('f' + str(i + 1)):>9}" for i in range(len(folds))) \
      + f"{'mean':>9}{'t(n=' + str(len(folds)) + ')':>9}{'>0':>5}"
print(hdr); print("-" * len(hdr))

out = {}
for c in convs:
    ics = np.array([fd[c]["ic_val"] for _, fd in folds], dtype=float)
    ok = np.isfinite(ics)
    m = float(ics[ok].mean()) if ok.any() else float("nan")
    se = float(ics[ok].std(ddof=1) / math.sqrt(ok.sum())) if ok.sum() > 1 else float("nan")
    t = m / se if se and se > 0 else float("nan")
    npos = int((ics[ok] > 0).sum())
    print(f"{c:<34}" + "".join(f"{v:>9.4f}" for v in ics)
          + f"{m:>9.4f}{t:>9.2f}{npos:>3}/{ok.sum()}")
    out[c] = {"ics": list(ics), "mean": m, "t": t, "n_pos": npos}

    # Edge in bps per bet, at the fold-mean IC, against the round trip. This is
    # the number the cost hurdle is applied to, so it is reported alongside --
    # but it inherits the mean's uncertainty and means nothing if t is small.
    eds = np.array([fd[c]["edge_bps"] for _, fd in folds], dtype=float)
    out[c]["edge_mean_bps"] = float(np.nanmean(eds))

print()
print("t here is ACROSS folds (n=%d); |t| > 2.78 is the 5%% two-sided bar at n=5." % len(folds))
print("The per-fold t inside each log is a WITHIN-fold statistic and is not this.")
best = max(out.items(), key=lambda kv: abs(kv[1]["t"]) if math.isfinite(kv[1]["t"]) else -1)
print()
for c, d in out.items():
    print(f"{c:<34} mean IC {d['mean']:+.4f}  t {d['t']:+.2f}  "
          f"mean edge {d['edge_mean_bps']:+.2f} bps  {d['n_pos']}/{len(folds)} folds positive")
