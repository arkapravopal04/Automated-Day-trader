"""Aggregate the overnight BOOK walk-forward, at the brief's bar.

P3's brief sets the pass condition as ALPHA/TURN moving toward 0.5-0.6 "with
the ratio holding above 2". AGENTS.md's close plan records that gate as
ratio > 1, which is a RELAXATION of the brief; this script reports against 2
and says so, because a gate that drifts between the brief and the write-up is
how a project talks itself into a result.

Reported on the lambda SELECTED ON TRAIN, never the best val row -- picking
lambda on val is one bit of lookahead per fold and it is the trap the sweep
exists to avoid.
"""
import glob, json, math, os, sys
import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else "logs/p3/on/book/book_f*_ib.json"
files = sorted(glob.glob(pat))
if not files:
    raise SystemExit(f"no folds matched {pat}")

rows = []
for f in files:
    d = json.load(open(f))
    hk = list(d["holds"])[0]
    h = d["holds"][hk]
    lam = h.get("chosen_lambda")
    sel = next((r for r in h["rows"] if r["lam"] == lam), None)
    if sel is None:
        rows.append((os.path.basename(f), lam, None, None)); continue
    rows.append((os.path.basename(f), lam, sel["train"], sel["val"]))

def col(key, which):
    return np.array([(r[3] if which == "val" else r[2]).get(key, np.nan)
                     if r[2] else np.nan for r in rows], dtype=float)

print(f"{len(rows)} folds from {pat}")
print(f"edge: {json.load(open(files[0]))['edge_kind']}\n")
hdr = (f"{'fold':<20}{'lam':>6}{'tr ratio':>10}{'VA RATIO':>10}{'va a/turn':>11}"
       f"{'va gross':>10}{'va cost':>9}{'va net':>9}{'va Shrp':>9}{'ex-top5':>9}")
print(hdr); print("-" * len(hdr))
for name, lam, tr, va in rows:
    if va is None:
        print(f"{name:<20}{'--':>6}  (no lambda kept the train book active)"); continue
    print(f"{name:<20}{lam:>6.2f}{tr['ratio']:>10.2f}{va['ratio']:>10.2f}"
          f"{va['alpha_per_turnover']:>11.3f}{va['gross_bps']:>10.3f}{va['cost_bps']:>9.3f}"
          f"{va['net_bps']:>9.3f}{va['sharpe']:>9.2f}{va['sharpe_ex_top5']:>9.2f}")

vr, vn, vs = col("ratio", "val"), col("net_bps", "val"), col("sharpe", "val")
va_t = col("alpha_per_turnover", "val")
def mt(a):
    a = a[np.isfinite(a)]
    if a.size < 2: return float("nan"), float("nan")
    se = a.std(ddof=1) / math.sqrt(a.size)
    return float(a.mean()), float(a.mean() / se) if se > 0 else float("nan")

print()
for label, arr, bar in (("ratio (brief's bar: > 2)", vr, 2.0),
                        ("alpha/turnover bps", va_t, None),
                        ("net bps / period", vn, 0.0),
                        ("net Sharpe (annual)", vs, None)):
    m, t = mt(arr)
    extra = ""
    if bar is not None:
        extra = f"   clears {bar:g} in {int((arr[np.isfinite(arr)] > bar).sum())}/{int(np.isfinite(arr).sum())} folds"
    print(f"  val {label:<28} mean {m:+.3f}   t(n={int(np.isfinite(arr).sum())}) {t:+.2f}{extra}")

print()
print("Read VA RATIO against 2. The train column is not evidence: lambda is chosen")
print("on it, so it is fitted by construction.")
