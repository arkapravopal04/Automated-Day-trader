"""Measure the effective spread at the OPEN against midday, from 1-minute bars.

WHY. `EnvConfig` prices spread at 0.0 bps beyond a half-tick floor, and it was
calibrated on intraday 5-minute bars. The overnight trade's exit leg is the
09:30 print, where spreads are materially wider than at midday, so the
overnight book has been charged an intraday cost for an opening-auction fill.
That is one of the three things flattering its ratio.

Rather than assume a multiplier, measure one. Corwin & Schultz (2012) recover a
proportional effective spread from consecutive periods' high/low alone: the
sum of two single-period ranges scales with volatility ONE way and with the
spread ANOTHER, and the two-period range separates them. No quote data needed,
which matters because quotes are P3 bullet 2 and are not in hand.

    beta  = E[ ln(H_t/L_t)^2 + ln(H_t1/L_t1)^2 ]
    gamma = ln( max(H_t,H_t1) / min(L_t,L_t1) )^2
    alpha = (sqrt(2b) - sqrt(b))/(3-2sqrt2) - sqrt(g/(3-2sqrt2))
    S     = 2(e^a - 1)/(1 + e^a)

Negative estimates are set to zero per the paper -- they arise when volatility
swamps the spread in a pair, and averaging them in as negative spreads would
bias the estimate down, which is the WRONG direction for a check whose purpose
is to stop understating cost.

    python eval/measure_open_spread.py --tickers 40
"""
import argparse, glob, json, math, os, sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

K = 3.0 - 2.0 * math.sqrt(2.0)


def corwin_schultz(h, l):
    """Proportional spread per adjacent pair, from 1-minute highs/lows."""
    h = np.asarray(h, float); l = np.asarray(l, float)
    ok = (h > 0) & (l > 0) & (h >= l)
    h = np.where(ok, h, np.nan); l = np.where(ok, l, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(h / l) ** 2
        beta = r[:-1] + r[1:]
        H2 = np.maximum(h[:-1], h[1:]); L2 = np.minimum(l[:-1], l[1:])
        gamma = np.log(H2 / L2) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K - np.sqrt(gamma / K)
        S = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return np.where(np.isfinite(S), np.maximum(S, 0.0), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/parquet_1min")
    ap.add_argument("--tickers", type=int, default=40)
    ap.add_argument("--json", default="logs/p3/open_spread.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw, "*.parquet")))[:args.tickers]
    buckets = {"open_1min": [], "open_5min": [], "midday": [], "close_5min": []}
    for f in files:
        df = pd.read_parquet(f, columns=["high", "low"])
        if len(df) < 1000:
            continue
        ny = df.index.tz_convert("America/New_York")
        mins = ny.hour * 60 + ny.minute - (9 * 60 + 30)
        S = corwin_schultz(df["high"].to_numpy(), df["low"].to_numpy())
        m = mins.to_numpy()[:-1]           # label each pair by its FIRST minute
        buckets["open_1min"].append(np.nanmedian(S[m == 0]))
        buckets["open_5min"].append(np.nanmedian(S[(m >= 0) & (m < 5)]))
        buckets["midday"].append(np.nanmedian(S[(m >= 120) & (m < 270)]))
        buckets["close_5min"].append(np.nanmedian(S[(m >= 385) & (m <= 389)]))

    out = {}
    print(f"{len(buckets['midday'])} tickers, Corwin-Schultz on 1-minute bars\n")
    print(f"{'bucket':<14}{'median spread bps':>20}{'vs midday':>12}")
    print("-" * 46)
    mid = float(np.nanmedian(buckets["midday"])) * 1e4
    for name in ("open_1min", "open_5min", "midday", "close_5min"):
        v = float(np.nanmedian(buckets[name])) * 1e4
        out[name] = v
        print(f"{name:<14}{v:>20.3f}{(v / mid if mid > 0 else np.nan):>11.2f}x")
    out["open_vs_midday_mult"] = out["open_1min"] / mid if mid else float("nan")
    print()
    print("The overnight trade's EXIT leg is the 09:30 print, so open_1min is the")
    print("bucket that applies to it. Entry is the 15:55 bar, close_5min.")
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(out, open(args.json, "w"), indent=2)
    print(f"[json] {args.json}")


if __name__ == "__main__":
    main()
