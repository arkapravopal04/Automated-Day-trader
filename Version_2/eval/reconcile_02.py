"""
eval/reconcile_02.py -- why does the rebuilt section 02 table not reproduce
section 02's numbers?

Section 02 reports, for the same reversal signal on close[t] -> close[t+1]:

    IC train 0.0545   IC val 0.0393   t(val) 27.2

The rebuild on the 1-minute panel returns IC val -0.0032 at t = -0.6. Before
any of that table can be read as a P3 result, the difference has to be
attributed to a specific methodological choice rather than waved at. This
script varies one choice at a time and prints what each is worth.

The choices under test, and why each is a candidate:

  clustering   A t of 27.2 on ~1.1M stacked observations is what a POOLED
               correlation gives: 0.0393 * sqrt(1.1e6) = 41. Block-clustered
               t-stats over day-sized blocks are far smaller, because
               observations inside one day share almost all of their outcome.
               If this is the whole story, the IC POINT ESTIMATES agree and
               only the t-stats differ.

  session gap  The rebuild masks the first bar of each session, where a 1-bar
               trailing return is the overnight gap rather than a 5-minute
               move. Measured: those 1.3% of bars carried 44% of the signal's
               variance. Section 02 does not mention such a mask.

  fill mask    The rebuild drops bars a ticker did not actually print, which
               otherwise enter as exact-zero returns against a live signal.

  split        paths.py splits 0.80/0.10/0.10 by row position, putting
               validation at 2025-06-24 -> 2026-01-27. Section 02 names its
               validation window as 2024-12 -> 2026-01, which is roughly twice
               as long and starts six months earlier.

Nothing here changes the rebuilt table. It exists to say which of the four
explains the disagreement, so the table can be quoted with that attached.
TEST IS NEVER READ.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import TRAIN_FRAC, VAL_FRAC  # noqa: E402
from eval.alpha_lab import BARS_PER_DAY, block_ic, cross_sectional_demean, load_panel  # noqa: E402
from eval.convention_table import (  # noqa: E402
    BAR_ORDINAL, convention_return_bps, load_price_columns,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def pooled_ic(pred, actual):
    """Unclustered Pearson IC over stacked observations, with its naive t.

    This is the estimator whose t-stat matches section 02's magnitude. It
    treats every (bar, ticker) cell as independent, which they are not.
    """
    ok = np.isfinite(pred) & np.isfinite(actual)
    x, y = pred[ok], actual[ok]
    if x.size < 100:
        return np.nan, np.nan, 0
    ic = float(np.corrcoef(x, y)[0, 1])
    t = ic * math.sqrt(max(x.size - 2, 1)) / math.sqrt(max(1 - ic * ic, 1e-12))
    return ic, t, int(x.size)


def main():
    panel = load_panel(None)
    P, day_id, sli = panel["P"], panel["day_id"], panel["session_last_idx"]
    tickers, index = panel["tickers"], panel["index"]
    T, N = P.shape
    i_train = int(T * TRAIN_FRAC)
    i_val = int(T * (TRAIN_FRAC + VAL_FRAC))
    print(f"[split] val {index[i_train].date()} -> {index[i_val - 1].date()}  "
          f"({i_val - i_train:,} bars)")

    prices, missing, present = load_price_columns(index, tickers, ["close"])
    close = prices["close"]

    fwd = convention_return_bps(close, close, 0, 1, sli,
                                entry_ord=BAR_ORDINAL["close"],
                                exit_ord=BAR_ORDINAL["close"], present=None)
    fwd_masked = convention_return_bps(close, close, 0, 1, sli,
                                       entry_ord=BAR_ORDINAL["close"],
                                       exit_ord=BAR_ORDINAL["close"], present=present)

    # the signal, in two versions: with and without the session-gap mask
    with np.errstate(divide="ignore", invalid="ignore"):
        trail = np.full_like(P, np.nan, dtype=np.float32)
        trail[1:] = np.log(P[1:] / P[:-1]) * 1e4
    raw_sig = cross_sectional_demean(-trail)

    idx = np.arange(T)
    first_of_day = np.r_[True, np.diff(day_id) != 0]
    day_start = np.maximum.accumulate(np.where(first_of_day, idx, 0))
    gap_bar = (idx - day_start) < 1
    trail_masked = trail.copy()
    trail_masked[gap_bar] = np.nan
    masked_sig = cross_sectional_demean(-np.where(present, trail_masked, np.nan))

    blocks = np.repeat((day_id // 2).astype(np.int64)[:, None], N, axis=1)

    variants = [
        ("section 02 as published",              raw_sig,    fwd,        "pooled"),
        ("+ block-clustered t",                  raw_sig,    fwd,        "block"),
        ("+ drop non-printing bars",             raw_sig,    fwd_masked, "block"),
        ("+ mask session-gap bar (the rebuild)", masked_sig, fwd_masked, "block"),
        ("rebuild, but pooled t",                masked_sig, fwd_masked, "pooled"),
    ]

    hdr = f"{'variant':<38}{'IC train':>10}{'IC val':>10}{'t (val)':>10}{'n val':>14}"
    print()
    print("close[t] -> close[t+1], the row section 02 reports as IC val 0.0393, t 27.2")
    print(hdr)
    print("-" * len(hdr))
    for label, sig, target, est in variants:
        tgt = cross_sectional_demean(target)
        if est == "pooled":
            ic_tr, _, _ = pooled_ic(sig[:i_train].ravel(), tgt[:i_train].ravel())
            ic_v, t_v, n_v = pooled_ic(sig[i_train:i_val].ravel(), tgt[i_train:i_val].ravel())
        else:
            ic_tr, _, _, _ = block_ic(sig[:i_train].ravel(), tgt[:i_train].ravel(),
                                      blocks[:i_train].ravel())
            ic_v, t_v, _, n_v = block_ic(sig[i_train:i_val].ravel(), tgt[i_train:i_val].ravel(),
                                         blocks[i_train:i_val].ravel())
        print(f"{label:<38}{ic_tr:>10.4f}{ic_v:>10.4f}{t_v:>10.1f}{n_v:>14,}")

    print()
    print("Read down the column: each row adds ONE choice to the row above it.")
    print("Whichever step moves IC val is the one that explains the disagreement;")
    print("a step that moves only t(val) is a standard-error question, not a")
    print("disagreement about the size of the effect.")


if __name__ == "__main__":
    main()
