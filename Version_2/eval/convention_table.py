"""
eval/convention_table.py -- section 02's table, rebuilt on a 1-minute panel.

WHAT SECTION 02 ESTABLISHED
---------------------------
The environment forms its observation through bar t, executes at `close[t]`
and marks at `close[t+1]`. A signal built from `close[t]` and scored against
`close[t] -> close[t+1]` shares a price print with its own target: if that
close printed on the bid the signal reads negative and the next return is
mechanically positive. That is Roll's bid-ask bounce, and it is not a trade
anyone can make.

Holding the signal FIXED and changing only the fill convention took IC(val)
from 0.0393 to 0.0060 -- roughly six and a half times of the apparent
five-minute alpha was bounce. Every tradeable convention landed in the same
place, IC 0.006-0.007 and edge 0.08-0.11 bps.

WHY IT IS WORTH RE-RUNNING NOW
------------------------------
Because at five-minute resolution that table had nothing to say. `open[t+1]`
was the ONLY price inside the fill bar that existed, so "enter at the next
bar's open" was not a choice, it was the only option, and the four tradeable
rows differed just in which aggregate they exited on. With 1-minute bars
underneath (see `intrabar.py`) the fill can land at the first minute's close,
at that minute's VWAP, or be worked across the first two minutes -- and the
spread between those rows is a direct measurement of how much edge is handed
to whoever is on the other side of the opening print.

If the intra-window rows sit at the same IC 0.006-0.007 as `open[t+1]`, then
entry timing is not where the loss is and the tradeable ceiling really is
~0.09 bps. If they recover materially toward the close-to-close row, the
bounce estimate was partly an entry-timing artefact and the tradeable edge is
larger than section 02 concluded.

THE SIGNAL IS HELD FIXED, WHICH IS THE WHOLE POINT
--------------------------------------------------
Fitting a signal against each convention's own forward return would change
the signal per row and measure nothing. The default `--signal reversal` is
target-free by construction -- the negated, cross-sectionally demeaned
trailing return -- so no fit happens at all and every row scores the same
numbers. `--signal ridge` is offered for comparability with `xsec_book.py`;
it fits ONCE, on train, against the close-to-close target, and that choice is
stated in the output because it is a choice.

EDGE, THE FOURTH COLUMN
-----------------------
Section 02's `Edge` is IC times the cross-sectional dispersion of the outcome
it is predicting:

    edge_bps = IC_val * std(cross-sectionally demeaned forward return, val)

which is the return a unit-variance-normalised signal earns per bet. It
reproduces the published column: 0.0393 x 14.2 = 0.558 and 0.0060 x 14.2 =
0.085, against 0.557 and 0.085 as printed.

TEST IS NEVER READ. Train and validation only, the same discipline the rest
of eval/ holds to.

Usage
-----
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min \\
        python eval/convention_table.py
    python eval/convention_table.py --signal ridge
    python eval/convention_table.py --lookback 3 --json logs/conventions.json
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

from paths import RAW_DIR, TRAIN_FRAC, VAL_FRAC  # noqa: E402
from eval.alpha_lab import (  # noqa: E402
    BARS_PER_DAY,
    block_ic,
    cross_sectional_demean,
    load_panel,
)
from eval.xsec_book import ridge_fit_chunked  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


# A convention is (label, entry_column, entry_offset, exit_column, exit_offset),
# offsets in bars relative to the decision bar t. The observation ends at t, so
# any entry offset of 0 is scoring a price the decision already saw -- which is
# exactly the defect the first row exists to quantify, and the reason it is
# labelled untradeable rather than dropped.
#
# The x_* columns come from intrabar.py and exist only in a directory built
# from a 1-minute cache. They are skipped with a note when absent, so this
# script still reproduces section 02's original five rows on a plain 5-minute
# panel.
CONVENTIONS = [
    ("close[t] -> close[t+1]",           "close",      0, "close",      1, "env today -- shares a print with its target"),
    ("open[t+1] -> open[t+2]",           "open",       1, "open",       2, "market orders, P1's execution frame"),
    ("vwap[t+1] -> vwap[t+2]",           "vwap",       1, "vwap",       2, "VWAP execution"),
    ("open[t+1] -> close[t+1]",          "open",       1, "close",      1, ""),
    ("vwap[t+1] -> close[t+2]",          "vwap",       1, "close",      2, "2-bar hold"),
    # --- rows that only exist with 1-minute bars underneath ---------------
    ("m1close[t+1] -> m1close[t+2]",     "x_close_m1", 1, "x_close_m1", 2, "fill at t+1's first-minute close"),
    ("m1vwap[t+1] -> m1vwap[t+2]",       "x_vwap_m1",  1, "x_vwap_m1",  2, "fill at t+1's first-minute VWAP"),
    ("m12vwap[t+1] -> m12vwap[t+2]",     "x_vwap_m12", 1, "x_vwap_m12", 2, "worked across t+1's first two minutes"),
    ("m1vwap[t+1] -> close[t+1]",        "x_vwap_m1",  1, "close",      1, "enter early, exit on the bar"),
    ("m1close[t+1] -> barvwap[t+1]",     "x_close_m1", 1, "x_vwap_full", 1, "exit at full-bar VWAP (not reachable live)"),
]

INTRABAR_COLUMNS = {"x_close_m1", "x_vwap_m1", "x_vwap_m12", "x_vwap_full"}

# Where inside its own bar each price is struck, as an ordinal. A trade must
# exit strictly LATER than it entered, and at 1-minute resolution "later" is no
# longer decided by the bar index alone: `open[t+1] -> close[t+1]` is a real
# round trip inside one bar, while `close[t+1] -> open[t+1]` is time travel.
# Comparing (bar_index, ordinal) lexicographically is what separates them, and
# it makes a malformed row fail loudly (zero valid entries) instead of quietly
# scoring a look-ahead.
#
# `x_vwap_m1` and `x_close_m1` share ordinal 1: one is the print that ends the
# first minute, the other the average across it, and neither can be exited into
# by the other.
BAR_ORDINAL = {
    "open": 0,
    "x_close_m1": 1, "x_vwap_m1": 1,
    "x_vwap_m12": 2,
    "x_vwap_full": 3, "vwap": 3,
    "close": 4,
}


def load_price_columns(index, tickers, columns):
    """{column: [T, N] float32} read from RAW_DIR onto the panel's timeline.

    Same reindex-ffill-bfill treatment `xsec_book.execution_frame` gives the
    fill column, for the same reason: the panel index is a union across
    tickers, so a name that did not print on a bar still needs a price to
    mark against, and the last trade is the honest one to use.

    Non-positive prices are masked before the fill -- a zero would produce an
    infinite log return rather than a missing one.

    `present` records, BEFORE any filling, whether the ticker actually had a
    row at each timestamp. That distinction is invisible afterwards and it
    matters here in a way it does not in `xsec_book`: there a filled price
    only marks a position that already exists, whereas this script's entire
    output is a correlation, and a filled bar contributes an exactly-zero
    return against a non-NaN signal -- a real-looking observation that adds
    nothing to the numerator and inflates the denominator, biasing every row's
    IC toward zero.
    """
    T, N = len(index), len(tickers)
    out = {c: np.full((T, N), np.nan, dtype=np.float32) for c in columns}
    present = np.zeros((T, N), dtype=bool)
    missing = set()
    for j, t in enumerate(tickers):
        rp = os.path.join(RAW_DIR, f"{t}.parquet")
        if not os.path.exists(rp):
            continue
        raw = pd.read_parquet(rp)
        present[:, j] = index.isin(raw.index)
        for c in columns:
            if c not in raw.columns:
                missing.add(c)
                continue
            col = raw[c].mask(raw[c] <= 0).reindex(index).ffill().bfill()
            out[c][:, j] = col.to_numpy(dtype=np.float32)
    return out, missing, present


def convention_return_bps(P_entry, P_exit, entry_off, exit_off, session_last_idx,
                          entry_ord=0, exit_ord=4, present=None):
    """log(exit / entry) in bps for one convention, session cap respected.

    The position is opened at bar ``t + entry_off`` and closed at bar
    ``t + exit_off``. `flatten_at_session_close` liquidates on the last bar of
    every session, so a trade whose exit falls past `session_last_idx` of the
    bar it ENTERED on is not a trade the system can place, and it returns NaN
    rather than a number nobody could have collected. That mirrors
    `alpha_lab.forward_return_bps`, generalised to the case where entry and
    exit read different columns at different offsets.

    Rows running off the end of the panel are NaN for the same reason -- there
    is no t+2 on the last bar, and extrapolating one would invent a fill.
    """
    T, N = P_entry.shape
    t = np.arange(T)
    e_idx = t + entry_off
    x_idx = t + exit_off

    later = (exit_off > entry_off) or (exit_off == entry_off and exit_ord > entry_ord)
    if not later:
        raise ValueError(
            f"convention exits at (bar+{exit_off}, ord {exit_ord}) which is not after "
            f"entry at (bar+{entry_off}, ord {entry_ord}) -- that is a look-ahead, "
            "not a fill convention"
        )
    ok = (e_idx >= 0) & (e_idx < T) & (x_idx >= 0) & (x_idx < T)
    if session_last_idx is not None:
        ok = ok & (x_idx <= np.take(session_last_idx, np.clip(e_idx, 0, T - 1)))

    out = np.full((T, N), np.nan, dtype=np.float32)
    if not ok.any():
        return out

    # A leg struck on a bar the ticker never printed is a filled price, not a
    # fill. Both legs must be real or the row is dropped -- see the note in
    # load_price_columns for why this matters more here than in xsec_book.
    cell_ok = None
    if present is not None:
        cell_ok = (present[np.clip(e_idx[ok], 0, T - 1)]
                   & present[np.clip(x_idx[ok], 0, T - 1)])
    a = P_entry[np.clip(e_idx[ok], 0, T - 1)]
    b = P_exit[np.clip(x_idx[ok], 0, T - 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.where((a > 0) & (b > 0), b / np.where(a > 0, a, np.nan), np.nan)) * 1e4
    if cell_ok is not None:
        r = np.where(cell_ok, r, np.nan)
    out[ok] = r.astype(np.float32)
    return out


def build_signal(kind, X, P, features, i_train, lookback, session_last_idx,
                 day_id=None, present=None):
    """[T, N] cross-sectionally demeaned signal, identical across every row.

    'reversal' is target-free: the negated trailing return over `lookback`
    bars, demeaned per bar. Nothing is fitted, so there is no train/val
    asymmetry and no way for a convention to influence its own score.

    'ridge' fits ONCE on train against the close-to-close target and is then
    frozen. Close-to-close is the reference because it is the convention the
    published table's first row uses; any other choice would quietly advantage
    whichever row shared it.
    """
    if kind == "reversal":
        with np.errstate(divide="ignore", invalid="ignore"):
            trail = np.full_like(P, np.nan, dtype=np.float32)
            trail[lookback:] = np.log(P[lookback:] / P[:-lookback]) * 1e4

        # THE OVERNIGHT GAP IS NOT A FIVE-MINUTE MOVE. On the first `lookback`
        # bars of a session the trailing return spans the close-to-open gap,
        # and that is a different effect at a different horizon: measured on
        # this panel, median |signal| is 24.0 bps at the session open against
        # 2.9 bps intraday, an 8.3x ratio, and those 1.3% of bars carried
        # 44.1% of the signal's total variance before this mask. Since the IC
        # is a Pearson correlation over day-sized blocks, one bar in 78 was
        # dominating the leverage in every block -- the table would have been
        # reporting overnight reversal while claiming to measure the intraday
        # effect. AGENTS.md puts median overnight |move| at 75.8 bps against
        # 9.6 midday, which is the same split arriving through the signal.
        if day_id is not None:
            T = P.shape[0]
            idx = np.arange(T)
            first_of_day = np.r_[True, np.diff(day_id) != 0]
            day_start = np.maximum.accumulate(np.where(first_of_day, idx, 0))
            trail[(idx - day_start) < lookback] = np.nan

        # A bar the ticker never printed carries a forward-filled price, so its
        # trailing return is a spurious exact zero. Same argument as the fill
        # legs; drop it rather than score it.
        if present is not None:
            trail = np.where(present, trail, np.nan)

        return cross_sectional_demean(-trail), (
            f"negated {lookback}-bar trailing return, cross-sectionally demeaned, "
            "session-gap bars and non-printing bars masked "
            "(target-free -- nothing is fitted)"
        )

    from eval.alpha_lab import forward_return_bps
    tgt = cross_sectional_demean(forward_return_bps(P, 1, session_last_idx))
    T, N, F = X.shape

    # Features are cross-sectionally demeaned before the fit, matching
    # `xsec_book.build_edge`. The target is demeaned, the book is
    # dollar-neutral, and a signal carrying a common per-bar offset would tilt
    # every name the same way; fitting the two sides on different centrings
    # would also make the two scripts' ridge edges incomparable.
    Xcs = cross_sectional_demean(X)
    Xf = Xcs[:i_train].reshape(-1, F)
    yf = tgt[:i_train].reshape(-1)
    ok = np.isfinite(yf) & np.isfinite(Xf).all(axis=1)

    # ridge_fit_chunked returns (beta, n_used), not beta -- unpacking it wrong
    # made the `is None` guard unreachable and handed einsum a tuple. It also
    # fits an UNPENALISED INTERCEPT as a trailing column, so beta has F+1
    # entries: coefficients are beta[:-1] and the intercept is beta[-1], the
    # same split build_edge applies.
    beta, n_used = ridge_fit_chunked(Xf[ok], yf[ok])
    if beta is None:
        raise SystemExit("ridge fit failed -- singular design")
    sig = (np.nan_to_num(Xcs).reshape(-1, F) @ beta[:-1] + beta[-1]).reshape(T, N).astype(np.float32)
    if present is not None:
        sig = np.where(present, sig, np.nan)
    return cross_sectional_demean(sig), (
        f"ridge over {F} features on {n_used:,} train rows, fit against the "
        "close[t]->close[t+1] target, then frozen across every row"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Section 02's execution-convention table, on the 1-minute panel.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", type=int, default=None, help="cap universe size")
    ap.add_argument("--signal", choices=("reversal", "ridge"), default="reversal",
                    help="held fixed across every row; see build_signal")
    ap.add_argument("--lookback", type=int, default=1,
                    help="bars of trailing return for --signal reversal (default 1)")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    panel = load_panel(args.tickers)
    X, P = panel["X"], panel["P"]
    features, tickers = panel["features"], panel["tickers"]
    sli = panel["session_last_idx"]
    T, N = P.shape

    i_train = int(T * TRAIN_FRAC)
    i_val = int(T * (TRAIN_FRAC + VAL_FRAC))
    print(f"[split] train 0:{i_train}  val {i_train}:{i_val}  test {i_val}:{T} (untouched)")
    print(f"[split] val window {panel['index'][i_train].date()} -> {panel['index'][i_val - 1].date()}")

    needed = sorted({c for _, ec, _, xc, _, _ in CONVENTIONS for c in (ec, xc)})
    prices, missing, present = load_price_columns(panel["index"], tickers, needed)
    print(f"[cells] {present.mean():.1%} of panel cells are real prints; "
          "the rest are forward-filled and are excluded from every row")
    if missing:
        print(f"[cols] absent from {RAW_DIR}: {sorted(missing)}")
        print("[cols] the intra-window rows need an intrabar.py output directory; "
              "skipping them and reproducing the original five rows only.")

    sig, sig_desc = build_signal(args.signal, X, P, features, i_train, args.lookback, sli,
                                 day_id=panel["day_id"], present=present)
    print(f"[signal] {args.signal}: {sig_desc}")
    print()

    hdr = (f"{'Execution convention':<32}{'IC train':>10}{'IC val':>9}{'t (val)':>9}"
           f"{'Edge':>9}{'sd fwd':>9}{'n val':>10}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for label, ec, eo, xc, xo, note in CONVENTIONS:
        if ec in missing or xc in missing:
            continue
        fwd = convention_return_bps(prices[ec], prices[xc], eo, xo, sli,
                                    entry_ord=BAR_ORDINAL[ec], exit_ord=BAR_ORDINAL[xc],
                                    present=present)
        tgt = cross_sectional_demean(fwd)

        h = max(xo - eo, 1)   # a same-bar round trip still carries one bar of risk
        block_days = int(math.ceil(h / BARS_PER_DAY)) + 1
        blocks = (panel["day_id"] // block_days).astype(np.int64)
        bl = np.repeat(blocks[:, None], N, axis=1)

        ic_tr, _, _, _ = block_ic(sig[:i_train].ravel(), tgt[:i_train].ravel(), bl[:i_train].ravel())
        ic_v, t_v, _, n_v = block_ic(sig[i_train:i_val].ravel(), tgt[i_train:i_val].ravel(),
                                     bl[i_train:i_val].ravel())
        seg = tgt[i_train:i_val]
        sd = float(np.nanstd(seg)) if np.isfinite(seg).any() else float("nan")
        edge = ic_v * sd if np.isfinite(ic_v) else np.nan

        print(f"{label:<32}{ic_tr:>10.4f}{ic_v:>9.4f}{t_v:>9.1f}{edge:>9.3f}{sd:>9.2f}{n_v:>10,}"
              + (f"   {note}" if note else ""))
        rows.append({"convention": label, "entry": [ec, eo], "exit": [xc, xo],
                     "ic_train": ic_tr, "ic_val": ic_v, "t_val": t_v,
                     "edge_bps": edge, "sd_fwd_bps": sd, "n_val": n_v, "note": note})

    print()
    print("Edge = IC(val) x sd of the cross-sectionally demeaned forward return, in bps:")
    print("the return a unit-variance signal earns per bet. Read the ratio between")
    print("rows, not the absolute level -- the level moves with the signal, the ratio")
    print("is the property of the execution convention.")
    print("Row 1 is NOT tradeable. It shares a price print with its own target and is")
    print("carried only to size the bid-ask bounce the other rows do not get.")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"signal": args.signal, "signal_desc": sig_desc,
                       "lookback": args.lookback, "n_tickers": N,
                       "raw_dir": RAW_DIR, "rows": rows}, fh, indent=2, default=float)
        print(f"\n[json] {args.json}")


if __name__ == "__main__":
    main()
