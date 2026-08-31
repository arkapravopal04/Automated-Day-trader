#!/usr/bin/env bash
# STEP 5, AMENDED -- the per-name cap with the ABSOLUTE bound it should have had.
#
# Rules are fixed in Appendix A of eval/PREREG_step5_risk_controls.md, written
# before this script was first run. The amendment closes a DEFECT: the relative
# cap KAPPA/n_selected permits 50% of gross on a six-name book, which is more
# than the 37% concentration Control A was written to forbid, and folds 1 and 5
# of the `both` arm did exactly that. The amended rule is
#
#     |w_i| <= min( KAPPA / n_selected , A )      KAPPA = 3.0, A = 0.10
#
# and it is TIGHTER THAN THE RULE IT REPLACES ON EVERY BAR. It cannot be a
# search for a better number; it can only make the reported book worse.
#
#   base   step-4 configuration, unchanged.        GATE 1 and GATE 2.
#   cap2   min(3.0x equal weight, 0.10 of gross)
#   earn   flat into scheduled earnings.           GATE 2.
#   both2  cap2 + earnings  ---> the new frozen reference, declared in advance
#
# TWO GATES, both of which must pass before any cell below them is read:
#   1. `base` reproduces logs/p3/on/freeze/ exactly, as in step 5.
#   2. `base` and `earn` reproduce logs/p3/on/cap/{base,earn}_* exactly. Neither
#      passes --max-weight-frac, so the absolute-cap code change must be inert in
#      both. If it is not, the change is not confined to the cap and nothing
#      below is admissible.
#
# Lambda is re-selected on TRAIN inside every arm, for the reason step 5 gave:
# a control is part of the strategy definition, so the book lambda is chosen on
# has to be the book that gets graded.
#
# Test is not read by any arm.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/cap2

CAL=data/earnings_calendar.csv
if [ ! -f "$CAL" ]; then
  echo "missing $CAL -- run: python eval/fetch_earnings_calendar.py"
  exit 1
fi

KAPPA=3.0
A=0.10

COMMON="--overnight --edge reversal --risk-scale vol \
  --open-spread-bps 1.464 --close-spread-bps 0.262 --carry-bps 0.20 \
  --min-names-frac 0.20"

run_arm () {                       # $1 arm name, $2.. extra flags
  arm="$1"; shift
  for k in 1 2 3 4 5; do
    TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
    echo "=== arm $arm fold $k TRAIN_FRAC=$TF ==="
    env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
      TRADING_RAW_DIR=data/parquet_agg5 \
      TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
      python -u eval/xsec_book.py $COMMON "$@" \
        --json logs/p3/on/cap2/${arm}_f${k}.json \
        > logs/p3/on/cap2/${arm}_f${k}.log 2>&1
    echo "    exit $?"
  done
}

run_arm base
run_arm cap2  --max-weight-mult $KAPPA --max-weight-frac $A
run_arm earn  --earnings-calendar "$CAL"
run_arm both2 --max-weight-mult $KAPPA --max-weight-frac $A --earnings-calendar "$CAL"

echo "=== STEP 5 (AMENDED) WALK-FORWARD COMPLETE ==="
