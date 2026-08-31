#!/usr/bin/env bash
# STEP 5 -- the two pre-registered risk controls, walk-forwarded.
#
# Rules are fixed in eval/PREREG_step5_risk_controls.md, written before this
# script was first run. Nothing here selects between the arms; the reference for
# step 6 is declared there to be `both`, whatever the four columns turn out to
# say.
#
#   base   the step-4 configuration, unchanged. THE REGRESSION GATE: it must
#          reproduce logs/p3/on/freeze/ to 0.00e+00, or the code change is not
#          inert when the controls are off and nothing else here is admissible.
#   cap    per-name weight cap at KAPPA = 3.0 x the bar's own equal weight
#   earn   flat into scheduled earnings, TRAIN AND VAL alike
#   both   ---> the new frozen reference
#
# Lambda is re-selected on TRAIN inside every arm. That is the point: a control
# is part of the strategy definition, so the book lambda is chosen on has to be
# the book that gets graded. Reading `base`'s lambda into the capped arm would
# grade a book nobody chose.
#
# Test is not read by any arm.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/cap

CAL=data/earnings_calendar.csv
if [ ! -f "$CAL" ]; then
  echo "missing $CAL -- run: python eval/fetch_earnings_calendar.py"
  exit 1
fi

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
        --json logs/p3/on/cap/${arm}_f${k}.json \
        > logs/p3/on/cap/${arm}_f${k}.log 2>&1
    echo "    exit $?"
  done
}

run_arm base
run_arm cap  --max-weight-mult 3.0
run_arm earn --earnings-calendar "$CAL"
run_arm both --max-weight-mult 3.0 --earnings-calendar "$CAL"

echo "=== STEP 5 WALK-FORWARD COMPLETE ==="
