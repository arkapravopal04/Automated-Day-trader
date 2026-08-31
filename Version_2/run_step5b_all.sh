#!/usr/bin/env bash
# STEP 5b -- re-run the two AMENDED arms, then the re-freeze.
#
# The two arms are re-run so they carry the feasible-bars concentration
# diagnostic. The books are DETERMINISTIC and both regression gates already
# passed, so no arm number moves; what is added is max_share measured over the
# bars that ADMIT a capped book, reported separately from the bars that admit
# none. A 2-name dollar-neutral book is 50/50 under any cap whatsoever, and an
# unconditional max makes the rule look broken on bars where no rule could
# succeed.
#
# base and earn are NOT re-run: neither passes --max-weight-frac, the diagnostic
# is empty for them, and leaving them untouched keeps gate 2 standing on the
# files it was actually checked against.
set -u
cd "$(dirname "$0")"

CAL=data/earnings_calendar.csv
COMMON="--overnight --edge reversal --risk-scale vol \
  --open-spread-bps 1.464 --close-spread-bps 0.262 --carry-bps 0.20 \
  --min-names-frac 0.20"

run_arm () {
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

run_arm cap2  --max-weight-mult 3.0 --max-weight-frac 0.10
run_arm both2 --max-weight-mult 3.0 --max-weight-frac 0.10 --earnings-calendar "$CAL"

bash run_overnight_freeze3_wf.sh

echo "=== STEP 5b + RE-FREEZE COMPLETE ==="
