#!/usr/bin/env bash
# STUDY B -- step 11. The holding-period sweep. MEASUREMENT ONLY.
#
# Declared in eval/PREREG_step10_14.md before it was run. NO HOLD IS SELECTED
# by this script and the frozen reference does not move on its output: picking
# the best cell of a five-point grid on five folds is exactly the thing the
# walk-forward protocol exists to prevent, and the grid is reported whole.
#
# What is being traded off. At hold 1 the overnight book does a full round trip
# every session -- turnover ~2 per session, which is why COST/TURN ~1.96 bps
# eats about half of gross. Hold h cuts turnover per session to 2/h BY
# CONSTRUCTION. What it buys that with is h-1 day sessions of exposure the
# reversal edge does not forecast, and whether that trade is worth taking is
# the question, not a foregone conclusion.
#
# Three things the code does that would otherwise be silent flatteries, all
# declared in the pre-registration:
#   * the edge is refitted on the h-night target, not the 1-night one;
#   * trailing_overnight_vol is lagged h sessions, not 1 -- shifting by one at
#     hold h would size a session on an outcome that has not finished;
#   * carry is charged h times per period, i.e. unchanged per session.
#
# Run on BOTH references. The amendment arm is the one the conclusion is drawn
# from; freeze3 is the pre-registered cross-check that says whether the
# amendment changed the conclusion, and it was declared before either ran.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/holds

CAL=data/earnings_calendar.csv
COMMON="--overnight --edge reversal --risk-scale vol \
  --open-spread-bps 1.464 --close-spread-bps 0.262 --carry-bps 0.20 \
  --min-names-frac 0.20 --max-weight-mult 3.0 --max-weight-frac 0.10 \
  --earnings-calendar $CAL"
HOLDS="--hold 1 --hold 2 --hold 3 --hold 4 --hold 5"
AMEND="--lam-select fixed --lam-fixed 1.0 --cap-realloc edge --cap-flat-if-infeasible"

run_ref () {
  ref="$1"; shift
  for k in 1 2 3 4 5; do
    TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
    echo "=== holds on $ref fold $k TRAIN_FRAC=$TF ==="
    env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
      TRADING_RAW_DIR=data/parquet_agg5 \
      TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
      python -u eval/xsec_book.py $COMMON $HOLDS "$@" \
        --json logs/p3/on/holds/${ref}_f${k}.json \
        > logs/p3/on/holds/${ref}_f${k}.log 2>&1
    echo "    exit $?"
  done
}

run_ref amendA $AMEND
run_ref freeze3

echo "=== STUDY B COMPLETE ==="
