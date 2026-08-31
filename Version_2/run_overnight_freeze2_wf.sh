#!/usr/bin/env bash
# STEP 6 -- THE RE-FROZEN REFERENCE RUN.
#
# Identical configuration to the `both` arm of run_overnight_cap_wf.sh, byte for
# byte on the arguments. It is RE-RUN rather than re-quoted, for the same reason
# run_overnight_freeze_wf.sh was: this project has twice had a number move
# between two runs of identical code, and a reference other work is graded
# against has to be one that reproduced at least once, on the cache as it stands
# today. eval/on_freeze_table.py then reports freeze2 against the cap-run arm
# cell by cell, and anything non-zero there means the cache moved underneath the
# result and the freeze does not stand.
#
# The variant is NOT chosen here. eval/PREREG_step5_risk_controls.md declared
# `both` -- cap plus earnings exclusion -- to be the new reference before any of
# the four arms had been run, on the grounds that both are risk constraints a
# book runs unconditionally. It is not the arm that validated best and it was
# not selected by looking.
#
# Output goes to a NEW directory so logs/p3/on/freeze/ (the step-4 reference)
# and logs/p3/on/cap/ (the four arms) both survive for comparison.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/freeze2
OPEN_SPREAD=1.464; CLOSE_SPREAD=0.262; CARRY=0.20
CAL=data/earnings_calendar.csv
KAPPA=3.0
for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k TRAIN_FRAC=$TF ==="
  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
    python -u eval/xsec_book.py --overnight --edge reversal --risk-scale vol \
      --open-spread-bps $OPEN_SPREAD --close-spread-bps $CLOSE_SPREAD --carry-bps $CARRY \
      --min-names-frac 0.20 \
      --max-weight-mult $KAPPA --earnings-calendar $CAL \
      --json logs/p3/on/freeze2/freeze2_f${k}.json \
      > logs/p3/on/freeze2/freeze2_f${k}.log 2>&1
  echo "    exit $?"
done
echo "=== RE-FREEZE WALK-FORWARD COMPLETE ==="
