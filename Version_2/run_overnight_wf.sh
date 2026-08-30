#!/usr/bin/env bash
# Step 1 of the P3 close plan, under step 0's protocol: price the overnight
# hold across the SAME five expanding-window folds run_walkforward.sh uses.
#
# The single split put validation in 2025-06 -> 2026-01, the window where the
# intraday reversal factor decayed to zero. That window produced IC +0.051 on
# the overnight row with t=3.5 -- which is either a real effect that survived a
# regime the intraday signal did not, or the same "you measured when you
# looked" artefact that voided every single-split number in this project. Five
# folds and a paired t decide it, and nothing else does.
#
# TEST (2025-12-26 ->) IS NEVER READ: no fold's val reaches it.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/wf

for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k  TRAIN_FRAC=$TF ==="
  # reversal is target-free and reads no features: panel-independent, one run.
  TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
  TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib \
    python -u eval/convention_table.py --overnight --signal reversal \
      --json logs/p3/on/wf/on_rev_f${k}.json \
      > logs/p3/on/wf/on_rev_f${k}.log 2>&1
  echo "    reversal exit $?"
  for panel in ib base; do
    TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_$panel \
      python -u eval/convention_table.py --overnight --signal ridge \
        --json logs/p3/on/wf/on_ridge_f${k}_${panel}.json \
        > logs/p3/on/wf/on_ridge_f${k}_${panel}.log 2>&1
    echo "    ridge $panel exit $?"
  done
done
echo "=== OVERNIGHT WALK-FORWARD COMPLETE ==="
