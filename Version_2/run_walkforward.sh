#!/usr/bin/env bash
# Expanding-window walk-forward for P3.
#
# The single 0.80/0.10/0.10 split put ALL of validation inside 2025Q3-2026Q1,
# the window in which the short-term reversal factor decayed to zero (measured:
# pooled bounce IC ran +0.005..+0.056 across the 19 quarters from 2020Q4 to
# 2025Q2, then -0.010, +0.002, +0.004). Train said the intrabar features lift
# ALPHA/TURN in all 15 cells; validation said every cell is negative. Neither
# is interpretable alone, because "did richer inputs help" is confounded with
# "when did you happen to measure".
#
# Fold k trains on everything before its window and scores the next ~7 months.
# No new estimator and no new split logic: paths.py already reads
# TRADING_TRAIN_FRAC / TRADING_VAL_FRAC, so a fold is the SAME audited
# xsec_book invoked with different fractions.
#
#   fold 1  train -> 2023-01-10   val 2023-01-10 -> 2023-08-15
#   fold 2  train -> 2023-08-15   val 2023-08-15 -> 2024-03-18
#   fold 3  train -> 2024-03-18   val 2024-03-18 -> 2024-10-18
#   fold 4  train -> 2024-10-18   val 2024-10-18 -> 2025-05-27
#   fold 5  train -> 2025-05-27   val 2025-05-27 -> 2025-12-26
#
# TEST (2025-12-26 -> 2026-08-28) IS NEVER READ. xsec_book and
# convention_table only ever touch train and val, and no fold's val reaches
# into that region.
#
# Configurations 2, 4 and 5 only. Row 6 (the lambda=1 cost hurdle) stands flat
# almost always -- on the full split it averaged 0.4 names and produced an
# ALPHA/TURN of 3.3 from a handful of bets. On a fold a third the length that
# is pure noise, and it costs a third of the runtime to produce.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/wf

for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  for panel in ib base; do
    echo "=== fold $k  panel $panel  TRAIN_FRAC=$TF ==="
    TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_$panel \
      python -u eval/ratio_table.py --configs 2,4,5 \
        --json logs/p3/wf/ratio_f${k}_${panel}.json \
        > logs/p3/wf/ratio_f${k}_${panel}.log 2>&1
    echo "    ratio exit $?"
  done
  # Panel-independent: the reversal signal is target-free and reads no
  # features, so one run per fold covers both panels.
  TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
  TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib \
    python -u eval/convention_table.py --signal reversal \
      --json logs/p3/wf/conv_f${k}.json > logs/p3/wf/conv_f${k}.log 2>&1
  echo "    conv exit $?"
done
echo "=== WALK-FORWARD COMPLETE ==="
