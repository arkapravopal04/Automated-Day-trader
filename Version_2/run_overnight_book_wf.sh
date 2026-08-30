#!/usr/bin/env bash
# The overnight hold at the BRIEF'S bar, across step 0's five folds.
#
# convention_table measures IC and edge-per-bet. That is a screen, not the
# test: P3's brief asks for ALPHA/TURN toward 0.5-0.6 with THE RATIO ABOVE 2,
# and only xsec_book computes those against the env's own cost model with
# sizing and name selection in place. (AGENTS.md's close plan records that
# gate as ratio > 1; that is a relaxation of the brief and is not used here.)
#
# TEST (2025-12-26 ->) IS NEVER READ.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/book

for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  for panel in ib base; do
    echo "=== fold $k  panel $panel  TRAIN_FRAC=$TF ==="
    TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_$panel \
      python -u eval/xsec_book.py --overnight \
        --json logs/p3/on/book/book_f${k}_${panel}.json \
        > logs/p3/on/book/book_f${k}_${panel}.log 2>&1
    echo "    exit $?"
  done
done
echo "=== OVERNIGHT BOOK WALK-FORWARD COMPLETE ==="
