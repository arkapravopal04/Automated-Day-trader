#!/usr/bin/env bash
# The overnight IC on the delisting-inclusive universe, directly comparable to
# logs/p3/on/wf (mean IC +0.0353, t 2.68 on the 100 survivors).
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/ic_uni
for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k TRAIN_FRAC=$TF ==="
  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
    python -u eval/convention_table.py --overnight --signal reversal \
      --json logs/p3/on/ic_uni/on_rev_f${k}.json > logs/p3/on/ic_uni/on_rev_f${k}.log 2>&1
  echo "    exit $?"
done
echo "=== IC UNIVERSE WALK-FORWARD COMPLETE ==="
