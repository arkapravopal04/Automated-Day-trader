#!/usr/bin/env bash
# Per-name / per-month / per-cell decomposition of every fold of the final
# overnight walk-forward. Lambda is READ from each fold's own json, so this
# decomposes the result that exists rather than searching for a new one.
#
# The fold-2 arm is also run on the survivor-only panel (processed_1min_ib) to
# separate "the loss" from "the universe the loss was measured on".
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/decomp

COMMON="--open-spread-bps 1.464 --close-spread-bps 0.262 --carry-bps 0.20 --risk-scale vol"

for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k TRAIN_FRAC=$TF (124-name delisting-inclusive) ==="
  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
    python -u eval/on_fold_decomp.py $COMMON \
      --from-json logs/p3/on/final/final_f${k}.json \
      --json logs/p3/on/decomp/f${k}_du.json \
      > logs/p3/on/decomp/f${k}_du.log 2>&1
  echo "    exit $?"
done

echo "=== fold 2, survivor-only 100-name panel ==="
env TRADING_TRAIN_FRAC=0.50 TRADING_VAL_FRAC=0.10 \
  TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib \
  python -u eval/on_fold_decomp.py $COMMON \
    --from-json logs/p3/on/uni/f2_ib.json \
    --json logs/p3/on/decomp/f2_ib.json \
    > logs/p3/on/decomp/f2_ib.log 2>&1
echo "    exit $?"
echo "=== DECOMPOSITION COMPLETE ==="
