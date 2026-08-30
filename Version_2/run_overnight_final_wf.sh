#!/usr/bin/env bash
# The overnight book, all three flatteries corrected, WITH A BREADTH FLOOR.
# The train book must average >=20% of the universe (~25 names) before its
# Sharpe may compete for lambda -- otherwise selection lands in the corner where
# a 1.7-name book posts a ratio of 2.93 off 43 active periods.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/final
OPEN_SPREAD=1.464; CLOSE_SPREAD=0.262; CARRY=0.20
for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k TRAIN_FRAC=$TF ==="
  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
    python -u eval/xsec_book.py --overnight --edge reversal --risk-scale vol \
      --open-spread-bps $OPEN_SPREAD --close-spread-bps $CLOSE_SPREAD --carry-bps $CARRY \
      --min-names-frac 0.20 \
      --json logs/p3/on/final/final_f${k}.json > logs/p3/on/final/final_f${k}.log 2>&1
  echo "    exit $?"
done
echo "=== FINAL WALK-FORWARD COMPLETE ==="
