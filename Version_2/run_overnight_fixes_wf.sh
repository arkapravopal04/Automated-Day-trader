#!/usr/bin/env bash
# The three flatteries, ablated one at a time, on the SAME five folds.
#
# Baseline (already run, logs/p3/on/book_rev) = ratio 2.37, and it is flattered
# three ways. Changing all three at once would tell us the answer moved without
# telling us what moved it, so each gets its own arm:
#
#   cost  measured per-leg spreads. Corwin-Schultz on the 1-minute bars puts
#         the effective spread at the 09:30 print at 2.93 bps against 0.068
#         midday -- 43x -- so the exit leg had been charged about a fifth of
#         its real spread. Half-spreads: 1.464 exit, 0.262 entry. Plus 0.20
#         bps of gross per night for borrow and financing, which is an
#         ASSUMPTION (no borrow data exists in this project), not a measurement.
#   risk  equal-risk instead of equal-dollar sizing, on causal trailing
#         overnight vol. `book_weights` has no risk term, and the reversal edge
#         is proportional to the trailing move, so the book systematically puts
#         its biggest positions on its most volatile names -- the mechanism
#         behind Sharpe ex-top5 collapsing in 3/5 folds.
#   both  the two together.
#
# The universe fix (delisted names) is a separate pipeline and is NOT in here.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/fixes

OPEN_SPREAD=1.464
CLOSE_SPREAD=0.262
CARRY=0.20

for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k  TRAIN_FRAC=$TF ==="
  base="TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib"

  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib \
    python -u eval/xsec_book.py --overnight --edge reversal \
      --open-spread-bps $OPEN_SPREAD --close-spread-bps $CLOSE_SPREAD --carry-bps $CARRY \
      --json logs/p3/on/fixes/cost_f${k}.json > logs/p3/on/fixes/cost_f${k}.log 2>&1
  echo "    cost exit $?"

  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib \
    python -u eval/xsec_book.py --overnight --edge reversal --risk-scale vol \
      --json logs/p3/on/fixes/risk_f${k}.json > logs/p3/on/fixes/risk_f${k}.log 2>&1
  echo "    risk exit $?"

  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib \
    python -u eval/xsec_book.py --overnight --edge reversal --risk-scale vol \
      --open-spread-bps $OPEN_SPREAD --close-spread-bps $CLOSE_SPREAD --carry-bps $CARRY \
      --json logs/p3/on/fixes/both_f${k}.json > logs/p3/on/fixes/both_f${k}.log 2>&1
  echo "    both exit $?"
done
echo "=== FIXES WALK-FORWARD COMPLETE ==="
