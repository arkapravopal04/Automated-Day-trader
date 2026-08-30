#!/usr/bin/env bash
# The overnight book on the signal that actually carries the effect.
#
# The IC walk-forward put the target-free reversal at mean IC +0.0353 (t 2.68)
# against ridge's +0.0254 (27f) and +0.0280 (15f). Pricing the book only on
# ridge would test the weaker signal. The reversal edge is panel-independent --
# it reads prices, not features -- so one run per fold covers both panels.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/book_rev
for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k  TRAIN_FRAC=$TF ==="
  TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
  TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib \
    python -u eval/xsec_book.py --overnight --edge reversal \
      --json logs/p3/on/book_rev/book_rev_f${k}.json \
      > logs/p3/on/book_rev/book_rev_f${k}.log 2>&1
  echo "    exit $?"
done
echo "=== OVERNIGHT REVERSAL BOOK WALK-FORWARD COMPLETE ==="
