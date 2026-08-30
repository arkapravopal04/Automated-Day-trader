#!/usr/bin/env bash
# The overnight book on the DELISTING-INCLUSIVE universe (124 names).
#
# Two arms, so the universe's effect is attributable rather than tangled with
# the cost and sizing fixes:
#   uni       universe change ONLY -- directly comparable to logs/p3/on/book_rev
#   unifix    universe + measured per-leg spreads + carry + equal-risk sizing,
#             i.e. all three flatteries corrected at once. THIS IS THE ANSWER.
#
# The 24 added names are large caps that DIED in-sample (scan_delisted.py):
# SIVB, FRC, TWTR, ATVI, PXD, XLNX and 18 others. FRC alone gaps -65% over the
# SVB weekend, 2023-03-10 close to 2023-03-13 open -- precisely the kind of
# overnight move a cross-sectional reversal book buys into and which a
# survivor-only universe contains exactly zero of.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/uni

OPEN_SPREAD=1.464; CLOSE_SPREAD=0.262; CARRY=0.20

for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k  TRAIN_FRAC=$TF ==="
  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
    python -u eval/xsec_book.py --overnight --edge reversal \
      --json logs/p3/on/uni/uni_f${k}.json > logs/p3/on/uni/uni_f${k}.log 2>&1
  echo "    uni exit $?"
  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
    python -u eval/xsec_book.py --overnight --edge reversal --risk-scale vol \
      --open-spread-bps $OPEN_SPREAD --close-spread-bps $CLOSE_SPREAD --carry-bps $CARRY \
      --json logs/p3/on/uni/unifix_f${k}.json > logs/p3/on/uni/unifix_f${k}.log 2>&1
  echo "    unifix exit $?"
done
echo "=== UNIVERSE WALK-FORWARD COMPLETE ==="
