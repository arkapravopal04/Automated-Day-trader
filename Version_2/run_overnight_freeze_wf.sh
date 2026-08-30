#!/usr/bin/env bash
# THE FROZEN REFERENCE RUN.
#
# Identical configuration to run_overnight_final_wf.sh -- byte for byte on the
# arguments. It is re-run rather than re-quoted for two reasons:
#
#   1. Steps 1-3 established that fold 2 requires NO change. Its loss is not a
#      data artefact (it reconciles exactly, and the driver is a real -24.8%
#      PANW earnings gap on 82M shares), not a universe artefact (the
#      survivor-only panel gives -2.39 against -2.42), and not a regime that
#      any causal marker separates ex ante. So the frozen baseline IS this
#      configuration, unmodified.
#   2. This project has twice had a number move between two runs of identical
#      code. A reference other work is judged against has to be one that was
#      reproduced at least once, on the cache as it stands today.
#
# Output goes to a NEW directory so logs/p3/on/final/ survives for comparison.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/freeze
OPEN_SPREAD=1.464; CLOSE_SPREAD=0.262; CARRY=0.20
for k in 1 2 3 4 5; do
  TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
  echo "=== fold $k TRAIN_FRAC=$TF ==="
  env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
    TRADING_RAW_DIR=data/parquet_agg5 TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
    python -u eval/xsec_book.py --overnight --edge reversal --risk-scale vol \
      --open-spread-bps $OPEN_SPREAD --close-spread-bps $CLOSE_SPREAD --carry-bps $CARRY \
      --min-names-frac 0.20 \
      --json logs/p3/on/freeze/freeze_f${k}.json > logs/p3/on/freeze/freeze_f${k}.log 2>&1
  echo "    exit $?"
done
echo "=== FREEZE WALK-FORWARD COMPLETE ==="
