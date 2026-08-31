#!/usr/bin/env bash
# AMENDMENT A -- steps 12, 13, 14. Seven arms, five folds, then the re-freeze.
#
# Rules fixed in eval/PREREG_step10_14.md BEFORE this script was first run:
# the lambda value, the reallocation mode, the breadth-floor test, the seven
# arms, and the declaration that the new reference is arm `A` whatever the
# columns turn out to say.
#
# The three clauses:
#   12  lambda FIXED at 1.0 on every fold. The train-Sharpe argmax was
#       separating candidates by 0.014 against a standard error of order 0.5,
#       and it let the treatment differ fold to fold.
#   13  --cap-realloc edge. The clip's released mass is water-filled back into
#       the SAME leg in proportion to remaining edge-over-cost, never across.
#       The `gross` rule funded clips out of the other leg -- FRC, 2023-03-10.
#   14  --cap-flat-if-infeasible. A bar whose two legs cannot each carry half
#       the gross under A = 0.10 admits no dollar-neutral book below the cap;
#       the book stands flat rather than reporting a position above its limit.
#
# GATE, and it is the first thing to read: `base` must reproduce
# logs/p3/on/freeze3/ EXACTLY. All three options are off in that arm, so the
# code change must be a no-op there. If it is not, nothing below it is
# admissible. eval/on_amendA_table.py checks it cell by cell.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/amendA logs/p3/on/freeze4

CAL=data/earnings_calendar.csv
COMMON="--overnight --edge reversal --risk-scale vol \
  --open-spread-bps 1.464 --close-spread-bps 0.262 --carry-bps 0.20 \
  --min-names-frac 0.20 --max-weight-mult 3.0 --max-weight-frac 0.10 \
  --earnings-calendar $CAL"

run_arm () {
  out="$1"; arm="$2"; shift 2
  for k in 1 2 3 4 5; do
    TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
    echo "=== arm $arm fold $k TRAIN_FRAC=$TF ==="
    env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
      TRADING_RAW_DIR=data/parquet_agg5 \
      TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
      python -u eval/xsec_book.py $COMMON "$@" \
        --json ${out}/${arm}_f${k}.json \
        > ${out}/${arm}_f${k}.log 2>&1
    echo "    exit $?"
  done
}

O=logs/p3/on/amendA
FIXED="--lam-select fixed --lam-fixed 1.0"
EDGE="--cap-realloc edge"
FLOOR="--cap-flat-if-infeasible"

run_arm $O base
run_arm $O lam      $FIXED
run_arm $O realloc  $EDGE
run_arm $O floor    $FLOOR
run_arm $O A        $FIXED $EDGE $FLOOR
run_arm $O A1se     --lam-select train-1se $EDGE $FLOOR
run_arm $O Anone    $FIXED --cap-realloc none $FLOOR

# THE RE-FREEZE. Re-run rather than re-quoted, for the reason freeze2 and
# freeze3 were: this project has twice had a number move between two runs of
# identical code, and a reference other work is graded against has to be one
# that reproduced at least once, on the cache as it stands today.
# eval/on_freeze_table.py then reports freeze4 against the amendA `A` arm cell
# by cell, and anything non-zero there means the cache moved underneath the
# result and the freeze does not stand.
run_arm logs/p3/on/freeze4 freeze4 $FIXED $EDGE $FLOOR

echo "=== AMENDMENT A + RE-FREEZE COMPLETE ==="
