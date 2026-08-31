#!/usr/bin/env bash
# STUDY C, APPENDIX A -- the EXIT-ONLY auction cell.
#
# Declared in Appendix A of eval/PREREG_step10_14.md before this was run.
#
# The two legs are NOT the same kind of fill and Study C's grid treated them as
# if they were. The 15:55 entry is a QUOTED fill -- a marketable order lifts an
# offer and gives up a tick on the snap, which is what the model charges, and no
# correction to it is warranted. The 09:30 exit is the OPENING CROSS, where an
# MOO order fills at one price and pays neither. It is also 5.6x the larger leg.
#
# And unlike the entry, correcting it costs NO INFORMATION: MOC entry closes at
# 15:50, which is why --exec-legs moc_moo had to move the decision back and why
# its IC collapsed. MOO entry stays open until 09:28, and the decision it
# implements was taken the previous afternoon. There is no cutoff to respect.
#
# phi = 0 is EVALUABLE here and was not in the body's grid: with the entry leg
# still paying a quoted spread the round trip stays strictly positive, so the
# pure-impact exit is a genuine reachable lower bound rather than a book that
# stands flat because the sizing divided by zero.
#
# NOTHING IS SELECTED. The deliverable is the breakeven phi.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/auction

CAL=data/earnings_calendar.csv
COMMON="--overnight --edge reversal --risk-scale vol \
  --open-spread-bps 1.464 --close-spread-bps 0.262 --carry-bps 0.20 \
  --min-names-frac 0.20 --max-weight-mult 3.0 --max-weight-frac 0.10 \
  --earnings-calendar $CAL"

# phi x 1.464, the MEASURED half-spread at the 09:30 cross. Entry untouched.
GRID="exit100:1.4640
exit050:0.7320
exit025:0.3660
exit000:0.0000"

echo "$GRID" | while IFS=: read -r cell ext; do
  for k in 1 2 3 4 5; do
    TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
    echo "=== freeze3/$cell fold $k TRAIN_FRAC=$TF  (exit auction $ext bps) ==="
    env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
      TRADING_RAW_DIR=data/parquet_agg5 \
      TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
      python -u eval/xsec_book.py $COMMON --exit-auction-bps "$ext" \
        --json logs/p3/on/auction/freeze3_${cell}_f${k}.json \
        > logs/p3/on/auction/freeze3_${cell}_f${k}.log 2>&1
    echo "    exit $?"
  done
done

echo "=== EXIT-ONLY CELL COMPLETE ==="
