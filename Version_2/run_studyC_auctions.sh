#!/usr/bin/env bash
# STUDY C -- step 10. Auction pricing. SENSITIVITY ONLY.
#
# Declared in eval/PREREG_step10_14.md before it was run, and the declaration
# that matters most is this one: NO AUCTION COST IS CHOSEN HERE AND THE FROZEN
# REFERENCE DOES NOT MOVE ON THIS OUTPUT. The ratio divides by cost, so a low
# enough assumption about the auction clears the brief's bar by arithmetic with
# no change whatever to the signal. There is no auction imbalance data in this
# project, so there is nothing to declare a value from, and the parameter is
# swept rather than picked.
#
# THE DEFECT BEING PRICED. The book enters at the end of one session and exits
# at the start of the next. Both moments are AUCTIONS -- the 16:00 closing cross
# and the 09:30 opening cross -- and every order in a cross fills at one
# clearing price. It does not lift an offer and it does not give up an adverse
# tick on the snap. The model charges both.
#
# TWO SEPARABLE THINGS, RUN SEPARATELY.
#
#   C1  --exec-legs moc_moo. The entry leg is currently struck at open[L], the
#       open of the session's LAST 5-minute bar -- 15:55, five minutes before
#       the cross, and a quoted fill. moc_moo moves it to close[L], the 16:00
#       print, AND moves the decision back to 15:50, the MOC entry cutoff. A
#       decision taken at 15:55 cannot be submitted as an MOC; filling it at the
#       cross anyway would be a five-minute look-ahead dressed as an execution
#       improvement. So the book here sees LESS than the 15:55 book, not more.
#       This is NOT a cost assumption -- it changes which price the trade is
#       struck on and which information the decision may use. It can go either
#       way.
#
#   C2  --entry-auction-bps / --exit-auction-bps. The leg is priced as
#       auction + commission + impact; the half-spread and the adverse tick
#       snap are DROPPED, not reduced, and impact is retained -- a large MOC
#       order moves the closing print whether or not it paid a spread.
#       Each leg's auction cost is phi x that leg's MEASURED quoted half-spread
#       (entry 0.262, exit 1.464), phi in {1.00, 0.50, 0.25, 0.00}.
#       phi = 0 is the pure-impact floor: a lower bound, not an estimate.
#
# Run on BOTH references, as declared -- the amendment arm for the conclusion,
# freeze3 as the cross-check that says whether the amendment changed it.
set -u
cd "$(dirname "$0")"
mkdir -p logs/p3/on/auction

CAL=data/earnings_calendar.csv
COMMON="--overnight --edge reversal --risk-scale vol \
  --open-spread-bps 1.464 --close-spread-bps 0.262 --carry-bps 0.20 \
  --min-names-frac 0.20 --max-weight-mult 3.0 --max-weight-frac 0.10 \
  --earnings-calendar $CAL"
AMEND="--lam-select fixed --lam-fixed 1.0 --cap-realloc edge --cap-flat-if-infeasible"

# phi x the MEASURED half-spread on each leg. Written out rather than computed
# in the loop so the numbers actually charged are readable in the script.
#            phi     entry    exit
GRID="phi100:0.2620:1.4640
phi050:0.1310:0.7320
phi025:0.0655:0.3660
phi000:0.0000:0.0000"

run_cell () {
  ref="$1"; cell="$2"; shift 2
  for k in 1 2 3 4 5; do
    TF=$(python -c "print(f'{0.30 + 0.10*$k:.2f}')")
    echo "=== $ref/$cell fold $k TRAIN_FRAC=$TF ==="
    env TRADING_TRAIN_FRAC=$TF TRADING_VAL_FRAC=0.10 \
      TRADING_RAW_DIR=data/parquet_agg5 \
      TRADING_PROCESSED_DIR=data/processed_1min_ib_du \
      python -u eval/xsec_book.py $COMMON "$@" \
        --json logs/p3/on/auction/${ref}_${cell}_f${k}.json \
        > logs/p3/on/auction/${ref}_${cell}_f${k}.log 2>&1
    echo "    exit $?"
  done
}

run_ref () {
  ref="$1"; shift
  extra="$*"
  # C1 alone -- the frame correction, ordinary quoted costs.
  run_cell "$ref" mocmoo --exec-legs moc_moo $extra
  # C2 alone -- the auction pricing, existing 15:55 entry frame.
  echo "$GRID" | while IFS=: read -r cell ent ext; do
    run_cell "$ref" "$cell" --entry-auction-bps "$ent" --exit-auction-bps "$ext" $extra
  done
  # The joint corners. The book this strategy actually describes is MOC in,
  # MOO out, priced as crosses; phi 1.00 and 0.00 bracket what that costs.
  run_cell "$ref" both100 --exec-legs moc_moo \
      --entry-auction-bps 0.2620 --exit-auction-bps 1.4640 $extra
  run_cell "$ref" both000 --exec-legs moc_moo \
      --entry-auction-bps 0.0 --exit-auction-bps 0.0 $extra
}

run_ref amendA $AMEND
run_ref freeze3

echo "=== STUDY C COMPLETE ==="
