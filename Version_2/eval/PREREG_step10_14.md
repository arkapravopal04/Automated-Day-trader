# Pre-registration -- steps 10-14

Written **before** any of the runs below was executed, and before any validation
split was read under any of the rules it declares. Committed in this state so
the selection rules cannot be reverse-engineered from the results they produced.
Nothing here may be edited after the first run in `run_step10_amendA.sh`
completes; corrections go in a dated appendix, as they did in
`eval/PREREG_step5_risk_controls.md`.

Baseline being modified: the step-6 re-frozen reference under Appendix A,
`logs/p3/on/freeze3/freeze3_f{k}.json` --

```
--overnight --edge reversal --risk-scale vol --open-spread-bps 1.464
--close-spread-bps 0.262 --carry-bps 0.20 --min-names-frac 0.20
--max-weight-mult 3.0 --max-weight-frac 0.10
--earnings-calendar data/earnings_calendar.csv
124-name panel, lambda selected on TRAIN, TEST NEVER READ
```

whose validation columns are, per fold and as the mean of five:

| | f1 | f2 | f3 | f4 | f5 | mean |
|---|---|---|---|---|---|---|
| lambda (train-chosen) | 1.00 | 0.50 | 0.25 | 0.50 | 0.50 | 0.55 |
| ALPHA/TURN bps | -0.81 | 0.06 | 0.80 | 4.52 | 3.04 | **1.52** |
| COST/TURN bps | 2.37 | 2.02 | 1.82 | 1.81 | 1.78 | **1.96** |
| ratio | -0.34 | 0.03 | 0.44 | 2.49 | 1.71 | **0.87** |
| net Sharpe | -0.95 | -2.68 | -1.35 | 1.86 | 1.19 | **-0.39** |

Pooled over the 743 disjoint validation sessions the ratio is **0.772**
(`logs/p3/on/indep/freeze3.json`). The step-4 reference, before either risk
control, pooled at **1.185**. Both numbers appear below and they are not
interchangeable; the arithmetic in Study C is stated against whichever
reference the arm was run on, and never against the more flattering one.

---

## The ordering, and why it is not the order of leverage

Study C has the most leverage and it is run **last of the three groups**. Steps
12-14 close defects in the rule; steps 10 and 11 measure things. A measurement
taken on a book with a known defect in it measures the defect as well, so the
amendment goes first and the two studies are run on the corrected reference.

1. **Amendment A** (steps 12, 13, 14) -- one amendment, three clauses, seven
   arms, then a re-freeze. Produces `logs/p3/on/freeze4/`.
2. **Study B** (step 11) -- the holding-period sweep, on the new reference.
   Reports; changes nothing.
3. **Study C** (step 10) -- the auction pricing, on the new reference. Reports
   a sensitivity curve; **changes nothing, and cannot become the headline.**

Both studies are also run against `freeze3` so the reader can see whether the
amendment changed their conclusion. That cross-check is declared here, before
either was run, precisely so that reporting only the flattering one is not
available afterwards.

---

# Amendment A -- steps 12, 13, 14

Three clauses, bundled into one amendment because they touch the same object --
how a bar's book is decided -- and splitting them into three would triple the
number of pre-registrations without adding a single independent decision.

## Clause 1 (step 12) -- lambda is fixed a priori

### The defect

Lambda is currently chosen per fold as the argmax of train Sharpe among the
candidates that clear the activity and breadth floors. On fold 3 of the
re-frozen reference that argmax discriminated between train Sharpes of
**-0.102 and -0.116** -- a gap of 0.014, against a Lo standard error on a train
Sharpe of order 0.5 -- and half a point of validation Sharpe moved on the
difference. A criterion that separates its two leading candidates by three
percent of its own standard error has no discriminating power; what it is
actually selecting on is noise.

The second defect is the one that matters more. Letting lambda fit per fold
means **the treatment differs across folds**: the re-frozen reference runs
lambda 1.00 on fold 1, 0.25 on fold 3 and 0.50 on the rest, so the five-fold
mean is the average of three different strategies and its standard error is not
the standard error of any of them. This is exactly the problem that was removed
from KAPPA in step 5 by fixing it a priori, and the same argument applies
unchanged.

### The rule

**`lambda = 1.0`, fixed, on every fold, in train and validation alike.**

`1.0` is declared from the hurdle's own definition and from no measurement in
this project. The hurdle admits a name when `|edge| > lambda * round_trip`, so
`lambda = 1` is the value at which a name is held **only if its expected edge
covers the round trip it will pay.** Below 1 the book is deliberately taking
trades whose expected edge is less than their cost and relying on the
cross-section to make that back; above 1 it is demanding a margin. One is the
economically neutral hurdle, it is the same threshold the project's own `ratio`
bar is stated at -- above 1 the book pays for itself -- and it is constant
across all five folds with no train statistic behind it that could drift.

It is **not** the modal train-chosen value. The train-chosen values are
{1.00, 0.50, 0.25, 0.50, 0.50}; the modal one is 0.50 and it is not being used,
because "the value the defective criterion picked most often" is the defective
criterion wearing a hat.

**The floors are still evaluated and a failure is reported loudly, and it does
not change the choice.** If lambda 1.0 produces a book below the breadth floor
on some fold, that is a result about that fold. Substituting a different lambda
there is how the fold-varying treatment gets back in.

### The alternative, also declared, also reported

`--lam-select train-1se`: take the train optimum, take its Lo standard error,
and choose the **largest** lambda whose train Sharpe is within one standard
error of it. Larger lambda is the conservative direction -- it trades less and
demands more edge per unit of cost -- so this is the 1-SE rule in its usual
sense. It is reported as an arm. It is **not** the reference: it still lets the
treatment vary across folds, and it is offered only as the answer to "what if
adaptivity is wanted anyway".

## Clause 2 (step 13) -- the cap's reallocation is made risk-aware

### The defect

`apply_weight_cap` enforces the per-name cap by clipping, re-demeaning over the
selected names, re-normalising back to gross 1, and iterating. The clip is
correct. The **re-normalisation is an unmanaged transfer**: the mass taken off
the clipped name is put back into the book somewhere, the rule never says
where, and it is free to cross from one leg to the other.

The tell is FRC on 2023-03-10 -- the SVB weekend, in fold 1. Clipping one name
pushed weight into a long on a regional bank that was days from failing. A
position limit that funds itself by growing unrelated positions is not a
position limit.

There is a second, sharper way to see it. Take a bar whose demeaned book has two
large longs, four longs at exactly zero weight, and six shorts. Under the current
rule the clip hands **0.075 of gross each to the four names the sizing had
assigned nothing to** -- names the hurdle admitted and the sizing then valued at
zero. Whatever the cap is for, it is not for that.

### The rule

**`--cap-realloc edge`.** Mass released by a clip is water-filled back into
**the same leg**, in proportion to the names' remaining pre-cap magnitudes --
which are exactly `(|edge| - lambda*cost) / cost / vol`, the quantity the sizing
already used. Names already at the cap take none of it; names the sizing valued
at zero take none of it. Dollar-neutrality is then restored by scaling the
**larger** leg down to the smaller, never the smaller up. Gross ends at
`2 * min(leg)`, at most 1, and below 1 whenever the cap bound harder than the
leg could absorb.

Two consequences are stated here so they cannot be reported as surprises:

- **The book will sometimes run under-invested.** That is the point. A binding
  limit that always finds somewhere to put the risk has not removed any.
  `gross_deployed` is recorded per bar and reported per fold.
- **Reported concentration is against realised gross.** On an under-invested bar
  a position at the cap is a larger share of the smaller book, so `max_share`
  can read above `A` where it previously could not. That is the honest
  measurement and it is not a regression.

`--cap-realloc none` -- clip and do not redeploy at all -- is run as a
decomposition arm. It is the strictest reading of "don't reallocate", and the
gap between `none` and `edge` is the value of redeploying within a leg.

## Clause 3 (step 14) -- a per-bar breadth floor

### The defect

Under `A = 0.10` a dollar-neutral book needs each leg to carry half the gross,
so it needs `0.5 / A = 5` names a side and `1 / A = 10` in all. Bars narrower
than that admit **no** book below the cap at any weighting whatsoever. The
enforcement currently returns the tightest book the bar admits and reports it as
`cap_infeasible`, which is why folds 1, 3 and 5 still read a max name share of
0.500 under a cap of 0.10.

Appendix A of the step-5 pre-registration declared that `cap_infeasible > 0` was
a result about breadth and was **not permitted to motivate a change to
`min_names`, to `A`, or to the breadth floors**, in step 5 or step 6. That was
the correct ruling at the time: the count had not been measured, and moving a
floor to make a fresh diagnostic go away is fitting. It has now been measured
across five folds, and what it reports is a **live defect** -- the reference book
holds positions above its own stated limit on a known, countable fraction of its
bars. That is a different thing from an open question, and it gets its own
pre-registration line rather than an exemption from the old one.

### The rule

**`--cap-flat-if-infeasible`.** On any bar where `cap * n_long < 0.5` or
`cap * n_short < 0.5`, the book stands **flat**. No book below the cap exists on
that bar, so every alternative reports a position above the limit.

This is a **strategy change**, not a diagnostic: it removes periods from the
traded sample. It is off by default in the code and it is declared here.

The test is on the **legs**, not on `n_selected`. Breadth alone
(`cap * n >= 1`) is necessary and not sufficient -- twenty longs and three
shorts passes it at `A = 0.10` and still admits no book, because three names
capped at 0.10 cannot carry a leg's half of gross. The existing `cap_infeasible`
counter keeps its Appendix-A breadth-only meaning so every count already in the
record reproduces; the leg test is a separate, sharper function and the flat
rule is stated against it.

## Arms, all five folds, all pre-declared

| arm | lambda | cap realloc | breadth floor |
|---|---|---|---|
| `base` | train-max | gross | -- |
| `lam` | **fixed 1.0** | gross | -- |
| `realloc` | train-max | **edge** | -- |
| `floor` | train-max | gross | **on** |
| `A` | **fixed 1.0** | **edge** | **on** |
| `A1se` | 1-SE | **edge** | **on** |
| `Anone` | **fixed 1.0** | **none** | **on** |

`base` must reproduce `logs/p3/on/freeze3/` **exactly**. It is the regression
gate on the code change: if the new options being off is not a no-op, nothing
below it is admissible. *(This gate has been run and passes at 0.00e+00 on every
sweep metric of every lambda row, train and validation, on fold 1 --
`logs/p3/on/regress/f1.json`. The full five-fold gate runs with the arms.)*

The three single-clause arms decompose the change; they are **not** candidates
to be selected between.

## The re-freeze (Amendment A's reference)

**The new frozen reference is arm `A`** -- lambda fixed at 1.0, `edge`
reallocation, breadth floor on -- declared here before any of the seven arms has
been run. All three clauses close defects in rules the book runs
unconditionally, so the reference is the book with all three applied. `A1se` and
`Anone` are reported to decompose the two choices that had a defensible
alternative, not to be selected between.

Output: `logs/p3/on/freeze4/`. `logs/p3/on/freeze3/` is retained, not deleted --
it is the record of what the defects cost.

## What is expected, so that it cannot be reported as a surprise

**Every clause is expected to make the reported book worse, and two of them are
expected to make it much worse.**

- **Lambda 1.0** is above the train-chosen value on four of five folds, so on
  those folds the book gets narrower and trades less. Fold 1 already ran at 1.00
  under the re-frozen reference and posted Sharpe -0.95 / ratio -0.34; the other
  four folds moving toward that configuration is the expected direction. The
  breadth floor is 24.8 names on fold 1 and the lambda-1.00 train book sat at
  25.2, so **fold 1 may stop clearing its own floor.** Under a fixed lambda that
  is reported, not repaired.
- **`edge` reallocation** is weakly tighter than `gross` on every bar: it removes
  an inflation channel and adds none. Gross can only fall.
- **The breadth floor** removes periods. It removes the narrowest ones, which are
  the ones the concentration diagnostics have been worst on, so `max_share`
  should improve and the traded sample should shrink.

The ratio is expected to fall for the reason step 5 already gave and which has
not changed: `(edge - lambda*cost)+ / cost` divides by cost, so the
concentration and the good ratio are the same mechanism, and every one of these
clauses removes concentration.

**If arm `A` improves on `freeze3`, that is a surprise and is to be reported as
one.** It would mean the defects were carrying the loss rather than the result,
and that claim would need the decomposition arms to support it.

---

# Study B -- step 11, the holding period

## What is being asked

Cost has two factors and the auction correction only attacks one. An overnight
book at hold 1 does a full round trip every session: turnover is ~2 per session,
which is why COST/TURN 1.96 bps consumes roughly half of gross. Holding `h`
nights cuts turnover per session to `2/h` **proportionally and by
construction**, so if any part of the reversal survives past one night the ratio
improves mechanically. The intraday book already showed the shape of this trade
-- hold 6 roughly doubled ALPHA/TURN over hold 1.

## What is being paid for it, stated up front

A multi-night hold carries the book through `h - 1` **day sessions**. The
reversal edge is fitted on, and forecasts, the overnight gap; it says nothing
about the day session. So hold `h` buys its lower turnover with intraday
exposure the signal does not forecast, and there is no reason from the signal to
expect that exposure to have positive expectation. **This may well not carry over
from the intraday result, and that is the question.**

## The grid, declared

`--hold 1 2 3 4 5`, on the new reference (arm `A`) and on `freeze3`. Nothing
else changes.

## Three implementation points, declared because each could have been a silent flattery

1. **The edge is refitted on the traded horizon.** At hold `h` the target is the
   `h`-night return, not the one-night return, for the reason the overnight
   masking already gave: a different horizon has a different conditional mean,
   and fitting on one while trading the other prices the wrong thing.
2. **The causal vol window is lagged by `h`, not by 1.**
   `trailing_overnight_vol` shifted by one session, which is correct at hold 1
   and a **look-ahead at every longer hold** -- session `s-1`'s `h`-night outcome
   does not finish until session `s+h-2`. The lag is now the hold. Getting this
   wrong would have made the longer holds look better for exactly the wrong
   reason.
3. **Carry scales with the hold.** The position is financed for `h` nights and
   the `h-1` days between them, so the charge is `h * carry_bps` per period, i.e.
   unchanged per session. Weekends remain one night's borrow -- the same
   idealisation the hold-1 book already makes.

## What is reported, and what is not decided

The full hold curve, per fold and pooled: ALPHA/TURN, COST/TURN, ratio, net
Sharpe, turnover per session, and book breadth. **No hold is selected.** The
reference stays at hold 1 unless a separate, later pre-registration adopts
another, and picking the best cell of a five-point grid on five folds is not a
thing this study is permitted to do.

---

# Study C -- step 10, the auction pricing

## The defect in the cost model

The book enters at the end of one session and exits at the start of the next.
Both of those moments are **auctions**: the 16:00 closing cross and the 09:30
opening cross. An MOC or MOO order does not lift an offer or hit a bid -- every
order in the cross is filled at one clearing price. So the model is charging a
quoted half-spread, plus an adverse tick snap, on two fills that pay neither.

The exit leg is where this is worth the most: it is charged 1.464 bps of measured
half-spread against the entry leg's 0.262, a factor of 5.6.

## Two separable things, and they are run separately

**C1 -- the execution frame.** The entry leg is currently struck at `open[L]`,
the open of the session's **last 5-minute bar**, i.e. **15:55** -- not the close.
That is a quoted fill five minutes before the cross, and it also puts the
15:55-to-16:00 move inside the held return. `--exec-legs moc_moo` moves the entry
to `close[L]`, the 16:00 print, which on consolidated data is the closing cross.

**And it moves the decision with it, because the MOC cutoff is 15:50.** NYSE and
Nasdaq stop accepting market-on-close orders at 15:50 ET. A decision taken on
data through 15:55 -- which is where the existing overnight book decides --
cannot be submitted as an MOC at all. Filling it at the 16:00 cross anyway would
be a five-minute look-ahead dressed up as an execution improvement, and it is
exactly the class of subsidy this project has spent three sessions removing.

So under `moc_moo` the decision bar moves back to `L-2`, whose close is **15:50**
-- the last moment an MOC can still be entered -- and the fill is the 16:00
cross. The held return is then exactly close-to-open, and **the book decides on
five minutes LESS information than the 15:55 book does.** The cutoff costs
information; it does not grant any. It is carried through the schedule, the
forward return and the causal volatility window alike.

This is **not** a cost assumption. It changes which price the trade is struck on
and which information the decision may use, and both changes are real. It can go
either way and it is walk-forwarded on its own.

**C2 -- the auction cost.** With `--entry-auction-bps` / `--exit-auction-bps` the
leg is priced as `auction_bps + commission + impact`: the half-spread and the
adverse tick snap are **dropped, not reduced**, and impact is retained unchanged,
because a large MOC order moves the closing print whether or not it paid a spread
to get there.

## No auction cost is declared as the value, and this is deliberate

**There is no auction imbalance data in this project, so there is no number to
declare.** Any value put here would be an assumption, and the ratio *divides by
cost*, so a low enough assumption clears the brief's bar by arithmetic with no
change whatever to the signal. That is not a result and it will not be reported
as one.

So the parameter is **swept, never chosen**, on a declared grid: each leg's
auction cost is `phi` times that leg's measured quoted half-spread, with

```
phi in {1.00, 0.50, 0.25, 0.00}

entry (16:00 cross): phi * 0.262 bps
exit  (09:30 cross): phi * 1.464 bps
```

`phi = 1.00` is **not** the same as the uncorrected book even so: it still drops
the adverse half-tick snap, which an auction cross does not pay either. The
uncorrected book is the `freeze3`/`freeze4` reference itself and it is quoted
beside every cell. `phi = 0.00` is the pure-impact floor -- the cross is free
apart from the size being crossed -- and is a lower bound, not an estimate. The
two intermediate points exist to make the curve readable, not because either is
believed.

## What is reported

1. The **whole curve**: ratio and net Sharpe against `phi`, per fold and pooled.
2. The **breakeven** `phi` at which pooled ratio reaches 2.0 and at which net
   Sharpe reaches 1.5 -- i.e. how cheap the auction would have to be for the book
   to clear its bars on this correction alone.
3. The **second-order effect on breadth.** Cheaper cost lets more names clear the
   hurdle, so the book widens; that is a change in what is held, not only in what
   it is charged, and it is reported separately from the arithmetic.

## The rule on how this may be quoted, fixed here

**A bar cleared at any `phi < 1` is a statement about the assumption, not about
the strategy**, and must be quoted with the `phi` that produced it and with the
uncorrected number beside it. The frozen reference does not move on Study C. It
moves when an auction cost is **measured**, and measuring one needs auction
imbalance and cross data this project does not have.

`--exec-legs moc_moo` at `phi = 1` is the one cell in Study C that is a candidate
for adoption on its own, because it is a correction to the model of reality
rather than an assumption about a price. Whether it is adopted is a later
decision and is not pre-registered here.

---

## Test

Untouched, under every arm of every group, as in steps 5 and 6.

---

# Appendix A -- 2026-08-31 -- the EXIT-ONLY auction cell. An addition to Study C.

Written **after** the Study C grid was run and **before** the cell it declares
has been run. It adds a cell; it does not edit the body, revise a result, or
move the reference. The distinction that makes this admissible is that the cell
is motivated by a **structural asymmetry between the two legs**, not by any
number the grid produced.

## The defect in Study C's grid

Study C swept `phi` on **both legs together**. That treats the two legs as the
same kind of fill. They are not:

| leg | when | what it is | quoted spread charged today |
|---|---|---|---|
| entry | 15:55 | a **quoted** fill -- a marketable order into the book | 0.262 bps |
| exit | 09:30 | the **opening cross** -- an MOO order fills at one price | 1.464 bps |

The entry leg is priced correctly as it stands. A market order at 15:55 lifts an
offer and gives up a tick on the snap, which is exactly what the model charges,
and **no correction to it is warranted.** Study C's `phi100` nonetheless dropped
the entry leg's tick snap along with the exit's, so every cell in the grid
mixes a justified correction with an unjustified one.

The exit leg is the one that is genuinely mispriced, and it is the larger of the
two by 5.6x. It is also the leg where the correction is **free of any
information cost**: `moc_moo` had to move the decision back to 15:50 because the
MOC cutoff is 15:50, but an MOO order can be entered until **09:28**, and the
decision it implements was taken the previous afternoon. So there is no cutoff
to respect and no signal to give up. `moc_moo`'s IC collapse cannot happen here.

## The cell

Entry leg **unchanged** -- `--close-spread-bps 0.262`, quoted, with the tick snap
and the `max(spread, half-tick)` floor intact. Exit leg priced as a cross:

```
--exit-auction-bps  phi * 1.464      phi in {1.00, 0.50, 0.25, 0.00}
```

Everything else is the `freeze3` reference configuration, unchanged, including
train-max lambda selection -- so the cell is directly comparable to the rest of
Study C rather than to a different selection rule.

**`phi = 0` is evaluable here, and it was not in the body's grid.** With both
legs zeroed the round-trip cost row was identically zero and `book_weights`
rejected every name on its `rt_cost > 0` guard, so the book stood 100% flat. With
the entry leg still paying a quoted spread the round trip stays strictly
positive, and the pure-impact exit is a genuine, reachable lower bound.

## What is expected, so that it cannot be reported as a surprise

- **COST/TURN falls and the ratio rises roughly as 1/cost.** Study C established
  that ALPHA/TURN is near-flat across the sweep (1.52 / 1.96 / 2.03 / 1.94), so
  cost acts mostly as a denominator on this book. The ratio bar of 2.0 is
  **expected to be cleared somewhere on this curve**, and clearing it is
  therefore not evidence about the strategy.
- **The Sharpe bar of 1.5 is expected NOT to be cleared at any phi.** The
  both-legs curve peaked at +0.84 and was non-monotone. Nothing here changes the
  return series' shape, only its level.
- **Lambda is still selected per cell by train-max**, which Amendment A showed to
  be a defective criterion. ALPHA/TURN carries that noise; the breakeven should
  be read with it.

## What this cell may and may not be used for

It **may** be quoted as the honest cost of the exit leg, because the correction
is structural and costs no information.

It **may not** move the frozen reference, on the same grounds as the body: the
auction cost is still an assumption, there is still no imbalance data in this
project, and **a bar cleared at any phi < 1 remains a statement about the
assumption and not about the strategy.** The deliverable is the breakeven phi.

It does **not** rehabilitate `both100`. That cell reads ALPHA/TURN -0.020 -- there
is no alpha for a smaller denominator to divide -- and the reason is the 15:50
MOC cutoff on the ENTRY leg, which this appendix does not touch.

## Test

Untouched.
