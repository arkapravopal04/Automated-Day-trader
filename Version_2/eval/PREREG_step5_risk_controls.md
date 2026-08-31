# Pre-registration -- step 5 risk controls

Written **before** any capped or calendar-excluded book was run, and before any
validation split was read under either control. Committed in this state so the
selection rules below cannot be reverse-engineered from the results they
produced. Nothing in this file may be edited after the first run in
`run_overnight_cap_wf.sh` completes; corrections go in a dated appendix.

Baseline being modified: the step-4 frozen reference,
`logs/p3/on/freeze/freeze_f{k}.json` --

```
--overnight --edge reversal --risk-scale vol --open-spread-bps 1.464
--close-spread-bps 0.262 --carry-bps 0.20 --min-names-frac 0.20
124-name panel, lambda selected on TRAIN, TEST NEVER READ
```

## Why these two and not others

Neither control is a signal filter and neither may reference a return. They are
the two constraints a real book runs unconditionally:

1. **No 37%-of-gross single name in a dollar-neutral book.** `book_weights`
   normalises by gross and stops; nothing bounds one name's share. Measured on
   fold 2: largest position median 11.4% of gross, p90 18.3%, max 37.3%.
2. **No reversal book deliberately holds an earnings print.** The date is
   published weeks ahead, so the exclusion is knowable at decision time in a way
   `abs(gap)` is not.

Both are stated as properties of the book, not as thresholds tuned on an
outcome. Fold 2 is not consulted in setting either value.

## Control A -- per-name weight cap

**Rule, fixed a priori:** on every bar, after selection, sizing, risk scaling
and dollar-neutral demeaning,

```
|w_i| <= KAPPA / n_selected   for all i,    KAPPA = 3.0
```

as a fraction of gross, where `n_selected` is that bar's own book size. The cap
is therefore **three times the equal weight of the book the bar actually holds**
-- it adapts to breadth without being fitted to anything.

**KAPPA = 3.0 is declared without reference to any measurement in this project.**
It is the round number for "no name is worth more than three names", it is
constant across all five folds, and there is no train statistic behind it that
could drift. The alternative offered -- a cap set at each fold's train p99 of
realised max-name weight -- is *not* used as the primary rule, because that
number is fitted to the train book's own concentration and moves fold to fold.
The train p99 is measured and reported as a **diagnostic only**, so the reader
can see where KAPPA=3.0 falls relative to it, and that measurement is made after
this file is fixed.

**Enforcement:** clip to the cap, re-demean over the selected names,
re-normalise gross, iterate to a fixed point (max 32 passes, tolerance 1e-6 on
`max|w| / gross`). Bars that fail to reach the fixed point are counted and
reported rather than silently accepted; a lopsided long/short split is the only
way this can happen and its frequency is a result, not a nuisance.

**Selection under the cap:** lambda is re-selected on TRAIN with the cap
already applied, under the unchanged floors (`--min-active-frac 0.10`,
`--min-names-frac 0.20`). The cap is part of the strategy definition, so the
book lambda is chosen on must be the book that is graded.

## Control B -- flat into scheduled earnings

**Rule:** a name is excluded from the book for any overnight period whose window
contains a scheduled earnings release.

The book's period dated session `D` is entered at `D`'s close and exited at the
next session's open, so:

| release timing | gaps at | period excluded |
|---|---|---|
| **AMC** on date `d` | next session's open | the last session with date `<= d` |
| **BMO** on date `d` | `d`'s open | the last session with date `< d` |
| **timing unknown** | either | **both** of the above |

Mapping is by trading-session adjacency, not calendar day, so a Monday-BMO
print excludes the preceding Friday and a holiday does not shift the window.
Unknown timing excludes both adjacent nights: over-exclusion is the only
direction in which this control can be wrong without silently leaving the
dangerous session held.

**Exclusion is applied in TRAIN and VALIDATION alike**, for the same reason the
cap is: lambda must be selected on the book being graded.

**Calendar:** `data/earnings_calendar.csv`, fetched from Yahoo Finance over
2020-07-01 to 2026-09-30 -- the full panel span plus a month of margin either
side, so no fold and not the sealed test has an edge effect. Fields are ticker,
ET date, and the BMO/AMC flag derived from the published release time
(`<= 09:30 ET` BMO, `>= 16:00 ET` AMC, anything else unknown). Coverage is
reported per ticker; index ETFs in the panel (SPY, QQQ, DIA, IWM, XL*) correctly
carry no rows, and any operating company with zero rows is named in the output
rather than passed over.

**Known idealisation, stated up front:** the calendar records the date the
release happened, used here as if it were the date scheduled in advance.
Companies announce the date two to four weeks ahead and occasionally move it.
This is the standard idealisation for an earnings exclusion and it is not
corrected for.

## Arms, all five folds, all pre-declared

| arm | cap | earnings exclusion |
|---|---|---|
| `base` | -- | -- |
| `cap` | KAPPA=3.0 | -- |
| `earn` | -- | yes |
| `both` | KAPPA=3.0 | yes |

`base` must reproduce `logs/p3/on/freeze/` **exactly**. It is the regression
gate on the code change: if the controls being off is not a no-op, nothing
below it is admissible.

## The re-freeze (step 6)

**The new frozen reference is the `both` arm**, declared here, before any of the
four arms has been run. It is not chosen by whichever arm validates best. Both
controls are risk constraints a book runs unconditionally, so the reference is
the book with both on; `cap` and `earn` are reported to decompose the change,
not to be selected between.

## What is expected, so that it cannot be reported as a surprise

The ratio is expected to **fall**. `(edge - lambda*cost)+ / cost` divides by
cost, so cheap names take large weights, and the hurdle leaves few survivors to
share the gross. The concentration and the good ratio are the same mechanism.
The quantity being measured in step 5 is the exchange rate between them, not
whether the controls "work".

Neither control is expected to fix fold 2. Zeroing beyond 6 sigma still left
Sharpe -2.16, the 94.8% body loses 490 bps on its own, and monthly IC is
negative Dec-Mar. Fold 2 has two independent problems and only one of them is
addressable here. **The headline stays "negative in one of five folds,
unexplained."**

## Test

Untouched, under every arm.

---

# Appendix A -- 2026-08-30 -- absolute cap. AMENDMENT, closing a defect in Control A.

**This is an amendment to a rule, not a revision of a result.** It is written
before the arms it defines have been run, and before any validation split has
been read under the amended rule. The body above is unedited, per its own terms.

## The defect

Control A's stated intent, quoted from `## Why these two and not others` above,
is:

> **No 37%-of-gross single name in a dollar-neutral book.**

The rule written to serve that intent -- `|w_i| <= KAPPA / n_selected` as a
fraction of gross -- does not serve it. `KAPPA / n_selected` is a *relative*
cap: it bounds a name against the breadth of the book it sits in, and says
nothing about the book's absolute concentration. At `KAPPA = 3.0` it permits

| n_selected | cap, as share of gross |
|---|---|
| 30 | 0.100 |
| 12 | 0.250 |
| 8 | 0.375 |
| **6** | **0.500** |

so for any book narrower than 8 names the cap permits *more* than the 37% that
motivated the control, and at 6 names it permits half the book in one position.

This is not a theoretical edge case; it is what the run did. In
`logs/p3/on/cap/`, VAL, largest single name as a share of gross:

| arm | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|
| `base` (uncapped) | 0.346 | 0.373 | 0.272 | 0.270 | 0.500 |
| `cap` | 0.250 | 0.273 | 0.200 | 0.242 | **0.500** |
| `both` | **0.500** | 0.273 | 0.200 | 0.242 | **0.500** |

`both` on fold 1 reads 0.500 against `base`'s 0.346: the capped book is *more*
concentrated in absolute terms than the uncapped one it was meant to constrain.
The corresponding `max_mult_max` is 3.00, so the cap bound -- on a six-name
bar, at exactly `3.0 / 6`. The control did what it said and failed what it was
for.

A rule that permits the outcome its own justification names as unacceptable is
a **defect in the rule**. Correcting it is not a result-driven revision, and the
distinction matters for whether this survives scrutiny later: the amended rule
below is strictly tighter than the one it replaces on every bar, in every fold,
so it cannot be a search for a better number. It can only make the reported
book worse.

## The amendment

Control A's enforcement line becomes, on every bar, after selection, sizing,
risk scaling and dollar-neutral demeaning:

```
|w_i| <= min( KAPPA / n_selected ,  A )   for all i,   as a fraction of gross

KAPPA = 3.0   (unchanged)
A     = 0.10  (new)
```

`KAPPA` is unchanged and keeps its original meaning: no name is worth more than
three names of the book actually held. `A` adds the floor the relative rule
never had: no name is more than a tenth of the book, however narrow the book
gets.

**`A = 0.10` is set from first principles and written down before running.**
Ten percent of gross is the conventional single-name limit; stated the other
way, it requires the book to be at least as diversified as a ten-name
equal-weight book before any name may take an outsized share. It is **not** read
off the max-share distribution measured above. That distribution is quoted here
only as evidence that the old rule failed, and the number 0.10 is not any
quantile of it -- it is below every one of the fifteen cells in the table, so it
could not have been fitted to them even accidentally. `A` is constant across all
five folds and there is no train statistic behind it that could drift.

## Feasibility, declared in advance

An absolute cap has a feasibility bound the relative one did not. Gross 1 spread
over `n_selected` names needs `A * n_selected >= 1`, i.e. **at least 10 names**;
dollar-neutrality additionally needs each leg to carry half the gross, i.e. at
least `0.5 / A = 5` names on the smaller side. `min_names` is 2, so bars
narrower than that will occur.

**Handling, fixed here:** the fixed-point iteration is unchanged (clip,
re-demean over the selected names, re-normalise gross, max 32 passes, tolerance
1e-6). On an infeasible bar it drives the book to the tightest weighting the bar
admits -- equal weight over the selected names -- and terminates there. Such
bars are **counted and reported** as `cap_infeasible` and they are reported
separately from `cap_unconverged`, which keeps its existing meaning of a
lopsided long/short split. Nothing is silently accepted above `A`: a bar that
ends above `A` ends there because no dollar-neutral book of that breadth exists
below it, and the count says how often that happened.

`cap_infeasible > 0` is a **result about the book's breadth**, not a nuisance to
be tuned away. It is not permitted to motivate a change to `min_names`, to
`A`, or to the breadth floors, either in this step or in step 6.

## What is expected, so that it cannot be reported as a surprise

The amended cap is tighter than the old one on every bar where
`KAPPA / n_selected > A`, i.e. every book narrower than 30 names. Book breadth
in `logs/p3/on/cap/`, `both` arm, VAL, is 17.6 / 33.5 / 37.5 / 41.6 / 44.1
names, so **`A` is the binding constraint on most of fold 1 and a minority of
folds 2-5**, and fold 1 is where the change will be largest. That is a
consequence of holding the conventional number fixed, not a reason to move it.

The ratio and Sharpe are expected to **fall further** than they did under the
relative cap, for the reason already given in the body: the `(edge - lambda*cost)+ / cost`
rule concentrates into cheap names, and the concentration and the good ratio are
the same mechanism. Fold 1 degraded under `both` (net Sharpe -0.73 against
`base`'s -0.05) and it is the fold this amendment binds hardest, so **fold 1
getting worse again is the expected outcome and will not be reported as
evidence against the controls.** If fold 1 instead improves, that is a surprise
and is to be reported as one -- it would mean the relative cap's permitted
50%-of-gross position was itself carrying the loss.

## Arms and the new reference

Run in `logs/p3/on/cap2/`, all four re-run so the amended code carries its own
regression gate:

| arm | cap | earnings exclusion |
|---|---|---|
| `base` | -- | -- |
| `cap2` | KAPPA=3.0, A=0.10 | -- |
| `earn` | -- | yes |
| `both2` | KAPPA=3.0, A=0.10 | yes |

**Two gates, both of which must pass before any cell below them is read:**

1. `base` must reproduce `logs/p3/on/freeze/` exactly, as before.
2. `base` and `earn` must reproduce `logs/p3/on/cap/base_*` and
   `logs/p3/on/cap/earn_*` exactly. Neither arm passes `A`, so the absolute-cap
   code change must be inert in both. If it is not, the change is not confined
   to the cap.

**The new frozen reference is `both2`**, declared here before it has been run,
on the same grounds the body gave for `both`: both controls are constraints a
real book runs unconditionally, and the reference is the book with both on. It
supersedes `logs/p3/on/freeze2/`, which is retained rather than deleted -- it is
the record of the defect.

## Test

Untouched, under every arm, as before.
