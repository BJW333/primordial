# TAXLOSS (C2) PRE-REGISTRATION -- sign BEFORE running validate_taxloss.py

## History (recorded honestly)
C2 = December distress cell from the zero-trial screening session:
train +4.73% (t +6.1), tail +9.79% (t +5.0); cohort gate PASSED
(check_c2_cohorts.py, 14/16 Decembers positive). A TAXLOSS_PREREG.md was
drafted in an earlier session but never committed; this file replaces it
on 2026-10-25, before either gate below has been run.

## Mechanism
Tax-loss selling: investors dump losers before year end regardless of
price; the selling has a deadline and is not information. Price is pushed
below value in December and recovers in January once it stops.

## Frozen cell (identical to check_c2_cohorts.py)
S&P 600 current membership. Signal: close <= lowest close of the prior 21
days AND return t-127 -> t-1 <= -30% AND date Dec 1-24; events >= 5 bars
apart. Buy next open, sell at the close 20 trading days after the signal.
Outcome = stock minus SPY over the same window, minus 36 bps. Unit = one
equal-weight mean per December cohort.

## Gate 1 -- placebo months
Same cell with the month replaced by each of Jan..Dec.
PASS only if December ranks #1 of 12 by across-cohort mean.

## Gate 2 -- cost stress
December across-cohort mean > 0 at 2x (72 bps) and 3x (108 bps) cost.

## Decision
Both pass -> Norgate PIT sp600 re-test under this same cell, then a
calendar-windowed runner, then forward paper. Either fails -> C2 closed.
S&P 400 / S&P 500 rows printed by the script are context only.
Logged to MECHANISM_MAP.md either way. One invocation; no edits to the
cell, window, distress threshold or cost after the look.

## Controls (2026-10-25)
Planted December rebound on synthetic data -> December #1 of 12, positive
at 3x cost, PASS (pipe check).

Signed: Blake Weiss
