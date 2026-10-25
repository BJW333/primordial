# DIPGAP PRE-REGISTRATION -- gap vs intraday dip split -- sign BEFORE running

## Lineage
dip5 (fixed 5%, S&P 500) VOID-dead on survivorship; dipz (3x ATR, sp400)
CLOSED. Read: a single-day drop mixes rebounders with continuers and they
average to zero. This family tests ONE separator.

## Hypothesis (direction stated now)
News drops keep drifting; non-news (flow) drops revert. Overnight gap is
the news proxy. Expected: INTRADAY drops beat GAP drops over 10 days, and
INTRADAY drops beat SPY.

## Frozen spec
Event: close-to-close drop > 5%. Gap share g = (open - prev close) /
(close - prev close). INTRADAY g <= 1/3, GAP g >= 2/3, mixed = report only.
Buy next open, hold 10 bars, exit at close. Return minus SPY, date-clustered t.
Costs RT: sp600 30 bps, sp400 20, sp500 10. Events with |10d raw| > 200%
dropped as data glitches (count printed). Split 2018-01-01.

## Grounds
- sp600 PRIMARY: fresh, current membership.
- sp400 REPLICATION: current membership; 3x-drop events seen in dipz.
- sp500 REPLICATION: PIT-gated; 5% events seen in dip5 (not this split).
Spread is conservative under survivorship (missing dead names mostly
belong to GAP). INTRADAY level is survivor-flattered on sp600/sp400.

## Decision rule
- (a) sp600 spread INTRADAY - GAP > 0, t > 2, in BOTH halves
- (b) sp600 INTRADAY net excess > 0, t > 2, in BOTH halves
- (r) sp400 and sp500 full-sample spreads both > 0
- (a)+(b)+(r): SURVIVES SCREEN -> PIT sp600 (Norgate), then genome +
  gauntlet + duplication check vs deep_state. Not a pass of the strategy.
- (a)+(r) only: mechanism real, long side not tradeable here. Short-the-gap
  would be a NEW family with its own prereg.
- Else: CLOSED. Logged to MECHANISM_MAP.md either way.

## Controls (verified 2026-10-25)
Planted (intraday rebound / gap drift): SURVIVES, spread t ~ 14 per half.
Null (random open placement, no plant): CLOSED.

## STOP
One invocation. No cutpoint, hold, threshold or cost edits after the look.
No adding close-location, volume or trend filters to this family.

Signed: __________
