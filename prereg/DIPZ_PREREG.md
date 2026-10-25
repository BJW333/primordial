# DIPZ PRE-REGISTRATION -- vol-scaled dip-buy -- sign BEFORE running

## Lineage
Successor to the fixed-5% S&P 500 dip-buy screen (dip5_screen.py,
2026-10-25): survivor mode looked strong (H=20 t +6.3); PIT with 40%
departed coverage collapsed it (t -1.5, post-2018 t -2.9). Formal verdict
VOID (coverage < 50%), directional read: survivorship was the edge.
The S&P 500 2010-26 sample is therefore SPENT for this family.

## Hypothesis
A fixed % trigger mixes shocks with routine noise across names of very
different volatility. Measuring the drop in units of the name's own
typical range isolates genuine overreactions, which revert.

## Trigger (frozen)
z_t = -(close_t/close_{t-1} - 1) / atr_pct_{t-1}, ATR 14d, lagged 1 bar.
Event z_t >= k. Buy next open, hold H bars, exit at close. 20 bps RT.

## Ground
manifests/sp400_symbols.txt, CURRENT membership, 2010-03 -> 2026-06.
Survivor-biased in the flattering direction, so this ground can only KILL.

## Primary cell (one, frozen): k = 3, H = 10
Measure: net return minus SPY over the same window, t clustered by date.
- PASS-SCREEN: mean > 0 and date-clustered t > 2 in BOTH halves
  (pre/post 2018-01-01). Not a pass of the idea -- owes PIT sp400
  (Norgate) under this same cell, then a prereg'd genome + gauntlet +
  duplication check vs deep_state.
- Anything else: CLOSED on sp400. Logged to MECHANISM_MAP.md.
- The k in {2,3,4} x H in {1,5,10,20} grid is context; no adjacent-cell
  promotion. --ground sp500pit output is context only (spent sample).

## Controls
Planted rebound (synthetic -8% drops + 10-day drift): SURVIVES, t ~ 11-14.
Null (Gaussian, no plant): CLOSED. Both verified 2026-10-25.

## STOP
One invocation of the primary run. No k/H/cost edits after the look.
Failed downloads may be re-fetched (data repair, not a new look) -- record it.

Signed: __________
