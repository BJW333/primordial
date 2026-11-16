# Night-only momentum and GAPSTEP -- SHELVED 2026-11-15

Shelved by Blake on 2026-11-15. Nothing here is live. Scripts are in `scripts/archive/`, outputs in
`logs/archive/`. All rules were written down before running; numbers are Primordial numbers
(2017-02 -> 2026-05, momentum_v2 STEADY top-20 picks, Alpaca SIP daily open/close), which run
hotter than QuantConnect.

## What was found
1. **Momentum picks earn everything overnight.** Close -> next open: about +36% a year before costs
   (Sharpe 1.54). Open -> close: about -6% a year. Held normally: +25.7%, Sharpe 0.80, worst drop -46%.
   (`structural_screen.py`)
2. **Night-only every day** (buy the close, sell the open): +23.1% / Sharpe 1.07 at 2 bp per side,
   +11.3% / 0.61 at 4 bp. About 10,000 trades a year, so costs decide it. Failed its rule narrowly
   (early-window return 21.8% vs 22.0% needed; fails at 2x cost).
3. **Night-only on cheap funds** (`night_screen.py`): stock funds (SPY QQQ IWM, from 1999) earn
   +12.4% a year overnight before costs but only +5.2% after; commodity funds were a 2016-26
   artefact (zero in 2012-21). Both failed.
4. **Step out only after a gap up** (`night_gap.py`, GAP1): sell at the open only if a stock opens
   more than 1% above the prior close, buy back at the close. +30.3% / Sharpe 1.02 overall, about
   1,060 round trips a year, barely cost-sensitive -- but worse than plain holding in 2017-21,
   because in 2020 gap-ups kept running all day. Failed the both-periods rule.
5. **GAPSTEP** (`gapstep.py`): GAP1 plus a switch -- step out only if, over the last 63 trading
   days, the strategy's own gap-up stocks lost money between open and close. +32.9% / Sharpe 1.00 /
   worst drop -41%; beat plain holding in both periods (30.4% vs 29.3%, 35.5% vs 22.1%) and at 2x
   costs; random control failed. PASSED its rule.

## Why it was shelved
- GAPSTEP is CHIMERA's stock picker plus one rule, without CHIMERA's risk controls: correlation
  with CHIMERA +0.69, worst drop -41%. It is a momentum upgrade, not a second strategy.
- The drawdown has no free fix: with CHIMERA's calm / normal / stressed sizing it drops to -22% but
  the return falls to +19.0% (a flat 59% size gives -25% and +19.8%).
- The switch was designed after seeing that 2020 broke the plain rule; the sample has one runaway
  rally, so the pass is weak evidence.
- Fills are assumed at the daily-bar open and close; real opening-auction (limit-on-open) and
  closing-auction orders were never tested.

## If it is ever reopened
- Test the step-out rule inside a CHIMERA test copy on QuantConnect with real limit-on-open and
  market-on-close orders, against the live version (25.3% a year, Sharpe 0.96, worst drop -20.8%).
- At CHIMERA-like sizing the rule added about 3.4% a year and trimmed the worst drop 27% -> 22%.
- Do not re-pick the 1% threshold, the 63-day window or the switch on this data.
