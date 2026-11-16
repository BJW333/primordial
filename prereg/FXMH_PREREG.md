# FXMH `_idea` PRE-REGISTRATION -- sign BEFORE any real-data run

## Question
Does the multi-horizon sign-vote score (mh_vote) add anything to a
Donchian 20d breakout on G10 FX daily -- the SECOND ground for a
hypothesis already dead on crypto daily (2026-09-09, MECHANISM_MAP), and
the FIRST two-sided test of it (FX is symmetric: no borrow, shorts are
just the mirror).

## Standing against this run, stated now
- mh_vote is dead on crypto: vote_alone was ACTIVELY negative (-2.47).
- FX family verdict 2026-11-15: 0/9 ideas passed; FX closed "unless a new
  mechanism-based idea appears". mh_vote is trend-following in the most
  mined FX universe, not a new mechanism. This run is charged as a
  deliberate exception for one reason: daily holds are the first FX idea
  where our cost structure doesn't kill the trade (0.7-1.0 bp/side vs
  multi-day holds).
- Expected outcome: all rows dead. The run buys a ledgered cross-asset
  close of mh_vote, nothing more.

## Signal (frozen; identical to crypto except windows in trading days)
Four DISJOINT segments [0,5) [5,10) [10,21) [21,42) trading days; each
votes +1/-1 on its return, 0 inside the dead zone (0.5 * sigma_d *
sqrt(seglen), sigma_d = 90d rolling std). Score in [-4, +4].

## Ground
8 G10 currencies vs USD (AUD NZD CAD NOK EUR GBP JPY CHF), the
fx_screen Dukascopy hourly cache (data/fx_1h/<CCY>.csv, UTC; CAD NOK
JPY CHF inverted to USD-per-unit with high/low swapped), resampled to
daily: signal 16:00 NY, fill same day 17:00 NY -- the fx_screen clock.
Costs: fx_screen per-side bp table, every row also at 2x. Known bias
carried over from fx_screen: spot only, no carry -- mild for G10 trend,
noted. Runner: scripts/fxmh_screen.py. Loader + resample verified on
the real cache (dates/coverage only; no signal or return was computed
or seen before signing).

## Frozen rows (4 judged + 1 control; exactly these, no post-look edits)
1. BASE        Donchian20 two-sided (up-break long / down-break short).
               PARENT/CONTROL at this ground.
2. VOTE_ALONE  score crossing to |2|, no breakout. PARENT.
3. GATE2       breakout AND same-sign |score| >= 2.   <- THE IDEA
4. GATE4       breakout AND unanimous score (+/-4). Sizing-ladder probe.
5. RANDOM      GATE2-count random entries on real prices, same exits.
               Must fail on real data.
All rows: 2.5 x ATR14 stop, 40d max hold, one position per currency, 10%
vol target per position, portfolio = sum of 8.

## Decision rule (pre-stated)
- PASS bar per row (all legs required): Sharpe >= 0.5 over 2012-07 ->
  2021-10 AND >= 0.7 over 2021-11 -> end AND whole-period Sharpe > 0 at
  2x costs AND >= 100 trades whole-period AND 50/50 equal-risk monthly
  blend with CHIMERA > 1.42 (the CHIM series embedded in fx_screen.py,
  copied verbatim into the runner; fx_screen's blend math).
- GATE2 is INTERESTING only if it passes AND its whole-period Sharpe
  exceeds BOTH parents'. GATE4 informs sizing only if GATE2 is
  INTERESTING and GATE4 >= GATE2.
- RANDOM passing any bar on real data = implementation bug, run void,
  fix and this prereg's single re-run allowance is consumed.
- Anything else: mh_vote CLOSED cross-asset (crypto + FX). The crypto
  sign-flip prohibition carries over unchanged: a bad result here does
  not license fading the vote on any ground.
- STOP: no threshold edits (|2|/|4|, dead zone 0.5, windows 5/10/21/42,
  stop/hold, vol target), no added rows, no window re-picks after any
  real-data look. Results logged to MECHANISM_MAP.md win or lose.
- Charged looks: 4 judged rows, 1 ground. 5th+ FX-family judged rows
  lifetime -> family false-pass math gets noted in the close-out.

## Controls (run BEFORE real data; verified in sandbox AND on this Mac)
- --smoke noise: all rows fail (10-universe calibration in sandbox:
  0 false passes with the 100-trade floor).
- --smoke planted: all rows pass AND each beats RANDOM's whole-period
  Sharpe by >= 0.5 (relative criterion: the stop/hold bracket alone
  harvests planted drift, so absolute RANDOM failure is required only
  on noise and real data).

## Order
1. Sign below. 2. git commit. 3. One real run:
   python3.10 scripts/fxmh_screen.py | tee logs/fxmh_screen.txt
4. Log + commit.

Signed: __________
