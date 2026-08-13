# Industry-ETF rotation — pre-registration DRAFT (2026-08-11, unsigned)

## The claim being tested
Ranking 20 industries by composite ROC, holding the top 5 (with a
top-10 exit band), inverse-vol weighted across every member ETF,
rebalanced monthly, earns more than randomly selecting industries with
the same turnover.

Frozen spec = QC project `industry_rotation`, main.py (v2, midnight
schedule / batched history), which ports scripts/test_volume_rotation.py
at `SIGNAL=roc DIRECTION=best MODE=cluster REBAL=monthly TOP_N=5
BUFFER=2.0`. Any change to signal, TOP_N, BUFFER, universe, weighting
or rebalance frequency ends this test and starts a new one.

## What the backtest already says (in-sample + holdout, Python)
    full    SR +0.97 | CAGR +15.82% | maxDD 25.7%
    train   SR +0.86 | 78.8th pct of turnover-matched null | +0.80 sd
    holdout SR +1.18 | 97.6th pct | +1.92 sd  -> above its null in BOTH
    equal-weight-all reference SR +0.84 (random industry picking earns
      +0.84 in holdout; only the excess over that is the claim)
    leave-one-out holdout range +1.04 .. +1.32 (no single ETF carries it)
    subsample 200x 40-of-58 universes: mean +1.09, 100% clear the bar

## HONEST TRIAL COUNT — RESOLVED 2026-08-13
The script's printed `luck bar over 12 attempted trials = +0.29` used
the DEFAULT count. The rotation family's real count (weekly/monthly;
sp400/sp500/sp500pit/ETF/crypto/leveraged; SIGNAL roc/obv/mfi/dvol;
MODE single/cluster; DIRECTION best/worst; survivor/PIT) is 25-40.
Re-run with TRIALS=30 -> bar +0.36, holdout +1.18 CLEARS (98th pct,
+1.92 sd, PSR 0.992). All other stats identical to the registered run.
This requirement is satisfied.

## Known open issue (must be resolved before capital, not before paper)
The sp400 NAME rotation and this ETF rotation correlate **+0.869**
(holdout +0.889, hit agreement 80%, 50/50 blend adds -0.01 SR). They are
the same edge expressed twice; only one gets traded, and the ETF version
is chosen because it dominates on every axis: holdout +1.18 vs +0.66,
+1.92 sd vs **-0.51 sd** against its own null, 2 bps vs 14 bps costs,
0.32 vs 0.69 turnover, and far less survivorship exposure. Note the
sp400 leg printed BELOW its null in holdout on 2026-08-11, which
contradicts an earlier "+2.72 sd" figure that has not been reproduced;
that discrepancy is a records question and is tracked separately.

## The paper test
Venue: QuantConnect paper (deep-state's shadow node), started
_____________ (fill in deploy date). Two independent witnesses are NOT
available for this one — it is QC-only — so a fill-quality caveat
applies that deep-state does not have.

Duration: 24 closed monthly rebalances (~2 years), or until a band
triggers, whichever comes first. Monthly strategies are slow to judge;
that is the cost of the frequency, and shortening it is not permitted.

PASS: realised SR >= +0.60 over the paper window AND cumulative return
      above the equal-weight-all-58-ETF reference over the same window.
FAIL: realised SR < 0 over 24 rebalances, OR a DAILY-basis drawdown
      deeper than 45%. (The LEAN backtest touched 35.0% intraday on
      2020-03-24 while healthy -- 25.7% was the MONTHLY-sampled maxDD.
      All SR figures in this document are monthly-basis; daily-basis
      full-period SR is ~0.58-0.60 and is NOT the pass metric.)
Anything else: INCONCLUSIVE, and the strategy stays unadopted.

The equal-weight reference is the point of this test. This strategy's
whole claim is beating random industry selection, and random selection
earns +0.84 SR here. A paper run that makes money but trails the
reference is a FAIL on the actual claim regardless of the P&L.

## Before any real capital
1. Honest TRIALS re-run: RESOLVED 2026-08-13 (see above).
2. RESOLVED 2026-08-13: LEAN backtest ("Creative Red Goat", v2
   main.py) reproduces on the monthly basis: SR +0.94 vs +0.97, maxDD
   27.6% vs 25.7%, CAGR 15.10% vs 15.82%, train +0.86 exact, holdout
   +1.11 vs +1.18. Daily-basis maxDD 35.0% (2020-03-24) is a sampling
   artifact, recorded above. Port is faithful.
3. This is one strategy, not two — the sp400 name rotation is retired as
   a duplicate, not held alongside.

Signed: ____________  date: __________
