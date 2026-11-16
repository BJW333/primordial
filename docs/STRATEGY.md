# industry_rotation — strategy description

## One-liner
Monthly momentum rotation across 20 US industries: hold the 5 strongest,
expressed through every member ETF, ride each industry until it stops
being strong, inverse-vol weighted. Long-only, no leverage.

## The mechanism (why it should pay)
Industry leadership persists. Capital flows into a sector over months,
not days — earnings cycles, commodity cycles, rate regimes all move
groups of stocks together, and the flows that chase them are slow
(funds rebalance monthly/quarterly, retail follows performance). Buying
what is already leading and holding until leadership actually fades
captures that persistence. The claim is NOT "momentum exists" (that
would be a crowded textbook trade at the single-stock level); the claim
is that industry-level selection beats random industry selection with
identical turnover — skill over diversification, measured against a
turnover-matched null.

## The rule, exactly
1. UNIVERSE: 20 industries expressed by 58 US sector ETFs (semis, oil
   E&P, oil services, regional banks, real estate, homebuilders,
   retail, biotech, insurance, aero/defense, gold miners, energy,
   financials, tech, industrials, health, staples, utilities,
   materials, discretionary).
2. SIGNAL: per ETF, composite ROC = mean of 5/21/63/126/252-day price
   returns, cross-sectionally z-scored across all eligible ETFs.
   Industry score = mean z of its member ETFs.
3. ENTRY: rank industries by score; target the top 5.
4. RIDE (hysteresis band): an industry already held is KEPT as long as
   it ranks inside the top 10 — it is not swapped just because
   something else edged ahead. Only when it decays past rank 10 is it
   dropped and replaced from the top of the ranking. This is what makes
   it "ride the rise" instead of churning the leaderboard monthly.
5. POSITIONS: hold EVERY member ETF of each chosen industry (cluster
   mode), weighted by inverse 63-day volatility, normalized to 100%.
   Typically ~15 ETFs at once.
6. REBALANCE: monthly. Signal reads data through the last close of the
   month; fills at the open of the first trading day.
7. CASH RULE: if every held industry scores negative, go flat. Inert in
   practice (fired 0 of 196 months) — kept for spec fidelity.
8. NO stops, NO shorts, NO leverage, NO intramonth action.

## Performance (backtest 2010-01 .. 2026-06, monthly basis)
    CAGR            ~15.1%  (Python original 15.8%, LEAN 15.1%)
    Sharpe          +0.94 monthly-basis (+0.58 daily-basis)
    maxDD           27.6% month-to-month; 35.0% intraday (2020-03-24)
    turnover        ~0.32 per rebalance; ~1%/day portfolio turnover
    reference       equal-weight all 58 ETFs earns SR +0.84 — the
                    edge is the excess over that, not the raw return
    validation      holdout SR +1.18 (97.6th pct of turnover-matched
                    null, +1.92 sd); above its null in BOTH train and
                    holdout; leave-one-out range +1.04..+1.32;
                    100% of 200 random 40-of-58 subuniverses clear;
                    honest 30-trial luck bar +0.36 — cleared
    capacity        QC estimates ~$320k (thinnest name: BBH)

## Status & caveats
- Backtest-validated, NOT adopted. Pre-registered 24-month QC paper
  test (ROTATION_PREREG.md) decides adoption: PASS needs realised
  monthly SR >= +0.60 AND beating the equal-weight-58 reference.
- Survivorship: the 58 ETFs are today's list. Sector ETFs carry far
  less survivorship than stock lists, but it is not zero.
- Same edge as the sp400 name rotation (corr +0.87) — that leg is
  retired as a duplicate; only this one trades.
- Expect a 2020-style event to draw ~35% intraday. FAIL band is 45%.
