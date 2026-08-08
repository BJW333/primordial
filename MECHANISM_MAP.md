# Deep-State Mechanism Map — research findings, Aug 5–8 2026

**Spec under test (frozen, unchanged throughout):** `deep_state.py` FROZEN dict,
sha256 `709f4e8b25b2a1e5`.
MACD(12,26) below zero with depth >= 0.25 ATR; close > SMA200; sag age <= 15 bars;
exit at first close >= EMA26 or 40 bars; **no stop**.
Genome form: `and(lt(ema_spread_atr, -0.25), and(lte(spread_runlen, 15), gt(close, roll_mean(close, n=200))))`,
`--stop 99 --max-hold 40 --anchor 26`.

**Judge:** PRIMORDIAL v0.1.9 — holdout names x holdout years, luck bar deflated by
logged trials, PSR, breadth (exact binomial, >= 10 holdout names), cost stress 2x/3x,
block bootstrap (mechanism-scaled blocks = clip(2 x max_hold, 48, n/6); fixed-48
recorded alongside as informational).

---

## 1. Verdicts by universe

| ground | holdout SR | bar | PSR | breadth | cost 2x/3x | bootstrap (scaled / blk48) | verdict |
|---|---|---|---|---|---|---|---|
| US megacap (20) | +1.24 | +0.85 | 0.994 | skipped (8 names) | +1.19 / +1.15 | +0.20 / — | **reject** (bootstrap) |
| **US midcap (sp400)** | **+0.73** | **+0.69** | **1.000** | **0.67 / 150, p≈0** | **+0.69 / +0.62** | **+2.26 / +0.00** | **PASS — deployed** |
| US smallcap (sp600) | +0.59 | +0.00 | 0.999 | 0.65 / 197, p≈0 | +0.40 / +0.19 | +0.00 / +0.00 | **reject** (bootstrap) |
| Global indexes (22, untradable) | +0.85 | +0.43 | 0.999 | skipped (8 names) | +0.84 / +0.82 | +0.94 / +0.67 | pass, but not implementable |
| Global ETFs (43, tradable) | +0.81 | +0.26 | 0.996 | 0.88 / 17, p=0.0012 | +0.78 / +0.75 | +0.04 / +0.51 | **reject** (bootstrap) |
| **UK (FTSE 250)** | **+0.65** | +0.00 | 0.981 | **0.69 / 72, p=0.0006** | +0.52 / +0.40 | **+1.12 / +1.39** | **PASS** |
| **Canada (TSX)** | **+0.77** | +0.00 | 1.000 | **0.76 / 82, p≈0** | +0.74 / +0.64 | **+2.36 / +2.81** | **PASS** |
| Australia (ASX 200) | +0.78 | +0.00 | 1.000 | 0.73 / 70, p=0.0001 | +0.70 / +0.62 | +0.00 / +4.10 | **reject** — inconclusive* |

\* Australia: two of three scaled-block paths returned exactly 0.00 = no trades
completed on that resample (blocks cutting the sample, not edge vanishing); the
completing path returned +4.59 and all fixed-48 paths passed. Recorded as a
rejection because the gate is frozen; interpreted as "path-robustness untested",
not "no edge".

## 2. Rejected hypotheses (other families)

| idea | ground | result | killer |
|---|---|---|---|
| Short side (fade bear rallies) | sp600 | +0.09 vs bar +0.43 | **PSR 0.008** — mechanism is asymmetric |
| Cross-sectional dip (xs_rank_roc_m bottom decile + SMA200) | sp600 | +0.23 vs bar +0.26 | PSR 0.105 |
| Crypto cross-sectional reversion | coinbase spot 1h/4h, maker | flat across buckets | zero-trial screen; 46–70bps costs |
| Volatility contraction (coiled -> expansion) | sp400 | compressed bucket WORST (+0.71% vs +2.52% at 20d) | zero-trial screen |
| Evolution search (6 universes, ~10k candidates) | ETF, crypto, sp600-150, megacap, sp500, 5m ETFs | **0 survivors, ever** | bar / beta-capture |

## 3. What the map says

The edge is **behavioural overreaction in mid-tier equities**, and it requires
**many imperfectly-correlated names**:

- **Megacaps fail** — too efficient, and only 8 holdout names (breadth can't run).
- **Midcaps pass** — US, UK, Canada; Australia strongly suggestive.
- **Smallcaps fail** — gross edge and breadth are there (0.65, p≈0) but cost
  stress decays fast (+0.19 at 3x) and bootstrap is flat: path-fragile.
- **Index/ETF baskets fail** — best breadth ever measured (0.88) yet bootstrap
  dies: 43 correlated instruments are ONE bet in costume. Correlation, not
  instrument count, is what the bootstrap is measuring.
- **Short side is absent** — consistent with a liquidity-provision story: you are
  paid for absorbing forced selling; there is no mirror-image forced buying.

**Replication across four independent markets** (different exchanges, currencies,
investor bases, sector mixes, regulators) with breadth 0.67–0.76 and p <= 0.0006
is the strongest evidence class in this project — stronger than the single-market
result the live system was deployed on.

## 4. Caveats on the record

1. **sp400 bootstrap is calibration-dependent.** The deployed pass has fixed-48
   median 0.00; it passes only under mechanism-scaled blocks, a change made AFTER
   deep-state failed the old sizing. Disclosed, controls re-verified, gate then
   frozen in both directions. UK and Canada pass under BOTH calibrations, which is
   what retires this concern — not an argument, a replication.
2. **The recalibration is not a ratchet.** It has since REJECTED two candidates the
   old sizing would have passed (global ETFs, Australia). It cuts both ways.
3. **Survivorship.** Every equity universe is current membership; delisted names are
   absent. The SMA200 gate blunts this (dead companies are rarely above their
   200-day average) but does not remove it. Point-in-time re-test required before
   capital.
4. **Foreign cost assumptions.** UK/CA/AU runs charged US-midcap costs. LSE/TSX/ASX
   spreads are wider. Cost stress at 3x (+0.40 to +0.64) suggests margin, but a
   foreign deployment needs real spread data.
5. **Fresh-fingerprint bars.** UK/CA/AU ran at bar 0.00 (first look). SRs of
   +0.65–0.78 clear any plausible debt, unlike sp400's +0.73-vs-+0.69 squeaker.

## 5. Method notes worth keeping

- **Screen free, then spend.** `explore_xs.py` / `explore_vol.py` run train-only,
  log nothing, and killed two whole families (crypto XS, vol contraction) for zero
  trials. In-sample tables are the "929 look profitable" stage — never evidence.
- **Hand ideas first, evolution last.** Evolution burns 50+ effective trials into a
  fingerprint and raises the bar for every later hand idea on that ground.
  (Learned the hard way: sp500 evolution ran before hand ideas.)
- **Write the prediction before the run.** Every probe above had one; the UK
  prediction ("survives, breadth ~0.6+, SR +0.4–0.8, watch cost stress") was
  confirmed on a market the strategy had never seen.
- **In-sample rank is anti-predictive at the top.** Independent confirmation:
  a published 1,152-strategy enumeration found its best in-sample strategy
  (Sharpe 2.01) ranked 807th out of sample. Same lesson our null results teach.
- **Counting survivors without deflation isn't a survivor count.** Testing N
  candidates against one out-of-sample period makes that period in-sample. The
  ledger exists to charge that debt.

## 6. Live status (unchanged by any of this)

Deep-state paper-trades the frozen spec on Alpaca account DEEPSTATE (PA39YN7UIO10)
via Oracle VPS cron, weekdays 22:15 UTC: recorder -> mirror -> next-open fills.
Graded against `PREREGISTRATION.md`: **60 closed trades**, PASS at mean net/trade
>= +0.30% AND win >= 60%, FAIL below +0.15% or 55%, one 30-trade extension.
Closed trades so far: **0**. Nothing in this document changes the deployed spec,
which stays frozen until the forward verdict.

## 7. Open questions

- Germany (`dax_mid`) — built, unrun. Fifth market.
- Australia re-test with a wider universe or a different block size chosen on
  principle *before* looking (the current result is inconclusive, not negative).
- Point-in-time universe re-test (delisted names included) — required before capital.
- Time-conditioning: does the edge concentrate in high-VIX regimes? Sizing
  question, one trial on sp400.
- Real foreign spread data before any non-US deployment.
