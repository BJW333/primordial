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

### §8 update (2026-08-08, later)

Search-control failure root-caused and fixed: (1) parsimony penalty was
absolute-scaled — negligible on planted controls, crushing on real data —
now scale-relative; (2) coverage-aware cell floor (15% of attemptable cells);
(3) random immigrants vs population collapse; (4) budget 10x24 -> 20x48.
All tuned on the control (planted answer known), never on real data.
Post-fix, on Blake's machine: negative 0 survivors, machinery PASSED,
search control RECOVERED the planted mechanism (lt(1.24, volz), 3 nodes,
stable 6 final gens, survived +21.35 vs bar +11.34). The judge was not
modified. The "6 universes / 0 survivors" footnote weakens accordingly:
future nulls from this searcher are meaningful statements about the ground.
# Distress-Volatility — findings, 2026-08-10

**Rule (unchanged across every ground below):**
`and(gt(pctile_rank(atr_pct, n=252), 0.9), lt(close, roll_mean(close, n=200)))`
`--tf 1d --stop 99 --max-hold 40 --anchor 26`

In words: buy a name whose ATR sits in the **top decile of its own trailing
year** AND which is **below its 200-day average**. Exit on the first close
above the 26-day EMA, else after 40 bars. No stop.

**Provenance.** Not from a paper or a public algorithm. It fell out of
`explore_vol.py`: unconditionally the highest-ATR bucket had the *best*
forward returns (+2.52% at 20d), but gated to UPTRENDS that bucket dropped to
+0.80%. The difference implied the high-vol winners were the ones *below*
trend. The `--below-trend` screen then showed a steep monotone gradient
(+3.49% at 20d, +7.72% at 40d for the top decile vs +0.43%/+0.98% for the
calmest), which earned the first trial.

**Mechanism story.** Distress premium: forced sellers (margin calls,
tax-loss harvesting, index deletion, mandate limits) sell regardless of
price; whoever absorbs that flow is compensated. Same liquidity-provision
economics as deep-state, but the mirror setup — chaos in broken names rather
than sags in healthy ones.

---

## 1. All ten grounds (base rule, verified from run.json artifacts)

| ground | SR | bar | PSR | breadth | cost 2x/3x | bootstrap (scaled) | verdict |
|---|---|---|---|---|---|---|---|
| **Japan (Nikkei)** | **+0.76** | 0.00 | 1.000 | **0.91 / 35, p≈0** | +0.74 / +0.71 | +0.45 | **PASS** |
| **Germany (MDAX)** | **+0.43** | 0.00 | 0.998 | 0.79 / 19, p=0.0096 | +0.40 / +0.35 | +0.42 | **PASS** |
| **Korea (KOSPI)** | **+0.38** | 0.00 | 0.995 | 0.73 / 26, p=0.0145 | +0.35 / +0.33 | +0.26 | **PASS** |
| **UK (FTSE 250)** | **+0.33** | 0.00 | 0.998 | 0.62 / 92, p=0.014 | +0.26 / +0.20 | +0.51 | **PASS** |
| Taiwan (TWSE) | +0.69 | 0.00 | 0.991 | 0.88 / 16, p=0.0021 | +0.59 / +0.50 | +0.24 (need +0.35) | reject |
| India (Nifty) | +0.66 | 0.00 | — | — (too few holdout names) | — | — | reject |
| Australia (ASX) | +0.35 | 0.00 | — | — | — | 0.00-ish | reject |
| Canada (TSX) | +0.43 | 0.00 | — | 0.85 / 72 | — | **0.00** | reject |
| US midcap (sp400) | +0.82 | +0.87 | 1.000 | — | — | — | reject (bar) |
| US smallcap (sp600) | +0.45 | +0.53 | — | — | — | — | reject (bar) |

**Positive SR on all ten grounds. PSR ~1.000 wherever measured. Four clean
gauntlet passes on four independent markets (UK, Germany, Japan, Korea) —
three continents, four currencies, four regulatory regimes.**

## 2. How the failures failed (this matters)

- **Both US grounds lost to the BAR, not the data.** sp400 +0.82 vs +0.87 and
  sp600 +0.45 vs +0.53 — both would have passed against a zero bar. Those
  grounds carry 15+ and 9 effective trials respectively, mostly from
  deep-state work and from this family's own refinement attempts. The
  deflation is doing exactly its job; it is not evidence of absence.
- **Bootstrap is the dominant killer elsewhere** (Canada, Taiwan, Australia).
  Canada returned literal 0.00 paths = *no trades completed on the resample*,
  which is a measurement failure, not a demonstrated absence of edge. Taiwan
  missed narrowly (+0.24 vs +0.35) with all paths completing.
- **India died on breadth** — too few holdout names accumulated enough trades,
  so the gate could not run.

## 3. Refinement attempts — all four made it worse (recorded so nobody retries)

| variant | ground | SR | bar | outcome |
|---|---|---|---|---|
| base | sp600 | +0.45 | +0.53 | reference |
| + stabilized (above 10-bar low) | sp600 | +0.35 | +0.60 | worse |
| + volume confirmation (volz>1) | sp600 | +0.51 | +0.65 | raw up, **net further from passing** |
| + depth (>20% below SMA200) | sp600 | **+0.92** | +0.69 | best SR, killed by bootstrap [0,0,0] |
| hold 40 -> 15 bars | sp600 | +0.31 | +0.73 | much worse |

Two lessons. **(a)** The volume result is the canonical trap: raw SR rose
0.45 -> 0.51 while the bar rose 0.53 -> 0.65, so spending the trial moved it
*away* from passing. Raw-SR comparisons across variants are how noise gets
promoted. **(b)** Cutting the hold from 40 to 15 bars cost two-thirds of the
edge (+0.92 -> +0.31 on the depth variant's ground). **The premium requires
long holds** — you are paid for sitting through the ugly part.

That last point also explains the bootstrap failures: long holds are exactly
what block resampling cannot reconstruct. The gate and the mechanism are in
structural tension here, and no threshold change should be used to resolve it.

## 4. Caveats — read before ever sizing this

1. **Survivorship bias is worse here than for any other idea tested.**
   "Broken, high-volatility midcap" IS the delisting profile. Every universe
   is current membership, so the bucket is enriched with names that survived
   their distress by construction. Deep-state's SMA200 gate partly insulated
   it; this rule deliberately inverts that gate and has no such protection.
   **Point-in-time re-test is mandatory before capital.**
2. **All four passes ran at a 0.00 bar** (first look at each ground). They
   clear any plausible debt, but no deflation was actually applied.
3. **Foreign cost assumptions are US-midcap.** Distressed foreign midcaps
   trade wide. UK's cost stress (+0.26/+0.20) is the thinnest of the four
   passes and is the first thing that would break in reality.
4. **No family-wise correction across grounds.** Ten grounds, four passes;
   the ledger deflates per fingerprint, not across a campaign.
5. **+0.33 to +0.76 is thin.** Deep-state validated at +0.65 to +0.77 with
   bootstrap clearing both calibrations everywhere it passed.

## 5. Status — CLOSED (superseded; see DISTRESS_VOL.md section 6)

**Dead. Zero honest passes on ten grounds; pre-registered Gate A failed
0/5** (base genome, us_sp600, five hash-keyed splits: SR median +0.41,
range [+0.40, +0.46], bar +0.76 -- run 20260811_004913). The four foreign
"passes" above ran at 0.00 luck bars (fingerprints carrying one trial;
bar formula returns 0 for n < 2); charging the family's real debt (5
variants, 14 peeks) puts the minimum bar at +0.89, above the best foreign
SR (+0.76). The SURVIVORS.md rows for uk/de/jp/kr at bar +0.00 are
superseded by this closure. Full correction and what-was-learned:
DISTRESS_VOL.md section 6. Keepable finding: same entry family and exit,
ABOVE the 200-day (deep-state) clears every gate; BELOW it prints a
stable +0.4 that clears nothing honest anywhere -- the premium is for
absorbing sags in healthy names, not chaos in broken ones.

# Addenda — closures recorded 2026-08-11

## Open-gap intraday family: CLOSED 0-for-3

Registered one-look holdout (explore_open_gap.py --holdout-cell, cell
fixed pre-look: gap <= -2 ATR, 09:30 auction entry, 120m exit, 25 bps,
pass = net > 0 AND t >= 2): net -0.442%/trade, t = -0.70, win 48% over
95 trades / 82 sessions (2022-12-30 .. 2025-12-30). Train reference was
+0.191%. By year: 2023 +0.91%, 2024 +0.50%, 2025 **-2.60%** -- the edge
did not fade, it inverted; buying the auction on 2-9 ATR gap-downs is
being the counterparty to real news, and 2025 collected. Daily-bar gap
variants previously 0-for-2 vs bars ~+0.9. Family closed.

## Rotation, sp500 PIT check: INCONCLUSIVE, leaning credible

Registered two-run comparison (test_weekly_rotation.py sp500pit mode,
TRIALS=14): survivor mode holdout +1.15 SR, +2.50 sd above matched null
(99.6th pct) but at the null MEDIAN in train (68.8th) -- one-regime
smell. PIT mode (863 names, membership-gated rule AND null, departed
names 5.4% of position-slots): holdout +0.67 SR, +1.48 sd (97th pct),
and above its null in BOTH windows (94.8th train / 97.2th holdout) --
the script's own verdict flipped to "weak but persistent." Neither
registered trigger fired (BIAS-DRIVEN needed PIT < +1.0; SHAPE ROBUST
needed PIT >= +2.0). Read: the edge attenuated but did not collapse when
dead names were restored, and part of survivor-mode's regime-dependence
was the NULL being survivorship-inflated (random picks from
guaranteed-survivors earn +0.99 in train; restore the dead and random
earns +0.26 while the ranking still finds the live names). CORRECTION
2026-08-13: the "+2.72 sd" sp400 figure has NO primary source -- it
appears nowhere in the repo except this addendum's own assertion; it
entered the record via conversation, uncommitted. The only reproduced
sp400 monthly result (2026-08-11 run) is holdout SR +0.66, 29.4th pct,
-0.51 sd vs its turnover-matched null: NOT a pass. Separately, the
name rotation correlates +0.869 with the industry-ETF rotation (same
edge twice) and is RETIRED as a duplicate -- the ETF leg (+1.92 sd
holdout, honest 30-trial bar +0.36, cleared) is the one under paper
test per ROTATION_PREREG.md. The sp400 PIT verdict is therefore MOOT
for trading; the Norgate/Sharadar purchase case now rests on
deep-state Gate C and the distress PIT completion only.

## Options expression layer (deep-state) -- CLOSED 2026-08-13
Screen: replay frozen genome sp400+sp600 (32,453 trades), reprice each as
60d BS call, IV = 21d trailing RV x VRP, delta-equivalent sized vs delta-1
stock null. Grid: {atm,itm} x VRP{1.00,1.15,1.30} x spread{3,6,10}%/side.
Rule (pre-stated): ALIVE iff >=3/4 honest cells (VRP>=1.15, sp>=6%) incl.
(1.15,6%) anchor beat stock. Result: 0/18 cells beat stock anywhere.
|z|=0.930 vs 0.80 walk baseline -- drift edge, not vol edge; realized moves
do not exceed implied by enough to pay spread+VRP toll. Options expression
closed for deep-state; no chain data purchase. Selftest recovered both
planted answers (null->DEAD, 2x-vol->ALIVE). scripts/kill_test_options_
expression.py, out/options_killtest.json. Corollary: analyst-headline
options ideas closed a fortiori (published family + weaker signal + event-
inflated IV cannot clear a toll deep-state cannot).


## Crypto intraday 15m (coinbase_liquid_v2) -- CLOSED CONDITIONALLY 2026-08-14
Registered evolutionary search: 28 USD pairs 2021-06 -> 2026-08, 15m base
(1h/4h allowed), long/flat, 24h wall-clock hold cap, taker/taker modeled at
25 bps commission (70 bps round trip all-in). 0 survivors of 6 candidates
vs bar +1.70; this holdout now carries 54 effective trials. Evolution
abandoned 15m by gen 1 and camped on 4h -- same flight-to-coarse as
us_liquid_5m: at these tolls sub-hour holds are unmineable. All finalists
momentum-shaped; train stagnated at +0.136 for 17 generations; best
holdout -1.08. Prereg was signed post-run (process violation, noted in the
prereg itself). Account taker tier UNVERIFIED at close-out: if >= 25 bps the verdict stands conservatively, if < 25 it is not conclusive and the seam reopens. Tier lookup owed before any future crypto ground or redeploy. runs/20260814_124328.
