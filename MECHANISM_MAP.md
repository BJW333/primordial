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

## Monthlow dip family -- CLOSED 2026-09-01
MONTHLOW_PREREG.md. Fresh 21d closing low AND >= 10% below 21d high, buy
next open, +5% take-profit (target_pct gene, added 2026-09-01), 20-bar cap,
no stop. us_sp600, 5 hash-keyed splits each.
| genome | median SR | range | bar | splits | trades/split | run |
|---|---|---|---|---|---|---|
| monthlow_dip | +0.19 | [+0.13, +0.26] | +0.87 | 0/5 | ~5,100 | 20260901_130125 |
| monthlow_dip_healthy (+SMA200) | +0.28 | [+0.21, +0.45] | +0.94 | 0/5 | ~1,300 | 20260901_142317 |
Neither pass. Healthy variant beat base (+0.09) but not the bar, so NOT
INTERESTING per rule. Read: fourth dip shape dead on sp600; health gate
again moves SR in the right direction; capped upside + 20-bar hold prints a
stable +0.2 vs deep-state's +0.59 on the same ground. Confirms "the premium
requires long holds" (DISTRESS_VOL sec 3). sp600 holdout now 24 trials.
Prereg signed late (split 1 started), recorded in the prereg. Family closed.

## Monthlow dip family, sp500 ground (Amendment 1) -- CLOSED 2026-09-01
| genome | median SR | range | bar | splits | trades/split | run |
|---|---|---|---|---|---|---|
| monthlow_dip | +0.34 | [+0.28, +0.36] | +1.19 | 0/5 | ~3,500 | 20260901_145616 |
| monthlow_dip_healthy | +0.30 | [+0.24, +0.40] | +1.20 | 0/5 | ~930 | 20260901_170907 |
Note vs pre-run prediction (base <= +0.15): base printed HIGHER on sp500
than sp600 (+0.34 vs +0.19), and the SMA200 gate added nothing here (it
added +0.09 on sp600). Read: the gate only pays where below-trend junk
exists to exclude; on large caps it is near-redundant. Raw dip premium
~+0.3 exists on both grounds but clears no honest bar anywhere. sp500
holdout now 75 trials. Family closed on all grounds; no further looks.

## sp500 deletion flow -- frozen cell FAILED 2026-09-01 (0 trials)
explore_index_deletion.py. 385 deletions 2009-2026, 119 scorable (yfinance
cannot fetch delisted paths -- 266 events missing, all numbers biased UP;
distress cell is where the bias binds hardest). Train: distress (n=26)
T->+60 SPY-adj +9.3% [t +0.9] -- FAILS frozen t>=2. No tail look taken.
Shape matches mechanism (sag into T, T-1 +2.6% = forced sell front-run,
recovery after) in both distress and non-distress blocks, but unscorable
at this coverage. Ground closed. Pre-committed next: sp600 deletion study
on Norgate PIT with delisted histories -- its own frozen study, not a
retry of this one.

## Crypto funding-extreme absorption -- screen DEAD 2026-09-01 (0 trials)
explore_crypto_funding.py, Coinbase INTX funding 2023-05->2026, BTC+ETH,
maker 46bps. Primary (fund3<=p10 AND ret3<0, +5d): -0.12% [t -0.2] n=84
on train -- FAILS, no tail look. Funding deciles show zero forward-return
gradient at daily hold; the liquidation-aftermath story does not survive
its own free proxy. Mirror (short) -2.07% [t -3.5]: short-side asymmetry
confirmed again. Funding archive (out/funding_*.csv) retained + growing.
Kraken venue's +1.40 [t +1.2] on 1y was noise, superseded. Seam closed at
daily granularity; anything faster is the archived-data class (Kalshi).

## EDGAR secondaries CORRECTION 2026-09-01: date-parse bug
All prior EDGAR numbers (primary +0.00, run-up +1.07 [t +8.3], distress
+5.08 [t +1.8]) were computed on MONTH-precision dates (form.idx offset
bug, assistant error) -- invalid in both directions, prior entry void.
Refetched day-precision, frozen study re-run as its FIRST valid look.

## EDGAR distress-issuer raise -- DEAD 2026-09-01 (0 trials)
Corrected-date study. Primary (all secondaries) legitimately dead: +0.19
[t +1.1] n=3,232 train, no tail look. Distress-issuer cell (train +3.12
[t +2.0] n=132, seen twice incl. bug era, so tail-ONLY design): tail
+0.09 [t +0.1] n=47. DEAD. Post-2021 distress raises are not absorbed
profitably. EDGAR event pipeline (5,110 day-precision 424B events,
self-updating via refetch) retained as infrastructure for the analyst
leg. Seam closed.

## Multi-horizon sign vote (mh_vote) -- family DEAD 2026-09-09 (15 trials)
MH_VOTE_PREREG.md, coinbase_daily_30 (coinbase_liquid_v2, 2022-01->
2026-06, spot taker, 1d), validate_genome --splits 5, runs 20260909_*.
Signal: four DISJOINT segment sign votes (bars 0-7/7-14/14-30/30-60,
vol-scaled dead zone), score in [-4,+4]; Donchian 20d cross as entry.

- base_donchian20: median SR -0.84 [-1.05,-0.36], 57-66 tr/split, 0/5.
  Dead as pre-stated. Baseline level set.
- vote_alone: median SR -2.47 [-5.03,-1.58], 37-46 tr/split, PSR <=
  0.13 everywhere, 0/5. WORSE than bare Donchian: entering after 2+
  weeks of unanimous uptrend bought tops. Direct evidence AGAINST the
  premise -- high trend-consistency scores predicted reversal, not
  continuation, on this ground.
- gate2 (the idea): 0/5, trades collapsed ~60 -> 12-20 per split with
  no improvement in survivors (pooled PSR 0.04-0.21). The vote
  decimated; it did not select. Printed median -923.53 is a DISPLAY
  ARTIFACT (all-stop-out cell, std ~ 0.003; fixed by the v0.1.8 clip),
  and one split's +0.00 was zero eligible cells, not breakeven. Both
  numbers' VERDICTS stand via pooled PSR, which never had the bug.
- gate4: run 20260909_161032. THREE trades total across 5 splits
  (1/0/0/1/1); every split zero eligible cells, printed +0.00 (the
  v0.1.8 artifact, live specimen). Unanimous +4 at a fresh 20d-high
  cross almost never co-occurs on this ground. Trade-starved as
  pre-stated, and moot regardless -- the sizing ladder was conditional
  on gate2 being INTERESTING. Trials charged: all four runs (ledger
  shows 18 effective on this holdout incl. synthetic-era smoke).

Family CLOSED on this ground per the decision rule -- both parents dead,
combo dead, nothing to size. mh_vote atom stays in the crypto adapter as
vocabulary (it is a computed feature, not doctrine). NOT licensed by this
close: sign-flipping the vote (fading high scores) -- that is a NEW
family owing its own prereg, and it starts skeptical because this
holdout leaked into its motivation. Windows/dead-zone re-tuning on this
ground: same prohibition, per the STOP clause.

## Breakout family, sp400 ground -- DEAD 2026-08-14 (4 trials)

Prereg BREAKOUT_PREREG.md (signed late but pre-result, 41aa9f4; violation
recorded there). sp400, --splits 5, all four 0/5:

| genome | run | 5-split median SR | holdout trades/split |
|---|---|---|---|
| breakout_donchian20 (control) | 20260814_133352 | -0.07 | ~2,650 |
| breakout_donchian55 (control) | 20260814_140304 | -0.14 | ~1,440 |
| breakout_compress | 20260814_143209 | -0.24 | ~1,140 |
| breakout_rvolconf | 20260814_163614 | -0.16 | ~1,050 |

Family CLOSED on sp400 per the decision rule; neither mechanism variant
beat the decayed baseline. rvolconf dead -> the analyst-price-target leg
gated on it is CLOSED with it (no data spend). sp600 half of the prereg
(4 candidates) was never run: recorded here as ABANDONED, not run-and-
hidden; running it now would be a fresh look owing a note in the prereg.

## Seconds-scale dip timing (BTC 1s) -- DEAD BY MECHANISM 2026-09-18 (0 trials)

scripts/dip_hazard.py, 48h / 989k Coinbase BTC-USD ticks, memoryless and
rising-hazard planted controls both recovered. 1s hazard P(dip ends | it
lasted k) never rises: 0.68 -> ~0.50 (flat=break), ~0.25-0.27 (flat=hold).
No exhaustion clock. "Wait the train-window average dip, then buy":
-0.09 bps gross @5s (break), -0.08 bps (hold); no k with n > 30 clears
1 bps gross vs ~50 bps taker RT. Dead by mechanism, not just costs; the
Kalshi wrapper idea closes with it. Off-repo screen, ledger not charged.

## Dip-count exhaustion (crypto minute-scale) -- REAL BUT UNTRADEABLE 2026-09-18 (0 trials)

scripts/dip_count_hazard.py (chain = consecutive lower-low dips; planted
controls recovered). Flat at <= 15s. At 1m-15m a weak, consistent
gradient: P(rebound | N) climbs ~4-5pp over N=1..5 on BTC (60s .486 ->
.535) and replicates on ETH/SOL/XRP/DOGE (5/5 assets). Best cells pay
3-16 bps vs ~70 bps taker RT (~15 maker): an order of magnitude short.
Hourly: present but low power. Daily: no read (sample-starved; raw tables
were pure bull-market drift). N=4 kink recurs across views -> covered by
runlen4 (see next entry). Off-repo screens, ledger not charged.

## Runlen crypto daily family -- DEAD 2026-09-18 (4 trials) -- PROCESS VIOLATION

**Violation, recorded not backfilled:** prereg/RUNLEN_PREREG.md was never
signed and was not committed before the four runs below; genomes and
prereg sat untracked until this entry. Results stand as observed; the
family gets no benefit of the doubt.

coinbase_daily, --splits 5, all 0/5:

| genome | run | 5-split median SR | holdout trades/split |
|---|---|---|---|
| runlen2 (2 down closes) | 20260918_014452 | -0.52 | ~560 |
| runlen3 | 20260918_020535 | -0.29 | ~345 |
| runlen4 | 20260918_022627 | -0.18 | ~190 |
| runlen3_deep (+3-day move bottom 10%) | 20260918_032726 | -0.59 | ~75 |

Dose-response: SR improves with k (2 -> 3 -> 4) but never crosses zero;
runlen3_deep worse than runlen3, so the depth filter adds nothing. The
minute-scale exhaustion gradient does not survive to a tradeable daily
edge. Family CLOSED; the dip-count seam is fully closed with it.

## Single-day dip-buying, US + intl equities -- SEAM CLOSED 2026-10-25 (off-ledger screens)

Idea: S&P name drops > X% in a day -> buy -> ride the rebound. Ten
screens, all off-ledger event studies with pre-stated rules in the script
headers; outcome measured vs SPY or vs the same-day index average.

| test | commit | result |
|---|---|---|
| dip5: fixed >5%, sp500 survivor vs PIT | (untracked) | survivor H20 t +6.3 -> PIT t -1.5, 2018+ t -2.9. VOID (41% dead-name coverage); direction: survivorship was the edge |
| dipz: drop >= 3x lagged ATR, sp400 kill-only | 165f4b4 | t -0.29, both halves < 0. CLOSED |
| dipgap: gap vs intraday drop split | dea8104 | spread WRONG sign on sp600/400/500. CLOSED. **Prereg unsigned at run (violation, recorded)** |
| dipband: 2%-X bands, close entry | eeec2fc | passes on sp400/600 only, sp500 PIT negative everywhere = survivorship fingerprint |
| dipbase: bands vs same-day index avg | 706e441 | dose-response rises every step on sp400/600, not on sp500 PIT; indistinguishable from asymmetric survivorship. Open only for Norgate PIT |
| dip_features: 17 features + GBM per index | ee79943 | NO SIGNAL (sp500 PIT model t +1.5). sp400/600 model t +6..+8 = log_dollarvol LOOK-AHEAD via current membership (small-then names that grew into the index) |
| market_trend_dips: WEAK (index < 200dma) vs STRONG, 8 intl | be599a3 | passed barely: 6/8, t +2.25 |
| market_trend_fresh: same spec, 6 fresh markets | 3a73f2b | DOES NOT REPLICATE: 3/6, -13 bps t -0.92. be599a3 = luck |
| dip_combo_search: 19,635 combos x H10/20 | dce4200 | none pass; discovery max t ~4-5 = null max |
| dip_combo_classic: +RSI/BB/MACD/stoch/MAs, 64,875 combos (look #2) | 7eeb903 | none pass |

Durable lessons:
- Current-membership lists manufacture dip-rebound edges. Any feature
  that proxies "will still be in the index" (small size, low liquidity,
  deep drawdown that later recovered) wins on survivor lists and becomes
  a severe falling knife on PIT data (big-drop & low-dollarvol combos:
  -350..-490 bps per 20d, t -7 on sp500 PIT 2018+).
- Same-day index baselines cancel uniform survivor lift, not the
  asymmetric part (big droppers delist most).
- Price/volume/classic-TA separators exhausted. Remaining levers: event
  data (earnings dates, news, downgrades), Norgate PIT sp400/sp600,
  deep_state forward paper (the multi-day dip edge already in the book).

## Insider-buys-the-dip (SEC Form 4) -- FAIL 2026-10-25 (off-ledger screen)

scripts/fetch_insider_buys.py + scripts/insider_dip_test.py, commit 647e052
(spec committed before the run). 191,955 open-market purchases (code P,
>= $25k, officers/directors), 11,888 tickers, 2006-2026; median filing lag
2 days. Dip = -10% over 10 trading days; INSIDER = purchase filed in the
prior 5 days; CONTROL = same dip, no insider filing in 20 days; 60-day hold,
outcome vs equal-weight index average.

| ground | MAIN spread 60d (all) | 2010-17 | 2018-26 |
|---|---|---|---|
| sp500 PIT (decides) | +74 bps t +0.71 | +253 t +1.42 | -74 t -0.59 |
| sp400 | (contaminated, see note) | | |
| sp600 | -152 t -1.20 | -385 t -2.30 | +41 t +0.22 |

LEVEL (insider dips vs index, sp500): +57 bps t +0.61. Clusters (2+
insiders) negative on sp500/sp400. Insider-dip vs insider-no-dip: no
difference -> the dip adds nothing. FAIL by the pre-stated rule.
Note: sp400 row (-1271 bps) is a bad Yahoo price; this script lacked the
|fwd| > 200% glitch filter the dip scripts use. Verdict unaffected.
Read: the insider-purchase effect likely lives below S&P size; at S&P
names a post-drop insider buy is largely priced in.

### Day summary 2026-10-25
Twelve tests of "buy the dip, ride the rebound" on S&P 400/500/600 plus 14
non-US markets: fixed %, volatility-scaled, gap vs intraday, drop bands,
bands vs index, 17-feature model, 39k and 130k indicator-combo searches,
market-regime (passed 8 markets, failed 6 fresh ones), insider buying.
All failed. Recurring killers: survivorship in current-membership lists
and multiple-testing luck. Remaining levers, if ever reopened: 8-K Item
2.02 earnings dates (no-news residual dip, innocent-bystander dip),
Norgate PIT sp400/sp600. deep_state forward paper continues.

## Tax-loss C2 placebo + cost gates -- FAIL 2026-10-25

scripts/validate_taxloss.py, prereg/TAXLOSS_PREREG.md (commit fb1200a or
earlier). Cell unchanged from check_c2_cohorts.py (sp600 current list).
GATE 1 placebo FAILS as frozen: December ranks #2 of 12 by cohort mean
(+7.47%, t +3.43, 14/16) behind July (+13.98%, t +1.21, one outlier year).
December is #1 by t, but re-ranking by t after the look is not allowed.
GATE 2 cost passes (+6.75% at 3x). Context: sp500 membership-gated Dec
+1.35% t +0.34, rank #6/12; sp400 Dec rank #2. Every sp600 month averages
+2.5% = survivor lift. Read: survivorship + concentration; C2 CLOSED.
Only a Norgate PIT sp600 run could reopen it, as a new look.

## Spinoff day-20 entry -- VOID (n) 2026-10-25, strongest signal of the day

scripts/fetch_spinoffs.py + scripts/spinoff_test.py (EDGAR 10-12B 2006+,
CIK->ticker via Form 4 data). 521 registrants, 395 with symbols, Yahoo
prices for 195, 142 usable (< 150 pre-stated minimum) -> VOID.
Buy close day 20, hold 120d vs IWM, 40 bps: +14.3% t +4.15; pre-2016
+17.2% t +2.97 (n 51); 2016+ +12.2% t +2.88 (n 91). Forced-selling window
days 0-20: -0.0% vs IWM -- the hypothesised dip did NOT appear; signal
looks like the published long-run spinoff premium, not a dip rebound.
Caveats: ~50% price coverage (acquired + bankrupt spinoffs missing), and
the >= 140-bar requirement drops fast deaths. NEXT: Norgate delisted
prices for the missing ~190 spinoffs, rerun unchanged.

## Spinoff rerun with Alpaca fallback -- PASS as frozen, weak on clean data 2026-10-25

coverage_probe.py: Alpaca returns full 2016+ history for 13/13 known
deaths/buyouts (SIVB, FRC, SBNY, BBBY, ATVI, TWTR, CELG, XLNX, PXD, HES,
WBA, JNPR, X) -> free survivorship repair for 2016+. Caveat: recycled
symbols (BBBY runs to 2026 = a different company); scripts must cut at
long gaps before relying on Alpaca for dead names.
spinoff_test.py --fallback alpaca (commit 2c3bd69, spec unchanged): +50
spinoffs recovered, n 181. ALL +8.4% t +2.69 -> PASS by the frozen rule.
BUT 2016+ (coverage ~95%) fell from +12.2% t +2.88 to +3.3% t +0.94 once
dead spinoffs returned; the pass rests on 2006-15 (coverage ~35%).
Forced-selling window -0.4% (no dip). Read: survivorship inflated it; on
clean data the spinoff premium is small/insignificant. Not for capital;
forward paper optional.

## Factor portfolio (S&P 500 top-20 monthly, survivorship-free) -- FAIL 2026-10-25

scripts/factor_portfolio.py, commit 2d2d826. S&P 500 PIT membership +
Alpaca prices incl. delisted (730 of 863 names), 2017-02 -> 2026-05,
112 rebalances, 10 bps/side, recycled-ticker gap guard.
PRIMARY MOM+LOWVOL: +7.5%/yr, Sharpe 0.61, -4.4%/yr vs equal-weight
(t -1.07), halves -3.8% / -5.0% -> FAIL. Context: MOM alone +18.2%/yr,
Sharpe 0.82, +7.2%/yr vs EW t +1.31 (not promoted -- context row);
LOWVOL -3.6%/yr vs EW; market switch worse. EW +11.6%, SPY +15.5%
(Sharpe 1.00). Read: low-vol fought growth leadership; SPY beat EW by
~3%/yr this decade, a headwind for any pick-from-the-list strategy.
Monthly returns saved (data/factor_portfolio_returns.csv) for an
industry_rotation dup check. Next candidate: quality from SEC financial
statement data sets.

## Turbo momentum (big-cap + vol-managed) -- primary FAIL; MOM_BIG strongest of the day 2026-10-25

scripts/turbo_portfolio.py, commit 8b99af4. Same survivorship-free S&P
500 machine; BIG = top 100 by 63d dollar volume; VM = scale to 15% target
vol (cap 1.5x, 4%/yr leverage cost). PRIMARY MOM_BIG_VM: +13.7%/yr,
Sharpe 0.95, max DD -13.7%, -1.8%/yr vs SPY t -0.54 -> FAIL. VM averaged
0.69x exposure (15% target << momentum's ~24% natural vol): cut drawdown,
gave up return -- target mis-set, not a dead lever.
Context: MOM_BIG +24.3%/yr, Sharpe 1.00 (= SPY), max DD -21.0% (SPY
-22.9%), +9.0%/yr vs SPY t +1.71 -- best risk/return of the day, not
significant, context row on twice-used data. Only clean test left is
forward: Alpaca paper bot for MOM_BIG (+ optional pre-registered VM with
24% target) proposed.

## Turbo momentum out-of-sample 2010-2016 -- does NOT replicate 2026-10-25

scripts/turbo_oos.py, commit 7fe239d. Frozen MOM_BIG and MOM_BIG_VM24
(24% target set before the run) on 2010-01 -> 2016-12 (no prior portfolio
test touched these years). S&P 500 PIT membership + Yahoo prices (member
coverage 338/497 in 2010 rising to 408/506 in 2016; buys only the 100
largest, where missing names matter least).
MOM_BIG +15.6%/yr Sharpe 0.94 (+2.3%/yr vs SPY t +0.77); MOM_BIG_VM24
+16.7%/yr Sharpe 0.92 (+3.7%/yr t +0.93); SPY +13.6%/yr Sharpe 1.08.
Both beat SPY on return, both fail the Sharpe >= SPY rule -> neither
replicates. Read: giant-stock momentum's extra return is extra risk
(higher-beta market), not an edge; 2017-2026 was the mega-cap decade.
Implication: leverage the best-Sharpe asset (SPY) directly, with a trend
switch -- testable on 1993+ SPY history.

## Residual momentum FAIL + last-5-years rescore 2026-10-25

scripts/residual_momentum.py: RESMOM_BIG last 5y (2021-06 -> 2026-05)
+5.2%/yr Sharpe 0.38 vs SPY +14.0%/0.93 -> FAIL (pre-stated).
scripts/last5.py rescored all saved portfolios on the last 5y vs SPY
leveraged to equal vol (4%/yr borrow): MOM +28.8%/yr Sharpe 1.08 beats
same-risk SPY +20.0% (t vs SPY +1.88); MOM_VM +15.4% Sharpe 1.07 DD -12%
beats +13.2%; MOM_BIG +27.7% beats +21.5%. Quality, lowvol, combos, resmom
all lose. Caveat: t < 2, AI mega-cap era; 2010-16 OOS lost on Sharpe.
Next: forward Alpaca paper bot, MOM + MOM_VM.

## Blow-up scan (accumulation / squeeze / laggard) -- all FAIL 2026-10-25

scripts/blowup_scan.py, commit 5fd7d8d. S&P 500 PIT, last 5y deciding.
Target: up >= 50% in 6 months (base rate 2.4% last 5y). Lift / t (NW):
ACCUM 0.59x -1.61 (+3.9%/yr), SQUEEZE 1.08x +0.14 (~2 picks/mo),
LAGGARD 1.21x +0.57 (-5.2%/yr) -> all FAIL. Context: MOM 3.61x t +3.17,
+26.6%/yr vs same-risk SPY +19.7% -- momentum is the best blow-up finder.
Limit: large caps rarely blow up; next = broad US universe incl. delisted.

## Exit rules for all-US steady momentum -- FAIL, no stop helps 2026-10-25

scripts/exit_rules.py. Rule picked on 2017-21 by Sharpe: NONE (hold the
month). Last 5y: NONE +23.0%/yr Sharpe 0.79 vs same-risk SPY +22.3%, SPY
Sharpe 0.91 -> FAIL (Sharpe). Every stop lowered return in both periods:
VOLSTOP (3x own monthly vol) +21.0%, TS35 +18.0%, TS20 +16.2%, BUFFER
(keep while top 60) +21.8% with ~half the trades, MA50 +3.3% (31 trades
per month). Read: these names rebound after dips; selling locks in losses.
The problem is the picks (momentum = levered beta), not the exits.
Next: earnings surprise (10-Q), alone and with momentum.

## Earnings surprise (SUE) at 10-Q filing date -- FAIL 2026-10-25

scripts/fetch_earnings.py + earnings_test.py, commit d4c1841. SEC
companyfacts, first-filed quarterly net income (no restatements), seasonal
surprise / 8q std. 108,964 surprises matched to prices (4,194 stocks).
Event = 10-Q/10-K FILING date (2-4 weeks after the press release).
Drift (top-minus-bottom decile, 3m after filing): 2017-21 +0.05% t +0.04;
last 5y +1.71% t +1.30 -> no drift left at filing date.
Last 5y: SUE_MOM (primary) +12.3%/yr Sharpe 0.60 vs same-risk SPY +18.3%
-> FAIL; SUE +2.2%. Next: re-time to 8-K item 2.02 (announcement) date.

## Momentum add-ons (regime/FIP/cluster/volume/invvol) -- FAIL 2026-10-25

scripts/momentum_plus.py, commit 93d643c. Steady momentum all-US base.
Selected on 2017-21 by Sharpe: REGIME (only add-on > BASE 0.98, at 1.00).
Last 5y: REGIME +17.6%/yr Sharpe 0.70 vs same-risk SPY +20.5% -> FAIL.
Non-selected, last 5y: INVVOL +29.1% Sharpe 1.02 DD -22.6% (vs same-risk
SPY +20.5%); FIP +24.4% Sharpe 1.00 DD -19.0% (vs +18.6%) -- both beat
SPY but both lost to BASE in 2017-21 -> forward-paper candidates only.
VOLUME worst (+9.5%), drags FIP_VOL to +16.7%. HEDGED skill check +4.4%/yr
t +0.41, pick beta 1.32 -> return is mostly levered market exposure.

## Insider clusters + same-month seasonality -- both FAIL 2026-10-25

scripts/insider_season.py, commit d782555. Both periods required.
INSIDER (3+ officers/directors buying >= $25k within 30 days, signal =
3rd filing date; >= $5, >= $5M/day; 30 bps): 4,749 clusters since 2015,
2,203 matched. Drift vs SPY: 2017-21 3m +3.1% (median +0.2%) t +0.48;
last 5y 3m +0.2% t +0.32, 6m -1.5% (median -7.8%) t -1.39. Portfolio
+6.1%/yr (2017-21), -5.4%/yr (last 5y) -> FAIL. Effect traded away.
SEASON (Heston-Sadka same-calendar-month, min 3 yrs, from 2019; >= $10,
>= $20M/day; 20 bps; ~89% monthly turnover): 2019-21 +34.6%/yr Sharpe
0.88 (SPY 1.22); last 5y -20.0%/yr, DD -76.9% -> FAIL.

## 13F best new ideas + buybacks -- both FAIL 2026-10-25

scripts/fetch_13f.py, fetch_shares.py, funds_buybacks.py, commit f76c55c.
BESTIDEA (each 20-500-position manager's largest new position, at 13F
filing date; 179,506 ideas, 9,777 managers; ranked by # managers): drift
vs SPY 3m -0.0% (both periods), 6m ~0 -> nothing left after the 45-day
lag. Portfolio last 5y +17.2%/yr Sharpe 0.86 vs same-risk SPY +16.8%,
SPY 0.91; 2017-21 +17.6% vs +18.9% -> FAIL (crowded large caps).
BUYBACK (1-25% 1y share shrink, cover-page counts at filing date):
2017-21 +8.1%/yr DD -45%; last 5y +11.4% vs same-risk SPY +18.0% -> FAIL.

## Crypto pump study v1 -- FADE signal real (untradeable as tested), RIDE invalid (look-ahead) 2026-10-26

scripts/pump_study.py, commit 73423ff. Binance archive, 686 USDT pairs incl.
delisted; pump = hour $vol >= 10x 7d median AND close >= +10%; 1/coin/72h.
Last 5y: 7,499 pumps, 545 coins. Coin minus BTC after detection: 1h -1.7%,
24h -4.3% (median -8.0%, 77% down), 7d -6.1% (median -11.4%). FADE 24h net
0.3%: mean +4.0% median +7.7% t +16.5 (tiny coins +5.6%); 2018-21 mean
+0.3% median +4.1%. Verdict PASS but unrealistic: assumes every tiny coin
shortable, 0.3% cost, BTC hedge, no stop.
RIDE 'PASS' (+3.8%, t +21.7) is INVALID: early 5m signal searched only
inside hours known to close +10% (look-ahead). Redo in pump_study2.py.

## Crypto pump study v2 -- RIDE FAIL, realistic FADE FAIL 2026-10-26

scripts/pump_study2.py, commit b08029a.
RIDE v2 (no look-ahead; 150 random coins, every 5m bar with >= 10x 24h
median volume and >= +5%, 1/coin/6h): 5,101 triggers last 5y. Buy at the
trigger close: +15m -0.06% (median -1.4%), +30m -0.39% (median -2.1%,
65% down), +60m -1.05% -> FAIL; v1 ride pass was look-ahead.
FADE v2 (coins with a Binance USD-M perp that month, short pump-hour close,
24h, +30% stop at stop/open, 1% cost, raw P&L): mean +0.7% median +4.8%
t +2.07; years 2022 +1.0%, 2023 -1.1%, 2024 -0.4%, 2025 +1.2%; 2018-21
-2.4% -> FAIL. Profit lives in non-shortable tiny coins (all-coins +2.2%
at same settings). Stop hit 15%; runaways to +626% in 24h (ALPACA).
Use: pump detector as an entry filter only.

## Commodities: 5 strategies x 20 futures -- SEASON PASS (pending roll check), rest FAIL 2026-10-26

scripts/commodities_test.py, commit 3e39538. Yahoo continuous front futures
(20 mkts, 2000+), CFTC disaggregated COT 2006+, ETF carry estimate. Equal
risk per market, 2 bps. Bar (tightened after noise check, pre-run):
last-5y Sharpe >= 0.5, 2007-21 >= 0.3, improves 50/50 blend with STEADY.
SEASON (sign of same-calendar-month mean, prior yrs >= 5): 2007-21 +2.2%/yr
Sharpe 0.67; last 5y +4.2%/yr Sharpe 1.36, DD -4%, 2022 +6%; corr with
stock momentum +0.01; blend Sharpe 0.79 -> 1.71; ETF long-only 0.81 -> PASS.
TREND last 5y 0.62 but 2007-21 0.16 -> FAIL. CARRY -0.37, COT 0.07,
REPORTS -0.33 -> FAIL. Caveat: continuous-futures roll jumps are
calendar-seasonal -> season_check.py before trusting.

## Commodity SEASON roll check -- DOES NOT SURVIVE (roll artifact) 2026-10-26

scripts/season_check.py, commit 939f566. Roll days detected as futures-vs-
ETF return divergence (>4x typical and >1.5%), ETF return substituted.
Same 11 ETF markets: raw futures 2007-21 Sharpe 0.53 / last 5y 1.14 ->
roll-cleaned 0.22 / 0.51 -> FAIL (bar 0.3 / 0.5). ETF-only long/short
-0.17 / -0.33. Top P&L markets hogs 28%, cattle 13% -- no ETF, uncleanable,
strong contract-month gaps. Read: SEASON pass was mostly Yahoo continuous-
futures roll jumps. Lesson: any calendar/seasonal futures test needs
roll-adjusted (back-adjusted) data.

## Commodity ETFs: ADX trend strength + volume -- all FAIL 2026-10-26

scripts/commodity_adx.py, commit f3fa2eb. 14 commodity ETFs (real prices
and volumes, no roll artifacts), daily, equal risk, 5 bps. Long/short
judged: ADX_TREND (ADX>25, DI direction) 2008-21 Sharpe 0.17 / last 5y
-0.01; ADX_VOL (+ volume surge >= 1.2) 0.23 / 0.44; ADX_RISING 0.17 /
-0.16; VOL_BREAKOUT (50d high/low on 2x volume, hold 20d) 0.46 / 0.23 ->
all FAIL. Context: TREND12 0.40 / 0.41; HOLD -0.11 / 0.91 -- long-only
'wins' last 5y are the commodity bull (inflation 2021-22, gold 2024-26),
not the signals. ADX adds nothing over plain trend.

## ADX + non-price data (CFTC flow/room/OI, EIA inventories) -- FAIL, ADX closed 2026-10-26

scripts/adx_fundamentals.py, commit 87fe71c. 12 commodity ETFs (energy 4
for inventories); CFTC used from Friday release, EIA from release day.
2008-21 / last-5y Sharpe: ADX alone 0.06/0.08; +FLOW 0.05/-0.26; +ROOM
0.13/0.34; +OI 0.06/0.19; +INV (crude/gasoline/natgas vs 5y same-week avg)
0.04/0.38 (2022 +6%). Candidate (best 2008-21) ADX_ROOM -> last 5y 0.34 <
0.5 -> FAIL. Non-price data helps ADX slightly but nothing near an edge;
time in market 10-23%. ADX closed.

## ADX + OBV + CMF weighting diagnostic (S&P 500 PIT) -- no consistent edge 2026-10-26

scripts/adx_obv_variants.py, commit 4f61e5a. Diagnostic, no pass/fail.
Live-price fix: adline_obv.py picked delisted 'zombie' names (stale ADX on
NaN prices) -> its yearly S_ADX_OBV table is superseded by TOP20_EW here.
Avg qualifiers/month: ADX+OBV 97, ADX+CMF 86, all three 80 (of 503).
10y growth vs controls: BASE_SIZE 4.69x; ALL_SIZE 4.62x; ADX_CMF_SIZE
4.66x; TRIPLE_SIZE 5.04x (2017-21 13.2%/0.85 vs ctrl 17.7%/1.01; last 5y
24.4%/1.25 vs 18.6%/0.95). BASE_EW 2.69x; ALL_EW 2.71x; TRIPLE_EW 2.77x;
TOP20_EW 3.59x (beats BASE_EW both periods, ~SPY 3.69x). Filters lag
2019/2021, win 2020/2024 -> momentum fingerprint; smaller DD 2017-21.
CMF ~ OBV. No filter beats its control in both periods on return.

## Attention spikes (Wikipedia) + short volume (FINRA Reg SHO) -- both FAIL 2026-10-27

scripts/fetch_attention_short.py + attention_short_test.py, commit 40e20e0.
S&P 500 PIT. ATTENTION: 744 companies' daily page views 2015+; spike = 7d
avg >= 3x prior 90d median, used next day. 1,183 spikes / 442 stocks.
21d vs SPY: 2017-21 +0.13% t +0.79; last 5y -0.16% t +0.41 -> FAIL (no
effect in large caps). SHORT: FINRA consolidated short interest returned
HTTP 204 for 2016-01 (treated as failure -> bug) -> fallback daily short-
sale volume ratio (20d), 2018-09+: Q1 +1.24%/mo ... Q5 +0.92%; spread
+1.5%/yr t +0.65, halves -0.4% / +3.4% -> FAIL. True short interest
untested (204 = empty window, fix pending).

## Attention follow-ups (same-day/direction, small caps, slow growth) -- all FAIL, consistent NEGATIVE drift 2026-10-27

scripts/attention2_test.py, commit e6786d8. 2,206 companies' page views.
Same day: spike-day |move| S&P 2.78% vs 1.40% (2.0x), SMALL 5.43% vs 1.98%
(2.7x); direction 51% up. Pre-stated 21d rule -> all FAIL. Consistent
sign: attention -> later underperformance. SMALL good-news spikes (next
close) 21d -2.56% t -1.60 / -1.24% t -2.37; 63d -3.44% t -1.75 / -6.24%
t -6.09. SMALL 7d spikes 63d -2.98% / -6.11% t -3.95. Attention growth
spreads negative: S&P -4.4/-3.6 %/yr (t -1.78), SMALL -6.1/-0.4. 63d
horizon is post-hoc -> fresh test on MID caps (non-S&P, above-median
liquidity, untouched) with a pre-stated 63d rule next.

## Attention FADE confirmed on MID caps (first PASS); rebound/bottom signals all FAIL 2026-10-27

scripts/attention_rebound.py, commit e37baae. Good-news attention spikes
(daily views >= 3x 90d median AND stock +2% that day), non-S&P liquid
stocks, vs equal-weight same-size benchmark, next-close entry.
MID (untouched universe, pre-stated 63d rule): 2017-21 -3.19% t -2.82
n 1067; last 5y -3.07% t -3.05 n 1314 -> FADE CONFIRMED (PASS).
Path: MID 126d -8.8%/-4.4%, 252d -22.8%/-11.7%; SMALL 126d -8.3%/-9.2%,
252d -16.7%/-17.3% -- decline persists a full year, no bottom.
REBOUND: 31 combos of A attn-normal, B round-trip, C vol-quiet, D price-
turn, E insider-buy + F time (median low day SMALL 81, MID 72). Selected
on 2017-21: SMALL A+C+D -> last 5y -0.36% t +0.26 FAIL; MID A+B+E ->
-4.21% t -1.90 FAIL. Use: avoid-filter / hedged short book (next: cost +
borrow-aware portfolio test).

## Momentum HYPE-only vs QUIET (post-hoc diagnostic) -- regime-flipping, no edge; attention line closed 2026-10-27

scripts/momentum_hype.py. Top-20 steady momentum among: all / hyped (good-
news spike in 2y) / quiet (Wikipedia page, no spike).
2017-21: STEADY 30.4% sh 0.98 | HYPE 19.1% 0.61 | QUIET 34.3% 1.27.
Last 5y: STEADY 23.0% 0.79 | HYPE 27.1% 0.96 | QUIET 15.4% 0.79.
Ranking flips between periods -> regime, not edge. Attention summary: fade
real (20-30% underperf over 2y, small/mid); short book fails net of 3-6%
borrow; no bottom; avoid filter hurts momentum. Line closed to avoid
forking paths. Open: forward paper short book on easy-to-borrow names.

## FADE TRADE FOUND: liquid-MID weak-trend short + 50% stop -- PASS (stop post-hoc) 2026-10-27

scripts/fade_predict.py: day-21 flags on good-news spikes. Selected on
2017-21 (15 rules): WEAK_TREND (12m ret < same-size avg) -16.0% t -4.32;
last 5y -15.3% t -8.26, vs unflagged -8.4% t -2.40 -> PASS. DILUTION
-23% recent but ~0 2017-21 -> FAIL. Insider sell ~ base rate.
scripts/fade_book2.py: real book, no trades dropped, liquid MID +
WEAK_TREND, 581 trades, short wins 65%, median -18%/-23%, but squeezes
(+1198%, +639%) -> +0.4% / +9.9% t 1.96 -> FAIL.
scripts/fade_stop.py (stop idea post-hoc): cover at close >= +50% over
entry -> +11.8% sh 0.95 dd -13% / +16.7% sh 1.39 t 2.73 -> PASS. Robust:
25% stop 15.2%/20.9%; 100% stop 10.0%/15.3%; 3% borrow still positive.
Next: verify worst trades aren't data errors; frozen-rule Alpaca paper bot.
