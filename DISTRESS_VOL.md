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

## 5. Status — CLOSED (superseded by section 6)

**Dead. Zero honest passes on ten grounds; Gate A failed 0/5.** The verdict
originally written here ("a real, replicated, thin market phenomenon") did
not survive the ledger audit or the pre-registered re-test below.

## 6. Correction and Gate A result (2026-08-11)

Sections 1-4 stand as the record of what was run. Their conclusions do not.

**"Four clean passes" -> zero.** All four ran at a 0.00 luck bar: their
fingerprints carried one trial each, and the bar formula returns 0 for
n < 2. The bar gate was ABSENT, not cleared. Charging any ground the
family's actual debt (5 variants, 14 peeks, via seed_ledger_distress.py)
puts the minimum bar at +0.89 — above the best foreign SR (+0.76, Japan).
The geography pattern ("works abroad, fails at home") was a map of
unseeded ledgers. The US grounds were the only ones with a real bar AND
the only direction survivorship bias permits trusting; both said no.
Every section-1 number was also a single draw of the pre-v0.1.10
fetch-order-unstable splitter (+0.92 vs +0.31 on one identical run pair).

**Gate A (pre-registered, DISTRESS_PREREG.md), run 20260811_004913:**
base genome, us_sp600, five deterministic hash-keyed splits:

    SR +0.40 +0.41 +0.46 +0.42 +0.40  | median +0.41
    range 0.06 wide | bar +0.76 | beat bar on 0/5

Per the prereg's stop-at-first-fail rule: the idea is DEAD. No Gate B,
no third country, no point-in-time spend, no LEAN port, no multi-market
sizing.

**What was actually learned (and is worth keeping):**
- The sign is stable: +0.40 to +0.46 on every split, every ground ever
  tested positive. There is a real tendency here — it is just far below
  what 14 trials of selection dredge up by luck, before survivorship,
  which for this below-trend rule shape is maximal and un-nulled.
- The v0.1.10 splitter's 0.06-wide range against the old +0.92/+0.31
  spread confirms the split fix: that spread was fetch noise, not market
  structure.
- Long-hold liquidity provision remains proven only in the form that
  passed everything: deep-state (above-trend). The below-trend mirror
  does not clear an honest bar. That asymmetry IS the mechanism map
  entry: the premium is for absorbing sags in healthy names, not chaos
  in broken ones — at least not at a size this framework can certify.

The PASSES JAPAN / PASSES KOREA / 6-passes commit messages in this
file's history are superseded by this section.
