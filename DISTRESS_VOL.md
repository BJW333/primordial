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

## 5. Status

**A real, replicated, thin market phenomenon — not yet a deployable edge.**
Four independent markets passed every gate; the effect's sign is positive on
all ten grounds tested. What it lacks versus deep-state: consistent bootstrap
survival, a point-in-time survivorship check, and any forward evidence.

**Next, in order:** (a) point-in-time re-test on delisted-inclusive data,
(b) if that holds, a LEAN implementation and a pre-registered paper forward
test with its own bands, (c) trading the passing markets together — four
thin uncorrelated streams beat one, and that is the honest way to raise
realized Sharpe here, not a fifth filter.
