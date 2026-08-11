# DISTRESS pre-registration (draft -- unsigned until Blake edits and signs)

## The record so far (all 2026-08-10, all PRE-v0.1.10 single-split draws)

Idea family: buy 1d names in a high-vol regime (atr_pct 252d percentile
> 0.9) trading BELOW their 200d mean; exit close > EMA26 or 40 days.
5 variants tried; 14 holdout peeks across 10 universes in one evening.

Where real luck bars existed (US), the family lost:
  sp400 base   +0.82 vs bar +0.87   FAIL
  sp600 base   +0.45 vs bar +0.53   FAIL
  sp600 stab   +0.35 vs bar +0.60   FAIL
  sp600 volconf+0.51 vs bar +0.65   FAIL
  sp600 deep   +0.92 vs bar +0.69   bootstrap 0.00 -> FAIL; rerun +0.31 vs
                                    +0.73 (old fetch-order split bug)
Where bars read 0.00 (fresh fingerprints, effective trials = 1), it
"passed": uk de jp kr tw au in. That geography pattern is NOT the idea
working abroad -- it is the location of unseeded ledgers. The bar was
absent, not beaten.

Extra survivorship warning: this rule buys BELOW-trend high-vol names from
current-membership lists. Delisted names are exactly the ones it would
have bought. Positive results on current membership are close to
uninformative; negative ones count.

## Frozen spec

genomes/distress_base.json  (sha256 b62a9753323f1e5e)
== the base rule only. The four mined variants (stabilized, volconf, deep,
deep_short) are CLOSED -- they exist as record in genomes/, not as
candidates. No new variants during evaluation.

## Test sequence (stop at first FAIL)

GATE A -- home ground, honest split
  python3.10 scripts/validate_genome.py --manifest manifests/us_sp600.yaml \
      --genome genomes/distress_base.json --splits 5
  PASS: median-across-splits SR > bar, and > bar on >= 4/5 splits.
  FAIL -> the idea is DEAD. Do not proceed to Gate B. "Maybe it only
  works abroad" is not a mechanism, it is where the unseeded ledgers were.

GATE B -- two pre-named foreign grounds, debt charged first
  python3.10 scripts/seed_ledger_distress.py --manifest manifests/jp_nikkei.yaml
  python3.10 scripts/seed_ledger_distress.py --manifest manifests/uk_ftse250.yaml
  then validate_genome --splits 5 on each.
  PASS: on BOTH: median SR > seeded bar; breadth p < 0.01; bootstrap gate
  passes on mechanism-scaled blocks. jp and uk are named NOW, before
  results; no third country if one fails.

GATE C -- point-in-time membership (only after A and B)
  Re-test on one universe with real historical constituents including
  delisted names (paid data: Norgate / Sharadar). This gate cannot be
  waived for this rule shape; it is the gate the rule is most likely to
  fail. No forward test, no deployment talk, before C.

## Ledger accounting

Gate A charges 5 peeks to sp600 (already at 9+). Gate B charges the
family debt (5 candidates, 14 peeks) plus 5 peeks per universe. That is
the cost of the evening of variant mining; it does not go away by
switching countries.

Signed: ____________  date: __________
(edit the thresholds if you disagree with them -- then sign. Unsigned,
this file is a proposal, not a registration.)

## RESULT (2026-08-11) -- CLOSED AT GATE A

Run 20260811_004913_idea, base genome sha b62a9753323f1e5e, us_sp600,
five splits: SR median +0.41, range [+0.40, +0.46], beat bar on 0/5
(bar +0.76). Stop-at-first-fail: the distress idea is DEAD. Gates B and
C never ran and must not run. Any future revival of this idea family
starts a NEW prereg and inherits this ledger debt.
