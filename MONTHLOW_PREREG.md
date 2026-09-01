# MONTHLOW `_idea` PRE-REGISTRATION -- sign BEFORE any command runs

## Question
Buy a US small-cap that closes at a fresh 21-day low while >= 10% below its
21-day high; take profit at +5% from fill, else flat after 20 bars, no stop.
Does this survive the gauntlet, and does gating to healthy names (above the
200-day) add anything?

## Ground
manifests/us_sp600.yaml ONLY. us_sp400 is excluded by the registered stop in
MR_PREREG.md (batch #2 returned 0 survivors; daily MR/momentum seam on
midcap US is CLOSED, and the sp400 bar now sits at +1.32 after 145 trials).
Current-membership universe: any pass owes a PIT re-test before deployment
talk. Runner: scripts/validate_genome.py --splits 5. Ledger charge per
genome: 1 candidate + up to 5 holdout peeks. Planned total: 2 candidates.
No other looks. Runs AFTER the already-registered sp600 breakout batch
(BREAKOUT_PREREG.md), which is also this idea's momentum leg -- no new
momentum genome is added here.

## Prior on record (read first)
Every dip shape on sp600 is dead on an honest bar: distress 0/5 splits,
one-day crash +0.08 vs +0.53, cross-sectional dip +0.23 vs +0.26. The only
dip family that has ever cleared every gate is deep-state, and it is
ABOVE-trend absorption with a 40-bar hold; cutting hold 40 -> 15 cost two
thirds of the distress edge. This family is a fresh-low entry with a
truncated 20-bar hold and a capped upside -- the record predicts dead.

## Frozen family (genomes/, exactly these two, no post-look edits)
1. monthlow_dip -- close <= 0.90 x prior 21d high AND close <= prior 21d
   closing low; stop 99A (none), target_pct 0.05, max hold 20.
2. monthlow_dip_healthy -- (1) AND close > SMA200: the deep-state health
   gate applied to a fresh-low entry.

Sizing (25%/trade, 4 slots) is NOT a gauntlet input: the judge scores
per-name edge on r-multiples. A slot-limited portfolio layer is built only
if something passes, never before.

## Decision rule (pre-stated)
- A genome PASSES only if the runner prints survival on 5/5 splits. Mixed
  splits are not a pass.
- monthlow_dip_healthy is INTERESTING only if it passes AND its 5-split
  median SR exceeds monthlow_dip's median. Beating null but not the base =
  nothing.
- Anything else: family CLOSED. If the healthy variant alone passes, it is a
  deep-state re-hit (MR_PREREG registered rule) unless its holdout trades
  correlate < 0.5 with deepstate_frozen on the same split -- record either
  way, do not iterate.
- STOP: no threshold edits (0.90, 21, 0.05, 20 are frozen), no added
  variants, no re-runs with new seeds after any holdout look. Results logged
  to MECHANISM_MAP.md win or lose.

## Order
1. Sign below. 2. git commit. 3. Run the 2 commands. 4. Log + commit.

Signed: __________
