# BREAKOUT `_idea` PRE-REGISTRATION -- sign BEFORE any command runs

## Question
Long breakout continuation on US mid/small-cap daily: does anything survive
the gauntlet, and do the two mechanism-conditioned variants add anything
over the plain (published, presumed-decayed) Donchian baseline?

## Grounds
manifests/us_sp400.yaml, manifests/us_sp600.yaml. Current-membership
universes: any pass owes a PIT re-test before deployment talk, per house
protocol. Runner: scripts/validate_genome.py --splits 5. Ledger charge per
genome per ground: 1 candidate + up to 5 holdout peeks. Planned total:
8 candidates (4 genomes x 2 grounds). No other looks.

## Frozen family (genomes/, exactly these four, no post-look edits)
1. breakout_donchian20 -- close crosses prior 20d high; 2.5A stop, 40 bar
   max hold. CONTROL: the textbook family, expected dead.
2. breakout_donchian55 -- 55d boundary, 3.0A stop, 60 bar hold. CONTROL at
   position scale.
3. breakout_compress -- 20d cross AND atr_pct in its bottom 30th pctile
   (252d): compression before the break (clustered stops, thin book story).
4. breakout_rvolconf -- 20d cross AND rvol_m above its 70th pctile (252d):
   flow-confirmed break vs noise poke.

## Decision rule (pre-stated)
- A genome PASSES a ground only if the runner prints survival on 5/5
  splits. Mixed splits are not a pass.
- A mechanism variant (3 or 4) is INTERESTING on a ground only if it
  passes AND its 5-split median SR exceeds breakout_donchian20's median on
  the same ground. Beating null but not the decayed baseline = nothing.
- Anything else: family CLOSED on that ground. Expected outcome, stated
  now: all four dead; the run exists to close the seam with a ledgered
  answer, not because the baseline is believed alive.
- STOP: no threshold edits, no added variants, no re-runs with new seeds
  after any holdout look. Results logged to MECHANISM_MAP.md win or lose.

## Order
1. Sign below. 2. git commit. 3. Run the 8 commands. 4. Log + commit.

Signed: __________
