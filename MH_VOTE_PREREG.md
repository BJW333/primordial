# MULTI-HORIZON VOTE `_idea` PRE-REGISTRATION -- sign BEFORE any command

## Question
Crypto daily, long/flat: does a disjoint-segment trend-consistency score
(mh_vote) add anything to a Donchian 20 breakout entry -- as a regime gate
now, as a conviction-sizing overlay at deployment?

## Signal (frozen; computed in atoms.py, nothing fitted)
mh_vote = sum of four sign votes on NON-OVERLAPPING return segments, in
bars: [0,7) [7,14) [14,30) [30,60). Vote 0 inside a vol-scaled dead zone
(0.5 * sigma_bar * sqrt(seglen)). Range [-4, +4].

## Ground
manifests/coinbase_daily.yaml (coinbase_liquid_v2, 2022-01..2026-06, spot
taker costs, 1d genomes). Runner: scripts/validate_genome.py --splits 5.
Ledger charge per genome: 1 candidate + up to 5 holdout peeks. Planned
total: 4 candidates. No other looks.

## Frozen family (genomes/, exactly these four, no post-look edits)
1. mhv_base_donchian20 -- close crosses prior 20d high. PARENT/CONTROL.
2. mhv_vote_alone -- entry when mh_vote crosses above 1.5 (score reaches
   +2). PARENT: the vote with no breakout.
3. mhv_gate2 -- Donchian entry, regime gate mh_vote > 1.5. THE IDEA.
4. mhv_gate4 -- Donchian entry, regime gate mh_vote > 3.5 (unanimous).
   MONOTONICITY PROBE, stands in for the sizing ladder.

All: long only, market at next open, 2.5 ATR stop, 40 bar max hold, no
target, no trail. Shorts deferred (crypto spot adapter is long/flat, and
the -2/-4 short legs were deferred at design time anyway). The +2 half /
+4 full SIZING ladder is a deployment-layer mapping: the engine sizes
binary, so it is tested here as the gate2-vs-gate4 comparison, not as
fractional positions.

## Decision rule (pre-stated)
- A genome PASSES only on 5/5 split survival. Mixed splits are not a pass.
- mhv_gate2 is INTERESTING only if it PASSES and its 5-split median SR
  exceeds BOTH parents' medians on this ground. Beating one parent = the
  other parent was doing the work = nothing.
- Sizing ladder is JUSTIFIED for deployment only if gate2 is INTERESTING
  and gate4's median SR >= gate2's (conviction monotone). gate4 passing
  while gate2 fails = a thin-trade-count fluke, not a result.
- Anything else: family CLOSED on this ground. Expected outcome, stated
  now: skeptical -- the vote is time-series momentum in a hat, and TS-mom
  entries start behind costs. The run exists for a ledgered answer.
- STOP: no threshold edits (1.5/3.5, dead zone 0.5, windows 7/14/30/60),
  no added variants, no re-runs with new seeds after any holdout look.
  Results logged to MECHANISM_MAP.md win or lose.

## Order
1. Sign below. 2. git commit. 3. Run the 4 commands. 4. Log + commit.

Signed: __________
