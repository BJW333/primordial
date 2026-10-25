# RUN-LENGTH REVERSAL `_idea` PRE-REGISTRATION -- sign BEFORE any command runs

## Question
Long dip-buy after k consecutive down daily closes on liquid crypto spot:
does any k clear the gauntlet at 70 bps round trip, is there a
dose-response in k, and does conditioning on dip DEPTH add anything over
bare run length?

## Provenance / scope note
This is the bar-scale analog of a seconds-scale run-length timing idea.
The seconds-scale original is OUT OF SCOPE here (framework floor = 5m
bars) and owes a separate hazard-curve measurement on 1s data; nothing in
this run confirms or kills the seconds mechanism. The 15m intraday ground
is deliberately NOT charged: sub-hour holds are already closed as
unmineable at crypto_spot_taker costs (see learnings); re-running that
seam buys nothing.

## Grounds
manifests/coinbase_daily.yaml only. Runner:
scripts/validate_genome.py --splits 5. Ledger charge per genome:
1 candidate + up to 5 holdout peeks. Planned total: 4 candidates
(4 genomes x 1 ground). No other looks. Note: this holdout has absorbed
18 effective trials (mh_vote); multiplicity skepticism applies to any
marginal pass.

## Frozen family (genomes/, exactly these four, no post-look edits)
1. runlen2 -- 2 consecutive down closes; 2.0A stop, 5 bar max hold.
2. runlen3 -- 3 consecutive down closes; same exits. PRIMARY.
3. runlen4 -- 4 consecutive down closes; same exits. Dose ladder top.
4. runlen3_deep -- runlen3 AND 3-day move in its deepest 10th pctile
   (252d): depth-conditioned dip (forced-seller / overshoot story).

## Decision rule (pre-stated)
- A genome PASSES only if the runner prints survival on 5/5 splits.
  Mixed splits are not a pass.
- Dose-response readout: median SR ordering across k=2,3,4 is recorded
  as mechanism evidence either way; non-monotone survivors are suspect.
- runlen3_deep is INTERESTING only if it passes AND its 5-split median
  SR exceeds runlen3's median. Beating null but not the bare run = nothing.
- Any survivor owes a duplication check against existing live/paper
  survivors (deep_state is mean-reversion on equities; different asset
  class does not waive the check) before any deployment talk.
- Anything else: family CLOSED on this ground. Expected outcome, stated
  now: short-horizon reversal on crypto daily is heavily mined; base rate
  says all four dead. The run exists to put a ledgered answer on the
  reversal side of coinbase_daily (mh_vote closed only the trend side).
- STOP: no k edits, no threshold edits, no added variants, no re-runs
  after any holdout look. Results logged to MECHANISM_MAP.md win or lose.

## Order
1. Sign below. 2. git commit. 3. Run the 4 commands. 4. Log + commit.

Signed: __________
