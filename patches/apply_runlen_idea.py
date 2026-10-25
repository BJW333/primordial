#!/usr/bin/env python3
"""apply_runlen_idea.py -- register the run-length reversal `_idea` family:
prereg skeleton + 4 frozen genome JSONs. No core changes. Runner is the
existing scripts/validate_genome.py.

Hypothesis (bar-scale analog of the seconds-scale dip-timing idea): after k
consecutive down DAILY closes on liquid crypto, forward returns are positive
enough to clear 70 bps round trip. The seconds-scale original is NOT testable
here (Primordial floor = 5m bars); it owes its own hazard-curve measurement
on 1s data before anyone believes the mechanism.

    python3.10 apply_runlen_idea.py
    # sign prereg/RUNLEN_PREREG.md, commit, THEN run the printed commands.

Idempotent: identical existing files print '='; divergent ones abort.
"""
import json
import os
import sys

T = lambda n: {"op": "term", "name": n, "ch": []}          # noqa: E731
C = lambda v: {"op": "const", "value": v, "ch": []}        # noqa: E731


def O(op, *ch, **kw):
    d = {"op": op, "ch": list(ch)}
    d.update(kw)
    return d


def genome(entry, stop, mh):
    return {"timeframe": "1d", "root_type": "bool", "entry_tree": entry,
            "regime_tree": None, "direction": "long",
            "entry_style": "market", "entry_param": 1, "stop_atr": stop,
            "target_r": 0.0, "time_stop": 0, "trail_mode": "none",
            "max_hold": mh, "anchor_ema": 0}


def RUN(k):
    """k consecutive down closes: max of the last k 1-bar diffs < 0."""
    return O("lt", O("roll_max", O("diff", T("close"), n=1), n=k), C(0))


# depth condition: 3-day dollar move in its deepest decile of the past year.
# pctile_rank is per-name time-series rank, so raw diff units are fine.
DEEP3 = O("lt",
          O("pctile_rank", O("roll_sum", O("diff", T("close"), n=1), n=3),
            n=252),
          C(0.10))

GENOMES = {
    "runlen2": genome(RUN(2), 2.0, 5),
    "runlen3": genome(RUN(3), 2.0, 5),
    "runlen4": genome(RUN(4), 2.0, 5),
    "runlen3_deep": genome(O("and", RUN(3), DEEP3), 2.0, 5),
}

PREREG = """# RUN-LENGTH REVERSAL `_idea` PRE-REGISTRATION -- sign BEFORE any command runs

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
"""


def write_new(path, content, label):
    if os.path.exists(path):
        if open(path).read() == content:
            print(f"  = {path}: {label} (already)")
            return
        sys.exit(f"REFUSING {path}: exists with different content. Diff it; "
                 "nothing further applied.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write(content)
    print(f"  + {path}: {label}")


def main():
    for name, g in GENOMES.items():
        write_new(os.path.join("genomes", name + ".json"),
                  json.dumps(g, indent=1) + "\n", "frozen genome")
    write_new(os.path.join("prereg", "RUNLEN_PREREG.md"), PREREG,
              "prereg skeleton")
    print("\nNext:")
    print('  # sign prereg/RUNLEN_PREREG.md, then:')
    print('  git add -A && git commit -m "runlen _idea family registered '
          '(4 genomes, coinbase_daily, signed)"')
    for name in GENOMES:
        print(f"  python3.10 scripts/validate_genome.py"
              f" --manifest manifests/coinbase_daily.yaml"
              f" --genome genomes/{name}.json --splits 5")


if __name__ == "__main__":
    main()
