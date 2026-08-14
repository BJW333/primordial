#!/usr/bin/env python3
"""apply_breakout_idea.py -- register the breakout `_idea` family: prereg
skeleton + 4 frozen genome JSONs. No core changes. Runner is the existing
scripts/validate_genome.py.

    python3.10 apply_breakout_idea.py
    # sign BREAKOUT_PREREG.md, commit, THEN run the printed commands.

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


CROSS20 = O("cross_above", T("close"),
            O("lag", O("roll_max", T("high"), n=20), n=1))
CROSS55 = O("cross_above", T("close"),
            O("lag", O("roll_max", T("high"), n=55), n=1))

GENOMES = {
    "breakout_donchian20": genome(CROSS20, 2.5, 40),
    "breakout_donchian55": genome(CROSS55, 3.0, 60),
    "breakout_compress": genome(
        O("and", CROSS20,
          O("lt", O("pctile_rank", T("atr_pct"), n=252), C(0.30))), 2.5, 40),
    "breakout_rvolconf": genome(
        O("and", CROSS20,
          O("gt", O("pctile_rank", T("rvol_m"), n=252), C(0.70))), 2.5, 40),
}

PREREG = """# BREAKOUT `_idea` PRE-REGISTRATION -- sign BEFORE any command runs

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
    write_new("BREAKOUT_PREREG.md", PREREG, "prereg skeleton")
    print("\nNext:")
    print('  # sign BREAKOUT_PREREG.md, then:')
    print('  git add -A && git commit -m "breakout _idea family registered '
          '(4 genomes, sp400+sp600, signed)"')
    for mfp in ("manifests/us_sp400.yaml", "manifests/us_sp600.yaml"):
        for name in GENOMES:
            print(f"  python3.10 scripts/validate_genome.py --manifest {mfp}"
                  f" --genome genomes/{name}.json --splits 5")


if __name__ == "__main__":
    main()
