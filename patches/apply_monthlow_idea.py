#!/usr/bin/env python3
"""apply_monthlow_idea.py -- register the monthlow `_idea` family: prereg
skeleton + 2 frozen genome JSONs. Requires apply_target_pct.py applied first
(refuses otherwise). Runner is the existing scripts/validate_genome.py.

    python3.10 apply_monthlow_idea.py
    # sign MONTHLOW_PREREG.md, commit, THEN run the printed commands.

Idempotent: identical existing files print '='; divergent ones abort.
"""
import json
import os
import sys

if "target_pct" not in open(os.path.join("primordial", "genome.py")).read():
    sys.exit("REFUSING: primordial/genome.py has no target_pct gene. Run "
             "apply_target_pct.py first. Nothing applied.")

T = lambda n: {"op": "term", "name": n, "ch": []}          # noqa: E731
C = lambda v: {"op": "const", "value": v, "ch": []}        # noqa: E731


def O(op, *ch, **kw):
    d = {"op": op, "ch": list(ch)}
    d.update(kw)
    return d


def genome(entry):
    return {"timeframe": "1d", "root_type": "bool", "entry_tree": entry,
            "regime_tree": None, "direction": "long",
            "entry_style": "market", "entry_param": 1, "stop_atr": 99.0,
            "target_r": 0.0, "time_stop": 0, "trail_mode": "none",
            "max_hold": 20, "anchor_ema": 0, "target_pct": 0.05}


# >= 10% below the prior 21-bar high AND a fresh 21-bar closing low
# (both windows lagged 1 so today's bar is not inside its own reference)
DIP = O("and",
        O("lte", O("div", T("close"),
                   O("lag", O("roll_max", T("high"), n=21), n=1)), C(0.90)),
        O("lte", T("close"), O("lag", O("roll_min", T("close"), n=21), n=1)))
HEALTHY = O("gt", T("close"), O("roll_mean", T("close"), n=200))

GENOMES = {
    "monthlow_dip": genome(DIP),
    "monthlow_dip_healthy": genome(O("and", DIP, HEALTHY)),
}

PREREG = """# MONTHLOW `_idea` PRE-REGISTRATION -- sign BEFORE any command runs

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
    write_new("MONTHLOW_PREREG.md", PREREG, "prereg skeleton")
    print("\nNext:")
    print('  # sign MONTHLOW_PREREG.md, then:')
    print('  git add -A && git commit -m "monthlow _idea family registered '
          '(2 genomes, sp600, signed)"')
    for name in GENOMES:
        print(f"  python3.10 scripts/validate_genome.py --manifest "
              f"manifests/us_sp600.yaml --genome genomes/{name}.json "
              f"--splits 5")


if __name__ == "__main__":
    main()
