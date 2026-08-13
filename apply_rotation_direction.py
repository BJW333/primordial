#!/usr/bin/env python3
"""apply_rotation_direction.py -- DIRECTION=best|worst on the industry
rotation, so the "buy the bottomed-out sector and ride the recovery"
version is testable with the same machinery, costs and null.

Run from repo root:  python3.10 apply_rotation_direction.py   (idempotent)"""
import os, sys

P = "scripts/test_volume_rotation.py"

def patch(old, new, label):
    s = open(P).read()
    if new in s:
        print(f"  = {label}: already applied"); return
    if s.count(old) != 1:
        sys.exit(f"ANCHOR problem ({label}) -- aborting, nothing written.")
    open(P, "w").write(s.replace(old, new)); print(f"  + {label}")

if not os.path.exists(P):
    sys.exit("run from the primordial repo root.")
patch("RIDE-UNTIL-IT-FADES is implemented as a hysteresis band, which is what", "DIRECTION=worst flips the sort: hold the BOTTOM-ranked industries and ride\nthe recovery instead of the rise. Everything else is held fixed, so best vs\nworst is a clean read on which way industry-level flow actually pays. Note\nthe null is direction-agnostic (random industries), so it prices selection,\nnot direction -- if worst beats the null and best does not, that is a real\nasymmetry and not an artifact of an easier benchmark.\n\nRIDE-UNTIL-IT-FADES is implemented as a hysteresis band, which is what", "docstring: DIRECTION")
patch("SIGNAL = os.environ.get(\"SIGNAL\", \"obv\")", "SIGNAL = os.environ.get(\"SIGNAL\", \"obv\")\n# DIRECTION=best  -> hold the TOP-ranked industries (momentum: ride the rise)\n# DIRECTION=worst -> hold the BOTTOM-ranked industries (reversion: buy the\n#   beaten-down sector and ride the recovery). Same machinery, same costs,\n#   same turnover-matched null -- only the sort flips, so the two are\n#   directly comparable and neither gets a hidden advantage.\nDIRECTION = os.environ.get(\"DIRECTION\", \"best\")\nif DIRECTION not in (\"best\", \"worst\"):\n    raise SystemExit(\"DIRECTION must be best or worst\")", "DIRECTION env flag")
patch("        ranked = sorted(sc, key=lambda k: sc[k], reverse=True)\n        pos = {k: i for i, k in enumerate(ranked)}", "        ranked = sorted(sc, key=lambda k: sc[k],\n                        reverse=(DIRECTION == \"best\"))\n        pos = {k: i for i, k in enumerate(ranked)}", "BandPicker: direction-aware sort")
patch("        if all(sc[k] < 0 for k in keep):\n            CASH_FIRES[0] += 1\n            if RULE_CASH:\n                self.book = []\n                return []", "        # The cash switch is a MOMENTUM idea: sit out when the whole top\n        # slice is falling. For DIRECTION=worst that condition is the\n        # entry premise, not an abort -- firing it there would delete the\n        # trade being tested. So it only applies to best.\n        if DIRECTION == \"best\" and all(sc[k] < 0 for k in keep):\n            CASH_FIRES[0] += 1\n            if RULE_CASH:\n                self.book = []\n                return []", "cash switch applies to best only")
print("\nCommit, then the pair (control first):")
print("  SIGNAL=roc DIRECTION=best  MODE=cluster python3.10 " + P)
print("  SIGNAL=roc DIRECTION=worst MODE=cluster python3.10 " + P)
