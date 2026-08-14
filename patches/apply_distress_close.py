#!/usr/bin/env python3
"""apply_distress_close.py -- correction + Gate A closure, one shot.

Run from the primordial repo root:  python3.10 apply_distress_close.py
Anchored edits; aborts writing nothing on any mismatch. Idempotent."""
import os, sys

def patch(path, old, new, label):
    src = open(path).read()
    if new in src:
        print(f"  = {label}: already applied"); return
    if old not in src:
        sys.exit(f"ANCHOR MISSING in {path} ({label}) -- aborting, nothing written.")
    if src.count(old) != 1:
        sys.exit(f"ANCHOR NOT UNIQUE in {path} ({label}) -- aborting, nothing written.")
    open(path, "w").write(src.replace(old, new)); print(f"  + {label}")

for _p in ("DISTRESS_VOL.md", "DISTRESS_PREREG.md"):
    if not os.path.exists(_p):
        sys.exit(f"{_p} not found -- run from the primordial repo root.")

patch("DISTRESS_VOL.md", "## 5. Status\n\n**A real, replicated, thin market phenomenon \u2014 not yet a deployable edge.**\nFour independent markets passed every gate; the effect's sign is positive on\nall ten grounds tested. What it lacks versus deep-state: consistent bootstrap\nsurvival, a point-in-time survivorship check, and any forward evidence.\n\n**Next, in order:** (a) point-in-time re-test on delisted-inclusive data,\n(b) if that holds, a LEAN implementation and a pre-registered paper forward\ntest with its own bands, (c) trading the passing markets together \u2014 four\nthin uncorrelated streams beat one, and that is the honest way to raise\nrealized Sharpe here, not a fifth filter.", "## 5. Status \u2014 CLOSED (superseded by section 6)\n\n**Dead. Zero honest passes on ten grounds; Gate A failed 0/5.** The verdict\noriginally written here (\"a real, replicated, thin market phenomenon\") did\nnot survive the ledger audit or the pre-registered re-test below.\n\n## 6. Correction and Gate A result (2026-08-11)\n\nSections 1-4 stand as the record of what was run. Their conclusions do not.\n\n**\"Four clean passes\" -> zero.** All four ran at a 0.00 luck bar: their\nfingerprints carried one trial each, and the bar formula returns 0 for\nn < 2. The bar gate was ABSENT, not cleared. Charging any ground the\nfamily's actual debt (5 variants, 14 peeks, via seed_ledger_distress.py)\nputs the minimum bar at +0.89 \u2014 above the best foreign SR (+0.76, Japan).\nThe geography pattern (\"works abroad, fails at home\") was a map of\nunseeded ledgers. The US grounds were the only ones with a real bar AND\nthe only direction survivorship bias permits trusting; both said no.\nEvery section-1 number was also a single draw of the pre-v0.1.10\nfetch-order-unstable splitter (+0.92 vs +0.31 on one identical run pair).\n\n**Gate A (pre-registered, DISTRESS_PREREG.md), run 20260811_004913:**\nbase genome, us_sp600, five deterministic hash-keyed splits:\n\n    SR +0.40 +0.41 +0.46 +0.42 +0.40  | median +0.41\n    range 0.06 wide | bar +0.76 | beat bar on 0/5\n\nPer the prereg's stop-at-first-fail rule: the idea is DEAD. No Gate B,\nno third country, no point-in-time spend, no LEAN port, no multi-market\nsizing.\n\n**What was actually learned (and is worth keeping):**\n- The sign is stable: +0.40 to +0.46 on every split, every ground ever\n  tested positive. There is a real tendency here \u2014 it is just far below\n  what 14 trials of selection dredge up by luck, before survivorship,\n  which for this below-trend rule shape is maximal and un-nulled.\n- The v0.1.10 splitter's 0.06-wide range against the old +0.92/+0.31\n  spread confirms the split fix: that spread was fetch noise, not market\n  structure.\n- Long-hold liquidity provision remains proven only in the form that\n  passed everything: deep-state (above-trend). The below-trend mirror\n  does not clear an honest bar. That asymmetry IS the mechanism map\n  entry: the premium is for absorbing sags in healthy names, not chaos\n  in broken ones \u2014 at least not at a size this framework can certify.\n\nThe PASSES JAPAN / PASSES KOREA / 6-passes commit messages in this\nfile's history are superseded by this section.", "DISTRESS_VOL.md: verdict corrected + Gate A result (section 6)")
patch("DISTRESS_PREREG.md", "Signed: ____________  date: __________\n(edit the thresholds if you disagree with them -- then sign. Unsigned,\nthis file is a proposal, not a registration.)\n", "Signed: ____________  date: __________\n(edit the thresholds if you disagree with them -- then sign. Unsigned,\nthis file is a proposal, not a registration.)\n\n## RESULT (2026-08-11) -- CLOSED AT GATE A\n\nRun 20260811_004913_idea, base genome sha b62a9753323f1e5e, us_sp600,\nfive splits: SR median +0.41, range [+0.40, +0.46], beat bar on 0/5\n(bar +0.76). Stop-at-first-fail: the distress idea is DEAD. Gates B and\nC never ran and must not run. Any future revival of this idea family\nstarts a NEW prereg and inherits this ledger debt.\n", "DISTRESS_PREREG.md: RESULT -- closed at Gate A")

print("\nDistress idea closed. Commit this so the record matches reality.")
