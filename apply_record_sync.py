#!/usr/bin/env python3
"""apply_record_sync.py -- bring MECHANISM_MAP.md in line with reality.

Fixes: (1) the stale uncorrected distress verdict that survived inside
MECHANISM_MAP.md (the cat-append copied DISTRESS_VOL.md BEFORE the
correction, so the map still claimed 4 passes and deployment next-steps);
(2) records the opengap holdout DEAD and rotation sp500 PIT INCONCLUSIVE
results, which currently exist only in terminal scrollback.

Run from repo root:  python3.10 apply_record_sync.py   (idempotent)"""
import os, sys

if not os.path.exists("MECHANISM_MAP.md"):
    sys.exit("run from the primordial repo root.")
src = open("MECHANISM_MAP.md").read()
OLD = "## 5. Status\n\n**A real, replicated, thin market phenomenon \u2014 not yet a deployable edge.**\nFour independent markets passed every gate; the effect's sign is positive on\nall ten grounds tested. What it lacks versus deep-state: consistent bootstrap\nsurvival, a point-in-time survivorship check, and any forward evidence.\n\n**Next, in order:** (a) point-in-time re-test on delisted-inclusive data,\n(b) if that holds, a LEAN implementation and a pre-registered paper forward\ntest with its own bands, (c) trading the passing markets together \u2014 four\nthin uncorrelated streams beat one, and that is the honest way to raise\nrealized Sharpe here, not a fifth filter."
NEW = "## 5. Status \u2014 CLOSED (superseded; see DISTRESS_VOL.md section 6)\n\n**Dead. Zero honest passes on ten grounds; pre-registered Gate A failed\n0/5** (base genome, us_sp600, five hash-keyed splits: SR median +0.41,\nrange [+0.40, +0.46], bar +0.76 -- run 20260811_004913). The four foreign\n\"passes\" above ran at 0.00 luck bars (fingerprints carrying one trial;\nbar formula returns 0 for n < 2); charging the family's real debt (5\nvariants, 14 peeks) puts the minimum bar at +0.89, above the best foreign\nSR (+0.76). The SURVIVORS.md rows for uk/de/jp/kr at bar +0.00 are\nsuperseded by this closure. Full correction and what-was-learned:\nDISTRESS_VOL.md section 6. Keepable finding: same entry family and exit,\nABOVE the 200-day (deep-state) clears every gate; BELOW it prints a\nstable +0.4 that clears nothing honest anywhere -- the premium is for\nabsorbing sags in healthy names, not chaos in broken ones.\n\n# Addenda \u2014 closures recorded 2026-08-11\n\n## Open-gap intraday family: CLOSED 0-for-3\n\nRegistered one-look holdout (explore_open_gap.py --holdout-cell, cell\nfixed pre-look: gap <= -2 ATR, 09:30 auction entry, 120m exit, 25 bps,\npass = net > 0 AND t >= 2): net -0.442%/trade, t = -0.70, win 48% over\n95 trades / 82 sessions (2022-12-30 .. 2025-12-30). Train reference was\n+0.191%. By year: 2023 +0.91%, 2024 +0.50%, 2025 **-2.60%** -- the edge\ndid not fade, it inverted; buying the auction on 2-9 ATR gap-downs is\nbeing the counterparty to real news, and 2025 collected. Daily-bar gap\nvariants previously 0-for-2 vs bars ~+0.9. Family closed.\n\n## Rotation, sp500 PIT check: INCONCLUSIVE, leaning credible\n\nRegistered two-run comparison (test_weekly_rotation.py sp500pit mode,\nTRIALS=14): survivor mode holdout +1.15 SR, +2.50 sd above matched null\n(99.6th pct) but at the null MEDIAN in train (68.8th) -- one-regime\nsmell. PIT mode (863 names, membership-gated rule AND null, departed\nnames 5.4% of position-slots): holdout +0.67 SR, +1.48 sd (97th pct),\nand above its null in BOTH windows (94.8th train / 97.2th holdout) --\nthe script's own verdict flipped to \"weak but persistent.\" Neither\nregistered trigger fired (BIAS-DRIVEN needed PIT < +1.0; SHAPE ROBUST\nneeded PIT >= +2.0). Read: the edge attenuated but did not collapse when\ndead names were restored, and part of survivor-mode's regime-dependence\nwas the NULL being survivorship-inflated (random picks from\nguaranteed-survivors earn +0.99 in train; restore the dead and random\nearns +0.26 while the ranking still finds the live names). The sp400\n+2.72 sd claim stays provisionally credible; final verdict deferred to\npoint-in-time sp400 data (Norgate/Sharadar), scheduled for the\ndeep-state Gate C moment. Nothing trades before that check."
if NEW in src:
    print("  = already applied"); sys.exit(0)
if src.count(OLD) != 1:
    sys.exit("ANCHOR problem in MECHANISM_MAP.md -- aborting, nothing written.")
open("MECHANISM_MAP.md","w").write(src.replace(OLD, NEW))
print("  + MECHANISM_MAP.md: stale distress verdict corrected; opengap DEAD")
print("    and rotation sp500 PIT INCONCLUSIVE recorded as addenda.")
