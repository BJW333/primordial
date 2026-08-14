#!/usr/bin/env python3
"""apply_crypto_close.py -- record the crypto 15m run honestly; close the
seam, hold it open, or mark it conditional, depending on what you know
about the account's actual Coinbase Advanced Trade 30-day taker rate.

    python3.10 apply_crypto_close.py                  # tier unknown right now
    python3.10 apply_crypto_close.py --taker-bps 60   # tier known -- also
                                                      # upgrades an earlier
                                                      # unverified close-out
                                                      # in place

No argument: prereg is signed post-run with the tier marked UNVERIFIED and
the conditional stated (>= 25 bps -> verdict stands conservatively; < 25 ->
not conclusive); the lookup is recorded as owed before any future crypto
ground. With --taker-bps N: the number is written, and the verdict wording
resolves to CLOSED (N >= 25) or SEAM OPEN + re-run instructions (N < 25).

Idempotent; anchored; exits loudly on missing anchors.
"""
import argparse
import sys

PREREG = "CRYPTO_INTRADAY_PREREG.md"
MMAP = "MECHANISM_MAP.md"
RUN_ID = "20260814_124328"
DATE = "2026-08-14"

TIER_UNV = ("UNVERIFIED at close-out. Model assumed 25 bps. If actual >= 25 "
            "the zero-survivor verdict stands conservatively; if < 25 it is "
            "NOT conclusive. Lookup owed before any future crypto ground.")
SIG_UNV = "Tier unverified at signing; verdict conditional as noted above."
MAP_UNV = ("Account taker tier UNVERIFIED at close-out: if >= 25 bps the "
           "verdict stands conservatively, if < 25 it is not conclusive and "
           "the seam reopens. Tier lookup owed before any future crypto "
           "ground or redeploy.")


def sub_once(path, old, new, label):
    s = open(path).read()
    if new in s:
        print(f"  = {path}: {label} (already)")
        return
    n = s.count(old)
    if n != 1:
        sys.exit(f"ANCHOR FAIL {path}: {label!r} found {n}x, need exactly 1. "
                 "Nothing further applied; diff the file.")
    open(path, "w").write(s.replace(old, new, 1))
    print(f"  + {path}: {label}")


def append_once(path, marker, block, label):
    if marker in open(path).read():
        print(f"  = {path}: {label} (already)")
        return
    open(path, "a").write(block)
    print(f"  + {path}: {label}")


def seam_block(header_suffix, tail):
    return f"""

## Crypto intraday 15m (coinbase_liquid_v2) -- {header_suffix} {DATE}
Registered evolutionary search: 28 USD pairs 2021-06 -> 2026-08, 15m base
(1h/4h allowed), long/flat, 24h wall-clock hold cap, taker/taker modeled at
25 bps commission (70 bps round trip all-in). 0 survivors of 6 candidates
vs bar +1.70; this holdout now carries 54 effective trials. Evolution
abandoned 15m by gen 1 and camped on 4h -- same flight-to-coarse as
us_liquid_5m: at these tolls sub-hour holds are unmineable. All finalists
momentum-shaped; train stagnated at +0.136 for 17 generations; best
holdout -1.08. Prereg was signed post-run (process violation, noted in the
prereg itself). {tail} runs/{RUN_ID}.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taker-bps", type=float, default=None,
                    help="ACTUAL Advanced Trade 30-day taker rate in bps; "
                         "omit to record the tier as unverified")
    ap.add_argument("--name", default="Blake Weiss")
    a = ap.parse_args()

    # ---------------------------------------------- tier unknown right now
    if a.taker_bps is None:
        sub_once(PREREG, "confirmed as: __________",
                 f"confirmed as: {TIER_UNV}", "tier recorded unverified")
        sub_once(PREREG, "Signed: __________",
                 f"Signed: {a.name} {DATE} -- POST-RUN: run {RUN_ID} "
                 "executed before this prereg was signed or the tier "
                 f"confirmed (process violation, recorded). {SIG_UNV}",
                 "signed post-run, annotated")
        append_once(MMAP, "## Crypto intraday 15m",
                    seam_block("CLOSED CONDITIONALLY", MAP_UNV),
                    "seam paragraph")
        print('\nNext:\n  git add -A && git commit -m "crypto 15m recorded: '
              '0/6 vs +1.70, fled to 4h; prereg signed post-run, tier '
              'unverified (conditional close)"')
        return

    # ------------------------------------------------------- tier known
    bps = a.taker_bps
    if not (0 < bps < 500):
        sys.exit(f"--taker-bps {bps} not plausible; give basis points "
                 "(60 = 0.60%).")
    stands = bps >= 25.0
    tier_note = (f"{bps:g} bps ACTUAL (>= 25 bps modeled: run costs were "
                 "conservative)" if stands else
                 f"{bps:g} bps ACTUAL (< 25 bps modeled: run OVERSTATED "
                 "costs; verdict not conclusive for this account)")
    verdict = ("Seam CLOSED at this cost structure; actual taker "
               f"{bps:g} bps >= modeled, verdict stands conservatively."
               if stands else
               f"Actual taker {bps:g} bps < modeled 25: tolls were "
               "OVERSTATED, zero survivors is NOT conclusive here. Seam "
               "OPEN pending re-run at real costs under a fresh prereg.")

    tier_old = TIER_UNV if TIER_UNV in open(PREREG).read() else "__________"
    sub_once(PREREG, f"confirmed as: {tier_old}",
             f"confirmed as: {tier_note}", "fee tier filled")

    p = open(PREREG).read()
    if "Tier verified later:" in p:
        print(f"  = {PREREG}: signature clause (already upgraded)")
    elif SIG_UNV in p:
        sub_once(PREREG, SIG_UNV, "Tier verified later: "
                 + ("verdict stands (>= modeled)." if stands else
                    "verdict NOT conclusive (< modeled); seam reopens per "
                    "MECHANISM_MAP."), "signature clause upgraded")
    else:
        sub_once(PREREG, "Signed: __________",
                 f"Signed: {a.name} {DATE} -- POST-RUN: run {RUN_ID} "
                 "executed before this prereg was signed or the tier "
                 "confirmed (process violation, recorded). "
                 + ("Verdict stands: real tolls >= modeled tolls, so zero "
                    "survivors is the conservative-valid answer." if stands
                    else "Verdict NOT conclusive at this account's real "
                    "costs; seam remains OPEN pending re-run at the actual "
                    "tier under a fresh prereg."),
                 "signed post-run, annotated")

    if MAP_UNV in open(MMAP).read():
        sub_once(MMAP, MAP_UNV, verdict, "seam verdict upgraded")
        if stands:
            sub_once(MMAP, "-- CLOSED CONDITIONALLY " + DATE,
                     "-- CLOSED " + DATE, "seam header upgraded")
        else:
            sub_once(MMAP, "-- CLOSED CONDITIONALLY " + DATE,
                     "-- RUN RECORDED, SEAM OPEN " + DATE,
                     "seam header upgraded")
    else:
        append_once(MMAP, "## Crypto intraday 15m",
                    seam_block("CLOSED" if stands else
                               "RUN RECORDED, SEAM OPEN", verdict),
                    "seam paragraph")

    print("\nNext:")
    if stands:
        print(f'  git add -A && git commit -m "crypto 15m closed: 0/6 vs '
              f'+1.70, fled to 4h; tier {bps:g}bps, prereg signed post-run '
              f'(noted)"')
    else:
        print(f'  git add -A && git commit -m "crypto 15m run recorded; '
              f'tier {bps:g}bps < modeled 25, verdict not conclusive, seam '
              f'open"')
        print(f"  # honest re-run: universe.py COST_TIERS "
              f"crypto_spot_taker commission 25.0 -> {bps:g}; fresh prereg "
              f"signed BEFORE; new seed")


if __name__ == "__main__":
    main()
