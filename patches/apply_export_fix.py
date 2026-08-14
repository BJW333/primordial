#!/usr/bin/env python3
"""apply_export_fix.py -- fix my bug: _export_series was appended AFTER the
`if __name__ == "__main__": sys.exit(main())` guard, so main() ran before the
def statement was ever executed -> NameError. This relocates the helper to
just above the guard in both rotation scripts. Idempotent; verifies by
compiling each file afterwards.

Run from repo root:  python3.10 apply_export_fix.py
"""
import ast
import os
import sys

GUARD = 'if __name__ == "__main__":'
MARK = "def _export_series("


def fix(path):
    src = open(path).read()
    if MARK not in src:
        sys.exit(f"{path}: _export_series not found -- run apply_series_compare.py first.")
    gi = src.index(GUARD)
    hi = src.index(MARK)
    if hi < gi:
        print(f"  = {path}: already in the right place")
        return
    # helper block runs from the blank lines before the def to end of file
    start = src.rindex("\n\n", 0, hi)
    helper = src[start:].rstrip("\n") + "\n"
    rest = src[:start].rstrip("\n") + "\n"
    gi2 = rest.index(GUARD)
    out = rest[:gi2].rstrip("\n") + "\n" + helper + "\n\n" + rest[gi2:]
    ast.parse(out)                       # refuse to write a broken file
    open(path, "w").write(out)
    print(f"  + {path}: helper moved above the main guard")


for p in ("scripts/test_volume_rotation.py", "scripts/test_weekly_rotation.py"):
    if not os.path.exists(p):
        sys.exit("run from the primordial repo root.")
    fix(p)
    ast.parse(open(p).read())
    if open(p).read().index(MARK) > open(p).read().index(GUARD):
        sys.exit(f"{p}: helper STILL after the guard -- aborting.")

print("\nBoth files compile and define _export_series before main runs.")
print("Re-run the two rotations, then compare_series.")
