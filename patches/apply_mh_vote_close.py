#!/usr/bin/env python3
"""apply_mh_vote_close.py -- log the mh_vote family close-out to
MECHANISM_MAP.md. Numbers transcribed from runs/20260909_{112705,124201,
130732,161032}_idea/run.json (base, vote_alone, gate2, gate4).

    python3.10 patches/apply_mh_vote_close.py

Idempotent; anchored; exits loudly if the map tail moved.
"""
import sys

MMAP = "MECHANISM_MAP.md"

# anchor: current final line of the EDGAR distress-issuer entry
ANCHOR = ("profitably. EDGAR event pipeline (5,110 day-precision 424B "
          "events,\nself-updating via refetch) retained as infrastructure "
          "for the analyst\nleg. Seam closed.\n")

ENTRY_HEAD = """
## Multi-horizon sign vote (mh_vote) -- family DEAD 2026-09-09 (15 trials)
MH_VOTE_PREREG.md, coinbase_daily_30 (coinbase_liquid_v2, 2022-01->
2026-06, spot taker, 1d), validate_genome --splits 5, runs 20260909_*.
Signal: four DISJOINT segment sign votes (bars 0-7/7-14/14-30/30-60,
vol-scaled dead zone), score in [-4,+4]; Donchian 20d cross as entry.

- base_donchian20: median SR -0.84 [-1.05,-0.36], 57-66 tr/split, 0/5.
  Dead as pre-stated. Baseline level set.
- vote_alone: median SR -2.47 [-5.03,-1.58], 37-46 tr/split, PSR <=
  0.13 everywhere, 0/5. WORSE than bare Donchian: entering after 2+
  weeks of unanimous uptrend bought tops. Direct evidence AGAINST the
  premise -- high trend-consistency scores predicted reversal, not
  continuation, on this ground.
- gate2 (the idea): 0/5, trades collapsed ~60 -> 12-20 per split with
  no improvement in survivors (pooled PSR 0.04-0.21). The vote
  decimated; it did not select. Printed median -923.53 is a DISPLAY
  ARTIFACT (all-stop-out cell, std ~ 0.003; fixed by the v0.1.8 clip),
  and one split's +0.00 was zero eligible cells, not breakeven. Both
  numbers' VERDICTS stand via pooled PSR, which never had the bug.
"""

GATE4 = """- gate4: run 20260909_161032. THREE trades total across 5 splits
  (1/0/0/1/1); every split zero eligible cells, printed +0.00 (the
  v0.1.8 artifact, live specimen). Unanimous +4 at a fresh 20d-high
  cross almost never co-occurs on this ground. Trade-starved as
  pre-stated, and moot regardless -- the sizing ladder was conditional
  on gate2 being INTERESTING. Trials charged: all four runs (ledger
  shows 18 effective on this holdout incl. synthetic-era smoke).
"""

ENTRY_TAIL = """
Family CLOSED on this ground per the decision rule -- both parents dead,
combo dead, nothing to size. mh_vote atom stays in the crypto adapter as
vocabulary (it is a computed feature, not doctrine). NOT licensed by this
close: sign-flipping the vote (fading high scores) -- that is a NEW
family owing its own prereg, and it starts skeptical because this
holdout leaked into its motivation. Windows/dead-zone re-tuning on this
ground: same prohibition, per the STOP clause.
"""


def main():
    entry = ENTRY_HEAD + GATE4 + ENTRY_TAIL

    s = open(MMAP).read()
    if "## Multi-horizon sign vote (mh_vote)" in s:
        print(f"  = {MMAP}: mh_vote close-out (already)")
        return
    if s.count(ANCHOR) != 1:
        sys.exit(f"REFUSING {MMAP}: anchor (EDGAR entry tail) not found "
                 "exactly once -- the map moved. Append by hand or fix the "
                 "anchor; nothing applied.")
    open(MMAP, "w").write(s.replace(ANCHOR, ANCHOR + entry, 1))
    print(f"  ~ {MMAP}: mh_vote family close-out")
    print("\nNext:")
    print('  git add -A && git commit -m "mh_vote family dead on '
          'coinbase_daily: both parents + combo 0/5; closed per prereg"')


if __name__ == "__main__":
    main()
