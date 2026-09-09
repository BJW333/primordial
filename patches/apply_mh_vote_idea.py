#!/usr/bin/env python3
"""apply_mh_vote_idea.py -- register the multi-horizon sign-vote `_idea`
family on crypto daily: one new atom (mh_vote), one new manifest, prereg
skeleton + 4 frozen genome JSONs. Runner is scripts/validate_genome.py.

CORE CHANGE (precedented, deep-state pattern): atoms.py gains a
`multi_horizon` block computing `mh_vote`, and the crypto adapter's
atom_blocks gains that block. mh_vote is a feature computed identically
for every symbol/split -- nothing is fitted -- so it is ledger-safe the
same way atr_pct is.

    python3.10 patches/apply_mh_vote_idea.py     # run from repo root
    # sign MH_VOTE_PREREG.md, commit, THEN run the printed commands.

Idempotent: applied patches / identical files print '='; divergent state
aborts loudly with nothing further applied.
"""
import json
import os
import sys

if not os.path.exists("primordial/atoms.py"):
    sys.exit("run from the repo root (primordial/atoms.py not found)")

T = lambda n: {"op": "term", "name": n, "ch": []}          # noqa: E731
C = lambda v: {"op": "const", "value": v, "ch": []}        # noqa: E731


def O(op, *ch, **kw):
    d = {"op": op, "ch": list(ch)}
    d.update(kw)
    return d


def genome(entry, regime, stop, mh):
    return {"timeframe": "1d", "root_type": "bool", "entry_tree": entry,
            "regime_tree": regime, "direction": "long",
            "entry_style": "market", "entry_param": 1, "stop_atr": stop,
            "target_r": 0.0, "time_stop": 0, "trail_mode": "none",
            "max_hold": mh, "anchor_ema": 0}


CROSS20 = O("cross_above", T("close"),
            O("lag", O("roll_max", T("high"), n=20), n=1))
VOTE = T("mh_vote")

GENOMES = {
    # parent 1: bare Donchian on this ground. CONTROL.
    "mhv_base_donchian20": genome(CROSS20, None, 2.5, 40),
    # parent 2: the vote by itself as the entry event (score reaches >= 2).
    "mhv_vote_alone": genome(O("cross_above", VOTE, C(1.5)), None, 2.5, 40),
    # the actual idea: Donchian entry gated by vote >= 2.
    "mhv_gate2": genome(CROSS20, O("gt", VOTE, C(1.5)), 2.5, 40),
    # monotonicity probe: same, vote == 4 only (all four segments agree).
    "mhv_gate4": genome(CROSS20, O("gt", VOTE, C(3.5)), 2.5, 40),
}

ATOM_FUNC = '''

def _multi_horizon(df, out):
    """Disjoint-segment sign vote (mh_vote family, 2026-09). Four
    NON-OVERLAPPING lookback segments in bars: [0,7) [7,14) [14,30) [30,60).
    Each segment votes on its close-to-close return: +1 up, -1 down, 0 if
    |ret| is inside a vol-scaled dead zone (0.5 * sigma_bar * sqrt(seglen),
    sigma_bar = 90-bar rolling std of bar returns). Sum in [-4, +4]; dead
    zone makes odd scores reachable. Windows are BARS: on crypto 1d these
    are calendar weeks/months because the tape never closes; on equities
    the same numbers run ~40% longer in wall-clock. Nothing is fitted."""
    c = df["close"]
    r = c.pct_change()
    sig = r.rolling(90, min_periods=30).std()
    vote = None
    for a, b in ((0, 7), (7, 14), (14, 30), (30, 60)):
        seg = c.shift(a) / c.shift(b).where(c.shift(b) > 0) - 1
        dz = 0.5 * sig * float(np.sqrt(b - a))
        v = pd.Series(np.sign(seg), index=c.index).where(seg.abs() >= dz, 0.0)
        vote = v if vote is None else vote + v
    out["mh_vote"] = vote

'''

MANIFEST = """name: coinbase_daily_30
asset_class: crypto
source: coinbase
symbols: coinbase_liquid_v2
base_timeframe: 1h               # coinbase serves no 1d; resampled up
allowed_timeframes: [1h, 4h, 1d]
start: "2022-01-01"
end: "2026-06-01"
cost_tier: crypto_spot_taker
holdout_name_frac: 0.4
holdout_time_frac: 0.3
embargo_bars: 60                 # covers the 60-bar mh_vote lookback at 1d
engines: [timing]
n_motifs: 6
motif_scales: [20, 40]
"""

PREREG = """# MULTI-HORIZON VOTE `_idea` PRE-REGISTRATION -- sign BEFORE any command

## Question
Crypto daily, long/flat: does a disjoint-segment trend-consistency score
(mh_vote) add anything to a Donchian 20 breakout entry -- as a regime gate
now, as a conviction-sizing overlay at deployment?

## Signal (frozen; computed in atoms.py, nothing fitted)
mh_vote = sum of four sign votes on NON-OVERLAPPING return segments, in
bars: [0,7) [7,14) [14,30) [30,60). Vote 0 inside a vol-scaled dead zone
(0.5 * sigma_bar * sqrt(seglen)). Range [-4, +4].

## Ground
manifests/coinbase_daily.yaml (coinbase_liquid_v2, 2022-01..2026-06, spot
taker costs, 1d genomes). Runner: scripts/validate_genome.py --splits 5.
Ledger charge per genome: 1 candidate + up to 5 holdout peeks. Planned
total: 4 candidates. No other looks.

## Frozen family (genomes/, exactly these four, no post-look edits)
1. mhv_base_donchian20 -- close crosses prior 20d high. PARENT/CONTROL.
2. mhv_vote_alone -- entry when mh_vote crosses above 1.5 (score reaches
   +2). PARENT: the vote with no breakout.
3. mhv_gate2 -- Donchian entry, regime gate mh_vote > 1.5. THE IDEA.
4. mhv_gate4 -- Donchian entry, regime gate mh_vote > 3.5 (unanimous).
   MONOTONICITY PROBE, stands in for the sizing ladder.

All: long only, market at next open, 2.5 ATR stop, 40 bar max hold, no
target, no trail. Shorts deferred (crypto spot adapter is long/flat, and
the -2/-4 short legs were deferred at design time anyway). The +2 half /
+4 full SIZING ladder is a deployment-layer mapping: the engine sizes
binary, so it is tested here as the gate2-vs-gate4 comparison, not as
fractional positions.

## Decision rule (pre-stated)
- A genome PASSES only on 5/5 split survival. Mixed splits are not a pass.
- mhv_gate2 is INTERESTING only if it PASSES and its 5-split median SR
  exceeds BOTH parents' medians on this ground. Beating one parent = the
  other parent was doing the work = nothing.
- Sizing ladder is JUSTIFIED for deployment only if gate2 is INTERESTING
  and gate4's median SR >= gate2's (conviction monotone). gate4 passing
  while gate2 fails = a thin-trade-count fluke, not a result.
- Anything else: family CLOSED on this ground. Expected outcome, stated
  now: skeptical -- the vote is time-series momentum in a hat, and TS-mom
  entries start behind costs. The run exists for a ledgered answer.
- STOP: no threshold edits (1.5/3.5, dead zone 0.5, windows 7/14/30/60),
  no added variants, no re-runs with new seeds after any holdout look.
  Results logged to MECHANISM_MAP.md win or lose.

## Order
1. Sign below. 2. git commit. 3. Run the 4 commands. 4. Log + commit.

Signed: __________
"""


def sub_once(path, old, new, label):
    s = open(path).read()
    if new in s:
        print(f"  = {path}: {label} (already)")
        return
    n = s.count(old)
    if n != 1:
        sys.exit(f"REFUSING {path}: anchor for '{label}' found {n} times "
                 "(need exactly 1). Diff it; nothing further applied.")
    open(path, "w").write(s.replace(old, new, 1))
    print(f"  ~ {path}: {label}")


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
    # 1) atoms.py: the block function + registry entry
    sub_once("primordial/atoms.py",
             "\n\nBLOCKS = {",
             ATOM_FUNC + "\nBLOCKS = {",
             "mh_vote atom block")
    sub_once("primordial/atoms.py",
             '    "volume": _volume,\n',
             '    "volume": _volume,\n    "multi_horizon": _multi_horizon,\n',
             "BLOCKS registry entry")
    # 2) universe.py: crypto adapter gains the block (crypto only; the
    #    "crypto" line disambiguates from the identical crypto_perp tuple)
    sub_once("primordial/universe.py",
             '        "crypto", always_open=True,\n'
             '        atom_blocks=("trend", "oscillators", "vol_structure", '
             '"distribution",\n'
             '                     "persistence", "microstructure", '
             '"volume"),',
             '        "crypto", always_open=True,\n'
             '        atom_blocks=("trend", "oscillators", "vol_structure", '
             '"distribution",\n'
             '                     "persistence", "microstructure", '
             '"volume", "multi_horizon"),',
             "crypto adapter block list")
    # 3) manifest, genomes, prereg
    write_new("manifests/coinbase_daily.yaml", MANIFEST, "crypto 1d ground")
    for name, g in GENOMES.items():
        write_new(os.path.join("genomes", name + ".json"),
                  json.dumps(g, indent=1) + "\n", "frozen genome")
    write_new("MH_VOTE_PREREG.md", PREREG, "prereg skeleton")
    print("\nNext:")
    print("  # sign MH_VOTE_PREREG.md, then:")
    print('  git add -A && git commit -m "mh_vote _idea family registered '
          '(atom + crypto 1d ground + 4 genomes, signed)"')
    for name in GENOMES:
        print("  python3.10 scripts/validate_genome.py --manifest "
              f"manifests/coinbase_daily.yaml --genome genomes/{name}.json"
              " --splits 5")


if __name__ == "__main__":
    main()
