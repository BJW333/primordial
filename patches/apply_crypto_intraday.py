#!/usr/bin/env python3
"""apply_crypto_intraday.py -- crypto intraday ground (hold <= 1 day). Idempotent.

What this adds, and why each piece exists:

  1. max_hold_hours (Manifest, default 0 = off): wall-clock cap on holding
     time, enforced at the GENE level. random/mutated genomes can only draw
     max_hold / time_stop values whose bar-count at the genome's timeframe
     fits inside the cap; a timeframe mutation re-clamps. With the default 0
     the choice lists are byte-identical to before, so every seeded equity
     run and all 26 pinned tests reproduce exactly.
  2. exclude_atoms (Manifest, default []): drop named terminals from the
     evolved set. Unknown names RAISE (mult-vs-mul lesson: a typo'd exclusion
     must be an error, not a silent no-op). First use: hod/dow/dom/
     dist_to_weekend on crypto -- published seasonality = decayed by mandate.
  3. coinbase_liquid_v2 preset: MATIC-USD -> POL-USD, RNDR-USD -> RENDER-USD,
     SAND/MANA dropped. A NEW preset, because editing coinbase_liquid in
     place would silently re-fingerprint every manifest that references it
     and orphan their ledger history. Never mutate a live preset.
  4. manifests/crypto_intraday_15m.yaml + CRYPTO_INTRADAY_PREREG.md skeleton.
  5. tests/test_crypto_intraday.py: cap invariants under random construction
     and mutation, legacy-stream guard, exclusion filter unit tests,
     long-only guard, and a planted-behavior backtest control (always-true
     entry must time out at exactly max_hold bars).

Run from repo root:  python3.10 apply_crypto_intraday.py
Then:                python3.10 -m pytest tests/ -q        (must be all green)
"""
import os
import sys

if not (os.path.exists("primordial/genome.py") and os.path.exists("MECHANISM_MAP.md")):
    sys.exit("run from the primordial repo root.")

n_changed = 0


def patch(path, old, new, label):
    global n_changed
    s = open(path).read()
    if new in s:
        print(f"  = {path}: {label} already applied")
        return
    if old not in s:
        sys.exit(f"ANCHOR MISSING in {path} ({label}) -- repo has drifted, "
                 "refusing to guess. Nothing partially applied beyond "
                 f"{n_changed} prior hunks; re-run after reconciling.")
    if s.count(old) != 1:
        sys.exit(f"ANCHOR NOT UNIQUE in {path} ({label}) -- refusing.")
    open(path, "w").write(s.replace(old, new))
    n_changed += 1
    print(f"  + {path}: {label}")


def write_new(path, content, label):
    global n_changed
    if os.path.exists(path):
        if open(path).read() == content:
            print(f"  = {path}: already present")
            return
        sys.exit(f"{path} exists with different content -- refusing to overwrite.")
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    open(path, "w").write(content)
    n_changed += 1
    print(f"  + {path}: {label}")


# ---------------------------------------------------------------- genome.py
patch("primordial/genome.py",
"""OPEN_SAFE_TERMS = {"gap_atr_open", "open"}""",
"""OPEN_SAFE_TERMS = {"gap_atr_open", "open"}

# legacy gene-choice ladders -- referenced verbatim when no hold cap is set,
# so pre-cap seeded runs reproduce bit-for-bit
HOLD_CHOICES = [48, 96, 192, 500]
TS_CHOICES = [0, 12, 24, 48, 96]


def hold_cap_bars(timeframe: str, max_hold_hours: float) -> int:
    \"\"\"Wall-clock hold cap -> bar count at this timeframe. 0 = uncapped.
    Wall-clock (not bars) because genomes mutate across timeframes: '1 day'
    must mean 96 bars at 15m and 24 bars at 1h, not one frozen number.\"\"\"
    if not max_hold_hours:
        return 0
    from .data import BAR_SECONDS
    return max(1, int(max_hold_hours * 3600.0 / BAR_SECONDS[timeframe]))


def _hold_choices(cap: int) -> list:
    if not cap:
        return HOLD_CHOICES
    return sorted({max(1, cap // 8), max(1, cap // 4), max(1, cap // 2), cap})


def _ts_choices(cap: int) -> list:
    if not cap:
        return TS_CHOICES
    return sorted({0, max(1, cap // 4), max(1, cap // 2), cap})""",
"hold-cap helpers")

patch("primordial/genome.py",
"""def random_genome(rng: random.Random, terminals, allowed_timeframes,
                  root_type="bool", allow_short=True) -> Genome:
    direction = rng.choice(["long"] + (["short"] if allow_short else []))
    return Genome(
        timeframe=rng.choice(allowed_timeframes),""",
"""def random_genome(rng: random.Random, terminals, allowed_timeframes,
                  root_type="bool", allow_short=True,
                  max_hold_hours=0.0) -> Genome:
    direction = rng.choice(["long"] + (["short"] if allow_short else []))
    tf = rng.choice(allowed_timeframes)      # same rng call order as before
    cap = hold_cap_bars(tf, max_hold_hours)
    return Genome(
        timeframe=tf,""",
"random_genome cap plumbing")

patch("primordial/genome.py",
"""        time_stop=rng.choice([0, 12, 24, 48, 96]),
        trail_mode=rng.choice(TRAIL_MODES),
        max_hold=rng.choice([48, 96, 192, 500]),""",
"""        time_stop=rng.choice(_ts_choices(cap)),
        trail_mode=rng.choice(TRAIL_MODES),
        max_hold=rng.choice(_hold_choices(cap)),""",
"random_genome capped choices")

patch("primordial/genome.py",
"""def mutate_genome(g: Genome, rng: random.Random, terminals,
                  allowed_timeframes) -> Genome:
    g = copy.deepcopy(g)
    r = rng.random()""",
"""def mutate_genome(g: Genome, rng: random.Random, terminals,
                  allowed_timeframes, max_hold_hours=0.0) -> Genome:
    g = copy.deepcopy(g)
    r = rng.random()""",
"mutate_genome signature")

patch("primordial/genome.py",
"""        tfs = [t for t in TIMEFRAMES if t in allowed_timeframes]
        i = tfs.index(g.timeframe)
        g.timeframe = tfs[max(0, min(len(tfs) - 1, i + rng.choice([-1, 1])))]""",
"""        tfs = [t for t in TIMEFRAMES if t in allowed_timeframes]
        i = tfs.index(g.timeframe)
        g.timeframe = tfs[max(0, min(len(tfs) - 1, i + rng.choice([-1, 1])))]
        cap = hold_cap_bars(g.timeframe, max_hold_hours)
        if cap:                       # coarser tf can push bar-holds past the
            g.max_hold = min(g.max_hold, cap)       # wall-clock cap: re-clamp
            g.time_stop = min(g.time_stop, cap)""",
"tf-shift re-clamp")

patch("primordial/genome.py",
"""        elif which == "ts":   g.time_stop = rng.choice([0, 12, 24, 48, 96])
        elif which == "trail": g.trail_mode = rng.choice(TRAIL_MODES)
        elif which == "hold": g.max_hold = rng.choice([48, 96, 192, 500])""",
"""        elif which == "ts":   g.time_stop = rng.choice(
            _ts_choices(hold_cap_bars(g.timeframe, max_hold_hours)))
        elif which == "trail": g.trail_mode = rng.choice(TRAIL_MODES)
        elif which == "hold": g.max_hold = rng.choice(
            _hold_choices(hold_cap_bars(g.timeframe, max_hold_hours)))""",
"mutate capped choices")

# ---------------------------------------------------------------- engine.py
patch("primordial/engine.py",
"""class Island:
    def __init__(self, terminals, allowed_timeframes, rng, pop_size,
                 root_type="bool", allow_short=True):
        self.terminals = terminals
        self.tfs = allowed_timeframes
        self.rng = rng
        self.pop = [random_genome(rng, terminals, allowed_timeframes,
                                  root_type=root_type, allow_short=allow_short)
                    for _ in range(pop_size)]""",
"""class Island:
    def __init__(self, terminals, allowed_timeframes, rng, pop_size,
                 root_type="bool", allow_short=True, max_hold_hours=0.0):
        self.terminals = terminals
        self.tfs = allowed_timeframes
        self.rng = rng
        self.max_hold_hours = max_hold_hours
        self.pop = [random_genome(rng, terminals, allowed_timeframes,
                                  root_type=root_type, allow_short=allow_short,
                                  max_hold_hours=max_hold_hours)
                    for _ in range(pop_size)]""",
"Island cap")

patch("primordial/engine.py",
"""           root_type="bool", allow_short=True, novelty_corr=0.95,
           verbose=True, on_generation=None):""",
"""           root_type="bool", allow_short=True, novelty_corr=0.95,
           verbose=True, on_generation=None, max_hold_hours=0.0):""",
"evolve signature")

patch("primordial/engine.py",
"""        islands.append(Island(terms, allowed_timeframes, irng, pop_size,
                              root_type, allow_short))""",
"""        islands.append(Island(terms, allowed_timeframes, irng, pop_size,
                              root_type, allow_short, max_hold_hours))""",
"island construction cap")

patch("primordial/engine.py",
"""                newpop.append(mutate_genome(c1, isl.rng, isl.terminals, isl.tfs))
                if len(newpop) < len(isl.pop):
                    newpop.append(mutate_genome(c2, isl.rng, isl.terminals, isl.tfs))""",
"""                newpop.append(mutate_genome(c1, isl.rng, isl.terminals,
                                            isl.tfs, isl.max_hold_hours))
                if len(newpop) < len(isl.pop):
                    newpop.append(mutate_genome(c2, isl.rng, isl.terminals,
                                                isl.tfs, isl.max_hold_hours))""",
"mutation cap")

patch("primordial/engine.py",
"""                    isl.pop[-i] = random_genome(
                        isl.rng, isl.terminals, isl.tfs)""",
"""                    isl.pop[-i] = random_genome(
                        isl.rng, isl.terminals, isl.tfs,
                        max_hold_hours=isl.max_hold_hours)""",
"immigrant cap")

# -------------------------------------------------------------- universe.py
patch("primordial/universe.py",
"""    context_symbols: list = dataclasses.field(default_factory=list)
    seed: int = 42""",
"""    context_symbols: list = dataclasses.field(default_factory=list)
    # wall-clock cap on holding time, enforced in the gene pool (0 = off).
    # NOT part of fingerprint(): like `engines`, it constrains the search,
    # not which holdout is being burned -- capped and uncapped searches on
    # the same ground share one ledger, which is the conservative accounting.
    max_hold_hours: float = 0.0
    # terminal names removed from the evolved set. Unknown names raise.
    exclude_atoms: list = dataclasses.field(default_factory=list)
    seed: int = 42""",
"Manifest fields")

patch("primordial/universe.py",
"""        "ALGO-USD", "SAND-USD", "MANA-USD", "GRT-USD", "IMX-USD", "RNDR-USD",
    ],
}""",
"""        "ALGO-USD", "SAND-USD", "MANA-USD", "GRT-USD", "IMX-USD", "RNDR-USD",
    ],
    # v2 2026-08: MATIC-USD -> POL-USD (Coinbase migrated Oct 2024),
    # RNDR-USD -> RENDER-USD; SAND/MANA dropped (illiquid). A NEW key on
    # purpose -- editing coinbase_liquid in place would silently change the
    # fingerprint of every manifest referencing it and orphan their ledger
    # history. NEVER mutate a live preset; version it. Ticker drift is
    # self-auditing: anything wrong here lands in run.json symbols_missing.
    "coinbase_liquid_v2": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD",
        "AVAX-USD", "LINK-USD", "DOT-USD", "POL-USD", "LTC-USD", "BCH-USD",
        "UNI-USD", "ATOM-USD", "XLM-USD", "ETC-USD", "FIL-USD", "APT-USD",
        "ARB-USD", "OP-USD", "NEAR-USD", "INJ-USD", "AAVE-USD", "MKR-USD",
        "ALGO-USD", "GRT-USD", "IMX-USD", "RENDER-USD",
    ],
}""",
"coinbase_liquid_v2 preset")

# -------------------------------------------------------------- pipeline.py
patch("primordial/pipeline.py",
"""def _prepare(mtf, adapter, train_syms, hold_syms, motif_defs,""",
"""def apply_exclusions(kept: list, all_terminals: list, exclude: list) -> list:
    \"\"\"Filter kept terminals by name. Validates against the PRE-dedupe list
    so an atom that was merged away doesn't read as a typo; a name that never
    existed raises (silent no-op exclusions are how the mult/mul class of bug
    ships). Returns a new list.\"\"\"
    if not exclude:
        return kept
    unknown = [a for a in exclude if a not in all_terminals]
    if unknown:
        raise ValueError(f"exclude_atoms not in terminal set: {unknown} -- "
                         f"check spelling against atoms.py")
    ex = set(exclude)
    return [k for k in kept if k not in ex]


def _prepare(mtf, adapter, train_syms, hold_syms, motif_defs,""",
"apply_exclusions helper")

patch("primordial/pipeline.py",
"""    kept, merges = A.dedupe(mtf_train[tfs[0]], terminals)""",
"""    kept, merges = A.dedupe(mtf_train[tfs[0]], terminals)
    kept = apply_exclusions(kept, terminals, mf.exclude_atoms)""",
"exclusion applied post-dedupe")

patch("primordial/pipeline.py",
"""    best, hall, n_eval = evolve(
        mtf_train, split.train_syms, mf.cost, kept, tfs,
        generations=generations, pop_size=pop_size, n_islands=n_islands,
        seed=seed, allow_short=allow_short, verbose=verbose)""",
"""    best, hall, n_eval = evolve(
        mtf_train, split.train_syms, mf.cost, kept, tfs,
        generations=generations, pop_size=pop_size, n_islands=n_islands,
        seed=seed, allow_short=allow_short, verbose=verbose,
        max_hold_hours=mf.max_hold_hours)""",
"evolve receives cap")

# --------------------------------------------------------------- manifest
write_new("manifests/crypto_intraday_15m.yaml",
"""# Crypto intraday, hold <= 1 day. Coinbase spot = LONG/FLAT ONLY (adapter
# enforces shortable=False); at these hold times the fee model IS the gate.
#
# cost_tier crypto_spot_taker = 35 bps/side (5 spread + 25 commission + 5
# slip) ~= Advanced Trade taker in the $100K-1M 30-day-volume band. CONFIRM
# against your actual tier before the registered run; edit COST_TIERS, not
# results. The maker tier + entry_style=limit (models non-fills) is the
# batch-2 lever if taker costs eat everything -- do not switch after looking.
#
# Session atoms (hod/dow/dom/weekend): published crypto seasonality =
# decayed by mandate -- and they are STRUCTURALLY absent: pipeline passes
# adapter.atom_blocks, which never includes "session" (valid_blocks() is
# unwired). Pinned by test_session_atoms_structurally_absent_for_crypto;
# exclude_atoms stays empty until something actually needs excluding.
#
# 4h stays in allowed_timeframes: under the 24h cap a 4h genome gets <= 6-bar
# holds -- cramped but legal; evolution decides if that corner is worth it.
#
# Cache build: ~160k 15m bars/name -> ~3 min/name at Coinbase's public rate
# limit, ~1.5h total, one-time. `source: omnifeed` uses your local cache
# instead if it is already warm.
name: crypto_intraday_15m
asset_class: crypto
source: coinbase                  # or omnifeed
symbols: coinbase_liquid_v2
base_timeframe: 15m
allowed_timeframes: [15m, 1h, 4h]
start: "2021-06-01"
end: "2026-08-01"
cost_tier: crypto_spot_taker
holdout_name_frac: 0.4
holdout_time_frac: 0.3
embargo_bars: 200                 # ~2 days at 15m >= 2x the max hold
engines: [timing, cross_sectional]
n_motifs: 6
motif_scales: [20, 40]
context_symbols: []               # no external streams; BTC/ETH are members
max_hold_hours: 24
exclude_atoms: []
""",
"crypto intraday manifest")

# ------------------------------------------------------------------ prereg
write_new("CRYPTO_INTRADAY_PREREG.md",
"""# Crypto intraday search -- pre-registration (2026-08-__)

WHAT: let primordial evolve on crypto_intraday_15m. Long/flat spot,
hold <= 24h (gene-level cap), taker costs. No rule handed to it.

WHY THIS GROUND: 24/7 venue, no borrow, no session boundary artifacts, and
a fee wall that mechanically kills the entire high-turnover end of the
space -- whatever survives 70 bps round trips at sub-day holds has to be
structural. New fingerprint; the bar comes from this search's own trial
count.

KNOWN-DECAYED, DO NOT RE-ACCEPT (published; excluded or audited):
  time-of-day / US-hours drift        session atoms structurally absent
                                      (pipeline never computes them; pinned
                                      by regression test)
  weekend effect                      same
  simple momentum / TSMOM on majors   published to death; a finalist whose
                                      entry is a trend-sign re-description
                                      is a re-hit, not a finding
  funding-rate carry                  spot ground, stream not present

REGISTERED WATCH-ITEM -- BTC-beta convergence: the sp400 batch-1 failure
mode (search flees per-name structure for market timing) maps here as
finalists collapsing onto BTC-proxy signals expressed through any liquid
alt. If >= half the finalists' entries read as majors-timing, batch 2 is
pre-committed as: drop BTC-USD/ETH-USD from members, engines
[cross_sectional] only. Registered NOW so it is a contingency, not mining.

FEE TIER: crypto_spot_taker assumes 25 bps taker commission. Actual
30-day-volume tier confirmed as: __________ (fill BEFORE the run; if it
differs, edit COST_TIERS first and note it here).

PASS = whatever the gauntlet says. No threshold edits, no variant mining
after seeing results. Zero survivors is the expected outcome and is a
complete result.

REGISTERED STOP: taker batch + (if triggered) the pre-committed batch 2
are the whole campaign on this fingerprint. 0 survivors from both closes
intraday crypto spot; next crypto effort would be a different ground
(perps with funding as carry), separately registered.

Signed: __________
""",
"prereg skeleton")

# ------------------------------------------------------------------- tests
write_new("tests/test_crypto_intraday.py",
'''"""Crypto intraday ground: hold-cap invariants, exclusion filter,
long-only guard, and a planted-behavior time-exit control."""
import random

import numpy as np
import pandas as pd
import pytest

from primordial import tree as T
from primordial.data import BAR_SECONDS
from primordial.genome import (Genome, random_genome, mutate_genome,
                               hold_cap_bars, HOLD_CHOICES, TS_CHOICES,
                               run_backtest)
from primordial.pipeline import apply_exclusions
from primordial.universe import ADAPTERS, COST_TIERS, Manifest, PRESETS

TERMS = ["close", "volz", "rsi_fast"] if True else []
TFS = ["15m", "1h", "4h"]
CAP_H = 24.0


def _ok(g):
    cap = hold_cap_bars(g.timeframe, CAP_H)
    assert g.max_hold * BAR_SECONDS[g.timeframe] <= CAP_H * 3600 + 1e-9
    assert g.max_hold <= cap and g.max_hold >= 1
    assert g.time_stop <= cap


def test_hold_cap_random_construction():
    rng = random.Random(7)
    for _ in range(300):
        _ok(random_genome(rng, TERMS, TFS, max_hold_hours=CAP_H))


def test_hold_cap_survives_mutation_and_tf_shifts():
    rng = random.Random(11)
    g = random_genome(rng, TERMS, TFS, max_hold_hours=CAP_H)
    for _ in range(500):
        g = mutate_genome(g, rng, TERMS, TFS, max_hold_hours=CAP_H)
        _ok(g)


def test_cap_off_reproduces_legacy_choice_sets():
    """cap=0 must sample from the exact pre-patch ladders -- this is the
    guard that seeded equity runs did not shift."""
    rng = random.Random(3)
    for _ in range(200):
        g = random_genome(rng, TERMS, TFS)          # default: no cap
        assert g.max_hold in HOLD_CHOICES
        assert g.time_stop in TS_CHOICES


def test_hold_cap_bars_wall_clock():
    assert hold_cap_bars("15m", 24) == 96
    assert hold_cap_bars("1h", 24) == 24
    assert hold_cap_bars("4h", 24) == 6
    assert hold_cap_bars("15m", 0) == 0             # off


def test_exclusions_filter_and_raise():
    kept = ["close", "hod", "volz"]
    all_t = ["close", "hod", "volz", "dow"]         # dow deduped away
    assert apply_exclusions(kept, all_t, []) == kept
    assert apply_exclusions(kept, all_t, ["hod", "dow"]) == ["close", "volz"]
    with pytest.raises(ValueError):
        apply_exclusions(kept, all_t, ["hodd"])     # typo must be fatal


def test_crypto_adapter_long_flat_only():
    assert ADAPTERS["crypto"].shortable is False
    rng = random.Random(5)
    for _ in range(200):
        g = random_genome(rng, TERMS, TFS, allow_short=False)
        assert g.direction == "long"


def test_manifest_defaults_off_and_v2_preset():
    assert Manifest.__dataclass_fields__["max_hold_hours"].default == 0.0
    v2 = PRESETS["coinbase_liquid_v2"]
    assert "MATIC-USD" not in v2 and "RNDR-USD" not in v2
    assert "POL-USD" in v2 and "RENDER-USD" in v2
    # live preset untouched -- its fingerprints must not move
    assert "MATIC-USD" in PRESETS["coinbase_liquid"]


def _flat_24_7(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")   # no gaps
    c = 100 + np.cumsum(rng.normal(0, 0.01, n))                  # tiny drift
    o = np.roll(c, 1); o[0] = c[0]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.02,
                         "low": np.minimum(o, c) - 0.02, "close": c,
                         "volume": np.full(n, 1e6)}, index=idx)


def test_planted_time_exit_at_max_hold():
    """Planted behavior: always-true entry, no stop/target/anchor in reach ->
    every trade must exit on the time axis at <= max_hold bars, and the
    always-true re-entry makes the modal duration exactly max_hold."""
    g = Genome(timeframe="15m", root_type="bool",
               entry_tree=T.node("gt", [T.term("close"), T.const(0.0)]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=50.0, target_r=0.0, time_stop=0,
               trail_mode="none", max_hold=96, anchor_ema=0)
    res = run_backtest(g, _flat_24_7(), COST_TIERS["crypto_spot_taker"],
                       bar_seconds=BAR_SECONDS["15m"])
    assert len(res.trades) >= 10
    durs = [(ex - en) / pd.Timedelta(minutes=15)
            for en, ex, side, r in res.trades]
    assert max(durs) <= 96
    assert max(durs) == 96          # the cap is what fired, not something else


def test_session_atoms_structurally_absent_for_crypto():
    """pipeline passes adapter.atom_blocks -- "session" is never in it, so
    hod/dow/dom/weekend cannot reach evolution. If someone wires
    valid_blocks() into _prepare later, this trips and forces the published-
    seasonality decision to be made consciously per-ground."""
    from primordial import atoms as A
    df = _flat_24_7(500)
    cols = A.compute_atoms(df, list(ADAPTERS["crypto"].atom_blocks),
                           ADAPTERS["crypto"].always_open)
    assert not {"hod", "dow", "dom", "dist_to_weekend"} & set(cols)
''',
"crypto intraday tests")

# --------------------------------------------------------------- changelog
CH = """
## crypto intraday ground (apply_crypto_intraday.py)
- Manifest.max_hold_hours: wall-clock hold cap enforced in the gene pool
  (random + mutation + tf-shift re-clamp). Default 0 = byte-identical legacy
  sampling; all prior seeded runs reproduce.
- Manifest.exclude_atoms: named-terminal removal post-dedupe, validated
  against the pre-dedupe list; unknown names raise (anti mult/mul).
- coinbase_liquid_v2 preset (MATIC->POL, RNDR->RENDER, SAND/MANA out) as a
  NEW key; live presets are never edited in place (fingerprint re-keying).
- manifests/crypto_intraday_15m.yaml + CRYPTO_INTRADAY_PREREG.md skeleton.
- tests/test_crypto_intraday.py: 8 tests incl. planted time-exit control.
"""
s = open("CHANGELOG.md").read()
if "apply_crypto_intraday.py" in s:
    print("  = CHANGELOG.md: already recorded")
else:
    open("CHANGELOG.md", "a").write(CH)
    n_changed += 1
    print("  + CHANGELOG.md")

print(f"\n{n_changed} hunk(s) applied." if n_changed else "\nNothing to do.")
print("""
Next:
  python3.10 -m pytest tests/ -q                       # all green required
  # fill FEE TIER + sign CRYPTO_INTRADAY_PREREG.md, commit, THEN:
  python3.10 scripts/fetch_data.py manifests/crypto_intraday_15m.yaml
  python3.10 -m primordial run --manifest manifests/crypto_intraday_15m.yaml \\
      --generations 20 --pop 40 --islands 3 --seed 42
""")
