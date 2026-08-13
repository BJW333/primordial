"""Crypto intraday ground: hold-cap invariants, exclusion filter,
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
