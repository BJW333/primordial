"""
target_pct gene guards (2026-09-01).

  off     -- default 0.0; legacy JSON loads with it off; a backtest with the
             gene at 0 is byte-identical to one that never had it
  fill    -- exits at exactly entry*(1+pct) on the first bar whose high
             reaches it, never before, never on the entry bar
  stop    -- a bar touching stop AND level resolves as the stop
  reach   -- mutation and crossover NEVER touch it (hand-only gene)
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from primordial import tree as T
from primordial.genome import (Genome, run_backtest, random_genome,
                               mutate_genome, crossover_genomes)
from primordial.universe import COST_TIERS
from test_primordial import make_df, planted_genome, make_planted

COST = COST_TIERS["equity_midcap"]          # adverse 7 bps, no carry


def _frame(path):
    """flat 100 for 30 bars, fire on bar 30, then a chosen close path."""
    n = 30 + len(path)
    close = np.array([100.0] * 30 + list(path))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({"open": close, "high": close * 1.001,
                       "low": close * 0.999, "close": close,
                       "volume": 1e6, "fire": 0.0}, index=idx)
    df.iloc[30, df.columns.get_loc("fire")] = 1.0
    return df


def _g(**kw):
    base = dict(timeframe="1d", root_type="bool",
                entry_tree=T.node("gt", [T.term("fire"), T.const(0.5)]),
                regime_tree=None, direction="long", entry_style="market",
                entry_param=1, stop_atr=99.0, target_r=0.0, time_stop=0,
                trail_mode="none", max_hold=200, anchor_ema=0)
    base.update(kw)
    return Genome(**base)


def test_off_is_identical_and_legacy_loads():
    legacy = json.loads(planted_genome().to_json())
    legacy.pop("target_pct", None)
    g = Genome.from_json(json.dumps(legacy))
    assert g.target_pct == 0.0
    df = make_planted(2000, seed=3)
    from primordial.atoms import compute_atoms
    compute_atoms(df, ["volume"], True)
    a = run_backtest(planted_genome(), df, COST)
    b = run_backtest(g, df, COST)
    assert a.trades == b.trades
    assert a.equity.equals(b.equity)


def test_fills_at_level_first_touch_not_entry_bar():
    # fire bar 30; entry at open of bar 31 (=100). level = 100*(1+adv)*1.05.
    # bar31 high 100.1 (and a planted 120 spike that must NOT count);
    # bar32 high 103.1 < level; bar33 high 106.1 >= level -> exit there.
    path = [100.0, 100.0, 103.0, 106.0, 110.0, 110.0]
    df = _frame(path)
    df.iloc[31, df.columns.get_loc("high")] = 120.0   # entry bar: must NOT count
    g = _g(target_pct=0.05)
    res = run_backtest(g, df, COST, bar_seconds=86400.0)
    assert res.n_trades == 1
    t0, t1, side, r = res.trades[0]
    assert df.index.get_loc(t0) == 31
    assert df.index.get_loc(t1) == 33                # first LATER bar reaching it
    entry = float(df["open"].iloc[31]) * (1 + COST.adverse_frac())
    level = entry * 1.05
    exit_px = level * (1 - COST.adverse_frac())
    want_r = (exit_px - entry) / (99.0 * _atr(df, 30))
    assert abs(r - want_r) < 1e-9


def _atr(df, t):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14, min_periods=5).mean().iloc[t])


def test_stop_beats_level_on_same_bar():
    path = [100.0, 100.0, 100.0]
    df = _frame(path)
    # entry bar can only stop out; keep it clear of the 1-ATR stop
    df.iloc[31, [df.columns.get_loc("high"), df.columns.get_loc("low")]] = 100.0
    # bar 32 spans both the stop and the level
    df.iloc[32, df.columns.get_loc("high")] = 130.0
    df.iloc[32, df.columns.get_loc("low")] = 50.0
    g = _g(target_pct=0.05, stop_atr=1.0)
    res = run_backtest(g, df, COST, bar_seconds=86400.0)
    assert res.n_trades == 1
    assert res.trades[0][3] < 0                       # resolved as the stop


def test_hand_only_gene_unreachable_by_evolution():
    g = _g(target_pct=0.05)
    rng = random.Random(0)
    for _ in range(300):
        assert mutate_genome(g, rng, ["fire"], ["1d"]).target_pct == 0.05
    a = random_genome(random.Random(1), ["fire"], ["1d"]); a.target_pct = 0.05
    b = random_genome(random.Random(2), ["fire"], ["1d"])
    for _ in range(50):
        a2, b2 = crossover_genomes(a, b, rng)
        assert a2.target_pct == 0.05 and b2.target_pct == 0.0
    assert "tgt5.0%" in g.describe()
