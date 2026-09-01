#!/usr/bin/env python3
"""apply_target_pct.py -- add a HAND-ONLY `target_pct` gene: exit at the
first bar (after entry) whose high reaches entry*(1+pct) (long) / low reaches
entry*(1-pct) (short). Fills at the level, stop checked first, never on the
entry bar -- same rules as target_r. Default 0.0 = off; every existing genome
JSON and every seeded run reproduces bit-for-bit. Evolution never sets it
(not in mutate/crossover), same policy as entry_style="session".

Touches: primordial/genome.py, primordial/runtime.py, scripts/make_genome.py,
tests/test_target_pct.py (new).

    python3.10 apply_target_pct.py && python3.10 -m pytest tests -q

Idempotent: each edit checks for its own marker first; a missing anchor
aborts with nothing further applied.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(ROOT, "primordial", "genome.py")):
    sys.exit("run from the primordial repo root")

EDITS = [
    # ---- genome.py: gene ---------------------------------------------------
    ("primordial/genome.py",
     "    anchor_ema: int = 0\n\n    # ---- serialization",
     "    anchor_ema: int = 0\n"
     "    # fixed-fraction take-profit (monthlow family, 2026-09-01): 0 = off;\n"
     "    # else exit at the level entry*(1 +/- pct). HAND-ONLY gene -- not in\n"
     "    # mutate/crossover, so evolution cannot reach it and old seeds\n"
     "    # reproduce. Checked after the stop, never on the entry bar.\n"
     "    target_pct: float = 0.0\n\n    # ---- serialization",
     "target_pct: float = 0.0"),
    # ---- genome.py: describe ----------------------------------------------
    ("primordial/genome.py",
     "        if self.anchor_ema:\n            d += f\" anchor{self.anchor_ema}\"\n        return d",
     "        if self.anchor_ema:\n            d += f\" anchor{self.anchor_ema}\"\n"
     "        if getattr(self, \"target_pct\", 0.0) > 0:\n"
     "            d += f\" tgt{self.target_pct:.1%}\"\n        return d",
     "d += f\" tgt{self.target_pct:.1%}\""),
    # ---- genome.py: backtest exit -----------------------------------------
    ("primordial/genome.py",
     "                exit_px = pos[\"target\"]\n"
     "            # time stop / max hold\n",
     "                exit_px = pos[\"target\"]\n"
     "            # fixed-fraction target: same discipline as target_r (stop\n"
     "            # first, level fill, never the entry bar); pos[\"entry\"] is\n"
     "            # the adverse-adjusted fill, so pct is measured from what was\n"
     "            # actually paid.\n"
     "            elif g.target_pct > 0 and (\n"
     "                    (side == \"long\" and\n"
     "                     h[t] >= pos[\"entry\"] * (1 + g.target_pct)) or\n"
     "                    (side == \"short\" and\n"
     "                     l[t] <= pos[\"entry\"] * (1 - g.target_pct))):\n"
     "                exit_px = pos[\"entry\"] * (1 + sgn * g.target_pct)\n"
     "            # time stop / max hold\n",
     "elif g.target_pct > 0 and ("),
    # ---- runtime.py: executor must know the exit ---------------------------
    ("primordial/runtime.py",
     "                \"anchor_ema\": getattr(g, \"anchor_ema\", 0)}",
     "                \"anchor_ema\": getattr(g, \"anchor_ema\", 0),\n"
     "                # 0 = off; else exit at entry*(1 +/- pct), level fill.\n"
     "                \"target_pct\": getattr(g, \"target_pct\", 0.0)}",
     "\"target_pct\": getattr(g, \"target_pct\", 0.0)"),
    # ---- make_genome.py: flag ---------------------------------------------
    ("scripts/make_genome.py",
     "    p.add_argument(\"--target\", type=float, default=0.0, help=\"R multiple\")\n",
     "    p.add_argument(\"--target\", type=float, default=0.0, help=\"R multiple\")\n"
     "    p.add_argument(\"--target-pct\", type=float, default=0.0,\n"
     "                   help=\"fixed-fraction take-profit, e.g. 0.05 (0 = off)\")\n",
     "--target-pct"),
    ("scripts/make_genome.py",
     "               max_hold=a.max_hold, anchor_ema=a.anchor)\n",
     "               max_hold=a.max_hold, anchor_ema=a.anchor,\n"
     "               target_pct=a.target_pct)\n",
     "target_pct=a.target_pct"),
]

TEST = '''"""
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
'''


def apply(path, old, new, marker):
    p = os.path.join(ROOT, path)
    src = open(p).read()
    if marker in src:
        print(f"  = {path}: {marker[:40]!r} (already)")
        return
    if src.count(old) != 1:
        sys.exit(f"REFUSING {path}: anchor found {src.count(old)}x (need 1). "
                 "Nothing further applied.")
    open(p, "w").write(src.replace(old, new, 1))
    print(f"  + {path}: {marker[:40]!r}")


def main():
    for path, old, new, marker in EDITS:
        apply(path, old, new, marker)
    tp = os.path.join(ROOT, "tests", "test_target_pct.py")
    if os.path.exists(tp):
        if open(tp).read() == TEST:
            print("  = tests/test_target_pct.py (already)")
        else:
            sys.exit("REFUSING tests/test_target_pct.py: exists, differs.")
    else:
        open(tp, "w").write(TEST)
        print("  + tests/test_target_pct.py")
    print("\nNext: python3.10 -m pytest tests -q   (expect 35 + 4 passing)")


if __name__ == "__main__":
    main()
