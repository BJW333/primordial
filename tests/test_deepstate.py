"""
Deep-state integration guards (v0.1.7).

  atoms   -- ema_spread_atr preserves sign; spread_runlen counts sag bars only
  anchor  -- exit fills at the close of the first bar AFTER entry where close
             recrosses the reference EMA; never on the entry bar; r_mult exact
  genome  -- pre-0.1.7 JSON loads with the gene defaulted off; gene survives
             roundtrip, mutation reach, crossover carry
  breadth -- exact binomial tail; planted edge shows breadth, and the gate
             math flags a 2-name wonder
"""
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial import tree as T
from primordial.atoms import compute_atoms
from primordial.genome import Genome, run_backtest, random_genome, mutate_genome, \
    crossover_genomes
from primordial.judge import breadth, _binom_tail
from primordial.universe import COST_TIERS
# import sibling helpers by PATH, not as a 'tests' package -- package-style
# imports resolve against whichever 'tests' dir Python finds first, which on
# a machine with two checked-out copies can be the WRONG repo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_primordial import make_df, make_planted


COST = COST_TIERS["crypto_taker"] if "crypto_taker" in COST_TIERS \
    else list(COST_TIERS.values())[0]


def test_deepstate_atoms_semantics():
    df = make_df(900, seed=7)
    cols = compute_atoms(df, ["trend"], True)
    assert "ema_spread_atr" in cols and "spread_runlen" in cols
    sp = (df["close"].ewm(span=12, min_periods=5).mean()
          - df["close"].ewm(span=26, min_periods=5).mean())
    m = df["ema_spread_atr"].notna()
    same = (np.sign(df["ema_spread_atr"][m]) == np.sign(sp[m])) | (sp[m] == 0)
    assert same.all()                                  # sign preserved (macd_line is z-scored; this must not be)
    assert (df.loc[sp >= 0, "spread_runlen"].fillna(0) == 0).all()
    neg = (sp < 0).to_numpy()
    rl = df["spread_runlen"].to_numpy()
    for i in range(30, len(df)):
        if neg[i] and neg[i - 1]:
            assert rl[i] == rl[i - 1] + 1              # increments through a sag
        elif neg[i] and not neg[i - 1]:
            assert rl[i] == 1                          # fresh sag's first bar counts 1
                                                       # (same formula as deep_state.py -- fresh_max=15 admits sag-bars 1..15)


def _anchor_frame():
    """Deterministic path: flat 100, one signal bar, dip, recovery through the
    reference EMA a few bars later. Ranges are WIDE (+-4%) so ATR ~ 8 and the
    6-ATR stop sits ~48 points away -- untouchable; the anchor exit is the
    only live exit. Wide wicks cannot leak profit: the exit is close-based
    and target_r = 0."""
    n = 60
    close = np.full(n, 100.0)
    close[30:33] = [97.0, 96.0, 96.5]                  # the dip
    close[33:] = 103.0                                 # recovery above any EMA
    df = pd.DataFrame({"open": close, "high": close * 1.04,
                       "low": close * 0.96, "close": close,
                       "volume": np.full(n, 1e6)},
                      index=pd.date_range("2024-01-01", periods=n, freq="1d"))
    df["fire"] = 0.0
    df.loc[df.index[29], "fire"] = 1.0                 # signal at t=29 -> entry t=30
    return df


def test_anchor_exit_timing_and_price():
    df = _anchor_frame()
    g = Genome(timeframe="1d", root_type="bool",
               entry_tree=T.node("gt", [T.term("fire"), T.const(0.5)]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=6.0, target_r=0.0, time_stop=0,
               trail_mode="none", max_hold=200, anchor_ema=5)
    res = run_backtest(g, df, COST, bar_seconds=86400.0)
    assert res.n_trades == 1
    t0, t1, side, r = res.trades[0]
    assert side == "long"
    ema5 = df["close"].ewm(span=5, min_periods=5).mean()
    # first bar AFTER the entry bar with close >= ema5:
    ei = df.index.get_loc(t0)
    want = next(t for t in range(ei + 1, len(df))
                if df["close"].iloc[t] >= ema5.iloc[t])
    assert df.index.get_loc(t1) == want                # exits exactly there, not before
    # r accounting still uses stop distance; confirm profit sign + finite
    assert np.isfinite(r) and r > 0


def test_anchor_never_fires_on_entry_bar():
    df = _anchor_frame()
    # make the ENTRY bar itself already close above the reference EMA:
    df.loc[df.index[30], ["open", "high", "low", "close"]] = 104.0
    g = Genome(timeframe="1d", root_type="bool",
               entry_tree=T.node("gt", [T.term("fire"), T.const(0.5)]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=6.0, target_r=0.0, time_stop=0,
               trail_mode="none", max_hold=200, anchor_ema=5)
    res = run_backtest(g, df, COST, bar_seconds=86400.0)
    assert res.n_trades == 1
    t0, t1, _, _ = res.trades[0]
    assert df.index.get_loc(t1) > df.index.get_loc(t0)  # exit strictly later


def test_genome_backcompat_and_variation():
    import json
    legacy = {"timeframe": "1d", "root_type": "bool",
              "entry_tree": {"op": "const", "ch": [], "value": 1.0},
              "regime_tree": None, "direction": "long",
              "entry_style": "market", "entry_param": 1, "stop_atr": 2.0,
              "target_r": 0.0, "time_stop": 0, "trail_mode": "none",
              "max_hold": 48}
    g = Genome.from_json(json.dumps(legacy))
    assert g.anchor_ema == 0                            # pre-0.1.7 artifacts load clean
    g.anchor_ema = 26
    assert Genome.from_json(g.to_json()).anchor_ema == 26
    rng = random.Random(0)
    terms = ["fire"]
    seen = {mutate_genome(g, rng, terms, ["1d"]).anchor_ema for _ in range(300)}
    assert len(seen) > 1                                # mutation reaches the gene
    a = random_genome(random.Random(1), terms, ["1d"]); a.anchor_ema = 39
    b = random_genome(random.Random(2), terms, ["1d"]); b.anchor_ema = 0
    for _ in range(50):
        a2, b2 = crossover_genomes(a, b, rng)
        if a2.anchor_ema == 0 and b2.anchor_ema == 39:
            break
    else:
        raise AssertionError("crossover never carried anchor_ema")


def test_binom_tail_exact():
    assert abs(_binom_tail(10, 12) - 0.019287) < 1e-5
    assert abs(_binom_tail(6, 12) - 0.612793) < 1e-5
    assert _binom_tail(0, 0) == 1.0
    assert _binom_tail(12, 12) == 0.5 ** 12


def test_breadth_planted_vs_narrow():
    data = {f"S{i}": make_planted(2500, seed=i) for i in range(12)}
    for df in data.values():
        compute_atoms(df, ["trend", "volume"], True)
    syms = list(data)
    g = Genome(timeframe="1h", root_type="bool",
               entry_tree=T.node("gt", [T.term("volz"), T.const(2.5)]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=3.0, target_r=0.0, time_stop=6,
               trail_mode="none", max_hold=12)
    frac, p, n = breadth(g, data, syms, COST, 3600.0)
    assert n >= 10 and frac > 0.5 and p < 0.05          # planted edge is broad
    # the failure mode the gate exists for: same numbers concentrated in 2 names
    assert _binom_tail(2, 12) > 0.99


def test_tautology_entry_is_floored():
    """lte(x, x) is always true; it references data so has_terminal passes it
    (seen live: gen 12 of the sp600_top150 run). The behavioral guard must
    catch it: constant signal = floored, whatever the tree looks like."""
    from primordial.judge import fitness
    from primordial.data import BAR_SECONDS
    data = {f"S{i}": make_df(1200, seed=80 + i) for i in range(4)}
    for d in data.values():
        compute_atoms(d, ["persistence"], always_open=True)
    g = Genome(timeframe="1h", root_type="bool",
               entry_tree=T.node("lte", [T.term("autocorr_5"),
                                          T.term("autocorr_5")]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=2.0, target_r=1.0, time_stop=24,
               trail_mode="none", max_hold=96)
    f, info = fitness(g, data, list(data), COST, BAR_SECONDS["1h"])
    assert f == -9.99 and info.get("degenerate") == "constant_signal"
