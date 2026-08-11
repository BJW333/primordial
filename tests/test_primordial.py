"""
The controls that keep the judge honest. If any of these fail, do not trust
any search result until they pass again.

  positive control -- a PLANTED edge must be detectable and score well
  lookahead control -- a cheating signal must produce an absurd Sharpe
                       (proves the harness can see signal when it exists)
  negative behavior -- random genomes on random data lose the costs
  plus unit guards: protected div, gap-through fills, genome/runtime
  round-trip, ledger keying, motif train-only mining, dedupe, splits embargo
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial import tree as T
from primordial.atoms import compute_atoms, dedupe
from primordial.data import _synthetic, resample_ohlcv, BAR_SECONDS
from primordial.genome import Genome, run_backtest, random_genome
from primordial.judge import (NameTimeSplit, Ledger, fold_cells, psr,
                              trade_sharpe, pbo)
from primordial.motifs import mine, add_motif_atoms
from primordial.universe import COST_TIERS, Manifest


def make_df(n=3000, seed=0, freq="1h"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq)
    r = rng.standard_normal(n) * 0.005
    close = 100 * np.exp(np.cumsum(r))
    o = np.roll(close, 1); o[0] = close[0]
    hi = np.maximum(o, close) * (1 + np.abs(rng.standard_normal(n)) * 0.002)
    lo = np.minimum(o, close) * (1 - np.abs(rng.standard_normal(n)) * 0.002)
    v = np.abs(rng.standard_normal(n) + 3) * 1e6
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": close,
                         "volume": v}, index=idx)


def make_planted(n=4000, seed=1):
    """Every ~40 bars: a volume-spike marker bar followed by a genuine +1.5%
    drift over the next 6 bars. Detectable via volz; tradable next-open."""
    df = make_df(n, seed)
    close = df["close"].to_numpy().copy()
    vol = df["volume"].to_numpy().copy()
    rng = np.random.default_rng(seed + 1)
    for i in range(50, n - 10, 40):
        vol[i] = vol[i] * 12
        drift = 0.015 * (1 + 0.2 * rng.standard_normal())
        # the +1.5% accrues over exactly the next 6 bars, then holds
        steps = np.exp(np.cumsum(np.full(6, drift / 6)))
        close[i + 1:i + 7] *= steps
        close[i + 7:] *= steps[-1]
    df["close"] = close
    df["open"] = np.roll(close, 1); df.iloc[0, df.columns.get_loc("open")] = close[0]
    df["high"] = np.maximum(df["open"], df["close"]) * 1.002
    df["low"] = np.minimum(df["open"], df["close"]) * 0.998
    df["volume"] = vol
    return df


def planted_genome():
    return Genome(timeframe="1h", root_type="bool",
                  entry_tree=T.node("gt", [T.term("volz"), T.const(2.5)]),
                  regime_tree=None, direction="long", entry_style="market",
                  entry_param=1, stop_atr=3.0, target_r=0.0, time_stop=6,
                  trail_mode="none", max_hold=12)


# ---------------- controls --------------------------------------------------
def test_positive_control_planted_edge():
    data = {}
    for i in range(6):
        df = make_planted(seed=10 + i)
        compute_atoms(df, ["volume", "trend"], always_open=True)
        data[f"S{i}"] = df
    g = planted_genome()
    cells, pooled, _ = fold_cells(g, data, list(data), COST_TIERS["crypto_spot_taker"],
                                  BAR_SECONDS["1h"])
    assert len(cells) >= 6
    assert np.median(cells) > 1.0, f"planted edge not detected: {cells}"
    assert psr(pooled) > 0.9


def test_lookahead_control_absurd_sharpe():
    """A signal built from FUTURE returns must produce huge Sharpe -- proves
    the harness transmits signal when it exists."""
    df = make_df(4000, seed=3)
    fut = pd.Series(df["close"]).pct_change().shift(-2).fillna(0).to_numpy()
    df["cheat"] = (fut > 0.004).astype(float)
    g = Genome(timeframe="1h", root_type="bool",
               entry_tree=T.node("gt", [T.term("cheat"), T.const(0.5)]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=3.0, target_r=0.0, time_stop=2,
               trail_mode="none", max_hold=4)
    res = run_backtest(g, df, COST_TIERS["equity_liquid"])
    rs = [t[3] for t in res.trades]
    assert len(rs) > 50
    assert trade_sharpe(rs, 2000) > 5.0


def test_negative_random_genomes_lose_costs():
    df = make_df(3000, seed=4)
    cols = compute_atoms(df, ["trend", "oscillators", "vol_structure"],
                         always_open=True)
    finals = []
    for seed in range(8):
        g = random_genome(random.Random(seed), cols, ["1h"])
        res = run_backtest(g, df, COST_TIERS["crypto_spot_taker"])
        if res.n_trades > 20:
            finals.append(res.equity.iloc[-1])
    assert finals, "no random genome traded"
    assert np.median(finals) < 1.0     # random trading loses the costs


# ---------------- unit guards ----------------------------------------------
def test_protected_division():
    df = make_df(200)
    t = T.node("div", [T.term("close"), T.const(0.0)])
    v = T.evaluate(t, df)
    assert np.all(v == 0)


def test_gap_through_stop_fills_at_open():
    df = make_df(300, seed=5)
    # force a gap: bar 100 opens far below prior close
    df.iloc[100, df.columns.get_loc("open")] = df["close"].iloc[99] * 0.90
    df.iloc[100, df.columns.get_loc("low")] = df["close"].iloc[99] * 0.89
    df.iloc[100, df.columns.get_loc("high")] = df["close"].iloc[99] * 0.99
    fire = np.zeros(len(df)); fire[98] = 1.0     # one-shot: enter at bar 99,
    df["fire"] = fire                            # guaranteed held into the gap
    g = Genome(timeframe="1h", root_type="bool",
               entry_tree=T.node("gt", [T.term("fire"), T.const(0.5)]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=1.0, target_r=0.0, time_stop=0,
               trail_mode="none", max_hold=5000)
    res = run_backtest(g, df, COST_TIERS["equity_liquid"])
    assert res.n_trades == 1
    worst = res.trades[0][3]
    assert worst < -1.5, f"gap-through should lose MORE than 1R, got {worst}"


def test_genome_and_runtime_roundtrip(tmp_path):
    from primordial.pipeline import export_survivor
    from primordial.runtime import Strategy
    df = make_planted(seed=20)
    cols = compute_atoms(df, ["volume", "trend"], always_open=True)
    g = planted_genome()
    mf = Manifest(name="t", asset_class="crypto", source="synthetic",
                  symbols=["S0"], base_timeframe="1h",
                  allowed_timeframes=["1h"], start="2023-01-01",
                  end="2024-01-01", cost_tier="crypto_spot_taker")
    art = export_survivor(g, mf, {}, {"survived": True})
    p = tmp_path / "s.json"; p.write_text(art)
    strat = Strategy.load(str(p))
    sig = strat.signal(df[["open", "high", "low", "close", "volume"]].copy())
    direct = g.signal(df)
    assert np.array_equal(np.nan_to_num(sig), np.nan_to_num(direct))
    assert strat.exit_spec()["stop_atr"] == g.stop_atr


def test_ledger_keying_and_monotone_bar(tmp_path):
    p = str(tmp_path / "ledger.json")
    a = Ledger(p, "manifest_A"); b = Ledger(p, "manifest_B")
    a.log("e", 1000); bar1 = a.luck_bar(0.25)
    a.log("e", 5000); bar2 = a.luck_bar(0.25)
    assert bar2 > bar1                        # bar rises with trials
    assert b.effective_trials() == 1          # B's holdout untouched by A


def test_motifs_mined_on_train_only():
    train = {f"S{i}": make_df(1500, seed=30 + i) for i in range(4)}
    hold = {f"H{i}": make_df(1500, seed=90 + i) for i in range(4)}
    m1 = mine(train, list(train), [20], 3, seed=7)
    m2 = mine(train, list(train), [20], 3, seed=7)   # holdout never passed in
    assert json.dumps(m1) == json.dumps(m2)
    cols = add_motif_atoms(hold, m1)                  # applying is fine
    assert all(c in hold["H0"].columns for c in cols)


def test_dedupe_collapses_clones():
    df = make_df(1000)
    df["a"] = df["close"].pct_change(20)
    df["b"] = df["a"] * 1.0000001          # clone
    df["c"] = df["close"].pct_change().rolling(30).std()
    kept, merges = dedupe(df, ["a", "b", "c"])
    assert "b" in merges and merges["b"] == "a" and "c" in kept


def test_split_outer_embargo():
    data = {f"S{i}": make_df(1000, seed=i) for i in range(10)}
    sp = NameTimeSplit(list(data), 0.4, 0.3, embargo_bars=50,
                       rng=random.Random(0))
    tr = sp.train(data); ho = sp.holdout(data)
    assert not (set(tr) & set(ho))
    s = sp.train_syms[0]; h = sp.hold_syms[0]
    gap = (data[h].index[int(1000 * 0.7)] - tr[s].index[-1])
    assert gap >= pd.Timedelta(hours=50)   # embargo enforced at the boundary


def test_split_deterministic_and_drop_stable():
    syms = [f"S{i}" for i in range(40)]
    # (a) fetch ORDER must not matter: shuffled input, same seed -> same split
    a = NameTimeSplit(list(syms), 0.3, 0.3, 10, random.Random(7))
    shuffled = list(syms); random.Random(99).shuffle(shuffled)
    b = NameTimeSplit(shuffled, 0.3, 0.3, 10, random.Random(7))
    assert set(a.hold_syms) == set(b.hold_syms)
    # (b) one symbol failing to fetch must not reshuffle everyone else:
    # every survivor keeps its side except at most the k-boundary name
    dropped = [s for s in syms if s != "S17"]
    c = NameTimeSplit(dropped, 0.3, 0.3, 10, random.Random(7))
    moved = (set(a.hold_syms) ^ set(c.hold_syms)) - {"S17"}
    assert len(moved) <= 1                 # boundary shift only
    # (c) different seed -> genuinely different split
    d = NameTimeSplit(list(syms), 0.3, 0.3, 10, random.Random(8))
    assert set(a.hold_syms) != set(d.hold_syms)
    # (d) membership() is the audit record
    m = a.membership()
    assert m["hold_syms"] == sorted(a.hold_syms)
    assert not (set(m["hold_syms"]) & set(m["train_syms"]))


def test_pbo_sane():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((12, 16))
    p_noise = pbo(noise, n_splits=60)
    assert 0.25 <= p_noise <= 0.75         # pure noise: IS best ~coin flip OOS
    skill = noise + np.linspace(0, 3, 12)[:, None]
    assert pbo(skill, n_splits=60) < p_noise   # real skill lowers PBO


def test_context_atoms_and_name_set_isolation():
    from primordial.atoms import add_context
    data = {f"S{i}": make_df(1200, seed=40 + i) for i in range(8)}
    bench = {"SPY": make_df(1200, seed=99)}
    train = list(data)[:5]
    hold = list(data)[5:]
    cols = add_context(data, train, bench)
    assert "ctx_breadth" in cols and "ctx_SPY_trend" in cols
    # train context must be computable from train names alone: recompute on a
    # dict that has NEVER seen holdout names and compare
    data2 = {s: make_df(1200, seed=40 + i) for i, s in
             enumerate([f"S{i}" for i in range(8)]) if s in train}
    add_context(data2, train, bench)
    a = data[train[0]]["ctx_breadth"].dropna()
    b = data2[train[0]]["ctx_breadth"].dropna()
    assert np.allclose(a.values, b.values), "train context leaked holdout names"


def test_shared_block_index_preserves_context_alignment():
    from primordial.judge import make_block_index, block_bootstrap
    import random as _r
    df1 = make_df(1000, seed=1)
    df2 = make_df(1000, seed=2)
    rng = _r.Random(0)
    ridx = make_block_index(1000, rng)
    b1 = block_bootstrap(df1, rng, ridx=ridx)
    b2 = block_bootstrap(df2, rng, ridx=ridx)
    # same source bars picked for both frames: their return CORRELATION
    # structure at t is preserved (both drawn from the same original t')
    r1 = df1["close"].pct_change().iloc[ridx].to_numpy()
    br1 = b1["close"].pct_change().to_numpy()
    m = np.isfinite(r1) & np.isfinite(br1)
    assert np.corrcoef(r1[m], br1[m])[0, 1] > 0.99


def test_no_same_bar_profit_on_random_walk():
    """THE gen-8/9 bug: always-fire limit entries harvested each entry bar's
    own range as profit -- Sharpe +4.6 long / +3.5 short on a PURE RANDOM
    WALK. After the fix both sides must be roughly flat-to-negative there."""
    rng = np.random.default_rng(0)
    n = 15000
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    r = rng.standard_normal(n) * 0.004
    close = 100 * np.exp(np.cumsum(r))
    o = np.roll(close, 1); o[0] = close[0]
    sp = np.abs(rng.standard_normal(n)) * 0.004
    df = pd.DataFrame({"open": o, "high": np.maximum(o, close) * (1 + sp),
                       "low": np.minimum(o, close) * (1 - sp), "close": close,
                       "volume": np.ones(n)}, index=idx)
    yr = (idx[-1] - idx[0]).total_seconds() / 31557600
    for side in ("long", "short"):
        g = Genome(timeframe="1h", root_type="bool",
                   entry_tree=T.node("gte", [T.const(2.0), T.const(1.0)]),
                   regime_tree=None, direction=side, entry_style="limit",
                   entry_param=1.0, stop_atr=1.6, target_r=1.0, time_stop=96,
                   trail_mode="breakeven_1r", max_hold=192)
        res = run_backtest(g, df, COST_TIERS["crypto_spot_maker"])
        rs = [t[3] for t in res.trades]
        assert len(rs) > 300
        sr = trade_sharpe(rs, len(rs) / yr)
        assert sr < 1.0, f"{side} always-fire limit SR {sr} -- same-bar leak?"
        assert np.mean(rs) < 0.05


def test_constant_entry_is_floored():
    from primordial.judge import fitness
    from primordial.data import BAR_SECONDS
    data = {f"S{i}": make_df(1500, seed=60 + i) for i in range(4)}
    for d in data.values():
        compute_atoms(d, ["trend"], always_open=True)
    g = Genome(timeframe="1h", root_type="bool",
               entry_tree=T.node("gte", [T.const(2.0), T.const(1.0)]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=2.0, target_r=1.0, time_stop=24,
               trail_mode="none", max_hold=96)
    f, info = fitness(g, data, list(data), COST_TIERS["crypto_spot_maker"],
                      BAR_SECONDS["1h"])
    assert f == -9.99 and info.get("degenerate") == "constant_entry"
