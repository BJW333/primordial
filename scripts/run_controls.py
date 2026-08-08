"""
Full-pipeline controls. Run BOTH after any change to the judge, backtest, or
engine. Slow (~5-10 min) by design -- they exercise the whole stack.

  negative: random walks, no edge exists      -> MUST produce 0 survivors
  machinery: a hand-built true genome on planted data -> MUST survive gauntlet
  search:   evolution on planted data         -> SHOULD find + pass survivors

If negative fails: the judge leaks. Stop trusting every result.
If machinery fails: a gate is killing real edges. Find which one.
If search fails but machinery passes: a budget/search problem, not a truth
problem -- raise generations/pop before touching any threshold.
"""
import os, sys, random, tempfile
import numpy as np, pandas as pd, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TMP = tempfile.mkdtemp(prefix="primordial_ctl_")
os.environ["PRIMORDIAL_CACHE"] = os.path.join(TMP, "cache")
os.environ["PRIMORDIAL_LEDGER"] = os.path.join(TMP, "ledger.json")

import primordial.data as D
import primordial.pipeline as P
from primordial import tree as T, motifs as M
from primordial.genome import Genome
from primordial.judge import NameTimeSplit, Ledger
from primordial.universe import Manifest


def _base_df(n, seed):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * 0.005
    close = 100 * np.exp(np.cumsum(r))
    o = np.roll(close, 1); o[0] = close[0]
    return pd.DataFrame({
        "open": o,
        "high": np.maximum(o, close) * 1.002,
        "low": np.minimum(o, close) * 0.998,
        "close": close,
        "volume": np.abs(rng.standard_normal(n) + 3) * 1e6})


def _planted(sym, timeframe, start, end):
    df = _base_df(9000, abs(hash(sym)) % 1000)
    close = df["close"].to_numpy().copy(); vol = df["volume"].to_numpy().copy()
    for i in range(60, len(df) - 10, 30):
        vol[i] *= 15
        steps = np.exp(np.cumsum(np.full(6, 0.025 / 6)))
        close[i + 1:i + 7] *= steps; close[i + 7:] *= steps[-1]
    df["close"] = close
    df["open"] = np.roll(close, 1); df.iloc[0, df.columns.get_loc("open")] = close[0]
    df["high"] = np.maximum(df["open"], df["close"]) * 1.002
    df["low"] = np.minimum(df["open"], df["close"]) * 0.998
    df["volume"] = vol
    df.index = pd.date_range(start, periods=len(df), freq="1h")
    return df


def _manifest(planted):
    m = yaml.safe_load(open(os.path.join(ROOT, "manifests/smoke_synthetic.yaml")))
    m["name"] = "planted_control" if planted else "negative_control"
    if planted:
        m["allowed_timeframes"] = ["1h"]
    p = os.path.join(TMP, m["name"] + ".yaml")
    yaml.safe_dump(m, open(p, "w"))
    return p


def _own_cache(name):
    """Each control gets its OWN cache dir. The negative control fetches
    plain random walks for the same symbols/dates the planted controls use;
    a shared cache serves those stale frames to the planted controls and the
    plant never enters the data (this exact bug once made the machinery
    control report a false failure)."""
    d = os.path.join(TMP, "cache_" + name)
    os.environ["PRIMORDIAL_CACHE"] = d
    return d


def negative_control():
    _own_cache("negative")
    print("\n=== NEGATIVE CONTROL (random walks; correct answer: 0 survivors) ===")
    survivors, _, _ = P.run(_manifest(False), generations=5, pop_size=16,
                            n_islands=2, verbose=True)
    assert len(survivors) == 0, "JUDGE LEAKS: survivors on random data"
    print("NEGATIVE CONTROL PASSED")


def machinery_control():
    _own_cache("machinery")
    print("\n=== MACHINERY CONTROL (true genome must survive the gauntlet) ===")
    D._synthetic = _planted
    mf = Manifest.load(_manifest(True))
    base = D.fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    mtf = D.multi_timeframe(base, mf.base_timeframe, mf.allowed_timeframes)
    rng = random.Random(mf.seed)
    split = NameTimeSplit(list(base.keys()), mf.holdout_name_frac,
                          mf.holdout_time_frac, mf.embargo_bars, rng)
    mtf_train = {tf: split.train(d) for tf, d in mtf.items()}
    mtf_hold = {tf: split.holdout(d) for tf, d in mtf.items()}
    motif_defs = M.mine(mtf_train["1h"], split.train_syms, mf.motif_scales,
                        mf.n_motifs, seed=mf.seed)
    P._prepare(mtf_train, mf.adapter, split.train_syms, [], motif_defs)
    P._prepare(mtf_hold, mf.adapter, [], split.hold_syms, motif_defs)
    g = Genome(timeframe="1h", root_type="bool",
               entry_tree=T.node("gt", [T.term("volz"), T.const(2.5)]),
               regime_tree=None, direction="long", entry_style="market",
               entry_param=1, stop_atr=3.0, target_r=0.0, time_stop=6,
               trail_mode="none", max_hold=12)
    ledger = Ledger(os.environ["PRIMORDIAL_LEDGER"], mf.fingerprint())
    ledger.log("control", 1, 1)
    survivors, report = P._gauntlet([(5.0, g)], mtf_train, mtf_hold, split,
                                    mf, ledger, rng, verbose=True)
    assert len(survivors) == 1, f"GAUNTLET KILLED A REAL EDGE: {report}"
    print("MACHINERY CONTROL PASSED")


def search_control():
    _own_cache("search")
    print("\n=== SEARCH CONTROL (evolution should find the planted edge) ===")
    D._synthetic = _planted
    survivors, _, _ = P.run(_manifest(True), generations=GENS, pop_size=POP,
                            n_islands=2, verbose=True)
    if survivors:
        print(f"SEARCH CONTROL PASSED ({len(survivors)} survivors)")
    else:
        print("SEARCH CONTROL: not found at this budget. Machinery control is "
              "the hard guarantee; raise --generations/--pop, do NOT touch "
              "thresholds.")


GENS, POP = 20, 48   # search-control budget; CLI-overridable

if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--generations", type=int, default=20)
    _p.add_argument("--pop", type=int, default=48)
    _p.add_argument("--only", choices=["negative", "machinery", "search"],
                    default=None)
    _a = _p.parse_args()
    GENS, POP = _a.generations, _a.pop
    if _a.only in (None, "negative"):
        negative_control()
    if _a.only in (None, "machinery"):
        machinery_control()
    if _a.only in (None, "search"):
        search_control()
