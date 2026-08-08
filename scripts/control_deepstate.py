"""
Equity-side machinery control: a planted DEEP-STATE-shaped edge must survive
the full gauntlet, expressed through the new vocabulary (ema_spread_atr,
spread_runlen, anchor_ema exit) and passing through the new breadth gate.

This is to the equity adapter what the planted volz edge is to crypto: if this
control fails after a change to the judge, backtest, atoms, or pipeline, a
gate is killing real dip-reversion edges -- find which one before trusting
any equity run.

  python scripts/control_deepstate.py          (~1-3 min)
"""
import os
import random
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TMP = tempfile.mkdtemp(prefix="primordial_dsctl_")
os.environ["PRIMORDIAL_CACHE"] = os.path.join(TMP, "cache")
os.environ["PRIMORDIAL_LEDGER"] = os.path.join(TMP, "ledger.json")

import primordial.data as D
import primordial.pipeline as P
from primordial import tree as T, motifs as M
from primordial.genome import Genome
from primordial.judge import NameTimeSplit, Ledger
from primordial.universe import Manifest

N_SYMS = 12


def _planted_dip(sym, timeframe, start, end):
    """Daily bars: gentle uptrend (stays above its long SMA), and every ~35
    bars a genuine deep-state event -- a 3-bar fresh sag that pulls the 12/26
    EMA spread below -0.25 ATR, followed by reversion back through the 26-EMA.
    The edge IS the mechanism the deep-state rule trades; nothing else is in
    the data."""
    import hashlib as _hl
    rng = np.random.default_rng(int(_hl.sha256(sym.encode()).hexdigest()[:8],
                                    16))  # hash() is per-process random
    n = 3000
    r = rng.standard_normal(n) * 0.009 + 0.0003          # drift keeps trend gate open
    for i in range(120, n - 12, 22):
        r[i:i + 3] = -0.016                              # the sag: fresh + deep
        r[i + 3:i + 8] = 0.014                           # the snap back to anchor
    close = 100 * np.exp(np.cumsum(r))
    o = np.roll(close, 1); o[0] = close[0]
    df = pd.DataFrame({
        "open": o,
        "high": np.maximum(o, close) * (1 + np.abs(rng.standard_normal(n)) * 0.004),
        "low": np.minimum(o, close) * (1 - np.abs(rng.standard_normal(n)) * 0.004),
        "close": close,
        "volume": np.abs(rng.standard_normal(n) + 3) * 1e6})
    df.index = pd.date_range(start, periods=n, freq="1D")
    return df


def _manifest():
    m = yaml.safe_load(open(os.path.join(ROOT, "manifests/smoke_synthetic.yaml")))
    m.update(name="deepstate_control", asset_class="equity",
             base_timeframe="1d", allowed_timeframes=["1d"],
             cost_tier="equity_liquid",
             symbols=[f"DS{i:02d}" for i in range(N_SYMS)],
             start="2014-01-02", end="2026-01-01")
    p = os.path.join(TMP, "deepstate_control.yaml")
    yaml.safe_dump(m, open(p, "w"))
    return p


def main():
    print("\n=== DEEP-STATE MACHINERY CONTROL (equity, daily, planted dips) ===")
    D._synthetic = _planted_dip
    mf = Manifest.load(_manifest())
    base = D.fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    mtf = D.multi_timeframe(base, mf.base_timeframe, mf.allowed_timeframes)
    rng = random.Random(mf.seed)
    split = NameTimeSplit(list(base.keys()), mf.holdout_name_frac,
                          mf.holdout_time_frac, mf.embargo_bars, rng)
    mtf_train = {tf: split.train(d) for tf, d in mtf.items()}
    mtf_hold = {tf: split.holdout(d) for tf, d in mtf.items()}
    motif_defs = M.mine(mtf_train["1d"], split.train_syms, mf.motif_scales,
                        mf.n_motifs, seed=mf.seed)
    P._prepare(mtf_train, mf.adapter, split.train_syms, [], motif_defs)
    P._prepare(mtf_hold, mf.adapter, [], split.hold_syms, motif_defs)

    # the frozen deep-state shape, hand-built from the new vocabulary:
    #   spread < -0.25 ATR  AND  sag fresh (runlen <= 15)  AND  in uptrend
    #   exit at the 26-EMA anchor; stop pushed to the 6-ATR rail (near-none)
    g = Genome(
        timeframe="1d", root_type="bool",
        entry_tree=T.node("and", [
            T.node("lt", [T.term("ema_spread_atr"), T.const(-0.40)]),
            T.node("and", [
                T.node("lte", [T.term("spread_runlen"), T.const(15)]),
                T.node("gt", [T.term("dist_sma_s"), T.const(0.0)]),
            ]),
        ]),
        regime_tree=None, direction="long", entry_style="market",
        entry_param=1, stop_atr=6.0, target_r=0.0, time_stop=0,
        trail_mode="none", max_hold=40, anchor_ema=26)

    ledger = Ledger(os.environ["PRIMORDIAL_LEDGER"], mf.fingerprint())
    ledger.log("control", 1, 1)
    survivors, report = P._gauntlet([(5.0, g)], mtf_train, mtf_hold, split,
                                    mf, ledger, rng, verbose=True)
    row = report[0]
    if "breadth" in row:
        print(f"breadth gate exercised: {row['breadth']}")
    slim = {k: v for k, v in row.items()
            if k not in ("holdout_equity", "genome_json")}
    assert len(survivors) == 1, \
        f"GAUNTLET KILLED THE DEEP-STATE EDGE -- find the gate: {slim}"
    print("DEEP-STATE MACHINERY CONTROL PASSED "
          f"(holdout SR {row['holdout_median_sharpe']:+.2f}, "
          f"anchor exit + breadth gate live)")


if __name__ == "__main__":
    main()
