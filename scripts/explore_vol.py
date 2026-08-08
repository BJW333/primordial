"""
ZERO-TRIAL screen: does volatility CONTRACTION predict expansion + direction?
Buckets every bar by where its ATR% sits in its own 252-bar history, then
measures forward returns. TRAIN names x TRAIN years only; nothing logged.

    python3.10 scripts/explore_vol.py --manifest manifests/us_sp400.yaml
"""
import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from primordial.data import fetch, multi_timeframe    # noqa: E402
from primordial.judge import NameTimeSplit            # noqa: E402
from primordial.universe import Manifest              # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--horizons", default="5,10,20,40")
    p.add_argument("--window", type=int, default=252,
                   help="lookback for the ATR percentile")
    p.add_argument("--trend-only", action="store_true",
                   help="restrict to bars above the 200-bar mean")
    a = p.parse_args()

    mf = Manifest.load(a.manifest)
    rng = random.Random(mf.seed)
    base = fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    mtf = multi_timeframe(base, mf.base_timeframe, mf.allowed_timeframes)
    tf = mf.base_timeframe
    split = NameTimeSplit(list(base.keys()), mf.holdout_name_frac,
                          mf.holdout_time_frac, mf.embargo_bars, rng)
    data = split.train(mtf[tf])
    syms = [s for s in split.train_syms if s in data]
    print(f"[explore_vol] {mf.name} {tf} | TRAIN ONLY: {len(syms)} names")

    closes = pd.DataFrame({s: data[s]["close"] for s in syms})
    highs = pd.DataFrame({s: data[s]["high"] for s in syms})
    lows = pd.DataFrame({s: data[s]["low"] for s in syms})
    tr = (highs - lows) / closes
    atr = tr.rolling(14).mean()
    pct = atr.rolling(a.window).rank(pct=True)
    gate = closes > closes.rolling(200).mean() if a.trend_only else None

    cost = mf.cost.adverse_frac() * 2 * 1e4
    print(f"  round-trip cost: {cost:.0f} bps | ATR percentile over "
          f"{a.window} bars"
          + (" | UPTREND ONLY" if a.trend_only else ""))
    hs = [int(x) for x in a.horizons.split(",")]
    print("\n  atr_pctile        n" + "".join(f"   +{h}b gross      net"
                                              for h in hs))
    for lo, hi in [(-0.01, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.0)]:
        mask = (pct > lo) & (pct <= hi)
        if gate is not None:
            mask = mask & gate
        row = f"  {max(lo,0):.1f}-{hi:.1f}  {int(mask.values.sum()):>10}"
        for h in hs:
            fwd = closes.shift(-h) / closes - 1.0
            v = fwd.values[mask.values]
            v = v[np.isfinite(v)]
            m = float(np.mean(v)) * 100 if len(v) else float("nan")
            row += f"   {m:+7.3f}%  {m - cost/100:+7.3f}%"
        print(row)
    print("\n  0.0-0.1 = most COMPRESSED vol (coiled). If contraction "
          "predicts\n  expansion UP, that bucket beats the rest net of costs."
          "\n  IN-SAMPLE ONLY. A shape earns ONE gauntlet trial.")


if __name__ == "__main__":
    main()
