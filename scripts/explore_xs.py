"""
ZERO-TRIAL explorer: does cross-sectional divergence mean-revert on this
universe? TRAIN names x TRAIN years only -- the holdout is never touched and
NOTHING is logged to the ledger. Use this to decide whether an idea is worth
one honest gauntlet trial; it is NOT evidence by itself (in-sample, no
deflation, no path robustness).

    python3.10 scripts/explore_xs.py --manifest manifests/coinbase_spot.yaml
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
    p.add_argument("--tf", default=None, help="timeframe (default: base)")
    p.add_argument("--horizons", default="4,12,24,72",
                   help="forward bars to measure")
    p.add_argument("--lookback", type=int, default=24,
                   help="bars over which divergence is measured")
    p.add_argument("--relative", action="store_true",
                   help="measure FORWARD returns vs universe median too "
                        "(strips beta: the real signal is the spread)")
    a = p.parse_args()

    mf = Manifest.load(a.manifest)
    tf = a.tf or mf.base_timeframe
    rng = random.Random(mf.seed)
    base = fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    mtf = multi_timeframe(base, mf.base_timeframe, mf.allowed_timeframes)
    if tf not in mtf:
        sys.exit(f"{tf} not in {list(mtf)}")
    split = NameTimeSplit(list(base.keys()), mf.holdout_name_frac,
                          mf.holdout_time_frac, mf.embargo_bars, rng)
    data = split.train(mtf[tf])
    syms = [s for s in split.train_syms if s in data]
    print(f"[explore_xs] {mf.name} {tf} | TRAIN ONLY: {len(syms)} names "
          f"(holdout {len(split.hold_syms)} untouched)")

    closes = pd.DataFrame({s: data[s]["close"] for s in syms}).dropna(
        how="all")
    lb = a.lookback
    ret_lb = closes.pct_change(lb)
    div = ret_lb.sub(ret_lb.median(axis=1), axis=0)
    rank = div.rank(axis=1, pct=True)

    cost_bps = mf.cost.adverse_frac() * 2 * 1e4
    print(f"  round-trip cost assumption: {cost_bps:.0f} bps "
          f"(manifest tier '{mf.cost_tier}')")
    print(f"  divergence measured over {lb} bars\n")

    hs = [int(x) for x in a.horizons.split(",")]
    buckets = [(-0.01, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print("  bucket        n" + "".join(f"   +{h}b gross      net" for h in hs))
    for lo, hi in buckets:
        mask = (rank > lo) & (rank <= hi)
        row = f"  {max(lo,0):.1f}-{hi:.1f}  {int(mask.values.sum()):>8}"
        for h in hs:
            fwd = closes.shift(-h) / closes - 1.0
            if a.relative:
                fwd = fwd.sub(fwd.median(axis=1), axis=0)
            vals = fwd.values[mask.values]
            vals = vals[np.isfinite(vals)]
            m = float(np.mean(vals)) * 100 if len(vals) else float("nan")
            row += f"   {m:+7.3f}%  {m - cost_bps/100:+7.3f}%"
        print(row)
    print("\n  low bucket = most UNDERperforming vs pack; high = most "
          "outperforming.\n  Mean reversion: low buckets positive, high "
          "negative, monotone between.\n  Momentum: the opposite. Noise: "
          "flat. Costs are already the hard part --\n  read the NET columns."
          "\n  IN-SAMPLE ONLY. A clean shape here earns ONE gauntlet trial.")


if __name__ == "__main__":
    main()
