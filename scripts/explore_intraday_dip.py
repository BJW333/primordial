"""
ZERO-TRIAL screen: do intraday dips get bought back? Buckets bars by how far
price has fallen from its recent rolling high (in ATR units), then measures
forward returns. TRAIN names x TRAIN years only; nothing logged.

    python3.10 scripts/explore_intraday_dip.py --manifest manifests/us_liquid_5m.yaml
"""
import argparse, os, random, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial.data import fetch, multi_timeframe      # noqa: E402
from primordial.judge import NameTimeSplit              # noqa: E402
from primordial.universe import Manifest                # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--lookback", type=int, default=12,
                   help="bars over which the rolling high is measured")
    p.add_argument("--horizons", default="3,6,12,24")
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
    print(f"[dip] {mf.name} {tf} | TRAIN ONLY: {len(syms)} names")

    C = pd.DataFrame({s: data[s]["close"] for s in syms})
    H = pd.DataFrame({s: data[s]["high"] for s in syms})
    L = pd.DataFrame({s: data[s]["low"] for s in syms})
    pc = C.shift(1)
    tr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()]).groupby(
        level=0).max()
    atr = tr.rolling(14, min_periods=5).mean()
    roll_hi = C.rolling(a.lookback, min_periods=3).max()
    dip = (C - roll_hi) / atr.where(atr > 0)     # <=0, in ATR units

    cost = mf.cost.adverse_frac() * 2 * 1e4
    hs = [int(x) for x in a.horizons.split(",")]
    print(f"  round-trip cost: {cost:.0f} bps | dip vs {a.lookback}-bar high\n")
    print("  dip (ATR)          n" + "".join(f"   +{h}b gross      net"
                                             for h in hs))
    for lo, hi in [(-99, -3.0), (-3.0, -2.0), (-2.0, -1.0),
                   (-1.0, -0.5), (-0.5, 0.0)]:
        m = (dip > lo) & (dip <= hi)
        row = f"  {max(lo,-9):+.1f} to {hi:+.1f} {int(m.values.sum()):>8}"
        for h in hs:
            fwd = C.shift(-h) / C - 1.0
            v = fwd.values[m.values]
            v = v[np.isfinite(v)]
            g = float(np.mean(v)) * 100 if len(v) else float("nan")
            row += f"   {g:>+9.4f}% {g - cost/100:>+8.4f}%"
        print(row)
    print("\n  Buy-the-dip = deeper dips show LARGER positive forward returns.")
    print("  Read the NET columns: at 5m the cost wall is the whole game.")
    print("  IN-SAMPLE ONLY. A clean shape earns ONE gauntlet trial.")


if __name__ == "__main__":
    main()
