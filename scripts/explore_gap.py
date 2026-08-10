"""
ZERO-TRIAL screen: do overnight gaps revert intraday? TRAIN names x TRAIN
years only, nothing logged. Buckets by gap size in ATR units, and splits
SYSTEMATIC (whole universe gapped) from IDIOSYNCRATIC (this name only) --
the distinction that decides whether a gap is flow or information.

    python3.10 scripts/explore_gap.py --manifest manifests/us_sp400.yaml
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
    p.add_argument("--trend-only", action="store_true",
                   help="restrict to names above their 200-bar mean")
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
    print(f"[explore_gap] {mf.name} | TRAIN ONLY: {len(syms)} names"
          + (" | UPTREND ONLY" if a.trend_only else ""))

    O = pd.DataFrame({s: data[s]["open"] for s in syms})
    C = pd.DataFrame({s: data[s]["close"] for s in syms})
    H = pd.DataFrame({s: data[s]["high"] for s in syms})
    L = pd.DataFrame({s: data[s]["low"] for s in syms})
    pc = C.shift(1)
    tr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()]).groupby(level=0).max()
    atr = tr.rolling(14, min_periods=5).mean()

    gap_atr = (O - pc) / atr.where(atr > 0)          # gap in ATR units
    o2c = C / O - 1.0                                 # open-to-close return
    med_gap = gap_atr.median(axis=1)                  # universe-wide gap
    rel_gap = gap_atr.sub(med_gap, axis=0)            # this name vs the pack
    gate = (C > C.rolling(200).mean()) if a.trend_only else None

    cost = mf.cost.adverse_frac() * 2 * 1e4
    print(f"  round-trip cost: {cost:.0f} bps | entry at OPEN, exit at CLOSE\n")

    def table(title, mask_extra=None):
        print(f"  --- {title} ---")
        print(f"  {'gap (ATR)':<14}{'n':>8}  {'o2c gross':>10} {'net':>9}")
        for lo, hi in [(-99, -2.0), (-2.0, -1.0), (-1.0, -0.5),
                       (-0.5, 0.5), (0.5, 1.0), (1.0, 99)]:
            m = (gap_atr > lo) & (gap_atr <= hi)
            if gate is not None:
                m = m & gate
            if mask_extra is not None:
                # mask_extra is per-DATE; broadcast it across all names
                m = m & pd.DataFrame(
                    np.repeat(mask_extra.values[:, None], m.shape[1], axis=1),
                    index=m.index, columns=m.columns)
            v = o2c.values[m.values]
            v = v[np.isfinite(v)]
            n = len(v)
            g = float(np.mean(v)) * 100 if n else float("nan")
            lbl = f"{max(lo,-9):+.1f} to {min(hi,9):+.1f}"
            print(f"  {lbl:<14}{n:>8}  {g:>+9.3f}% {g - cost/100:>+8.3f}%")
        print()

    table("ALL GAPS")
    # systematic = whole universe gapped the same way; idiosyncratic = not
    sysmask = (med_gap.abs() > 0.3)
    table("SYSTEMATIC (universe-wide gap day)", sysmask)
    table("IDIOSYNCRATIC (quiet universe, this name gapped)", ~sysmask)
    print("  Reversion = negative gaps show POSITIVE open-to-close.")
    print("  Compare systematic vs idiosyncratic: that split is the thesis.")
    print("  IN-SAMPLE ONLY. Daily bars = a lower bound (no intraday timing).")


if __name__ == "__main__":
    main()
