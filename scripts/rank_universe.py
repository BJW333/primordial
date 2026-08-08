"""
Rank a symbols file by TRADABILITY and keep the top N.

    python3.10 scripts/rank_universe.py manifests/sp600_symbols.txt \
        --manifest manifests/us_sp600.yaml --top 150

Reads bars from the primordial cache (run fetch_data.py first), ranks by
median daily DOLLAR VOLUME over the FIRST HALF of the window (late listers
use whatever history they have there, so recent IPOs rank low -- that is
conservative, not a bug), and writes <input>_top<N>.txt.

Why first-half only: ranking on full-period liquidity leaks a little --
names that performed well GREW their volume, so "most liquid over the whole
window" quietly favors winners. Early-window liquidity is set before most of
the outcomes you'll be judged on.

NEVER rank a universe by returns/Sharpe/trendiness. That selects names on
the target using the whole period (holdout included) and no split can undo
it: every downstream result inherits the bias. Liquidity is a legitimate
criterion because it's about executability, not outcome.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from primordial.data import fetch          # noqa: E402
from primordial.universe import Manifest   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("symbols_file")
    p.add_argument("--manifest", required=True,
                   help="manifest supplying source/timeframe/dates")
    p.add_argument("--top", type=int, default=150)
    a = p.parse_args()

    mf = Manifest.load(a.manifest)
    with open(a.symbols_file) as f:
        syms = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

    print(f"ranking {len(syms)} names by early-window median dollar volume...")
    scores = {}
    for i, s in enumerate(syms, 1):
        try:
            got = fetch(mf.source, [s], mf.base_timeframe, mf.start, mf.end)
        except Exception:
            continue
        if s not in got:
            continue
        df = got[s]
        half = df.iloc[: max(len(df) // 2, 250)]
        dv = (half["close"] * half["volume"]).replace(0, np.nan).dropna()
        if len(dv) >= 100:
            scores[s] = float(dv.median())
        if i % 100 == 0:
            print(f"  {i}/{len(syms)} scored")

    ranked = sorted(scores, key=scores.get, reverse=True)[: a.top]
    base, ext = os.path.splitext(a.symbols_file)
    out = f"{base}_top{a.top}{ext}"
    with open(out, "w") as f:
        f.write(f"# top {a.top} of {len(scores)} scored names by early-window "
                f"median dollar volume\n")
        f.write("\n".join(sorted(ranked)) + "\n")
    lo = scores[ranked[-1]] / 1e6
    hi = scores[ranked[0]] / 1e6
    print(f"kept {len(ranked)} -> {out}  (dollar-vol range "
          f"${lo:,.1f}M - ${hi:,.1f}M/day)")


if __name__ == "__main__":
    main()
