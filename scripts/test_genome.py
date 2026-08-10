#!/usr/bin/env python3
"""
Score a HAND-WRITTEN rule with primordial's judge.

    python3.10 scripts/test_genome.py manifests/us_liquid_5m.yaml

WHY THIS ISN'T JUST `primordial run`:
The normal gauntlet estimates its luck bar from the SPREAD of a candidate
population's holdout Sharpes (MAD-robust, needs >=3 candidates). One
hand-written rule has no population, so there is no bar. This script builds
one honestly: it runs your rule AND N random genomes over the same panel,
same folds, same costs, and asks whether your rule beats the null
distribution its own competitors produce.

WHAT IS BEING TESTED HERE (the QC SPY_SMA sample algorithm):
    long  when close > SMA(5) > SMA(30)
    short when close < SMA(5) < SMA(30)
    flat  before the close, stop-and-reverse otherwise

Ported exactly -- d_sma5 and d_sma30 are injected as real atom columns, so
the tree compares the same quantities the original does. Two departures,
both unavoidable and both noted in the output:
  1. A Genome is long OR short, not stop-and-reverse. The long and short
     legs are scored SEPARATELY. A rule that only works as a combined
     always-in machine will show up as two mediocre legs, which is
     information, not a bug.
  2. The 5%-daily-drawdown kill switch has no genome equivalent. It is a
     risk overlay, not an edge, and cannot create one -- omitting it tests
     the signal rather than the circuit breaker.

The rule is scored across the WHOLE manifest panel, not just SPY. A rule
that only works on the one symbol it was written for is a rule that was
fitted to that symbol.
"""
from __future__ import annotations

import random
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial import atoms as A                      # noqa: E402
from primordial import data as D                       # noqa: E402
from primordial import tree as T                       # noqa: E402
from primordial.data import BAR_SECONDS                # noqa: E402
from primordial.genome import Genome, random_genome    # noqa: E402
from primordial.judge import (fitness, fold_cells,        # noqa: E402
                              NameTimeSplit)
from primordial.universe import Manifest               # noqa: E402

N_NULL = int(os.environ.get("N_NULL", "200"))
TF = os.environ.get("TF", "5m")
SESSION_BARS = {"5m": 78, "15m": 26, "30m": 13, "1h": 7}


def inject_sma(df: pd.DataFrame, fast: int = 5, slow: int = 30) -> None:
    """The exact quantities the QC algorithm compares, as atom columns."""
    c = df["close"]
    df[f"d_sma{fast}"] = c / c.rolling(fast, min_periods=fast).mean() - 1.0
    df[f"d_sma{slow}"] = c / c.rolling(slow, min_periods=slow).mean() - 1.0


def sma_genome(direction: str, tf: str, fast: int = 5, slow: int = 30) -> Genome:
    """
    close > SMA(f) > SMA(s)  <=>  d_smaF > 0  AND  d_smaF < d_smaS
    (if SMA(f) > SMA(s) then c/SMA(f) < c/SMA(s), so d_smaF < d_smaS)
    """
    f_t, s_t = T.term(f"d_sma{fast}"), T.term(f"d_sma{slow}")
    if direction == "long":
        tree = T.node("and", [T.node("gt", [f_t, T.const(0.0)]),
                              T.node("lt", [f_t, s_t])])
    else:
        tree = T.node("and", [T.node("lt", [f_t, T.const(0.0)]),
                              T.node("gt", [f_t, s_t])])
    hold = SESSION_BARS.get(tf, 78)
    return Genome(timeframe=tf, root_type="bool", entry_tree=tree,
                  regime_tree=None, direction=direction,
                  entry_style="market", entry_param=0.5,
                  stop_atr=99.0,          # original has no stop, only a
                  target_r=0.0,           # daily kill switch
                  time_stop=hold, trail_mode="none", max_hold=hold)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mf = Manifest.load(sys.argv[1])
    print(f"[{mf.name}] {len(mf.symbols)} syms | tf {TF} | tier {mf.cost_tier} "
          f"| {mf.cost.adverse_frac() * 2 * 1e4:.1f} bps round trip")

    base = D.fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    mtf = D.multi_timeframe(base, mf.base_timeframe, [TF])
    data = mtf[TF]
    blocks = list(A.BLOCKS.keys())
    for s, df in data.items():
        A.compute_atoms(df, blocks, False)
        inject_sma(df)
    cols = [c for c in next(iter(data.values())).columns
            if c not in ("open", "high", "low", "close", "volume")]

    # same split the pipeline uses: hold out NAMES and the TAIL of time,
    # with an embargo at the boundary. Same seed => same split as a real run.
    split = NameTimeSplit(list(data), mf.holdout_name_frac,
                          mf.holdout_time_frac, mf.embargo_bars,
                          random.Random(mf.seed))
    d_tr, d_ho = split.train(data), split.holdout(data)
    tr = [s for s in split.train_syms if s in d_tr]
    ho = [s for s in split.hold_syms if s in d_ho]
    print(f"panel: {len(tr)} train names / {len(ho)} holdout names "
          f"(time tail {mf.holdout_time_frac:.0%} held back)\n")

    rows = []
    for d in ("long", "short"):
        g = sma_genome(d, TF)
        f, meta = fitness(g, d_tr, tr, mf.cost, BAR_SECONDS[TF])
        cells, pooled, _ = fold_cells(g, d_ho, ho, mf.cost, BAR_SECONDS[TF], k=2)
        med = float(np.median(cells)) if cells else 0.0
        rows.append((d, f, meta, med, len(cells)))
        print(f"SMA {d:5s} | train fit {f:+.3f} "
              f"({meta.get('n_cells', 0)} cells, {meta.get('n_trades', 0)} trades) "
              f"| holdout med SR {med:+.3f} over {len(cells)} cells")
        if meta.get("degenerate"):
            print(f"          DEGENERATE: {meta['degenerate']}")

    # ---- the null: same panel, same folds, same costs, random structure ----
    print(f"\nbuilding null from {N_NULL} random genomes at {TF} ...")
    rr = random.Random(12345)
    null = []
    for i in range(N_NULL):
        rg = random_genome(rr, cols, [TF])
        try:
            cells, _, _ = fold_cells(rg, d_ho, ho, mf.cost, BAR_SECONDS[TF], k=2)
        except Exception:
            continue
        if len(cells) >= 6:
            null.append(float(np.median(cells)))
    if len(null) < 30:
        print(f"  only {len(null)} usable null genomes -- bar unreliable")
        return 1

    null = np.array(null)
    bar95 = float(np.percentile(null, 95))
    print(f"  null holdout med SR: mean {null.mean():+.3f} "
          f"sd {null.std(ddof=1):.3f} | 95th pct = {bar95:+.3f} "
          f"(n={len(null)})")

    print("\n" + "=" * 62)
    for d, f, meta, med, nc in rows:
        pct = float((null < med).mean() * 100.0)
        verdict = "BEATS NULL" if med > bar95 else "indistinguishable from null"
        print(f"SMA {d:5s}: holdout med SR {med:+.3f} | {pct:.0f}th pct of null "
              f"| {verdict}")
    print("=" * 62)
    print("A rule at the 50th percentile of random genomes is a random genome.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
