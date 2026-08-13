"""
compare_series.py -- are two rotation strategies the same edge twice?

    python3.10 scripts/compare_series.py out/series/A.csv out/series/B.csv

Reads two period-return series written by the rotation scripts, aligns them
on DATE (inner join -- different rebalance calendars are fine, only shared
dates are compared), and reports:

  overlap        how many periods actually line up. Under 24, stop; the
                 correlation is not measurable.
  corr           Pearson on the overlap. This is the headline.
  corr, holdout  same on the last 30% of shared periods -- correlation in
                 the window where both were validated matters more than
                 correlation over a full sample that includes both train
                 sets.
  hit agreement  fraction of periods where both are up or both are down.
                 Correlation can be dragged by a few shared crashes; this
                 is the blunter check.
  combined       equal-weight blend of the two, and its SR against each
                 leg's SR. If the blend's SR is not meaningfully above the
                 better leg, holding both buys nothing.

READ IT THIS WAY (registered before looking):
  corr >= 0.80   the same edge expressed twice. Trade ONE -- whichever has
                 the better cost profile and the cleaner data story. The
                 second adds fees, not diversification.
  0.50-0.80      overlapping but not identical. A blend may help; the
                 honest test is whether combined SR beats the better leg by
                 more than the extra cost, which this prints.
  < 0.50         genuinely different exposures. Two streams beat one.

Note both series are gross of the correlation of their ERRORS: two rules on
overlapping universes in the same decade share regimes, so some correlation
is structural, not evidence of a shared signal.
"""
import sys

import numpy as np
import pandas as pd


def load(p):
    df = pd.read_csv(p)
    if "date" not in df.columns or "ret" not in df.columns:
        sys.exit(f"{p}: expected columns date,ret")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.set_index("date")["ret"].astype(float).sort_index()


def sr(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    per_yr = 12.0
    return float(x.mean() / x.std(ddof=1) * np.sqrt(per_yr))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    j = pd.concat({"a": a, "b": b}, axis=1).dropna()
    n = len(j)
    print(f"  {sys.argv[1]}: {len(a)} periods  {a.index.min().date()} .. "
          f"{a.index.max().date()}")
    print(f"  {sys.argv[2]}: {len(b)} periods  {b.index.min().date()} .. "
          f"{b.index.max().date()}")
    print(f"  overlap: {n} shared dates")
    if n < 24:
        sys.exit("  fewer than 24 shared periods -- correlation not "
                 "measurable. Align the rebalance calendars first.")

    c = float(j["a"].corr(j["b"]))
    cut = int(n * 0.7)
    c_ho = float(j["a"].iloc[cut:].corr(j["b"].iloc[cut:]))
    agree = float(((j["a"] > 0) == (j["b"] > 0)).mean())
    print(f"\n  corr (all {n})      {c:+.3f}")
    print(f"  corr (holdout {n - cut})  {c_ho:+.3f}")
    print(f"  hit agreement       {agree:.0%}")

    sa, sb = sr(j["a"]), sr(j["b"])
    blend = 0.5 * j["a"] + 0.5 * j["b"]
    sc = sr(blend)
    better = max(sa, sb)
    print(f"\n  SR a {sa:+.2f} | SR b {sb:+.2f} | 50/50 blend {sc:+.2f} "
          f"({sc - better:+.2f} vs better leg)")

    print("\n  ===== READING =====")
    if c >= 0.80:
        print(f"  SAME EDGE TWICE (corr {c:+.2f}). Trade one. The second "
              f"leg adds costs, not diversification.")
    elif c >= 0.50:
        print(f"  OVERLAPPING (corr {c:+.2f}). Blend only if the +"
              f"{sc - better:.2f} SR gain survives the extra turnover -- "
              f"and it is not free, both legs pay full spread.")
    else:
        print(f"  DISTINCT (corr {c:+.2f}). Two streams; the blend gain "
              f"({sc - better:+.2f}) is real diversification.")
    print("  Caveat: shared decade and overlapping universes create "
          "structural correlation independent of signal overlap.")


if __name__ == "__main__":
    main()
