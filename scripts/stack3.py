"""
stack3.py -- SIZING STUDY (no pass/fail): momentum + two-way futures trend +
attention fade, every reasonable combination, side by side.

Pieces (same as before, nothing re-tuned):
  MOM    all-US steady momentum, STEADY (data/momentum_v2_returns.csv)
  TREND  two-way trend sleeve, 14 markets (trend_ls.py), return above cash
  FADE   liquid-MID weak-trend short book + 50% stop (fade_stop.py primary),
         hedged; modeled as stock shorting (a puts version would differ)
Overlays (TREND, FADE) are charged 1%/yr financing per 100% of size.

Run (repo root, Alpaca keys exported):
    python3.10 scripts/stack3.py | tee logs/stack3.txt
"""
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import portfolio_combo as PC          # noqa: E402
import trend_ls as TL                 # noqa: E402
import trend_sleeve as TS             # noqa: E402

SP = 0.01 / 12
COMBOS = [  # (label, trend size, fade size)
    ("MOM alone", 0, 0),
    ("MOM + TREND 100%", 1.0, 0),
    ("MOM + FADE 50%", 0, 0.5),
    ("MOM + FADE 100%", 0, 1.0),
    ("MOM + TREND 50% + FADE 50%", 0.5, 0.5),
    ("MOM + TREND 100% + FADE 50%", 1.0, 0.5),
    ("MOM + TREND 100% + FADE 100%", 1.0, 1.0),
]


def main():
    m = pd.read_csv(TS.MOMF, index_col=0, parse_dates=True)["STEADY"]
    m.index = m.index.to_period("M")
    px = TS.load([x for x in TL.MKTS if x != "SPY"] + ["SPY"])
    tr = TL.sleeve(px)["ret"]
    fd = PC.monthly(PC.fade_daily())
    idx = m.index.intersection(tr.index).intersection(fd.index)
    m, tr, fd = m.reindex(idx), tr.reindex(idx), fd.reindex(idx)
    c = pd.concat([m, tr, fd], axis=1, keys=["MOM", "TREND", "FADE"]).corr()
    print(f"\n  {len(idx)} months {idx[0]} -> {idx[-1]}")
    print("  correlations: " + "  ".join(
        f"{a}/{b} {c.loc[a, b]:+.2f}" for a, b in
        (("MOM", "TREND"), ("MOM", "FADE"), ("TREND", "FADE"))))
    print(f"\n  {'stack':<30} | {'all ann':>8} {'sh':>5} {'worst':>6} | "
          f"{'2017-21':>8} | {'last 5y':>8} {'worst':>6} | {'needs':<20}")
    books = {}
    for lab, kt, kf in COMBOS:
        b = m + kt * tr + kf * fd - (kt + kf) * SP
        books[lab] = b
        a, s0, s1 = TS.stats(b), TS.stats(b[b.index < TS.RECENT]), \
            TS.stats(b[b.index >= TS.RECENT])
        need = ("normal account" if kt == 0 and kf == 0 else
                ("futures" if kf == 0 else
                 ("shorting/puts" if kt == 0 else "futures + shorting/puts")))
        print(f"  {lab:<30} | {a[0]*100:>+7.1f}% {a[2]:>5.2f} "
              f"{a[3]*100:>5.0f}% | {s0[0]*100:>+7.1f}% | "
              f"{s1[0]*100:>+7.1f}% {s1[3]*100:>5.0f}% | {need:<20}")
    yr = pd.DataFrame(books)
    yr = (1 + yr).groupby(yr.index.year).prod() - 1
    cols = ["MOM alone", "MOM + TREND 100%", "MOM + TREND 100% + FADE 50%",
            "MOM + TREND 100% + FADE 100%"]
    print("\n  by year:  " + "".join(f"{x[:26]:>27}" for x in cols))
    for y, row in yr[cols].iterrows():
        print(f"  {y:<8} " + "".join(f"{v*100:>+26.1f}%" for v in row))
    print("\n  SIZING STUDY -- FADE stop is post-hoc; ETF stand-ins for futures;"
          " expect live results well below these.")


if __name__ == "__main__":
    main()
