"""
turbo_oos.py -- out-of-sample test of the two momentum candidates on
2010-2016, years no portfolio test has touched.

Candidates (frozen now, from turbo_portfolio.py, commit 8b99af4):
  MOM_BIG      100 largest S&P 500 members (63-day dollar volume), top 20
               by 12-1 momentum, equal weight, monthly, 10 bps/side
  MOM_BIG_VM24 the same, exposure scaled to min(24% / realised vol of the
               last 126 daily returns, 1.5); 24% = momentum's own natural
               vol, set BEFORE this run; leverage charged 4%/yr

Data: S&P 500 point-in-time membership + Yahoo prices (Alpaca starts in
2016). Yahoo is missing many dead S&P names, but this test only buys from
the 100 LARGEST members -- giants rarely die, so the gap matters far less
here than in the dip tests. Coverage is printed.

PRE-STATED (2010-01 -> 2016-12, benchmark SPY)
  REPLICATES if a candidate beats SPY (mean monthly excess > 0) AND its
  Sharpe is at least SPY's. t is printed but not required: 7 years can't
  give t > 2 for an edge of this size even if it is real -- direction is
  what an out-of-sample check can honestly confirm.
  Both replicate -> paper trade both. One -> paper trade that one.
  Neither -> the 2017-2026 result was that decade, not momentum.

Run (repo root; Yahoo prices are cached from the dip runs):
    python3.10 scripts/turbo_oos.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import factor_portfolio as FP  # noqa: E402
import turbo_portfolio as T   # noqa: E402

ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from primordial.data import fetch  # noqa: E402

Y_START, Y_END = "2009-01-01", "2026-06-01"     # = existing Yahoo cache keys
OOS_END = pd.Timestamp("2017-01-31")


def main():
    raw_members, union = FP.membership()
    members = lambda d: {s.replace(".", "-") for s in raw_members(d)}  # noqa
    syms = [s.replace(".", "-") for s in union]
    print(f"  {len(syms)} names ever in the S&P 500 -- Yahoo prices (cache)")
    data = fetch("yfinance", syms, "1d", Y_START, Y_END, min_bars=60)
    spy = fetch("yfinance", ["SPY"], "1d", Y_START, Y_END)["SPY"]["close"]
    px = pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()
    vol = pd.DataFrame({s: d["volume"] for s, d in data.items()}).reindex(
        px.index)
    px, vol = px.loc[:OOS_END], vol.loc[:OOS_END]

    # coverage: share of members each year-end that have price data
    cov = []
    for y in range(2010, 2017):
        d = px.index[px.index.year == y][-1]
        m = members(d)
        cov.append(f"{y}:{sum(s in px.columns and np.isfinite(px.at[d, s]) for s in m)}/{len(m)}")  # noqa
    print("  members with prices: " + "  ".join(cov))

    FP.FIRST = pd.Timestamp("2010-01-29")
    T.TARGET_VOL = 0.24                          # frozen before this run
    res = T.run(px, vol, spy, members)
    res = res[res.index <= OOS_END]
    res = res.rename(columns={"MOM_BIG_VM": "MOM_BIG_VM24",
                              "MOM_BIG_VM_scale": "MOM_BIG_VM24_scale"})

    print(f"\n  {len(res)} monthly rebalances, {res.index[0].date()} -> "
          f"{res.index[-1].date()} (never used by any portfolio test)\n")
    print(f"  {'portfolio':<13} {'annual':>8} {'vol':>7} {'sharpe':>7} "
          f"{'max DD':>8} {'vs SPY/yr':>10} {'t':>6} {'exposure':>9}")
    out = {}
    for col in ("MOM_BIG", "MOM_BIG_VM24", "MOM", "SPY", "EW"):
        a, v, s, dd = FP.stats(res[col])
        ex = res[col] - res["SPY"]
        exs = (f"{ex.mean()*1200:>+9.1f}% {FP.tstat(ex):>+6.2f}"
               if col != "SPY" else f"{'':>10} {'':>6}")
        sc = (f"{res[col + '_scale'].mean():>8.2f}x"
              if col + "_scale" in res else "")
        print(f"  {col:<13} {a*100:>+7.1f}% {v*100:>6.1f}% {s:>7.2f} "
              f"{dd*100:>7.1f}% {exs} {sc}")
        out[col] = (ex.mean(), s)

    s_spy = FP.stats(res["SPY"])[2]
    print("\n  ===== VERDICT (pre-stated, 2010-2016 out of sample) =====")
    winners = []
    for c in ("MOM_BIG", "MOM_BIG_VM24"):
        ok = out[c][0] > 0 and out[c][1] >= s_spy
        winners += [c] if ok else []
        print(f"  {c:<13} beats SPY: {'YES' if out[c][0] > 0 else 'no '} | "
              f"Sharpe {out[c][1]:.2f} vs SPY {s_spy:.2f}: "
              f"{'YES' if out[c][1] >= s_spy else 'no'} -> "
              f"{'REPLICATES' if ok else 'does not replicate'}")
    if winners:
        print(f"  NEXT: paper trade {' and '.join(winners)} on Alpaca.")
    else:
        print("  NEITHER replicates -- the 2017-2026 result was that decade's "
              "mega-cap run, not a durable momentum edge.")


if __name__ == "__main__":
    main()
