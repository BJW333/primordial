"""
vol_reduce.py -- LOWER THE VOLATILITY of the stack (momentum + trend +
fade) without giving up the return-per-risk. Momentum (~33%/yr swings,
top 20 stocks) is almost all of the stack's risk; trend and fade swing
~12-13%.

IDEAS (all rules fixed before running):
  A  MORE STOCKS: momentum top 40 and top 60 (same engine as momentum_v2.py,
     only TOP changed) vs top 20.
  B  VOL-TARGETED MOMENTUM: each month, exposure = 20% / momentum's realized
     volatility over the prior 12 months (monthly returns), capped 0.5x-1.5x;
     unused money in T-bills, borrowed money at T-bills + 1%.
  C  RISK-BALANCED STACK: engine sizes inverse to their own volatility
     (measured over the first 24 months only, then frozen): momentum gets
     less, trend/fade more; then the whole mix scaled to 20%/yr volatility
     (scale also measured on the first 24 months only).
  D  VOL-TARGETED STACK: current stack (MOM + TREND 50% + FADE 50%) with the
     same B-style targeting at 20%/yr on the whole portfolio.

PRE-STATED: an idea WORKS if it cuts volatility by >= 25% vs the current
stack while keeping Sharpe >= the current stack's in BOTH 2017-02 -> 2021-05
and 2021-06 -> 2026-05.

Run (repo root, Alpaca keys exported; reruns momentum at top 40/60, ~10 min):
    python3.10 scripts/vol_reduce.py | tee logs/vol_reduce.txt
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_us as U                 # noqa: E402
import momentum_v2 as MV              # noqa: E402
import portfolio_combo as PC          # noqa: E402
import trend_ls as TL                 # noqa: E402
import trend_sleeve as TS             # noqa: E402

SP = 0.01 / 12
TV, LO, HI = 0.20, 0.5, 1.5
RECENT = TS.RECENT


def momentum_top(n):
    p = os.path.join(TS.ROOT, "data", f"momentum_top{n}.csv")
    if os.path.exists(p):
        s = pd.read_csv(p, index_col=0, parse_dates=True)["STEADY"]
    else:
        px, vol, spy, feed = U.load()
        U.MIN_PX = MV.MIN_PX
        sets, ever = U.liquid_members(px, vol, MV.MIN_DV if feed == "sip"
                                      else MV.MIN_DV * 0.03)
        px = px[ever]
        spy = spy.reindex(px.index).ffill()
        days = sorted(sets)

        def members(t):
            past = [d for d in days if d <= t]
            return sets[past[-1]] if past else set()
        MV.TOP = n
        res, _ = MV.run(px, spy, members)
        MV.TOP = 20
        res.to_csv(p)
        s = res["STEADY"]
    s.index = s.index.to_period("M")
    return s


def target(x, cash, tv=TV):
    """Scale by tv / trailing-12-month vol, measured BEFORE each month."""
    rv = x.rolling(12, min_periods=12).std().shift(1) * np.sqrt(12)
    k = (tv / rv).clip(LO, HI).fillna(1.0)
    return k * x + (1 - k) * cash - (k - 1).clip(lower=0) * SP, k


def stats_row(lab, b, base=None):
    a, s0, s1 = TS.stats(b), TS.stats(b[b.index < RECENT]), \
        TS.stats(b[b.index >= RECENT])
    print(f"  {lab:<34} | {a[0]*100:>+7.1f}% {a[1]*100:>4.0f}% {a[2]:>5.2f} "
          f"{a[3]*100:>5.0f}% | {s0[0]*100:>+7.1f}% {s0[2]:>5.2f} | "
          f"{s1[0]*100:>+7.1f}% {s1[2]:>5.2f} {s1[3]*100:>5.0f}%")
    return a, s0, s1


def main():
    m20 = pd.read_csv(TS.MOMF, index_col=0, parse_dates=True)["STEADY"]
    m20.index = m20.index.to_period("M")
    m40, m60 = momentum_top(40), momentum_top(60)
    px = TS.load([x for x in TL.MKTS if x != "SPY"] + ["SPY"])
    sl = TL.sleeve(px)
    tr, cash = sl["ret"], sl["cash"]
    fd = PC.monthly(PC.fade_daily())
    idx = m20.index.intersection(tr.index).intersection(fd.index) \
        .intersection(m40.index).intersection(m60.index)
    m20, m40, m60, tr, cash, fd = (s.reindex(idx) for s in
                                   (m20, m40, m60, tr, cash, fd))

    def stack(m):
        return m + 0.5 * tr + 0.5 * fd - SP

    books = {}
    books["CURRENT: MOM20 + T50 + F50"] = stack(m20)
    books["MOM top 20 alone"] = m20
    books["A  MOM top 40 alone"] = m40
    books["A  MOM top 60 alone"] = m60
    books["A  stack with MOM top 40"] = stack(m40)
    books["A  stack with MOM top 60"] = stack(m60)
    vm, k = target(m20, cash)
    books["B  vol-targeted MOM alone"] = vm
    books["B  stack with vol-targeted MOM"] = vm + 0.5 * tr + 0.5 * fd - SP
    first = idx[:24]
    vols = {"m": m20.loc[first].std(), "t": tr.loc[first].std(),
            "f": fd.loc[first].std()}
    inv = {k_: 1 / v for k_, v in vols.items()}
    tot = sum(inv.values())
    w = {k_: v / tot for k_, v in inv.items()}
    raw = w["m"] * m20 + w["t"] * tr + w["f"] * fd
    scale = TV / (raw.loc[first].std() * np.sqrt(12))
    books[f"C  risk-balanced (M{w['m']*scale:.2f}/T{w['t']*scale:.2f}/"
          f"F{w['f']*scale:.2f})"] = scale * raw - max(scale - 1, 0) * SP
    vs, ks = target(books["CURRENT: MOM20 + T50 + F50"], cash)
    books["D  vol-targeted whole stack"] = vs

    print(f"\n  {len(idx)} months {idx[0]} -> {idx[-1]} | B avg exposure "
          f"{k.mean():.2f}x, D avg {ks.mean():.2f}x")
    print(f"\n  {'book':<34} | {'all ann':>8} {'vol':>5} {'sh':>5} {'worst':>6}"
          f" | {'2017-21':>8} {'sh':>5} | {'last 5y':>8} {'sh':>5} {'worst':>6}")
    res = {k_: stats_row(k_, b) for k_, b in books.items()}
    base = res["CURRENT: MOM20 + T50 + F50"]
    print("\n  ===== VERDICT (pre-stated: vol cut >= 25% AND Sharpe >= current "
          "in both periods) =====")
    for k_, (a, s0, s1) in res.items():
        if k_.startswith(("A  stack", "B  stack", "C", "D")):
            ok = (a[1] <= 0.75 * base[0][1] and s0[2] >= base[1][2]
                  and s1[2] >= base[2][2])
            print(f"  {k_:<34} vol {a[1]*100:>3.0f}% vs {base[0][1]*100:.0f}% "
                  f"-> {'WORKS' if ok else 'no'}")
    print("  (backtest; FADE modeled as shorting; ETF stand-ins for futures)")


if __name__ == "__main__":
    main()
