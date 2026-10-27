"""
trend_ls.py -- TWO-WAY trend following across stocks, bonds, commodities,
currencies and foreign markets (the managed-futures style), stacked on
momentum. trend_sleeve.py showed the LONG-ONLY version is too weak; this
tests the full version before anything is built with real futures.

ETF prices stand in for futures (daily, dividends included). A futures
position earns the asset's return ABOVE CASH whether long or short, so:
position return = sign x weight x (ETF return - T-bill return).

MARKETS (14):  stocks  SPY EFA EEM
               bonds   TLT IEF TIP
               commods GLD SLV DBC USO DBA
               FX      UUP FXY FXE
RULE (monthly, fixed before running): sign = +1 if the 12-month return
beats cash, else -1 (bet against it). Each market sized to the SAME RISK:
weight = 0.30 / its 63-day volatility / 14 (capped at 1.0 each), which
makes the whole sleeve swing roughly 8-12%/yr. 0.05% cost per unit traded.

STACK: momentum (STEADY, ~26.6%/yr) 100% + sleeve at 50/100/150%, minus 1%/yr
financing spread per 100% (futures need little cash; this is conservative).

PRE-STATED: the sleeve ADDS if MOM + TREND 100% beats momentum alone on
Sharpe in BOTH 2017-02 -> 2021-05 and 2021-06 -> 2026-05 AND on overall
annual return.

Run (repo root, Alpaca keys exported):
    python3.10 scripts/trend_ls.py | tee logs/trend_ls.txt
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trend_sleeve as TS             # noqa: E402

MKTS = ["SPY", "EFA", "EEM", "TLT", "IEF", "TIP", "GLD", "SLV", "DBC", "USO",
        "DBA", "UUP", "FXY", "FXE"]
TARGET, CAP, COST, SPREAD = 0.30, 1.0, 0.0005, 0.01
KS = (0.5, 1.0, 1.5)
RECENT = TS.RECENT


def sleeve(px):
    names = [n for n in MKTS if n in px.columns]
    me = px.resample("ME").last().index
    r = px.pct_change()
    vol = r.rolling(63, min_periods=50).std() * np.sqrt(252)
    rows, prev = [], pd.Series(0.0, index=names)
    for k in range(len(me) - 1):
        t = px.index[px.index <= me[k]][-1]
        t2 = px.index[px.index <= me[k + 1]][-1]
        i = px.index.get_loc(t)
        if i < 252:
            continue
        r12 = px.iloc[i][names] / px.iloc[i - 252][names] - 1
        cash12 = px.iloc[i]["BIL"] / px.iloc[i - 252]["BIL"] - 1
        sign = np.where(r12 > cash12, 1.0, -1.0)
        w = (TARGET / vol.iloc[i][names] / len(names)).clip(upper=CAP)
        w = (w * sign).fillna(0.0)
        m = px.loc[t2] / px.loc[t] - 1
        ex = m[names] - m["BIL"]
        ret = (w * ex).sum() - COST * (w - prev).abs().sum()
        prev = w
        rows.append({"p": t2.to_period("M"), "ret": ret, "cash": m["BIL"],
                     "long": int((w > 0).sum()), "short": int((w < 0).sum()),
                     "gross": w.abs().sum()})
    return pd.DataFrame(rows).set_index("p")


def main():
    argparse.ArgumentParser().parse_args()
    m = pd.read_csv(TS.MOMF, index_col=0, parse_dates=True)
    mom = m["STEADY"]
    mom.index = mom.index.to_period("M")
    px = TS.load([x for x in MKTS if x != "SPY"] + ["SPY"])
    sl = sleeve(px)
    idx = mom.index.intersection(sl.index)
    mom, tr, cash = mom.reindex(idx), sl["ret"].reindex(idx), \
        sl["cash"].reindex(idx)
    print(f"\n  {len(idx)} months {idx[0]} -> {idx[-1]} | avg {sl['long'].mean():.1f}"
          f" long / {sl['short'].mean():.1f} short, gross {sl['gross'].mean():.2f}x"
          f" | correlation vs momentum "
          f"{pd.concat([mom, tr], axis=1).corr().iloc[0, 1]:+.2f}")
    books = {"MOM": mom,
             "MOM x1.15 lev": 1.15 * mom - 0.15 * (cash + SPREAD / 12),
             "TREND alone (excess)": tr}
    for k in KS:
        books[f"MOM + TREND {k:.0%}"] = mom + k * tr - k * SPREAD / 12
    print(f"\n  {'book':<22} | {'all ann':>8} {'vol':>5} {'sh':>5} "
          f"{'worst':>6} | {'2017-21':>8} {'sh':>5} | {'last 5y':>8} {'sh':>5}"
          f" {'worst':>6}")
    res = {}
    for k, b in books.items():
        a, s0, s1 = TS.stats(b), TS.stats(b[b.index < RECENT]), \
            TS.stats(b[b.index >= RECENT])
        res[k] = (a, s0, s1)
        flag = "  <- 30%+" if a[0] >= 0.30 and s1[0] >= 0.30 else ""
        print(f"  {k:<22} | {a[0]*100:>+7.1f}% {a[1]*100:>4.0f}% "
              f"{a[2]:>5.2f} {a[3]*100:>5.0f}% | {s0[0]*100:>+7.1f}% "
              f"{s0[2]:>5.2f} | {s1[0]*100:>+7.1f}% {s1[2]:>5.2f} "
              f"{s1[3]*100:>5.0f}%{flag}")
    cols = ["MOM", "TREND alone (excess)", "MOM + TREND 100%"]
    yr = pd.DataFrame({c: books[c] for c in cols})
    yr = (1 + yr).groupby(yr.index.year).prod() - 1
    print("\n  by year:  " + "".join(f"{c[:20]:>21}" for c in cols))
    for y, row in yr.iterrows():
        print(f"  {y:<8} " + "".join(f"{x*100:>+20.1f}%" for x in row))
    (ma, m0, m1), (pa, p0, p1) = res["MOM"], res["MOM + TREND 100%"]
    ok = p0[2] > m0[2] and p1[2] > m1[2] and pa[0] > ma[0]
    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"  two-way trend sleeve adds to momentum  {'PASS' if ok else 'FAIL'}")
    print("  (ETF stand-ins for futures; backtest -- expect less live)")


if __name__ == "__main__":
    main()
