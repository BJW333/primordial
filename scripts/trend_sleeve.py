"""
trend_sleeve.py -- a LONG-ONLY multi-asset trend-following sleeve to STACK on
top of all-US steady momentum, aiming at ~30%/yr with less pain than
plain leverage. No shorting anywhere.

SLEEVE (monthly, ETFs, prices include dividends):
  NONEQ  bonds / gold / commodities / currencies -- the assets that tend to
         trend when stocks don't:
         TLT IEF TIP LQD GLD SLV DBC USO DBA UUP FXY
  ALL    NONEQ + foreign stocks and real estate: EFA EEM VNQ
  Each month-end: an asset is HELD if its 12-month return beats cash (BIL,
  T-bills) -- otherwise its slot sits in cash. Slot sizes = inverse of each
  asset's 63-day volatility (equal RISK per asset), across ALL assets, so
  fewer trends = more cash. 0.05% cost per unit traded.

STACKING: momentum (data/momentum_v2_returns.csv, STEADY = the ~26%/yr
version) at 100% + the sleeve's return ABOVE CASH at k = 50/100/150/200%
of the account, minus 1%/yr financing spread per 100% (the way an overlay
is held with leverage). Also shown: momentum alone levered 1.15x / 1.3x.

PRE-STATED (primary = NONEQ at 100%): the sleeve is USEFUL if
  MOM + NONEQ 100% has a HIGHER Sharpe than MOM in BOTH 2017-02 -> 2021-05
  and 2021-06 -> 2026-05, AND its worst drop is no worse than MOM's overall.

Run (repo root, Alpaca keys exported; downloads ~15 ETF histories once):
    python3.10 scripts/trend_sleeve.py | tee logs/trend_sleeve.txt
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
import attention_short_test as AS     # noqa: E402
import factor_portfolio as FP         # noqa: E402

ROOT = os.path.dirname(HERE)
MOMF = os.path.join(ROOT, "data", "momentum_v2_returns.csv")
NONEQ = ["TLT", "IEF", "TIP", "LQD", "GLD", "SLV", "DBC", "USO", "DBA",
         "UUP", "FXY"]
EQ = ["EFA", "EEM", "VNQ"]
RECENT = pd.Period("2021-06", "M")
COST, SPREAD = 0.0005, 0.01
KS = (0.5, 1.0, 1.5, 2.0)


def load(syms):
    out = {}
    for s in syms + ["BIL"]:
        df = FP.alpaca(s)
        if df is None or not len(df):
            print(f"  (no data for {s})")
            continue
        out[s] = AS.norm_index(df)["close"]
    px = pd.DataFrame(out).sort_index().ffill()
    print(f"  ETFs: {', '.join(c for c in px.columns)} | "
          f"{px.index[0]:%Y-%m-%d} -> {px.index[-1]:%Y-%m-%d}")
    return px


def sleeve(px, names):
    names = [n for n in names if n in px.columns]
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
        iv = 1 / vol.iloc[i][names]
        w = (iv / iv.sum()).where(r12 > cash12, 0.0).fillna(0.0)
        cash_w = 1 - w.sum()
        m = px.loc[t2] / px.loc[t] - 1
        ret = (w * m[names]).sum() + cash_w * m["BIL"] - \
            COST * (w - prev).abs().sum()
        prev = w
        rows.append({"p": t2.to_period("M"), "ret": ret, "cash": m["BIL"],
                     "held": int((w > 0).sum())})
    return pd.DataFrame(rows).set_index("p")


def stats(x):
    x = x.dropna()
    ann = (1 + x).prod() ** (12 / len(x)) - 1
    vol = x.std() * np.sqrt(12)
    eq = (1 + x).cumprod()
    return ann, vol, x.mean() * 12 / vol if vol else np.nan, \
        (eq / eq.cummax() - 1).min()


def main():
    argparse.ArgumentParser().parse_args()
    m = pd.read_csv(MOMF, index_col=0, parse_dates=True)
    mom = m["STEADY"]
    mom.index = mom.index.to_period("M")
    px = load(NONEQ + EQ)
    sl = {"NONEQ": sleeve(px, NONEQ), "ALL": sleeve(px, NONEQ + EQ)}
    idx = mom.index.intersection(sl["NONEQ"].index)
    mom = mom.reindex(idx)
    cash = sl["NONEQ"]["cash"].reindex(idx)
    books = {"MOM": mom,
             "MOM x1.15 lev": 1.15 * mom - 0.15 * (cash + SPREAD / 12),
             "MOM x1.30 lev": 1.30 * mom - 0.30 * (cash + SPREAD / 12)}
    for name, s in sl.items():
        sr = s["ret"].reindex(idx)
        books[f"{name} alone"] = sr
        for k in KS:
            books[f"MOM + {name} {k:.0%}"] = mom + k * (sr - cash) - \
                k * SPREAD / 12
    print(f"\n  {len(idx)} months {idx[0]} -> {idx[-1]} | NONEQ holds avg "
          f"{sl['NONEQ']['held'].mean():.1f} of {len(NONEQ)} assets")
    for name, s in sl.items():
        c = pd.concat([mom, s["ret"].reindex(idx)], axis=1).corr().iloc[0, 1]
        print(f"  correlation {name} vs momentum: {c:+.2f}")
    print(f"\n  {'book':<18} | {'all ann':>8} {'vol':>5} {'sh':>5} "
          f"{'worst':>6} | {'2017-21':>8} {'sh':>5} | {'last 5y':>8} {'sh':>5}"
          f" {'worst':>6}")
    res = {}
    for k, b in books.items():
        a = stats(b)
        s0 = stats(b[b.index < RECENT])
        s1 = stats(b[b.index >= RECENT])
        res[k] = (a, s0, s1)
        flag = "  <- 30%+" if a[0] >= 0.30 and s1[0] >= 0.30 else ""
        print(f"  {k:<18} | {a[0]*100:>+7.1f}% {a[1]*100:>4.0f}% "
              f"{a[2]:>5.2f} {a[3]*100:>5.0f}% | {s0[0]*100:>+7.1f}% "
              f"{s0[2]:>5.2f} | {s1[0]*100:>+7.1f}% {s1[2]:>5.2f} "
              f"{s1[3]*100:>5.0f}%{flag}")
    cols = ["MOM", "NONEQ alone", "MOM + NONEQ 100%", "MOM x1.15 lev"]
    yr = pd.DataFrame({c: books[c] for c in cols})
    yr = (1 + yr).groupby(yr.index.year).prod() - 1
    print("\n  by year:  " + "".join(f"{c:>18}" for c in cols))
    for y, row in yr.iterrows():
        print(f"  {y:<8} " + "".join(f"{x*100:>+17.1f}%" for x in row))
    (ma, m0, m1), (pa, p0, p1) = res["MOM"], res["MOM + NONEQ 100%"]
    ok = p0[2] > m0[2] and p1[2] > m1[2] and pa[3] >= ma[3]
    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"  NONEQ trend sleeve useful on top of momentum  "
          f"{'PASS' if ok else 'FAIL'}")
    print("  (backtest; expect live results below these numbers)")


if __name__ == "__main__":
    main()
