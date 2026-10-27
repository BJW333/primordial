"""
adx_obv_variants.py -- DIAGNOSTIC (no pass/fail): is the ADX+OBV filter
itself useful, or did "top 20, equal weight" drive the pattern seen in
adline_obv.py (lags tech-led bull years, wins stress years)?

Same S&P 500 point-in-time universe, Alpaca daily OHLCV incl. delisted.
QUALIFIER at a month-end: ADX(14) > 25, +DI > -DI, OBV above its 50-day
average. Monthly rebalance, 10 bps per unit of weight traded.
SIZE = 63-day average dollar volume (a stand-in for company size; the
BASE_SIZE control checks it tracks SPY).

  BASE_SIZE   control: ALL members, size-weighted      (should ~ SPY)
  BASE_EW     control: ALL members, equal weight       (~ equal-weight S&P)
  ALL_SIZE    every qualifier, size-weighted   -> compare with BASE_SIZE
  ALL_EW      every qualifier, equal weight    -> compare with BASE_EW
  TOP50_SIZE  50 highest-ADX qualifiers, size-weighted
  TOP20_EW    the original (adline_obv.py S_ADX_OBV, no SPY fill)

Run (repo root; prices cached; keys only checked):
    python3.10 scripts/adx_obv_variants.py | tee logs/adx_obv_variants.txt
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
import adline_obv as X        # noqa: E402
import blowup_scan as B       # noqa: E402
import commodity_adx as A     # noqa: E402

VARS = ["BASE_SIZE", "BASE_EW", "ALL_SIZE", "ALL_EW", "TOP50_SIZE",
        "TOP20_EW"]
COST = 0.0010


def run(px, H, L, V, spy, members):
    idx = px.index
    a, pdi, ndi = A.adx(H, L, px)
    obv = X.obv_up(px, V)
    size = (px * V).rolling(63, min_periods=40).mean()
    spy = spy.reindex(idx).ffill()
    me = [d for d in px.resample("ME").last().index if d >= X.S_FIRST]
    rows, prev = [], {k: pd.Series(dtype=float) for k in VARS}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        m = members(t)
        mem = [s for s in px.columns if s in m and np.isfinite(px.at[t, s])
               and np.isfinite(size.at[t, s])]
        at, pt, nt, ot = a.loc[t], pdi.loc[t], ndi.loc[t], obv.loc[t]
        q = [s for s in mem if np.isfinite(at[s]) and at[s] > 25
             and pt[s] > nt[s] and bool(ot[s])]
        top50 = list(at[q].sort_values(ascending=False).index[:50])
        top20 = list(at[q].sort_values(ascending=False).index[:20])
        seg = px.loc[t:t2, mem]
        mret = (seg.ffill().iloc[-1] / seg.iloc[0] - 1).fillna(0)
        sz = size.loc[t]

        def sw(names):
            w = sz[names].clip(lower=0)
            return w / w.sum() if w.sum() > 0 else w

        def ew(names):
            return pd.Series(1 / len(names), names) if names else \
                pd.Series(dtype=float)
        books = {"BASE_SIZE": sw(mem), "BASE_EW": ew(mem),
                 "ALL_SIZE": sw(q), "ALL_EW": ew(q),
                 "TOP50_SIZE": sw(top50), "TOP20_EW": ew(top20)}
        row = {"month": t2, "SPY": spy.asof(t2) / spy.asof(t) - 1,
               "n_qual": len(q), "n_mem": len(mem)}
        for name, w in books.items():
            old = prev[name]
            turn = w.subtract(old, fill_value=0).abs().sum()
            row[name] = float((w * mret.reindex(w.index).fillna(0)).sum()) \
                - turn * COST
            prev[name] = w
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def report(res):
    print(f"\n  qualifiers per month: avg {res['n_qual'].mean():.0f} of "
          f"{res['n_mem'].mean():.0f} members (min {res['n_qual'].min()}, "
          f"max {res['n_qual'].max()})")
    print(f"\n  {'portfolio':<11} | {'2017-21 ann':>11} {'sh':>5} {'worst':>6}"
          f" | {'last5y ann':>10} {'sh':>5} {'worst':>6} | {'10y growth':>10}")
    for k in ["SPY"] + VARS:
        o = res.loc[res.index < X.S_RECENT, k]
        n = res.loc[res.index >= X.S_RECENT, k]
        a0, _, s0, d0 = B.stats(o)
        a1, _, s1, d1 = B.stats(n)
        g = (1 + res[k]).prod()
        print(f"  {k:<11} | {a0*100:>+10.1f}% {s0:>5.2f} {d0*100:>5.0f}% | "
              f"{a1*100:>+9.1f}% {s1:>5.2f} {d1*100:>5.0f}% | {g:>9.2f}x")
    print(f"\n  by year:  {'year':<5} " + " ".join(f"{k:>10}" for k in
                                                   ["SPY"] + VARS))
    for y, g in res.groupby(res.index.year):
        print(f"            {y:<5} " + " ".join(
            f"{((1 + g[k]).prod() - 1)*100:>+9.1f}%" for k in ["SPY"] + VARS))
    print("\n  READ: ALL_SIZE vs BASE_SIZE and ALL_EW vs BASE_EW isolate the "
          "filter itself;")
    print("        BASE_SIZE vs SPY checks the size stand-in; BASE_EW vs "
          "BASE_SIZE = the equal-weight effect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        import turbo_portfolio as T
        px, vol, spy, members = T.smoke()
        res = run(px, px * 1.01, px * 0.99, vol, spy, members)
    else:
        if not os.environ.get("ALPACA_API_KEY"):
            sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
        res = run(*X.load_stocks())
        res.to_csv(os.path.join(X.C.ROOT, "data", "adx_obv_variants.csv"))
    report(res)


if __name__ == "__main__":
    main()
