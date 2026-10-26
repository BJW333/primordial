"""
last5.py -- re-score every saved portfolio on the LAST FIVE YEARS only
(2021-06 -> 2026-05), plus "SPY scaled up to the same risk" for each.

Reads the monthly-return files the earlier runs already saved -- no new
downloads:
  data/turbo_portfolio_returns.csv      MOM, MOM_BIG, MOM_VM, MOM_BIG_VM
  data/factor_portfolio_returns.csv     factor portfolios
  data/quality_portfolio_returns.csv    QUALITY, QUALITY+MOM
  data/residual_momentum_returns.csv    RESMOM_BIG, RESMOM (if run)

For each portfolio: annual return, volatility, Sharpe, worst drop, excess
vs SPY per year (t), and SPY_matched = SPY leveraged to the portfolio's
volatility (borrowing above 1x charged 4%/yr). Beating SPY_matched is the
real test: it means the picker beats just buying more SPY.

Run (repo root):
    python3.10 scripts/last5.py
    python3.10 scripts/last5.py --start 2021-06-01
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["turbo_portfolio_returns.csv", "factor_portfolio_returns.csv",
         "quality_portfolio_returns.csv", "residual_momentum_returns.csv"]
LEV_COST = 0.04
SKIP = ("_turn", "_scale", "coverage")


def stats(r):
    r = r.dropna()
    n = len(r)
    ann = (1 + r).prod() ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sh = r.mean() * 12 / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return ann, vol, sh, dd


def tstat(x):
    x = x.dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    a = ap.parse_args()
    start = pd.Timestamp(a.start)
    rows = []
    for f in FILES:
        p = os.path.join(a.data, f)
        if not os.path.exists(p):
            print(f"  (skip {f}: not found)")
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df = df[df.index >= start]
        if "SPY" not in df or df.empty:
            continue
        spy = df["SPY"]
        _, spy_vol, _, _ = stats(spy)
        for c in df.columns:
            if c in ("SPY", "risk_on") or any(c.endswith(s) for s in SKIP):
                continue
            r = df[c]
            ann, vol, sh, dd = stats(r)
            ex = r - spy
            lev = vol / spy_vol
            matched = lev * spy - max(lev - 1, 0) * LEV_COST / 12
            m_ann, _, m_sh, m_dd = stats(matched)
            rows.append((f.replace("_returns.csv", ""), c, ann, vol, sh, dd,
                         ex.mean() * 12, tstat(ex), lev, m_ann, m_dd))
        spy_row = stats(spy)
    if not rows:
        raise SystemExit("no saved return files found in " + a.data)

    n = len(df)
    print(f"\n  LAST FIVE YEARS: {df.index[0].date()} -> {df.index[-1].date()}"
          f" ({n} months)")
    s = spy_row
    print(f"  SPY: {s[0]*100:+.1f}%/yr, vol {s[1]*100:.1f}%, Sharpe "
          f"{s[2]:.2f}, worst drop {s[3]*100:.1f}%\n")
    print(f"  {'file':<18} {'portfolio':<13} {'annual':>7} {'vol':>6} "
          f"{'sharpe':>6} {'worst':>7} {'vs SPY':>7} {'t':>6} | "
          f"{'SPY x lev':>9} {'same-risk SPY':>13} {'beats it?':>9}")
    for (fn, c, ann, vol, sh, dd, ex, t, lev, m_ann, m_dd) in rows:
        win = "YES" if ann > m_ann else "no"
        print(f"  {fn:<18} {c:<13} {ann*100:>+6.1f}% {vol*100:>5.1f}% "
              f"{sh:>6.2f} {dd*100:>6.1f}% {ex*100:>+6.1f}% {t:>+6.2f} | "
              f"{lev:>8.2f}x {m_ann*100:>+12.1f}% {win:>9}")
    print("\n  'beats it?' = portfolio return > SPY leveraged to the same "
          "volatility (4%/yr borrow). NO means: just buy more SPY.")


if __name__ == "__main__":
    main()
