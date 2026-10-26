"""
residual_momentum.py -- momentum with the market exposure stripped out
(Blitz, Huij & Martens 2011), on the survivorship-free S&P 500 machine.
DECIDING WINDOW: the last five years.

Why: raw momentum's extra return on 2026-10-25 turned out to be extra
market risk (it bought high-beta stocks). Residual momentum ranks each
stock on how much it beat what its market exposure alone predicts.

RESIDUAL MOMENTUM SCORE at each month-end, per stock:
  regress its daily returns on SPY's over days t-252 .. t-21 (skip the
  last month, like 12-1 momentum); score = sum of residuals / standard
  deviation of residuals (beat-the-market, per unit of its own noise).
  Needs a full 231-day window.

VARIANTS (monthly, top 20 equal weight, 10 bps/side, PIT membership,
Alpaca prices incl. delisted)
  RESMOM_BIG  residual momentum within the 100 largest  -- PRIMARY
  RESMOM      residual momentum, whole list
  MOM_BIG     plain momentum within the 100 largest (yesterday's candidate)

PRE-STATED
  window  2021-06-01 -> 2026-05-29 (last five years) DECIDES;
          2017-02 -> 2021-05 shown as context only
  PASS    RESMOM_BIG beats SPY on annual return AND on Sharpe in the
          deciding window. t vs SPY is printed (60 months is short).
  Pass -> earnings-surprise signal next, then paper trading.

Run (repo root, Alpaca keys exported; prices cached):
    python3.10 scripts/residual_momentum.py
    python3.10 scripts/residual_momentum.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import factor_portfolio as FP  # noqa: E402
import turbo_portfolio as T   # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "residual_momentum_returns.csv")
RECENT = pd.Timestamp("2021-06-01")
VARIANTS = ["RESMOM_BIG", "RESMOM", "MOM_BIG"]


def resmom(px, spy_r, t_i, names):
    """Residual 12-1 momentum, vectorised over names with full windows."""
    lo, hi = t_i - 252, t_i - 21
    if lo < 1:
        return pd.Series(dtype=float)
    R = px[names].iloc[lo - 1:hi + 1].pct_change(fill_method=None).iloc[1:]
    x = spy_r.reindex(R.index).values
    R = R.loc[:, R.notna().all()]
    if R.empty or np.isnan(x).any():
        return pd.Series(dtype=float)
    Y = R.values
    xm = x - x.mean()
    beta = (xm[:, None] * (Y - Y.mean(0))).sum(0) / (xm ** 2).sum()
    alpha = Y.mean(0) - beta * x.mean()
    res = Y - alpha - np.outer(x, beta)
    sd = res.std(0, ddof=1)
    return pd.Series(res.sum(0) / np.where(sd > 0, sd, np.nan), R.columns)


def run(px, vol, spy, members):
    me = [d for d in px.resample("ME").last().index if d >= FP.FIRST]
    idx = px.index
    spy_r = spy.reindex(idx).ffill().pct_change(fill_method=None)
    dv = (px * vol).rolling(63, min_periods=40).mean()
    rows, prev = [], {}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        mom, _lv, c = FP.scores(px, t_i)
        if mom is None:
            continue
        mem = members(t)
        hist = px.iloc[:t_i + 1].notna().sum()
        elig = [s for s in px.columns
                if s in mem and np.isfinite(c.get(s, np.nan))
                and hist[s] >= 253 and np.isfinite(mom.get(s, np.nan))]
        if len(elig) < 150:
            continue
        rm = resmom(px, spy_r, t_i, elig).dropna()
        size = dv.loc[t, elig].dropna()
        big = list(size.sort_values(ascending=False).index[:T.BIG_N])
        seg = px.loc[t:t2, elig]
        mret = seg.ffill().iloc[-1] / seg.iloc[0] - 1
        row = {"month": t2, "SPY": spy.asof(t2) / spy.asof(t) - 1,
               "EW": mret.mean()}
        pools = {"RESMOM_BIG": rm[[s for s in big if s in rm.index]],
                 "RESMOM": rm, "MOM_BIG": mom[big]}
        for name, sc in pools.items():
            pick = list(sc.sort_values(ascending=False).index[:FP.TOP])
            w = {s: 1 / FP.TOP for s in pick}
            old = prev.get(name, {})
            turn = sum(abs(w.get(s, 0) - old.get(s, 0))
                       for s in set(w) | set(old))
            prev[name] = w
            row[name] = mret[pick].mean() - turn * FP.COST
            row[name + "_turn"] = turn
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def table(res, label):
    print(f"\n  == {label}: {len(res)} months, {res.index[0].date()} -> "
          f"{res.index[-1].date()} ==")
    print(f"  {'portfolio':<12} {'annual':>8} {'vol':>7} {'sharpe':>7} "
          f"{'max DD':>8} {'vs SPY/yr':>10} {'t':>6} {'turnover':>9}")
    out = {}
    for col in VARIANTS + ["SPY", "EW"]:
        a, v, s, dd = FP.stats(res[col])
        ex = res[col] - res["SPY"]
        exs = (f"{ex.mean()*1200:>+9.1f}% {FP.tstat(ex):>+6.2f}"
               if col != "SPY" else f"{'':>10} {'':>6}")
        to = (f"{res[col + '_turn'].mean()*12:>8.1f}x"
              if col + "_turn" in res else "")
        print(f"  {col:<12} {a*100:>+7.1f}% {v*100:>6.1f}% {s:>7.2f} "
              f"{dd*100:>7.1f}% {exs} {to}")
        out[col] = (a, s)
    return out


def smoke():
    px, vol, spy, members = T.smoke()
    rng = np.random.default_rng(8)
    # planted: big names with positive ALPHA (drift beyond market beta)
    big = px.columns[::2]
    alpha = np.exp(np.cumsum(np.full(len(px), 0.0005)))
    for i, s in enumerate(big):
        if i % 4 == 1:
            px[s] = px[s] * alpha
    return px, vol, spy, members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy, members = smoke()
    else:
        if not os.environ.get("ALPACA_API_KEY"):
            sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
        members, union = FP.membership()
        px, spy = FP.load_prices(union, members)
        vol = T.volumes(px)
    res = run(px, vol, spy, members)
    if not a.smoke:
        res.to_csv(OUT)
    table(res[res.index < RECENT], "CONTEXT 2017-2021")
    rec = table(res[res.index >= RECENT], "DECIDING: LAST FIVE YEARS")

    p, s = rec["RESMOM_BIG"], rec["SPY"]
    print("\n  ===== VERDICT (pre-stated, RESMOM_BIG vs SPY, last 5 years) "
          "=====")
    if a.smoke:
        print("  smoke: planted alpha in big names (pipe check)")
    ok = p[0] > s[0] and p[1] > s[1]
    print(f"  return {p[0]*100:+.1f}% vs SPY {s[0]*100:+.1f}% | Sharpe "
          f"{p[1]:.2f} vs SPY {s[1]:.2f}")
    print("  PASS -- a real stock picker on recent data. Next: earnings-"
          "surprise signal, then paper trading." if ok else
          "  FAIL -- residual momentum does not beat SPY on both return and "
          "Sharpe over the last five years.")


if __name__ == "__main__":
    main()
