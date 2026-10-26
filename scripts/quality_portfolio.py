"""
quality_portfolio.py -- the survivorship-free S&P 500 top-20 portfolio
(factor_portfolio.py, commit 2d2d826) with a QUALITY signal from SEC
financial statements.

Needs data/fundamentals.csv.gz (fetch_sec_fundamentals.py) and Alpaca
keys exported (prices are cached from the factor_portfolio run).

QUALITY score at each month-end, from the latest 10-K FILED BEFORE that
day (and covering a fiscal year that ended within the last 18 months):
  average percentile rank of
    roa      net income / assets             (higher = better)
    gpa      gross profit / assets           (higher = better)
    lev      liabilities / assets            (lower = better)
    roa_std  3-year std-dev of ROA           (lower = better)
  needs at least 2 of the 4 (banks have no gross profit)

PRE-STATED (primary = QUALITY; QUALITY+MOM and MOM are context)
  same universe, rebalance, top 20, costs (10 bps/side), benchmark
  (equal-weight of all eligible members) and halves as factor_portfolio.
  PASS: monthly excess vs equal-weight > 0 with t > 2 over 2017-2026 AND
  positive in both halves (2017-2021, 2022-2026).
  Pass -> duplication check vs industry_rotation, then forward paper.

Run (repo root):
    python3.10 scripts/quality_portfolio.py
    python3.10 scripts/quality_portfolio.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import factor_portfolio as FP  # noqa: E402

ROOT = os.path.dirname(HERE)
FUND = os.path.join(ROOT, "data", "fundamentals.csv.gz")
OUT = os.path.join(ROOT, "data", "quality_portfolio_returns.csv")
METRICS = {"roa": True, "gpa": True, "lev": False, "roa_std": False}


def norm(t):
    return str(t).upper().replace("-", ".")


def quality_at(fund, t, names):
    """Latest filed-before-t 10-K per ticker -> composite quality rank."""
    f = fund[(fund["filed"] < t)
             & (fund["period"] >= t - pd.DateOffset(months=18))]
    f = f[f["key"].isin(names)].sort_values("filed").groupby("key").tail(1)
    f = f.set_index("key")
    ranks = []
    for m, higher in METRICS.items():
        r = f[m].rank(pct=True, ascending=higher)
        ranks.append(r)
    R = pd.concat(ranks, axis=1)
    ok = R.notna().sum(axis=1) >= 2
    return R[ok].mean(axis=1)


def run(px, spy, members, fund):
    me = [d for d in px.resample("ME").last().index if d >= FP.FIRST]
    idx = px.index
    keymap = {s: norm(s) for s in px.columns}
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
        if len(elig) < 100:
            continue
        q = quality_at(fund, t, {keymap[s] for s in elig})
        q.index = [s for s in q.index]
        inv = {keymap[s]: s for s in elig}
        q = pd.Series({inv[k2]: v for k2, v in q.items() if k2 in inv})
        qe = [s for s in elig if s in q.index]
        if len(qe) < 100:
            continue
        r_mom = mom[qe].rank(pct=True)
        r_q = q[qe].rank(pct=True)
        combos = {"QUALITY": r_q, "QUALITY+MOM": (r_q + r_mom) / 2,
                  "MOM": r_mom}
        seg = px.loc[t:t2, elig]
        mret = seg.ffill().iloc[-1] / seg.iloc[0] - 1
        row = {"month": t2, "EW": mret.mean(),
               "SPY": spy.asof(t2) / spy.asof(t) - 1,
               "coverage": len(qe) / len(elig)}
        for name, sc in combos.items():
            pick = list(sc.sort_values(ascending=False).index[:FP.TOP])
            w = {s: 1 / FP.TOP for s in pick}
            old = prev.get(name, {})
            turn = sum(abs(w.get(s, 0) - old.get(s, 0))
                       for s in set(w) | set(old))
            row[name] = mret[pick].mean() - turn * FP.COST
            row[name + "_turn"] = turn
            prev[name] = w
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def report(res):
    print(f"\n  {len(res)} monthly rebalances, {res.index[0].date()} -> "
          f"{res.index[-1].date()} | quality data for "
          f"{res['coverage'].mean():.0%} of eligible stocks on average\n")
    print(f"  {'portfolio':<14} {'annual':>8} {'vol':>7} {'sharpe':>7} "
          f"{'max DD':>8} {'vs EW/yr':>9} {'t':>6} {'turnover':>9}")
    for col in ("QUALITY", "QUALITY+MOM", "MOM", "EW", "SPY"):
        a, v, s, dd = FP.stats(res[col])
        ex = res[col] - res["EW"]
        exs = (f"{ex.mean()*1200:>+8.1f}% {FP.tstat(ex):>+6.2f}"
               if col != "EW" else f"{'':>9} {'':>6}")
        to = (f"{res[col + '_turn'].mean()*12:>8.1f}x"
              if col + "_turn" in res else "")
        print(f"  {col:<14} {a*100:>+7.1f}% {v*100:>6.1f}% {s:>7.2f} "
              f"{dd*100:>7.1f}% {exs} {to}")
    print("\n  halves (QUALITY excess vs EW, per year):")
    for lab, part in (("2017-2021", res[res.index < FP.SPLIT]),
                      ("2022-2026", res[res.index >= FP.SPLIT])):
        ex = part["QUALITY"] - part["EW"]
        print(f"    {lab}: {ex.mean()*1200:+.1f}%/yr  t {FP.tstat(ex):+.2f}")


def smoke():
    (px, spy), members, _ = FP.smoke_data()
    rng = np.random.default_rng(4)
    # planted: quality = each stock's hidden drift (+ noise), refiled yearly
    drift = np.log(px.iloc[-1] / px.iloc[0]) / len(px)
    rows = []
    for s in px.columns:
        for y in range(2012, 2027):
            g = drift[s] * 2500 + rng.normal(0, 0.05)
            rows.append({"ticker": s, "filed": pd.Timestamp(y, 3, 1),
                         "period": pd.Timestamp(y - 1, 12, 31), "roa": g,
                         "gpa": g + rng.normal(0, 0.05),
                         "lev": -g + rng.normal(0, 0.05),
                         "roa_std": rng.uniform(0, 0.1)})
    return px, spy, members, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, spy, members, fund = smoke()
    else:
        if not os.environ.get("ALPACA_API_KEY"):
            sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
        if not os.path.exists(FUND):
            sys.exit("run fetch_sec_fundamentals.py first")
        fund = pd.read_csv(FUND, parse_dates=["filed", "period"])
        members, union = FP.membership()
        px, spy = FP.load_prices(union, members)
    fund = fund.dropna(subset=["ticker"]).copy()
    fund["key"] = fund["ticker"].map(norm)
    res = run(px, spy, members, fund)
    report(res)
    if not a.smoke:
        res.to_csv(OUT)
        print(f"\n  monthly returns -> {OUT}")
    ex = res["QUALITY"] - res["EW"]
    h1, h2 = ex[ex.index < FP.SPLIT].mean(), ex[ex.index >= FP.SPLIT].mean()
    t = FP.tstat(ex)
    print("\n  ===== VERDICT (pre-stated, primary QUALITY vs equal-weight) "
          "=====")
    if a.smoke:
        print("  smoke: planted quality signal (pipe check)")
    if ex.mean() > 0 and t > 2 and h1 > 0 and h2 > 0:
        print(f"  PASS -- {ex.mean()*1200:+.1f}%/yr over the equal-weight S&P "
              f"after costs (t {t:+.2f}), both halves positive. Next: dup "
              f"check vs industry_rotation, forward paper.")
    else:
        print(f"  FAIL -- {ex.mean()*1200:+.1f}%/yr vs equal-weight, "
              f"t {t:+.2f}, halves {h1*1200:+.1f}% / {h2*1200:+.1f}%.")


if __name__ == "__main__":
    main()
