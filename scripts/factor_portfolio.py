"""
factor_portfolio.py -- monthly stock-picking portfolio on the S&P 500,
survivorship-free (membership checked by date + Alpaca prices that keep
delisted stocks), 2017-2026.

Each month-end: score every stock that was an S&P 500 member that day,
hold the top 20 equal-weighted until the next month-end, then rebalance.
A stock that stops trading mid-month (bought out, bankrupt) exits at its
last price and the slot sits in cash until the next rebalance.

SCORES (price-only; quality/value need SEC financial data -- next step)
  MOM        12-month return skipping the most recent month (t-252 -> t-21)
  LOWVOL     minus the standard deviation of daily returns, last 126 days
  MOM+LOWVOL average of the two cross-sectional percentile ranks

PRE-STATED (primary = MOM+LOWVOL; MOM and LOWVOL alone are context)
  costs      10 bps per side on traded weight
  benchmark  equal-weight average of ALL eligible members (the fair
             comparison -- same stocks, no picking) and SPY
  PASS       monthly excess return vs the equal-weight benchmark > 0 with
             t > 2 over 2017-2026 AND positive in both halves
             (2017-2021, 2022-2026)
  context    the same portfolio with a market switch (cash when SPY is
             below its 200-day average at the rebalance) -- reported only
  Pass -> SEC-fundamentals combos (quality/value), duplication check vs
  industry_rotation (monthly returns saved to data/factor_portfolio_
  returns.csv), then forward paper.

ALPACA SYMBOL REUSE GUARD: a ticker whose price history has a gap longer
than 40 calendar days is cut at the gap -- only the part of history that
overlaps the stock's S&P membership is used.

Run (repo root; needs ALPACA_API_KEY / ALPACA_SECRET_KEY exported):
    python3.10 scripts/factor_portfolio.py
    python3.10 scripts/factor_portfolio.py --smoke
"""
import argparse
import csv
import os
import sys
from bisect import bisect_right

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial import data as D  # noqa: E402

MEMB = os.path.join(ROOT, "manifests", "sp500_pit_membership.csv")
OUT = os.path.join(ROOT, "data", "factor_portfolio_returns.csv")
A_START, END = "2016-01-01", "2026-06-01"
TOP, COST = 20, 0.0010
FIRST = pd.Timestamp("2017-01-31")
SPLIT = pd.Timestamp("2022-01-01")


def membership():
    rows = [r for r in csv.reader(open(MEMB))][1:]
    dates = [r[0] for r in rows]
    sets_ = [frozenset(t.strip() for t in r[1].split(",")) for r in rows]
    union = sorted(set().union(*sets_))

    def members(day):
        i = bisect_right(dates, str(day.date())) - 1
        return sets_[i] if i >= 0 else frozenset()
    return members, union


def alpaca(sym):
    p = D._cache_path("alpaca", sym, "1d", A_START, END)
    if os.path.exists(p):
        return pd.read_parquet(p)
    try:
        df = D._fetch_one("alpaca", sym, "1d", A_START, END)
    except Exception:  # noqa: BLE001
        return None
    if df is not None and len(df):
        os.makedirs(D.cache_dir(), exist_ok=True)
        df.to_parquet(p)
    return df


def cut_gaps(s):
    """Split at >40-day gaps (recycled tickers); return list of segments."""
    s = s[~s.index.duplicated()].dropna()
    if s.empty:
        return []
    br = np.flatnonzero(np.diff(s.index.values).astype("timedelta64[D]")
                        > np.timedelta64(40, "D"))
    cuts = [0] + list(br + 1) + [len(s)]
    return [s.iloc[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 20]


def load_prices(union, members, smoke=None):
    if smoke is not None:
        return smoke
    closes = {}
    for k, sym in enumerate(union):
        df = alpaca(sym)
        if df is None or df.empty:
            continue
        segs = cut_gaps(df["close"])
        # keep the segment that overlaps this name's membership the most
        best, score = None, -1
        for seg in segs:
            mo = seg.resample("ME").last().index
            n = sum(sym in members(d) for d in mo)
            if n > score:
                best, score = seg, n
        if best is not None and score > 0:
            closes[sym] = best
        if k % 100 == 0:
            print(f"    prices {k}/{len(union)}")
    px = pd.DataFrame(closes).sort_index()
    spy = alpaca("SPY")["close"]
    return px, spy


def scores(px, t_i):
    win = px.iloc[max(0, t_i - 260):t_i + 1]
    c = win.iloc[-1]
    mom = win.iloc[-22] / win.iloc[-253] - 1 if len(win) >= 253 else None
    vol = win.pct_change().iloc[-126:].std()
    return mom, -vol, c


def run(px, spy, members, label_switch=False):
    me = px.resample("ME").last().index
    me = [d for d in me if d >= FIRST]
    idx = px.index
    rows, prev = [], {}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        mom, lowvol, c = scores(px, t_i)
        if mom is None:
            continue
        mem = members(t)
        hist = px.iloc[:t_i + 1].notna().sum()
        elig = [s for s in px.columns
                if s in mem and np.isfinite(c.get(s, np.nan))
                and hist[s] >= 253 and np.isfinite(mom.get(s, np.nan))
                and np.isfinite(lowvol.get(s, np.nan))]
        if len(elig) < 100:
            continue
        r_mom = mom[elig].rank(pct=True)
        r_lv = lowvol[elig].rank(pct=True)
        combos = {"MOM": r_mom, "LOWVOL": r_lv,
                  "MOM+LOWVOL": (r_mom + r_lv) / 2}
        # month return per eligible stock: to t2, or to last price if it
        # stopped trading (cash afterwards)
        seg = px.loc[t:t2, elig]
        last = seg.ffill().iloc[-1]
        mret = last / seg.iloc[0] - 1
        ew = mret.mean()
        spy_t, spy_t2 = spy.asof(t), spy.asof(t2)
        spy_r = spy_t2 / spy_t - 1
        spy_sma = spy.loc[:t].iloc[-200:].mean()
        risk_on = spy_t >= spy_sma
        row = {"month": t2, "EW": ew, "SPY": spy_r, "risk_on": risk_on}
        for name, sc in combos.items():
            pick = list(sc.sort_values(ascending=False).index[:TOP])
            w = {s: 1 / TOP for s in pick}
            old = prev.get(name, {})
            turn = sum(abs(w.get(s, 0) - old.get(s, 0))
                       for s in set(w) | set(old))
            gross = mret[pick].mean()
            row[name] = gross - turn * COST
            row[name + "_switch"] = row[name] if risk_on else 0.0
            row[name + "_turn"] = turn
            prev[name] = w
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def stats(r):
    r = r.dropna()
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return ann, vol, (r.mean() * 12) / vol if vol else np.nan, dd


def tstat(x):
    x = x.dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 \
        else np.nan


def report(res):
    print(f"\n  {len(res)} monthly rebalances, {res.index[0].date()} -> "
          f"{res.index[-1].date()}\n")
    print(f"  {'portfolio':<20} {'annual':>8} {'vol':>7} {'sharpe':>7} "
          f"{'max DD':>8} {'vs EW/yr':>9} {'t':>6} {'turnover':>9}")
    cols = ["MOM", "LOWVOL", "MOM+LOWVOL", "MOM+LOWVOL_switch", "EW", "SPY"]
    for col in cols:
        a, v, s, dd = stats(res[col])
        ex = res[col] - res["EW"]
        exs = f"{ex.mean()*12*100:>+8.1f}% {tstat(ex):>+6.2f}" \
            if col not in ("EW",) else f"{'':>9} {'':>6}"
        tcol = col.replace("_switch", "") + "_turn"
        to = f"{res[tcol].mean()*12:>8.1f}x" if tcol in res else ""
        print(f"  {col:<20} {a*100:>+7.1f}% {v*100:>6.1f}% {s:>7.2f} "
              f"{dd*100:>7.1f}% {exs} {to}")
    print("\n  halves (MOM+LOWVOL excess vs EW, per year):")
    for lab, part in (("2017-2021", res[res.index < SPLIT]),
                      ("2022-2026", res[res.index >= SPLIT])):
        ex = part["MOM+LOWVOL"] - part["EW"]
        print(f"    {lab}: {ex.mean()*12*100:+.1f}%/yr  t {tstat(ex):+.2f}  "
              f"({len(part)} months)")


def smoke_data():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2016-01-01", "2026-05-29")
    n, k = len(idx), 300
    drift = rng.normal(0, 0.0004, k)          # persistent winners
    vol = rng.uniform(0.01, 0.03, k)
    lowvol_premium = 0.0003 * (0.02 - vol) / 0.01
    r = rng.normal(0, 1, (n, k)) * vol + drift + 0.0003 - 0.5 * vol ** 2 \
        + lowvol_premium                      # planted: MOM and LOWVOL
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), idx,
                      [f"S{i}" for i in range(k)])
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
                    idx)
    names = frozenset(px.columns)
    return (px, spy), (lambda d: names), list(px.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        (px, spy), members, union = smoke_data()
    else:
        if not os.environ.get("ALPACA_API_KEY"):
            sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
        members, union = membership()
        print(f"  {len(union)} names ever in the S&P 500 -- loading Alpaca "
              f"prices (cached after the first run)")
        px, spy = load_prices(union, members)
        print(f"  prices for {px.shape[1]} names")
    res = run(px, spy, members)
    report(res)
    if not a.smoke:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        res.to_csv(OUT)
        print(f"\n  monthly returns -> {OUT} (for the industry_rotation "
              f"duplication check)")

    ex = res["MOM+LOWVOL"] - res["EW"]
    h1 = ex[ex.index < SPLIT].mean()
    h2 = ex[ex.index >= SPLIT].mean()
    t = tstat(ex)
    print("\n  ===== VERDICT (pre-stated, primary MOM+LOWVOL vs equal-weight) "
          "=====")
    if a.smoke:
        print("  smoke: planted momentum persistence (pipe check)")
    if ex.mean() > 0 and t > 2 and h1 > 0 and h2 > 0:
        print(f"  PASS -- {ex.mean()*1200:+.1f}%/yr over the equal-weight S&P "
              f"after costs (t {t:+.2f}), both halves positive. Next: quality/"
              f"value from SEC data, dup check vs industry_rotation, paper.")
    else:
        print(f"  FAIL -- {ex.mean()*1200:+.1f}%/yr vs equal-weight, "
              f"t {t:+.2f}, halves {h1*1200:+.1f}% / {h2*1200:+.1f}%.")


if __name__ == "__main__":
    main()
