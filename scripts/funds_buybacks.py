"""
funds_buybacks.py -- two more non-momentum stock pickers, each tested on
its own. Needs data/us_universe/, data/13f_best_ideas.csv.gz
(fetch_13f.py) and data/shares_outstanding.csv.gz (fetch_shares.py).

BESTIDEA  hedge-fund best new ideas
  For each manager (20-500 positions): its largest NEW position each
  quarter, known from the 13F filing date (up to 45 days late -- kept).
  Each month-end: stocks that were a best new idea of any manager filed
  in the last 120 days, ranked by how many managers picked it, then by
  average weight; top 20, 1/20 each, unused slots in SPY.
  Also printed: raw drift 3 and 6 months after each idea became public.

BUYBACK   companies shrinking their share count
  Shares outstanding from the cover of each 10-Q/10-K (filing date).
  Change = latest count vs the count reported ~1 year earlier. Stocks
  that shrank >= 1% (and < 25%, which would be a reverse split, not a
  buyback); top 20 biggest shrinkers; unused slots in SPY.

Universe for both: >= $10 and >= $20M/day (point-in-time), monthly,
20 bps per trade.

PRE-STATED (two ideas at once -> BOTH periods must pass, as before)
  PASS if in 2017-2021 AND in the last five years it beats SPY leveraged
  to the same volatility (4%/yr borrow) AND has Sharpe above SPY's.

Run (repo root):
    python3.10 scripts/funds_buybacks.py --smoke
    python3.10 scripts/funds_buybacks.py | tee logs/funds_buybacks.txt
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
import blowup_us as U        # noqa: E402
import earnings_test as E    # noqa: E402
import insider_season as S   # noqa: E402
import momentum_v2 as M      # noqa: E402

ROOT = os.path.dirname(HERE)
IDEAS = os.path.join(ROOT, "data", "13f_best_ideas.csv.gz")
SHARES = os.path.join(ROOT, "data", "shares_outstanding.csv.gz")
OUT = os.path.join(ROOT, "data", "funds_buybacks_returns.csv")
SLOTS, COST = 20, 0.0020


def norm(tk):
    return tk.astype(str).str.upper().str.strip().str.replace(
        "-", ".", regex=False)


def book(px, spy, t, t2, pick, prev):
    seg = px.loc[t:t2, pick]
    mret = (seg.ffill().iloc[-1] / seg.iloc[0] - 1).fillna(0) \
        if pick else pd.Series(dtype=float)
    spy_m = spy.asof(t2) / spy.asof(t) - 1
    hold = set(pick)
    trades = len(hold - prev) + len(prev - hold)
    ret = (mret.sum() + (SLOTS - len(pick)) * spy_m) / SLOTS \
        - trades / SLOTS * COST
    return ret, spy_m, hold


def months(px):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    for k in range(len(me) - 1):
        yield idx[idx <= me[k]][-1], idx[idx <= me[k + 1]][-1]


def run_bestidea(px, spy, members, ev):
    rows, prev = [], set()
    for t, t2 in months(px):
        live = ev[(ev["filed"] <= t)
                  & (ev["filed"] > t - pd.Timedelta(days=120))
                  & ev["col"].isin(members(t))]
        g = live.groupby("col").agg(n=("mgr", "nunique"),
                                    w=("weight", "mean"))
        pick = list(g.sort_values(["n", "w"], ascending=False)
                    .index[:SLOTS])
        ret, spy_m, prev = book(px, spy, t, t2, pick, prev)
        rows.append({"month": t2, "BESTIDEA": ret, "SPY": spy_m,
                     "n": len(pick), "multi": int((g["n"] >= 2).sum())})
    return pd.DataFrame(rows).set_index("month")


def run_buyback(px, spy, members, sh):
    rows, prev = [], set()
    for t, t2 in months(px):
        mem = members(t)
        now = sh[(sh["filed"] <= t) & (sh["filed"] > t - pd.Timedelta(
            days=200))].groupby("col")["shares"].last()
        then = sh[(sh["filed"] <= t - pd.Timedelta(days=330))
                  & (sh["filed"] > t - pd.Timedelta(days=550))] \
            .groupby("col")["shares"].last()
        chg = (now / then - 1).dropna()
        chg = chg[chg.index.isin(mem) & (chg <= -0.01) & (chg > -0.25)]
        pick = list(chg.sort_values().index[:SLOTS])
        ret, spy_m, prev = book(px, spy, t, t2, pick, prev)
        rows.append({"month": t2, "BUYBACK": ret, "SPY": spy_m,
                     "n": len(pick), "median_shrink": chg.median()})
    return pd.DataFrame(rows).set_index("month")


def smoke():
    px, vol, spy = M.smoke()
    rng = np.random.default_rng(31)
    px = px.copy()
    ideas, shares = [], []
    cols = list(px.columns)
    for j, c in enumerate(cols[::4]):            # planted best ideas
        for i in range(300 + j * 5, len(px) - 130, 200):
            ideas.append((f"m{j}", px.index[i], c, 0.08))
            ramp = np.ones(len(px))
            ramp[i + 1:i + 64] = np.linspace(1, 1.10, 63)
            ramp[i + 64:] = 1.10
            px[c] = px[c] * ramp
    for j, c in enumerate(cols):                 # share counts; 1/6 shrink
        n = 1e8
        shrink = j % 6 == 0
        for i in range(0, len(px), 63):
            n *= 0.98 if shrink else 1.01
            shares.append((c, px.index[i], n))
        if shrink:
            px[c] = px[c] * np.exp(np.arange(len(px)) * 0.0004)
    ev = pd.DataFrame(ideas, columns=["mgr", "filed", "col", "weight"])
    sh = pd.DataFrame(shares, columns=["col", "filed", "shares"])
    return px, vol, spy, ev, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy, ev, sh = smoke()
        feed = "sip"
    else:
        for p in (IDEAS, SHARES):
            if not os.path.exists(p):
                sys.exit(f"missing {p} -- run fetch_13f.py / "
                         "fetch_shares.py first")
        px, vol, spy, feed = U.load()
        ev = pd.read_csv(IDEAS, dtype={"mgr": str, "ticker": str,
                                       "cusip": str},
                         parse_dates=["filed", "period"]).dropna(
                             subset=["ticker"])
        ev["ticker"] = norm(ev["ticker"])
        ev = E.map_to_columns(ev, px)
        sh = pd.read_csv(SHARES, dtype={"cik": str, "ticker": str},
                         parse_dates=["filed"]).dropna(subset=["ticker"])
        sh["ticker"] = norm(sh["ticker"])
        sh = E.map_to_columns(sh, px).sort_values("filed")
    spy = spy.reindex(px.index).ffill()
    members, ever = S.universe(px, vol, feed, 10.0, 20e6)
    px = px[ever]
    ev = ev[ev["col"].isin(ever)].sort_values("filed")
    sh = sh[sh["col"].isin(ever)].sort_values("filed")

    print(f"\n  BESTIDEA: {len(ev)} best new ideas on liquid stocks, "
          f"{ev['mgr'].nunique()} managers")
    print("  raw drift after the 13F became public (minus SPY):")
    d = S.drift(ev, px, spy)
    S.report_drift(d[d["filed"] < M.RECENT], "2017-2021")
    S.report_drift(d[d["filed"] >= M.RECENT], "last 5 years")
    rb = run_bestidea(px, spy, members, ev)
    print(f"  avg stocks held {rb['n'].mean():.1f} of {SLOTS}; avg stocks "
          f"picked by 2+ managers {rb['multi'].mean():.1f}")
    ok_b = S.judge(rb, "BESTIDEA", "BESTIDEA portfolio")

    print(f"\n  BUYBACK: {len(sh)} share counts on liquid stocks")
    rk = run_buyback(px, spy, members, sh)
    print(f"  avg stocks held {rk['n'].mean():.1f} of {SLOTS}; median "
          f"1-year share change among shrinkers "
          f"{rk['median_shrink'].median()*100:+.1f}%")
    ok_k = S.judge(rk, "BUYBACK", "BUYBACK portfolio")

    if not a.smoke:
        rb[["BESTIDEA"]].join(rk[["BUYBACK", "SPY"]], how="outer") \
            .to_csv(OUT)
    print("\n  ===== VERDICT (pre-stated: both periods must beat same-risk "
          "SPY and SPY's Sharpe) =====")
    if a.smoke:
        print("  smoke: planted best ideas + shrinkers (pipe check)")
    print(f"  BESTIDEA: {'PASS' if ok_b else 'FAIL'}")
    print(f"  BUYBACK:  {'PASS' if ok_k else 'FAIL'}")


if __name__ == "__main__":
    main()
