"""
momentum_more.py -- a second batch of add-ons on steady momentum (all
liquid US stocks >= $10 and >= $20M/day, 12-1 return / volatility, top 20,
monthly, 20 bps/trade). Needs data/us_universe/ and
data/earnings_quarterly.csv.gz.

ADD-ONS
  BASE        steady momentum as is
  PROFIT      skip stocks whose last 4 reported quarters of net income sum
              to a loss (known as of the latest SEC filing before t)
  NOBADNEWS   skip stocks whose latest earnings (filed within 120 days)
              were a negative surprise
  LOWBETA     of the top 60, the 20 that move least with SPY (past year)
  LEADER_DIP  of the top 60, the 20 that fell most over the last month
              (buy the pullback, but only in leaders)
  CONSIST     rank by the WEAKEST of its 3-month, 6-month and 12-1 month
              steady momentum ranks (all horizons must agree)
  NEARHIGH    of the top 60, the 20 closest to their 52-week high
  STACK       every add-on whose 2017-2021 Sharpe beat BASE's, combined:
              filters (PROFIT, NOBADNEWS) first, then the top-60 re-rank
              by the sum of the chosen add-ons' ranks

PRE-STATED (select on 2017-02 -> 2021-05, judge on the last five years)
  1. candidate = STACK if its 2017-21 Sharpe >= the best single variant,
     else the best single variant (BASE included)
  2. PASS if the candidate, 2021-06 -> 2026-05, beats SPY leveraged to
     the same volatility (4%/yr borrow) AND has Sharpe > SPY's.

Run (repo root):
    python3.10 scripts/momentum_more.py --smoke
    python3.10 scripts/momentum_more.py | tee logs/momentum_more.txt
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
import blowup_scan as B      # noqa: E402
import blowup_us as U        # noqa: E402
import earnings_test as E    # noqa: E402
import momentum_v2 as M      # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "momentum_more_returns.csv")
FILTERS = ["PROFIT", "NOBADNEWS"]
RERANKS = ["LOWBETA", "LEADER_DIP", "CONSIST", "NEARHIGH"]
ADDONS = FILTERS + RERANKS
SINGLES = ["BASE"] + ADDONS
POOL = 60


def fund_events(e):
    """Per company-quarter: surprise + trailing-4-quarter net income."""
    s = E.surprises(e)
    e = e.sort_values(["cik", "end"]).drop_duplicates(["cik", "end"])
    e["ttm"] = e.groupby("cik")["ni"].transform(
        lambda x: x.rolling(4, min_periods=4).sum())
    s = s.merge(e[["cik", "end", "ttm"]], on=["cik", "end"], how="left")
    return s


def latest(ev, t, days):
    g = ev[(ev["filed"] <= t) & (ev["filed"] > t - pd.Timedelta(days=days))]
    return g.groupby("col")[["sue", "ttm"]].last()


def steady_at(px, r, t_i, elig, look):
    W = px[elig].iloc[t_i - look:t_i + 1]
    vol = r[elig].iloc[t_i - 251:t_i + 1].std() * np.sqrt(252)
    mom = W.iloc[-22 if look > 63 else -1] / W.iloc[0] - 1
    return (mom / vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def choose(use, steady, info, beta, r21, nhigh, consist):
    order = list(steady.dropna().sort_values(ascending=False).index)
    if "CONSIST" in use and len(use) == 1:
        order = list(consist.dropna().sort_values(ascending=False).index)
    if "PROFIT" in use:
        bad = set(info.index[info["ttm"] <= 0])
        order = [s for s in order if s not in bad]
    if "NOBADNEWS" in use:
        bad = set(info.index[info["sue"] < 0])
        order = [s for s in order if s not in bad]
    pool = order[:POOL]
    ranks = []
    if "LOWBETA" in use:
        ranks.append(beta.reindex(pool).rank())
    if "LEADER_DIP" in use:
        ranks.append(r21.reindex(pool).rank())
    if "NEARHIGH" in use:
        ranks.append(nhigh.reindex(pool).rank(ascending=False))
    if "CONSIST" in use and len(use) > 1:
        ranks.append(consist.reindex(pool).rank(ascending=False))
    if ranks:
        sc = pd.concat(ranks, axis=1).mean(axis=1)
        return list(sc.sort_values().index[:M.TOP])
    return pool[:M.TOP]


def run(px, spy, members, ev, configs):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    spy_r = spy.reindex(idx).ffill().pct_change().fillna(0)
    rows = []
    prev = {k: set() for k in configs}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        if t_i < 260:
            continue
        mem = members(t)
        cnt = px.iloc[t_i - 252:t_i + 1].notna().sum()
        elig = [s for s in mem if cnt.get(s, 0) >= 240]
        if len(elig) < 100:
            continue
        steady = steady_at(px, r, t_i, elig, 252)
        s6 = steady_at(px, r, t_i, elig, 126)
        s3 = steady_at(px, r, t_i, elig, 63)
        consist = pd.concat([steady.rank(pct=True), s6.rank(pct=True),
                             s3.rank(pct=True)], axis=1).min(axis=1)
        top = list(steady.dropna().sort_values(ascending=False).index[:300])
        R = r[top].iloc[t_i - 251:t_i + 1].fillna(0)
        x = spy_r.iloc[t_i - 251:t_i + 1].values
        xm = x - x.mean()
        beta = pd.Series((xm[:, None] * (R.values - R.values.mean(0)))
                         .sum(0) / (xm ** 2).sum(), top)
        P = px[elig]
        r21 = P.iloc[t_i] / P.iloc[t_i - 21] - 1
        nhigh = P.iloc[t_i] / P.iloc[t_i - 251:t_i + 1].max()
        info = latest(ev, t, 120) if ev is not None else \
            pd.DataFrame(columns=["sue", "ttm"])
        info_p = latest(ev, t, 200) if ev is not None else info
        seg = px.loc[t:t2, elig]
        mret = seg.ffill().iloc[-1] / seg.iloc[0] - 1
        row = {"month": t2, "SPY": spy.asof(t2) / spy.asof(t) - 1}
        for name, use in configs.items():
            inf = info.copy()
            if "PROFIT" in use:
                inf = info_p[["ttm"]].join(info[["sue"]], how="outer")
            pick = choose(use, steady, inf, beta, r21, nhigh, consist)
            old = prev[name]
            trades = len(set(pick) - old) + len(old - set(pick))
            row[name] = mret[pick].mean() - trades / M.TOP * M.COST
            row[name + "_beta"] = beta.reindex(pick).mean()
            prev[name] = set(pick)
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def table(res, cols, label):
    print(f"\n  ===== {label}: {len(res)} months, {res.index[0].date()} -> "
          f"{res.index[-1].date()} =====")
    print(f"  {'variant':<11} {'annual':>7} {'vol':>6} {'sharpe':>6} "
          f"{'worst':>7} {'same-risk SPY':>13} {'beta':>5}")
    _, spy_v, spy_s, _ = B.stats(res["SPY"])
    out = {}
    for v in cols:
        a, vol, s, dd = B.stats(res[v])
        lev = vol / spy_v
        m_a = B.stats(lev * res["SPY"]
                      - max(lev - 1, 0) * M.LEV_COST / 12)[0]
        print(f"  {v:<11} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
              f"{dd*100:>6.1f}% {m_a*100:>+12.1f}% "
              f"{res[v + '_beta'].mean():>5.2f}")
        out[v] = (a, s, m_a)
    a, vol, s, dd = B.stats(res["SPY"])
    print(f"  {'SPY':<11} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
          f"{dd*100:>6.1f}%")
    return out, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy = M.smoke()
        feed = "sip"
        rng = np.random.default_rng(9)
        ev = pd.DataFrame([(c, px.index[i], rng.normal(), rng.normal(1, 1))
                           for c in px.columns
                           for i in range(260, len(px), 63)],
                          columns=["col", "filed", "sue", "ttm"])
    else:
        px, vol, spy, feed = U.load()
        e = pd.read_csv(E.EARN, dtype={"cik": str, "ticker": str},
                        parse_dates=["end", "filed"]).dropna(
                            subset=["ticker"])
        ev = E.map_to_columns(fund_events(e), px)
        print(f"  {len(ev)} earnings records matched to price histories")
    U.MIN_PX = M.MIN_PX
    min_dv = M.MIN_DV if feed == "sip" else M.MIN_DV * 0.03
    sets, ever = U.liquid_members(px, vol, min_dv)
    px = px[ever]
    ev = ev[ev["col"].isin(ever)]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731

    configs = {k: {k} for k in ADDONS}
    configs["BASE"] = set()
    res = run(px, spy, members, ev, configs)
    old = res.index < M.RECENT
    sel, _ = table(res[old], SINGLES, "STEP 1 -- SELECT ON 2017-2021")
    stack = [k for k in ADDONS if sel[k][1] > sel["BASE"][1]]
    print(f"\n  add-ons that beat BASE's Sharpe in 2017-2021: "
          f"{', '.join(stack) or 'none'}")
    best = max(SINGLES, key=lambda k: sel[k][1])
    cand = best
    if len(stack) >= 2:
        rs = run(px, spy, members, ev, {"STACK": set(stack)})
        res = res.join(rs[["STACK", "STACK_beta"]])
        s_stack = B.stats(res.loc[old, "STACK"])[2]
        print(f"  STACK ({'+'.join(stack)}) 2017-2021 Sharpe {s_stack:.2f} "
              f"vs best single {best} {sel[best][1]:.2f}")
        if s_stack >= sel[best][1]:
            cand = "STACK"
    print(f"  CANDIDATE (fixed before looking at the last 5 years): {cand}")
    if not a.smoke:
        res.to_csv(OUT)
    cols = SINGLES + (["STACK"] if "STACK" in res else [])
    rec, spy_s = table(res[~old], cols, "STEP 2 -- JUDGE ON LAST FIVE YEARS")
    yr = res.loc[~old, list(dict.fromkeys([cand, "BASE", "SPY"]))]
    print("  by year: " + " | ".join(
        f"{y}: " + " ".join(f"{c} {((1 + g[c]).prod() - 1)*100:+.0f}%"
                            for c in yr.columns)
        for y, g in yr.groupby(yr.index.year)))
    a_, s_, m_a = rec[cand]
    ok = a_ > m_a and s_ > spy_s
    print(f"\n  ===== VERDICT (pre-stated: chosen on 2017-2021, judged on "
          f"last 5 years) =====")
    print(f"  {cand}: return {a_*100:+.1f}% vs same-risk SPY {m_a*100:+.1f}%"
          f" | Sharpe {s_:.2f} vs SPY {spy_s:.2f} -> "
          f"{'PASS -- forward paper trading next' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
