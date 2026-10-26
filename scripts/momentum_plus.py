"""
momentum_plus.py -- add-ons stacked on steady momentum (momentum_v2
STEADY: all liquid US stocks >= $10 and >= $20M/day, 12-1 return divided
by volatility, top 20, monthly, 20 bps/trade, no stops).

ADD-ONS (each attacks a weakness the earlier runs showed)
  BASE     steady momentum as is
  REGIME   market switch: hold the picks only while the equal-weight index
           of all liquid US stocks is above its 200-day average; else SPY
  FIP      smooth climbers ("frog in the pan", Da-Gurun-Warachka 2014):
           of the top 60, the 20 whose gain came from many small up days
           (score = share of down days - share of up days, lowest wins)
  CLUSTER  no crowding: walk down the ranking, skip a stock whose past-
           year daily returns correlate > 0.6 with one already picked
  VOLUME   money flowing in: of the top 60, the 20 with the biggest rise
           in dollar volume (last 3 months vs last 12)
  INVVOL   same picks as BASE, weighted by 1 / volatility
  FIP_VOL  smooth climbers AND money flowing in: of the top 60 by steady
           momentum, the 20 with the best combined FIP + volume-rise rank
           (the requested combo -- always a candidate, chosen on 2017-21
           like every other variant)
  HEDGED   BASE minus its market exposure (beta x SPY) -- skill check;
           not a candidate, printed for information
  STACK    every add-on among REGIME/FIP/CLUSTER/VOLUME/INVVOL whose
           2017-2021 Sharpe beat BASE's, combined

PRE-STATED (select on 2017-02 -> 2021-05, judge on the last five years)
  1. candidate = STACK if its 2017-21 Sharpe >= the best single add-on,
     else that best single add-on (BASE included as a choice)
  2. PASS if the candidate, over 2021-06 -> 2026-05, beats SPY leveraged
     to the same volatility (4%/yr borrow) AND has Sharpe > SPY's.

Run (repo root; needs data/us_universe/):
    python3.10 scripts/momentum_plus.py --smoke
    python3.10 scripts/momentum_plus.py | tee logs/momentum_plus.txt
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
import momentum_v2 as M      # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "momentum_plus_returns.csv")
ADDONS = ["REGIME", "FIP", "CLUSTER", "VOLUME", "INVVOL"]
SINGLES = ["BASE"] + ADDONS + ["FIP_VOL"]
POOL, CORR_MAX = 60, 0.6


class Ctx:
    """Everything known at month-end t, computed once per month."""

    def __init__(self, px, dv, r, t_i, elig, spy_r):
        W = px[elig].iloc[t_i - 252:t_i + 1]
        R = r[elig].iloc[t_i - 251:t_i + 1]
        self.R = R
        self.dvol = R.std() * np.sqrt(252)
        mom = W.iloc[-22] / W.iloc[0] - 1
        self.steady = (mom / self.dvol.replace(0, np.nan)) \
            .replace([np.inf, -np.inf], np.nan).dropna() \
            .sort_values(ascending=False)
        self.pool = list(self.steady.index[:POOL])
        P = R[self.pool]
        self.fip = np.sign(mom[self.pool]) * ((P < 0).mean() - (P > 0).mean())
        d = dv[self.pool].iloc[t_i - 251:t_i + 1]
        self.flow = d.iloc[-63:].mean() / d.mean().replace(0, np.nan)
        x = spy_r.iloc[t_i - 251:t_i + 1].values
        xm = x - np.nanmean(x)
        Y = P.fillna(0).values
        self.beta = pd.Series((xm[:, None] * (Y - Y.mean(0))).sum(0)
                              / (xm ** 2).sum(), self.pool)


def choose(c, use):
    """Pick list + weights for a set of add-ons."""
    if "FIP" in use and "VOLUME" in use:
        sc = c.fip.rank(ascending=True) + c.flow.rank(ascending=False)
        order = list(sc.sort_values().index)
    elif "FIP" in use:
        order = list(c.fip.sort_values().index)
    elif "VOLUME" in use:
        order = list(c.flow.sort_values(ascending=False).index)
    else:
        order = list(c.steady.index)
    if "CLUSTER" in use:
        C = c.R[[s for s in order[:200] if s in c.R]].corr(min_periods=200)
        pick = []
        for s in order:
            if s not in C:
                continue
            if all(not (C.at[s, p] > CORR_MAX) for p in pick):
                pick.append(s)
            if len(pick) == M.TOP:
                break
    else:
        pick = order[:M.TOP]
    if "INVVOL" in use:
        w = 1 / c.dvol[pick]
        w = w / w.sum()
    else:
        w = pd.Series(1 / len(pick), pick)
    return pick, w


def run(px, vol, spy, members, stack=None):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    dv = px * vol
    ew = (1 + r.clip(-0.5, 1).mean(axis=1).fillna(0)).cumprod()
    ew_ma = ew.rolling(200, min_periods=150).mean()
    spy_r = spy.reindex(idx).ffill().pct_change().fillna(0)
    configs = {k: {k} for k in ADDONS}
    configs["BASE"] = set()
    configs["FIP_VOL"] = {"FIP", "VOLUME"}
    if stack is not None:
        configs = {"STACK": set(stack)}
    rows = []
    prev = {k: set() for k in list(configs) + ["HEDGED"]}
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
        c = Ctx(px, dv, r, t_i, elig, spy_r)
        seg = px.loc[t:t2, elig]
        mret = seg.ffill().iloc[-1] / seg.iloc[0] - 1
        spy_m = spy.asof(t2) / spy.asof(t) - 1
        risk_on = ew.iloc[t_i] > ew_ma.iloc[t_i]
        row = {"month": t2, "SPY": spy_m, "risk_on": risk_on}
        for name, use in configs.items():
            pick, w = choose(c, use)
            if "REGIME" in use and not risk_on:
                hold, ret = {"SPY"}, spy_m
            else:
                hold, ret = set(pick), float((mret[pick] * w).sum())
            old = prev[name]
            trades = len(hold - old) + len(old - hold)
            row[name] = ret - trades / M.TOP * M.COST
            prev[name] = hold
            if name == "BASE":
                b = float(c.beta[[s for s in pick if s in c.beta]].mean())
                row["HEDGED"] = row["BASE"] - b * spy_m
                row["beta"] = b
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def table(res, cols, label):
    print(f"\n  ===== {label}: {len(res)} months, {res.index[0].date()} -> "
          f"{res.index[-1].date()} =====")
    print(f"  {'variant':<8} {'annual':>7} {'vol':>6} {'sharpe':>6} "
          f"{'worst':>7} {'same-risk SPY':>13}")
    _, spy_v, spy_s, _ = B.stats(res["SPY"])
    out = {}
    for v in cols:
        a, vol, s, dd = B.stats(res[v])
        lev = vol / spy_v
        m_a = B.stats(lev * res["SPY"]
                      - max(lev - 1, 0) * M.LEV_COST / 12)[0]
        print(f"  {v:<8} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
              f"{dd*100:>6.1f}% {m_a*100:>+12.1f}%")
        out[v] = (a, s, m_a)
    a, vol, s, dd = B.stats(res["SPY"])
    print(f"  {'SPY':<8} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
          f"{dd*100:>6.1f}%")
    return out, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy = M.smoke()
        feed = "sip"
    else:
        px, vol, spy, feed = U.load()
    U.MIN_PX = M.MIN_PX
    min_dv = M.MIN_DV if feed == "sip" else M.MIN_DV * 0.03
    sets, ever = U.liquid_members(px, vol, min_dv)
    px, vol = px[ever], vol[ever]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731

    res = run(px, vol, spy, members)
    old = res.index < M.RECENT
    sel, _ = table(res[old], SINGLES + ["HEDGED"],
                   "STEP 1 -- SELECT ON 2017-2021")
    base_s = sel["BASE"][1]
    stack = [k for k in ADDONS if sel[k][1] > base_s]
    print(f"\n  add-ons that beat BASE's Sharpe in 2017-2021: "
          f"{', '.join(stack) or 'none'}")
    best = max(SINGLES, key=lambda k: sel[k][1])
    cand = best
    if len(stack) >= 2:
        rs = run(px, vol, spy, members, stack=stack)
        res = res.join(rs[["STACK"]])
        s_stack = B.stats(res.loc[old, "STACK"])[2]
        print(f"  STACK ({'+'.join(stack)}) 2017-2021 Sharpe {s_stack:.2f} "
              f"vs best single {best} {sel[best][1]:.2f}")
        if s_stack >= sel[best][1]:
            cand = "STACK"
    print(f"  CANDIDATE (fixed before looking at the last 5 years): {cand}")
    if not a.smoke:
        res.to_csv(OUT)

    cols = SINGLES + ["HEDGED"] + (["STACK"] if "STACK" in res else [])
    rec, spy_s = table(res[~old], cols, "STEP 2 -- JUDGE ON LAST FIVE YEARS")
    yr = res.loc[~old, list(dict.fromkeys([cand, "BASE", "SPY"]))]
    print("  by year: " + " | ".join(
        f"{y}: " + " ".join(f"{c} {((1 + g[c]).prod() - 1)*100:+.0f}%"
                            for c in yr.columns)
        for y, g in yr.groupby(yr.index.year)))
    h = res.loc[~old, "HEDGED"]
    print(f"  skill check: HEDGED (market removed) {h.mean()*1200:+.1f}%/yr "
          f"t {h.mean() / (h.std() / np.sqrt(len(h))):+.2f}, avg beta of "
          f"picks {res.loc[~old, 'beta'].mean():.2f}")
    a_, s_, m_a = rec[cand]
    ok = a_ > m_a and s_ > spy_s
    print(f"\n  ===== VERDICT (pre-stated: chosen on 2017-2021, judged on "
          f"last 5 years) =====")
    print(f"  {cand}: return {a_*100:+.1f}% vs same-risk SPY {m_a*100:+.1f}%"
          f" | Sharpe {s_:.2f} vs SPY {spy_s:.2f} -> "
          f"{'PASS -- forward paper trading next' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
