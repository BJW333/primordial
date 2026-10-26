"""
exit_rules.py -- fix the sell rule. Same machine as momentum_v2.py (all
liquid US stocks >= $10 and >= $20M/day, steady-gain momentum rank, top 20,
monthly rebalance, 20 bps/trade); only the EXIT changes.

The 20% trailing stop sold ~5 of 20 holdings a month and cost ~7%/yr:
these stocks swing ~33%/yr, so a 20% dip is routine noise.

EXIT RULES (checked on every daily close)
  NONE      hold the month (= STEADY in momentum_v2)
  TS20      sell 20% below peak since purchase (the old rule)
  TS35      sell 35% below peak
  VOLSTOP   sell 3x the stock's own typical monthly move below peak
            (3 * daily vol of the past year * sqrt(21), clipped 10%-50%):
            calm stocks get tight stops, wild ones loose stops
  MA50      sell when it closes below its 50-day average price
  BUFFER    no price stop; at each rebalance KEEP a holding while it is
            still in the top 60, only replace ones that fell out
In every rule, a sold stock that is still top-ranked at the next
rebalance is bought back (rule 5, re-entry, built in).

SELECTION THEN CONFIRMATION (pre-stated, so the winner can't be luck)
  1. pick the rule with the best Sharpe on 2017-02 -> 2021-05 ONLY
  2. that one rule must, on the last five years (2021-06 -> 2026-05),
     beat SPY leveraged to the same volatility AND have Sharpe > SPY's.
  The other rules' last-5-year numbers are printed but do not count.

Run (repo root; needs data/us_universe/):
    python3.10 scripts/exit_rules.py --smoke
    python3.10 scripts/exit_rules.py | tee logs/exit_rules.txt
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
OUT = os.path.join(ROOT, "data", "exit_rules_returns.csv")
RULES = ["NONE", "TS20", "TS35", "VOLSTOP", "MA50", "BUFFER"]
KEEP_RANK = 60


def ride(px, ma, pick, t, t2, peaks, rule, dist):
    """Daily ride t -> t2. Returns returns, sold set, peaks to carry."""
    seg = px.loc[t:t2, pick].ffill()
    mseg = ma.loc[t:t2, pick] if rule == "MA50" else None
    base = seg.iloc[0]
    rets, sold, newpk = {}, set(), {}
    for s in pick:
        p = seg[s].values
        if not np.isfinite(base[s]):
            rets[s] = 0.0
            continue
        pk = max(peaks.get(s, base[s]), base[s])
        out = None
        if rule in ("TS20", "TS35", "VOLSTOP"):
            d = dist[s]
            for v in p[1:]:
                if not np.isfinite(v):
                    continue
                pk = max(pk, v)
                if v <= pk * (1 - d):
                    out = v
                    break
        elif rule == "MA50":
            m = mseg[s].values
            for v, mv in zip(p[1:], m[1:]):
                if np.isfinite(v) and np.isfinite(mv) and v < mv:
                    out = v
                    break
        if out is not None:
            rets[s] = out / base[s] - 1
            sold.add(s)
        else:
            fin = p[np.isfinite(p)]
            rets[s] = (fin[-1] if len(fin) else base[s]) / base[s] - 1
            newpk[s] = max(pk, np.nanmax(p)) if len(fin) else pk
    return rets, sold, newpk


def run(px, spy, members):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    ma = px.rolling(50, min_periods=40).mean()
    rows = []
    held = {k: set() for k in RULES}
    peaks = {k: {} for k in RULES}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        if t_i < 260:
            continue
        mem = members(t)
        win = px.iloc[t_i - 252:t_i + 1]
        cnt = win.notna().sum()
        elig = [s for s in mem if cnt.get(s, 0) >= 240]
        if len(elig) < 100:
            continue
        W = win[elig]
        dvol = r[elig].iloc[t_i - 251:t_i + 1].std()
        steady = (W.iloc[-22] / W.iloc[0] - 1) / (dvol * np.sqrt(252))
        rank = steady.replace([np.inf, -np.inf], np.nan).dropna() \
            .sort_values(ascending=False)
        top = list(rank.index[:M.TOP])
        keep_set = set(rank.index[:KEEP_RANK])
        row = {"month": t2, "SPY": spy.asof(t2) / spy.asof(t) - 1,
               "n_elig": len(elig)}
        for rule in RULES:
            if rule == "BUFFER":
                kept = [s for s in rank.index if s in held[rule]
                        and s in keep_set]
                pick = kept + [s for s in top if s not in kept]
                pick = pick[:M.TOP]
            else:
                pick = top
            dist = {}
            if rule in ("TS20", "TS35"):
                dist = {s: 0.20 if rule == "TS20" else 0.35 for s in pick}
            elif rule == "VOLSTOP":
                dist = {s: float(np.clip(3 * dvol.get(s, 0.02) * np.sqrt(21),
                                         0.10, 0.50)) for s in pick}
            carry = {s: v for s, v in peaks[rule].items()
                     if s in pick and s in held[rule]}
            rets, sold, newpk = ride(px, ma, pick, t, t2, carry, rule, dist)
            old = held[rule]
            trades = len(set(pick) - old) + len(old - set(pick)) + len(sold)
            row[rule] = np.mean([rets[s] for s in pick]) - \
                trades / M.TOP * M.COST
            row[rule + "_sold"] = len(sold)
            row[rule + "_trades"] = trades
            held[rule] = set(pick) - sold
            peaks[rule] = newpk
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def table(res, label):
    print(f"\n  ===== {label}: {len(res)} months, {res.index[0].date()} -> "
          f"{res.index[-1].date()} =====")
    print(f"  {'rule':<8} {'annual':>7} {'vol':>6} {'sharpe':>6} "
          f"{'worst':>7} {'same-risk SPY':>13} {'sold/mo':>7} "
          f"{'trades/mo':>9}")
    _, spy_v, spy_s, _ = B.stats(res["SPY"])
    out = {}
    for rule in RULES:
        a, v, s, dd = B.stats(res[rule])
        lev = v / spy_v
        m_a = B.stats(lev * res["SPY"]
                      - max(lev - 1, 0) * M.LEV_COST / 12)[0]
        print(f"  {rule:<8} {a*100:>+6.1f}% {v*100:>5.1f}% {s:>6.2f} "
              f"{dd*100:>6.1f}% {m_a*100:>+12.1f}% "
              f"{res[rule + '_sold'].mean():>7.1f} "
              f"{res[rule + '_trades'].mean():>9.1f}")
        out[rule] = (a, s, m_a)
    a, v, s, dd = B.stats(res["SPY"])
    print(f"  {'SPY':<8} {a*100:>+6.1f}% {v*100:>5.1f}% {s:>6.2f} "
          f"{dd*100:>6.1f}%")
    return out, spy_s


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
    px = px[ever]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731
    res = run(px, spy, members)
    if not a.smoke:
        res.to_csv(OUT)
    old = res.index < M.RECENT
    sel, _ = table(res[old], "STEP 1 -- SELECT ON 2017-2021")
    best = max(RULES, key=lambda k: sel[k][1])
    print(f"\n  selected on 2017-2021 (best Sharpe): {best}")
    rec, spy_s = table(res[~old], "STEP 2 -- CONFIRM ON LAST FIVE YEARS")
    yr = res.loc[~old, list(dict.fromkeys([best, "NONE", "SPY"]))]
    print("  by year: " + " | ".join(
        f"{y}: {best} {((1 + g[best]).prod() - 1)*100:+.0f}% NONE "
        f"{((1 + g['NONE']).prod() - 1)*100:+.0f}% SPY "
        f"{((1 + g['SPY']).prod() - 1)*100:+.0f}%"
        for y, g in yr.groupby(yr.index.year)))
    a_, s_, m_a = rec[best]
    ok = a_ > m_a and s_ > spy_s
    print(f"\n  ===== VERDICT (pre-stated: rule chosen on 2017-2021, "
          f"judged on last 5 years) =====")
    print(f"  {best}: return {a_*100:+.1f}% vs same-risk SPY {m_a*100:+.1f}%"
          f" | Sharpe {s_:.2f} vs SPY {spy_s:.2f} -> "
          f"{'PASS -- forward paper trading next' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
