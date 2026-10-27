"""
momentum_avoid.py -- does the confirmed attention fade make the all-US
steady momentum system better if we simply NEVER BUY recently hyped stocks?

Long-only, so no borrow fees -- the reason the short book failed.

Base system = momentum_v2.py exactly (all liquid US stocks, price >= $10,
>= $20M/day, rank by 12-1 return / volatility, top 20 equal weight,
monthly rebalance, 20 bps per trade; STEADY_TS adds the 20% trailing stop).

AVOID rule: at each rebalance, drop any stock that had a GOOD-NEWS
attention spike (daily Wikipedia views >= 3x prior-90-day median AND the
stock up > 2% that day) in the prior 504 trading days (2 years -- the fade
lasted through year two). Spikes known by the rebalance day only (spike
day strictly before the rebalance close). Then take the next-best 20.
Stocks without a Wikipedia page are never dropped.

PRE-STATED: STEADY_TS_AVOID PASSES if it beats STEADY_TS on BOTH annual
return AND Sharpe in BOTH 2017-02 -> 2021-05 and 2021-06 -> 2026-05.
STEADY vs STEADY_AVOID shown as a second look (not judged).

Run (repo root):
    python3.10 scripts/momentum_avoid.py --smoke
    python3.10 scripts/momentum_avoid.py | tee logs/momentum_avoid.txt
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
import attention_rebound as AR        # noqa: E402
import attention_short_test as AS     # noqa: E402
import blowup_scan as B               # noqa: E402
import blowup_us as U                 # noqa: E402
import momentum_v2 as MV              # noqa: E402

VARIANTS = ["STEADY", "STEADY_AVOID", "STEADY_TS", "STEADY_TS_AVOID"]
LOOKBACK = 504


def spike_table(views, px):
    AR.START = pd.Timestamp("2015-10-01")
    cols = set(px.columns)
    ev = AR.good_spikes(views, px, lambda d: cols)
    print(f"  good-news spikes found (all sizes, 2015-10+): {len(ev)} in "
          f"{ev['sym'].nunique()} stocks")
    return {s: np.sort(g["d"].values) for s, g in ev.groupby("sym")}


def hyped(spk, syms, t, t_back):
    out = set()
    for s in syms:
        a = spk.get(s)
        if a is None:
            continue
        j = np.searchsorted(a, np.datetime64(t))      # spikes strictly < t
        if j and a[j - 1] > np.datetime64(t_back):
            out.add(s)
    return out


def run(px, spy, members, spk):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= MV.FIRST]
    r = px.pct_change(fill_method=None)
    rows = []
    prev = {v: set() for v in VARIANTS}
    peaks = {v: {} for v in VARIANTS}
    held_ts = {v: set() for v in VARIANTS}
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
        mom = W.iloc[-22] / W.iloc[0] - 1
        vol = r[elig].iloc[t_i - 251:t_i + 1].std() * np.sqrt(252)
        steady = (mom / vol.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        bad = hyped(spk, steady.index[:200], t, idx[max(t_i - LOOKBACK, 0)])
        row = {"month": t2, "SPY": spy.asof(t2) / spy.asof(t) - 1}
        base_pick = list(steady.index[:MV.TOP])
        row["dropped"] = len([s for s in base_pick if s in bad])
        for v in VARIANTS:
            cand = [s for s in steady.index if not ("AVOID" in v and s in bad)]
            pick = cand[:MV.TOP]
            stop = MV.STOP if "_TS" in v else 0
            pk = {s: peaks[v][s] for s in pick
                  if s in peaks[v] and s in held_ts[v]} if stop else {}
            rets, stopped, newpk = MV.month_path(px, pick, t, t2, pk, stop)
            old = prev[v]
            trades = len(set(pick) - old) + len(old - set(pick)) + len(stopped)
            row[v] = np.mean([rets[s] for s in pick]) - trades / MV.TOP * \
                MV.COST
            held = set(pick) - stopped
            prev[v] = held
            if stop:
                peaks[v], held_ts[v] = newpk, held
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def report(res, label):
    print(f"\n  ===== {label}: {len(res)} months | momentum top-20 picks "
          f"dropped as hyped: {res['dropped'].mean():.1f}/month =====")
    print(f"  {'variant':<16} {'annual':>7} {'vol':>6} {'sharpe':>6} "
          f"{'worst':>7}")
    out = {}
    for v in VARIANTS + ["SPY"]:
        a, vol, s, dd = B.stats(res[v])
        out[v] = (a, s)
        print(f"  {v:<16} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
              f"{dd*100:>6.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy = MV.smoke()
        feed = "sip"
        views = pd.DataFrame(np.random.default_rng(5).lognormal(
            5, 0.6, (len(px.index), 30)), px.index, px.columns[:30])
    else:
        px, vol, spy, feed = U.load()
        views = AS.norm_index(pd.read_parquet(AS.ATT))
    U.MIN_PX = MV.MIN_PX
    sets, ever = U.liquid_members(px, vol, MV.MIN_DV if feed == "sip"
                                  else MV.MIN_DV * 0.03)
    spk = spike_table(views, px)
    px = px[ever]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)

    def members(t):
        past = [d for d in days if d <= t]
        return sets[past[-1]] if past else set()
    res = run(px, spy, members, spk)
    old = res.index < MV.RECENT
    r0 = report(res[old], "2017-2021")
    r1 = report(res[~old], "LAST FIVE YEARS")
    yr = (1 + res[VARIANTS + ["SPY"]]).groupby(res.index.year).prod() - 1
    print("\n  by year:   " + "".join(f"{v:>17}" for v in yr.columns))
    for y, row in yr.iterrows():
        print(f"  {y:<9} " + "".join(f"{x*100:>+16.1f}%" for x in row))
    ok = all(r[ "STEADY_TS_AVOID"][0] > r["STEADY_TS"][0] and
             r["STEADY_TS_AVOID"][1] > r["STEADY_TS"][1] for r in (r0, r1))
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    print(f"  STEADY_TS_AVOID beats STEADY_TS (return + Sharpe, both "
          f"periods)  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
