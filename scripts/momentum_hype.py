"""
momentum_hype.py -- DIAGNOSTIC (post-hoc, no pass/fail). momentum_avoid.py
showed that dropping hyped stocks from steady momentum cut returns ~10-13%/yr
in BOTH periods -- so the hyped momentum picks were the big winners.
This flips it: what if momentum ONLY buys stocks that had a good-news
attention spike in the past 2 years ("trend + a real story")?

Variants (same engine/costs as momentum_v2.py, top 20, monthly):
  STEADY / STEADY_TS            normal steady momentum (with/without stop)
  HYPE / HYPE_TS                top 20 steady momentum among hyped stocks only
  QUIET / QUIET_TS              top 20 among stocks with a Wikipedia page but
                                NO spike in 2y (fair control: same "has a
                                page" group, so it isn't just a size effect)
Because the idea came from looking at momentum_avoid.py results, these
numbers can't be a clean pass -- a good result here means: paper-trade it
forward from today with the rules frozen.

Run (repo root):
    python3.10 scripts/momentum_hype.py | tee logs/momentum_hype.txt
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
import attention_short_test as AS     # noqa: E402
import blowup_us as U                 # noqa: E402
import momentum_avoid as MA           # noqa: E402
import momentum_v2 as MV              # noqa: E402

VARIANTS = ["STEADY", "HYPE", "QUIET", "STEADY_TS", "HYPE_TS", "QUIET_TS"]


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
        bad = MA.hyped(spk, steady.index, t, idx[max(t_i - MA.LOOKBACK, 0)])
        page = set(spk)
        row = {"month": t2, "SPY": spy.asof(t2) / spy.asof(t) - 1}
        for v in VARIANTS:
            if v.startswith("HYPE"):
                cand = [s for s in steady.index if s in bad]
            elif v.startswith("QUIET"):
                cand = [s for s in steady.index if s in page and s not in bad]
            else:
                cand = list(steady.index)
            pick = cand[:MV.TOP]
            row[v + "_n"] = len(pick)
            if not pick:
                row[v] = 0.0
                continue
            stop = MV.STOP if v.endswith("_TS") else 0
            pk = {s: peaks[v][s] for s in pick
                  if s in peaks[v] and s in held_ts[v]} if stop else {}
            rets, stopped, newpk = MV.month_path(px, pick, t, t2, pk, stop)
            old = prev[v]
            trades = len(set(pick) - old) + len(old - set(pick)) + len(stopped)
            row[v] = np.mean([rets[s] for s in pick]) - trades / len(pick) * \
                MV.COST
            held = set(pick) - stopped
            prev[v] = held
            if stop:
                peaks[v], held_ts[v] = newpk, held
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy = MV.smoke()
        feed = "sip"
        views = pd.DataFrame(np.random.default_rng(5).lognormal(
            5, 0.6, (len(px.index), 60)), px.index, px.columns[:60])
    else:
        px, vol, spy, feed = U.load()
        views = AS.norm_index(pd.read_parquet(AS.ATT))
    U.MIN_PX = MV.MIN_PX
    sets, ever = U.liquid_members(px, vol, MV.MIN_DV if feed == "sip"
                                  else MV.MIN_DV * 0.03)
    spk = MA.spike_table(views, px)
    px = px[ever]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)

    def members(t):
        past = [d for d in days if d <= t]
        return sets[past[-1]] if past else set()
    res = run(px, spy, members, spk)
    MA.VARIANTS = VARIANTS
    res["dropped"] = 20 - res["HYPE_n"].clip(upper=20)
    old = res.index < MV.RECENT
    print(f"\n  avg picks available: HYPE {res['HYPE_n'].mean():.1f}/20, "
          f"QUIET {res['QUIET_n'].mean():.1f}/20")
    MA.report(res[old], "2017-2021 (ignore 'dropped' line)")
    MA.report(res[~old], "LAST FIVE YEARS (ignore 'dropped' line)")
    yr = (1 + res[VARIANTS + ["SPY"]]).groupby(res.index.year).prod() - 1
    print("\n  by year:  " + "".join(f"{v:>11}" for v in yr.columns))
    for y, row in yr.iterrows():
        print(f"  {y:<8} " + "".join(f"{x*100:>+10.1f}%" for x in row))
    print("\n  DIAGNOSTIC ONLY (idea came from momentum_avoid results): a "
          "strong HYPE result -> freeze rules, paper-trade forward.")


if __name__ == "__main__":
    main()
