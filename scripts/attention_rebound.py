"""
attention_rebound.py -- after a good-news attention spike, small/mid caps
fade (attention2_test.py). Can we (1) confirm the fade on fresh mid caps,
and (2) catch the BOTTOM in real time and buy the rebound?

EVENTS: good-news attention spikes -- daily Wikipedia views >= 3x their
prior-90-day median AND the stock up > 2% that day; first per stock in
30 days. Universes (point-in-time, non-S&P, liquid >= $5 and >= $5M/day):
SMALL = below the median dollar volume, MID = above. Benchmark = equal-
weight average of the same universe.

REAL-TIME "BOTTOM" SIGNALS, checked each day from 5 to 252 trading days
after the spike (views used with a 1-day lag):
  A  ATTN_NORMAL  7-day avg views <= 1.5x the pre-spike normal level
  B  ROUND_TRIP   close <= the close the day before the spike
  C  VOL_QUIET    10-day avg volume <= 1.2x the pre-spike 60-day average
  D  PRICE_TURN   fell >= 10% from the spike-day close at some point, and
                  now closes above its 10-day average
  E  INSIDER_BUY  an officer/director open-market buy filed after the spike
  F  TIME         enter at the typical days-to-low, measured on 2017-2021
                  spikes only
  Combos: every combination of A-E (31) plus F alone. Entry = the close
  AFTER the first day all of a combo's signals are true; hold 63 trading
  days; return minus the benchmark.

PRE-STATED
  FADE (MID caps, untouched so far): 63-day return after the spike
  (entering at the next close) is NEGATIVE with t < -2.5 in both 2017-
  2021-05 and 2021-06 -> 2026-05 -> "fade confirmed".
  REBOUND (per universe): pick the combo with the highest t-stat among
  those with >= 30 trades on 2017-2021 spikes ONLY; that one PASSES if on
  last-five-year spikes its 63-day return vs the benchmark is > 0 with
  t > 2.5 and >= 30 trades.
  t-stats: monthly averages of trades, 3-lag Newey-West.

Run (repo root; all data already downloaded):
    python3.10 scripts/attention_rebound.py --smoke
    python3.10 scripts/attention_rebound.py | tee logs/attention_rebound.txt
"""
import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attention_short_test as AS  # noqa: E402
import blowup_scan as B            # noqa: E402
import blowup_us as U              # noqa: E402
import factor_portfolio as FP      # noqa: E402

ROOT = os.path.dirname(HERE)
INS = os.path.join(ROOT, "data", "insider_buys.csv.gz")
RECENT = pd.Timestamp("2021-06-01")
START = pd.Timestamp("2017-01-01")
SIGS = ["A", "B", "C", "D", "E"]
NAMES = {"A": "ATTN_NORMAL", "B": "ROUND_TRIP", "C": "VOL_QUIET",
         "D": "PRICE_TURN", "E": "INSIDER_BUY", "F": "TIME"}
COMBOS = [c for k in range(1, 6) for c in itertools.combinations(SIGS, k)]
HOLD, WIN, MINN = 63, 252, 30


# ------------------------------------------------------------- universes
def universes(sp_members):
    px, vol, spy, feed = U.load()
    px = px[[c for c in px.columns if "~" not in c]]
    vol = vol[px.columns]
    U.MIN_PX = 5.0
    sets, _ = U.liquid_members(px, vol, 5e6 if feed == "sip" else 1.5e5)
    dv = (px * vol).rolling(63, min_periods=40).mean()
    days = sorted(sets)
    small, mid = {}, {}
    for d in days:
        sp = sp_members(d)
        live = [s for s in sets[d] if s not in sp]
        med = dv.loc[d, live].median() if live else np.nan
        small[d] = {s for s in live if dv.at[d, s] < med}
        mid[d] = {s for s in live if dv.at[d, s] >= med}
    out = {}
    r = px.pct_change(fill_method=None).clip(-0.5, 1.0)
    for name, grp in (("SMALL", small), ("MID", mid)):
        def mem(t, g=grp):
            past = [d for d in days if d <= t]
            return g[past[-1]] if past else set()
        mask = pd.DataFrame(False, px.index, px.columns)
        for d in days:
            rows = (px.index > d - pd.offsets.MonthEnd(1)) & (px.index <= d)
            cols = [c for c in grp[d] if c in px.columns]
            mask.loc[rows, cols] = True
        bench = (1 + r.where(mask).mean(axis=1).fillna(0)).cumprod()
        out[name] = (mem, bench)
        print(f"  {name}: ~{np.mean([len(v) for v in grp.values()]):.0f} "
              f"stocks per month")
    return px, vol, out


# ---------------------------------------------------------------- events
def good_spikes(views, px, members):
    v = views.reindex(columns=[c for c in views.columns if c in px.columns])
    v = v.asfreq("D")
    base = v.shift(1).rolling(90, min_periods=60).median()
    ratio = (v / base.replace(0, np.nan)).reindex(px.index)
    r = px.pct_change(fill_method=None)
    ev = []
    for c in ratio.columns:
        x, y = ratio[c], r[c]
        hit = x.index[(x >= 3) & (y > 0.02)]
        last = None
        for d in hit:
            if d < START or (last is not None and (d - last).days < 30):
                continue
            if c not in members(d):
                continue
            last = d
            ev.append((c, d))
    return pd.DataFrame(ev, columns=["sym", "d"])


def insider_dates():
    if not os.path.exists(INS):
        return {}
    ins = pd.read_csv(INS, usecols=["ticker", "filing_date"],
                      parse_dates=["filing_date"]).dropna()
    ins["ticker"] = ins["ticker"].astype(str).str.upper().str.replace(
        "-", ".", regex=False)
    return {k: np.sort(g["filing_date"].values) for k, g in
            ins.groupby("ticker")}


# ------------------------------------------------------------- per event
def event_arrays(ev, px, vol, views, bench, ins):
    """For each event: signal arrays over the window + forward returns."""
    idx = px.index
    vday = views.asfreq("D")
    v7 = vday.rolling(7, min_periods=5).mean().shift(1)     # 1-day lag
    vbase = vday.shift(7).rolling(90, min_periods=60).median()
    ma10 = px.rolling(10, min_periods=8).mean()
    vol10 = vol.rolling(10, min_periods=8).mean()
    b = bench.reindex(idx).ffill()
    pxf = px.ffill()
    out = []
    for s, d in zip(ev["sym"], ev["d"]):
        p0 = idx.get_loc(d)
        if p0 < 61 or p0 + 6 >= len(idx):
            continue
        hi = min(p0 + WIN, len(idx) - 1)
        w = np.arange(p0 + 5, hi)
        c = pxf[s].values
        spike_close, pre_close = c[p0], c[p0 - 1]
        pre_vol = np.nanmean(vol[s].values[p0 - 60:p0])
        base_v = vbase[s].asof(d) if s in vbase else np.nan
        sig = {}
        if s in v7 and np.isfinite(base_v) and base_v > 0:
            vv = v7[s].reindex(idx[w]).values
            sig["A"] = vv <= 1.5 * base_v
        else:
            sig["A"] = np.zeros(len(w), bool)
        sig["B"] = c[w] <= pre_close
        sig["C"] = vol10[s].values[w] <= 1.2 * pre_vol
        fell = np.minimum.accumulate(c[p0:hi]) <= 0.9 * spike_close
        sig["D"] = fell[5:5 + len(w)] & (c[w] > ma10[s].values[w])
        f = ins.get(s)
        if f is not None and len(f):
            after = f[f > np.datetime64(d)]
            first = after[0] if len(after) else None
            sig["E"] = (idx[w].values >= first) if first is not None \
                else np.zeros(len(w), bool)
        else:
            sig["E"] = np.zeros(len(w), bool)
        for k in sig:
            sig[k] = np.nan_to_num(sig[k].astype(float), nan=0).astype(bool)
        low_day = int(np.nanargmin(c[p0:hi])) if hi > p0 else 0

        def fwd(i):
            if i + HOLD >= len(idx) or not np.isfinite(c[i]):
                return np.nan
            return c[i + HOLD] / c[i] - b.iloc[i + HOLD] / b.iloc[i]
        path = {h: (c[p0 + 1 + h] / c[p0 + 1] - b.iloc[p0 + 1 + h] /
                    b.iloc[p0 + 1]) if p0 + 1 + h < len(idx) else np.nan
                for h in (21, 63, 126, 252)}
        out.append({"sym": s, "d": d, "p0": p0, "w": w, "sig": sig,
                    "low_day": low_day, "path": path, "fwd": fwd})
    return out


def trades(evs, combo, time_lag=None):
    rows = []
    for e in evs:
        if time_lag is not None:
            i = e["p0"] + time_lag
        else:
            m = np.ones(len(e["w"]), bool)
            for k in combo:
                m &= e["sig"][k]
            hit = np.flatnonzero(m)
            if not len(hit):
                continue
            i = e["w"][hit[0]] + 1                 # next close
        r = e["fwd"](i)
        if np.isfinite(r) and abs(r) < 3:
            rows.append((e["d"], r))
    return pd.DataFrame(rows, columns=["d", "r"])


def tstat(tr):
    if len(tr) < 5:
        return np.nan
    m = tr.groupby(tr["d"].dt.to_period("M"))["r"].mean()
    return B.nw_t(m, 3)


# ---------------------------------------------------------------- report
def run_universe(name, evs):
    print(f"\n  ===== {name}: {len(evs)} good-news spikes =====")
    old = [e for e in evs if e["d"] < RECENT]
    new = [e for e in evs if e["d"] >= RECENT]
    print("  path after the spike (vs benchmark, entering next close):")
    for lab, grp in (("2017-21", old), ("last 5y", new)):
        cells = []
        for h in (21, 63, 126, 252):
            x = pd.Series([e["path"][h] for e in grp]).dropna()
            x = x[x.abs() < 3]
            cells.append(f"+{h}d {x.mean()*100:+.1f}%")
        print(f"    {lab:<8} " + "  ".join(cells))
    fade = {}
    for lab, grp in (("2017-21", old), ("last 5y", new)):
        tr = pd.DataFrame([(e["d"], e["path"][63]) for e in grp],
                          columns=["d", "r"]).dropna()
        tr = tr[tr["r"].abs() < 3]
        fade[lab] = (tr["r"].mean(), tstat(tr), len(tr))
    print("  FADE (63d after spike): " + " | ".join(
        f"{k} {v[0]*100:+.2f}% t {v[1]:+.2f} n {v[2]}"
        for k, v in fade.items()))
    lag = int(np.median([e["low_day"] for e in old])) if old else 63
    print(f"  typical days from spike to the low (2017-21 only): {lag}")
    table = []
    for combo in COMBOS:
        t0 = trades(old, combo)
        t1 = trades(new, combo)
        table.append(("+".join(combo), t0, t1))
    t0 = trades(old, (), time_lag=lag)
    t1 = trades(new, (), time_lag=lag)
    table.append(("F", t0, t1))
    print(f"\n  {'combo':<12} | {'2017-21: n':>10} {'mean':>7} {'t':>6} | "
          f"{'last5y: n':>9} {'mean':>7} {'t':>6}")
    sel = []
    for lab, a, b in table:
        ta, tb = tstat(a), tstat(b)
        sel.append((lab, len(a), ta, len(b), b["r"].mean() if len(b) else
                    np.nan, tb))
        if len(a) >= 10 or len(b) >= 10:
            print(f"  {lab:<12} | {len(a):>10} "
                  f"{(a['r'].mean() if len(a) else np.nan)*100:>+6.2f}% "
                  f"{ta:>+6.2f} | {len(b):>9} "
                  f"{(b['r'].mean() if len(b) else np.nan)*100:>+6.2f}% "
                  f"{tb:>+6.2f}")
    ok = [x for x in sel if x[1] >= MINN and np.isfinite(x[2])]
    if not ok:
        print("  no combo with >= 30 trades in 2017-21")
        return fade, None, False
    best = max(ok, key=lambda x: x[2])
    lab, n0, t0v, n1, m1, t1v = best
    passed = bool(n1 >= MINN and m1 > 0 and t1v > 2.5)
    print(f"\n  SELECTED on 2017-21 (highest t, >= 30 trades): {lab} "
          f"({' + '.join(NAMES[k] for k in lab.split('+'))}), t {t0v:+.2f}")
    print(f"  judged on last 5y: n {n1}, mean {m1*100:+.2f}% vs benchmark, "
          f"t {t1v:+.2f} -> {'PASS' if passed else 'FAIL'}")
    return fade, lab, passed


def smoke():
    px, spy, members, views, sm, _ = AS.smoke()
    rng = np.random.default_rng(31)
    vol = pd.DataFrame(rng.lognormal(12, 0.3, px.shape), px.index,
                       px.columns)
    bench = px.mean(axis=1)
    return px, vol, views, {"SMALL": (members, bench)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, views, unis = smoke()
        ins = {}
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = universes(sp_members)
        ins = insider_dates()
        print(f"  insider-buy histories for {len(ins)} tickers")
    verdict = {}
    for name, (mem, bench) in unis.items():
        ev = good_spikes(views, px, mem)
        evs = event_arrays(ev, px, vol, views, bench, ins)
        fade, lab, passed = run_universe(name, evs)
        verdict[f"REBOUND {name} ({lab})"] = passed
        if name == "MID":
            (m0, t0, _), (m1, t1, _) = fade["2017-21"], fade["last 5y"]
            verdict["FADE confirmed on MID caps"] = bool(
                m0 < 0 and m1 < 0 and t0 < -2.5 and t1 < -2.5)
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    for k, v in verdict.items():
        print(f"  {k:<40} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
