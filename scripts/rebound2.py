"""
rebound2.py -- second round of "will this faded stock come back?" signals,
each aimed at a REASON a stock actually rebounds (round 1 -- earnings beat,
market washout, trend back, capitulation -- all failed).

EVENTS: good-news attention spikes on non-S&P liquid small + mid caps.
FADE DETECTED = >= 10% behind the same-size benchmark since the spike.
Checked daily from day 42 to day 378 after the spike.

SIGNALS (each must happen while the fade is detected; all point-in-time):
  Q  SQUEEZE_FUEL  heavily shorted (20-day short share of volume in the top
                   20% of all stocks that day, FINRA daily data, 2018-08+)
                   AND ignition: up >= 10% vs the benchmark over 5 days on
                   5-day volume >= 2x its 60-day average
  A  REIGNITE      a NEW good-news attention spike >= 30 trading days after
                   the first one (the crowd comes back)
  V  ACTIVIST      a new Schedule 13D (someone took a 5%+ stake to push for
                   change) filed after the spike
  Y  BUYBACK       share count down >= 2% vs a year earlier (latest SEC
                   filing) -- the company is buying its own stock
  Rules: every combination (15). Buy at the close AFTER the first day the
  rule is true; hold 126 trading days; return minus the benchmark; trades
  beyond +/-300% dropped.

PRE-STATED (stricter: ~80 rebound tests already run tonight):
  Pick the rule with the highest t (>= 30 trades) on 2017-2021-05 spikes;
  PASS if on 2021-06+ spikes its mean > 0 with t > 3.0 (>= 30 trades).

Run (repo root):
    python3.10 scripts/fetch_rebound_data.py --email you@example.com
    python3.10 scripts/rebound2.py | tee logs/rebound2.txt
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
import attention_rebound as AR        # noqa: E402
import attention_short_test as AS     # noqa: E402
import factor_portfolio as FP         # noqa: E402
import fade_predict as P              # noqa: E402
import rebound_signals as RS          # noqa: E402

ROOT = os.path.dirname(HERE)
SVF = os.path.join(ROOT, "data", "short", "short_volume_all.parquet")
ADF = os.path.join(ROOT, "data", "activist_13d.csv")
SIGS = ["Q", "A", "V", "Y"]
NAMES = {"Q": "SQUEEZE_FUEL", "A": "REIGNITE", "V": "ACTIVIST",
         "Y": "BUYBACK"}
RULES = [c for k in range(1, 5) for c in itertools.combinations(SIGS, k)]
LO, HI, HOLD = RS.LO, RS.HI, RS.HOLD
MINN, BAR = 30, 3.0


def short_rank(px):
    if not os.path.exists(SVF):
        print("  (no short_volume_all.parquet -- SQUEEZE_FUEL all False)")
        return None
    sv = pd.read_parquet(SVF)
    sv["ticker"] = sv["ticker"].str.replace("-", ".", regex=False)
    sv = sv[sv["ticker"].isin(px.columns)]
    sh = sv.pivot_table(index="date", columns="ticker", values="short",
                        aggfunc="sum")
    tot = sv.pivot_table(index="date", columns="ticker", values="total",
                         aggfunc="sum")
    sh.index, tot.index = pd.to_datetime(sh.index), pd.to_datetime(tot.index)
    ratio = sh.rolling(20, min_periods=15).sum() / \
        tot.rolling(20, min_periods=15).sum()
    rk = ratio.rank(axis=1, pct=True)
    print(f"  short-volume ranks: {rk.shape[1]} stocks, "
          f"{rk.index[0]:%Y-%m-%d} -> {rk.index[-1]:%Y-%m-%d}")
    return rk.reindex(px.index).reindex(columns=px.columns)


def activist_dates():
    if not os.path.exists(ADF):
        print("  (no activist_13d.csv -- ACTIVIST all False)")
        return {}
    a = pd.read_csv(ADF, parse_dates=["filed"])
    a["ticker"] = a["ticker"].astype(str).str.upper()
    return {k: np.sort(g["filed"].values) for k, g in a.groupby("ticker")}


def events(ev, all_ev, px, vol, bench, srank, act, sh):
    idx = px.index
    pxf = px.ffill()
    cv = pxf.values
    cols = {c: j for j, c in enumerate(px.columns)}
    b = bench.reindex(idx).ffill().values
    v5 = vol.rolling(5, min_periods=4).mean().reindex_like(px).values
    v60 = vol.shift(5).rolling(60, min_periods=40).mean().reindex_like(
        px).values
    sr = srank.values if srank is not None else None
    later = {k: np.sort(g["d"].values) for k, g in all_ev.groupby("sym")}
    out = []
    for s, d in zip(ev["sym"], ev["d"]):
        p0 = idx.get_loc(d)
        if p0 + LO + HOLD + 2 >= len(idx):
            continue
        j = cols[s]
        c = cv[:, j]
        w = np.arange(p0 + LO, min(p0 + HI, len(idx) - HOLD - 2))
        if not len(w) or not np.isfinite(c[p0 + 1]):
            continue
        rel = c[w] / c[p0 + 1] - b[w] / b[p0 + 1]
        faded = np.nan_to_num(rel, nan=0) <= -0.10
        sig = {}
        r5 = c[w] / c[w - 5] - b[w] / b[w - 5]
        ign = (np.nan_to_num(r5, nan=0) >= 0.10) & \
            (np.nan_to_num(v5[w, j], nan=0) >= 2 * np.nan_to_num(
                v60[w, j], nan=np.inf))
        heavy = np.nan_to_num(sr[w, j], nan=0) >= 0.8 if sr is not None \
            else np.zeros(len(w), bool)
        sig["Q"] = ign & heavy
        lt = later.get(s, np.array([], "datetime64[ns]"))
        lt = lt[lt >= np.datetime64(idx[min(p0 + 30, len(idx) - 1)])]
        if len(lt):
            sig["A"] = idx[w].values >= lt[0]
        else:
            sig["A"] = np.zeros(len(w), bool)
        f = act.get(s)
        if f is not None and len(f):
            aft = f[f > np.datetime64(d)]
            sig["V"] = idx[w].values >= aft[0] if len(aft) else \
                np.zeros(len(w), bool)
        else:
            sig["V"] = np.zeros(len(w), bool)
        y = np.zeros(len(w), bool)
        if s in sh:
            for k in range(0, len(w), 21):          # monthly check, held
                ch = P.dilution(sh, s, idx[w[k]])
                if np.isfinite(ch) and ch <= -0.02:
                    y[k:k + 21] = True
        sig["Y"] = y
        for k_ in sig:
            sig[k_] = sig[k_] & faded

        def fwd(i, c=c):
            if not (np.isfinite(c[i]) and np.isfinite(c[i + HOLD])):
                return np.nan
            return c[i + HOLD] / c[i] - b[i + HOLD] / b[i]
        out.append({"d": d, "w": w, "sig": sig, "faded": faded, "fwd": fwd})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, views, unis = AR.smoke()
        srank, act, sh = None, {}, {}
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = AR.universes(sp_members)
        srank, act, sh = short_rank(px), activist_dates(), P.load_shares()
        print(f"  activist-13D histories {len(act)} tickers, share counts "
              f"{len(sh)} tickers")
    evs = []
    for name, (mem, bench) in unis.items():
        ev = AR.good_spikes(views, px, mem)
        evs += events(ev, ev, px, vol, bench, srank, act, sh)
    old = [e for e in evs if e["d"] < AR.RECENT]
    new = [e for e in evs if e["d"] >= AR.RECENT]
    print(f"\n  {len(evs)} spikes; {len(old)} in 2017-21, {len(new)} 2021-06+")
    print(f"\n  {'rule':<40} | {'2017-21 n':>9} {'mean':>7} {'med':>7} "
          f"{'win':>4} {'t':>6} | {'2021+ n':>7} {'mean':>7} {'med':>7} "
          f"{'win':>4} {'t':>6}")
    sel = []
    for rule in [None] + RULES:
        lab = "BASELINE: fade detected" if rule is None else \
            "+".join(NAMES[k] for k in rule)
        s0, s1 = RS.line(RS.trades(old, rule)), RS.line(RS.trades(new, rule))
        if rule is not None:
            sel.append((rule, s0, s1))

        def f(s):
            return (f"{s[0]:>7} {s[1]*100:>+6.2f}% {s[2]*100:>+6.2f}% "
                    f"{s[3]*100:>3.0f}% {s[4]:>+6.2f}")
        print(f"  {lab:<40} | {f(s0):>40} | {f(s1)}")
    ok = [x for x in sel if x[1][0] >= MINN and np.isfinite(x[1][4])]
    if not ok:
        print("\n  no rule with >= 30 trades in 2017-21 -> FAIL")
        return
    rule, s0, s1 = max(ok, key=lambda x: x[1][4])
    lab = " + ".join(NAMES[k] for k in rule)
    p = bool(s1[0] >= MINN and s1[1] > 0 and s1[4] > BAR)
    print(f"\n  SELECTED on 2017-21 (highest t): {lab} (n {s0[0]}, "
          f"{s0[1]*100:+.2f}%, t {s0[4]:+.2f})")
    print(f"  2021-06+: n {s1[0]}, {s1[1]*100:+.2f}% vs benchmark, t "
          f"{s1[4]:+.2f} (bar t > {BAR}) -> {'PASS' if p else 'FAIL'}")
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    print(f"  rebound signal round 2 ({lab})  {'PASS' if p else 'FAIL'}")


if __name__ == "__main__":
    main()
