"""
momentum_variants.py -- change WHICH stocks the book picks instead of timing
the same 20. Momentum's worst crash (Feb 2021 -> 2022) was a STYLE crash:
the top 20 were all the same kind of hot growth stock and fell together
while the S&P rose. Three published momentum versions were built to avoid
exactly that. Same universe, costs and 30% lock as CHIMERA; only the
ranking changes.

BOOKS (monthly, US common stocks, price >= $10, >= $20M/day, top 20 equal
weight, 20 bps per trade -- identical to momentum_v2.py STEADY except the
score)
  STEADY    12-1 return / 1y volatility                 (CHIMERA today)
  RESIDUAL  Blitz, Huij & Martens (2011): rank on the part of each stock's
            12-1 return NOT explained by the market (daily regression on
            SPY over days t-252..t-21; sum of leftover returns / their
            volatility). Strips out "it just went up with the market /
            the hot style". Paper: about half the crash risk.
  FROG      Da, Gurun & Warachka (2014, "Frog in the Pan"): among the 100
            best 12-1 returns, take the 20 whose gains came in many small
            steps (lowest "information discreteness" = sign(return) x
            (% down days - % up days)). Paper: continuous winners keep
            winning; jumpy winners reverse.
  HIGH52    George & Hwang (2004): closest to its 52-week high
            (steady score breaks ties). Paper: no long-run reversal.

Each is shown plain and with the 30% yearly lock (lock -> SPY until Jan 1,
cash = T-bills from Ken French's file you already downloaded).

PRE-STATED (the three variants come from the literature, fixed now; no
others will be tried on this data)
  A variant replaces STEADY only if VARIANT+LOCK30 beats STEADY+LOCK30 on
  BOTH Sharpe and worst drop over 2017-03 -> 2026-05, AND has the higher
  Sharpe in the last five years (2021-06 -> 2026-05). If two pass, the
  higher full-period Sharpe wins. Pass -> port to QC.

Run (repo root; rebuilds 4 books, ~3-6 minutes):
    python3.10 scripts/momentum_variants.py --smoke
    python3.10 scripts/momentum_variants.py | tee logs/momentum_variants.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_us as U              # noqa: E402
import crash_warning_search as CW  # noqa: E402
import momentum_v2 as M            # noqa: E402

START, END, RECENT = pd.Timestamp("2017-03-01"), pd.Timestamp("2026-06-01"), \
    pd.Timestamp("2021-06-01")
NAMES = ["STEADY", "RESIDUAL", "FROG", "HIGH52"]


def scores(px, r, spy_r, elig, t_i):
    W = px[elig].iloc[t_i - 252:t_i + 1]
    mom = W.iloc[-22] / W.iloc[0] - 1
    R = r[elig].iloc[t_i - 251:t_i + 1]
    vol = R.std() * np.sqrt(252)
    steady = mom / vol.replace(0, np.nan)
    out = {"STEADY": steady}
    # RESIDUAL: regress on SPY over the 12-1 window
    Rw = R.iloc[:-21].fillna(0.0).values
    x = spy_r.iloc[t_i - 251:t_i - 20].fillna(0.0).values
    xc = x - x.mean()
    beta = (xc[:, None] * (Rw - Rw.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()
    resid = Rw - Rw.mean(axis=0) - beta[None, :] * xc[:, None]
    out["RESIDUAL"] = pd.Series(resid.sum(axis=0) / resid.std(axis=0), index=elig)
    # FROG: among the top 100 raw 12-1 returns, the most continuous gains
    top = mom.dropna().sort_values(ascending=False).index[:100]
    Rt = R[top].iloc[:-21]
    idisc = np.sign(mom[top]) * ((Rt < 0).mean() - (Rt > 0).mean())
    out["FROG"] = -idisc.reindex(elig)                     # low ID = good
    out["HIGH52"] = W.iloc[-1] / W.max() + steady.rank(pct=True) * 1e-6
    return out


def build(px, spy, members):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    spy_r = spy.pct_change()
    pxf = px.ffill()
    daily = {n: [] for n in NAMES}
    prev = {n: set() for n in NAMES}
    picks = {n: {} for n in NAMES}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        if t_i < 260:
            continue
        cnt = px.iloc[t_i - 252:t_i + 1].notna().sum()
        elig = [s for s in members(t) if cnt.get(s, 0) >= 240]
        if len(elig) < 100:
            continue
        sc = scores(px, r, spy_r, elig, t_i)
        for n in NAMES:
            s = sc[n].replace([np.inf, -np.inf], np.nan).dropna()
            pick = list(s.sort_values(ascending=False).index[:M.TOP])
            seg = pxf.loc[t:t2, pick]
            d = (seg / seg.iloc[0]).mean(axis=1).pct_change().iloc[1:]
            if len(d):
                d.iloc[0] -= len(set(pick) ^ prev[n]) / M.TOP * M.COST
            prev[n] = set(pick)
            picks[n][t2] = set(pick)
            daily[n].append(d)
    res = pd.DataFrame({n: pd.concat(v) for n, v in daily.items()})
    res["SPY"] = spy_r.reindex(res.index).fillna(0)
    return res, picks


def stats(r):
    r = np.asarray(r)
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    eq = np.cumprod(1 + r)
    return ann, r.mean() / r.std() * np.sqrt(252), (eq / np.maximum.accumulate(eq) - 1).min()


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
    res, picks = build(px, spy, members)
    res = res[(res.index >= START) & (res.index < END)]
    if a.smoke:
        rf = pd.Series(0.0001, res.index)
    else:
        rf = CW.load_french()["RF"].reindex(res.index).ffill().fillna(0)
    print(f"  {res.index[0].date()} -> {res.index[-1].date()}, {len(res)} days"
          f"{'  [SYNTHETIC]' if a.smoke else ''}")

    locked = {n: CW.lock_path(res[n], res["SPY"], rf, None) for n in NAMES}
    out = {}
    print(f"\n  {'book':<10} {'plain: annual  Sharpe  worst':>30}   "
          f"{'+LOCK30: annual  Sharpe  worst':>32}   last5y Sharpe  2021-22 drop")
    for n in NAMES:
        p = stats(res[n])
        lk = stats(locked[n])
        rec = stats(locked[n][locked[n].index >= RECENT])
        seg = locked[n][(locked[n].index >= "2021-01-01") & (locked[n].index <= "2022-12-31")]
        eq = (1 + seg).cumprod()
        out[n] = (lk[1], lk[2], rec[1])
        print(f"  {n:<10} {p[0]*100:>+12.1f}% {p[1]:>6.2f} {p[2]*100:>6.1f}%   "
              f"{lk[0]*100:>+14.1f}% {lk[1]:>6.2f} {lk[2]*100:>6.1f}%   "
              f"{rec[1]:>12.2f}  {(eq/eq.cummax()-1).min()*100:>10.1f}%")
    sp = stats(res["SPY"])
    print(f"  {'SPY':<10} {sp[0]*100:>+12.1f}% {sp[1]:>6.2f} {sp[2]*100:>6.1f}%")

    print("\n  year by year, with the 30% lock")
    print("  year " + "".join(f"{n:>10}" for n in NAMES))
    for y in sorted(set(res.index.year)):
        print(f"  {y} " + "".join(
            f"{((1 + locked[n][locked[n].index.year == y]).prod() - 1)*100:>+9.0f}%"
            for n in NAMES))

    print("\n  how different are they from STEADY?")
    for n in NAMES[1:]:
        c = res[n].corr(res["STEADY"])
        ov = np.mean([len(picks[n][m] & picks["STEADY"].get(m, set())) / M.TOP
                      for m in picks[n]])
        print(f"    {n:<9} daily correlation {c:.2f}, same stocks {ov*100:.0f}% of the time")

    s0, d0, r0 = out["STEADY"]
    winners = [n for n in NAMES[1:] if out[n][0] > s0 and out[n][1] > d0 and out[n][2] > r0]
    print("\n  ===== VERDICT (pre-stated, each vs STEADY+LOCK30) =====")
    for n in NAMES[1:]:
        s, d, r = out[n]
        print(f"    {n:<9} Sharpe {s:.2f} vs {s0:.2f} | worst {d*100:.1f}% vs {d0*100:.1f}% "
              f"| last5y {r:.2f} vs {r0:.2f} -> {'PASS' if n in winners else 'FAIL'}")
    if winners:
        best = max(winners, key=lambda n: out[n][0])
        print(f"    -> replace STEADY with {best} (port to QC)")
    else:
        print("    -> keep STEADY")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")


if __name__ == "__main__":
    main()
