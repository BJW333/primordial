"""
momentum_cap.py -- ONE pre-stated test: does capping how many of the top 20
can come from the same industry fix STEADY momentum's -56% crash?

WHY: QuantConnect showed plain STEADY momentum at 24.9%/yr but a -56% worst
drop (Feb 2021 -> 2023). The drop began while the S&P was making new highs,
so market filters can't see it (vol target and 200-day filter both failed
on QC 2026-10-27). Suspected cause: in early 2021 the top 20 were all one
bet (speculative growth / EV / solar) and collapsed together.

BOOKS (monthly, US common stocks, price >= $10, >= $20M/day, top 20 by
12-1 return / 1y volatility, equal weight, 20 bps per trade -- exactly
momentum_v2.py STEADY, the version running on QuantConnect)
  STEADY   plain top 20
  CAPPED   walk down the same ranking; skip a stock if 4 already picked
           share its industry; stop at 20

INDUSTRY (no outside data, point-in-time, covers delisted stocks): at each
rebalance a stock is matched to the industry ETF its MARKET-REMOVED daily
returns (stock minus SPY) moved with most over the last 252 days, using
the 26 ETFs of sector_tailwind.py (10 SPDR sectors + XOP OIH XME SLX GDX
KRE XBI SMH ITB IYT TAN XRT JETS URA LIT KIE, each minus SPY). Best
correlation < 0.30 -> no industry (never capped).

RETURNS are built DAILY (buy-and-hold within the month), so the worst drop
is comparable to QuantConnect's, not the softer monthly number.

PRE-STATED (settings fixed before running; no second try with other caps,
thresholds or ETF lists):
  window 2017-03 -> 2026-05 (same as the QC run)
  PASS   CAPPED Sharpe > STEADY Sharpe  AND  CAPPED worst daily drop is
         shallower than STEADY's. Pass -> port to QuantConnect as a switch.
  2017-21 and last-5y splits printed as context only.

Run (repo root; ETF histories are cached from sector_tailwind.py, else
Alpaca keys must be exported):
    python3.10 scripts/momentum_cap.py --smoke
    python3.10 scripts/momentum_cap.py | tee logs/momentum_cap.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_us as U          # noqa: E402
import momentum_v2 as M        # noqa: E402
import turbo_portfolio as T    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "data", "momentum_cap_daily.csv")
CAP, MINCORR, CAND = 4, 0.30, 150
START, END = pd.Timestamp("2017-03-01"), pd.Timestamp("2026-06-01")
RECENT = pd.Timestamp("2021-06-01")


def industries(r, er, spy_r, cands, t_i):
    """stock -> industry ETF (or None), from market-removed correlations."""
    sl = slice(t_i - 251, t_i + 1)
    S = r[cands].iloc[sl].sub(spy_r.iloc[sl], axis=0)
    E = er.iloc[sl].sub(spy_r.iloc[sl], axis=0)
    out = {}
    for s in cands:
        c = E.corrwith(S[s]).dropna()
        out[s] = c.idxmax() if len(c) and c.max() >= MINCORR else None
    return out


def run(px, spy, members, etf):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    pxf = px.ffill()
    spy_r = spy.pct_change()
    er = etf.pct_change()
    daily = {"STEADY": [], "CAPPED": []}
    prev = {"STEADY": set(), "CAPPED": set()}
    diag = []
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
        sc = (mom / vol.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        ranked = list(sc.index[:CAND])
        ind = industries(r, er, spy_r, ranked, t_i)
        plain = ranked[:M.TOP]
        capped, used = [], Counter()
        for s in ranked:
            g = ind[s]
            if g is not None and used[g] >= CAP:
                continue
            capped.append(s)
            if g is not None:
                used[g] += 1
            if len(capped) == M.TOP:
                break
        pg = Counter(ind[s] for s in plain if ind[s] is not None)
        top_g, top_n = pg.most_common(1)[0] if pg else ("-", 0)
        diag.append({"month": t2, "swapped": len(set(plain) - set(capped)),
                     "top_group": top_g, "top_n": top_n,
                     "matched": sum(ind[s] is not None for s in plain)})
        for name, pick in (("STEADY", plain), ("CAPPED", capped)):
            seg = pxf.loc[t:t2, pick]
            path = (seg / seg.iloc[0]).mean(axis=1)
            d = path.pct_change().iloc[1:]
            trades = len(set(pick) ^ prev[name])
            if len(d):
                d.iloc[0] -= trades / M.TOP * M.COST
            daily[name].append(d)
            prev[name] = set(pick)
    res = pd.DataFrame({k: pd.concat(v) for k, v in daily.items()})
    res["SPY"] = spy_r.reindex(res.index)
    return res, pd.DataFrame(diag).set_index("month")


def stats(x):
    x = x.dropna()
    ann = (1 + x).prod() ** (252 / len(x)) - 1
    v = x.std() * np.sqrt(252)
    eq = (1 + x).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return ann, v, x.mean() * 252 / v, dd.min(), peak, trough


def report(res, label):
    print(f"\n  ===== {label}: {res.index[0].date()} -> "
          f"{res.index[-1].date()} =====")
    print(f"  {'book':<7} {'annual':>7} {'vol':>6} {'sharpe':>6} "
          f"{'worst':>7}   worst drop from -> to")
    out = {}
    for c in ["STEADY", "CAPPED", "SPY"]:
        a, v, s, dd, pk, tr = stats(res[c])
        print(f"  {c:<7} {a*100:>+6.1f}% {v*100:>5.1f}% {s:>6.2f} "
              f"{dd*100:>6.1f}%   {pk:%Y-%m} -> {tr:%Y-%m}")
        out[c] = (s, dd)
    return out


def smoke():
    px, vol, spy = M.smoke()
    rng = np.random.default_rng(0)
    cols = rng.choice(px.columns, size=min(40, px.shape[1]), replace=False)
    etf = pd.DataFrame({f"E{j}": px[cols[j::5]].ffill().mean(axis=1)
                        for j in range(5)})
    return px, vol, spy, etf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy, etf = smoke()
        feed = "sip"
    else:
        import sector_tailwind as ST   # noqa: E402
        px, vol, spy, feed = U.load()
    U.MIN_PX = M.MIN_PX
    min_dv = M.MIN_DV if feed == "sip" else M.MIN_DV * 0.03
    sets, ever = U.liquid_members(px, vol, min_dv)
    px = px[ever]
    spy = spy.reindex(px.index).ffill()
    if not a.smoke:
        etf = ST.load_etfs(px.index).drop(columns=["SPY"])
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731
    res, diag = run(px, spy, members, etf)
    res = res[(res.index >= START) & (res.index < END)]
    if not a.smoke:
        res.to_csv(OUT)

    full = report(res, "DECIDING: FULL PERIOD (QC window)")
    report(res[res.index < RECENT], "context 2017-2021")
    report(res[res.index >= RECENT], "context last five years")

    print("\n  ===== how concentrated was the plain top 20? =====")
    print(f"  stocks matched to an industry: {diag['matched'].mean():.1f} of 20"
          f" | cap changed {(diag['swapped'] > 0).mean()*100:.0f}% of months,"
          f" avg {diag['swapped'].mean():.1f} stocks swapped")
    print(f"  months with 5+ from one industry: {(diag['top_n'] > CAP).sum()}"
          f" of {len(diag)}")
    hot = diag[(diag.index >= "2020-10-01") & (diag.index <= "2021-06-30")]
    for m, row in hot.iterrows():
        print(f"    {m:%Y-%m}: biggest group {row['top_group']:<5} "
              f"{row['top_n']:>2} of 20 | {row['swapped']} swapped by cap")
    print("  by year: " + " | ".join(
        f"{y}: STEADY {((1 + g['STEADY']).prod() - 1)*100:+.0f}% "
        f"CAPPED {((1 + g['CAPPED']).prod() - 1)*100:+.0f}%"
        for y, g in res.groupby(res.index.year)))

    (s0, d0), (s1, d1) = full["STEADY"], full["CAPPED"]
    ok = s1 > s0 and d1 > d0
    print("\n  ===== VERDICT (pre-stated, full period) =====")
    print(f"  Sharpe {s1:.2f} vs {s0:.2f} | worst drop {d1*100:.1f}% vs "
          f"{d0*100:.1f}% -> "
          f"{'PASS -- port the cap to QuantConnect' if ok else 'FAIL'}")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")


if __name__ == "__main__":
    main()
