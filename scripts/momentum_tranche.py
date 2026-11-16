"""
momentum_tranche.py -- fix the HOLDINGS, not the timing.

THE PROBLEM: losing years came from leadership flips (Q4 2018, Jan 2022,
H1 2023). The whole book re-ranks once a month on the 1st, so after a flip it
holds yesterday's winners for up to a month. Faster on/off switches (REGIME
weekly, stops) failed because they also miss rebounds.

IDEA 1 -- TRANCHES (Newfound Research, "rebalance timing luck"): split the
  book into 4 equal sub-books that each re-rank monthly on a DIFFERENT week
  (trading day 0, 5, 10, 15 after month-end). Every week a quarter of the
  money moves to the fresh top 20. After a flip, 25% has rotated within a
  week, 50% within two -- without ever cutting exposure, so rebounds are not
  missed. Newfound measured ~4.4%/yr of pure luck from the rebalance DATE
  alone in momentum; tranching averages that luck away.
  This script also shows how lucky "rebalance on the 1st" has been for us.

IDEA 2 -- SPEED BLEND (Goulding-Harvey-Mazzoleni, "Momentum Turning Points":
  blends of fast and slow momentum beat either alone): half the book ranked on
  12-month momentum (as now), half on 6-month momentum (return 6 months ago ->
  1 month ago / 6-month volatility). The 6-month half picks up new leaders
  sooner after a flip.

Rejected before testing: cheap crash insurance (buying puts when markets are
calm) -- research (2015, S&P 1996-2014) finds puts lose money even when
volatility is low.

All books: same universe / filters / top 20 / 20 bps costs as STEADY
(momentum_cap.py). Account = 70% book + 30% T-bills, 30% lock -> SPY, and
with REGIME on top (calm 100 / normal 70 / stressed 35, monthly, from Ken
French US winners as in regime_sizer.py). Stock data: 2017-03 -> 2026-05 only.

PRE-STATED DECISIONS
  TRANCHES  adopt if, with REGIME, TRANCHE4 beats the AVERAGE of the 4
            single-date versions on Sharpe (above T-bills) AND worst drop.
            (Judged against the average, not the 1st-of-month version,
            because one date can be lucky.)
  SPEED     adopt only if, with REGIME and tranches, the blend beats the
            12-month-only book on Sharpe AND worst drop over the full period
            AND has a higher Sharpe in BOTH halves (2017-03 -> 2021-05 and
            2021-06 -> 2026-05), with no more losing years.

Run (repo root; stock data in data/us_universe/, French data on disk):
    python3.10 scripts/momentum_tranche.py --smoke
    python3.10 scripts/momentum_tranche.py | tee logs/momentum_tranche.txt
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
import blowup_us as U              # noqa: E402
import crash_warning_search as CW  # noqa: E402
import dd2_lab as D2               # noqa: E402
import judge as J                  # noqa: E402
import momentum_v2 as M            # noqa: E402
import regime_sizer as RS          # noqa: E402

OFFSETS = (0, 5, 10, 15)
P3 = ("2017-03-01", "2026-05-29")
SPLIT = pd.Timestamp("2021-06-01")
BAD = ["2018-10", "2018-11", "2018-12", "2020-02", "2020-03", "2022-01",
       "2023-03", "2023-10", "2025-02", "2025-03", "2025-04"]


def book(px, members, offset, lookback):
    """Daily returns of one sub-book: re-ranks monthly, `offset` trading days
    after month-end, on (lookback-day return to 1 month ago) / volatility."""
    idx = px.index
    r = px.pct_change(fill_method=None)
    pxf = px.ffill()
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    me_i = [idx.get_indexer([idx[idx <= d][-1]])[0] for d in me]
    reb = sorted({i + offset for i in me_i if i + offset < len(idx)})
    daily, prev = [], set()
    min_bars = int(lookback * 240 / 252)
    for k in range(len(reb) - 1):
        t_i, t2_i = reb[k], reb[k + 1]
        if t_i < lookback + 10:
            continue
        t, t2 = idx[t_i], idx[t2_i]
        win = px.iloc[t_i - lookback:t_i + 1]
        cnt = win.notna().sum()
        elig = [s for s in members(t) if cnt.get(s, 0) >= min_bars]
        if len(elig) < 100:
            continue
        W = win[elig]
        mom = W.iloc[-22] / W.iloc[0] - 1
        vol = r[elig].iloc[t_i - lookback + 1:t_i + 1].std() * np.sqrt(252)
        sc = (mom / vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan) \
            .dropna().sort_values(ascending=False)
        pick = list(sc.index[:M.TOP])
        seg = pxf.loc[t:t2, pick]
        path = (seg / seg.iloc[0]).mean(axis=1)
        dd = path.pct_change().iloc[1:]
        if len(dd):
            dd.iloc[0] -= len(set(pick) ^ prev) / M.TOP * M.COST
        daily.append(dd)
        prev = set(pick)
    return pd.concat(daily)


def account(b, spy, cash, reg, regime):
    D = D2.prep(b, spy.reindex(b.index).fillna(0), cash.reindex(b.index).ffill().fillna(0))
    D["reg"] = reg.reindex(b.index).ffill().fillna(RS.NORMAL).values
    return (RS.sim(D, ("none",)) if regime else RS.sim(D, fixed=0.70)), D["c"]


def row(label, r, rf):
    a, s, w = J.stats(r, rf)
    lo, ny = J.losing(r)
    print(f"    {label:<30}{a*100:>+7.1f}%{s:>7.2f}{w*100:>8.1f}%   {lo}/{ny}")
    return a, s, w, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = CW.smoke_data() if a.smoke else CW.load_french()
    reg_us = RS.regime_sizes(d["W"])
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
    members = lambda t: sets[max(x for x in days if x <= t)]  # noqa: E731
    spy_r = spy.pct_change()
    cash = d["RF"].reindex(px.index).ffill().fillna(0)

    print("  building sub-books (8 = 4 rebalance weeks x 12m / 6m momentum) ...")
    B = {}
    for lb, tag in ((252, "12m"), (126, "6m")):
        for off in OFFSETS:
            B[(tag, off)] = book(px, members, off, lb)
    p3 = (str(B[("12m", 15)].index[0].date()), str(B[("12m", 0)].index[-1].date())) \
        if a.smoke else P3
    common = None
    for s in B.values():
        common = s.index if common is None else common.intersection(s.index)
    common = common[(common >= p3[0]) & (common <= p3[1])]
    B = {k: v.reindex(common).fillna(0) for k, v in B.items()}
    books = {f"12m week{o // 5 + 1}": B[("12m", o)] for o in OFFSETS}
    books["TRANCHE4 12m"] = sum(B[("12m", o)] for o in OFFSETS) / 4
    books["TRANCHE4 12m+6m blend"] = sum(B[(t, o)] for t in ("12m", "6m") for o in OFFSETS) / 8
    books["12m+6m blend, week1 only"] = (B[("12m", 0)] + B[("6m", 0)]) / 2
    print(f"  period {common[0].date()} -> {common[-1].date()} | week1 = rebalance on "
          f"the 1st (today's CHIMERA)")

    res = {}
    for regime in (False, True):
        print(f"\n  ===== {'WITH REGIME' if regime else 'FIXED 70%'} "
              f"(70% book or REGIME size + T-bills, 30% lock) =====")
        print(f"    {'book':<30}{'annual':>8}{'Sharpe':>7}{'worst':>9}   losing yrs")
        for nm, b in books.items():
            r, rf = account(b, spy_r, cash, reg_us, regime)
            res[(regime, nm)] = (r, rf, row(nm, r, rf))
        singles = [res[(regime, f"12m week{i}")][2] for i in (1, 2, 3, 4)]
        avg = np.mean(singles, axis=0)
        print(f"    {'(average of the 4 weeks)':<30}{avg[0]*100:>+7.1f}%{avg[1]:>7.2f}{avg[2]*100:>8.1f}%")
        rank = sorted(range(4), key=lambda i: -singles[i][1]).index(0) + 1
        print(f"    rebalancing on the 1st ranks #{rank} of 4 on Sharpe "
              f"(spread {min(x[1] for x in singles):.2f} to {max(x[1] for x in singles):.2f})")

    # ---------------- decisions (with REGIME)
    singles = [res[(True, f"12m week{i}")][2] for i in (1, 2, 3, 4)]
    avg = np.mean(singles, axis=0)
    t4 = res[(True, "TRANCHE4 12m")][2]
    ok_t = t4[1] > avg[1] and t4[2] >= avg[2] - 1e-9
    bl = res[(True, "TRANCHE4 12m+6m blend")]
    tr = res[(True, "TRANCHE4 12m")]
    halves = []
    for m in (bl[0].index < SPLIT, bl[0].index >= SPLIT):
        halves.append((J.stats(bl[0][m], bl[1][m])[1], J.stats(tr[0][m], tr[1][m])[1]))
    ok_s = (bl[2][1] > tr[2][1] and bl[2][2] >= tr[2][2] - 1e-9 and bl[2][3] <= tr[2][3]
            and all(x > y for x, y in halves))

    print("\n  ===== BAD MONTHS (with REGIME): week1 / TRANCHE4 / TRANCHE4 blend =====")
    w1, t4r, blr = res[(True, "12m week1")][0], tr[0], bl[0]
    for m in BAD:
        sel = w1.index.strftime("%Y-%m") == m
        if sel.any():
            g = lambda r: ((1 + r[sel]).prod() - 1) * 100  # noqa: E731
            print(f"    {m}: {g(w1):+6.1f}%  {g(t4r):+6.1f}%  {g(blr):+6.1f}%")
    print("  by year (with REGIME): week1 / TRANCHE4 / TRANCHE4 blend")
    for y in sorted(set(w1.index.year)):
        g = lambda r: ((1 + r[r.index.year == y]).prod() - 1) * 100  # noqa: E731
        print(f"    {y}: {g(w1):+5.0f}% {g(t4r):+5.0f}% {g(blr):+5.0f}%")

    print("\n  ===== VERDICT (pre-stated, with REGIME) =====")
    print(f"    TRANCHES: Sharpe {t4[1]:.2f} vs 4-week average {avg[1]:.2f}, worst "
          f"{t4[2]*100:.1f}% vs {avg[2]*100:.1f}% -> {'ADOPT' if ok_t else 'FAIL'}")
    print(f"    SPEED BLEND: Sharpe {bl[2][1]:.2f} vs {tr[2][1]:.2f}, worst {bl[2][2]*100:.1f}% vs "
          f"{tr[2][2]*100:.1f}%, losing yrs {bl[2][3]} vs {tr[2][3]}, halves "
          f"{halves[0][0]:.2f}/{halves[0][1]:.2f} and {halves[1][0]:.2f}/{halves[1][1]:.2f} -> "
          f"{'ADOPT' if ok_s else 'FAIL'}")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
