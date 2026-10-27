"""
attention2_test.py -- three follow-ups to attention_short_test.py (where
Wikipedia view spikes predicted nothing for S&P 500 stocks).

UNIVERSES
  SP     S&P 500 point-in-time members (Alpaca prices incl. delisted)
  SMALL  liquid US stocks (>= $5, >= $5M/day, point-in-time) that are NOT
         S&P members and are below the median dollar volume of the liquid
         list that month (data/us_universe/). Benchmark: equal-weight
         average of SMALL each day (so a small-cap rally isn't credited to
         attention), with SPY shown too.

PART 1 -- SAME DAY + NEWS DIRECTION (both universes)
  Daily spike: views on day d >= 3x their median over the prior 90 days.
  a) contemporaneous: average size of the day-d move on spike days vs
     other days; average signed move.
  b) direction split: GOOD = spike day with the stock up > 2% that day,
     BAD = down > 2%. Forward return vs benchmark over 5/21/63 trading
     days, entering at the close of day d+1 (views for d are complete by
     then). Also shown: entry at the close of day d (optimistic -- the
     Wikipedia day ends 4 hours after the US close).
  PASS: GOOD or BAD shows the same-sign 21-day drift in 2017-2021-05 and
  2021-06 -> 2026-05 with |t| > 2.5 in each (d+1 entry).

PART 2 -- SMALL-CAP SPIKES (the original 7-day spike test, SMALL universe)
  spike = 7-day avg views >= 3x prior 90-day median, first in 30 days.
  PASS: same sign 21-day drift vs benchmark both periods, |t| > 2.5 each.
  (Part 1's direction split on SMALL is the other half of this test.)

PART 3 -- SLOW STEADY ATTENTION GROWTH (separate; both universes)
  Each month-end: growth = median daily views over the last 91 days /
  median over the same 91 days one year earlier. Fifths by growth;
  SPREAD = fastest-growing fifth minus fastest-shrinking fifth, equal
  weight, held one month, 10 bps per trade.
  PASS: SPREAD mean > 0 with t > 2.5 over all months AND positive in both
  periods.

t-stats: monthly averages of events, 3-lag Newey-West.
Needs data/attention/views.parquet (run fetch_attention_all.py first for
the small caps).

Run (repo root):
    python3.10 scripts/attention2_test.py --smoke
    python3.10 scripts/attention2_test.py | tee logs/attention2.txt
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
import attention_short_test as AS  # noqa: E402
import blowup_scan as B            # noqa: E402
import blowup_us as U              # noqa: E402
import factor_portfolio as FP      # noqa: E402

FIRST, RECENT = AS.FIRST, AS.RECENT
COST = 0.0010


# --------------------------------------------------------------- helpers
def forward(px, bench, ev, lag):
    """ev: sym, d (spike calendar day). Entry close = first trading day
    >= d (lag 0) or the one after (lag 1)."""
    idx = px.index
    pxf = px.ffill()
    b = bench.reindex(idx).ffill()
    rows = []
    for s, d in zip(ev["sym"], ev["d"]):
        pos = idx.searchsorted(d) + lag
        if pos >= len(idx) or not np.isfinite(px[s].iloc[pos]):
            continue
        rec = {"e": idx[pos]}
        for h in (5, 21, 63):
            if pos + h < len(idx):
                rec[h] = pxf[s].iloc[pos + h] / px[s].iloc[pos] - \
                    b.iloc[pos + h] / b.iloc[pos]
        rows.append(rec)
    d = pd.DataFrame(rows)
    return d[(d[21].abs() < 2) | d[21].isna()] if len(d) else d


def drift_line(d, lab):
    out = {}
    for h in (5, 21, 63):
        x = d.dropna(subset=[h]) if len(d) else d
        if len(x) < 20:
            out[h] = (np.nan, np.nan)
            continue
        m = x.groupby(x["e"].dt.to_period("M"))[h].mean()
        t = B.nw_t(m, 3)
        out[h] = (x[h].mean(), t)
    n = len(d.dropna(subset=[21])) if len(d) else 0
    print(f"      {lab:<24} " + "  ".join(
        f"+{h}d {out[h][0]*100:+.2f}% (t {out[h][1]:+.2f})" for h in
        (5, 21, 63)) + f"  n {n}")
    return out[21]


def split_periods(d):
    return d[d["e"] < RECENT], d[d["e"] >= RECENT]


def judge(old, new):
    (m0, t0), (m1, t1) = old, new
    return bool(np.isfinite(t0) and np.isfinite(t1) and np.sign(m0) ==
                np.sign(m1) and abs(t0) > 2.5 and abs(t1) > 2.5)


# ------------------------------------------------------------ universes
def sp_universe():
    members, union = FP.membership()
    px, spy = FP.load_prices(union, members)
    px = AS.norm_index(px)
    spy = AS.norm_index(spy.to_frame()).iloc[:, 0]
    return px, spy, members


def small_universe(sp_members):
    px, vol, spy, feed = U.load()
    px = px[[c for c in px.columns if "~" not in c]]
    vol = vol[px.columns]
    U.MIN_PX = 5.0
    sets, ever = U.liquid_members(px, vol, 5e6 if feed == "sip" else 1.5e5)
    dv = (px * vol).rolling(63, min_periods=40).mean()
    days = sorted(sets)
    small = {}
    for d in days:
        sp = sp_members(d)
        live = [s for s in sets[d] if s not in sp]
        if not live:
            small[d] = set()
            continue
        med = dv.loc[d, live].median()
        small[d] = {s for s in live if dv.at[d, s] < med}
    members = lambda t: small[max(d for d in days if d <= t)]  # noqa
    keep = sorted(set().union(*small.values()))
    px = px[keep]
    r = px.pct_change(fill_method=None).clip(-0.5, 1.0)
    mask = pd.DataFrame(False, px.index, px.columns)
    for d in days:
        rows = (px.index > d - pd.offsets.MonthEnd(1)) & (px.index <= d)
        cols = [c for c in small[d] if c in px.columns]
        mask.loc[rows, cols] = True
    bench = (1 + r.where(mask).mean(axis=1).fillna(0)).cumprod()
    spy = spy.reindex(px.index).ffill()
    print(f"  SMALL universe: {len(keep)} stocks ever, ~"
          f"{np.mean([len(v) for v in small.values()]):.0f} per month")
    return px, spy, members, bench


# ---------------------------------------------------------------- part 1
def daily_spikes(views, px, members):
    v = views.reindex(columns=[c for c in views.columns if c in px.columns])
    v = v.asfreq("D")
    base = v.shift(1).rolling(90, min_periods=60).median()
    ratio = v / base.replace(0, np.nan)
    return ratio


def part1(views, px, members, bench, label):
    print(f"\n  ----- PART 1 ({label}): same day + news direction -----")
    ratio = daily_spikes(views, px, members)
    r = px.pct_change(fill_method=None)
    rr = ratio.reindex(px.index)              # calendar day == trading day
    big, normal, signed = [], [], []
    ev = []
    for c in rr.columns:
        x, y = rr[c], r[c]
        ok = x.notna() & y.notna()
        spike = ok & (x >= 3)
        if spike.any():
            big.append(y[spike].abs())
            signed.append(y[spike])
        normal.append(y[ok & (x < 1.5)].abs())
        last = None
        for d in x[spike].index:
            if last is not None and (d - last).days < 30:
                continue
            if c not in members(d):
                continue
            last = d
            ev.append((c, d, y.loc[d]))
    big = pd.concat(big) if big else pd.Series(dtype=float)
    normal = pd.concat(normal) if normal else pd.Series(dtype=float)
    signed = pd.concat(signed) if signed else pd.Series(dtype=float)
    print(f"    same day: average move on spike days {big.mean()*100:.2f}% "
          f"vs {normal.mean()*100:.2f}% on normal days "
          f"({big.mean() / normal.mean():.1f}x); average signed move "
          f"{signed.mean()*100:+.2f}%; {(signed > 0).mean():.0%} up")
    ev = pd.DataFrame(ev, columns=["sym", "d", "move"])
    res = {}
    for grp, sel in (("GOOD (up >2% on spike day)", ev["move"] > 0.02),
                     ("BAD (down >2% on spike day)", ev["move"] < -0.02)):
        g = ev[sel]
        print(f"    {grp}: {len(g)} spikes")
        for lag, lab in ((1, "enter next close"), (0, "enter same close*")):
            d = forward(px, bench, g, lag)
            if not len(d):
                continue
            o, n = split_periods(d)
            r0 = drift_line(o, f"{lab}, 2017-21")
            r1 = drift_line(n, f"{lab}, last 5y")
            if lag == 1:
                res[grp] = judge(r0, r1)
    print("      * optimistic: Wikipedia's day ends 4h after the US close")
    return any(res.values())


# ---------------------------------------------------------------- part 2
def part2(views, px, members, bench):
    print("\n  ----- PART 2 (SMALL): 7-day spikes, as in the S&P test -----")
    ev = AS.attention_events(views, px, members)
    ev = ev.rename(columns={"e": "d"})
    print(f"    {len(ev)} spikes in {ev['sym'].nunique()} stocks")
    d = forward(px, bench, ev, 0)       # e is already the next close
    o, n = split_periods(d)
    r0 = drift_line(o, "vs small-cap avg, 2017-21")
    r1 = drift_line(n, "vs small-cap avg, last 5y")
    return judge(r0, r1)


# ---------------------------------------------------------------- part 3
def part3(views, px, members, label):
    print(f"\n  ----- PART 3 ({label}): slow steady attention growth -----")
    v = views.reindex(columns=[c for c in views.columns if c in px.columns])
    v = v.asfreq("D")
    now = v.rolling(91, min_periods=70).median()
    growth = now / now.shift(365).replace(0, np.nan)
    rows, prev1, prev5 = [], set(), set()
    for t, t2 in AS.months(px):
        mem = [s for s in px.columns if s in members(t)
               and np.isfinite(px.at[t, s])]
        g = growth.asof(t) if hasattr(growth, "asof") else None
        g = growth.loc[:t].iloc[-1] if len(growth.loc[:t]) else None
        if g is None:
            continue
        g = g.reindex(mem).replace([np.inf, -np.inf], np.nan).dropna()
        if len(g) < 50:
            continue
        q = pd.qcut(g.rank(method="first"), 5, labels=False)
        mret = AS.month_ret(px, t, t2, mem)
        s1, s5 = set(q.index[q == 4]), set(q.index[q == 0])
        turn = len(s1 ^ prev1) / max(len(s1), 1) + \
            len(s5 ^ prev5) / max(len(s5), 1)
        prev1, prev5 = s1, s5
        rows.append({"month": t2, "SPREAD": mret[list(s1)].mean() -
                     mret[list(s5)].mean() - turn * COST,
                     "GROW": mret[list(s1)].mean(),
                     "SHRINK": mret[list(s5)].mean(),
                     "ALL": mret.mean(), "n": len(g)})
    res = pd.DataFrame(rows).set_index("month")
    x = res["SPREAD"]
    t = x.mean() / (x.std() / np.sqrt(len(x)))
    o = x[x.index < RECENT].mean()
    n = x[x.index >= RECENT].mean()
    print(f"    {len(res)} months, ~{res['n'].mean():.0f} stocks/month with a "
          f"year of view history")
    print(f"    avg monthly: growing fifth {res['GROW'].mean()*100:+.2f}%, "
          f"shrinking fifth {res['SHRINK'].mean()*100:+.2f}%, all "
          f"{res['ALL'].mean()*100:+.2f}%")
    print(f"    SPREAD {x.mean()*1200:+.1f}%/yr t {t:+.2f} | 2017-21 "
          f"{o*1200:+.1f}%/yr, last 5y {n*1200:+.1f}%/yr")
    return bool(x.mean() > 0 and t > 2.5 and o > 0 and n > 0)


# ----------------------------------------------------------------- smoke
def smoke():
    px, spy, members, views, sm, _ = AS.smoke()
    return px, spy, members, views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    verdict = {}
    if a.smoke:
        px, spy, members, views = smoke()
        verdict["1 same-day/direction (SP)"] = part1(views, px, members,
                                                     spy, "SP")
        verdict["2 small-cap spikes"] = part2(views, px, members, spy)
        verdict["3 attention growth (SP)"] = part3(views, px, members, "SP")
    else:
        if not os.environ.get("ALPACA_API_KEY"):
            sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        print(f"  page views for {views.shape[1]} companies")
        px, spy, members = sp_universe()
        spy = spy.reindex(px.index).ffill()
        verdict["1 same-day/direction (SP)"] = part1(views, px, members,
                                                     spy, "SP")
        verdict["3 attention growth (SP)"] = part3(views, px, members, "SP")
        spx, sspy, smem, bench = small_universe(members)
        have = [c for c in spx.columns if c in views.columns]
        print(f"  SMALL stocks with page views: {len(have)}")
        if len(have) < 200:
            print("  too few small caps with views -- run "
                  "fetch_attention_all.py first; SMALL parts skipped")
        else:
            verdict["1 same-day/direction (SMALL)"] = part1(
                views, spx, smem, bench, "SMALL")
            verdict["2 small-cap spikes"] = part2(views, spx, smem, bench)
            verdict["3 attention growth (SMALL)"] = part3(views, spx, smem,
                                                          "SMALL")
    print("\n  ===== VERDICT (pre-stated) =====")
    for k, v in verdict.items():
        print(f"  {k:<32} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
