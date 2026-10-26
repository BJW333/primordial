"""
insider_season.py -- two non-momentum stock pickers, each tested on its
own. Needs data/us_universe/ (build_us_universe.py) and
data/insider_buys.csv.gz (fetch_insider_buys.py).

INSIDER  insider cluster buys
  A cluster = 3+ DIFFERENT officers/directors buying their own company's
  stock on the open market (each >= $25k) with SEC filing dates inside 30
  days. Signal date = filing date of the 3rd buyer (the day it became
  public). One cluster per stock per 90 days.
  Portfolio: each month-end, hold stocks with a cluster in the last 90
  days (up to 20, biggest total $ bought first), 1/20 each; empty slots
  sit in SPY. Universe >= $5 and >= $5M/day traded (insiders cluster in
  smaller companies); 30 bps per trade.
  Also printed: the raw drift -- return 3 and 6 months after the signal
  minus SPY, averaged over every cluster.

SEASON   same-calendar-month seasonality (Heston & Sadka 2008)
  Score = the stock's average return in the coming calendar month across
  all prior years with data (min 3 years; prices start 2016, so the first
  scores exist in 2019). Top 20, equal weight, monthly. Universe >= $10
  and >= $20M/day; 20 bps per trade.

PRE-STATED (two ideas tested at once, so BOTH periods must pass)
  A strategy PASSES if, in 2017-2021 (SEASON: 2019-2021) AND in the last
  five years (2021-06 -> 2026-05), it beats SPY leveraged to the same
  volatility (4%/yr borrow) AND has Sharpe above SPY's.

Run (repo root):
    python3.10 scripts/insider_season.py --smoke
    python3.10 scripts/insider_season.py | tee logs/insider_season.txt
"""
import argparse
import os
import sys
import warnings
from collections import deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_scan as B      # noqa: E402
import blowup_us as U        # noqa: E402
import earnings_test as E    # noqa: E402
import momentum_v2 as M      # noqa: E402

ROOT = os.path.dirname(HERE)
INS = os.path.join(ROOT, "data", "insider_buys.csv.gz")
OUT = os.path.join(ROOT, "data", "insider_season_returns.csv")
SLOTS, LEV_COST = 20, 0.04
INS_COST, SEA_COST = 0.0030, 0.0020


# ---------------------------------------------------------------- INSIDER
def clusters(df, need=3, window=30, gap=90):
    df = df.sort_values("filing_date")
    out = []
    for tk, g in df.groupby("ticker", sort=False):
        q = deque()
        last = None
        for d, o, v in zip(g["filing_date"], g["owner_cik"], g["value"]):
            q.append((d, o, v))
            while q and (d - q[0][0]).days > window:
                q.popleft()
            owners = {x[1] for x in q}
            if len(owners) >= need and (last is None
                                         or (d - last).days > gap):
                out.append((tk, d, len(owners), sum(x[2] for x in q)))
                last = d
    return pd.DataFrame(out, columns=["ticker", "filed", "n", "value"])


def drift(ev, px, spy):
    idx = px.index
    pxf = px.ffill()
    s = spy.reindex(idx).ffill()
    rows = []
    for c, f in zip(ev["col"], ev["filed"]):
        i = idx.searchsorted(f) + 1          # next trading day's close
        if i >= len(idx) or not np.isfinite(px[c].iloc[i]):
            continue
        rec = {"filed": f}
        for h in (63, 126):
            if i + h < len(idx):
                rec[h] = (pxf[c].iloc[i + h] / px[c].iloc[i]
                          - s.iloc[i + h] / s.iloc[i])
        rows.append(rec)
    d = pd.DataFrame(rows)
    d = d[(d[63].abs() < 3) | d[63].isna()]
    return d


def report_drift(d, lab):
    for h, name in ((63, "3 months"), (126, "6 months")):
        x = d.dropna(subset=[h])
        m = x.groupby(x["filed"].dt.to_period("M"))[h].mean()
        print(f"    {lab:<13} {name}: {x[h].mean()*100:+.1f}% vs SPY "
              f"(median {x[h].median()*100:+.1f}%), t {B.nw_t(m, 3):+.2f}, "
              f"{len(x)} clusters")


def run_insider(px, spy, members, ev):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    rows, prev = [], set()
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        mem = members(t)
        live = ev[(ev["filed"] <= t) & (ev["filed"] > t - pd.Timedelta(
            days=91)) & ev["col"].isin(mem)]
        live = live.sort_values("value", ascending=False) \
            .drop_duplicates("col")
        pick = list(live["col"][:SLOTS])
        seg = px.loc[t:t2, pick]
        mret = (seg.ffill().iloc[-1] / seg.iloc[0] - 1).fillna(0) \
            if pick else pd.Series(dtype=float)
        spy_m = spy.asof(t2) / spy.asof(t) - 1
        hold = set(pick)
        trades = len(hold - prev) + len(prev - hold)
        ret = (mret.sum() + (SLOTS - len(pick)) * spy_m) / SLOTS \
            - trades / SLOTS * INS_COST
        rows.append({"month": t2, "INSIDER": ret, "SPY": spy_m,
                     "INSIDER_n": len(pick)})
        prev = hold
    return pd.DataFrame(rows).set_index("month")


# ----------------------------------------------------------------- SEASON
def run_season(px, spy, members):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    mpx = px.resample("ME").last()
    mr = mpx.pct_change(fill_method=None)
    rows, prev = [], set()
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        nxt = me[k + 1].month
        past = mr[(mr.index.month == nxt) & (mr.index < me[k])]
        cnt = past.notna().sum()
        score = past.mean()[cnt >= 3]
        mem = members(t)
        score = score[score.index.isin(mem)].dropna()
        if len(score) < 100:
            continue
        pick = list(score.sort_values(ascending=False).index[:SLOTS])
        seg = px.loc[t:t2, pick]
        mret = seg.ffill().iloc[-1] / seg.iloc[0] - 1
        hold = set(pick)
        trades = len(hold - prev) + len(prev - hold)
        rows.append({"month": t2, "SEASON": mret.mean()
                     - trades / SLOTS * SEA_COST,
                     "SPY": spy.asof(t2) / spy.asof(t) - 1,
                     "SEASON_turn": trades / (2 * SLOTS)})
        prev = hold
    return pd.DataFrame(rows).set_index("month")


# ----------------------------------------------------------------- report
def judge(res, col, label):
    print(f"\n  {label}")
    print(f"  {'period':<13} {'months':>6} {'annual':>7} {'vol':>6} "
          f"{'sharpe':>6} {'worst':>7} {'same-risk SPY':>13} "
          f"{'SPY sharpe':>10}")
    ok_all = True
    for lab, part in (("2017-2021", res[res.index < M.RECENT]),
                      ("last 5 years", res[res.index >= M.RECENT])):
        if len(part) < 12:
            print(f"  {lab:<13} too few months")
            ok_all = False
            continue
        a, v, s, dd = B.stats(part[col])
        _, sv, ss, _ = B.stats(part["SPY"])
        lev = v / sv
        m_a = B.stats(lev * part["SPY"] - max(lev - 1, 0) * LEV_COST / 12)[0]
        ok = a > m_a and s > ss
        ok_all &= ok
        print(f"  {lab:<13} {len(part):>6} {a*100:>+6.1f}% {v*100:>5.1f}% "
              f"{s:>6.2f} {dd*100:>6.1f}% {m_a*100:>+12.1f}% {ss:>10.2f}  "
              f"{'ok' if ok else 'miss'}")
    yr = res[[col, "SPY"]]
    print("  by year: " + " | ".join(
        f"{y}: {((1 + g[col]).prod() - 1)*100:+.0f}% vs SPY "
        f"{((1 + g['SPY']).prod() - 1)*100:+.0f}%"
        for y, g in yr.groupby(yr.index.year)))
    return ok_all


def universe(px, vol, feed, min_px, min_dv):
    U.MIN_PX = min_px
    dv = min_dv if feed == "sip" else min_dv * 0.03
    sets, ever = U.liquid_members(px, vol, dv)
    days = sorted(sets)
    return (lambda t: sets[max(d for d in days if d <= t)]), ever


def smoke():
    px, vol, spy = M.smoke()
    rng = np.random.default_rng(13)
    px = px.copy()
    # planted insider clusters followed by +12% drift over 3 months
    rows = []
    for j, c in enumerate(px.columns[::5]):
        for i in range(300 + j * 7, len(px) - 130, 250):
            for o in range(3):
                rows.append((px.index[i - 10 + 4 * o], c, f"o{o}", 1e5))
            ramp = np.ones(len(px))
            ramp[i + 1:i + 64] = np.linspace(1, 1.12, 63)
            ramp[i + 64:] = 1.12
            px[c] = px[c] * ramp
    ins = pd.DataFrame(rows, columns=["filing_date", "ticker", "owner_cik",
                                      "value"])
    # planted seasonality: a few stocks gain +4% every March
    for c in px.columns[1::9]:
        mar = (px.index.month == 3).astype(float)
        px[c] = px[c] * np.exp(np.cumsum(mar * 0.04 / 21))
    return px, vol, spy, ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy, ins = smoke()
        feed = "sip"
    else:
        if not os.path.exists(INS):
            sys.exit("run fetch_insider_buys.py first")
        px, vol, spy, feed = U.load()
        ins = pd.read_csv(INS, dtype={"owner_cik": str, "ticker": str},
                          parse_dates=["filing_date"])
        lo, hi = ins["filing_date"].min(), ins["filing_date"].max()
        print(f"  {len(ins)} insider purchases, {lo.date()} -> {hi.date()}")
    spy = spy.reindex(px.index).ffill()

    # INSIDER
    ins = ins.dropna(subset=["ticker", "owner_cik", "filing_date"])
    ins["ticker"] = ins["ticker"].astype(str).str.upper() \
        .str.replace("-", ".", regex=False)
    cl = clusters(ins[ins["filing_date"] >= "2015-01-01"])
    mem_i, ever_i = universe(px, vol, feed, 5.0, 5e6)
    ev = E.map_to_columns(cl, px[ever_i])
    print(f"\n  INSIDER: {len(cl)} clusters (3+ insiders in 30 days) since "
          f"2015, {len(ev)} matched to a liquid-ever stock")
    print("  raw drift after the cluster became public (minus SPY):")
    d = drift(ev, px, spy)
    report_drift(d[d["filed"] < M.RECENT], "2017-2021")
    report_drift(d[d["filed"] >= M.RECENT], "last 5 years")
    ri = run_insider(px[ever_i], spy, mem_i, ev)
    print(f"  avg stocks held: {ri['INSIDER_n'].mean():.1f} of {SLOTS} "
          f"(rest in SPY)")
    ok_i = judge(ri, "INSIDER", "INSIDER portfolio")

    # SEASON
    mem_s, ever_s = universe(px, vol, feed, 10.0, 20e6)
    rs = run_season(px[ever_s], spy, mem_s)
    print(f"\n  SEASON: {len(rs)} months scored, first {rs.index[0].date()};"
          f" avg turnover {rs['SEASON_turn'].mean()*100:.0f}% of the book "
          f"per month")
    ok_s = judge(rs, "SEASON", "SEASON portfolio")

    if not a.smoke:
        ri.join(rs[["SEASON"]], how="outer").to_csv(OUT)
    print("\n  ===== VERDICT (pre-stated: both periods must beat same-risk "
          "SPY and SPY's Sharpe) =====")
    if a.smoke:
        print("  smoke: planted clusters + March seasonality (pipe check)")
    print(f"  INSIDER: {'PASS' if ok_i else 'FAIL'}")
    print(f"  SEASON:  {'PASS' if ok_s else 'FAIL'}")


if __name__ == "__main__":
    main()
