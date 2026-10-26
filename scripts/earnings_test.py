"""
earnings_test.py -- earnings surprise: stocks that just reported profits
far above their usual pattern tend to keep drifting up for weeks
("post-earnings drift"). The first NON-PRICE signal tested: it runs on
news about the business, not on the chart.

Needs data/earnings_quarterly.csv.gz (fetch_earnings.py) and
data/us_universe/ (build_us_universe.py).

SURPRISE SCORE (per company-quarter, known only from its SEC filing day)
  change = net income this quarter - same quarter last year
  surprise = change / std-dev of that change over the previous 8 quarters
  (min 4) -- "how unusual is this quarter for THIS company"
  Event date = the 10-Q/10-K filing day. Companies announce earnings by
  press release 2-4 weeks earlier, so this is a LATE, conservative
  timestamp: any edge found is what's left after the first reaction.

PORTFOLIOS (all liquid US stocks >= $10 and >= $20M/day point-in-time,
top 20 equal weight, monthly, 20 bps/trade; a stock counts only if its
latest filing came out within the last 90 days)
  SUE       top 20 by surprise
  SUE_MOM   average of surprise rank and steady-momentum rank -- PRIMARY
            (business beating AND stock already climbing)
  MOM       steady momentum alone (momentum_v2 STEADY) -- context
Also printed: the drift itself -- 3-month return after filing, top 10%
of surprises minus bottom 10%, per filing month (t, 6 lags).

PRE-STATED (primary SUE_MOM, last five years 2021-06 -> 2026-05)
  PASS: return beats SPY leveraged to the same volatility (4%/yr borrow)
  AND Sharpe above SPY's. 2017-2021 shown as context.

Run (repo root):
    python3.10 scripts/earnings_test.py --smoke
    python3.10 scripts/earnings_test.py | tee logs/earnings_test.txt
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
EARN = os.path.join(ROOT, "data", "earnings_quarterly.csv.gz")
OUT = os.path.join(ROOT, "data", "earnings_test_returns.csv")
VARIANTS = ["SUE", "SUE_MOM", "MOM"]
PRIMARY = "SUE_MOM"
FRESH = pd.Timedelta(days=90)


def surprises(e):
    """Seasonal net-income surprise per company-quarter."""
    e = e.dropna(subset=["ni", "end", "filed"]).sort_values(["cik", "end"])
    e = e.drop_duplicates(["cik", "end"], keep="first")
    out = []
    for cik, g in e.groupby("cik", sort=False):
        g = g.set_index("end")
        ni = g["ni"]
        prev = ni.reindex(ni.index - pd.DateOffset(years=1),
                          method="nearest", tolerance=pd.Timedelta(days=20))
        d = pd.Series(ni.values - prev.values, ni.index)
        sd = d.shift(1).rolling(8, min_periods=4).std()
        s = d / sd.replace(0, np.nan)
        out.append(pd.DataFrame({"cik": cik, "ticker": g["ticker"].values,
                                 "end": ni.index, "filed": g["filed"].values,
                                 "sue": s.values}))
    df = pd.concat(out, ignore_index=True)
    df["sue"] = df["sue"].clip(-10, 10)
    return df.dropna(subset=["sue"])


def map_to_columns(ev, px):
    """Ticker + filing date -> the price column (recycled tickers are
    split into SYM, SYM~1, ... by build_us_universe)."""
    first = px.apply(pd.Series.first_valid_index)
    last = px.apply(pd.Series.last_valid_index)
    by_base = {}
    for c in px.columns:
        by_base.setdefault(c.split("~")[0], []).append(c)
    cols = []
    for tk, f in zip(ev["ticker"], ev["filed"]):
        tk = str(tk).upper().replace("-", ".")
        hit = None
        for c in by_base.get(tk, []):
            if first[c] <= f <= last[c]:
                hit = c
                break
        cols.append(hit)
    ev = ev.assign(col=cols).dropna(subset=["col"])
    return ev


def drift_study(ev, px, start, end):
    """3-month return after filing: top-decile minus bottom-decile
    surprise, per filing month."""
    pxf = px.ffill()
    idx = px.index
    ev = ev[(ev["filed"] >= start) & (ev["filed"] < end)]
    rets = []
    for c, f in zip(ev["col"], ev["filed"]):
        i = idx.searchsorted(f) + 1
        if i + 63 >= len(idx):
            rets.append(np.nan)
            continue
        p0, p1 = pxf[c].iloc[i], pxf[c].iloc[i + 63]
        rets.append(p1 / p0 - 1 if p0 and np.isfinite(p0) else np.nan)
    ev = ev.assign(r=rets).dropna(subset=["r"])
    ev = ev[ev["r"].abs() < 3]
    ev["m"] = ev["filed"].dt.to_period("M")
    spread = []
    for m, g in ev.groupby("m"):
        if len(g) < 50:
            continue
        hi = g["sue"].quantile(0.9)
        lo = g["sue"].quantile(0.1)
        spread.append(g.loc[g["sue"] >= hi, "r"].mean()
                      - g.loc[g["sue"] <= lo, "r"].mean())
    sp = pd.Series(spread)
    return sp.mean(), B.nw_t(sp), len(ev)


def run(px, spy, members, ev):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    ev = ev.sort_values("filed")
    rows = []
    prev = {v: set() for v in VARIANTS}
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
        dvol = r[elig].iloc[t_i - 251:t_i + 1].std() * np.sqrt(252)
        steady = ((W.iloc[-22] / W.iloc[0] - 1) / dvol.replace(0, np.nan)) \
            .replace([np.inf, -np.inf], np.nan).dropna()
        recent = ev[(ev["filed"] <= t) & (ev["filed"] > t - FRESH)]
        sue = recent.groupby("col")["sue"].last()
        sue = sue[sue.index.isin(elig)]
        both = sue.index.intersection(steady.index)
        combo = (sue[both].rank(pct=True) + steady[both].rank(pct=True)) / 2
        scores = {"SUE": sue, "SUE_MOM": combo, "MOM": steady}
        seg = px.loc[t:t2, elig]
        mret = seg.ffill().iloc[-1] / seg.iloc[0] - 1
        row = {"month": t2, "SPY": spy.asof(t2) / spy.asof(t) - 1,
               "n_sue": len(sue)}
        for v in VARIANTS:
            sc = scores[v]
            pick = list(sc.sort_values(ascending=False).index[:M.TOP])
            if len(pick) < M.TOP:
                row[v] = np.nan
                continue
            old = prev[v]
            trades = len(set(pick) - old) + len(old - set(pick))
            row[v] = mret[pick].mean() - trades / M.TOP * M.COST
            prev[v] = set(pick)
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def table(res, label):
    print(f"\n  ===== {label}: {len(res)} months, {res.index[0].date()} -> "
          f"{res.index[-1].date()} | ~{res['n_sue'].mean():.0f} stocks with a"
          f" fresh surprise/month =====")
    print(f"  {'portfolio':<9} {'annual':>7} {'vol':>6} {'sharpe':>6} "
          f"{'worst':>7} {'same-risk SPY':>13} {'vs SPY/yr':>9} {'t':>6}")
    _, spy_v, spy_s, _ = B.stats(res["SPY"])
    out = {}
    for v in VARIANTS:
        x = res[v].dropna()
        a, vol, s, dd = B.stats(x)
        lev = vol / spy_v
        sp = res.loc[x.index, "SPY"]
        m_a = B.stats(lev * sp - max(lev - 1, 0) * M.LEV_COST / 12)[0]
        ex = x - sp
        t = ex.mean() / (ex.std() / np.sqrt(len(ex)))
        print(f"  {v:<9} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
              f"{dd*100:>6.1f}% {m_a*100:>+12.1f}% {ex.mean()*1200:>+8.1f}% "
              f"{t:>+6.2f}")
        out[v] = (a, s, m_a)
    a, vol, s, dd = B.stats(res["SPY"])
    print(f"  {'SPY':<9} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
          f"{dd*100:>6.1f}%")
    yr = res[[PRIMARY, "SPY"]]
    print("  by year: " + " | ".join(
        f"{y}: {PRIMARY} {((1 + g[PRIMARY]).prod() - 1)*100:+.0f}% SPY "
        f"{((1 + g['SPY']).prod() - 1)*100:+.0f}%"
        for y, g in yr.groupby(yr.index.year)))
    return out, spy_s


def smoke():
    px, vol, spy = M.smoke()
    rng = np.random.default_rng(21)
    # unit check of the surprise math: flat profits, then a jump
    ends = pd.date_range("2018-03-31", periods=16, freq="QE")
    ni = np.r_[100 + rng.normal(0, 3, 12), 160, 100, 100, 100]
    s = surprises(pd.DataFrame({"cik": "1", "ticker": "X", "end": ends,
                                "filed": ends + pd.Timedelta(days=35),
                                "ni": ni}))
    z = s.set_index("end")["sue"]
    assert z.loc[ends[12]] > 3, z
    print(f"  surprise unit check: jump quarter scores {z.loc[ends[12]]:+.1f}"
          " OK")
    # planted: random surprise each quarter; price drifts with it for 2 mo
    rows = []
    px = px.copy()
    for c in px.columns:
        for i in range(300, len(px) - 50, 63):
            sue = rng.normal()
            f = px.index[i]
            rows.append((c, c, f, sue))
            ramp = np.ones(len(px))
            ramp[i + 1:i + 43] = np.exp(np.linspace(0, 0.03 * sue, 42))
            ramp[i + 43:] = np.exp(0.03 * sue)
            px[c] = px[c] * ramp
    ev = pd.DataFrame(rows, columns=["col", "ticker", "filed", "sue"])
    return px, vol, spy, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy, ev = smoke()
        feed = "sip"
    else:
        if not os.path.exists(EARN):
            sys.exit("run fetch_earnings.py first")
        px, vol, spy, feed = U.load()
        e = pd.read_csv(EARN, dtype={"cik": str, "ticker": str},
                        parse_dates=["end", "filed"])
        e = e.dropna(subset=["ticker"])
        print(f"  {len(e)} company-quarters with a ticker")
        ev = surprises(e)
        ev = map_to_columns(ev, px)
        print(f"  {len(ev)} surprises matched to a price history "
              f"({ev['col'].nunique()} stocks)")
    U.MIN_PX = M.MIN_PX
    min_dv = M.MIN_DV if feed == "sip" else M.MIN_DV * 0.03
    sets, ever = U.liquid_members(px, vol, min_dv)
    px = px[ever]
    ev = ev[ev["col"].isin(ever)]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731

    print("\n  THE DRIFT ITSELF (3-month return after filing, top 10% "
          "surprise minus bottom 10%):")
    for lab, s0, s1 in (("2017-2021", M.FIRST, M.RECENT),
                        ("last 5 years", M.RECENT,
                         pd.Timestamp("2026-12-31"))):
        m, t, n = drift_study(ev, px, s0, s1)
        print(f"    {lab:<13} {m*100:+.2f}% per 3 months  t {t:+.2f}  "
              f"({n} filings)")

    res = run(px, spy, members, ev)
    if not a.smoke:
        res.to_csv(OUT)
    old = res.index < M.RECENT
    if old.sum() > 6:
        table(res[old], "CONTEXT 2017-2021")
    rec, spy_s = table(res[~old], "DECIDING: LAST FIVE YEARS")
    a_, s_, m_a = rec[PRIMARY]
    ok = a_ > m_a and s_ > spy_s
    print(f"\n  ===== VERDICT (pre-stated, {PRIMARY}, last 5 years) =====")
    print(f"  return {a_*100:+.1f}% vs same-risk SPY {m_a*100:+.1f}% | "
          f"Sharpe {s_:.2f} vs SPY {spy_s:.2f} -> "
          f"{'PASS -- forward paper trading next' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
