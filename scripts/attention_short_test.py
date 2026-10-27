"""
attention_short_test.py -- two NON-PRICE signals on S&P 500 stocks
(point-in-time members incl. delisted, Alpaca prices).
Needs data/attention/views.parquet and data/short/ from
fetch_attention_short.py.

ATTENTION (Wikipedia page views)
  spike ratio on day d = average views d-6..d / median views d-97..d-7.
  EVENT: ratio >= 3 (first in 30 days per stock), stock an S&P member.
  Views for day d are only used from the next trading day's close.
  Measured: return 5 / 21 / 63 trading days later minus SPY.
  Portfolios (context): ATTN_LONG = up to 20 stocks with a spike in the
  last month (biggest first), SPY fill; ATTN_AVOID = equal-weight S&P
  minus those stocks. Monthly, 10 bps per trade.
  PASS (pre-stated): the 21-day effect has the SAME SIGN in 2017-02 ->
  2021-05 and 2021-06 -> 2026-05 with |t| > 2.5 in each (t from monthly
  averages, 3-lag Newey-West).

SHORT INTEREST (FINRA; fallback: daily short-sale volume)
  days to cover = shares short / average daily volume, used from 12 days
  after the settlement date (publication lag). Fallback measure: 20-day
  shorted share of volume, used from the next day.
  Each month: S&P members sorted into fifths by shorting.
  SPREAD = least-shorted fifth minus most-shorted fifth, equal weight,
  held one month.
  PASS (pre-stated): SPREAD mean > 0 with t > 2.5 over all available
  months AND positive in both halves of the available history.
  Context: each fifth vs SPY; AVOID_SHORTED = equal-weight S&P minus the
  most-shorted fifth, vs equal-weight S&P.

Run (repo root):
    python3.10 scripts/attention_short_test.py --smoke
    python3.10 scripts/attention_short_test.py | tee logs/attention_short.txt
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
import blowup_scan as B        # noqa: E402
import factor_portfolio as FP  # noqa: E402

ROOT = os.path.dirname(HERE)
ATT = os.path.join(ROOT, "data", "attention", "views.parquet")
SI = os.path.join(ROOT, "data", "short", "short_interest.parquet")
SV = os.path.join(ROOT, "data", "short", "short_volume.parquet")
FIRST, RECENT = pd.Timestamp("2017-02-28"), pd.Timestamp("2021-06-01")
COST = 0.0010


def norm_index(df):
    ix = pd.to_datetime(df.index)
    if ix.tz is not None:
        ix = ix.tz_convert(None)
    df.index = ix.normalize()
    return df


def months(px):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= FIRST]
    for k in range(len(me) - 1):
        yield idx[idx <= me[k]][-1], idx[idx <= me[k + 1]][-1]


def month_ret(px, t, t2, names):
    if not names:
        return pd.Series(dtype=float)
    seg = px.loc[t:t2, names]
    return (seg.ffill().iloc[-1] / seg.iloc[0] - 1).fillna(0)


# --------------------------------------------------------------- attention
def attention_events(views, px, members):
    v = views.reindex(columns=[c for c in views.columns if c in px.columns])
    v = v.asfreq("D")
    recent = v.rolling(7, min_periods=6).mean()
    base = v.shift(7).rolling(91, min_periods=60).median()
    ratio = recent / base.replace(0, np.nan)
    idx = px.index
    rows = []
    for c in ratio.columns:
        r = ratio[c].dropna()
        last = None
        for d, x in r[r >= 3].items():
            if last is not None and (d - last).days < 30:
                continue
            pos = idx.searchsorted(d + pd.Timedelta(days=1))
            if pos >= len(idx):
                continue
            e = idx[pos]                      # first close after day d
            if c not in members(e) or not np.isfinite(px.at[e, c]):
                continue
            last = d
            rows.append((c, e, x))
    return pd.DataFrame(rows, columns=["sym", "e", "ratio"])


def drift(ev, px, spy):
    idx = px.index
    pxf = px.ffill()
    s = spy.reindex(idx).ffill()
    out = []
    for c, e in zip(ev["sym"], ev["e"]):
        i = idx.get_loc(e)
        rec = {"e": e}
        for h in (5, 21, 63):
            if i + h < len(idx):
                rec[h] = pxf[c].iloc[i + h] / px[c].iloc[i] - \
                    s.iloc[i + h] / s.iloc[i]
        out.append(rec)
    d = pd.DataFrame(out)
    return d[(d[21].abs() < 2) | d[21].isna()]


def drift_report(d, lab):
    res = {}
    for h in (5, 21, 63):
        x = d.dropna(subset=[h])
        m = x.groupby(x["e"].dt.to_period("M"))[h].mean()
        t = B.nw_t(m, 3)
        res[h] = (x[h].mean(), t)
        print(f"    {lab:<13} +{h:>2}d: {x[h].mean()*100:+.2f}% vs SPY (median "
              f"{x[h].median()*100:+.2f}%), t {t:+.2f}, {len(x)} spikes")
    return res


def attention_books(ev, px, spy, members):
    rows, prev = [], {"ATTN_LONG": set(), "ATTN_AVOID": set(),
                      "BASE_EW": set()}
    for t, t2 in months(px):
        mem = [s for s in px.columns if s in members(t)
               and np.isfinite(px.at[t, s])]
        win = ev[(ev["e"] <= t) & (ev["e"] > t - pd.Timedelta(days=31))]
        spk = list(win.sort_values("ratio", ascending=False)["sym"]
                   .drop_duplicates())
        spk = [s for s in spk if s in mem]
        mret = month_ret(px, t, t2, mem)
        spy_m = spy.asof(t2) / spy.asof(t) - 1
        row = {"month": t2, "SPY": spy_m, "n_spikes": len(spk)}
        books = {"ATTN_LONG": spk[:20],
                 "ATTN_AVOID": [s for s in mem if s not in set(spk)],
                 "BASE_EW": mem}
        for k, names in books.items():
            hold = set(names)
            n = 20 if k == "ATTN_LONG" else max(len(names), 1)
            r = mret[names].sum() / n
            if k == "ATTN_LONG":
                r += (20 - len(names)) / 20 * spy_m
            turn = len(hold ^ prev[k]) / n
            row[k] = r - turn * COST
            prev[k] = hold
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


# ----------------------------------------------------------- short interest
def short_measure():
    if os.path.exists(SI):
        si = pd.read_parquet(SI)
        si = si[si["adv"] > 0]
        si["val"] = si["short"] / si["adv"]
        si["known"] = si["date"] + pd.Timedelta(days=12)
        yrs = (si["date"].max() - si["date"].min()).days / 365
        if yrs >= 6 or not os.path.exists(SV):
            return si[["ticker", "known", "val"]], "days to cover (FINRA)"
    sv = pd.read_parquet(SV).sort_values("date")
    sv["s20"] = sv.groupby("ticker")["short"].transform(
        lambda x: x.rolling(20, min_periods=15).sum())
    sv["t20"] = sv.groupby("ticker")["total"].transform(
        lambda x: x.rolling(20, min_periods=15).sum())
    sv["val"] = sv["s20"] / sv["t20"]
    sv["known"] = sv["date"] + pd.Timedelta(days=1)
    return sv[["ticker", "known", "val"]].dropna(), \
        "shorted share of volume, 20d (FINRA daily)"


def short_books(px, spy, members, sm):
    sm = sm.sort_values("known")
    rows, prev1, prev5 = [], set(), set()
    for t, t2 in months(px):
        mem = [s for s in px.columns if s in members(t)
               and np.isfinite(px.at[t, s])]
        cur = sm[(sm["known"] <= t) & (sm["known"] > t - pd.Timedelta(
            days=45))].groupby("ticker")["val"].last()
        cur = cur[cur.index.isin(mem)].dropna()
        if len(cur) < 100:
            continue
        q = pd.qcut(cur.rank(method="first"), 5, labels=False)
        mret = month_ret(px, t, t2, mem)
        row = {"month": t2, "SPY": spy.asof(t2) / spy.asof(t) - 1,
               "n": len(cur)}
        for k in range(5):
            row[f"Q{k+1}"] = mret[q.index[q == k]].mean()
        s1, s5 = set(q.index[q == 0]), set(q.index[q == 4])
        turn = len(s1 ^ prev1) / max(len(s1), 1) + \
            len(s5 ^ prev5) / max(len(s5), 1)
        prev1, prev5 = s1, s5
        row["SPREAD"] = row["Q1"] - row["Q5"] - turn * COST
        row["turn"] = turn
        top = s5
        row["AVOID_SHORTED"] = mret[[s for s in mem if s not in top]].mean()
        row["BASE_EW"] = mret[mem].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


# --------------------------------------------------------------- reports
def table(res, cols, lab):
    print(f"\n  {lab}")
    print(f"  {'portfolio':<14} | {'2017-21 ann':>11} {'sh':>5} | "
          f"{'last5y ann':>10} {'sh':>5} {'worst':>6}")
    for k in cols:
        o = res.loc[res.index < RECENT, k].dropna()
        n = res.loc[res.index >= RECENT, k].dropna()
        a0, _, s0, _ = B.stats(o) if len(o) > 6 else (np.nan,) * 4
        a1, _, s1, dd = B.stats(n) if len(n) > 6 else (np.nan,) * 4
        print(f"  {k:<14} | {a0*100:>+10.1f}% {s0:>5.2f} | {a1*100:>+9.1f}% "
              f"{s1:>5.2f} {dd*100:>5.0f}%")


def smoke():
    import turbo_portfolio as T
    px, vol, spy, members = T.smoke()
    rng = np.random.default_rng(23)
    px = px.copy()
    days = pd.date_range(px.index[0], px.index[-1], freq="D")
    views = pd.DataFrame(rng.lognormal(7, 0.2, (len(days), px.shape[1])),
                         days, px.columns)
    for j, c in enumerate(px.columns[::3]):     # spikes, then +5% drift
        for i in range(400 + j * 11, len(days) - 120, 180):
            views.iloc[i:i + 5, views.columns.get_loc(c)] *= 8
            d = days[i + 5]
            k = px.index.searchsorted(d)
            ramp = np.ones(len(px))
            ramp[k + 1:k + 22] = np.linspace(1, 1.05, 21)
            ramp[k + 22:] = 1.05
            px[c] = px[c] * ramp
    rows = []                                   # shorted names drift down
    heavy = set(px.columns[1::4])
    for c in px.columns:
        base = 8.0 if c in heavy else 2.0
        for d in pd.date_range(px.index[0], px.index[-1], freq="SMS"):
            rows.append((c, d + pd.Timedelta(days=12),
                         base * rng.lognormal(0, 0.2)))
        if c in heavy:
            px[c] = px[c] * np.exp(-np.arange(len(px)) * 0.0004)
    sm = pd.DataFrame(rows, columns=["ticker", "known", "val"])
    return px, spy, members, views, sm, "smoke days to cover"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, spy, members, views, sm, slab = smoke()
    else:
        if not os.environ.get("ALPACA_API_KEY"):
            sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
        members, union = FP.membership()
        px, spy = FP.load_prices(union, members)
        px, spy = norm_index(px), norm_index(spy.to_frame()).iloc[:, 0]
        views = norm_index(pd.read_parquet(ATT)) if os.path.exists(ATT) \
            else None
        try:
            sm, slab = short_measure()
        except FileNotFoundError:
            sm, slab = None, None
    spy = spy.reindex(px.index).ffill()
    verdict = {}

    if views is not None:
        print("\n  ===== ATTENTION (Wikipedia page-view spikes) =====")
        ev = attention_events(views, px, members)
        print(f"  {len(ev)} spikes in {ev['sym'].nunique()} stocks")
        d = drift(ev, px, spy)
        r0 = drift_report(d[d["e"] < RECENT], "2017-2021")
        r1 = drift_report(d[d["e"] >= RECENT], "last 5 years")
        ab = attention_books(ev, px, spy, members)
        table(ab, ["ATTN_LONG", "ATTN_AVOID", "BASE_EW", "SPY"],
              f"portfolios (avg {ab['n_spikes'].mean():.0f} spike stocks "
              f"per month)")
        (m0, t0), (m1, t1) = r0[21], r1[21]
        verdict["ATTENTION"] = (np.sign(m0) == np.sign(m1) and abs(t0) > 2.5
                                and abs(t1) > 2.5)
    if sm is not None:
        print(f"\n  ===== SHORT INTEREST: {slab} =====")
        sb = short_books(px, spy, members, sm)
        print(f"  {len(sb)} months, {sb.index[0]:%Y-%m} -> {sb.index[-1]:%Y-%m}"
              f", ~{sb['n'].mean():.0f} stocks per month")
        print("  avg monthly return by fifth (Q1 least shorted ... Q5 most): "
              + "  ".join(f"Q{k} {sb[f'Q{k}'].mean()*100:+.2f}%"
                          for k in range(1, 6))
              + f"  | SPY {sb['SPY'].mean()*100:+.2f}%")
        x = sb["SPREAD"].dropna()
        t = x.mean() / (x.std() / np.sqrt(len(x)))
        h = len(x) // 2
        h1, h2 = x.iloc[:h].mean(), x.iloc[h:].mean()
        print(f"  monthly turnover of the two legs: {sb['turn'].mean()*50:.0f}%"
              " of positions")
        print(f"  SPREAD (least minus most shorted): {x.mean()*1200:+.1f}%/yr"
              f" t {t:+.2f} | first half {h1*1200:+.1f}%/yr, second half "
              f"{h2*1200:+.1f}%/yr")
        table(sb, ["SPREAD", "AVOID_SHORTED", "BASE_EW", "SPY"],
              "portfolios")
        verdict["SHORT"] = x.mean() > 0 and t > 2.5 and h1 > 0 and h2 > 0

    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: planted spike drift + shorted-stock decline")
    for k, v in verdict.items():
        print(f"  {k:<10} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
