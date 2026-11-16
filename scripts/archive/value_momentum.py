"""
value_momentum.py -- does a VALUE book (cheap stocks) cancel momentum's bad
years? Research ("Value and Momentum Everywhere", Asness, Moskowitz &
Pedersen 2013): value and momentum tend to move in opposite directions.
Momentum's worst stretches (Feb 2021, 2022) were rotations from hot growth
into cheap stocks -- exactly when value should win.

BOOKS (monthly, US common stocks, price >= $10, >= $20M/day, top 20 equal
weight, 20 bps per trade -- same universe and mechanics as CHIMERA)
  STEADY  12-1 return / 1y volatility (CHIMERA)
  VALUE   average percentile rank of
            E/P  last 10-K net income / market cap
            B/P  last 10-K shareholders' equity / market cap (if > 0)
          10-K must be filed BEFORE the month-end and cover a fiscal year
          ended within 18 months; market cap = RAW (unadjusted) month-end
          close x shares outstanding from the latest SEC cover filed within
          15 months before. All point in time.
  BLENDS  rebalanced to fixed weights each month
          VM50 = 50% STEADY + 50% VALUE      (PRIMARY)
          VM70 = 70% STEADY + 30% VALUE      (secondary)
  SIZE BASELINES (same weights, VALUE replaced by T-bills)
          T50 = 50% STEADY + 50% T-bills, T70 = 70% STEADY + 30% T-bills
  Every book is shown plain and with the 30% yearly lock (account up 30%
  this year -> SPY until Jan 1). Sharpe here is measured ABOVE the T-bill
  rate (like QuantConnect), so the T-bill blends get no free credit.

PRE-STATED (fixed before running)
  VM50+LOCK30 passes if it beats STEADY+LOCK30 on BOTH Sharpe and worst
  drop (2017-03 -> 2026-05) AND has a higher Sharpe than T50+LOCK30 (value
  must earn its place, not just shrink the bet). If VM50 fails, VM70+LOCK30
  gets the same test against STEADY+LOCK30 and T70+LOCK30. Pass -> port the
  value book to QC (QuantConnect has point-in-time P/E and P/B built in).
  Caveat: Primordial's books look rosier in bad years than QC's; the
  comparison between books is what counts here.

DATA (all on your Mac already, except raw prices)
  data/sec_fsds/*.parsed.csv          (fetch_sec_fundamentals.py)
  data/cik_tickers.csv.gz             (ticker <-> SEC id by year)
  data/shares_outstanding.csv.gz      (fetch_shares.py)
  data/us_universe/raw_monthly.parquet  downloaded once by THIS script from
                                      Alpaca (monthly bars, no split
                                      adjustment) -> export Alpaca keys

Run (repo root):
    python3.10 scripts/value_momentum.py --smoke
    python3.10 scripts/value_momentum.py | tee logs/value_momentum.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_us as U              # noqa: E402
import crash_warning_search as CW  # noqa: E402
import momentum_v2 as M            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "us_universe", "raw_monthly.parquet")
START, END, RECENT = pd.Timestamp("2017-03-01"), pd.Timestamp("2026-06-01"), \
    pd.Timestamp("2021-06-01")


# ------------------------------------------------------------- inputs
def raw_monthly(symbols):
    """Unadjusted month-end closes from Alpaca (cached)."""
    if os.path.exists(RAW):
        return pd.read_parquet(RAW)
    import build_us_universe as B   # noqa: E402
    h = B.hdr()
    out = []
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 100):
        chunk, token = syms[i:i + 100], None
        while True:
            prm = {"symbols": ",".join(chunk), "timeframe": "1Month",
                   "start": "2015-01-01", "end": B.END, "limit": 10000,
                   "adjustment": "raw", "feed": "sip"}
            if token:
                prm["page_token"] = token
            js = B.get(B.DATA, prm, h)
            for s, bars in (js.get("bars") or {}).items():
                out += [(s, b["t"][:7], b["c"]) for b in bars]
            token = js.get("next_page_token")
            if not token:
                break
            time.sleep(0.25)
        print(f"    raw prices {min(i + 100, len(syms))}/{len(syms)}", flush=True)
    df = pd.DataFrame(out, columns=["symbol", "month", "close"])
    wide = df.pivot_table("close", "month", "symbol", aggfunc="last")
    wide.to_parquet(RAW)
    return wide


def load_fund():
    files = glob.glob(os.path.join(DATA, "sec_fsds", "*.parsed.csv"))
    if not files:
        sys.exit("no data/sec_fsds/*.parsed.csv -- run fetch_sec_fundamentals.py")
    f = pd.concat([pd.read_csv(p, dtype={"cik": str}) for p in files])
    f = f.drop_duplicates("adsh")
    f["filed"] = pd.to_datetime(f["filed"].astype(str), format="%Y%m%d", errors="coerce")
    f["period"] = pd.to_datetime(f["period"].astype(str), format="%Y%m%d", errors="coerce")
    f["cik"] = f["cik"].str.lstrip("0")
    return f.dropna(subset=["filed", "period"])[["cik", "filed", "period", "ni", "equity"]]


def load_shares():
    p = os.path.join(DATA, "shares_outstanding.csv.gz")
    if not os.path.exists(p):
        sys.exit("no data/shares_outstanding.csv.gz -- run fetch_shares.py")
    s = pd.read_csv(p, dtype={"cik": str}, parse_dates=["filed"])
    s["cik"] = s["cik"].str.lstrip("0")
    return s[s["shares"] > 0][["cik", "filed", "shares"]]


def load_cik_map():
    t = pd.read_csv(os.path.join(DATA, "cik_tickers.csv.gz"),
                    dtype={"cik": str, "ticker": str})
    t["cik"] = t["cik"].str.lstrip("0")
    t["ticker"] = t["ticker"].str.upper().str.replace("-", ".", regex=False)
    return {(r.ticker, int(r.year)): r.cik for r in t.itertuples()}


def value_scores(t, elig, fund, shares, cikmap, raw):
    month = t.strftime("%Y-%m")
    if month not in raw.index:
        return pd.Series(dtype=float), 0.0
    f = fund[(fund["filed"] < t) & (fund["period"] >= t - pd.DateOffset(months=18))]
    f = f.sort_values("filed").groupby("cik").tail(1).set_index("cik")
    sh = shares[(shares["filed"] < t) & (shares["filed"] >= t - pd.DateOffset(months=15))]
    sh = sh.sort_values("filed").groupby("cik").tail(1).set_index("cik")["shares"]
    rawm = raw.loc[month]
    ep, bp = {}, {}
    for s in elig:
        base = s.split("~")[0]
        cik = cikmap.get((base, t.year)) or cikmap.get((base, t.year - 1))
        if cik is None or cik not in f.index or cik not in sh.index:
            continue
        px = rawm.get(base, np.nan)
        if not np.isfinite(px) or px <= 0:
            continue
        mcap = px * sh[cik]
        ni, eq = f.at[cik, "ni"], f.at[cik, "equity"]
        if np.isfinite(ni):
            ep[s] = ni / mcap
        if np.isfinite(eq) and eq > 0:
            bp[s] = eq / mcap
    E, Bk = pd.Series(ep), pd.Series(bp)
    E = E[(E > -1) & (E < 1)]                 # drop share-count/split glitches
    Bk = Bk[Bk < 10]
    sc = pd.concat([E.rank(pct=True), Bk.rank(pct=True)], axis=1).mean(axis=1)
    sc = sc[sc.index.isin(E.index)]
    return sc, len(sc) / max(len(elig), 1)


# -------------------------------------------------------------- books
def build(px, spy, members, valfn):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    pxf = px.ffill()
    daily = {"STEADY": [], "VALUE": []}
    prev = {"STEADY": set(), "VALUE": set()}
    cov = []
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
        W = px[elig].iloc[t_i - 252:t_i + 1]
        steady = (W.iloc[-22] / W.iloc[0] - 1) / (
            r[elig].iloc[t_i - 251:t_i + 1].std() * np.sqrt(252)).replace(0, np.nan)
        val, c = valfn(t, elig)
        cov.append(c)
        if len(val) < 40:
            continue
        for n, sc in (("STEADY", steady), ("VALUE", val)):
            sc = sc.replace([np.inf, -np.inf], np.nan).dropna()
            pick = list(sc.sort_values(ascending=False).index[:M.TOP])
            seg = pxf.loc[t:t2, pick]
            d = (seg / seg.iloc[0]).mean(axis=1).pct_change().iloc[1:]
            if len(d):
                d.iloc[0] -= len(set(pick) ^ prev[n]) / M.TOP * M.COST
            prev[n] = set(pick)
            daily[n].append(d)
    res = pd.DataFrame({n: pd.concat(v) for n, v in daily.items()}).dropna()
    res["SPY"] = spy.pct_change().reindex(res.index).fillna(0)
    return res, float(np.mean(cov)) if cov else 0.0


def blend(a, b, w):
    """w in a, 1-w in b, back to target weights at each month start."""
    out = []
    for _, g in pd.concat([a, b], axis=1, keys=["a", "b"]).groupby(a.index.to_period("M")):
        path = w * (1 + g["a"]).cumprod() + (1 - w) * (1 + g["b"]).cumprod()
        out.append(path.pct_change().fillna(path.iloc[0] - 1))
    return pd.concat(out)


def stats(r, rf):
    """annual return, Sharpe ABOVE T-bills (fair to the T-bill blends), worst."""
    rf = rf.reindex(r.index).fillna(0).values
    r = np.asarray(r)
    eq = np.cumprod(1 + r)
    return (np.prod(1 + r) ** (252 / len(r)) - 1, (r - rf).mean() / r.std() * np.sqrt(252),
            (eq / np.maximum.accumulate(eq) - 1).min())


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

    if a.smoke:
        rng = np.random.default_rng(4)
        fwd = px.pct_change(21, fill_method=None).shift(-21)

        def valfn(t, elig):     # planted: value score = noisy NEGATIVE of momentum
            past = px[elig].pct_change(252, fill_method=None).loc[t]
            return (-past + rng.normal(0, 0.3, len(elig))).dropna(), 1.0
        rf = None
    else:
        print("  loading SEC fundamentals, share counts, raw prices ...")
        fund, shares, cikmap = load_fund(), load_shares(), load_cik_map()
        raw = raw_monthly([s.split("~")[0] for s in px.columns])

        def valfn(t, elig):
            return value_scores(t, elig, fund, shares, cikmap, raw)
        rf = None
    res, cov = build(px, spy, members, valfn)
    res = res[(res.index >= START) & (res.index < END)]
    rf = (pd.Series(0.0001, res.index) if a.smoke else
          CW.load_french()["RF"].reindex(res.index).ffill().fillna(0))
    print(f"  {res.index[0].date()} -> {res.index[-1].date()} | value data for "
          f"{cov*100:.0f}% of eligible stocks{'  [SYNTHETIC]' if a.smoke else ''}")

    books = {"STEADY": res["STEADY"], "VALUE": res["VALUE"],
             "VM50": blend(res["STEADY"], res["VALUE"], 0.5),
             "VM70": blend(res["STEADY"], res["VALUE"], 0.7),
             "T50": blend(res["STEADY"], rf, 0.5),
             "T70": blend(res["STEADY"], rf, 0.7)}
    locked = {k: CW.lock_path(v, res["SPY"], rf, None) for k, v in books.items()}

    print(f"\n  {'book':<7} {'plain: annual Sharpe  worst':>28}   "
          f"{'+LOCK30: annual Sharpe  worst':>30}   last5y Sharpe")
    S = {}
    for k in books:
        p, l = stats(books[k], rf), stats(locked[k], rf)
        rec = stats(locked[k][locked[k].index >= RECENT], rf)
        S[k] = l
        print(f"  {k:<7} {p[0]*100:>+11.1f}% {p[1]:>6.2f} {p[2]*100:>6.1f}%   "
              f"{l[0]*100:>+13.1f}% {l[1]:>6.2f} {l[2]*100:>6.1f}%   {rec[1]:>10.2f}")
    sp = stats(res["SPY"], rf)
    print(f"  {'SPY':<7} {sp[0]*100:>+11.1f}% {sp[1]:>6.2f} {sp[2]*100:>6.1f}%")

    c = res["STEADY"].corr(res["VALUE"])
    mo = (1 + res[["STEADY", "VALUE"]]).groupby(res.index.to_period("M")).prod() - 1
    bad = mo["STEADY"] < mo["STEADY"].quantile(0.2)
    print(f"\n  STEADY vs VALUE: daily correlation {c:.2f} | in STEADY's worst 20% of "
          f"months VALUE averaged {mo['VALUE'][bad].mean()*100:+.1f}% "
          f"(STEADY {mo['STEADY'][bad].mean()*100:+.1f}%)")

    print("\n  year by year, with the 30% lock")
    cols = ["STEADY", "VALUE", "VM50", "VM70", "T50", "T70"]
    print("  year " + "".join(f"{k:>9}" for k in cols))
    for y in sorted(set(res.index.year)):
        print(f"  {y} " + "".join(
            f"{((1 + locked[k][locked[k].index.year == y]).prod() - 1)*100:>+8.0f}%"
            for k in cols))

    print("\n  ===== VERDICT (pre-stated) =====")
    s0, d0 = S["STEADY"][1], S["STEADY"][2]
    result = None
    for k, base in (("VM50", "T50"), ("VM70", "T70")):
        s, d = S[k][1], S[k][2]
        ok = s > s0 and d > d0 and s > S[base][1]
        print(f"    {k}+LOCK30: Sharpe {s:.2f} vs STEADY {s0:.2f} and {base} "
              f"{S[base][1]:.2f} | worst {d*100:.1f}% vs {d0*100:.1f}% -> "
              f"{'PASS' if ok else 'FAIL'}")
        if ok:
            result = k
            break
    print("    " + (f"-> add the value book to CHIMERA as {result} (port to QC)"
                    if result else "-> keep CHIMERA as STEADY + lock"))
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")


if __name__ == "__main__":
    main()
