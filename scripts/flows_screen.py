"""
flows_screen.py -- Primordial screen for the five UNSEEN futures/commodity
ideas (free data only). QuantConnect comes after something holds up here.

WHY ETFs: Yahoo's continuous futures (CL=F ...) jump at every contract roll,
so their daily returns are wrong. Futures-based ETFs (USO, UNG, CORN ...)
hold and roll the contracts, so their returns are what a futures holder
earns. Physical metal ETFs (GLD SLV PPLT PALL) track spot; minus T-bills
they match a futures holder's return. Every return here is in EXCESS of
T-bills (BIL), like a futures position.

DATA (free, fetched on your Mac, cached)
  * Yahoo daily (auto-adjusted) via primordial.data: commodity ETFs, their
    front-month futures (only for the carry estimate), IEI, IEF, BIL
  * CFTC disaggregated futures-only, producers/merchants ("hedgers"),
    2006+ -> data/cftc/ (same cache as commodities_test.py)
  * 10-year Treasury note auction dates (new + reopenings), 2009+, from
    fiscaldata.treasury.gov -> data/auctions_10y.csv

MARKETS (12): crude USO/CL, natgas UNG/NG, gasoline UGA/RB, copper CPER/HG,
  corn CORN/ZC, wheat WEAT/ZW, soybeans SOYB/ZS, sugar CANE/SB, and physical
  gold GLD, silver SLV, platinum PPLT, palladium PALL.
  CARRY estimate = ETF 12m return - front-futures 12m price change (= what
  rolling paid or cost; curve shape). Physical metals: carry = -T-bill 12m.

THE FIVE (all rules fixed here, before any real-data run)
  HEDGE_PR   monthly: hedgers' long share of their positions (latest report
             released by the last business day: report date + 3 days);
             long the bottom third (hedgers most short), short the top third
  PRESSURE   monthly: long = top carry third AND bottom hedger third;
             short = bottom carry third AND top hedger third; else flat;
             needs >= 8 markets with both signals
  MONTH_END  long IEI + IEF over the last 3 trading days of every month
  AUCTION    short IEF from 5 trading days before each 10-year note auction
             to the auction-day close, then long for 3 trading days
  TSY_CAL    MONTH_END + AUCTION, equal risk
  control RANDOM  random monthly long/short/flat on the 12 commodities
  Sizing: each position = 10% / its own 60-day volatility / positions.
  Costs per unit traded: commodity ETFs 5 bp, Treasury ETFs 2 bp.

PRE-STATED PASS, ALL of (same bar as futures_lab.py):
  (a) Sharpe >= 0.7 over the last five years (2021-11 -> end of data)
  (b) Sharpe >= 0.5 over 2012-07 -> 2021-10
  (c) a 50/50 equal-risk monthly blend with CHIMERA-REGIME-GUARD's QC paper
      returns (2017-02 -> 2026-05) beats CHIMERA's monthly Sharpe (1.42)
  5 judged rows -> ~1.5% chance a pure-noise row passes.
  A PASS here earns a QuantConnect test on real futures; nothing more.

Run (repo root):
    python3.10 scripts/flows_screen.py --smoke     # planted answers
    python3.10 scripts/flows_screen.py | tee logs/flows_screen.txt
"""
import argparse
import io
import os
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
CFTC_DIR = os.path.join(ROOT, "data", "cftc")
AUC_CSV = os.path.join(ROOT, "data", "auctions_10y.csv")
START, END = "2009-01-01", "2026-11-03"
EARLY = (pd.Timestamp("2012-07-01"), pd.Timestamp("2021-11-01"))
RECENT = (pd.Timestamp("2021-11-01"), pd.Timestamp("2099-01-01"))
TGT, COST_C, COST_T = 0.10, 0.0005, 0.0002
UA = {"User-Agent": "Mozilla/5.0"}

# name: (ETF, front future or None for physical, CFTC code)
MKT = {"crude": ("USO", "CL=F", "067651"), "natgas": ("UNG", "NG=F", "023651"),
       "gasoline": ("UGA", "RB=F", "111659"), "copper": ("CPER", "HG=F", "085692"),
       "corn": ("CORN", "ZC=F", "002602"), "wheat": ("WEAT", "ZW=F", "001602"),
       "soybeans": ("SOYB", "ZS=F", "005602"), "sugar": ("CANE", "SB=F", "080732"),
       "gold": ("GLD", None, "088691"), "silver": ("SLV", None, "084691"),
       "platinum": ("PPLT", None, "076651"), "palladium": ("PALL", None, "075651")}
CHIM = {"2017-02": 0.05638, "2017-03": -0.00502, "2017-04": 0.0516, "2017-05": 0.05944, "2017-06": -0.04716, "2017-07": 0.08036, "2017-08": 0.04121, "2017-09": -0.0082, "2017-10": 0.01275, "2017-11": 0.06163, "2017-12": 0.00984, "2018-01": 0.05409, "2018-02": -0.04113, "2018-03": -0.00773, "2018-04": -0.00085, "2018-05": 0.02454, "2018-06": -0.02166, "2018-07": -0.03758, "2018-08": 0.04259, "2018-09": 0.01471, "2018-10": -0.09646, "2018-11": -0.00262, "2018-12": -0.03574, "2019-01": 0.05253, "2019-02": 0.02498, "2019-03": 0.01921, "2019-04": -0.01715, "2019-05": -0.0174, "2019-06": 0.06728, "2019-07": 0.04436, "2019-08": 0.01406, "2019-09": -0.03595, "2019-10": 0.02579, "2019-11": 0.03318, "2019-12": 0.08462, "2020-01": 0.00542, "2020-02": -0.0347, "2020-03": -0.03995, "2020-04": 0.10273, "2020-05": 0.02383, "2020-06": 0.08567, "2020-07": 0.04235, "2020-08": -0.0517, "2020-09": -0.00151, "2020-10": -0.00627, "2020-11": 0.17142, "2020-12": 0.03592, "2021-01": 0.20194, "2021-02": 0.1025, "2021-03": 0.05749, "2021-04": 0.04693, "2021-05": 0.0169, "2021-06": 0.00755, "2021-07": 0.01409, "2021-08": 0.02306, "2021-09": -0.03898, "2021-10": 0.06648, "2021-11": -0.02027, "2021-12": 0.05834, "2022-01": -0.13833, "2022-02": 0.03776, "2022-03": 0.07138, "2022-04": 0.04416, "2022-05": 0.01971, "2022-06": -0.03329, "2022-07": 0.01483, "2022-08": 0.04692, "2022-09": 0.00549, "2022-10": 0.01113, "2022-11": -0.07182, "2022-12": -0.03874, "2023-01": -0.00446, "2023-02": 0.00744, "2023-03": -0.01724, "2023-04": 0.00149, "2023-05": 0.04634, "2023-06": -0.00555, "2023-07": 0.01982, "2023-08": -0.02555, "2023-09": -0.03562, "2023-10": -0.05146, "2023-11": 0.03769, "2023-12": 0.07371, "2024-01": 0.04383, "2024-02": 0.10113, "2024-03": 0.02381, "2024-04": -0.00117, "2024-05": 0.09167, "2024-06": 0.00586, "2024-07": 0.03194, "2024-08": 0.02881, "2024-09": 0.02354, "2024-10": -0.01373, "2024-11": 0.05931, "2024-12": -0.05531, "2025-01": 0.04376, "2025-02": -0.03464, "2025-03": 0.01805, "2025-04": 0.01726, "2025-05": 0.0183, "2025-06": 0.02993, "2025-07": 0.00038, "2025-08": 0.00485, "2025-09": 0.17491, "2025-10": -0.00345, "2025-11": 0.01746, "2025-12": -0.00032, "2026-01": 0.11679, "2026-02": 0.11839, "2026-03": 0.04071, "2026-04": 0.05341, "2026-05": 0.00883}


# ===================================================================== data
def load_prices():
    from primordial.data import fetch
    etfs = [v[0] for v in MKT.values()] + ["IEI", "IEF", "BIL"]
    futs = [v[1] for v in MKT.values() if v[1]]
    d = fetch("yfinance", etfs + futs, "1d", START, END, min_bars=200)
    px = pd.DataFrame({s: d[s]["close"] for s in etfs + futs if s in d})
    px.index = pd.to_datetime(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    miss = [s for s in etfs + futs if s not in px]
    print(f"  prices: {px.shape[1]} series {px.index[0].date()} -> {px.index[-1].date()}"
          + (f"  MISSING {miss}" if miss else ""))
    return px


def load_auctions():
    import requests
    if os.path.exists(AUC_CSV):
        ds = pd.read_csv(AUC_CSV)["date"].tolist()
    else:
        url = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/"
               "accounting/od/auctions_query")
        p = {"fields": "auction_date,security_type,security_term,original_security_term",
             "filter": "security_type:eq:Note,auction_date:gte:2009-01-01",
             "page[size]": "10000"}
        data = requests.get(url, params=p, headers=UA, timeout=60).json()["data"]
        ds = sorted({r["auction_date"] for r in data
                     if "10-Year" in (r.get("original_security_term") or "")
                     or (r.get("security_term") or "").startswith("10-Year")})
        os.makedirs(os.path.dirname(AUC_CSV), exist_ok=True)
        pd.DataFrame({"date": ds}).to_csv(AUC_CSV, index=False)
    print(f"  10-year note auctions: {len(ds)} ({ds[0]} -> {ds[-1]})")
    return ds


def _norm(c):
    return "".join(ch for ch in c.lower() if ch.isalnum())


def load_hedgers():
    import requests
    os.makedirs(CFTC_DIR, exist_ok=True)
    files = ["fut_disagg_txt_hist_2006_2016.zip"] + \
        [f"fut_disagg_txt_{y}.zip" for y in range(2017, 2027)]
    frames = []
    for fn in files:
        p = os.path.join(CFTC_DIR, fn)
        if not os.path.exists(p):
            r = requests.get("https://www.cftc.gov/files/dea/history/" + fn,
                             headers=UA, timeout=180)
            if not r.ok:
                print(f"    CFTC {fn}: HTTP {r.status_code}, skipped")
                continue
            open(p, "wb").write(r.content)
        z = zipfile.ZipFile(p)
        frames.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),
                                  dtype=str, low_memory=False))
    c = pd.concat(frames, ignore_index=True)
    cols = {_norm(x): x for x in c.columns}

    def col(key):
        hits = [v for n, v in cols.items() if n.startswith(_norm(key))]
        if not hits:
            raise KeyError(f"no CFTC column like {key}")
        return hits[0]
    d = pd.DataFrame({
        "code": c[col("CFTC_Contract_Market_Code")].str.strip(),
        "date": pd.to_datetime(c[col("Report_Date_as_YYYY")]),
        "lo": pd.to_numeric(c[col("Prod_Merc_Positions_Long_All")], errors="coerce"),
        "sh": pd.to_numeric(c[col("Prod_Merc_Positions_Short_All")], errors="coerce")})
    code2m = {v[2]: m for m, v in MKT.items()}
    d = d[d["code"].isin(code2m)].dropna()
    d["m"] = d["code"].map(code2m)
    d["hp"] = d["lo"] / (d["lo"] + d["sh"])
    d["avail"] = d["date"] + pd.Timedelta(days=3)          # Friday release
    w = d.drop_duplicates(["m", "date"]).pivot(index="avail", columns="m", values="hp")
    print(f"  CFTC hedgers: {w.shape[1]} markets, {w.index.min().date()} -> "
          f"{w.index.max().date()}")
    return w.sort_index()


def build(px, hedgers):
    bil = px["BIL"].pct_change(fill_method=None).fillna(0.0)
    X, carry = {}, {}
    for m, (etf, fut, _) in MKT.items():
        if etf not in px:
            continue
        r = px[etf].pct_change(fill_method=None).clip(-0.25, 0.25)
        X[m] = r - bil
        e12 = px[etf] / px[etf].shift(252) - 1
        if fut and fut in px:
            carry[m] = e12 - (px[fut] / px[fut].shift(252) - 1)
        else:
            carry[m] = -(1 + bil).rolling(252).apply(np.prod, raw=True) + 1
    X = pd.DataFrame(X)
    T = pd.DataFrame({t: px[t].pct_change(fill_method=None) - bil
                      for t in ("IEI", "IEF") if t in px})
    me = month_ends(X.index)
    hp = None
    if hedgers is not None:
        hp = hedgers.reindex(hedgers.index.union(me)).ffill(limit=10).reindex(me)
    return {"X": X, "carry": pd.DataFrame(carry).reindex(X.index), "T": T,
            "hp": hp}


# ==================================================================== maths
def month_ends(idx):
    s = pd.Series(idx, idx)
    return pd.DatetimeIndex(sorted(s.groupby([idx.year, idx.month]).max()))


def vol60(R):
    return R.rolling(60, min_periods=40).std() * np.sqrt(252)


def size(sign, vol):
    v = vol.reindex(sign.index)
    n = (sign.fillna(0) != 0).sum(axis=1).replace(0, np.nan)
    return (sign * (TGT / v)).div(n, axis=0).replace([np.inf, -np.inf], np.nan)


def book(R, W, cost):
    W = W.reindex(R.index).ffill().fillna(0.0).reindex(columns=R.columns).fillna(0.0)
    pnl = (W.shift(1) * R.fillna(0.0)).sum(axis=1)
    return pnl - W.diff().abs().sum(axis=1).fillna(0.0) * cost


def eq_risk(*xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    df = pd.concat(xs, axis=1).fillna(0.0)
    sd = df.std().replace(0, np.nan)
    return (df / sd).mean(axis=1) * sd.mean()


# =============================================================== strategies
def s_hedge(D):
    if D["hp"] is None:
        return None
    X = D["X"]
    hp = D["hp"].reindex(columns=X.columns)
    rk = hp.rank(axis=1, pct=True)
    s = pd.DataFrame(0.0, hp.index, X.columns)
    s[rk <= 1 / 3] = 1.0
    s[rk > 2 / 3] = -1.0
    s = s.where(hp.notna())
    s[hp.notna().sum(axis=1) < 8] = np.nan
    return book(X, size(s, vol60(X)), COST_C)


def s_pressure(D):
    if D["hp"] is None:
        return None
    X = D["X"]
    me = D["hp"].index
    hp = D["hp"].reindex(columns=X.columns)
    car = D["carry"].reindex(me)
    ok = hp.notna() & car.notna()
    hr = hp.where(ok).rank(axis=1, pct=True)
    cr = car.where(ok).rank(axis=1, pct=True)
    s = pd.DataFrame(0.0, me, X.columns)
    s[(cr > 2 / 3) & (hr <= 1 / 3)] = 1.0
    s[(cr <= 1 / 3) & (hr > 2 / 3)] = -1.0
    s = s.where(ok)
    s[ok.sum(axis=1) < 8] = np.nan
    return book(X, size(s, vol60(X)), COST_C)


def s_month_end(D):
    T = D["T"]
    if T.empty:
        return None
    idx = T.index
    ym = idx.year * 12 + idx.month
    last = np.flatnonzero(np.r_[ym[1:] != ym[:-1], True])
    w = pd.DataFrame(0.0, idx, T.columns)
    for p in last:
        if p >= 3:
            w.iloc[p - 3:p] = 1.0              # set at close of day -3, earn last 3 days
    return book(T, w * (TGT / vol60(T)) / T.shape[1], COST_T)


def s_auction(D):
    T = D["T"]
    if "IEF" not in T or not D.get("auctions"):
        return None
    R = T[["IEF"]]
    idx = R.index
    w = pd.Series(0.0, idx)
    for a in D["auctions"]:
        a = pd.Timestamp(a)
        pos = idx.searchsorted(a)
        if pos >= len(idx) or pos < 5 or abs((idx[pos] - a).days) > 3:
            continue
        w.iloc[pos - 5:pos] = -1.0             # held into the auction-day close
        w.iloc[pos:pos + 3] = 1.0              # 3 days after
    return book(R, w.to_frame("IEF") * (TGT / vol60(R)), COST_T)


def s_random(D):
    X = D["X"]
    me = month_ends(X.index)
    rng = np.random.default_rng(7)
    s = pd.DataFrame(rng.choice([-1.0, 0.0, 1.0], size=(len(me), X.shape[1])),
                     me, X.columns)
    return book(X, size(s, vol60(X)), COST_C)


# =================================================================== judge
def stats(x):
    x = x.dropna()
    if len(x) < 120 or x.std() == 0:
        return (np.nan,) * 4
    eq = (1 + x).cumprod()
    vol = x.std() * np.sqrt(252)
    return eq.iloc[-1] ** (252 / len(x)) - 1, vol, x.mean() * 252 / vol, \
        (eq / eq.cummax() - 1).min()


def win(x, w):
    return x[(x.index >= w[0]) & (x.index < w[1])]


def blend(daily):
    m = (1 + daily).resample("ME").prod() - 1
    m.index = m.index.strftime("%Y-%m")
    j = pd.concat([pd.Series(CHIM), m], axis=1, keys=["c", "s"]).dropna()
    if len(j) < 36 or j["s"].std() == 0:
        return np.nan, np.nan, np.nan
    sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
    b = 0.5 * j["c"] + 0.5 * j["s"] * (j["c"].std() / j["s"].std())
    return sh(j["c"]), sh(b), j["c"].corr(j["s"])


def report(D):
    S = {"HEDGE_PR": s_hedge(D), "PRESSURE": s_pressure(D),
         "MONTH_END": s_month_end(D), "AUCTION": s_auction(D)}
    S["TSY_CAL"] = eq_risk(S["MONTH_END"], S["AUCTION"])
    S["control RANDOM"] = s_random(D)
    print(f"\n  {'strategy':<15}| 2012-21 ann  shp | last5y ann  vol  shp  worst  2022 |"
          f" corrCHIM  CHIM  blend | verdict")
    for k, x in S.items():
        if x is None:
            print(f"  {k:<15}| no data")
            continue
        a0, _, s0, _ = stats(win(x, EARLY))
        a1, v1, s1, dd1 = stats(win(x, RECENT))
        y22 = (1 + x[x.index.year == 2022]).prod() - 1
        csh, bsh, corr = blend(x)
        ok = (s1 >= 0.7) and (s0 >= 0.5) and (bsh > csh)
        print(f"  {k:<15}| {a0*100:+7.1f}% {s0:5.2f} | {a1*100:+7.1f}% {v1*100:4.0f}% "
              f"{s1:5.2f} {dd1*100:5.0f}% {y22*100:+5.0f}% | {corr:+6.2f} {csh:6.2f} "
              f"{bsh:6.2f} | {'PASS' if ok else 'fail'}")
    print("  CHIM/blend = monthly Sharpe 2017-02..2026-05; pass needs (a) last-5y >= 0.7, "
          "(b) 2012-21 >= 0.5, (c) blend > CHIM")


# =================================================================== smoke
def smoke():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-04", "2026-10-30")
    n = len(idx)
    X = pd.DataFrame(rng.normal(0, 0.015, (n, len(MKT))), idx, list(MKT))
    carry = pd.DataFrame(np.repeat(rng.normal(0, 0.1, (n // 21 + 1, len(MKT))), 21,
                                   axis=0)[:n], idx, list(MKT))
    me = month_ends(idx)
    hp = pd.DataFrame(rng.uniform(0.3, 0.7, (len(me), len(MKT))), me, list(MKT))
    cr, hr = carry.reindex(me).rank(axis=1, pct=True), hp.rank(axis=1, pct=True)
    good = ((cr > 2 / 3) & (hr <= 1 / 3)).astype(float) - \
        ((cr <= 1 / 3) & (hr > 2 / 3)).astype(float)
    for i in range(len(me) - 1):                       # plant PRESSURE
        sel = (idx > me[i]) & (idx <= me[i + 1])
        X.loc[sel] += good.iloc[i].values * 0.004
    T = pd.DataFrame(rng.normal(0, 0.004, (n, 2)), idx, ["IEI", "IEF"])
    ym = idx.year * 12 + idx.month
    for p in np.flatnonzero(np.r_[ym[1:] != ym[:-1], True]):   # plant MONTH_END
        T.iloc[p - 2:p + 1] += 0.0012
    auc = [str(pd.Timestamp(y, m, 12).date()) for y in range(2010, 2027)
           for m in range(1, 13) if idx[10] < pd.Timestamp(y, m, 12) < idx[-10]]
    for a in auc:                                      # plant AUCTION on IEF
        p = idx.searchsorted(pd.Timestamp(a))
        T.iloc[p - 4:p + 1, 1] -= 0.0015
        T.iloc[p + 1:p + 4, 1] += 0.0015
    return {"X": X, "carry": carry, "T": T, "hp": hp, "auctions": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        D = smoke()
        print("  smoke: PRESSURE, MONTH_END, AUCTION (+TSY_CAL) planted -> should PASS; "
              "RANDOM must fail; HEDGE_PR partly overlaps the planted PRESSURE")
    else:
        px = load_prices()
        try:
            hed = load_hedgers()
        except Exception as e:  # noqa: BLE001
            print(f"  CFTC failed ({e}); HEDGE_PR/PRESSURE skipped")
            hed = None
        D = build(px, hed)
        try:
            D["auctions"] = load_auctions()
        except Exception as e:  # noqa: BLE001
            print(f"  auction dates failed ({e}); AUCTION skipped")
            D["auctions"] = []
    report(D)


if __name__ == "__main__":
    main()
