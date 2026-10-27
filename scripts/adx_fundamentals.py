"""
adx_fundamentals.py -- ADX (trend strength, from price) paired with
NON-PRICE information, on commodity ETFs (real prices, no roll artifacts).

COMBOS (direction d = +1 if +DI > -DI else -1; "strong" = ADX > 25)
  ADX_FLOW   strong trend AND big speculators (CFTC managed money) added to
             positions in direction d over the last 4 weekly reports
  ADX_ROOM   strong trend AND speculators not crowded in direction d
             (net position below its 3-year 80th percentile for longs,
             above the 20th for shorts)
  ADX_OI     strong trend AND open interest up >= 5% over 4 weeks (new
             money entering, not just short covering)
  ADX_INV    energy only (USO, BNO, UGA, UNG): strong UP trend while
             inventories are BELOW their 5-year same-week average
             (shortage) -> long; strong DOWN trend while ABOVE (glut) ->
             short. EIA weekly: crude stocks ex-SPR, gasoline stocks,
             natural gas storage
  context:   ADX_TREND (ADX alone, as in commodity_adx.py)

Data timing: CFTC positions are as of Tuesday, released Friday (+3 days);
EIA petroleum week ends Friday, released Wednesday (+5); EIA gas storage
week ends Friday, released Thursday (+6). Each is used only from its
release day; positions trade on the next day's return.
ETF -> data: USO/BNO crude, UGA gasoline, UNG natgas, GLD gold, SLV
silver, CPER copper, PPLT platinum, CORN corn, WEAT wheat, SOYB soybeans,
CANE sugar (DBA/DBB baskets dropped -- no single CFTC market).
Equal risk per ETF, 5 bps per unit traded, long/short.

PRE-STATED (4 combos, one shot)
  1. CANDIDATE = the combo with the best Sharpe over 2008 -> 2021-05
  2. it PASSES if over the last five years (2021-06 -> 2026-05): Sharpe
     >= 0.5 AND a 50/50 equal-risk blend with stock momentum (momentum_v2
     STEADY) beats momentum alone. Fail -> ADX is closed.

Needs data/cftc/ (from commodities_test.py). EIA files are downloaded
to data/eia/ (needs: pip3.10 install xlrd).

Run (repo root):
    python3.10 scripts/adx_fundamentals.py --smoke
    python3.10 scripts/adx_fundamentals.py | tee logs/adx_fundamentals.txt
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
sys.path.insert(0, HERE)
import commodities_test as C  # noqa: E402
import commodity_adx as A     # noqa: E402

EIA_DIR = os.path.join(C.ROOT, "data", "eia")
MAP = {"USO": "crude", "BNO": "crude", "UGA": "gasoline", "UNG": "natgas",
       "GLD": "gold", "SLV": "silver", "CPER": "copper", "PPLT": "platinum",
       "CORN": "corn", "WEAT": "wheat", "SOYB": "soybeans", "CANE": "sugar"}
EIA = {  # inventory series: (url, release lag days, sheet)
    "crude": ("https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls", 5),
    "gasoline": ("https://www.eia.gov/dnav/pet/hist_xls/WGTSTUS1w.xls", 5),
    "natgas": ("https://www.eia.gov/dnav/ng/hist_xls/"
               "NW2_EPG0_SWO_R48_BCFw.xls", 6),
}
COMBOS = ["ADX_FLOW", "ADX_ROOM", "ADX_OI", "ADX_INV"]


# ------------------------------------------------------------------- data
def cot_frames():
    frames = []
    for fn in sorted(os.listdir(C.CFTC_DIR)):
        if fn.endswith(".zip"):
            z = zipfile.ZipFile(os.path.join(C.CFTC_DIR, fn))
            frames.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),
                                      dtype=str, low_memory=False))
    c = pd.concat(frames, ignore_index=True)

    def col(key):
        return [x for x in c.columns if key.lower() in x.lower()][0]
    c = pd.DataFrame({
        "code": c[col("CFTC_Contract_Market_Code")].str.strip(),
        "date": pd.to_datetime(c[col("Report_Date_as_YYYY-MM-DD")]),
        "oi": pd.to_numeric(c[col("Open_Interest_All")], errors="coerce"),
        "mml": pd.to_numeric(c[col("M_Money_Positions_Long_All")],
                             errors="coerce"),
        "mms": pd.to_numeric(c[col("M_Money_Positions_Short_All")],
                             errors="coerce")}).dropna()
    code2m = {v[1]: m for m, v in C.MKT.items()}
    c = c[c["code"].isin(code2m)]
    c["mkt"] = c["code"].map(code2m)
    c["net"] = (c["mml"] - c["mms"]) / c["oi"]
    c = c.drop_duplicates(["mkt", "date"])
    net = c.pivot(index="date", columns="mkt", values="net").sort_index()
    oi = c.pivot(index="date", columns="mkt", values="oi").sort_index()
    return net, oi


def eia_series():
    import requests
    os.makedirs(EIA_DIR, exist_ok=True)
    out = {}
    for name, (url, lag) in EIA.items():
        p = os.path.join(EIA_DIR, os.path.basename(url))
        if not os.path.exists(p):
            r = requests.get(url, timeout=120,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            open(p, "wb").write(r.content)
        x = pd.read_excel(p, sheet_name=None, header=None)
        best = None
        for _, df in x.items():                 # find the data sheet
            d = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            v = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            s = pd.Series(v.values, d.values).dropna()
            s = s[s.index.notna()]
            if len(s) > 300 and (best is None or len(s) > len(best)):
                best = s
        best = best[~best.index.duplicated()].sort_index()
        best.index = best.index + pd.Timedelta(days=lag)   # release day
        out[name] = best
        print(f"  EIA {name}: {len(best)} weeks, {best.index[0]:%Y} -> "
              f"{best.index[-1]:%Y-%m-%d}")
    return out


def seasonal_gap(s):
    """Inventory minus its average for the same week over the prior 5 yrs."""
    wk = s.index.isocalendar().week.values
    yr = s.index.year.values
    gap = []
    for i in range(len(s)):
        m = (wk == wk[i]) & (yr < yr[i]) & (yr >= yr[i] - 5)
        gap.append(s.iloc[i] - s.values[m].mean() if m.sum() >= 3
                   else np.nan)
    return pd.Series(gap, s.index)


# ---------------------------------------------------------------- signals
def to_daily(weekly, idx, release_lag_days=0):
    w = weekly.copy()
    w.index = w.index + pd.Timedelta(days=release_lag_days)
    return w.reindex(idx, method="ffill", limit=10)


def combos(H, L, Cl, net, oi, inv):
    a, pdi, ndi = A.adx(H, L, Cl)
    d = np.sign(pdi - ndi)
    strong = a > 25
    cols = [s for s in Cl.columns if s in MAP]
    idx = Cl.index
    netd = pd.DataFrame({s: to_daily(net[MAP[s]], idx, 3)
                         for s in cols if MAP[s] in net})
    flow = pd.DataFrame({s: to_daily(net[MAP[s]].diff(4), idx, 3)
                         for s in cols if MAP[s] in net})
    pct = net.rolling(156, min_periods=104).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    pctd = pd.DataFrame({s: to_daily(pct[MAP[s]], idx, 3)
                         for s in cols if MAP[s] in pct})
    oig = pd.DataFrame({s: to_daily(oi[MAP[s]].pct_change(4), idx, 3)
                        for s in cols if MAP[s] in oi})
    base = d.where(strong, 0.0).where(a.notna())[cols]
    S = {"ADX_TREND": base}
    S["ADX_FLOW"] = base.where(np.sign(flow) == d[cols], 0.0) \
        .where(base.notna() & flow.notna())
    room = ((d[cols] > 0) & (pctd < 0.8)) | ((d[cols] < 0) & (pctd > 0.2))
    S["ADX_ROOM"] = base.where(room, 0.0).where(base.notna() & pctd.notna())
    S["ADX_OI"] = base.where(oig >= 0.05, 0.0).where(base.notna()
                                                     & oig.notna())
    ecols = [s for s in cols if MAP[s] in inv]
    if ecols:
        gap = pd.DataFrame({s: inv[MAP[s]].reindex(idx, method="ffill",
                                                   limit=10) for s in ecols})
        sig = pd.DataFrame(0.0, idx, ecols)
        sig[(base[ecols] > 0) & (gap < 0)] = 1.0
        sig[(base[ecols] < 0) & (gap > 0)] = -1.0
        S["ADX_INV"] = sig.where(base[ecols].notna() & gap.notna())
    return S


# ------------------------------------------------------------------ smoke
def smoke():
    H, L, Cl, V = A.smoke()
    rng = np.random.default_rng(12)
    wk = pd.date_range("2006-01-03", "2026-10-20", freq="W-TUE")
    mk = sorted(set(MAP.values()))
    net = pd.DataFrame(np.cumsum(rng.normal(0, 0.02, (len(wk), len(mk))), 0),
                       wk, mk)
    oi = pd.DataFrame(1e5 * np.exp(np.cumsum(rng.normal(0, 0.03,
                                                        (len(wk), len(mk))),
                                             0)), wk, mk)
    inv = {k: pd.Series(rng.normal(0, 10, 900),
                        pd.date_range("2006-01-06", periods=900, freq="W-FRI")
                        + pd.Timedelta(days=5)) for k in EIA}
    return H, L, Cl, V, net, oi, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        H, L, Cl, V, net, oi, inv = smoke()
        print("  smoke: planted price trends; positioning/inventories are "
              "noise")
    else:
        H, L, Cl, V = A.load()
        net, oi = cot_frames()
        print(f"  CFTC: {net.shape[1]} markets, {net.index[0]:%Y} -> "
              f"{net.index[-1]:%Y-%m-%d}")
        try:
            raw = eia_series()
            inv = {k: seasonal_gap(v) for k, v in raw.items()}
        except Exception as e:  # noqa: BLE001
            print(f"  EIA download/parse failed ({e}); ADX_INV skipped")
            inv = {}
    R = Cl.pct_change(fill_method=None).clip(-0.25, 0.25)
    vol = R.rolling(60, min_periods=40).std() * np.sqrt(252)
    S = combos(H, L, Cl, net, oi, inv)
    if a.smoke:
        mom_m = C.monthly(pd.Series(np.random.default_rng(9).normal(
            0.0005, 0.01, len(R)), R.index))
    else:
        mom_m, _ = C.stock_momentum()
    print(f"\n  {'combo':<10} {'ETFs':>4} | {'2008-21 ann':>11} {'sh':>5} | "
          f"{'last5y ann':>10} {'sh':>5} {'worst':>6} {'2022':>5} | "
          f"{'mom':>5} {'blend':>5} | {'time in market':>14}")
    res = {}
    for k in ["ADX_TREND"] + COMBOS:
        if k not in S:
            print(f"  {k:<10} no data")
            continue
        W = S[k]
        d = A.book(W, R[W.columns], vol[W.columns])
        a0, s0, a1, v1, s1, dd, y22 = A.split(d)
        msh, bsh, _ = C.blend_sharpe(mom_m, C.monthly(d), C.RECENT, C.DEC1)
        live = (W.abs() > 0).sum(axis=1) / W.notna().sum(axis=1)
        print(f"  {k:<10} {W.shape[1]:>4} | {a0*100:>+10.1f}% {s0:>5.2f} | "
              f"{a1*100:>+9.1f}% {s1:>5.2f} {dd*100:>5.0f}% {y22*100:>+4.0f}%"
              f" | {msh:>5.2f} {bsh:>5.2f} | {live.mean()*100:>12.0f}%")
        if k in COMBOS:
            res[k] = (s0, s1, bsh > msh)
    cand = max(res, key=lambda k: res[k][0])
    s0, s1, helps = res[cand]
    ok = s1 >= 0.5 and helps
    print(f"\n  CANDIDATE (best 2008-2021 Sharpe, chosen before looking at "
          f"the last 5 years): {cand} ({s0:.2f})")
    print("  ===== VERDICT (pre-stated) =====")
    print(f"  {cand}: last-5y Sharpe {s1:.2f} (need >= 0.5), improves the "
          f"momentum blend: {helps} -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
