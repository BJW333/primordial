"""
commodities_test.py -- five commodity strategies across 20 futures markets,
plus whether each one helps the stock momentum system.

DATA (free; downloaded on your Mac, cached):
  * Yahoo continuous front-month futures (CL=F, GC=F, ZC=F ...), 2000+,
    through primordial.data (cache ~/.primordial_cache)
  * Commodity ETFs for the carry estimate and an Alpaca long-only version
  * CFTC Commitments of Traders, disaggregated futures-only, 2006+
    (yearly zips from cftc.gov -> data/cftc/)
  Daily futures returns are clipped at +/-25% (front-month rolls and the
  April 2020 negative oil print are not real returns).

STRATEGIES (each market sized to equal risk: weight = 10% / its own
volatility / number of markets; positions set at a close, earn from the
next day; 2 bps per unit traded)
  TREND   long if the last 12 months' return > 0, short if < 0; monthly
  CARRY   ESTIMATE: (ETF 12m return - futures price 12m change) = what
          rolling paid or cost = curve shape. Long top third, short bottom
          third; monthly; only the ~10 markets with a futures ETF
  COT     speculators' (managed money) net position as % of open interest,
          vs its own last 3 years: fade the top 10% (short) and bottom 10%
          (long). Weekly; usable from Friday's release (+3 days after the
          Tuesday report date), acted on the next trading day
  SEASON  for each market and calendar month: sign of that month's
          average return in all PRIOR years (min 5); monthly
  REPORTS oil (Wednesday inventory report) and natural gas (Thursday
          storage report): if the report-day move is > 1 standard
          deviation, follow it for the next 5 trading days

PRE-STATED (5 strategies; each judged on its own)
  PASS if ALL of (tightened after a pure-noise check let a random
  strategy through the looser 0.4 / > 0 bar -- set before any real run):
    (a) Sharpe >= 0.5 over the last five years (2021-06 -> 2026-05)
    (b) Sharpe >= 0.3 over 2007 -> 2021-05 (it has to have worked before)
    (c) blending it 50/50 (equal risk) with stock momentum (momentum_v2
        STEADY) raises momentum's last-five-year Sharpe
  Also printed (not judged): 2022 return, correlation with stock momentum,
  and a long-only ETF version you could run on Alpaca.

Run (repo root):
    python3.10 scripts/commodities_test.py --smoke
    python3.10 scripts/commodities_test.py | tee logs/commodities_test.txt
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
MOM = os.path.join(ROOT, "data", "momentum_v2_returns.csv")
SPY_US = os.path.join(ROOT, "data", "us_universe", "spy.parquet")
START, END = "2000-01-01", "2026-10-24"
CTX0, RECENT, DEC1 = (pd.Timestamp("2007-01-01"), pd.Timestamp("2021-06-01"),
                      pd.Timestamp("2026-06-01"))
TGT, COST, ETF_COST = 0.10, 0.0002, 0.0005

# name: (yahoo future, CFTC code, ETF or None)
MKT = {
    "crude":     ("CL=F", "067651", "USO"),
    "natgas":    ("NG=F", "023651", "UNG"),
    "heatoil":   ("HO=F", "022651", None),
    "gasoline":  ("RB=F", "111659", "UGA"),
    "gold":      ("GC=F", "088691", "GLD"),
    "silver":    ("SI=F", "084691", "SLV"),
    "copper":    ("HG=F", "085692", "CPER"),
    "platinum":  ("PL=F", "076651", "PPLT"),
    "corn":      ("ZC=F", "002602", "CORN"),
    "wheat":     ("ZW=F", "001602", "WEAT"),
    "kcwheat":   ("KE=F", "001612", None),
    "soybeans":  ("ZS=F", "005602", "SOYB"),
    "soyoil":    ("ZL=F", "007601", None),
    "soymeal":   ("ZM=F", "026603", None),
    "sugar":     ("SB=F", "080732", "CANE"),
    "coffee":    ("KC=F", "083731", None),
    "cocoa":     ("CC=F", "073732", None),
    "cotton":    ("CT=F", "033661", None),
    "cattle":    ("LE=F", "057642", None),
    "hogs":      ("HE=F", "054642", None),
}
STRATS = ["TREND", "CARRY", "COT", "SEASON", "REPORTS"]


# ------------------------------------------------------------------- data
def load_prices():
    from primordial.data import fetch
    fut = {m: v[0] for m, v in MKT.items()}
    etf = {m: v[2] for m, v in MKT.items() if v[2]}
    d = fetch("yfinance", list(fut.values()) + list(etf.values()), "1d",
              START, END, min_bars=200)
    F = pd.DataFrame({m: d[s]["close"] for m, s in fut.items() if s in d})
    E = pd.DataFrame({m: d[s]["close"] for m, s in etf.items() if s in d})
    F.index = pd.to_datetime(F.index).normalize()
    E.index = pd.to_datetime(E.index).normalize()
    missing = [m for m in MKT if m not in F]
    print(f"  futures: {F.shape[1]} markets {F.index[0].date()} -> "
          f"{F.index[-1].date()}" + (f" (missing: {missing})" if missing
                                     else ""))
    print(f"  ETFs: {', '.join(f'{m}={MKT[m][2]}' for m in E.columns)}")
    return F.sort_index(), E.sort_index()


def load_cot():
    import requests
    os.makedirs(CFTC_DIR, exist_ok=True)
    files = ["fut_disagg_txt_hist_2006_2016.zip"] + \
        [f"fut_disagg_txt_{y}.zip" for y in range(2017, 2027)]
    frames = []
    for fn in files:
        p = os.path.join(CFTC_DIR, fn)
        if not os.path.exists(p):
            r = requests.get("https://www.cftc.gov/files/dea/history/" + fn,
                             timeout=120,
                             headers={"User-Agent": "Mozilla/5.0"})
            if not r.ok:
                print(f"    CFTC {fn}: HTTP {r.status_code}, skipped")
                continue
            open(p, "wb").write(r.content)
        z = zipfile.ZipFile(p)
        frames.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])),
                                  dtype=str, low_memory=False))
    c = pd.concat(frames, ignore_index=True)

    def col(key):
        hits = [x for x in c.columns if key.lower() in x.lower()]
        return hits[0]
    c = pd.DataFrame({
        "code": c[col("CFTC_Contract_Market_Code")].str.strip(),
        "date": pd.to_datetime(c[col("Report_Date_as_YYYY-MM-DD")]),
        "oi": pd.to_numeric(c[col("Open_Interest_All")], errors="coerce"),
        "mml": pd.to_numeric(c[col("M_Money_Positions_Long_All")],
                             errors="coerce"),
        "mms": pd.to_numeric(c[col("M_Money_Positions_Short_All")],
                             errors="coerce")})
    code2m = {v[1]: m for m, v in MKT.items()}
    c = c[c["code"].isin(code2m)].dropna()
    c["mkt"] = c["code"].map(code2m)
    c["net"] = (c["mml"] - c["mms"]) / c["oi"]
    c = c.drop_duplicates(["mkt", "date"]).sort_values("date")
    print(f"  CFTC positioning: {c['mkt'].nunique()} markets, "
          f"{c['date'].min().date()} -> {c['date'].max().date()}")
    return c.pivot(index="date", columns="mkt", values="net")


# ------------------------------------------------------------- strategies
def month_ends(idx):
    s = pd.Series(idx, idx)
    return set(s.groupby([idx.year, idx.month]).max())


def run_book(R, W, cost):
    """Daily P&L of weights W (set at close t, earn r_{t+1})."""
    W = W.reindex(R.index).ffill().fillna(0.0)
    pnl = (W.shift(1) * R).sum(axis=1)
    turn = W.diff().abs().sum(axis=1).fillna(0)
    return pnl - turn * cost


def risk_w(sign, vol):
    n = sign.notna().sum(axis=1).replace(0, np.nan)
    return (sign * (TGT / vol)).div(n, axis=0)


def sig_trend(F, R, vol):
    me = month_ends(F.index)
    r12 = F / F.shift(252) - 1
    s = np.sign(r12).loc[F.index.isin(me)].dropna(how="all")
    return risk_w(s, vol.reindex(s.index))


def sig_carry(F, E, vol):
    me = month_ends(F.index)
    cols = [m for m in E.columns if m in F.columns]
    carry = (E[cols] / E[cols].shift(252) - 1) - \
        (F[cols].reindex(E.index) / F[cols].reindex(E.index).shift(252) - 1)
    carry = carry.loc[carry.index.isin(me)].dropna(how="all")
    rk = carry.rank(axis=1, pct=True)
    s = pd.DataFrame(0.0, rk.index, rk.columns)
    s[rk >= 2 / 3] = 1.0
    s[rk <= 1 / 3] = -1.0
    s = s.where(carry.notna())
    return risk_w(s, vol.reindex(s.index)[cols])


def sig_cot(F, cot, vol):
    if cot is None or cot.empty:
        return pd.DataFrame()
    pct = cot.rolling(156, min_periods=104).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    s = pd.DataFrame(0.0, pct.index, pct.columns)
    s[pct >= 0.9] = -1.0
    s[pct <= 0.1] = 1.0
    s = s.where(pct.notna())
    s.index = s.index + pd.Timedelta(days=3)          # Friday release
    s = s.reindex(F.index, method="ffill", limit=7)
    return risk_w(s, vol[s.columns])


def sig_season(F, vol):
    m = F.resample("ME").last()
    mr = m / m.shift(1) - 1
    rows = {}
    for d in mr.index:
        nxt = (d + pd.offsets.MonthEnd(1))
        past = mr[(mr.index.month == nxt.month) & (mr.index <= d)]
        ok = past.notna().sum() >= 5
        rows[d] = np.sign(past.mean()).where(ok)
    s = pd.DataFrame(rows).T
    idx = F.index
    last_day = {d: idx[idx <= d][-1] for d in s.index if (idx <= d).any()}
    s = s.loc[list(last_day)]
    s.index = [last_day[d] for d in s.index]
    return risk_w(s, vol.reindex(s.index))


def sig_reports(F, R, vol):
    out = pd.DataFrame(0.0, F.index, [c for c in ("crude", "natgas")
                                      if c in F.columns])
    dvol = R.rolling(60, min_periods=40).std().shift(1)
    for mkt, wd in (("crude", 2), ("natgas", 3)):
        if mkt not in F:
            continue
        z = R[mkt] / dvol[mkt]
        pos = pd.Series(0.0, F.index)
        idx = np.flatnonzero((F.index.weekday == wd) & (z.abs() > 1).values)
        for i in idx:
            pos.iloc[i:i + 5] = np.sign(z.iloc[i])   # set at close of day i
        out[mkt] = pos
    return out * (TGT / vol[out.columns]) / out.shape[1]


def etf_long_only(W, E, F):
    """Same signals, long side only, on the matching ETF (Alpaca-style)."""
    cols = [m for m in W.columns if m in E.columns]
    if not cols:
        return None
    RE = E[cols].pct_change(fill_method=None).clip(-0.25, 0.25)
    vol_e = RE.rolling(60, min_periods=40).std() * np.sqrt(252)
    Wl = W[cols].clip(lower=0)
    Wl = (Wl > 0).astype(float).where(W[cols].notna()) * \
        (TGT / vol_e.reindex(Wl.index)) / len(cols)
    return run_book(RE, Wl, ETF_COST)


# ----------------------------------------------------------------- report
def stats(daily):
    x = daily.dropna()
    if len(x) < 60:
        return (np.nan,) * 4
    eq = (1 + x).cumprod()
    yrs = len(x) / 252
    ann = eq.iloc[-1] ** (1 / yrs) - 1
    v = x.std() * np.sqrt(252)
    return ann, v, (x.mean() * 252 / v if v else np.nan), \
        (eq / eq.cummax() - 1).min()


def monthly(daily):
    return (1 + daily).resample("ME").prod() - 1


def stock_momentum():
    if os.path.exists(MOM):
        m = pd.read_csv(MOM, index_col=0, parse_dates=True)["STEADY"]
        m.index = m.index.to_period("M").to_timestamp("M")
        return m, "stock momentum (momentum_v2 STEADY)"
    s = pd.read_parquet(SPY_US)["SPY"].resample("ME").last().pct_change()
    return s, "SPY (momentum_v2 returns not found)"


def blend_sharpe(mom_m, strat_m, s0, s1):
    j = pd.concat([mom_m, strat_m], axis=1, keys=["m", "c"]).dropna()
    j = j[(j.index >= s0) & (j.index < s1)]
    if len(j) < 24:
        return np.nan, np.nan, np.nan
    sc = j["m"].std() / j["c"].std() if j["c"].std() > 0 else 0
    b = 0.5 * j["m"] + 0.5 * sc * j["c"]
    sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
    return sh(j["m"]), sh(b), j["m"].corr(j["c"])


def report(books, etf_books, mom_m, mom_lab):
    print(f"\n  benchmark for (c): {mom_lab}")
    print(f"\n  {'strategy':<8} | {'2007-2021: ann':>14} {'sharpe':>6} | "
          f"{'last 5y: ann':>12} {'vol':>5} {'sharpe':>6} {'worst':>6} "
          f"{'2022':>6} | {'corr':>5} {'mom alone':>9} {'blend':>6} | "
          f"{'ETF long-only 5y':>16}")
    verdict = {}
    for k, d in books.items():
        a0, _, s0, _ = stats(d[(d.index >= CTX0) & (d.index < RECENT)])
        a1, v1, s1, dd1 = stats(d[(d.index >= RECENT) & (d.index < DEC1)])
        y22 = (1 + d[d.index.year == 2022]).prod() - 1
        msh, bsh, corr = blend_sharpe(mom_m, monthly(d), RECENT, DEC1)
        e = etf_books.get(k)
        es = stats(e[(e.index >= RECENT) & (e.index < DEC1)]) \
            if e is not None else (np.nan,) * 4
        print(f"  {k:<8} | {a0*100:>+13.1f}% {s0:>6.2f} | {a1*100:>+11.1f}% "
              f"{v1*100:>4.0f}% {s1:>6.2f} {dd1*100:>5.0f}% {y22*100:>+5.0f}% "
              f"| {corr:>+5.2f} {msh:>9.2f} {bsh:>6.2f} | "
              f"{es[0]*100:>+7.1f}% sh {es[2]:>4.2f}")
        verdict[k] = (s1 >= 0.5) and (s0 >= 0.3) and (bsh > msh)
    return verdict


# ------------------------------------------------------------------ smoke
def smoke():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2000-01-03", "2026-10-23")
    n = len(idx)
    F, E = {}, {}
    for j, m in enumerate(MKT):
        mu = np.zeros(n)
        if j % 2 == 0:                              # planted slow trends
            regime = np.sign(np.sin(np.arange(n) / 400 + j))
            mu = regime * 0.0012
        r = mu + rng.normal(0, 0.015, n)
        F[m] = 100 * np.exp(np.cumsum(r))
        if MKT[m][2]:
            E[m] = F[m] * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    F = pd.DataFrame(F, idx)
    E = pd.DataFrame(E, idx)
    wk = pd.date_range("2006-01-03", "2026-10-20", freq="W-TUE")
    cot = pd.DataFrame(rng.normal(0, 0.1, (len(wk), len(MKT))), wk,
                       list(MKT))
    return F, E, cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        F, E, cot = smoke()
        print("  smoke: half the markets carry planted multi-month trends "
              "(TREND should pass); COT/SEASON/CARRY/REPORTS are noise")
    else:
        F, E = load_prices()
        try:
            cot = load_cot()
        except Exception as e:  # noqa: BLE001
            print(f"  CFTC download failed ({e}); COT skipped")
            cot = None
    R = F.pct_change(fill_method=None).clip(-0.25, 0.25)
    vol = (R.rolling(60, min_periods=40).std() * np.sqrt(252)).shift(0)
    sigs = {"TREND": sig_trend(F, R, vol), "CARRY": sig_carry(F, E, vol),
            "COT": sig_cot(F, cot, vol), "SEASON": sig_season(F, vol),
            "REPORTS": sig_reports(F, R, vol)}
    books, etf_books = {}, {}
    for k, W in sigs.items():
        if W is None or W.empty:
            print(f"  {k}: no signal (data missing)")
            continue
        books[k] = run_book(R[W.columns], W, COST)
        e = etf_long_only(W, E, F)
        if e is not None:
            etf_books[k] = e
    if a.smoke:
        mom_m = monthly(pd.Series(np.random.default_rng(9).normal(
            0.0005, 0.01, len(F)), F.index))
        mom_lab = "random stand-in (smoke)"
    else:
        mom_m, mom_lab = stock_momentum()
    verdict = report(books, etf_books, mom_m, mom_lab)
    print("\n  ===== VERDICT (pre-stated: last-5y Sharpe >= 0.5, 2007-21 "
          "Sharpe >= 0.3, improves momentum blend) =====")
    for k in STRATS:
        if k in verdict:
            print(f"  {k:<8} {'PASS' if verdict[k] else 'FAIL'}")
    if not a.smoke:
        out = pd.DataFrame(books)
        out.to_csv(os.path.join(ROOT, "data", "commodities_daily.csv"))


if __name__ == "__main__":
    main()
