"""
futures_mr_orb.py -- Primordial screen: futures MEAN REVERSION (daily spreads)
and ORB / intraday breakout (hourly bars). Free data only. QuantConnect comes
after something holds up here.

==================== PART 1: MEAN REVERSION (daily, 2009 -> now) ============
Each spread is a price gap the physical economy pulls back together. Signal is
the same for all four (fixed before any real-data run):
  z = (spread - its 60-day average) / its 60-day std
  enter when |z| >= 1.5 (bet it shrinks back), exit at |z| <= 0.5 or after
  20 days; after a 20-day timeout wait for |z| <= 0.5 before re-entering.
  Size: 10% / the spread's own 60-day volatility. Decided at the close,
  earned the next day.
  FLY         Treasury 2s5s10s butterfly = 2*5y - 2y - 10y yield (FRED,
              clean daily). Trading it = long 5y vs short 2y+10y, sized so a
              parallel rate move doesn't matter. Cost 0.5bp of fly per trade.
  CRACK       gasoline vs crude: UGA vs USO (futures ETFs, so no roll jumps);
              hedge ratio = 120-day beta, lagged a day. Cost 5bp per leg.
  CRUSH       soybean board crush: meal + oil vs beans, Yahoo ZM ZL ZS front
              futures, weights = each product's value per bushel crushed.
              Yahoo front-month prices jump at contract rolls, so any day a
              leg moves > 4x its 60-day std counts as a possible roll and is
              dropped. ROUGH -- QC re-tests on clean futures. Cost 2bp per leg.
  CORN_WHEAT  WEAT vs CORN (futures ETFs), as CRACK.
  MR_BOOK     the four, equal risk.
  SPREAD_CONF combined idea: CRACK and CORN_WHEAT, but only fade a spread
              when the futures curves agree it's temporary: e.g. gasoline
              rich vs crude is faded only if gasoline's curve is NOT tighter
              than crude's (carry = ETF 12m return - front futures 12m move).
  control RANDOM  random long/short/flat on CRACK, redrawn every 20 days.
PASS (all): Sharpe >= 0.7 last five years (2021-11 ->), >= 0.5 over
  2012-07 -> 2021-10, and a 50/50 equal-risk blend with CHIMERA's monthly
  returns (2017-02 -> 2026-05) beats CHIMERA alone (1.42).

==================== PART 2: ORB / INTRADAY (hourly bars) ===================
Yahoo only gives ~730 days of hourly futures bars (5-minute: 60 days). Bars
are cached in data/intraday/ and APPENDED each run, so history grows if you
re-run every few months. Markets (flat every night, never ES/NQ):
  CL NG GC SI HG ZN ZB 6E 6J 6B. Session window, New York time:
  opening range = the 9:00-10:00 bar; trading 10:00 -> exit 15:00.
  Costs: 1 tick slippage + $2.50 commission per side, every trade.
  ORB         first break of the opening range high/low -> enter there
              (stop order); stop at the other side; else exit 15:00. A bar
              that breaks both sides at once = a full stopped-out loss (the
              second break stops the first; skipping it would be look-ahead).
  NR7_ORB     ORB only on days after the narrowest session range of 7 days.
  INTRA_MOM   prior 15:00 close -> today 14:00 close sets the direction of the
              14:00-15:00 bar (Baltussen et al. 2021, across futures).
  NOISE_BAND  long above open*(1+band), short below open*(1-band), band =
              14-day average |move from open| at that hour; checked each hour
              close (Zarattini, Aziz & Barbon 2024, moved onto futures).
  ORB_COMBO   combined idea: ORB only on NR7 days AND only in the direction
              the first hour closed; first break the wrong way = no trade.
  control RANDOM  half the days, random direction 10:00 -> 15:00.
  Each market's day return x 10% / its 60-day daily vol, averaged.
PROMISING (not PASS -- two years is too short to pass): Sharpe >= 1.4 over
  the whole sample (~t-stat 2 on two years) AND positive in both halves.
  PROMISING earns a QC minute-data test back to ~2010.

Run (repo root):
    python3.10 scripts/futures_mr_orb.py --smoke    # planted + null worlds
    python3.10 scripts/futures_mr_orb.py | tee logs/futures_mr_orb.txt
    (--daily / --intraday to run one half)
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
INTRA_DIR = os.path.join(DATA, "intraday")
START, END = "2009-01-01", "2026-11-05"
EARLY = (pd.Timestamp("2012-07-01"), pd.Timestamp("2021-11-01"))
RECENT = (pd.Timestamp("2021-11-01"), pd.Timestamp("2099-01-01"))
TGT, COST_ETF, COST_FUT, COST_FLY = 0.10, 0.0005, 0.0002, 0.005
Z_WIN, H_WIN, Z_IN, Z_OUT, MAX_HOLD = 60, 120, 1.5, 0.5, 20
COMM = 2.50
UA = {"User-Agent": "Mozilla/5.0"}
# intraday: symbol -> (tick, contract multiplier)
INTRA = {"CL=F": (0.01, 1000), "NG=F": (0.001, 10000), "GC=F": (0.10, 100),
         "SI=F": (0.005, 5000), "HG=F": (0.0005, 25000), "ZN=F": (1 / 64, 1000),
         "ZB=F": (1 / 32, 1000), "6E=F": (0.00005, 125000),
         "6J=F": (0.0000005, 12500000), "6B=F": (0.0001, 62500)}
DAILY_SYMS = ["UGA", "USO", "CORN", "WEAT", "BIL", "RB=F", "CL=F", "ZW=F", "ZC=F",
              "ZS=F", "ZM=F", "ZL=F"]
CHIM = {"2017-02": 0.05638, "2017-03": -0.00502, "2017-04": 0.0516, "2017-05": 0.05944, "2017-06": -0.04716, "2017-07": 0.08036, "2017-08": 0.04121, "2017-09": -0.0082, "2017-10": 0.01275, "2017-11": 0.06163, "2017-12": 0.00984, "2018-01": 0.05409, "2018-02": -0.04113, "2018-03": -0.00773, "2018-04": -0.00085, "2018-05": 0.02454, "2018-06": -0.02166, "2018-07": -0.03758, "2018-08": 0.04259, "2018-09": 0.01471, "2018-10": -0.09646, "2018-11": -0.00262, "2018-12": -0.03574, "2019-01": 0.05253, "2019-02": 0.02498, "2019-03": 0.01921, "2019-04": -0.01715, "2019-05": -0.0174, "2019-06": 0.06728, "2019-07": 0.04436, "2019-08": 0.01406, "2019-09": -0.03595, "2019-10": 0.02579, "2019-11": 0.03318, "2019-12": 0.08462, "2020-01": 0.00542, "2020-02": -0.0347, "2020-03": -0.03995, "2020-04": 0.10273, "2020-05": 0.02383, "2020-06": 0.08567, "2020-07": 0.04235, "2020-08": -0.0517, "2020-09": -0.00151, "2020-10": -0.00627, "2020-11": 0.17142, "2020-12": 0.03592, "2021-01": 0.20194, "2021-02": 0.1025, "2021-03": 0.05749, "2021-04": 0.04693, "2021-05": 0.0169, "2021-06": 0.00755, "2021-07": 0.01409, "2021-08": 0.02306, "2021-09": -0.03898, "2021-10": 0.06648, "2021-11": -0.02027, "2021-12": 0.05834, "2022-01": -0.13833, "2022-02": 0.03776, "2022-03": 0.07138, "2022-04": 0.04416, "2022-05": 0.01971, "2022-06": -0.03329, "2022-07": 0.01483, "2022-08": 0.04692, "2022-09": 0.00549, "2022-10": 0.01113, "2022-11": -0.07182, "2022-12": -0.03874, "2023-01": -0.00446, "2023-02": 0.00744, "2023-03": -0.01724, "2023-04": 0.00149, "2023-05": 0.04634, "2023-06": -0.00555, "2023-07": 0.01982, "2023-08": -0.02555, "2023-09": -0.03562, "2023-10": -0.05146, "2023-11": 0.03769, "2023-12": 0.07371, "2024-01": 0.04383, "2024-02": 0.10113, "2024-03": 0.02381, "2024-04": -0.00117, "2024-05": 0.09167, "2024-06": 0.00586, "2024-07": 0.03194, "2024-08": 0.02881, "2024-09": 0.02354, "2024-10": -0.01373, "2024-11": 0.05931, "2024-12": -0.05531, "2025-01": 0.04376, "2025-02": -0.03464, "2025-03": 0.01805, "2025-04": 0.01726, "2025-05": 0.0183, "2025-06": 0.02993, "2025-07": 0.00038, "2025-08": 0.00485, "2025-09": 0.17491, "2025-10": -0.00345, "2025-11": 0.01746, "2025-12": -0.00032, "2026-01": 0.11679, "2026-02": 0.11839, "2026-03": 0.04071, "2026-04": 0.05341, "2026-05": 0.00883}


# ============================================================ shared maths
def stats(x):
    x = x.dropna()
    if len(x) < 120 or x.std() == 0:
        return (np.nan,) * 4
    eq = (1 + x).cumprod()
    vol = x.std() * np.sqrt(252)
    return eq.iloc[-1] ** (252 / len(x)) - 1, vol, x.mean() * 252 / vol, \
        (eq / eq.cummax() - 1).min()


def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 20 and x.std() > 0 else np.nan


def win(x, w):
    return x[(x.index >= w[0]) & (x.index < w[1])]


def blend(daily):
    m = (1 + daily).resample("ME").prod() - 1
    m.index = m.index.strftime("%Y-%m")
    j = pd.concat([pd.Series(CHIM), m], axis=1, keys=["c", "s"]).dropna()
    if len(j) < 12 or j["s"].std() == 0:
        return np.nan, np.nan, np.nan, len(j)
    sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
    b = 0.5 * j["c"] + 0.5 * j["s"] * (j["c"].std() / j["s"].std())
    return sh(j["c"]), sh(b), j["c"].corr(j["s"]), len(j)


def eq_risk(*xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    df = pd.concat(xs, axis=1).fillna(0.0)
    sd = df.std().replace(0, np.nan)
    return (df / sd).mean(axis=1) * sd.mean()


# ===================================================== PART 1: daily data
def load_daily():
    from primordial.data import fetch
    d = fetch("yfinance", DAILY_SYMS, "1d", START, END, min_bars=200)
    px = pd.DataFrame({s: d[s]["close"] for s in DAILY_SYMS if s in d})
    px.index = pd.to_datetime(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    miss = [s for s in DAILY_SYMS if s not in px]
    print(f"  daily prices: {px.shape[1]} series {px.index[0].date()} -> "
          f"{px.index[-1].date()}" + (f"  MISSING {miss}" if miss else ""))
    return px


def load_fred():
    import requests
    os.makedirs(DATA, exist_ok=True)
    out = {}
    for s in ("DGS2", "DGS5", "DGS10"):
        p = os.path.join(DATA, f"fred_{s}.csv")
        if not os.path.exists(p) or time.time() - os.path.getmtime(p) > 86400:
            for k in range(4):                 # FRED is slow sometimes: retry
                try:
                    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv"
                                     f"?id={s}&cosd=2008-01-01", headers=UA, timeout=180)
                    r.raise_for_status()
                    break
                except Exception:  # noqa: BLE001
                    if k == 3:
                        raise
                    time.sleep(10)
            open(p, "w").write(r.text)
        d = pd.read_csv(p)
        out[s] = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values,
                           pd.to_datetime(d.iloc[:, 0]))
    f = pd.DataFrame(out).dropna()
    f = f[f.index >= START]
    print(f"  FRED yields: {f.index[0].date()} -> {f.index[-1].date()}")
    return f


# ================================================ PART 1: spread machinery
def hysteresis(z, allow_long=None, allow_short=None):
    pos, p, held, cool = np.zeros(len(z)), 0, 0, False
    for i, v in enumerate(z):
        if not np.isfinite(v):
            p, held, cool = 0, 0, False
            continue
        if cool and abs(v) <= Z_OUT:
            cool = False
        if p != 0:
            held += 1
            if abs(v) <= Z_OUT or np.sign(v) == p:
                p, held = 0, 0
            elif held >= MAX_HOLD:
                p, held, cool = 0, 0, True
        if p == 0 and not cool:
            if v >= Z_IN and (allow_short is None or allow_short[i]):
                p, held = -1, 0
            elif v <= -Z_IN and (allow_long is None or allow_long[i]):
                p, held = 1, 0
        pos[i] = p
    return pos


def mr(dL, cost, allow_long=None, allow_short=None):
    """dL = daily change of the spread (a return for pairs, yield for FLY)."""
    dL = dL.fillna(0.0)
    L = dL.cumsum()
    z = (L - L.rolling(Z_WIN, min_periods=40).mean()) / \
        L.rolling(Z_WIN, min_periods=40).std()
    al = None if allow_long is None else allow_long.reindex(dL.index).fillna(False).values
    ash = None if allow_short is None else allow_short.reindex(dL.index).fillna(False).values
    pos = pd.Series(hysteresis(z.values, al, ash), dL.index)
    vol = dL.replace(0, np.nan).rolling(Z_WIN, min_periods=40).std() * np.sqrt(252)
    w = (pos * (TGT / vol)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return w.shift(1).fillna(0.0) * dL - w.diff().abs().fillna(0.0) * cost


def pair(rA, rB):
    h = (rA.rolling(H_WIN, min_periods=80).cov(rB) /
         rB.rolling(H_WIN, min_periods=80).var()).shift(1)
    return rA - h * rB, h


def build_daily(px, fred):
    bil = px["BIL"].pct_change(fill_method=None).fillna(0.0) if "BIL" in px else 0.0
    ex = lambda t: px[t].pct_change(fill_method=None).clip(-0.25, 0.25) - bil  # noqa
    car = lambda e, f: px[e] / px[e].shift(252) - px[f] / px[f].shift(252)  # noqa
    D = {}
    if fred is not None and len(fred):
        D["fly"] = (2 * fred["DGS5"] - fred["DGS2"] - fred["DGS10"]).diff()
    for name, a, b, fa, fb in (("crack", "UGA", "USO", "RB=F", "CL=F"),
                               ("grain", "WEAT", "CORN", "ZW=F", "ZC=F")):
        if a in px and b in px:
            rs, h = pair(ex(a), ex(b))
            D[name] = (rs, h)
            if fa in px and fb in px:
                D[name + "_carry"] = car(a, fa) - car(b, fb)
    if all(s in px for s in ("ZS=F", "ZM=F", "ZL=F")):
        r = {s: np.log(px[s]).diff() for s in ("ZS=F", "ZM=F", "ZL=F")}
        roll = pd.concat([(r[s].abs() > 4 * r[s].rolling(60, min_periods=40).std()
                           .shift(1)) for s in r], axis=1).any(axis=1)
        bean = px["ZS=F"] / 100
        wm, wo = (0.022 * px["ZM=F"] / bean).shift(1), (0.11 * px["ZL=F"] / bean).shift(1)
        rs = (wm * r["ZM=F"] + wo * r["ZL=F"] - r["ZS=F"]).where(~roll, 0.0)
        D["crush"] = (rs, wm + wo)
        print(f"  CRUSH: {int(roll.sum())} possible roll days dropped")
    return D


def daily_strats(D):
    S = {}
    S["FLY"] = mr(D["fly"], COST_FLY) if "fly" in D else None
    S["CRACK"] = mr(D["crack"][0], COST_ETF * (1 + D["crack"][1].abs())) \
        if "crack" in D else None
    S["CRUSH"] = mr(D["crush"][0], COST_FUT * (1 + D["crush"][1])) \
        if "crush" in D else None
    S["CORN_WHEAT"] = mr(D["grain"][0], COST_ETF * (1 + D["grain"][1].abs())) \
        if "grain" in D else None
    S["MR_BOOK"] = eq_risk(S["FLY"], S["CRACK"], S["CRUSH"], S["CORN_WHEAT"])
    conf = []
    for k in ("crack", "grain"):
        if k in D and k + "_carry" in D:
            dc = D[k + "_carry"]
            # long spread (A cheap) only if A's curve is at least as tight as B's;
            # short spread (A rich) only if A's curve is NOT tighter than B's
            conf.append(mr(D[k][0], COST_ETF * (1 + D[k][1].abs()),
                           allow_long=dc >= 0, allow_short=dc <= 0))
    S["SPREAD_CONF"] = eq_risk(*conf) if conf else None
    if "crack" in D:
        rs = D["crack"][0].fillna(0.0)
        rng = np.random.default_rng(7)
        blocks = np.repeat(rng.choice([-1.0, 0.0, 1.0], len(rs) // 20 + 1), 20)[:len(rs)]
        vol = rs.replace(0, np.nan).rolling(60, min_periods=40).std() * np.sqrt(252)
        w = (pd.Series(blocks, rs.index) * TGT / vol).replace([np.inf, -np.inf],
                                                              np.nan).fillna(0.0)
        S["control RANDOM"] = w.shift(1).fillna(0.0) * rs - w.diff().abs().fillna(0.0) \
            * COST_ETF * (1 + D["crack"][1].abs().fillna(1.0))
    return S


def report_daily(S):
    print(f"\n  PART 1 MEAN REVERSION (daily)\n  {'strategy':<15}| 2012-21 ann  shp |"
          f" last5y ann  vol  shp  worst  2022 | corrCHIM  CHIM  blend | verdict")
    for k, x in S.items():
        if x is None:
            print(f"  {k:<15}| no data")
            continue
        x = x.dropna()
        a0, _, s0, _ = stats(win(x, EARLY))
        a1, v1, s1, dd1 = stats(win(x, RECENT))
        y22 = (1 + x[x.index.year == 2022]).prod() - 1
        csh, bsh, corr, _ = blend(x)
        ok = (s1 >= 0.7) and (s0 >= 0.5) and (bsh > csh)
        print(f"  {k:<15}| {a0*100:+7.1f}% {s0:5.2f} | {a1*100:+7.1f}% {v1*100:4.0f}% "
              f"{s1:5.2f} {dd1*100:5.0f}% {y22*100:+5.0f}% | {corr:+6.2f} {csh:6.2f} "
              f"{bsh:6.2f} | {'PASS' if ok else 'fail'}")
    print("  pass = last-5y Sharpe >= 0.7 AND 2012-21 >= 0.5 AND blend > CHIM")


# ================================================== PART 2: intraday data
def load_hourly(sym):
    import yfinance as yf
    os.makedirs(INTRA_DIR, exist_ok=True)
    p = os.path.join(INTRA_DIR, sym.replace("=", "_") + "_1h.csv")
    old = pd.read_csv(p, index_col=0, parse_dates=True) if os.path.exists(p) else None
    h = yf.download(sym, period="729d", interval="1h", auto_adjust=False,
                    progress=False)
    if h is not None and len(h):
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        h = h.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        idx = h.index if h.index.tz is not None else h.index.tz_localize("UTC")
        h.index = idx.tz_convert("America/New_York").tz_localize(None)
    else:
        h = None
    h = pd.concat([x for x in (old, h) if x is not None])
    h = h[~h.index.duplicated(keep="last")].sort_index()
    h.to_csv(p)
    return h


def sessions(h):
    """-> list of (date, or_bar(o,h,l,c), trade bars ndarray [5 x ohlc] for 10..14)."""
    h = h[(h.index.hour >= 9) & (h.index.hour < 15)]
    out = []
    for d, g in h.groupby(h.index.date):
        hrs = list(g.index.hour)
        if sorted(set(hrs)) != [9, 10, 11, 12, 13, 14] or len(hrs) != 6:
            continue
        g = g.sort_index()
        a = g[["open", "high", "low", "close"]].values.astype(float)
        out.append((pd.Timestamp(d), a[0], a[1:]))
    return out


def orb_trade(ORH, ORL, tr, c, allow):
    o, hi, lo, cl = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]
    pos, entry = 0, np.nan
    for i in range(len(cl)):
        if pos == 0:
            up, dn = hi[i] > ORH, lo[i] < ORL
            if up and dn:
                # both sides hit inside one bar: whichever broke first was then
                # stopped by the other -> a full loss (skipping it = look-ahead)
                if not (allow(1) or allow(-1)):
                    return 0.0, 0
                return (ORL - ORH - 2 * c) / ORH, 1
            if up or dn:
                d = 1 if up else -1
                if not allow(d):
                    return 0.0, 0
                pos = d
                entry = (max(o[i], ORH) + c) if d > 0 else (min(o[i], ORL) - c)
        else:
            if pos > 0 and lo[i] <= ORL:
                return (min(o[i], ORL) - c - entry) / entry, 1
            if pos < 0 and hi[i] >= ORH:
                return (entry - max(o[i], ORH) - c) / entry, 1
    if pos == 0:
        return 0.0, 0
    return pos * (cl[-1] - c * pos - entry) / entry, 1


def per_market(h, tick, mult, seed):
    ses = sessions(h)
    if len(ses) < 60:
        return None
    c = tick + COMM / mult
    rng = np.random.default_rng(seed)
    dates = [s[0] for s in ses]
    rngs = np.array([max(s[1][1], s[2][:, 1].max()) - min(s[1][2], s[2][:, 2].min())
                     for s in ses])
    close = np.array([s[2][-1, 3] for s in ses])
    # moves from open at each hour 9..14, for NOISE_BAND
    mv = np.array([np.abs(np.r_[s[1][3], s[2][:, 3]] / s[1][0] - 1) for s in ses])
    rows = []
    for t, (d, ob, tr) in enumerate(ses):
        ORH, ORL = ob[1], ob[2]
        r = {"date": d}
        r["ORB"], n1 = orb_trade(ORH, ORL, tr, c, lambda x: True)
        nr7 = t >= 7 and rngs[t - 1] <= rngs[t - 7:t].min()
        r["NR7_ORB"], n2 = orb_trade(ORH, ORL, tr, c, lambda x: True) if nr7 else (0.0, 0)
        first = np.sign(ob[3] - ob[0])
        r["ORB_COMBO"], n3 = orb_trade(ORH, ORL, tr, c, lambda x: x == first) \
            if nr7 and first != 0 else (0.0, 0)
        r["INTRA_MOM"], n4 = 0.0, 0
        if t > 0 and (d - dates[t - 1]).days <= 4:
            c13, c14 = tr[3, 3], tr[4, 3]
            sgn = np.sign(c13 / close[t - 1] - 1)
            if sgn != 0:
                r["INTRA_MOM"], n4 = sgn * (c14 / c13 - 1) - 2 * c / c13, 1
        r["NOISE_BAND"], n5 = 0.0, 0
        if t >= 14:
            band = mv[t - 14:t].mean(axis=0)              # hours 9..14
            cls = np.r_[ob[3], tr[:, 3]]
            prev, pnl = 0, 0.0
            for j in range(5):                            # decide at closes 9..13
                x = cls[j] / ob[0] - 1
                tgt = 1 if x > band[j] else (-1 if x < -band[j] else 0)
                pnl += tgt * (cls[j + 1] / cls[j] - 1) - abs(tgt - prev) * c / cls[j]
                n5 += int(tgt != 0 and prev == 0)
                prev = tgt
            pnl -= abs(prev) * c / cls[5]
            r["NOISE_BAND"] = pnl
        r["control RANDOM"], n6 = 0.0, 0
        if rng.random() < 0.5:
            dd = rng.choice([-1, 1])
            r["control RANDOM"], n6 = dd * (tr[-1, 3] / ob[3] - 1) - 2 * c / ob[3], 1
        r.update({"n_ORB": n1, "n_NR7_ORB": n2, "n_ORB_COMBO": n3, "n_INTRA_MOM": n4,
                  "n_NOISE_BAND": n5, "n_control RANDOM": n6})
        rows.append(r)
    df = pd.DataFrame(rows).set_index("date")
    dret = pd.Series(close, dates).pct_change()
    vol = (dret.rolling(60, min_periods=20).std() * np.sqrt(252)).shift(1)
    df["size"] = (TGT / vol).clip(upper=50)
    return df


INTRA_ROWS = ["ORB", "NR7_ORB", "INTRA_MOM", "NOISE_BAND", "ORB_COMBO", "control RANDOM"]


def report_intraday(M):
    M = {k: v for k, v in M.items() if v is not None}
    if not M:
        print("  no intraday data")
        return
    print(f"\n  PART 2 ORB / INTRADAY (hourly)  markets: {', '.join(M)}")
    first = min(v.index[0] for v in M.values()).date()
    last = max(v.index[-1] for v in M.values()).date()
    print(f"  sample {first} -> {last}")
    print(f"  {'strategy':<15}| trades  win  net/trade | ann    shp  half1 half2  worst |"
          f" corrCHIM(mo) | verdict")
    for k in INTRA_ROWS:
        book = pd.concat({m: v[k] * v["size"] for m, v in M.items()}, axis=1)
        book = book.mean(axis=1).dropna()
        tr = pd.concat([v.loc[v["n_" + k] > 0, k] for v in M.values()])
        ntr = int(sum(v["n_" + k].sum() for v in M.values()))
        a, _, s, dd = stats(book)
        mid = len(book) // 2
        h1, h2 = sharpe(book.iloc[:mid]), sharpe(book.iloc[mid:])
        _, _, corr, nm = blend(book)
        ok = (s >= 1.4) and (h1 > 0) and (h2 > 0)
        print(f"  {k:<15}| {ntr:6d} {(tr > 0).mean()*100:3.0f}% {tr.mean()*1e4:+7.1f}bp |"
              f" {a*100:+5.1f}% {s:5.2f} {h1:5.2f} {h2:5.2f} {dd*100:5.0f}% |"
              f" {corr:+5.2f} ({nm:2d}mo) | {'PROMISING' if ok else 'fail'}")
    print("  promising = Sharpe >= 1.4 AND both halves > 0 -> earns a QC minute test")
    print("  net/trade = average trade after 1 tick + $2.50 each side, unscaled")


# =================================================================== smoke
def smoke_daily(k):
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2009-01-02", "2026-10-30")
    n = len(idx)

    def ou(sd):
        s = np.zeros(n)
        for i in range(1, n):
            s[i] = s[i - 1] * (1 - k) + rng.normal(0, sd)
        return s
    rw = lambda sd, n0=0.0: n0 + np.cumsum(rng.normal(0, sd, n))  # noqa: E731
    px = pd.DataFrame(index=idx)
    px["BIL"] = np.exp(rw(0.0001))
    for a, b in (("UGA", "USO"), ("WEAT", "CORN")):
        lb = rw(0.02, 4)
        px[b] = np.exp(lb)
        px[a] = np.exp(lb + ou(0.01) + rw(0.004))
    for f in ("RB=F", "CL=F", "ZW=F", "ZC=F"):
        px[f] = np.exp(rw(0.02, 4))
    lb = rw(0.015, np.log(1300))
    s = ou(0.01)
    px["ZS=F"] = np.exp(lb)
    px["ZM=F"] = np.exp(lb - np.log(1300) + np.log(400) + s + rw(0.003))
    px["ZL=F"] = np.exp(lb - np.log(1300) + np.log(60) + s + rw(0.003))
    y2, y10 = rw(0.05, 2), rw(0.06, 3)
    f = ou(0.03) + rw(0.01)
    fred = pd.DataFrame({"DGS2": y2, "DGS5": (y2 + y10 + f) / 2, "DGS10": y10}, idx)
    return px, fred


def smoke_intraday(mu):
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2024-11-01", periods=500)
    M = {}
    for j, (sym, (tick, mult)) in enumerate(INTRA.items()):
        px0 = 100.0
        sd = 0.004
        rows, p = [], px0
        for d in days:
            dr = rng.choice([-1, 1])
            for hr in (2, 9, 10, 11, 12, 13, 14, 16):    # 2/16 = overnight, ignored
                # a fine price path (15s steps), so highs/lows are true path extremes
            # and fills at the breakout level are realistic
                m = dr * mu * sd / 240 if 9 <= hr <= 14 else 0.0
                path = p * np.exp(np.cumsum(rng.normal(m, sd / np.sqrt(240), 240)))
                o, p = path[0], path[-1]
                rows.append((d + pd.Timedelta(hours=hr), o, path.max(), path.min(), p))
        h = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close"]).set_index("t")
        M[sym] = per_market(h, 0.01, 1e6, j)    # 1 tick = 1bp at price 100
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--daily", action="store_true")
    ap.add_argument("--intraday", action="store_true")
    a = ap.parse_args()
    both = not (a.daily or a.intraday)
    if a.smoke:
        for label, k, mu in (("PLANTED (all but RANDOM should pass/promise)", 0.08, 0.35),
                             ("NULL (everything should fail)", 0.0, 0.0)):
            print(f"\n================ SMOKE {label}")
            if both or a.daily:
                px, fred = smoke_daily(k)
                report_daily(daily_strats(build_daily(px, fred)))
            if both or a.intraday:
                report_intraday(smoke_intraday(mu))
        return
    if both or a.daily:
        px = load_daily()
        try:
            fred = load_fred()
        except Exception as e:  # noqa: BLE001
            print(f"  FRED failed ({e}); FLY skipped")
            fred = None
        report_daily(daily_strats(build_daily(px, fred)))
    if both or a.intraday:
        M = {}
        for j, (sym, (tick, mult)) in enumerate(INTRA.items()):
            try:
                h = load_hourly(sym)
                M[sym] = per_market(h, tick, mult, j)
                print(f"  {sym}: {len(h)} hourly bars {h.index[0].date()} -> "
                      f"{h.index[-1].date()}, {0 if M[sym] is None else len(M[sym])} "
                      f"full sessions")
            except Exception as e:  # noqa: BLE001
                print(f"  {sym}: failed ({e})")
        report_intraday(M)


if __name__ == "__main__":
    main()
