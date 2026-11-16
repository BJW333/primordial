"""
fx_screen.py -- FX ideas on hourly data. PRE-REGISTERED 2026-11-10, before any real-data run.
Blake noticed AUD and NZD move in unison; the trade is the GAP between linked
currencies, plus the home-hours pattern. Rules below are fixed -- no tuning after results.

Currencies (all as USD per unit): AUD NZD CAD NOK EUR GBP JPY CHF
Data: Dukascopy free hourly candles 2004 -> 2026-10 (downloaded once, cached in data/fx_1h/).
Costs (per side, ~1 tick + commission on CME futures, bps): EUR .7 GBP .9 CHF .7 JPY .9
      AUD .9 NZD 1.0 CAD .8 NOK 3.0.  Every row is also run at 2x costs.
Daily clock: signal at the 16:00 New York close, trade at the 17:00 New York close (1 hour later).

  1 PAIR      AUD/NZD gap (log AUD - log NZD), z = (gap - 60d avg) / 60d std.
              z > 2 -> short the gap, z < -2 -> long. Exit when z crosses 0, stop when |z| > 4,
              max 15 days. $1 long one leg, $1 short the other.
  2 NEWS      Same as PAIR, but only enter if NO Reserve Bank of Australia or of New Zealand
              decision fell in the last 5 trading days (today included).
              Passes only if it also beats NEWS_DAY (the same rule entering ONLY after a decision).
              RBA: first Tuesday of each month except January 2004-2023, published dates 2024-26.
              RBNZ: announcement dates scraped from rbnz.govt.nz (cached in data/rbnz_ocr_dates.csv);
              if they can't be loaded, NEWS is NOT RUN (no substitute calendar).
  3 BLOC      AUD NZD CAD NOK. Each currency's gap to the 4-currency average, z-scored over 60d.
              Daily weights = -(z - avg z), scaled to $1 long / $1 short; rebalanced daily.
  4 CATCHUP   Hourly. AUD-minus-NZD move in one hour > 2 std (std of the last 480 hours) ->
              bet the gap closes (NZD catches up) for the next 3 hours.
  5 HOME      Hourly. Short EUR GBP CHF NOK 07:00-12:00 London and short JPY AUD NZD 09:00-15:00
              Tokyo (their home hours); long all seven 12:00-16:00 New York (after London closes).
              1/7 each. CAD left out (its home hours are US hours).
Controls (must FAIL): RANDOM (random AUD/NZD entries, 5-day hold), CATCHUP_SHUF (same signals
              shuffled in time), HOME_SHIFT6 (HOME with every window moved 6 hours later).
Pass bar (same as every daily idea): Sharpe >= 0.5 2012-07 -> 2021-10, >= 0.7 2021-11 -> end,
  50/50 equal-risk monthly blend with CHIMERA > CHIMERA's 1.42, AND whole-period Sharpe > 0 at 2x costs.

Run (repo root):
    python3.10 scripts/fx_screen.py --smoke        # synthetic checks (planted rows PASS, noise rows fail)
    python3.10 scripts/fx_screen.py | tee logs/fx_screen.txt
    python3.10 scripts/fx_screen.py --quick     # only AUD NZD CAD NOK -> ideas 1-4 (HOME needs all 8)
First real run downloads ~2,200 small files (10-30 min; Dukascopy throttles). If it stops with a
throttling message, wait 10-15 min and rerun: it resumes from where it stopped.
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import datetime as dt
import lzma
import os
import re
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "fx_1h")
RBNZ_CSV = os.path.join(ROOT, "data", "rbnz_ocr_dates.csv")

INST = {"AUD": ("AUDUSD", False), "NZD": ("NZDUSD", False), "CAD": ("USDCAD", True),
        "NOK": ("USDNOK", True), "EUR": ("EURUSD", False), "GBP": ("GBPUSD", False),
        "JPY": ("USDJPY", True), "CHF": ("USDCHF", True)}
COST_BP = {"EUR": 0.7, "GBP": 0.9, "CHF": 0.7, "JPY": 0.9, "AUD": 0.9, "NZD": 1.0,
           "CAD": 0.8, "NOK": 3.0}
START, END = "2004-01-01", "2026-11-01"
EARLY = ("2012-07-01", "2021-11-01")
RECENT = ("2021-11-01", "2100-01-01")
BLOC = ["AUD", "NZD", "CAD", "NOK"]
EU, ASIA = ["EUR", "GBP", "CHF", "NOK"], ["JPY", "AUD", "NZD"]

CHIM = {"2017-02": 0.05638, "2017-03": -0.00502, "2017-04": 0.0516, "2017-05": 0.05944, "2017-06": -0.04716, "2017-07": 0.08036, "2017-08": 0.04121, "2017-09": -0.0082, "2017-10": 0.01275, "2017-11": 0.06163, "2017-12": 0.00984, "2018-01": 0.05409, "2018-02": -0.04113, "2018-03": -0.00773, "2018-04": -0.00085, "2018-05": 0.02454, "2018-06": -0.02166, "2018-07": -0.03758, "2018-08": 0.04259, "2018-09": 0.01471, "2018-10": -0.09646, "2018-11": -0.00262, "2018-12": -0.03574, "2019-01": 0.05253, "2019-02": 0.02498, "2019-03": 0.01921, "2019-04": -0.01715, "2019-05": -0.0174, "2019-06": 0.06728, "2019-07": 0.04436, "2019-08": 0.01406, "2019-09": -0.03595, "2019-10": 0.02579, "2019-11": 0.03318, "2019-12": 0.08462, "2020-01": 0.00542, "2020-02": -0.0347, "2020-03": -0.03995, "2020-04": 0.10273, "2020-05": 0.02383, "2020-06": 0.08567, "2020-07": 0.04235, "2020-08": -0.0517, "2020-09": -0.00151, "2020-10": -0.00627, "2020-11": 0.17142, "2020-12": 0.03592, "2021-01": 0.20194, "2021-02": 0.1025, "2021-03": 0.05749, "2021-04": 0.04693, "2021-05": 0.0169, "2021-06": 0.00755, "2021-07": 0.01409, "2021-08": 0.02306, "2021-09": -0.03898, "2021-10": 0.06648, "2021-11": -0.02027, "2021-12": 0.05834, "2022-01": -0.13833, "2022-02": 0.03776, "2022-03": 0.07138, "2022-04": 0.04416, "2022-05": 0.01971, "2022-06": -0.03329, "2022-07": 0.01483, "2022-08": 0.04692, "2022-09": 0.00549, "2022-10": 0.01113, "2022-11": -0.07182, "2022-12": -0.03874, "2023-01": -0.00446, "2023-02": 0.00744, "2023-03": -0.01724, "2023-04": 0.00149, "2023-05": 0.04634, "2023-06": -0.00555, "2023-07": 0.01982, "2023-08": -0.02555, "2023-09": -0.03562, "2023-10": -0.05146, "2023-11": 0.03769, "2023-12": 0.07371, "2024-01": 0.04383, "2024-02": 0.10113, "2024-03": 0.02381, "2024-04": -0.00117, "2024-05": 0.09167, "2024-06": 0.00586, "2024-07": 0.03194, "2024-08": 0.02881, "2024-09": 0.02354, "2024-10": -0.01373, "2024-11": 0.05931, "2024-12": -0.05531, "2025-01": 0.04376, "2025-02": -0.03464, "2025-03": 0.01805, "2025-04": 0.01726, "2025-05": 0.0183, "2025-06": 0.02993, "2025-07": 0.00038, "2025-08": 0.00485, "2025-09": 0.17491, "2025-10": -0.00345, "2025-11": 0.01746, "2025-12": -0.00032, "2026-01": 0.11679, "2026-02": 0.11839, "2026-03": 0.04071, "2026-04": 0.05341, "2026-05": 0.00883}


# =================================================================== data
REC = np.dtype([("t", ">u4"), ("a", ">u4"), ("b", ">u4"), ("c", ">u4"), ("d", ">u4"),
                ("v", ">f4")])


RAW = os.path.join(DATA, "raw")      # one small file per instrument-month -> restarts resume
PAUSE = 0.5                          # seconds between requests (Dukascopy throttles fast clients)


class Throttled(Exception):
    pass


def fetch_month(sess, inst, y, m, tries=8):
    url = f"https://datafeed.dukascopy.com/datafeed/{inst}/{y}/{m - 1:02d}/BID_candles_hour_1.bi5"
    for k in range(tries):
        try:
            r = sess.get(url, timeout=30)
        except Exception:  # noqa: BLE001  (timeouts, dropped connections)
            r = None
        if r is not None and r.status_code == 404:
            return None
        if r is not None and r.status_code == 200:
            if not r.content:
                return None
            raw = lzma.decompress(r.content)
            rec = np.frombuffer(raw[:len(raw) // 24 * 24], dtype=REC)
            if not len(rec):
                return None
            t0 = pd.Timestamp(y, m, 1)
            return pd.DataFrame({"a": rec["a"], "b": rec["b"], "c": rec["c"], "d": rec["d"],
                                 "volume": rec["v"].astype(float)},
                                index=t0 + pd.to_timedelta(rec["t"].astype(np.int64), unit="s"))
        wait = min(10 * 2 ** k, 180)
        code = r.status_code if r is not None else "no answer"
        print(f"      Dukascopy slowing us down ({code}) on {inst} {y}-{m:02d}, "
              f"waiting {wait}s (try {k + 1}/{tries})", flush=True)
        time.sleep(wait)
    raise Throttled(url)


def ohlc_order(df):
    """Dukascopy candle field order; picked by the one that respects low <= open,close <= high."""
    for o, c, lo, hi in [("a", "b", "c", "d"), ("a", "d", "c", "b"), ("a", "b", "d", "c")]:
        ok = (df[lo] <= df[[o, c]].min(axis=1)) & (df[hi] >= df[[o, c]].max(axis=1))
        if ok.mean() > 0.99:
            return o, c, lo, hi
    raise SystemExit("  could not work out the candle field order -- tell Claude")


def session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    return s


def get_instrument(sess, inst):
    """All months for one instrument, using/creating the month cache. Raises Throttled."""
    d = os.path.join(RAW, inst)
    os.makedirs(d, exist_ok=True)
    frames = []
    for y in range(int(START[:4]), int(END[:4]) + 1):
        n = 0
        for m in range(1, 13):
            if pd.Timestamp(y, m, 1) >= pd.Timestamp(END):
                break
            f = os.path.join(d, f"{y}_{m:02d}.csv")
            if os.path.exists(f + ".empty"):
                continue
            if os.path.exists(f):
                df = pd.read_csv(f, index_col=0, parse_dates=True)
            else:
                time.sleep(PAUSE)
                df = fetch_month(sess, inst, y, m)
                if df is None:
                    open(f + ".empty", "w").close()
                    continue
                df.to_csv(f)
            frames.append(df)
            n += len(df)
        print(f"    {inst} {y}: {n:,} hourly bars", flush=True)
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def write_csv(df, inst, path, scale):
    o, c, lo, hi = ohlc_order(df)
    out = pd.DataFrame({"open": df[o] / scale, "high": df[hi] / scale, "low": df[lo] / scale,
                        "close": df[c] / scale, "volume": df["volume"]})
    out.index.name = "time"
    out.to_csv(path)
    print(f"  {inst}: {len(out):,} bars {out.index[0]} -> {out.index[-1]}, "
          f"first close {out.close.iloc[0]:.4f}, last {out.close.iloc[-1]:.4f}", flush=True)


THROTTLE_MSG = ("\n  Dukascopy is throttling downloads from your connection. Everything fetched so far "
                "is saved.\n  Wait 10-15 minutes and run the same command again -- it picks up where "
                "it stopped.")


def download(curs=None):
    os.makedirs(DATA, exist_ok=True)
    sess = session()
    for cur, (inst, _) in INST.items():
        if curs is not None and cur not in curs:
            continue
        p = os.path.join(DATA, f"{cur}.csv")
        if os.path.exists(p):
            continue
        try:
            df = get_instrument(sess, inst)
        except Throttled:
            raise SystemExit(THROTTLE_MSG)
        if df is None:
            print(f"  {inst}: NO DATA")
            continue
        write_csv(df, inst, p, 1e3 if "JPY" in inst else 1e5)


def load_real(curs=None):
    download(curs)
    px = {}
    for cur, (inst, inv) in INST.items():
        if curs is not None and cur not in curs:
            continue
        df = pd.read_csv(os.path.join(DATA, f"{cur}.csv"), index_col=0, parse_dates=True)
        df = df[(df.volume > 0) & (df.close > 0)]
        px[cur] = 1 / df.close if inv else df.close
    P = pd.DataFrame(px).sort_index()
    P = P[(P.index >= START) & (P.index < END)]
    P.index = P.index.tz_localize("UTC")
    return P.ffill(limit=3).dropna(how="all")


# ----------------------------------------------------------- central bank days
def rba_dates():
    d = []
    for y in range(2004, 2024):
        for m in range(2, 13):
            first = dt.date(y, m, 1)
            d.append(first + dt.timedelta(days=(1 - first.weekday()) % 7))  # first Tuesday
    d += [dt.date.fromisoformat(x) for x in [
        "2024-02-06", "2024-03-19", "2024-05-07", "2024-06-18", "2024-08-06", "2024-09-24",
        "2024-11-05", "2024-12-10", "2025-02-18", "2025-04-01", "2025-05-20", "2025-07-08",
        "2025-08-12", "2025-09-30", "2025-11-04", "2025-12-09", "2026-02-03", "2026-03-17",
        "2026-05-05", "2026-06-16", "2026-08-11", "2026-09-29"]]
    return sorted(d)


MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
          "November|December")


def rbnz_dates():
    if os.path.exists(RBNZ_CSV):
        return sorted(pd.to_datetime(pd.read_csv(RBNZ_CSV)["date"]).dt.date)
    try:
        import requests
        html = ""
        for url in ["https://www.rbnz.govt.nz/monetary-policy/official-cash-rate-decisions",
                    "https://www.rbnz.govt.nz/en/monetary-policy/monetary-policy-decisions"]:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            if r.ok:
                html += r.text
        found = {dt.datetime.strptime(f"{a} {b} {c}", "%d %B %Y").date()
                 for a, b, c in re.findall(rf"\b(\d{{1,2}}) ({MONTHS}) (\d{{4}})\b", html)}
        found = sorted(x for x in found if dt.date(1999, 11, 1) <= x <= dt.date.today())
        if len(found) < 150:
            print(f"  RBNZ page gave only {len(found)} dates -> NEWS not run. To fix: save the "
                  f"announcement dates as {RBNZ_CSV} (one column 'date').")
            return None
        os.makedirs(os.path.dirname(RBNZ_CSV), exist_ok=True)
        pd.DataFrame({"date": found}).to_csv(RBNZ_CSV, index=False)
        print(f"  RBNZ: {len(found)} announcement dates {found[0]} -> {found[-1]} (cached)")
        return found
    except Exception as e:  # noqa: BLE001
        print(f"  RBNZ dates failed ({e}) -> NEWS not run")
        return None


# =================================================================== daily clock
def daily(P):
    ny = P.index.tz_convert("America/New_York")
    wd = ny.dayofweek < 5
    C = P[wd & (ny.hour == 15)].copy()       # bar 15:00-16:00 NY -> 16:00 close (signal)
    E = P[wd & (ny.hour == 16)].copy()       # bar 16:00-17:00 NY -> 17:00 close (trade)
    C.index = pd.DatetimeIndex(ny[wd & (ny.hour == 15)].date)
    E.index = pd.DatetimeIndex(ny[wd & (ny.hour == 16)].date)
    idx = C.index.intersection(E.index)
    return np.log(C.loc[idx]), np.log(E.loc[idx])


def book(pos, rets, cmult):
    """pos, rets: DataFrames (days x currencies). pos decided at day t, earns t -> t+1."""
    pos = pos.fillna(0)
    held = pos.shift(1).fillna(0)
    cost = pd.Series(COST_BP)[pos.columns] * 1e-4 * cmult
    return (held * rets).sum(axis=1) - (pos.diff().abs().fillna(pos.abs()) * cost).sum(axis=1)


def pair_positions(z, allow=None):
    s = np.zeros(len(z))
    pos, held = 0, 0
    zv = z.values
    al = np.ones(len(z), bool) if allow is None else allow.reindex(z.index).fillna(False).values
    for i, x in enumerate(zv):
        if np.isnan(x):
            pos = 0
        elif pos == 0:
            if al[i] and x > 2:
                pos, held = -1, 0
            elif al[i] and x < -2:
                pos, held = 1, 0
        else:
            held += 1
            if (pos == -1 and (x <= 0 or x > 4)) or (pos == 1 and (x >= 0 or x < -4)) or held >= 15:
                pos = 0
        s[i] = pos
    return pd.Series(s, index=z.index)


def s_pair(C, E, cmult, allow=None):
    gap = C["AUD"] - C["NZD"]
    z = (gap - gap.rolling(60).mean()) / gap.rolling(60).std()
    p = pair_positions(z, allow)
    pos = pd.DataFrame({"AUD": p, "NZD": -p})
    return book(pos, E[["AUD", "NZD"]].diff(), cmult)


def news_allow(idx, dates, news_day):
    hit = pd.Series(0.0, index=idx)
    ds = pd.DatetimeIndex(pd.to_datetime(sorted(set(dates))))
    hit.loc[hit.index.isin(ds)] = 1.0
    # decisions on non-trading days count on the next trading day
    for d in ds[~ds.isin(idx)]:
        nxt = idx[idx > d]
        if len(nxt):
            hit.loc[nxt[0]] = 1.0
    recent = hit.rolling(5, min_periods=1).max() > 0
    return recent if news_day else ~recent


def s_bloc(C, E, cmult):
    L = C[BLOC]
    dev = L.sub(L.mean(axis=1), axis=0)
    z = (dev - dev.rolling(60).mean()) / dev.rolling(60).std()
    x = -(z.sub(z.mean(axis=1), axis=0))
    w = x.div(x.abs().sum(axis=1), axis=0) * 2
    return book(w, E[BLOC].diff(), cmult)


def s_random(C, E, cmult, seed=11):
    rng = np.random.default_rng(seed)
    s = np.zeros(len(C))
    pos, held = 0, 0
    for i in range(len(C)):
        if pos == 0 and rng.random() < 0.05:
            pos, held = rng.choice([-1, 1]), 0
        elif pos != 0:
            held += 1
            if held >= 5:
                pos = 0
        s[i] = pos
    p = pd.Series(s, index=C.index)
    return book(pd.DataFrame({"AUD": p, "NZD": -p}), E[["AUD", "NZD"]].diff(), cmult)


# =================================================================== hourly
def hourly_returns(P):
    R = np.log(P).diff()
    gap = P.index.to_series().diff() > pd.Timedelta(hours=3)
    R[gap.values] = 0.0                       # never earn across a weekend / data gap
    return R.fillna(0)


def to_daily(x):
    d = x.index.tz_convert("America/New_York").date
    return x.groupby(pd.DatetimeIndex(d)).sum()


def hourly_book(pos, R, cmult):
    cost = pd.Series(COST_BP)[pos.columns] * 1e-4 * cmult
    pnl = (pos.shift(1).fillna(0) * R[pos.columns]).sum(axis=1) \
        - (pos.diff().abs().fillna(0) * cost).sum(axis=1)
    return to_daily(pnl)


def s_catchup(R, cmult, shuffle=False):
    g = R["AUD"] - R["NZD"]
    sd = g.rolling(480, min_periods=240).std().shift(1)
    sig = pd.Series(np.where(g > 2 * sd, -1.0, np.where(g < -2 * sd, 1.0, 0.0)), index=R.index)
    if shuffle:
        sig = pd.Series(np.random.default_rng(5).permutation(sig.values), index=R.index)
    p = sig.replace(0, np.nan).ffill(limit=2).fillna(0)   # held for the next 3 hours
    return hourly_book(pd.DataFrame({"AUD": p, "NZD": -p}), R, cmult)


def home_positions(idx):
    lon = idx.tz_convert("Europe/London")
    nyc = idx.tz_convert("America/New_York")
    tok = idx.tz_convert("Asia/Tokyo")
    us = np.asarray((nyc.dayofweek < 5) & (nyc.hour >= 12) & (nyc.hour < 16))
    eu = np.asarray((lon.dayofweek < 5) & (lon.hour >= 7) & (lon.hour < 12))
    asia = np.asarray((tok.dayofweek < 5) & (tok.hour >= 9) & (tok.hour < 15))
    pos = {}
    for c in EU:
        pos[c] = np.where(eu, -1.0, np.where(us, 1.0, 0.0)) / 7
    for c in ASIA:
        pos[c] = np.where(asia, -1.0, np.where(us, 1.0, 0.0)) / 7
    return pos


def s_home(R, cmult, shift_h=0):
    pos = pd.DataFrame(home_positions(R.index + pd.Timedelta(hours=shift_h)), index=R.index)
    return hourly_book(pos, R, cmult)


# =================================================================== judging
def stats(x):
    x = x.dropna()
    if len(x) < 120 or x.std() == 0:
        return (np.nan,) * 4
    eq = np.exp(x.cumsum())
    vol = x.std() * np.sqrt(252)
    return x.mean() * 252, vol, x.mean() * 252 / vol, (eq / eq.cummax() - 1).min()


def win(x, w):
    return x[(x.index >= w[0]) & (x.index < w[1])]


def blend(daily_ret):
    m = np.exp(daily_ret.groupby(daily_ret.index.to_period("M")).sum()) - 1
    m.index = m.index.strftime("%Y-%m")
    j = pd.concat([pd.Series(CHIM), m], axis=1, keys=["c", "s"]).dropna()
    if len(j) < 36 or j["s"].std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    sh = lambda v: v.mean() / v.std() * np.sqrt(12)  # noqa: E731
    b = 0.5 * j["c"] + 0.5 * j["s"] * (j["c"].std() / j["s"].std())
    return sh(j["c"]), sh(b), j["c"].corr(j["s"]), j.loc[j["c"] < -0.03, "s"].mean()


def run(P, rbnz):
    C, E = daily(P)
    R = hourly_returns(P)
    allow = news_day = None
    if rbnz is not None:
        cb = rba_dates() + list(rbnz)
        allow = news_allow(C.index, cb, news_day=False)
        news_day = news_allow(C.index, cb, news_day=True)
    home_ok = all(c in P.columns for c in EU + ASIA)
    rows = {
        "1 PAIR": lambda k: s_pair(C, E, k),
        "2 NEWS": (lambda k: s_pair(C, E, k, allow)) if allow is not None else None,
        "  NEWS_DAY": (lambda k: s_pair(C, E, k, news_day)) if news_day is not None else None,
        "3 BLOC": lambda k: s_bloc(C, E, k),
        "4 CATCHUP": lambda k: s_catchup(R, k),
        "5 HOME": (lambda k: s_home(R, k)) if home_ok else None,
        "control RANDOM": lambda k: s_random(C, E, k),
        "control CATCHUP_SHUF": lambda k: s_catchup(R, k, shuffle=True),
        "control HOME_SHIFT6": (lambda k: s_home(R, k, shift_h=6)) if home_ok else None,
    }
    print(f"\n  data {P.index[0].date()} -> {P.index[-1].date()}, {len(C):,} days")
    print(f"  {'idea':<21}| 2012-21 ann  shp | last5y ann  vol  shp  worst | whole shp  2x-cost |"
          f" corrCHIM CHIM blend badM | trades/yr | verdict")
    res = {}
    for name, f in rows.items():
        if f is None:
            why = "no RBNZ calendar" if "NEWS" in name else "--quick: EUR GBP JPY CHF not loaded"
            print(f"  {name:<21}| NOT RUN ({why})")
            continue
        x, x2 = f(1.0), f(2.0)
        a0, _, s0, _ = stats(win(x, EARLY))
        a1, v1, s1, d1 = stats(win(x, RECENT))
        _, _, sw, _ = stats(x)
        _, _, sw2, _ = stats(x2)
        csh, bsh, corr, bad = blend(x)
        tr = (x2 - x).abs().gt(0).sum() / (len(x) / 252)
        ok = (s0 >= 0.5) and (s1 >= 0.7) and (bsh > csh) and (sw2 > 0)
        res[name] = (ok, s0, s1)
        print(f"  {name:<21}| {a0*100:+6.1f}% {s0:5.2f} | {a1*100:+6.1f}% {v1*100:4.1f}% {s1:5.2f}"
              f" {d1*100:5.0f}% | {sw:6.2f} {sw2:8.2f} | {corr:+6.2f} {csh:5.2f} {bsh:5.2f}"
              f" {bad*100:+5.1f}% | {tr:9.0f} | {'PASS' if ok else 'fail'}")
    if "2 NEWS" in res and "  NEWS_DAY" in res and res["2 NEWS"][0]:
        better = res["2 NEWS"][1] > res["  NEWS_DAY"][1] and res["2 NEWS"][2] > res["  NEWS_DAY"][2]
        print(f"  NEWS extra check (beats NEWS_DAY in both windows): {'yes' if better else 'NO -> NEWS fails'}")
    bad_ctrl = [k for k, v in res.items() if k.startswith("control") and v[0]]
    print("  controls: " + ("all fail (good)" if not bad_ctrl else
                            f"WARNING {', '.join(bad_ctrl)} passed -> the test setup is suspect"))
    print("  ann = average yearly return (log), unlevered: $1 long + $1 short for pairs, 1/7 per "
          "currency for HOME.\n  pass = 2012-21 shp >= 0.5, last5y shp >= 0.7, blend > CHIM, "
          "2x-cost whole shp > 0.  badM = avg month when CHIMERA < -3%.")
    return res


# =================================================================== smoke
def synth(planted, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2004-01-04 22:00", "2026-10-30 20:00", freq="h", tz="UTC")
    wd, hr = idx.dayofweek, idx.hour
    keep = ~((wd == 5) | ((wd == 6) & (hr < 22)) | ((wd == 4) & (hr >= 21)))
    idx = idx[keep]
    n = len(idx)
    vol = 0.0013
    com = rng.normal(0, vol, n)
    r = {}
    for c in INST:
        beta = 0.8 if c in BLOC else 0.3
        r[c] = beta * com + np.sqrt(1 - beta ** 2) * rng.normal(0, vol, n)
    if planted:
        u = np.zeros(n)                        # AUD/NZD gap that mean-reverts over ~2 weeks
        e = rng.normal(0, 0.003, n)
        for i in range(1, n):
            u[i] = u[i - 1] * (1 - 1 / 120) + e[i]
        du = np.diff(u, prepend=0)
        r["AUD"] = r["AUD"] + du / 2
        r["NZD"] = r["NZD"] - du / 2
        r["NZD"] = r["NZD"] + 0.5 * np.roll(r["AUD"] - com * 0.8, 1)    # NZD catches up
        hp = home_positions(idx)
        for c, p in hp.items():
            r[c] = r[c] + p * 7 * 0.0002          # 2 bp/hour home-hours drift
    P = pd.DataFrame({c: np.exp(np.cumsum(v)) for c, v in r.items()}, index=idx)
    days = pd.date_range("2004-01-01", "2026-10-30", freq="W-WED")
    rbnz = sorted(rng.choice(days, size=8 * 23, replace=False).astype("datetime64[D]").astype(object))
    return P, rbnz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="only AUD NZD CAD NOK: runs ideas 1-4 now, HOME later")
    a = ap.parse_args()
    if a.smoke:
        for planted in (False, True):
            print(f"\n===== SMOKE {'PLANTED (rows 1-5 should PASS, controls fail)' if planted else 'NOISE (everything should fail)'}")
            P, rbnz = synth(planted)
            run(P, rbnz)
        return
    P = load_real(BLOC if a.quick else None)
    run(P, rbnz_dates())


if __name__ == "__main__":
    main()
