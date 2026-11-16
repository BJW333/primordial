"""
spy_noise_band.py -- intraday MOMENTUM on the S&P 500 (the "noise band"),
exact rules of Zarattini, Aziz & Barbon (2024), "Beat the Market: An
Effective Intraday Momentum Strategy for S&P500 ETF (SPY)". No tuning.

THE RULES (fixed by the paper, not by us)
  noise band  for each minute of the day, sigma = average over the last 14
              days of |price / today's open - 1| at that same minute
              upper = max(today's open, yesterday's close) x (1 + sigma)
              lower = min(today's open, yesterday's close) x (1 - sigma)
  decisions   only at :00 and :30, from 10:00 to 15:30
  entry       price above upper -> long;  below lower -> short
  exit        long: price below max(upper, VWAP) at a decision time
              short: price above min(lower, VWAP);  everything flat at 16:00
  sizing      leverage = min(4, 2% / SPY's daily volatility over 14 days)
  costs       priced as E-mini S&P futures (ES): 1 tick + $2.50 per side
              (~0.4-0.8 bp of value per side); --micro: MES costs

DATA  SPY 1-minute bars from Alpaca (SIP), 2016 -> now, 9:30-16:00 ET.
      Raw (tradable) prices. Same Alpaca keys as before.
      The paper's sample is 2007 -> early 2024, so 2016-2024 is a
      REPLICATION; 2024-06 -> now is genuinely after publication.

PASS (fixed before any real-data run): ALL of
  (a) Sharpe >= 1.0, 2016 -> now, after ES costs
  (b) after publication (2024-06-01 -> now): Sharpe >= 0.5
  (c) positive in at least 7 of the 10 full years 2016-2025
  (d) at 2x costs: Sharpe >= 0.7
  (e) runs WITH CHIMERA: monthly correlation <= 0.3 AND a 50/50 equal-risk
      blend beats CHIMERA alone (1.42 monthly Sharpe, 2017-02 -> 2026-05)
NULL CONTROL (automatic): same rules on the same data with each day's
  minutes shuffled (keeps the day's size, destroys intraday trends) -> must fail.

Run (repo root):
    python3.10 scripts/spy_noise_band.py --smoke
    python3.10 scripts/spy_noise_band.py | tee logs/spy_noise_band.txt
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
MIN_DIR = os.path.join(ROOT, "data", "minute")
PUB = pd.Timestamp("2024-06-01")
TARGET, MAX_LEV, LOOK = 0.02, 4.0, 14
CHIM = {"2017-02": 0.05638, "2017-03": -0.00502, "2017-04": 0.0516, "2017-05": 0.05944, "2017-06": -0.04716, "2017-07": 0.08036, "2017-08": 0.04121, "2017-09": -0.0082, "2017-10": 0.01275, "2017-11": 0.06163, "2017-12": 0.00984, "2018-01": 0.05409, "2018-02": -0.04113, "2018-03": -0.00773, "2018-04": -0.00085, "2018-05": 0.02454, "2018-06": -0.02166, "2018-07": -0.03758, "2018-08": 0.04259, "2018-09": 0.01471, "2018-10": -0.09646, "2018-11": -0.00262, "2018-12": -0.03574, "2019-01": 0.05253, "2019-02": 0.02498, "2019-03": 0.01921, "2019-04": -0.01715, "2019-05": -0.0174, "2019-06": 0.06728, "2019-07": 0.04436, "2019-08": 0.01406, "2019-09": -0.03595, "2019-10": 0.02579, "2019-11": 0.03318, "2019-12": 0.08462, "2020-01": 0.00542, "2020-02": -0.0347, "2020-03": -0.03995, "2020-04": 0.10273, "2020-05": 0.02383, "2020-06": 0.08567, "2020-07": 0.04235, "2020-08": -0.0517, "2020-09": -0.00151, "2020-10": -0.00627, "2020-11": 0.17142, "2020-12": 0.03592, "2021-01": 0.20194, "2021-02": 0.1025, "2021-03": 0.05749, "2021-04": 0.04693, "2021-05": 0.0169, "2021-06": 0.00755, "2021-07": 0.01409, "2021-08": 0.02306, "2021-09": -0.03898, "2021-10": 0.06648, "2021-11": -0.02027, "2021-12": 0.05834, "2022-01": -0.13833, "2022-02": 0.03776, "2022-03": 0.07138, "2022-04": 0.04416, "2022-05": 0.01971, "2022-06": -0.03329, "2022-07": 0.01483, "2022-08": 0.04692, "2022-09": 0.00549, "2022-10": 0.01113, "2022-11": -0.07182, "2022-12": -0.03874, "2023-01": -0.00446, "2023-02": 0.00744, "2023-03": -0.01724, "2023-04": 0.00149, "2023-05": 0.04634, "2023-06": -0.00555, "2023-07": 0.01982, "2023-08": -0.02555, "2023-09": -0.03562, "2023-10": -0.05146, "2023-11": 0.03769, "2023-12": 0.07371, "2024-01": 0.04383, "2024-02": 0.10113, "2024-03": 0.02381, "2024-04": -0.00117, "2024-05": 0.09167, "2024-06": 0.00586, "2024-07": 0.03194, "2024-08": 0.02881, "2024-09": 0.02354, "2024-10": -0.01373, "2024-11": 0.05931, "2024-12": -0.05531, "2025-01": 0.04376, "2025-02": -0.03464, "2025-03": 0.01805, "2025-04": 0.01726, "2025-05": 0.0183, "2025-06": 0.02993, "2025-07": 0.00038, "2025-08": 0.00485, "2025-09": 0.17491, "2025-10": -0.00345, "2025-11": 0.01746, "2025-12": -0.00032, "2026-01": 0.11679, "2026-02": 0.11839, "2026-03": 0.04071, "2026-04": 0.05341, "2026-05": 0.00883}


# ===================================================================== data
def _alpaca_year(sym, y, key, sec):
    import requests
    os.makedirs(MIN_DIR, exist_ok=True)
    p = os.path.join(MIN_DIR, f"{sym}_{y}_sip.csv.gz")
    now = pd.Timestamp.now(tz="UTC")
    if os.path.exists(p) and not (y == now.year and
                                  time.time() - os.path.getmtime(p) > 86400):
        return pd.read_csv(p, index_col=0, parse_dates=True)
    start = pd.Timestamp(f"{y}-01-01", tz="UTC")
    end = min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), now - pd.Timedelta(minutes=20))
    url = f"https://data.alpaca.markets/v2/stocks/{sym}/bars"
    hd = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    prm = {"timeframe": "1Min", "start": start.isoformat(), "end": end.isoformat(),
           "limit": 10000, "adjustment": "raw", "feed": "sip"}
    rows, tok = [], None
    while True:
        if tok:
            prm["page_token"] = tok
        for k in range(6):
            r = requests.get(url, headers=hd, params=prm, timeout=60)
            if r.status_code == 429:
                time.sleep(3 + 3 * k)
                continue
            break
        r.raise_for_status()
        j = r.json()
        rows += j.get("bars") or []
        tok = j.get("next_page_token")
        if not tok:
            break
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York") \
        .dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                            "v": "volume"})[["open", "high", "low", "close", "volume"]]
    df.to_csv(p)
    print(f"    {sym} {y}: {len(df):,} bars")
    return df


def load_spy():
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("  export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
    df = pd.concat([_alpaca_year("SPY", y, key, sec)
                    for y in range(2016, pd.Timestamp.today().year + 1)]).sort_index()
    return df[~df.index.duplicated(keep="last")]


def to_days(df):
    """-> list of (date, minute-of-day array, open, high, low, close, volume),
    regular hours, full days only (>= 380 of 390 minutes)."""
    mod = df.index.hour * 60 + df.index.minute
    df = df[(mod >= 570) & (mod < 960)]
    out = []
    for d, g in df.groupby(df.index.normalize()):
        if len(g) < 380:
            continue
        m = (g.index.hour * 60 + g.index.minute).values - 570
        out.append((d, m, g["open"].values, g["high"].values, g["low"].values,
                    g["close"].values, g["volume"].values))
    return out


def shuffle_days(days, seed=99):
    rng = np.random.default_rng(seed)
    out = []
    for d, m, o, h, l, c, v in days:
        prev = np.r_[o[0], c[:-1]]
        gap, ret = o / prev, c / o
        p = rng.permutation(len(c))
        gap, ret, vv = gap[p], ret[p], v[p]
        gap[0] = 1.0
        px = o[0] * np.cumprod(gap * ret)
        oo = np.r_[o[0], px[:-1]] * gap
        out.append((d, m, oo, np.maximum(oo, px), np.minimum(oo, px), px, vv))
    return out


# ================================================================ strategy
def backtest(days, cost_mult=1.0, micro=False):
    """daily returns of the paper's rules (fraction of capital)."""
    per_side_pts = 0.25 + (0.62 / 5 if micro else 2.50 / 50)   # ES/MES points
    moves = np.full((len(days), 390), np.nan)
    for i, (d, m, o, h, l, c, v) in enumerate(days):
        moves[i, m] = np.abs(c / o[0] - 1)
    closes = np.array([x[5][-1] for x in days])
    dret = np.r_[np.nan, closes[1:] / closes[:-1] - 1]
    vol14 = pd.Series(dret).rolling(LOOK).std().shift(1).values
    rets, dates, trades = [], [], 0
    for i in range(LOOK + 1, len(days)):
        d, m, o, h, l, c, v = days[i]
        sigma = np.nanmean(moves[i - LOOK:i], axis=0)            # last 14 days
        lev = min(MAX_LEV, TARGET / vol14[i]) if vol14[i] > 0 else 0.0
        O, Cp = o[0], closes[i - 1]
        up_base, dn_base = max(O, Cp), min(O, Cp)
        vwap = np.cumsum((h + l + c) / 3 * v) / np.maximum(np.cumsum(v), 1)
        cost = per_side_pts / (10.0 * O) * cost_mult           # ES ~ 10 x SPY
        pos, last_px, pnl = 0.0, None, 0.0
        for j in range(len(m)):
            t = m[j] + 1                                       # bar end, minutes after 9:30
            px = c[j]
            if last_px is not None:
                pnl += pos * (px / last_px - 1)
            last_px = px
            if t >= 390 or j == len(m) - 1:                    # 16:00: flat
                pnl -= abs(pos) * cost
                pos = 0.0
                break
            if t < 30 or t % 30 != 0:                          # decisions 10:00..15:30
                continue
            s = sigma[min(m[j], 389)]
            if not np.isfinite(s):
                continue
            ub, lb = up_base * (1 + s), dn_base * (1 - s)
            new = pos
            if pos > 0 and px < max(ub, vwap[j]):
                new = 0.0
            elif pos < 0 and px > min(lb, vwap[j]):
                new = 0.0
            if new == 0.0:
                if px > ub:
                    new = lev
                elif px < lb:
                    new = -lev
            if new != pos:
                pnl -= abs(new - pos) * cost
                trades += int(new != 0)
                pos = new
        rets.append(pnl)
        dates.append(d)
    return pd.Series(rets, pd.DatetimeIndex(dates)), trades


# =================================================================== judge
def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 20 and x.std() > 0 else np.nan


def blend(daily):
    m = (1 + daily).resample("ME").prod() - 1
    m.index = m.index.strftime("%Y-%m")
    j = pd.concat([pd.Series(CHIM), m], axis=1, keys=["c", "s"]).dropna()
    if len(j) < 24 or j["s"].std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
    b = 0.5 * j["c"] + 0.5 * j["s"] * j["c"].std() / j["s"].std()
    bad = j[j["c"] < -0.03]["s"].mean()                 # strategy in CHIMERA's bad months
    return sh(j["c"]), sh(b), j["c"].corr(j["s"]), bad


def report(days, label, full=True, micro=False):
    r, n = backtest(days, micro=micro)
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    eq = (1 + r).cumprod()
    s = sharpe(r)
    post = r[r.index >= PUB]
    print(f"\n  ===== {label}: {len(r):,} days {r.index[0].date()} -> {r.index[-1].date()}")
    print(f"  {n:,} entries ({n / yrs:.0f}/yr) | {eq.iloc[-1] ** (1 / yrs) * 100 - 100:+.1f}%/yr"
          f" | Sharpe {s:+.2f} | worst drop {(eq / eq.cummax() - 1).min() * 100:.0f}%"
          f" | after publication Sharpe {sharpe(post):+.2f} ({len(post)} days)")
    by = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    print("  by year: " + "  ".join(f"{y} {v * 100:+.0f}%" for y, v in by.items()))
    full_years = by[(by.index >= 2016) & (by.index <= 2025)]
    if not full:
        ok = s >= 1.0
        print(f"  VERDICT: {'PASS' if ok else 'FAIL'} (Sharpe >= 1.0 check only)")
        return
    s2 = sharpe(backtest(days, cost_mult=2.0, micro=micro)[0])
    csh, bsh, cor, bad = blend(r)
    print(f"  2x costs Sharpe {s2:+.2f} | with CHIMERA (monthly): corr {cor:+.2f}, "
          f"CHIM {csh:.2f} -> blend {bsh:.2f} | avg month when CHIMERA < -3%: "
          f"{bad * 100:+.1f}%")
    ok = {"a Sharpe >= 1.0 (2016-now)": s >= 1.0,
          "b after publication Sharpe >= 0.5": sharpe(post) >= 0.5,
          f"c >= 7/10 full years positive ({(full_years > 0).sum()})":
              (full_years > 0).sum() >= 7,
          "d 2x costs Sharpe >= 0.7": s2 >= 0.7,
          "e corr <= 0.3 and blend > CHIM": (cor <= 0.3) and (bsh > csh)}
    for k, v in ok.items():
        print(f"    {k:<38} {'yes' if v else 'NO'}")
    print(f"  VERDICT: {'PASS' if all(ok.values()) else 'FAIL'}")


# =================================================================== smoke
def synth(trend, seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", "2026-10-30")
    out, px = [], 200.0
    for d in dates:
        sd = 0.0006
        drift = rng.choice([-1, 0, 0, 1]) * trend * sd   # some days trend, some don't
        r = rng.normal(drift, sd, 390)
        c = px * np.exp(np.cumsum(r))
        o = np.r_[px, c[:-1]]
        w = np.abs(rng.normal(0, sd / 3, (2, 390)))
        out.append((d, np.arange(390), o, np.maximum(o, c) * (1 + w[0]),
                    np.minimum(o, c) * (1 - w[1]), c,
                    rng.integers(1e4, 5e4, 390).astype(float)))
        px = c[-1] * np.exp(rng.normal(0, 0.004))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--micro", action="store_true", help="MES costs")
    a = ap.parse_args()
    if a.smoke:
        report(synth(0.0), "SMOKE NOISE (must FAIL)", full=False, micro=a.micro)
        report(synth(0.12), "SMOKE PLANTED trend days (must PASS)", full=False, micro=a.micro)
        return
    days = to_days(load_spy())
    print(f"  data: SPY 1-min (Alpaca SIP), {len(days):,} full days; costs as "
          f"{'MES' if a.micro else 'ES'}")
    report(days, "REAL DATA", full=True, micro=a.micro)
    report(shuffle_days(days), "NULL CONTROL: minutes shuffled within each day (must FAIL)",
           full=False, micro=a.micro)


if __name__ == "__main__":
    main()
