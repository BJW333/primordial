"""
gold_choch.py -- Primordial screen of the "premium zone + CHoCH" setup on GOLD,
15-minute bars (from Blake's TheTradingAcademy screenshot, 2026-11-10).

THE SETUP, MECHANICAL (all rules fixed here, before any real-data run)
  short (long is the exact mirror):
  1 zone    the swing high H that starts the setup sits in the TOP HALF of the
            last ZONE bars' range (ICT "premium"; longs need the bottom half)
  2 trend   on the 15m chart the last two confirmed swing lows are rising
            (higher low), and H formed after the newer one
  3 CHoCH   a bar CLOSES below that newer higher low (first close below)
  4 entry   sell limit at RETRACE (50%) of the leg from H down to the lowest
            low since H; good for TTL bars (1 day); cancelled if price
            reaches the target first
  5 stop    H + 0.1 x ATR(14);  skip if the risk is < 0.3 x ATR (micro-setups)
  6 target  entry - RR x risk (RR 3, like the screenshot's ~1:3 box)
  7 exit    stop, target, or MAX_HOLD bars (5 days) at the close
  swings = fractal pivots: a high higher than PIVOT bars on each side, known
  only PIVOT bars later (no look-ahead). One trade at a time.

FILLS (conservative -- the old ctrl_vwap_reversion lesson: limit-fill
  optimism can fake an edge)
  * a limit entry fills only if price trades THROUGH it by 1 tick
  * the bar that fills you is checked for the stop (assume stop first);
    any bar touching both stop and target = stop
  * stops gap-fill at the open if price jumped past them
  * costs per side: 1 tick slippage + commission, on EVERY fill
    (GC: $0.10/oz tick + $2.50 per 100 oz; --micro: MGC $0.62 per 10 oz)

DATA
  default : GLD 1-minute bars from Alpaca (SIP), 2016 -> now, regular hours
            9:30-16:00 ET, built into 15-minute bars; gold price ~= GLD x 10.8
            for costs. Keys: ALPACA_API_KEY / ALPACA_SECRET_KEY (env or .env).
            Caveat: GLD trades only US hours; gold futures trade ~23h.
  --csv F : real gold 15m bars (e.g. XAUUSD or GC): timestamp,open,high,low,
            close[,volume]; run with --ratio 1

PASS (canonical rules; ALL of)
  (a) expectancy >= +0.10 R per trade after costs, t-stat >= 2
  (b) positive expectancy in both halves (2016-2020, 2021-now)
  (c) beats the 95th percentile of RANDOM-BRACKET controls: same number of
      trades at random times and directions, same stop size (in ATR) and
      RR -> asks "does the SHAPE add anything over a 1:3 bracket?"
  (d) at least 8 of 12 neighbouring variants positive
      (PIVOT 2/3/5 x RR 2/3 x ZONE 52/130 bars)
  (e) risk 0.5% per trade, monthly blend with CHIMERA beats CHIMERA (1.42
      monthly Sharpe over 2017-02 -> 2026-05)
NULL CONTROL (automatic): same rules on the same data with each day's
  15-minute bars shuffled (keeps sizes, destroys any shape) -> must fail.

Run (repo root):
    python3.10 scripts/gold_choch.py --smoke      # planted edge must pass, noise fail
    python3.10 scripts/gold_choch.py | tee logs/gold_choch.txt
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
SPLIT = pd.Timestamp("2021-01-01")
CANON = dict(pivot=3, zone=130, retr=0.5, ttl=26, rr=3.0, hold=130)
RISK = 0.005
CHIM = {"2017-02": 0.05638, "2017-03": -0.00502, "2017-04": 0.0516, "2017-05": 0.05944, "2017-06": -0.04716, "2017-07": 0.08036, "2017-08": 0.04121, "2017-09": -0.0082, "2017-10": 0.01275, "2017-11": 0.06163, "2017-12": 0.00984, "2018-01": 0.05409, "2018-02": -0.04113, "2018-03": -0.00773, "2018-04": -0.00085, "2018-05": 0.02454, "2018-06": -0.02166, "2018-07": -0.03758, "2018-08": 0.04259, "2018-09": 0.01471, "2018-10": -0.09646, "2018-11": -0.00262, "2018-12": -0.03574, "2019-01": 0.05253, "2019-02": 0.02498, "2019-03": 0.01921, "2019-04": -0.01715, "2019-05": -0.0174, "2019-06": 0.06728, "2019-07": 0.04436, "2019-08": 0.01406, "2019-09": -0.03595, "2019-10": 0.02579, "2019-11": 0.03318, "2019-12": 0.08462, "2020-01": 0.00542, "2020-02": -0.0347, "2020-03": -0.03995, "2020-04": 0.10273, "2020-05": 0.02383, "2020-06": 0.08567, "2020-07": 0.04235, "2020-08": -0.0517, "2020-09": -0.00151, "2020-10": -0.00627, "2020-11": 0.17142, "2020-12": 0.03592, "2021-01": 0.20194, "2021-02": 0.1025, "2021-03": 0.05749, "2021-04": 0.04693, "2021-05": 0.0169, "2021-06": 0.00755, "2021-07": 0.01409, "2021-08": 0.02306, "2021-09": -0.03898, "2021-10": 0.06648, "2021-11": -0.02027, "2021-12": 0.05834, "2022-01": -0.13833, "2022-02": 0.03776, "2022-03": 0.07138, "2022-04": 0.04416, "2022-05": 0.01971, "2022-06": -0.03329, "2022-07": 0.01483, "2022-08": 0.04692, "2022-09": 0.00549, "2022-10": 0.01113, "2022-11": -0.07182, "2022-12": -0.03874, "2023-01": -0.00446, "2023-02": 0.00744, "2023-03": -0.01724, "2023-04": 0.00149, "2023-05": 0.04634, "2023-06": -0.00555, "2023-07": 0.01982, "2023-08": -0.02555, "2023-09": -0.03562, "2023-10": -0.05146, "2023-11": 0.03769, "2023-12": 0.07371, "2024-01": 0.04383, "2024-02": 0.10113, "2024-03": 0.02381, "2024-04": -0.00117, "2024-05": 0.09167, "2024-06": 0.00586, "2024-07": 0.03194, "2024-08": 0.02881, "2024-09": 0.02354, "2024-10": -0.01373, "2024-11": 0.05931, "2024-12": -0.05531, "2025-01": 0.04376, "2025-02": -0.03464, "2025-03": 0.01805, "2025-04": 0.01726, "2025-05": 0.0183, "2025-06": 0.02993, "2025-07": 0.00038, "2025-08": 0.00485, "2025-09": 0.17491, "2025-10": -0.00345, "2025-11": 0.01746, "2025-12": -0.00032, "2026-01": 0.11679, "2026-02": 0.11839, "2026-03": 0.04071, "2026-04": 0.05341, "2026-05": 0.00883}


# ===================================================================== data
def _env_keys():
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    p = os.path.join(ROOT, ".env")
    if (not k or not s) and os.path.exists(p):
        for line in open(p):
            if "=" in line and not line.strip().startswith("#"):
                a, b = line.split("=", 1)
                a, b = a.strip(), b.strip().strip('"').strip("'")
                if a in ("ALPACA_API_KEY", "APCA_API_KEY_ID") and not k:
                    k = b
                if a in ("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY") and not s:
                    s = b
    if not k or not s:
        sys.exit("  set ALPACA_API_KEY and ALPACA_SECRET_KEY (export or .env)")
    return k, s


def _alpaca_year(sym, y, key, sec, feed):
    import requests
    os.makedirs(MIN_DIR, exist_ok=True)
    p = os.path.join(MIN_DIR, f"{sym}_{y}_{feed}.csv.gz")
    now = pd.Timestamp.now(tz="UTC")
    if os.path.exists(p) and not (y == now.year and
                                  time.time() - os.path.getmtime(p) > 86400):
        return pd.read_csv(p, index_col=0, parse_dates=True)
    start = pd.Timestamp(f"{y}-01-01", tz="UTC")
    end = min(pd.Timestamp(f"{y + 1}-01-01", tz="UTC"), now - pd.Timedelta(minutes=20))
    url = f"https://data.alpaca.markets/v2/stocks/{sym}/bars"
    hd = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    prm = {"timeframe": "1Min", "start": start.isoformat(), "end": end.isoformat(),
           "limit": 10000, "adjustment": "raw", "feed": feed}
    rows, tok = [], None
    while True:
        if tok:
            prm["page_token"] = tok
        for k in range(5):
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
    if df.empty:
        return df
    df.index = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York") \
        .dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                            "v": "volume"})[["open", "high", "low", "close", "volume"]]
    df.to_csv(p)
    print(f"    {sym} {y}: {len(df):,} bars")
    return df


def load_alpaca(sym="GLD", first=2016):
    key, sec = _env_keys()
    frames, feed = [], "sip"
    for y in range(first, pd.Timestamp.today().year + 1):
        try:
            frames.append(_alpaca_year(sym, y, key, sec, feed))
        except Exception as e:  # noqa: BLE001
            if feed == "sip":
                print(f"  SIP feed refused ({e}); using IEX feed")
                feed = "iex"
                frames.append(_alpaca_year(sym, y, key, sec, feed))
            else:
                raise
    return pd.concat([f for f in frames if len(f)]).sort_index()


def to_15m(df):
    """1-minute regular-hours bars -> 15-minute bars at 9:30, 9:45 ... 15:45."""
    df = df[~df.index.duplicated(keep="last")]
    mod = df.index.hour * 60 + df.index.minute
    df = df[(mod >= 570) & (mod < 960)]
    mod = df.index.hour * 60 + df.index.minute
    key = df.index.normalize() + pd.to_timedelta(570 + (mod - 570) // 15 * 15, unit="m")
    g = df.groupby(key)
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last()}).dropna()


def load_csv(path):
    d = pd.read_csv(path, header=None)
    try:
        float(d.iloc[0, 1])
    except ValueError:
        d = pd.read_csv(path)
    d = d.iloc[:, :5]
    d.columns = ["t", "open", "high", "low", "close"]
    d.index = pd.to_datetime(d["t"])
    return d.drop(columns="t").astype(float).sort_index()


def arrays(bars):
    o, h, l, c = (bars[k].values.astype(float) for k in ("open", "high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().shift(0).values
    return {"o": o, "h": h, "l": l, "c": c, "atr": atr, "t": bars.index}


def shuffled(bars, seed=99):
    """null: each day's 15m bars in random order (sizes kept, shape destroyed)."""
    rng = np.random.default_rng(seed)
    out = []
    for _, g in bars.groupby(bars.index.normalize()):
        o, h, l, c = (g[k].values for k in ("open", "high", "low", "close"))
        prev = np.r_[o[0], c[:-1]]
        gap, ret = o / prev, c / o
        up, dn = h / np.maximum(o, c), l / np.minimum(o, c)
        p = rng.permutation(len(g))
        gap, ret, up, dn = gap[p], ret[p], up[p], dn[p]
        gap[0] = 1.0
        px, rows = o[0], []
        for i in range(len(g)):
            oo = px * gap[i]
            cc = oo * ret[i]
            rows.append((oo, max(oo, cc) * up[i], min(oo, cc) * dn[i], cc))
            px = cc
        out.append(pd.DataFrame(rows, g.index, ["open", "high", "low", "close"]))
    return pd.concat(out)


# ================================================================== engine
def bracket(A, t0, d, entry, stop, target, hold, slip, comm, fill_bar=None):
    """run an open trade from bar t0 (the fill bar) -> (R, exit index)."""
    o, h, l, c = A["o"], A["h"], A["l"], A["c"]
    risk = abs(stop - entry)
    n = len(c)
    for i in range(t0, min(n, t0 + hold + 1)):
        if d < 0:
            if h[i] >= stop:
                ex = max(o[i], stop) + slip if i > t0 else max(stop, entry) + slip
                return (entry - ex - 2 * comm) / risk, i
            if i > t0 and l[i] <= target - slip:
                ex = min(o[i], target)
                return (entry - ex - 2 * comm) / risk, i
        else:
            if l[i] <= stop:
                ex = min(o[i], stop) - slip if i > t0 else min(stop, entry) - slip
                return (ex - entry - 2 * comm) / risk, i
            if i > t0 and h[i] >= target + slip:
                ex = max(o[i], target)
                return (ex - entry - 2 * comm) / risk, i
    i = min(n - 1, t0 + hold)
    ex = c[i] - d * slip
    return (d * (ex - entry) - 2 * comm) / risk, i


def simulate(A, p, slip, comm):
    """slip/comm arrays (price units per side, one per bar). Returns trades."""
    o, h, l, c, atr = A["o"], A["h"], A["l"], A["c"], A["atr"]
    n, k = len(c), p["pivot"]
    zhi = pd.Series(h).rolling(p["zone"], min_periods=p["zone"]).max().values
    zlo = pd.Series(l).rolling(p["zone"], min_periods=p["zone"]).min().values
    sh, sl, trades = [], [], []
    pend, busy_until = None, -1
    for i in range(n):
        # pending limit order: fill / expire / invalidate (orders from bar < i)
        if pend is not None and i > pend["t0"]:
            d, e = pend["d"], pend["entry"]
            if i > pend["exp"]:
                pend = None
            elif (d < 0 and h[i] >= e + slip[i]) or (d > 0 and l[i] <= e - slip[i]):
                fill = max(o[i], e) if d < 0 else min(o[i], e)
                fill = fill - d * slip[i]                     # slippage on entry
                R, j = bracket(A, i, d, fill, pend["stop"], pend["target"], p["hold"],
                               slip[i], comm[i])
                trades.append((i, j, d, R, pend["risk_atr"]))
                busy_until, pend = j, None
            elif (d < 0 and l[i] <= pend["target"]) or (d > 0 and h[i] >= pend["target"]):
                pend = None                                   # ran without us
        # swings confirmed at bar i (pivot at j = i-k)
        j = i - k
        if j - k >= 0:
            w = h[j - k:j + k + 1]
            if h[j] == w.max() and (w == h[j]).sum() == 1:
                sh.append((j, h[j]))
            w = l[j - k:j + k + 1]
            if l[j] == w.min() and (w == l[j]).sum() == 1:
                sl.append((j, l[j]))
        if pend is not None or i <= busy_until or i < 1 or not (atr[i] > 0) \
                or not np.isfinite(zhi[i]):
            continue
        mid = (zhi[i] + zlo[i]) / 2
        # bearish CHoCH in premium
        if len(sl) >= 2 and sh:
            (_, L1), (iL2, L2), (iH, H) = sl[-2], sl[-1], sh[-1]
            if L2 > L1 and iH > iL2 and c[i] < L2 <= c[i - 1] and H >= mid:
                leg = l[iH:i + 1].min()
                e = H - p["retr"] * (H - leg)
                st = H + 0.1 * atr[i]
                r = st - e
                if r >= 0.3 * atr[i]:
                    pend = dict(d=-1, entry=e, stop=st, target=e - p["rr"] * r, t0=i,
                                exp=i + p["ttl"], risk_atr=r / atr[i])
                    continue
        # bullish CHoCH in discount
        if len(sh) >= 2 and sl:
            (_, H1), (iH2, H2), (iL, L) = sh[-2], sh[-1], sl[-1]
            if H2 < H1 and iL > iH2 and c[i] > H2 >= c[i - 1] and L <= mid:
                leg = h[iL:i + 1].max()
                e = L + p["retr"] * (leg - L)
                st = L - 0.1 * atr[i]
                r = e - st
                if r >= 0.3 * atr[i]:
                    pend = dict(d=1, entry=e, stop=st, target=e + p["rr"] * r, t0=i,
                                exp=i + p["ttl"], risk_atr=r / atr[i])
    return trades


def random_brackets(A, trades, p, slip, comm, seed):
    """same count / stop size (ATR units) / RR, random time and direction."""
    rng = np.random.default_rng(seed)
    o, c, atr = A["o"], A["c"], A["atr"]
    ok = np.flatnonzero(np.isfinite(atr) & (atr > 0))[:-2]
    Rs = []
    for (_, _, _, _, ra) in trades:
        t = int(rng.choice(ok))
        d = rng.choice([-1, 1])
        e = o[t + 1] + d * slip[t + 1]                       # market entry next open
        r = ra * atr[t]
        st, tg = e - d * r, e + d * p["rr"] * r
        R, _ = bracket(A, t + 1, d, e, st, tg, p["hold"], slip[t + 1], comm[t + 1])
        Rs.append(R)
    return np.array(Rs)


# =================================================================== judge
def tstat(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else 0.0


def daily_returns(A, trades):
    if not trades:
        return pd.Series(dtype=float)
    t = A["t"]
    s = pd.Series([R * RISK for (_, j, _, R, _) in trades],
                  pd.DatetimeIndex([t[j] for (_, j, _, _, _) in trades]).normalize())
    days = pd.DatetimeIndex(sorted(set(t.normalize())))
    return s.groupby(level=0).sum().reindex(days, fill_value=0.0)


def blend(daily):
    m = (1 + daily).resample("ME").prod() - 1
    m.index = m.index.strftime("%Y-%m")
    j = pd.concat([pd.Series(CHIM), m], axis=1, keys=["c", "s"]).dropna()
    if len(j) < 24 or j["s"].std() == 0:
        return np.nan, np.nan, np.nan
    sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
    b = 0.5 * j["c"] + 0.5 * j["s"] * j["c"].std() / j["s"].std()
    return sh(j["c"]), sh(b), j["c"].corr(j["s"])


def evaluate(A, label, slip, comm, full=True):
    print(f"\n  ===== {label}: {len(A['c']):,} bars {A['t'][0].date()} -> {A['t'][-1].date()}")
    tr = simulate(A, CANON, slip, comm)
    if len(tr) < 20:
        print(f"  only {len(tr)} trades -> FAIL")
        return
    R = np.array([x[3] for x in tr])
    when = np.array([A["t"][x[1]] for x in tr])
    yrs = (A["t"][-1] - A["t"][0]).days / 365.25
    r1, r2 = R[when < SPLIT], R[when >= SPLIT]
    longs = np.array([x[2] > 0 for x in tr])
    print(f"  canonical {CANON}")
    print(f"  trades {len(R)} ({len(R) / yrs:.0f}/yr, {longs.mean() * 100:.0f}% long) | "
          f"win {np.mean(R > 0) * 100:.0f}% | expectancy {R.mean():+.3f} R (t {tstat(R):+.2f})"
          f" | longs {R[longs].mean():+.3f} R, shorts {R[~longs].mean():+.3f} R")
    print(f"  halves: 2016-20 {r1.mean() if len(r1) else np.nan:+.3f} R ({len(r1)}) | "
          f"2021-now {r2.mean() if len(r2) else np.nan:+.3f} R ({len(r2)})")
    yr = pd.Series(R, pd.DatetimeIndex(when)).groupby(pd.DatetimeIndex(when).year).mean()
    print("  by year (R/trade): " + "  ".join(f"{y} {v:+.2f}" for y, v in yr.items()))
    rnd = [random_brackets(A, tr, CANON, slip, comm, s).mean() for s in range(20)]
    p95 = float(np.percentile(rnd, 95))
    print(f"  random 1:{CANON['rr']:.0f} brackets (20 runs): mean {np.mean(rnd):+.3f} R, "
          f"95th pct {p95:+.3f} R")
    ok_grid, n_grid = None, 0
    if full:
        res = []
        for pv in (2, 3, 5):
            for rr in (2.0, 3.0):
                for z in (52, 130):
                    q = dict(CANON, pivot=pv, rr=rr, zone=z)
                    t2 = simulate(A, q, slip, comm)
                    m = np.mean([x[3] for x in t2]) if t2 else np.nan
                    res.append((pv, rr, z, len(t2), m))
        n_grid = sum(1 for r in res if np.isfinite(r[4]) and r[4] > 0)
        print("  grid (pivot/RR/zone: trades, R): " + " | ".join(
            f"{a}/{b:.0f}/{c}: {d} {e:+.2f}" for a, b, c, d, e in res))
        ok_grid = n_grid >= 8
    dr = daily_returns(A, tr)
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else np.nan
    csh, bsh, cor = blend(dr)
    print(f"  at {RISK * 100:.1f}% risk/trade: {dr.mean() * 252 * 100:+.1f}%/yr, Sharpe {sh:+.2f};"
          f" vs CHIMERA corr {cor:+.2f}, CHIM {csh:.2f} -> blend {bsh:.2f}")
    checks = {"a expectancy>=0.10R & t>=2": R.mean() >= 0.10 and tstat(R) >= 2,
              "b both halves > 0": len(r1) > 0 and len(r2) > 0 and r1.mean() > 0
              and r2.mean() > 0,
              "c beats random 95th pct": R.mean() > p95}
    if full:
        checks[f"d grid >= 8/12 positive ({n_grid})"] = ok_grid
    checks["e blend > CHIM"] = bool(bsh > csh) if np.isfinite(bsh) else False
    for kk, v in checks.items():
        print(f"    {kk:<32} {'yes' if v else 'NO'}")
    print(f"  VERDICT: {'PASS' if all(checks.values()) else 'FAIL'}")


def cost_arrays(A, ratio, micro):
    tick = 0.10
    comm = (0.62 / 10) if micro else (2.50 / 100)
    n = len(A["c"])
    return np.full(n, tick / ratio), np.full(n, comm / ratio)


# =================================================================== smoke
def synth(seed, plant=None):
    """GLD-like 15m bars from a fine (1-minute) random path; `plant` adds drift
    in a trade's direction for 20 bars after each listed fill bar."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", "2026-10-30")
    nb = 26
    r = rng.normal(0, 0.0004, (len(days) * nb, 15))
    if plant:
        for (i, d) in plant:
            r[i:i + 20] += d * 0.00012
    gaps = np.zeros(len(days) * nb)
    gaps[::nb] = rng.normal(0, 0.003, len(days))
    path = 120.0 * np.exp(np.cumsum(r, axis=1) + np.cumsum(r.sum(axis=1) + gaps)[:, None]
                          - r.sum(axis=1)[:, None])
    o = np.r_[120.0, path[:-1, -1]] * np.exp(gaps)
    c = path[:, -1]
    h = np.maximum(path.max(axis=1), o)
    l = np.minimum(path.min(axis=1), o)
    idx = (days.repeat(nb) + pd.to_timedelta(np.tile(570 + 15 * np.arange(nb), len(days)),
                                             unit="m"))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, idx)


def smoke(ratio, micro):
    base = synth(1)
    A = arrays(base)
    slip, comm = cost_arrays(A, ratio, micro)
    evaluate(A, "SMOKE NOISE (must FAIL)", slip, comm, full=False)
    tr = simulate(A, CANON, slip, comm)
    planted = synth(1, plant=[(x[0], x[2]) for x in tr])
    A2 = arrays(planted)
    evaluate(A2, "SMOKE PLANTED (a/b/c must pass)", slip, comm, full=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--csv", help="gold 15m bars csv instead of Alpaca GLD")
    ap.add_argument("--ratio", type=float, default=None,
                    help="gold $/oz per price unit (GLD 10.8, gold csv 1)")
    ap.add_argument("--micro", action="store_true", help="MGC costs")
    a = ap.parse_args()
    if a.smoke:
        smoke(10.8, a.micro)
        return
    if a.csv:
        bars, ratio, src = load_csv(a.csv), a.ratio or 1.0, os.path.basename(a.csv)
    else:
        bars, ratio, src = to_15m(load_alpaca("GLD")), a.ratio or 10.8, "GLD 15m (Alpaca)"
    print(f"  data: {src}, {len(bars):,} 15m bars; costs as "
          f"{'MGC' if a.micro else 'GC'} (1 tick + commission per side)")
    A = arrays(bars)
    slip, comm = cost_arrays(A, ratio, a.micro)
    evaluate(A, "REAL DATA", slip, comm, full=True)
    N = arrays(shuffled(bars))
    evaluate(N, "NULL CONTROL: shuffled 15m bars (must FAIL)", slip, comm, full=False)


if __name__ == "__main__":
    main()
