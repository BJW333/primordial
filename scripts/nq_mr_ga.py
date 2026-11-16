"""
nq_mr_ga.py -- a genetic algorithm that EVOLVES the entry and exit rules of an
intraday mean-reversion strategy on the Nasdaq-100 (E-mini NQ), in Primordial.

Unlike the MaxAI paper (whose GA only tunes 3 learning settings and picks the
winner on its own test data), this GA evolves the trade rules themselves and
keeps three time periods strictly apart.

DATA
  default : QQQ 1-minute bars from Alpaca (free, 2016 -> now), regular hours
            only. QQQ tracks the same index as NQ minute by minute; costs are
            charged in NQ terms (NQ ~= 41 x QQQ). Keys: export ALPACA_API_KEY
            and ALPACA_SECRET_KEY (or put them in a .env at the repo root).
            Cached per year in data/minute/.
  --csv F : real NQ 1-minute bars (e.g. FirstRate Data), columns
            timestamp(ET),open,high,low,close,volume; run with --mult 1.

THE STRATEGY (one long or short position at a time, flat by the close)
  anchor   = today's VWAP or an N-minute EMA (both restart each morning)
  stretch  z = (price - anchor) / its own recent typical distance
  entry    z <= -E -> buy next minute's open ; z >= +E -> short (side gene)
  exit     back toward the anchor (exit gene: 20% of the way ... 20% past it),
           or a stop = K x (average daily range of the last 10 days),
           or max holding time, or the end of the trading window
  filters  trading window; skip trend days (price already far from today's
           open, in daily-range units); optional "turn" bar (the minute must
           close back toward the anchor); cooldown after each trade

GENOME (12 genes, each 0..1, mapped to these ranges)
  anchor VWAP|EMA . lookback 10-240 min . entry E 1.0-4.0 . exit 20-120% of
  the way back . stop K 0.05-0.8 daily ranges . max hold 5-240 min . window
  start 9:45-14:00 . window length 1-6 h (ends by 15:50) . side long|short|both
  . trend-day filter 0.3-3.0 daily ranges (or off) . turn bar yes|no .
  cooldown 0-30 min

GA (fixes the paper's two weak spots: premature convergence and fragile peaks)
  * 3 independent populations (different seeds), 80 genomes each, up to 40
    generations, stop after 8 with no improvement
  * tournament selection (3), blend crossover (prob 0.8), mutation (prob 0.25
    per gene, sd 0.08), 8 elites kept, 10% fresh random genomes each
    generation to keep diversity
  * FITNESS (training years only) = mean yearly Sharpe - 0.5 x the spread of
    the yearly Sharpes - 0.1 x max drawdown (in units of a year's volatility);
    a big penalty for fewer than 40 or more than 3000 trades a year. Each
    genome is scored as the AVERAGE of itself and 2 slightly jittered copies,
    so the GA prefers wide flat hilltops over one lucky spike.
  * costs inside the fitness: 1 NQ tick slippage + $2.50 commission per side
    (= 0.375 NQ points per side, ~$15 a round trip). --micro: MNQ costs.

THREE PERIODS (nothing leaks forward)
  train      2016-01 -> 2020-12   GA evolves here only
  validate   2021-01 -> 2022-12   the 30 best distinct training genomes are
                                  scored here; the top 5 form the ENSEMBLE
                                  (each trades 1/5 size)
  test       2023-01 -> now       looked at ONCE. Re-running after changing
                                  anything because of the test numbers makes
                                  the test worthless -- decide before running.

PASS (judged on the test period, ensemble, decided before any real run):
  (a) Sharpe >= 1.0 after 1-tick costs
  (b) Sharpe >= 0.5 with one extra tick of slippage each side
  (c) positive in at least 2 of 3 full test years (2023, 2024, 2025)
  (d) a 50/50 equal-risk monthly blend with CHIMERA beats CHIMERA alone
      over their overlap (2023-01 -> 2026-05)
NULL CONTROL (runs automatically): the whole pipeline again on the same data
  with each day's 1-minute moves shuffled (same volatility and daily range,
  no intraday pattern). It must FAIL; its training fitness shows how high
  fitness can climb on pure noise.

Run (repo root):
    python3.10 -m pip install numba          # once (makes the backtest fast)
    python3.10 scripts/nq_mr_ga.py --smoke   # planted edge must pass, noise fail
    python3.10 scripts/nq_mr_ga.py | tee logs/nq_mr_ga.txt
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    from numba import njit
    NUMBA = True
except ImportError:                                     # works, just slow
    NUMBA = False

    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MIN_DIR = os.path.join(ROOT, "data", "minute")
LOG_DIR = os.path.join(ROOT, "logs")
TRAIN = ("2016-01-01", "2021-01-01")
VALID = ("2021-01-01", "2023-01-01")
TEST = ("2023-01-01", "2099-01-01")
NQ_PER_QQQ = 41.0
COST_NQ = 0.25 + 2.50 / 20          # 1 tick + $2.50 commission, NQ points/side
COST_MNQ = 0.25 + 0.62 / 2          # 1 tick + $0.62 commission, MNQ points/side
TICK = 0.25
NG = 12
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
    cur = y == now.year
    if os.path.exists(p) and not (cur and time.time() - os.path.getmtime(p) > 86400):
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


def load_alpaca(sym="QQQ", first=2016):
    key, sec = _env_keys()
    frames = []
    feed = "sip"
    for y in range(first, pd.Timestamp.today().year + 1):
        try:
            frames.append(_alpaca_year(sym, y, key, sec, feed))
        except Exception as e:  # noqa: BLE001
            if feed == "sip":
                print(f"  SIP feed refused ({e}); using IEX feed (thinner volume)")
                feed = "iex"
                frames.append(_alpaca_year(sym, y, key, sec, feed))
            else:
                raise
    return pd.concat([f for f in frames if len(f)]).sort_index()


def load_csv(path):
    d = pd.read_csv(path, header=None)
    try:
        float(d.iloc[0, 1])
    except ValueError:
        d = pd.read_csv(path)
    d = d.iloc[:, :6]
    d.columns = ["t", "open", "high", "low", "close", "volume"]
    d.index = pd.to_datetime(d["t"])
    return d.drop(columns="t").astype(float).sort_index()


def prepare(df):
    """regular hours only -> flat arrays + per-day index (no look-ahead)."""
    df = df[~df.index.duplicated(keep="last")]
    mod = df.index.hour * 60 + df.index.minute
    df = df[(mod >= 570) & (mod < 960)].copy()
    df["mod"] = df.index.hour * 60 + df.index.minute
    day = df.index.normalize()
    cnt = pd.Series(1, df.index).groupby(day).sum()
    good = cnt[cnt >= 300].index                       # drop half days / gaps
    df = df[day.isin(good)]
    day = df.index.normalize()
    codes, dates = pd.factorize(day)
    ds = np.searchsorted(codes, np.arange(len(dates)), side="left")
    de = np.searchsorted(codes, np.arange(len(dates)), side="right")
    h, l = df["high"].values, df["low"].values
    rng = np.array([h[a:b].max() - l[a:b].min() for a, b in zip(ds, de)])
    U = pd.Series(rng).shift(1).rolling(10, min_periods=10).mean().values
    return {"o": df["open"].values.astype(np.float64), "h": h.astype(np.float64),
            "l": l.astype(np.float64), "c": df["close"].values.astype(np.float64),
            "v": np.maximum(df["volume"].values.astype(np.float64), 1.0),
            "mod": df["mod"].values.astype(np.int64), "ds": ds.astype(np.int64),
            "de": de.astype(np.int64), "U": U, "dopen": df["open"].values[ds],
            "dates": pd.DatetimeIndex(dates), "years": pd.DatetimeIndex(dates).year.values}


def shuffled(D, seed=99):
    """null data: each day's 1-minute bars reshuffled (moves, wicks, gaps and
    volumes keep their size; their ORDER inside the day is destroyed)."""
    rng = np.random.default_rng(seed)
    o, h, l, c, v = (D[k].copy() for k in ("o", "h", "l", "c", "v"))
    for a, b in zip(D["ds"], D["de"]):
        prev = np.r_[o[a], c[a:b - 1]]
        gap, ret = o[a:b] / prev, c[a:b] / o[a:b]
        up, dn = h[a:b] / np.maximum(o[a:b], c[a:b]), l[a:b] / np.minimum(o[a:b], c[a:b])
        p = rng.permutation(b - a)
        gap, ret, up, dn, vv = gap[p], ret[p], up[p], dn[p], v[a:b][p]
        gap[0] = 1.0
        px = o[a]
        for i in range(b - a):
            oo = px * gap[i]
            cc = oo * ret[i]
            o[a + i], c[a + i] = oo, cc
            h[a + i], l[a + i] = max(oo, cc) * up[i], min(oo, cc) * dn[i]
            v[a + i] = vv[i]
            px = cc
    E = dict(D)
    E.update(o=o, h=h, l=l, c=c, v=v)
    rngs = np.array([h[a:b].max() - l[a:b].min() for a, b in zip(D["ds"], D["de"])])
    E["U"] = pd.Series(rngs).shift(1).rolling(10, min_periods=10).mean().values
    E["dopen"] = o[D["ds"]]
    return E


# ================================================================== engine
@njit(cache=True)
def sim(o, h, l, c, v, mod, ds, de, U, dopen, anchor, n, ez, xf, sk, mh, t0, t1,
        side, tk, conf, cd, cost):
    nd = ds.shape[0]
    pnl = np.zeros(nd)
    ntr = 0
    a = 2.0 / (n + 1.0)
    for d in range(nd):
        s, e = ds[d], de[d]
        if not (U[d] > 0):
            continue
        pos, ep, stop, held, pend, cool = 0, 0.0, 0.0, 0, 0, 0
        ema, cpv, cv, s2, nb = c[s], 0.0, 0.0, 0.0, 0
        for i in range(s, e):
            last = i == e - 1
            if pend == 9 and pos != 0:                       # exit at this open
                pnl[d] += (o[i] - cost * pos - ep) * pos
                pos, pend, cool = 0, 0, cd
            elif pend == 1 or pend == -1:                    # entry at this open
                pos = pend
                ep = o[i] + cost * pos
                stop = ep - pos * sk * U[d]
                held, pend = 0, 0
                ntr += 1
            if pos == 1 and l[i] <= stop:                    # stop inside the bar
                pnl[d] += min(o[i], stop) - cost - ep
                pos, cool = 0, cd
            elif pos == -1 and h[i] >= stop:
                pnl[d] += ep - (max(o[i], stop) + cost)
                pos, cool = 0, cd
            tp = (h[i] + l[i] + c[i]) / 3.0                  # indicators, bar i
            cpv += tp * v[i]
            cv += v[i]
            ema += a * (c[i] - ema)
            anc = ema
            if anchor == 0:
                anc = cpv / cv
            dev = c[i] - anc
            s2 = (1.0 - a) * s2 + a * dev * dev
            nb += 1
            z = dev / np.sqrt(s2) if s2 > 0 else 0.0
            if pos != 0:
                held += 1
                lvl = -ez + xf * ez                          # long exit level
                ex = (pos == 1 and z >= lvl) or (pos == -1 and z <= -lvl)
                if held >= mh or mod[i] >= t1:
                    ex = True
                if ex and not last:
                    pend = 9
            elif cool > 0:
                cool -= 1
            elif (not last) and nb >= 15 and t0 <= mod[i] < t1 and pend == 0:
                if abs(c[i] - dopen[d]) <= tk * U[d]:
                    upb = c[i] > c[i - 1]
                    dnb = c[i] < c[i - 1]
                    if side != 1 and z <= -ez and (conf == 0 or upb):
                        pend = 1
                    elif side != 0 and z >= ez and (conf == 0 or dnb):
                        pend = -1
            if last and pos != 0:                            # flat by the close
                pnl[d] += (c[i] - cost * pos - ep) * pos
                pos = 0
    return pnl, ntr


def decode(g):
    g = np.clip(np.asarray(g, float), 0, 1)
    t0 = int(585 + g[6] * 255)
    return dict(anchor=0 if g[0] < 0.5 else 1,
                n=int(round(10 * 24 ** g[1])),
                ez=1.0 + 3.0 * g[2], xf=0.2 + 1.0 * g[3], sk=0.05 + 0.75 * g[4],
                mh=int(round(5 * 48 ** g[5])), t0=t0,
                t1=int(min(950, t0 + 60 + 300 * g[7])),
                side=0 if g[8] < 1 / 3 else (1 if g[8] < 2 / 3 else 2),
                tk=1e9 if g[9] > 0.9 else 0.3 + 3.0 * g[9],
                conf=0 if g[10] < 0.5 else 1, cd=int(round(30 * g[11])))


def describe(g):
    p = decode(g)
    hm = lambda m: f"{m // 60}:{m % 60:02d}"  # noqa: E731
    return (f"{'VWAP' if p['anchor'] == 0 else 'EMA'} n={p['n']} | enter |z|>={p['ez']:.2f}"
            f" | exit {p['xf'] * 100:.0f}% back | stop {p['sk']:.2f} rng | hold<={p['mh']}m"
            f" | {hm(p['t0'])}-{hm(p['t1'])} | {['long', 'short', 'both'][p['side']]}"
            f" | trend<={'off' if p['tk'] > 1e8 else f'{p[chr(116)+chr(107)]:.1f}'}"
            f" | turn={'y' if p['conf'] else 'n'} | cool {p['cd']}m")


def run(D, g, days, cost):
    p = decode(g)
    pnl, ntr = sim(D["o"], D["h"], D["l"], D["c"], D["v"], D["mod"], D["ds"][days],
                   D["de"][days], D["U"][days], D["dopen"][days], p["anchor"], p["n"],
                   p["ez"], p["xf"], p["sk"], p["mh"], p["t0"], p["t1"], p["side"],
                   p["tk"], p["conf"], p["cd"], cost)
    return pnl / D["dopen"][days], ntr


def sharpe(r):
    s = r.std()
    return r.mean() / s * np.sqrt(252) if s > 0 and len(r) > 20 else np.nan


def fitness(r, ntr, years):
    if r.std() == 0:
        return -9.0
    ys = [sharpe(r[years == y]) for y in np.unique(years) if (years == y).sum() >= 60]
    ys = [x for x in ys if np.isfinite(x)]
    if not ys:
        return -9.0
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(eq) - eq).max() / (r.std() * np.sqrt(252))
    tpy = ntr / (len(r) / 252)
    pen = 0.0 if 40 <= tpy <= 3000 else 3.0
    return float(np.mean(ys) - 0.5 * np.std(ys) - 0.1 * dd - pen)


# ===================================================================== GA
def ga(D, days, cost, seed, pop=80, gens=40, patience=8, quiet=False):
    rng = np.random.default_rng(seed)
    yrs = D["years"][days]

    def score(g):
        fs = []
        for j in range(3):                      # itself + 2 jittered copies
            gg = g if j == 0 else np.clip(g + rng.normal(0, 0.04, NG), 0, 1)
            r, n = run(D, gg, days, cost)
            fs.append(fitness(r, n, yrs))
        return float(np.mean(fs))

    P = rng.random((pop, NG))
    F = np.array([score(g) for g in P])
    hof = list(zip(F, P.copy()))
    best, flat = F.max(), 0
    for gen in range(gens):
        order = np.argsort(-F)
        kids = [P[i].copy() for i in order[:8]]                 # elites
        while len(kids) < pop - pop // 10:
            pa = P[max(rng.choice(pop, 3), key=lambda i: F[i])]
            pb = P[max(rng.choice(pop, 3), key=lambda i: F[i])]
            if rng.random() < 0.8:                              # blend crossover
                lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
                span = hi - lo
                k = rng.uniform(lo - 0.3 * span, hi + 0.3 * span)
            else:
                k = pa.copy()
            m = rng.random(NG) < 0.25
            k[m] += rng.normal(0, 0.08, m.sum())
            kids.append(np.clip(k, 0, 1))
        while len(kids) < pop:
            kids.append(rng.random(NG))                         # immigrants
        P = np.array(kids)
        F = np.array([F[order[i]] if i < 8 else score(P[i]) for i in range(pop)])
        hof += list(zip(F[8:], P[8:].copy()))
        if F.max() > best + 1e-6:
            best, flat = F.max(), 0
        else:
            flat += 1
        if not quiet:
            print(f"    seed {seed} gen {gen + 1:2d}: best {F.max():+.3f}  "
                  f"median {np.median(F):+.3f}")
        if flat >= patience:
            break
    return hof


def pick(hof, k=30, dist=0.08):
    hof = sorted(hof, key=lambda t: -t[0])
    out = []
    for f, g in hof:
        if all(np.abs(g - h).max() > dist for _, h in out):
            out.append((f, g))
        if len(out) == k:
            break
    return out


def period(D, a, b):
    return np.flatnonzero((D["dates"] >= a) & (D["dates"] < b))


def pipeline(D, cost, label, pop, gens, seeds, quiet):
    tr, va, te = period(D, *TRAIN), period(D, *VALID), period(D, *TEST)
    print(f"\n  ===== {label}: train {len(tr)} days, validate {len(va)}, test {len(te)}")
    t = time.time()
    hof = []
    for s in seeds:
        hof += ga(D, tr, cost, s, pop, gens, quiet=quiet)
    cands = pick(hof)
    print(f"  GA: {len(hof)} genomes scored in {time.time() - t:.0f}s; best training "
          f"fitness {cands[0][0]:+.2f}")
    vs = []
    for f, g in cands:
        r, n = run(D, g, va, cost)
        vs.append((sharpe(r), f, g))
    vs.sort(key=lambda x: -np.nan_to_num(x[0], nan=-9))
    top = vs[:5]
    for i, (sv, f, g) in enumerate(top):
        print(f"   #{i + 1} train fit {f:+.2f}  valid Sharpe {sv:+.2f}  {describe(g)}")

    def ens(days, c):
        rs, nt = [], 0
        for _, _, g in top:
            r, n = run(D, g, days, c)
            rs.append(r)
            nt += n
        return np.mean(rs, axis=0), nt / len(top)
    out = {}
    for nm, days in (("train", tr), ("validate", va), ("test", te)):
        r, nt = ens(days, cost)
        r2, _ = ens(days, cost + TICK * D["unit"])
        yrs = D["years"][days]
        out[nm] = (r, days)
        line = f"  ENSEMBLE {nm:<8} Sharpe {sharpe(r):+.2f} (2 ticks {sharpe(r2):+.2f})" \
            f"  ann {r.mean() * 252 * 100:+5.1f}%/contract-notional  " \
            f"trades/yr {nt / (len(days) / 252):5.0f}"
        if nm == "test":
            ys = {y: sharpe(r[yrs == y]) for y in np.unique(yrs)}
            line += "\n    by year: " + "  ".join(f"{y} {s:+.2f}" for y, s in ys.items())
            full = [s for y, s in ys.items() if (yrs == y).sum() >= 240]
            m = pd.Series(r, D["dates"][days])
            m = ((1 + m).resample("ME").prod() - 1)
            m.index = m.index.strftime("%Y-%m")
            j = pd.concat([pd.Series(CHIM), m], axis=1, keys=["c", "s"]).dropna()
            if len(j) >= 12 and j["s"].std() > 0:
                sh = lambda x: x.mean() / x.std() * np.sqrt(12)  # noqa: E731
                b = 0.5 * j["c"] + 0.5 * j["s"] * j["c"].std() / j["s"].std()
                csh, bsh, cor = sh(j["c"]), sh(b), j["c"].corr(j["s"])
            else:
                csh = bsh = cor = np.nan
            line += f"\n    vs CHIMERA ({len(j)} months): corr {cor:+.2f}, CHIM {csh:.2f}" \
                f" -> blend {bsh:.2f}"
            ok = [sharpe(r) >= 1.0, sharpe(r2) >= 0.5,
                  sum(s > 0 for s in full) >= 2 if len(full) >= 3 else False, bsh > csh]
            line += f"\n    PASS checks a/b/c/d: {['yes' if x else 'no' for x in ok]}" \
                f"  ->  {'PASS' if all(ok) else 'FAIL'}"
        print(line)
    return [g.tolist() for _, _, g in top], out


# =================================================================== smoke
def synth(planted, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", "2026-10-30")
    mins = np.arange(570, 960)
    rows_t, O, H, L, C, V = [], [], [], [], [], []
    px = 120.0
    for d in dates:
        drift = rng.normal(0, 0.0004, len(mins)).cumsum()
        x = np.zeros(len(mins))
        if planted:
            for i in range(1, len(mins)):
                x[i] = x[i - 1] * 0.95 + rng.normal(0, 0.0006)
        lp = np.log(px) + drift + x
        c = np.exp(lp)
        o = np.r_[px, c[:-1]]
        wig = np.abs(rng.normal(0, 0.0002, (2, len(mins))))
        O.append(o)
        C.append(c)
        H.append(np.maximum(o, c) * (1 + wig[0]))
        L.append(np.minimum(o, c) * (1 - wig[1]))
        V.append(rng.integers(500, 5000, len(mins)))
        rows_t.append(d + pd.to_timedelta(mins, unit="m"))
        px = c[-1] * np.exp(rng.normal(0, 0.004))           # overnight gap
    idx = pd.DatetimeIndex(np.concatenate(rows_t))
    return pd.DataFrame({"open": np.concatenate(O), "high": np.concatenate(H),
                         "low": np.concatenate(L), "close": np.concatenate(C),
                         "volume": np.concatenate(V).astype(float)}, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--csv", help="NQ 1-minute CSV instead of Alpaca QQQ")
    ap.add_argument("--mult", type=float, default=None,
                    help="NQ points per price unit (QQQ 41, NQ csv 1)")
    ap.add_argument("--micro", action="store_true", help="MNQ costs")
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gens", type=int, default=40)
    a = ap.parse_args()
    if not NUMBA:
        print("  numba not installed -> very slow. python3.10 -m pip install numba")
    cpts = COST_MNQ if a.micro else COST_NQ
    if a.smoke:
        for lab, pl in (("SMOKE PLANTED (must PASS)", True), ("SMOKE NOISE (must FAIL)", False)):
            D = prepare(synth(pl))
            D["unit"] = 1 / NQ_PER_QQQ
            pipeline(D, cpts / NQ_PER_QQQ, lab, 40, 15, [1], quiet=True)
        return
    if a.csv:
        df, mult = load_csv(a.csv), a.mult or 1.0
        src = f"NQ csv {os.path.basename(a.csv)}"
    else:
        df, mult = load_alpaca("QQQ"), a.mult or NQ_PER_QQQ
        src = "QQQ 1-min (Alpaca), NQ costs"
    D = prepare(df)
    D["unit"] = 1 / mult
    print(f"  data: {src}: {len(D['c']):,} bars, {len(D['dates'])} days "
          f"{D['dates'][0].date()} -> {D['dates'][-1].date()}; cost/side "
          f"{cpts:.3f} NQ pts ({'MNQ' if a.micro else 'NQ'})")
    top, _ = pipeline(D, cpts / mult, "REAL DATA", a.pop, a.gens, [1, 2, 3], quiet=False)
    os.makedirs(LOG_DIR, exist_ok=True)
    json.dump({"source": src, "genomes": top, "decoded": [decode(g) for g in top]},
              open(os.path.join(LOG_DIR, "nq_mr_ga_genomes.json"), "w"), indent=1,
              default=float)
    if not a.no_null:
        N = shuffled(D)
        N["unit"] = D["unit"]
        pipeline(N, cpts / mult, "NULL CONTROL (shuffled minutes, must FAIL)",
                 a.pop, a.gens, [1, 2, 3], quiet=True)


if __name__ == "__main__":
    main()
