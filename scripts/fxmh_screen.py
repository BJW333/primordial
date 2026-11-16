#!/usr/bin/env python3
"""fxmh_screen.py -- multi-horizon sign vote (mh_vote) on G10 FX, daily.
REGISTERED 2026-11-15 (prereg/FXMH_PREREG.md) BEFORE any real-data run.
Second ground for a hypothesis already dead on crypto daily (2026-09-09).

Signal: four DISJOINT trading-day segments [0,5) [5,10) [10,21) [21,42);
each votes +1/-1 on its return, 0 inside a vol dead zone (0.5 * sigma_d *
sqrt(seglen), sigma_d = 90d rolling std). Score in [-4, +4]. First
TWO-SIDED test of the idea: up-break + score >= +2 -> long, down-break +
score <= -2 -> short (FX symmetric, no borrow).

Rows (4 judged + controls):
  BASE        Donchian20 two-sided, no vote
  VOTE_ALONE  score crossing to |2|, no breakout
  GATE2       Donchian AND |score| >= 2 same sign      <- the idea
  GATE4       Donchian AND score = +/-4 (unanimous)
  NULL        GATE2 on per-currency shuffled daily returns (must fail)

All: 2.5 x ATR14 stop, 40 trading-day max hold, one position per
currency, 10% ann vol target per position (60d), fx_screen cost table
per side, every row also at 2x costs.

Clock (fx_screen convention): signals from 16:00-NY daily snapshots,
fills at the SAME day's 17:00-NY price.

Usage:
    python3.10 scripts/fxmh_screen.py --smoke noise      # all rows fail
    python3.10 scripts/fxmh_screen.py --smoke planted    # rows pass, NULL fails
    python3.10 scripts/fxmh_screen.py                    # real: data/fx_1h/
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

CCYS = ["AUD", "NZD", "CAD", "NOK", "EUR", "GBP", "JPY", "CHF"]
COST_BP = {"EUR": .7, "GBP": .9, "CHF": .7, "JPY": .9,
           "AUD": .9, "NZD": 1.0, "CAD": .8, "NOK": 3.0}   # per side
SEGS = [(0, 5), (5, 10), (10, 21), (21, 42)]
DEAD = 0.5
DON_N, ATR_N, STOP_A, MAX_HOLD = 20, 14, 2.5, 40
VOL_WIN, POS_VOL = 60, 0.10
W1 = ("2012-07-01", "2021-10-31")       # bar >= 0.5
W2_START = "2021-11-01"                 # bar >= 0.7
NY = "America/New_York"

# CHIMERA monthly returns, copied verbatim from scripts/fx_screen.py
CHIM = {"2017-02": 0.05638, "2017-03": -0.00502, "2017-04": 0.0516, "2017-05": 0.05944, "2017-06": -0.04716, "2017-07": 0.08036, "2017-08": 0.04121, "2017-09": -0.0082, "2017-10": 0.01275, "2017-11": 0.06163, "2017-12": 0.00984, "2018-01": 0.05409, "2018-02": -0.04113, "2018-03": -0.00773, "2018-04": -0.00085, "2018-05": 0.02454, "2018-06": -0.02166, "2018-07": -0.03758, "2018-08": 0.04259, "2018-09": 0.01471, "2018-10": -0.09646, "2018-11": -0.00262, "2018-12": -0.03574, "2019-01": 0.05253, "2019-02": 0.02498, "2019-03": 0.01921, "2019-04": -0.01715, "2019-05": -0.0174, "2019-06": 0.06728, "2019-07": 0.04436, "2019-08": 0.01406, "2019-09": -0.03595, "2019-10": 0.02579, "2019-11": 0.03318, "2019-12": 0.08462, "2020-01": 0.00542, "2020-02": -0.0347, "2020-03": -0.03995, "2020-04": 0.10273, "2020-05": 0.02383, "2020-06": 0.08567, "2020-07": 0.04235, "2020-08": -0.0517, "2020-09": -0.00151, "2020-10": -0.00627, "2020-11": 0.17142, "2020-12": 0.03592, "2021-01": 0.20194, "2021-02": 0.1025, "2021-03": 0.05749, "2021-04": 0.04693, "2021-05": 0.0169, "2021-06": 0.00755, "2021-07": 0.01409, "2021-08": 0.02306, "2021-09": -0.03898, "2021-10": 0.06648, "2021-11": -0.02027, "2021-12": 0.05834, "2022-01": -0.13833, "2022-02": 0.03776, "2022-03": 0.07138, "2022-04": 0.04416, "2022-05": 0.01971, "2022-06": -0.03329, "2022-07": 0.01483, "2022-08": 0.04692, "2022-09": 0.00549, "2022-10": 0.01113, "2022-11": -0.07182, "2022-12": -0.03874, "2023-01": -0.00446, "2023-02": 0.00744, "2023-03": -0.01724, "2023-04": 0.00149, "2023-05": 0.04634, "2023-06": -0.00555, "2023-07": 0.01982, "2023-08": -0.02555, "2023-09": -0.03562, "2023-10": -0.05146, "2023-11": 0.03769, "2023-12": 0.07371, "2024-01": 0.04383, "2024-02": 0.10113, "2024-03": 0.02381, "2024-04": -0.00117, "2024-05": 0.09167, "2024-06": 0.00586, "2024-07": 0.03194, "2024-08": 0.02881, "2024-09": 0.02354, "2024-10": -0.01373, "2024-11": 0.05931, "2024-12": -0.05531, "2025-01": 0.04376, "2025-02": -0.03464, "2025-03": 0.01805, "2025-04": 0.01726, "2025-05": 0.0183, "2025-06": 0.02993, "2025-07": 0.00038, "2025-08": 0.00485, "2025-09": 0.17491, "2025-10": -0.00345, "2025-11": 0.01746, "2025-12": -0.00032, "2026-01": 0.11679, "2026-02": 0.11839, "2026-03": 0.04071, "2026-04": 0.05341, "2026-05": 0.00883}


# ---------------------------------------------------------------- data
INV = {"CAD", "NOK", "JPY", "CHF"}   # cached as USDXXX; invert to USD-per-unit
                                      # (same INST map as fx_screen.py)


def load_hourly(ccy):
    """data/fx_1h/<CCY>.csv (fx_screen cache): UTC-naive time,open,high,
    low,close,volume. Inverted pairs flipped to USD-per-unit, with
    high/low swapped (1/low is the high)."""
    p = f"data/fx_1h/{ccy}.csv"
    if not os.path.exists(p):
        sys.exit(f"{p} missing -- run scripts/fx_screen.py once to build "
                 "the Dukascopy cache, or fix the path.")
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df[(df["volume"] > 0) & (df["close"] > 0)]
    df.index = df.index.tz_localize("UTC")
    out = pd.DataFrame({"close": df["close"].astype(float),
                        "high": df["high"].astype(float),
                        "low": df["low"].astype(float)})
    if ccy in INV:
        out = pd.DataFrame({"close": 1 / out["close"],
                            "high": 1 / out["low"],
                            "low": 1 / out["high"]})
    return out.sort_index()


def to_daily(h):
    """16:00-NY signal snapshot + 17:00-NY fill price + daily H/L."""
    loc = h.tz_convert(NY)
    day = loc.index.date
    g = loc.groupby(day)
    def at_hour(hr):
        s = loc[loc.index.hour == hr]["close"]
        return s.groupby(s.index.date).last()
    d = pd.DataFrame({"sig": at_hour(16), "fill": at_hour(17),
                      "high": g["high"].max(), "low": g["low"].min()})
    d.index = pd.to_datetime(d.index)
    d = d[d.index.dayofweek < 5].dropna(subset=["sig"])
    d["fill"] = d["fill"].fillna(d["sig"])
    return d


# ---------------------------------------------------------------- signal
def mh_vote(sig):
    r = sig.pct_change()
    sd = r.rolling(90, min_periods=30).std()
    vote = pd.Series(0.0, index=sig.index)
    for a, b in SEGS:
        seg = sig.shift(a) / sig.shift(b) - 1
        dz = DEAD * sd * math.sqrt(b - a)
        vote = vote + np.sign(seg).where(seg.abs() >= dz, 0.0).fillna(0.0)
    return vote


def features(d):
    d = d.copy()
    d["vote"] = mh_vote(d["sig"])
    d["dhi"] = d["high"].rolling(DON_N).max().shift(1)
    d["dlo"] = d["low"].rolling(DON_N).min().shift(1)
    pc = d["sig"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(),
                    (d["low"] - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_N).mean()
    d["up"] = (d["sig"] > d["dhi"]) & (d["sig"].shift(1) <= d["dhi"].shift(1))
    d["dn"] = (d["sig"] < d["dlo"]) & (d["sig"].shift(1) >= d["dlo"].shift(1))
    d["vup"] = (d["vote"] >= 2) & (d["vote"].shift(1) < 2)
    d["vdn"] = (d["vote"] <= -2) & (d["vote"].shift(1) > -2)
    d["pvol"] = d["sig"].pct_change().rolling(VOL_WIN).std() * math.sqrt(252)
    return d


def entries(d, row):
    if row == "BASE":
        return d["up"], d["dn"]
    if row == "VOTE_ALONE":
        return d["vup"], d["vdn"]
    if row == "GATE2":
        return d["up"] & (d["vote"] >= 2), d["dn"] & (d["vote"] <= -2)
    if row == "GATE4":
        return d["up"] & (d["vote"] >= 4), d["dn"] & (d["vote"] <= -4)
    raise ValueError(row)


# ---------------------------------------------------------------- engine
def run_ccy(d, row, cost_bp):
    """One currency, one row. Daily pnl series (vol-targeted notional)."""
    lo, sh = entries(d, row)
    pnl = pd.Series(0.0, index=d.index)
    pos = 0; i = 0; idx = d.index
    ent = stop = notion = 0.0; held = 0
    fill = d["fill"].values; low = d["low"].values; high = d["high"].values
    atr = d["atr"].values; pv = d["pvol"].values
    lov, shv = lo.values, sh.values
    n_ent = 0
    for i in range(1, len(idx)):
        if pos != 0:
            held += 1
            px0, px1 = fill[i - 1], fill[i]
            exit_px = None
            if pos > 0 and low[i] <= stop:
                exit_px = min(stop, px1)
            elif pos < 0 and high[i] >= stop:
                exit_px = max(stop, px1)
            elif held >= MAX_HOLD:
                exit_px = px1
            end = exit_px if exit_px is not None else px1
            pnl.iloc[i] += pos * notion * (end - px0) / px0
            if exit_px is not None:
                pnl.iloc[i] -= notion * cost_bp * 1e-4
                pos = 0
        if pos == 0 and not (np.isnan(atr[i]) or np.isnan(pv[i]) or pv[i] <= 0):
            sgn = 1 if lov[i] else (-1 if shv[i] else 0)
            if sgn:
                pos = sgn; ent = fill[i]; held = 0; n_ent += 1
                stop = ent - sgn * STOP_A * atr[i]
                notion = min(POS_VOL / pv[i], 3.0)
                pnl.iloc[i] -= notion * cost_bp * 1e-4
    return pnl, n_ent


def portfolio(data, row, mult=1.0, random_seed=None):
    """random_seed: RANDOM-entry control -- same prices, same exits, but
    entries moved to random dates matching the row's per-currency entry
    count and long/short mix. Tests the signal's TIMING; must fail."""
    rng = np.random.default_rng(random_seed)
    parts = []
    for c, d0 in data.items():
        d = features(d0)
        if random_seed is not None:
            lo, sh = entries(d, row)
            ok = d["atr"].notna() & d["pvol"].notna() & (d["pvol"] > 0)
            cand = np.flatnonzero(ok.values)
            nl, ns = int(lo.sum()), int(sh.sum())
            pick = rng.choice(cand, size=min(nl + ns, len(cand)),
                              replace=False)
            rl = pd.Series(False, index=d.index)
            rs = pd.Series(False, index=d.index)
            rl.iloc[pick[:nl]] = True
            rs.iloc[pick[nl:]] = True
            d = d.copy()
            d["up"], d["dn"] = rl, rs
            d["vup"], d["vdn"] = rl, rs
            d["vote"] = pd.Series(np.where(rl, 4.0,
                                  np.where(rs, -4.0, 0.0)), index=d.index)
        parts.append(run_ccy(d, row, COST_BP[c] * mult))
    pnl = pd.concat([p for p, _ in parts], axis=1).fillna(0.0).sum(axis=1)
    return pnl, sum(n for _, n in parts)


# ---------------------------------------------------------------- stats
def sharpe(p):
    p = p[p.index.dayofweek < 5]
    sd = p.std()
    return float(p.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0


def ann(p):
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    return float(p.sum() / yrs) if yrs > 0 else 0.0


MIN_TRADES = 100  # whole-period; starved rows cannot pass (noise calib)


def verdict(p, p2x, blend, trades):
    s1 = sharpe(p.loc[W1[0]:W1[1]])
    s2 = sharpe(p.loc[W2_START:])
    s2x = sharpe(p2x)
    ok = s1 >= 0.5 and s2 >= 0.7 and s2x > 0 and trades >= MIN_TRADES and \
        (blend is None or blend > 1.42)
    return s1, s2, s2x, ok


def chimera_blend(p):
    """fx_screen's blend math: 50/50 equal-risk monthly with CHIM."""
    m = p.groupby(p.index.to_period("M")).sum()
    m.index = m.index.strftime("%Y-%m")
    j = pd.concat([pd.Series(CHIM), m], axis=1, keys=["c", "s"]).dropna()
    if len(j) < 36 or j["s"].std() == 0:
        return None
    b = 0.5 * j["c"] + 0.5 * j["s"] * (j["c"].std() / j["s"].std())
    return float(b.mean() / b.std() * math.sqrt(12))


# ---------------------------------------------------------------- smoke
def synth(mode, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-02", "2026-10-31", freq="B")
    n = len(idx)
    if mode == "noise":
        r = rng.normal(0, 0.006, n)
    else:  # planted: modest persistent drift regimes -- strong enough
        # for trend-TIMED entries, weak enough that random timing fails
        drift = np.repeat(rng.choice([-1, 1], size=n // 120 + 1) * 0.0009,
                          120)[:n]
        r = drift + rng.normal(0, 0.005, n)
    s = pd.Series(100 * np.cumprod(1 + r), index=idx)
    return pd.DataFrame({"sig": s, "fill": s * (1 + rng.normal(0, 2e-4, n)),
                         "high": s * 1.004, "low": s * 0.996})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", choices=["noise", "planted"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-seeds", type=int, default=0,
                    help="run N extra noise universes, report false passes")
    a = ap.parse_args()

    if a.smoke:
        data = {c: synth(a.smoke, 100 * (a.seed + 1) + i)
                for i, c in enumerate(CCYS)}
        print(f"[smoke {a.smoke} seed {a.seed}]")
    else:
        data = {c: to_daily(load_hourly(c)) for c in CCYS}
        print("[real data] data/fx_1h ->",
              {c: f"{d.index[0].date()}..{d.index[-1].date()}"
               for c, d in list(data.items())[:2]}, "...")

    print(f"{'row':<11} {'2012-21':>8} {'2021-now':>9} {'whole':>6} "
          f"{'2x-shp':>7} {'ann':>7} {'tr':>5} {'blend':>6}  verdict")
    whole = {}
    for row in ("BASE", "VOTE_ALONE", "GATE2", "GATE4"):
        p, tr = portfolio(data, row)
        p2, _ = portfolio(data, row, mult=2.0)
        bl = chimera_blend(p) if not a.smoke else None
        s1, s2, s2x, ok = verdict(p, p2, bl, tr)
        whole[row] = sharpe(p)
        print(f"{row:<11} {s1:>8.2f} {s2:>9.2f} {whole[row]:>6.2f} "
              f"{s2x:>7.2f} {ann(p):>6.1%} {tr:>5} "
              f"{bl if bl is not None else float('nan'):>6.2f}"
              f"  {'PASS' if ok else 'fail'}")
    p, tr = portfolio(data, "GATE2", random_seed=7)
    p2, _ = portfolio(data, "GATE2", mult=2.0, random_seed=7)
    s1, s2, s2x, ok = verdict(p, p2, None, tr)
    rw = sharpe(p)
    print(f"{'RANDOM':<11} {s1:>8.2f} {s2:>9.2f} {rw:>6.2f} {s2x:>7.2f} "
          f"{ann(p):>6.1%} {tr:>5} {'':>6}  "
          + ("(judged relatively below)" if a.smoke == "planted"
             else ("PASS (BAD!)" if ok else "fail (correct)")))
    if a.smoke == "planted":
        # the exit bracket harvests planted drift even with random timing,
        # so the planted criterion is RELATIVE: every signal row must beat
        # RANDOM's whole-period Sharpe by >= 0.5.
        sep = all(w > rw + 0.5 for w in whole.values())
        print(f"[planted verdict] rows beat RANDOM by >=0.5 whole-Sharpe: "
              f"{'PASS' if sep else 'FAIL'} "
              f"(RANDOM {rw:.2f} vs rows {min(whole.values()):.2f}..)"
              )
    if a.smoke == "noise":
        print("[noise verdict] all rows must print fail above; RANDOM too.")
    if a.noise_seeds:
        fp = {r: 0 for r in ("BASE", "VOTE_ALONE", "GATE2", "GATE4")}
        for sd in range(a.noise_seeds):
            nd = {c: synth("noise", 9000 + 100 * sd + i)
                  for i, c in enumerate(CCYS)}
            for row in fp:
                p, tr = portfolio(nd, row)
                p2, _ = portfolio(nd, row, mult=2.0)
                _, _, _, ok = verdict(p, p2, None, tr)
                fp[row] += ok
        print(f"[noise calibration, {a.noise_seeds} universes] "
              f"false passes: {fp}")


if __name__ == "__main__":
    main()
