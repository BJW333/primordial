"""
fx_grid.py -- the TikTok grid bot, rebuilt on the AUD/NZD gap instead of gold.
PRE-REGISTERED 2026-11-15, before any real-data run. No tuning after results.

Why the gap: the video's gold grid made its money from gold tripling (1,330 -> 4,000) and blew up
14 times when it doubled down. A currency gap has no long-run drift, so a grid on it only earns
if the gap really snaps back. Fixed size per step -- NO doubling, ever.

Data: same cache as scripts/fx_screen.py (data/fx_1h/, Dukascopy hourly 2004 -> 2026-10).
      Run fx_screen.py first (it downloads), or this script downloads on first run.

The grid (hourly, AUD/NZD gap = log AUD - log NZD)
  anchor  = 20-day (480-hour) average of the gap
  step    = 0.5 x the std of the gap's 1-day change over the last 60 days (1,440 hours)
  gap k steps ABOVE the anchor -> hold k short units (short AUD / long NZD); k steps BELOW -> k long
  each unit closes one step back toward the anchor (classic grid take-profit); max 5 units a side
  unit = $0.20 long one leg + $0.20 short the other (5 units = $1 / $1, same as fx_screen PAIR)
  orders go in one hour after the signal close; CME futures costs per leg (fx_screen.COST_BP)

Variants (lockouts, fixed now)
  G1 NO_STOP     the grid as above, nothing else
  G2 BASKET_STOP close every unit when the gap reaches 6 steps (one past the last level);
                 that side stays locked until the gap is back within 1 step of the anchor
  G3 STOP+NEWS   G2, and no NEW units within 5 trading days of an RBA or RBNZ decision
                 (calendar from fx_screen; NOT RUN if the RBNZ dates can't be loaded)
  G4 STOP+VOL    G2, and no NEW units while the gap's 5-day volatility is > 2x its 1-year median
  G2_BLOC6       G2 on all six pairs of AUD NZD CAD NOK, 1/6 each (replication, not a separate pass)
Control (must FAIL)
  RANDOM_GAP     G2 on a fake AUD/NZD gap built from the real hourly gap moves shuffled in time
                 (same size moves, no snap-back)
Pass bar (same as always): Sharpe >= 0.5 2012-07 -> 2021-10, >= 0.7 2021-11 -> end, 50/50 equal-risk
  monthly blend with CHIMERA > 1.42, whole-period Sharpe > 0 at 2x costs.
  Four variants are judged -> one pass is weaker evidence than it looks (about 4 tries).
Also printed: basket stops ("wipeouts" in the video's sense), time in market, worst single episode.

Run (repo root):
    python3.10 scripts/fx_grid.py --smoke
    python3.10 scripts/fx_grid.py | tee logs/fx_grid.txt
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx_screen as fx  # noqa: E402

UNIT = 0.2
MAX_UNITS = 5
STOP_K = 6


def grid_units(dev, step, stop_k=None, allow=None):
    """Signed units held after each hourly close (+ = long the gap, - = short the gap)."""
    n = len(dev)
    out = np.zeros(n)
    stops = np.zeros(n, bool)
    al = np.ones(n, bool) if allow is None else allow
    u = 0
    locked = {1: False, -1: False}           # 1 = gap above anchor (short side), -1 = below
    for i in range(n):
        d, s = dev[i], step[i]
        if np.isnan(d) or np.isnan(s) or s <= 0:
            u = 0
            continue
        k = d / s
        for side in (1, -1):
            if locked[side] and side * k < 1:
                locked[side] = False
        if stop_k is not None and u != 0 and abs(k) >= stop_k and np.sign(k) == -np.sign(u):
            u = 0
            locked[int(np.sign(k))] = True
            stops[i] = True
        while u < 0 and k <= (-u - 1):        # short units take profit one step back
            u += 1
        while u > 0 and -k <= (u - 1):        # long units take profit one step back
            u -= 1
        if al[i]:
            while u <= 0 and not locked[1] and -u < MAX_UNITS and k >= (-u + 1):
                u -= 1
            while u >= 0 and not locked[-1] and u < MAX_UNITS and -k >= (u + 1):
                u += 1
        out[i] = u
    return out, stops


def gap_inputs(P, a, b):
    g = np.log(P[a]) - np.log(P[b])
    anchor = g.rolling(480, min_periods=240).mean()
    step = 0.5 * g.diff(24).rolling(1440, min_periods=720).std()
    return g, (g - anchor).values, step.values


def vol_allow(g):
    rv = g.diff().rolling(120, min_periods=60).std()
    med = rv.rolling(6048, min_periods=2000).median()
    return ~(rv > 2 * med).values


def hourly_news_allow(idx, rbnz):
    days = pd.DatetimeIndex(sorted(set(idx.tz_convert("America/New_York").normalize()
                                       .tz_localize(None))))
    ok = fx.news_allow(days, fx.rba_dates() + list(rbnz), news_day=False)
    d = pd.DatetimeIndex(idx.tz_convert("America/New_York").normalize().tz_localize(None))
    return ok.reindex(d).fillna(True).values


def pair_book(P, R, a, b, cmult, stop_k=None, allow=None):
    _, dev, step = gap_inputs(P, a, b)
    u, stops = grid_units(dev, step, stop_k, allow)
    us = pd.Series(u, index=P.index).shift(1).fillna(0)          # one-hour order delay
    pos = pd.DataFrame({a: us * UNIT, b: -us * UNIT})
    return fx.hourly_book(pos, R, cmult), u, stops


def episodes(u, pnl_h):
    """worst loss from opening units to flat again (one 'cycle')."""
    worst, cur, open_ = 0.0, 0.0, False
    for x, p in zip(u, pnl_h):
        if x != 0:
            open_ = True
            cur += p
        elif open_:
            worst, cur, open_ = min(worst, cur), 0.0, False
    return min(worst, cur)


def control_P(P, seed=3):
    g = np.log(P["AUD"]) - np.log(P["NZD"])
    dg = g.diff().fillna(0).values
    rng = np.random.default_rng(seed)
    g2 = g.iloc[0] + np.cumsum(rng.permutation(dg))
    Q = P.copy()
    Q["AUD"] = P["NZD"] * np.exp(g2)
    return Q


def run(P, rbnz):
    R = fx.hourly_returns(P)
    g = np.log(P["AUD"]) - np.log(P["NZD"])
    news = hourly_news_allow(P.index, rbnz) if rbnz is not None else None
    rows = {
        "G1 NO_STOP": dict(stop_k=None, allow=None),
        "G2 BASKET_STOP": dict(stop_k=STOP_K, allow=None),
        "G3 STOP+NEWS": dict(stop_k=STOP_K, allow=news) if news is not None else None,
        "G4 STOP+VOL": dict(stop_k=STOP_K, allow=vol_allow(g)),
    }
    print(f"\n  data {P.index[0].date()} -> {P.index[-1].date()}  (AUD/NZD grid, max {MAX_UNITS} "
          f"units of ${UNIT:.2f}/${UNIT:.2f}, no doubling)")
    print(f"  {'variant':<17}| 2012-21 ann  shp | last5y ann  vol  shp  worst | whole shp 2x-cost |"
          f" corrCHIM CHIM blend | stops in-mkt worst-cycle | verdict")

    res = {}
    for name, kw in rows.items():
        if kw is None:
            print(f"  {name:<17}| NOT RUN (no RBNZ calendar)")
            continue
        x, u, st = pair_book(P, R, "AUD", "NZD", 1.0, **kw)
        x2, _, _ = pair_book(P, R, "AUD", "NZD", 2.0, **kw)
        # worst cycle in account terms (hourly pnl, before daily summing)
        _, dev, step = gap_inputs(P, "AUD", "NZD")
        us = pd.Series(u, index=P.index).shift(1).fillna(0)
        ph = (us * UNIT * (R["AUD"] - R["NZD"])).values
        wc = episodes(us.values, ph)
        a0, _, s0, _ = fx.stats(fx.win(x, fx.EARLY))
        a1, v1, s1, d1 = fx.stats(fx.win(x, fx.RECENT))
        _, _, sw, _ = fx.stats(x)
        _, _, sw2, _ = fx.stats(x2)
        csh, bsh, corr, _ = fx.blend(x)
        ok = (s0 >= 0.5) and (s1 >= 0.7) and (bsh > csh) and (sw2 > 0)
        res[name] = ok
        print(f"  {name:<17}| {a0*100:+6.1f}% {s0:5.2f} | {a1*100:+6.1f}% {v1*100:4.1f}% {s1:5.2f}"
              f" {d1*100:5.0f}% | {sw:6.2f} {sw2:7.2f} | {corr:+6.2f} {csh:5.2f} {bsh:5.2f} |"
              f" {int(st.sum()):5d} {(u != 0).mean()*100:5.0f}% {wc*100:+7.1f}% | "
              f"{'PASS' if ok else 'fail'}")

    # replication: G2 on all six bloc pairs
    parts, parts2 = [], []
    for a, b in itertools.combinations(fx.BLOC, 2):
        parts.append(pair_book(P, R, a, b, 1.0, stop_k=STOP_K)[0])
        parts2.append(pair_book(P, R, a, b, 2.0, stop_k=STOP_K)[0])
    x = pd.concat(parts, axis=1).fillna(0).mean(axis=1)
    x2 = pd.concat(parts2, axis=1).fillna(0).mean(axis=1)
    a0, _, s0, _ = fx.stats(fx.win(x, fx.EARLY))
    a1, v1, s1, d1 = fx.stats(fx.win(x, fx.RECENT))
    _, _, sw, _ = fx.stats(x)
    _, _, sw2, _ = fx.stats(x2)
    csh, bsh, corr, _ = fx.blend(x)
    print(f"  {'G2_BLOC6 (repl.)':<17}| {a0*100:+6.1f}% {s0:5.2f} | {a1*100:+6.1f}% {v1*100:4.1f}%"
          f" {s1:5.2f} {d1*100:5.0f}% | {sw:6.2f} {sw2:7.2f} | {corr:+6.2f} {csh:5.2f} {bsh:5.2f} |")

    # control
    Q = control_P(P)
    RQ = fx.hourly_returns(Q)
    x, u, st = pair_book(Q, RQ, "AUD", "NZD", 1.0, stop_k=STOP_K)
    x2, _, _ = pair_book(Q, RQ, "AUD", "NZD", 2.0, stop_k=STOP_K)
    a0, _, s0, _ = fx.stats(fx.win(x, fx.EARLY))
    a1, v1, s1, d1 = fx.stats(fx.win(x, fx.RECENT))
    _, _, sw, _ = fx.stats(x)
    _, _, sw2, _ = fx.stats(x2)
    csh, bsh, corr, _ = fx.blend(x)
    cok = (s0 >= 0.5) and (s1 >= 0.7) and (bsh > csh) and (sw2 > 0)
    print(f"  {'control RANDOM_GAP':<17}| {a0*100:+6.1f}% {s0:5.2f} | {a1*100:+6.1f}% {v1*100:4.1f}%"
          f" {s1:5.2f} {d1*100:5.0f}% | {sw:6.2f} {sw2:7.2f} | {corr:+6.2f} {csh:5.2f} {bsh:5.2f} |"
          f" {int(st.sum()):5d} {(u != 0).mean()*100:5.0f}%         | {'PASS' if cok else 'fail'}")
    print("  control: " + ("fails (good)" if not cok else "PASSED -> the test setup is suspect"))
    print("  ann = average yearly return (log) at max $1/$1; stops = basket stops; in-mkt = share of "
          "hours holding units;\n  worst-cycle = biggest loss from first unit opened to flat again. "
          "pass = 2012-21 >= 0.5, last5y >= 0.7, blend > CHIM, 2x-cost > 0.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        for planted in (False, True):
            print(f"\n===== SMOKE {'PLANTED snap-back gap (G1-G4 should PASS, control fail)' if planted else 'NOISE (everything should fail)'}")
            P, rbnz = fx.synth(planted)
            run(P, rbnz)
        return
    P = fx.load_real(fx.BLOC)          # the grid only needs AUD NZD CAD NOK
    run(P, fx.rbnz_dates())


if __name__ == "__main__":
    main()
