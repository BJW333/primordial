"""
fx_active.py -- ACTIVE FX (5+ trades a week). PRE-REGISTERED 2026-11-15, before any real-data run.

  1 FIXREV   The dollar rises into the big daily fixes and falls back after (Krohn-Mueller-Whelan,
             "FX fixings and returns around the clock", 1999-2019: ~2 bp per swing, shrinking after 2014).
             Basket = the 8 majors vs USD (1/8 each, $1 total). Every weekday, two windows:
               Tokyo fix 9:55 Tokyo : LONG USD 08:00-10:00 Tokyo, then SHORT USD 10:00-12:00 Tokyo
               London fix 16:00 London: LONG USD 14:00-16:00 London, then SHORT USD 16:00-18:00 London
             ~10 trades a week (enter, flip, exit per window).
  2 SPILL    US stocks' day -> risk currencies overnight. Each US trading day, if SPY's close-to-close
             move is beyond +/-0.5%: SPY down -> long JPY $1, short AUD $0.5 + NZD $0.5 (risk-off);
             SPY up -> the reverse. Hold 17:00 -> 02:00 New York (the thin Asian evening). ~2-3 a week.
Controls (must FAIL): FIXREV_SHIFT (same windows moved 5 hours later), SPILL_RANDOM (random sign on
  the same days).
Costs: fx_screen.COST_BP per side (CME futures), also at 2x; the BEFORE-COSTS Sharpe and bp per trade
  day are printed too -- that number decides whether a limit-order version is worth building.
Pass bar (same as always): Sharpe >= 0.5 2012-07 -> 2021-10, >= 0.7 2021-11 -> end, 50/50 equal-risk
  monthly blend with CHIMERA > 1.42, whole-period Sharpe > 0 at 2x costs.

Run (repo root, uses data already on the Mac):
    python3.10 scripts/fx_active.py --smoke
    python3.10 scripts/fx_active.py | tee logs/fx_active.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fx_screen as fx  # noqa: E402

MAJ = ["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "NOK"]
ERAS = [("2004-2013", "2004-01-01", "2014-01-01"), ("2014-2019", "2014-01-01", "2020-01-01"),
        ("2020-2026", "2020-01-01", "2100-01-01")]


def from_earn(earn):
    """earn[h] = exposure wanted DURING bar h -> position held after the close of bar h-1."""
    return earn.shift(-1).fillna(0.0)


# =================================================================== 1 FIXREV
def fixrev_positions(P, shift_h=0):
    idx = P.index + pd.Timedelta(hours=shift_h)
    tok = idx.tz_convert("Asia/Tokyo")
    lon = idx.tz_convert("Europe/London")
    th, tw = np.asarray(tok.hour), np.asarray(tok.dayofweek) < 5
    lh, lw = np.asarray(lon.hour), np.asarray(lon.dayofweek) < 5
    usd = np.zeros(len(P))                                    # +1 = long USD
    usd[tw & ((th == 8) | (th == 9))] = 1.0
    usd[tw & ((th == 10) | (th == 11))] = -1.0
    usd[lw & ((lh == 14) | (lh == 15))] = 1.0
    usd[lw & ((lh == 16) | (lh == 17))] = -1.0
    earn = pd.DataFrame(np.outer(-usd, np.ones(len(MAJ)) / len(MAJ)), index=P.index, columns=MAJ)
    return from_earn(earn), int(((np.diff(np.r_[0, usd]) != 0) & (usd != 0)).sum())


# =================================================================== 2 SPILL
def spill_positions(P, eq, random_sign=False, seed=41):
    r = eq["SPY"].pct_change()
    sig = pd.Series(np.where(r < -0.005, -1.0, np.where(r > 0.005, 1.0, 0.0)), index=r.index)
    if random_sign:
        rng = np.random.default_rng(seed)
        nz = sig != 0
        sig[nz] = rng.choice([-1.0, 1.0], size=int(nz.sum()))
    ny = P.index.tz_convert("America/New_York")
    sess = pd.DatetimeIndex((ny - pd.Timedelta(hours=17)).date)   # 17:00 NY day d -> session d
    hr = np.asarray(ny.hour)
    hold = (hr >= 17) | (hr <= 1)
    s = sig.reindex(sess).fillna(0.0).values * hold
    # SPY down (s=-1): long JPY, short AUD/NZD
    earn = pd.DataFrame({"JPY": -s * 1.0, "AUD": s * 0.5, "NZD": s * 0.5}, index=P.index)
    n = int((sig != 0).sum())
    return from_earn(earn), n


# =================================================================== judging
def eras_bp(x):
    out = []
    for name, a, b in ERAS:
        w = x[(x.index >= a) & (x.index < b)]
        act = w[w != 0]
        out.append(f"{name} {act.mean() * 1e4:+.2f}" if len(act) > 50 else f"{name} n/a")
    return "  ".join(out)


def judge(name, pos, n, R, years):
    x0 = fx.hourly_book(pos, R, 0.0)
    x, x2 = fx.hourly_book(pos, R, 1.0), fx.hourly_book(pos, R, 2.0)
    a0, _, s0, _ = fx.stats(fx.win(x, fx.EARLY))
    a1, v1, s1, d1 = fx.stats(fx.win(x, fx.RECENT))
    _, _, sw, _ = fx.stats(x)
    _, _, sw2, _ = fx.stats(x2)
    _, _, sg, _ = fx.stats(x0)
    _, _, sg_r, _ = fx.stats(fx.win(x0, fx.RECENT))
    csh, bsh, corr, bad = fx.blend(x)
    ok = (s0 >= 0.5) and (s1 >= 0.7) and (bsh > csh) and (sw2 > 0)
    print(f"  {name:<19}| {a0*100:+6.2f}% {s0:5.2f} | {a1*100:+6.2f}% {v1*100:4.1f}% {s1:5.2f}"
          f" {d1*100:5.0f}% | {sw:6.2f} {sw2:7.2f} | {sg:6.2f} {sg_r:6.2f} | {corr:+6.2f} {bsh:5.2f} |"
          f" {n / years / 52:5.1f} | {'PASS' if ok else 'fail'}")
    print(f"  {'':<19}  before-costs bp per active day: {eras_bp(x0)}")
    return ok


def run(P, eq):
    R = fx.hourly_returns(P)
    C, _ = fx.daily(P)
    years = len(C) / 252
    print(f"\n  data {P.index[0].date()} -> {P.index[-1].date()}, {len(C):,} days")
    print(f"  {'idea':<19}| 2012-21 ann   shp | last5y ann   vol  shp  worst | whole 2x-cost |"
          f" gross whole/last5y | corrCHIM blend | tr/wk | verdict")
    res = {}
    res["1 FIXREV"] = judge("1 FIXREV", *fixrev_positions(P), R, years)
    res["2 SPILL"] = judge("2 SPILL", *spill_positions(P, eq), R, years)
    res["control FIX_SHIFT5"] = judge("control FIX_SHIFT5", *fixrev_positions(P, shift_h=5), R, years)
    res["control SPILL_RANDOM"] = judge("control SPILL_RANDOM", *spill_positions(P, eq, True), R, years)
    bad = [k for k, v in res.items() if k.startswith("control") and v]
    print("  controls: " + ("all fail (good)" if not bad else
                            f"WARNING {', '.join(bad)} passed -> the test setup is suspect"))
    print(f"  CHIMERA monthly Sharpe 1.42. ann = average yearly return (log) on $1. gross = Sharpe"
          " before any costs.\n  pass = 2012-21 >= 0.5, last5y >= 0.7, blend > 1.42, 2x-cost > 0.")


# =================================================================== smoke
def synth(planted, seed=6):
    P, _ = fx.synth(False, seed=seed)
    P = P[MAJ].copy()
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2004-01-01", "2026-10-30")
    eq = pd.DataFrame({"SPY": np.exp(np.cumsum(rng.normal(0.0004, 0.011, len(days))))}, index=days)
    if planted:
        R = np.log(P).diff().fillna(0)
        fp, _ = fixrev_positions(P)
        R = R + fp.shift(1).fillna(0) * len(MAJ) * 0.0006           # 6 bp/hour in the fix windows
        P = np.exp(R.cumsum())
        sp, _ = spill_positions(P, eq)
        R = np.log(P).diff().fillna(0)
        R[["JPY", "AUD", "NZD"]] += sp.shift(1).fillna(0) * 0.0005    # 5 bp/hour overnight
        P = np.exp(R.cumsum())
    return P, eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        for planted in (False, True):
            print(f"\n===== SMOKE {'PLANTED (rows 1-2 should PASS, controls fail)' if planted else 'NOISE (everything should fail)'}")
            run(*synth(planted))
        return
    from fx_calendar import load_equities
    P = fx.load_real(MAJ)
    eq = load_equities(fx.START, fx.END)
    run(P, eq)


if __name__ == "__main__":
    main()
