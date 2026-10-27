"""
portfolio_combo.py -- SIZING STUDY (no pass/fail, nothing new is being
tested): can STACKING what we already have reach ~30%/yr, and at what risk?

Pieces (all already built; monthly returns 2017-02 -> 2026-05):
  FADE      the liquid-MID WEAK_TREND short book with a 50% stop
            (fade_stop.py primary: 0.5% borrow, 0.2%/side) -- hedged, so it
            earns ON TOP of whatever else you hold
  SPY       buy and hold
  ADX_OBV   TOP20_EW from adx_obv_variants.py (data/adx_obv_variants.csv)
  MOM       all-US steady momentum + 20% stop, STEADY_TS from momentum_v2.py
            (data/momentum_v2_returns.csv)

Mixes = base (100%) + FADE overlay at 50% / 100% / 150% of the account.
Extra financing charged on the overlay: 2%/yr per 100% of overlay (the
short proceeds fund most of the hedge; this is a conservative haircut).
Gross exposure = 100% base + 2 x overlay (short leg + hedge leg).
  Reg-T margin (normal account) allows ~200% gross -> overlay <= 50%.
  Portfolio margin (usually $100k+) allows far more.
Also shown: each base levered 1.5x (4%/yr borrowing) for comparison.

Run (repo root):
    python3.10 scripts/portfolio_combo.py | tee logs/portfolio_combo.txt
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attention_rebound as AR        # noqa: E402
import attention_short_test as AS     # noqa: E402
import blowup_us as U                 # noqa: E402
import factor_portfolio as FP         # noqa: E402
import fade_predict as P              # noqa: E402
import fade_stop as FS                # noqa: E402

ROOT = os.path.dirname(HERE)
RECENT, END = AR.RECENT, pd.Timestamp("2026-06-01")
OVER = (0.5, 1.0, 1.5)
FIN, LEV_COST = 0.02, 0.04


def monthly(s):
    s = s.dropna()
    return (1 + s).groupby(s.index.to_period("M")).prod() - 1


def fade_daily():
    P.CLIP = False
    views = AS.norm_index(pd.read_parquet(AS.ATT))
    sp_members, _ = FP.membership()
    px, vol, unis = AR.universes(sp_members)
    mem, bench = unis["MID"]
    dv = (px * vol).rolling(63, min_periods=40).mean()
    ev = AR.good_spikes(views, px, mem)
    df = P.build("MID", ev, px, bench, {}, {}, mem, dv)
    tr = df[df["liq"] & df["W"]].sort_values("d")
    r, _, _ = FS.book(tr, px, bench, 0.005, FS.PRIMARY)
    return r[(r.index >= AR.START) & (r.index < END)]


def load_base():
    out = {}
    spy = AS.norm_index(pd.read_parquet(os.path.join(U.DIR, "spy.parquet")))
    out["SPY"] = monthly(spy["SPY"].pct_change())
    p = os.path.join(ROOT, "data", "adx_obv_variants.csv")
    if os.path.exists(p):
        a = pd.read_csv(p, index_col=0, parse_dates=True)
        if "TOP20_EW" in a:
            out["ADX_OBV"] = a["TOP20_EW"].groupby(
                a.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    p = os.path.join(ROOT, "data", "momentum_v2_returns.csv")
    if os.path.exists(p):
        m = pd.read_csv(p, index_col=0, parse_dates=True)
        out["MOM"] = m["STEADY"].groupby(m.index.to_period("M")).apply(
            lambda x: (1 + x).prod() - 1)
    return out


def stats(m):
    m = m.dropna()
    if len(m) < 12:
        return np.nan, np.nan, np.nan, np.nan
    ann = (1 + m).prod() ** (12 / len(m)) - 1
    vol = m.std() * np.sqrt(12)
    eq = (1 + m).cumprod()
    return ann, vol, (m.mean() * 12) / vol if vol else np.nan, \
        (eq / eq.cummax() - 1).min()


def main():
    argparse.ArgumentParser().parse_args()
    fade = monthly(fade_daily())
    base = load_base()
    mixes = {"FADE alone": fade}
    for k, b in base.items():
        b = b[(b.index >= fade.index[0]) & (b.index <= fade.index[-1])]
        mixes[k] = b
        mixes[f"{k} x1.5 lev"] = 1.5 * b - 0.5 * LEV_COST / 12
        for o in OVER:
            j = pd.concat([b, fade], axis=1, keys=["b", "f"]).dropna()
            mixes[f"{k} + FADE {o:.0%}"] = j["b"] + o * j["f"] - \
                o * FIN / 12
    print(f"\n  months: {fade.index[0]} -> {fade.index[-1]}   "
          f"(corr FADE vs SPY: {pd.concat([fade, base['SPY']], axis=1).corr().iloc[0, 1]:+.2f})")
    print(f"\n  {'mix':<22} {'gross':>6} | {'all ann':>8} {'vol':>5} "
          f"{'sh':>5} {'worst':>6} | {'2017-21':>8} | {'last 5y':>8} "
          f"{'worst':>6}")
    for k, m in mixes.items():
        g = "100%"
        for o in OVER:
            if f"FADE {o:.0%}" in k:
                g = f"{100 + 200 * o:.0f}%"
        if "lev" in k:
            g = "150%"
        if k == "FADE alone":
            g = "200%"
        a, v, s, dd = stats(m)
        a0 = stats(m[m.index < RECENT.to_period("M")])[0]
        a1, _, _, dd1 = stats(m[m.index >= RECENT.to_period("M")])
        flag = "  <- 30%+" if a1 >= 0.30 else ""
        print(f"  {k:<22} {g:>6} | {a*100:>+7.1f}% {v*100:>4.0f}% {s:>5.2f} "
              f"{dd*100:>5.0f}% | {a0*100:>+7.1f}% | {a1*100:>+7.1f}% "
              f"{dd1*100:>5.0f}%{flag}")
    best = [k for k in mixes if "+ FADE 100%" in k]
    yr = pd.DataFrame({k: mixes[k] for k in ["SPY", "FADE alone"] + best})
    yr = (1 + yr).groupby(yr.index.year).prod() - 1
    print("\n  by year:  " + "".join(f"{c[:16]:>17}" for c in yr.columns))
    for y, row in yr.iterrows():
        print(f"  {y:<8} " + "".join(f"{x*100:>+16.1f}%" for x in row))
    print("\n  SIZING STUDY ONLY -- the FADE stop is post-hoc and unproven; "
          "expect live results 30-50% below backtest.")


if __name__ == "__main__":
    main()
