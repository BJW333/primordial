"""
commodity_adx.py -- ADX trend-strength and volume signals on commodity
ETFs (real tradeable prices and volumes: no futures roll jumps, which
faked the SEASON result in commodities_test.py).

ETFs: USO BNO UNG UGA GLD SLV CPER PPLT CORN WEAT SOYB CANE DBA DBB
Daily OHLCV via primordial.data (Yahoo, cached). Signal at a close,
position earns from the next day. Each ETF sized to equal risk
(10% / its volatility / number live). 5 bps per unit traded.

INDICATORS (Wilder, 14 days): ADX = trend strength, +DI/-DI = direction.
Volume surge = 20-day avg volume / 100-day avg volume.

VARIANTS
  ADX_TREND     long if ADX > 25 and +DI > -DI; short if ADX > 25 and
                -DI > +DI; flat otherwise
  ADX_VOL       ADX_TREND, only while volume surge >= 1.2
  ADX_RISING    ADX_TREND, only while ADX is higher than 5 days ago
  VOL_BREAKOUT  close at a 50-day high on >= 2x 50-day avg volume -> long
                20 days; 50-day low on >= 2x volume -> short 20 days
  context:      TREND12 (sign of 12-month return), HOLD (always long)
  Each also run LONG-ONLY (short signals -> flat), as on Alpaca.

PRE-STATED (4 variants, long/short versions judged)
  PASS if ALL: Sharpe >= 0.5 last five years (2021-06 -> 2026-05);
  Sharpe >= 0.3 over 2008 -> 2021-05; 50/50 equal-risk blend with stock
  momentum (momentum_v2 STEADY) has a higher last-5y Sharpe than
  momentum alone.

Run (repo root):
    python3.10 scripts/commodity_adx.py --smoke
    python3.10 scripts/commodity_adx.py | tee logs/commodity_adx.txt
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
import commodities_test as C  # noqa: E402

ETFS = ["USO", "BNO", "UNG", "UGA", "GLD", "SLV", "CPER", "PPLT", "CORN",
        "WEAT", "SOYB", "CANE", "DBA", "DBB"]
CTX0 = pd.Timestamp("2008-01-01")
TGT, COST = 0.10, 0.0005
VARIANTS = ["ADX_TREND", "ADX_VOL", "ADX_RISING", "VOL_BREAKOUT"]
CONTEXT = ["TREND12", "HOLD"]


def load():
    sys.path.insert(0, C.ROOT)
    from primordial.data import fetch
    d = fetch("yfinance", ETFS, "1d", C.START, C.END, min_bars=500)
    get = lambda k: pd.DataFrame({s: d[s][k] for s in ETFS if s in d})  # noqa
    H, L, Cl, V = get("high"), get("low"), get("close"), get("volume")
    for x in (H, L, Cl, V):
        x.index = pd.to_datetime(x.index).normalize()
    print(f"  {Cl.shape[1]} ETFs: " + ", ".join(
        f"{s} {Cl[s].first_valid_index():%Y}" for s in Cl.columns))
    return H.sort_index(), L.sort_index(), Cl.sort_index(), V.sort_index()


def wilder(x, n=14):
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(H, L, Cl, n=14):
    up = H.diff()
    dn = -L.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.DataFrame({c: np.maximum.reduce([
        (H[c] - L[c]).values, (H[c] - Cl[c].shift()).abs().values,
        (L[c] - Cl[c].shift()).abs().values]) for c in H.columns}, H.index)
    atr = wilder(tr, n)
    pdi = 100 * wilder(pdm, n) / atr
    ndi = 100 * wilder(ndm, n) / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return wilder(dx, n), pdi, ndi


def signals(H, L, Cl, V):
    a, pdi, ndi = adx(H, L, Cl)
    surge = V.rolling(20, min_periods=15).mean() / \
        V.rolling(100, min_periods=60).mean()
    direction = np.sign(pdi - ndi)
    trend = direction.where(a > 25, 0.0).where(a.notna())
    S = {"ADX_TREND": trend,
         "ADX_VOL": trend.where(surge >= 1.2, 0.0).where(trend.notna()),
         "ADX_RISING": trend.where(a > a.shift(5), 0.0).where(trend.notna())}
    hi50 = Cl.rolling(50, min_periods=50).max()
    lo50 = Cl.rolling(50, min_periods=50).min()
    v50 = V.rolling(50, min_periods=40).mean().shift(1)
    ev = pd.DataFrame(0.0, Cl.index, Cl.columns)
    ev[(Cl >= hi50) & (V >= 2 * v50)] = 1.0
    ev[(Cl <= lo50) & (V >= 2 * v50)] = -1.0
    hold = ev.replace(0.0, np.nan).ffill(limit=19).fillna(0.0)
    S["VOL_BREAKOUT"] = hold.where(hi50.notna())
    S["TREND12"] = np.sign(Cl / Cl.shift(252) - 1)
    S["HOLD"] = pd.DataFrame(1.0, Cl.index, Cl.columns).where(Cl.notna())
    return S


def book(S, R, vol, long_only=False):
    s = S.clip(lower=0) if long_only else S
    n = s.notna().sum(axis=1).replace(0, np.nan)
    W = (s * (TGT / vol)).div(n, axis=0)
    return C.run_book(R, W, COST)


def split(d):
    a0, _, s0, _ = C.stats(d[(d.index >= CTX0) & (d.index < C.RECENT)])
    a1, v1, s1, dd = C.stats(d[(d.index >= C.RECENT) & (d.index < C.DEC1)])
    y22 = (1 + d[d.index.year == 2022]).prod() - 1
    return a0, s0, a1, v1, s1, dd, y22


def smoke():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2006-01-02", "2026-10-23")
    n = len(idx)
    H, L, Cl, V = {}, {}, {}, {}
    for j, s in enumerate(ETFS):
        regime = np.sign(np.sin(np.arange(n) / 300 + j)) if j % 2 == 0 \
            else np.zeros(n)
        r = regime * 0.0015 + rng.normal(0, 0.015, n)
        c = 50 * np.exp(np.cumsum(r))
        rng_hl = np.abs(rng.normal(0, 0.008, n))
        Cl[s], H[s], L[s] = c, c * (1 + rng_hl), c * (1 - rng_hl)
        V[s] = rng.lognormal(13, 0.3, n) * (1 + 0.5 * (regime != 0))
    f = lambda d: pd.DataFrame(d, idx)  # noqa: E731
    return f(H), f(L), f(Cl), f(V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        H, L, Cl, V = smoke()
        print("  smoke: half the ETFs carry planted multi-month trends")
    else:
        H, L, Cl, V = load()
    R = Cl.pct_change(fill_method=None).clip(-0.25, 0.25)
    vol = R.rolling(60, min_periods=40).std() * np.sqrt(252)
    S = signals(H, L, Cl, V)
    if a.smoke:
        mom_m = C.monthly(pd.Series(np.random.default_rng(9).normal(
            0.0005, 0.01, len(R)), R.index))
        lab = "random stand-in (smoke)"
    else:
        mom_m, lab = C.stock_momentum()
    print(f"\n  benchmark for the blend test: {lab}")
    print(f"\n  {'variant':<13} {'side':<10} | {'2008-21 ann':>11} {'sh':>5} |"
          f" {'last5y ann':>10} {'vol':>4} {'sh':>5} {'worst':>6} {'2022':>5}"
          f" | {'mom':>5} {'blend':>5}")
    verdict = {}
    for k in VARIANTS + CONTEXT:
        for lo in (False, True):
            d = book(S[k], R, vol, lo)
            a0, s0, a1, v1, s1, dd, y22 = split(d)
            msh, bsh, _ = C.blend_sharpe(mom_m, C.monthly(d), C.RECENT,
                                         C.DEC1)
            side = "long-only" if lo else "long/short"
            print(f"  {k:<13} {side:<10} | {a0*100:>+10.1f}% {s0:>5.2f} | "
                  f"{a1*100:>+9.1f}% {v1*100:>3.0f}% {s1:>5.2f} {dd*100:>5.0f}%"
                  f" {y22*100:>+4.0f}% | {msh:>5.2f} {bsh:>5.2f}")
            if k in VARIANTS and not lo:
                verdict[k] = s1 >= 0.5 and s0 >= 0.3 and bsh > msh
    print("\n  ===== VERDICT (pre-stated: last-5y Sharpe >= 0.5, 2008-21 "
          ">= 0.3, improves the momentum blend; long/short versions) =====")
    if a.smoke:
        print("  smoke: planted trends (ADX variants should pass)")
    for k in VARIANTS:
        print(f"  {k:<13} {'PASS' if verdict[k] else 'FAIL'}")


if __name__ == "__main__":
    main()
