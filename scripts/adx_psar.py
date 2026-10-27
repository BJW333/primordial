"""
adx_psar.py -- Wilder's own pairing: ADX (is there a strong trend?) +
Parabolic SAR (trailing stop that times entries and exits), on the
commodity ETFs from commodity_adx.py (real prices, no roll artifacts).

PARABOLIC SAR: classic settings, acceleration 0.02, step 0.02, max 0.20.
  SAR below price = uptrend, above = downtrend; a flip = signal.

VARIANTS (checked at each daily close, earn from the next day; equal risk
per ETF; 5 bps per unit traded)
  ADX_PSAR   JUDGED. Enter long on a SAR flip to "below price" if ADX > 25
             and +DI > -DI; enter short on a flip to "above price" if
             ADX > 25 and -DI > +DI. Exit on the next SAR flip (flat until
             a new qualifying flip).
  PSAR       context: always in, long/short with the SAR (stop-and-reverse)
  ADX_TREND  context: ADX > 25 with DI direction (commodity_adx.py)

PRE-STATED: ADX_PSAR PASSES if Sharpe >= 0.3 over 2008 -> 2021-05 AND
>= 0.5 over the last five years (2021-06 -> 2026-05) AND a 50/50 equal-
risk blend with stock momentum (momentum_v2 STEADY) beats momentum alone.

Run (repo root; ETF prices cached):
    python3.10 scripts/adx_psar.py --smoke
    python3.10 scripts/adx_psar.py | tee logs/adx_psar.txt
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
import commodity_adx as A     # noqa: E402


def psar(h, l, af0=0.02, step=0.02, amax=0.20):
    """Parabolic SAR direction for one series: +1 uptrend, -1 downtrend."""
    h, l = np.asarray(h, float), np.asarray(l, float)
    n = len(h)
    out = np.full(n, np.nan)
    ok = np.flatnonzero(np.isfinite(h) & np.isfinite(l))
    if len(ok) < 3:
        return out
    i0 = ok[0]
    up = True
    sar, ep, af = l[i0], h[i0], af0
    for i in range(i0 + 1, n):
        if not (np.isfinite(h[i]) and np.isfinite(l[i])):
            out[i] = out[i - 1]
            continue
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1] if np.isfinite(l[i - 1]) else sar)
            if l[i] < sar:                       # flip to downtrend
                up, sar, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep, af = h[i], min(af + step, amax)
        else:
            sar = max(sar, h[i - 1] if np.isfinite(h[i - 1]) else sar)
            if h[i] > sar:                       # flip to uptrend
                up, sar, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep, af = l[i], min(af + step, amax)
        out[i] = 1.0 if up else -1.0
    return out


def signals(H, L, Cl):
    a, pdi, ndi = A.adx(H, L, Cl)
    d = np.sign(pdi - ndi)
    P = pd.DataFrame({c: psar(H[c].values, L[c].values) for c in Cl.columns},
                     Cl.index)
    flip = P.ne(P.shift()) & P.shift().notna()
    pos = pd.DataFrame(0.0, Cl.index, Cl.columns)
    for c in Cl.columns:
        p = 0.0
        out = np.zeros(len(Cl))
        pv, fv, av, dv = (P[c].values, flip[c].values, a[c].values,
                          d[c].values)
        for i in range(len(Cl)):
            if fv[i]:
                p = 0.0                           # any flip exits
                if np.isfinite(av[i]) and av[i] > 25 and dv[i] == pv[i]:
                    p = pv[i]                     # qualifying flip enters
            out[i] = p
        pos[c] = out
    pos = pos.where(a.notna() & P.notna())
    trend = d.where(a > 25, 0.0).where(a.notna())
    return {"ADX_PSAR": pos, "PSAR": P, "ADX_TREND": trend}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        H, L, Cl, V = A.smoke()
        print("  smoke: half the ETFs carry planted multi-month trends")
        # unit check: a clean uptrend then downtrend flips SAR once
        x = np.r_[np.linspace(10, 20, 60), np.linspace(20, 10, 60)]
        s = psar(x * 1.01, x * 0.99)
        flips = int((np.diff(s[np.isfinite(s)]) != 0).sum())
        print(f"  SAR unit check: up-then-down path -> {flips} flip(s), "
              f"ends {'down' if s[-1] < 0 else 'up'} (expect 1, down)")
        mom_m = C.monthly(pd.Series(np.random.default_rng(9).normal(
            0.0005, 0.01, len(Cl)), Cl.index))
    else:
        H, L, Cl, V = A.load()
        mom_m, _ = C.stock_momentum()
    R = Cl.pct_change(fill_method=None).clip(-0.25, 0.25)
    vol = R.rolling(60, min_periods=40).std() * np.sqrt(252)
    S = signals(H, L, Cl)
    print(f"\n  {'variant':<10} | {'2008-21 ann':>11} {'sh':>5} | "
          f"{'last5y ann':>10} {'sh':>5} {'worst':>6} {'2022':>5} | "
          f"{'mom':>5} {'blend':>5} | {'in market':>9} {'trades/yr':>9}")
    res = {}
    for k in ("ADX_PSAR", "PSAR", "ADX_TREND"):
        W = S[k]
        d = A.book(W, R, vol)
        a0, s0, a1, v1, s1, dd, y22 = A.split(d)
        msh, bsh, _ = C.blend_sharpe(mom_m, C.monthly(d), C.RECENT, C.DEC1)
        live = ((W.abs() > 0).sum(axis=1) / W.notna().sum(axis=1)).mean()
        yrs = len(W) / 252
        trades = (W.diff().abs() > 0).sum().sum() / max(W.shape[1], 1) / yrs
        print(f"  {k:<10} | {a0*100:>+10.1f}% {s0:>5.2f} | {a1*100:>+9.1f}% "
              f"{s1:>5.2f} {dd*100:>5.0f}% {y22*100:>+4.0f}% | {msh:>5.2f} "
              f"{bsh:>5.2f} | {live*100:>8.0f}% {trades:>9.1f}")
        res[k] = (s0, s1, bsh > msh)
    s0, s1, helps = res["ADX_PSAR"]
    ok = s0 >= 0.3 and s1 >= 0.5 and helps
    print("\n  ===== VERDICT (pre-stated, ADX_PSAR) =====")
    print(f"  2008-21 Sharpe {s0:.2f} (need >= 0.3), last 5y {s1:.2f} (need "
          f">= 0.5), improves momentum blend: {helps} -> "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
