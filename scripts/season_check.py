"""
season_check.py -- is commodity SEASON (commodities_test.py, commit
3e39538: last-5y Sharpe 1.36, 2007-21 0.67) real, or an artifact of
Yahoo's continuous futures, whose contract-roll jumps are themselves
calendar-seasonal (a holder never earns those jumps)?

CHECKS (rules fixed before running)
  1. ROLL-CLEANED: in the markets with a futures ETF, a roll jump shows up
     as a day where futures and ETF returns disagree sharply (|diff| > 4x
     its 60-day typical size and > 1.5%). On those days use the ETF's
     return instead. Re-run SEASON (signals AND P&L) on the cleaned series,
     same markets, and compare with raw futures on the same markets.
  2. ETF-ONLY: SEASON with signals and P&L purely from ETF prices,
     long/short (ETF histories start 2006-2011, so fewer years).
  3. CONCENTRATION: SEASON run one market at a time -> share of total P&L.

SEASON SURVIVES if the ROLL-CLEANED version keeps Sharpe >= 0.5 over the
last five years AND >= 0.3 over 2007 -> 2021-05, AND no single market
supplies more than 50% of the full-universe P&L.

Run (repo root; prices are cached by commodities_test.py):
    python3.10 scripts/season_check.py | tee logs/season_check.txt
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import commodities_test as C  # noqa: E402


def sharpe_split(d):
    a0, _, s0, _ = C.stats(d[(d.index >= C.CTX0) & (d.index < C.RECENT)])
    a1, _, s1, dd = C.stats(d[(d.index >= C.RECENT) & (d.index < C.DEC1)])
    return a0, s0, a1, s1, dd


def season_book(P, R):
    vol = R.rolling(60, min_periods=40).std() * np.sqrt(252)
    W = C.sig_season(P, vol)
    return C.run_book(R[W.columns], W, C.COST)


def line(lab, d):
    a0, s0, a1, s1, dd = sharpe_split(d)
    print(f"  {lab:<34} 2007-21 {a0*100:>+5.1f}%/yr sh {s0:>5.2f} | last 5y "
          f"{a1*100:>+5.1f}%/yr sh {s1:>5.2f} worst {dd*100:>4.0f}%")
    return s0, s1


def main(F=None, E=None):
    if F is None:
        F, E = C.load_prices()
    R = F.pct_change(fill_method=None).clip(-0.25, 0.25)
    RE = E.pct_change(fill_method=None).clip(-0.25, 0.25)
    cols = [m for m in E.columns if m in F.columns]

    print("\n  ===== 1. ROLL-CLEANED futures (ETF markets only) =====")
    diff = (R[cols] - RE[cols].reindex(R.index))
    typ = diff.abs().rolling(60, min_periods=40).median()
    roll = (diff.abs() > 4 * typ) & (diff.abs() > 0.015)
    Rc = R[cols].where(~roll, RE[cols].reindex(R.index))
    n_roll = roll.sum()
    print("  roll-jump days replaced: " + ", ".join(
        f"{m} {int(n)}" for m, n in n_roll.items()))
    Pc = (1 + Rc.fillna(0)).cumprod().where(F[cols].notna())
    line("raw futures, same markets", season_book(F[cols], R[cols]))
    s0c, s1c = line("ROLL-CLEANED futures", season_book(Pc, Rc))

    print("\n  ===== 2. ETF-ONLY (signals + P&L from ETF prices) =====")
    line("ETF long/short", season_book(E[cols], RE[cols]))

    print("\n  ===== 3. CONCENTRATION (each market alone, full universe) "
          "=====")
    full = season_book(F, R)
    tot = full[full.index >= C.CTX0].sum()
    shares = {}
    for m in F.columns:
        d = season_book(F[[m]], R[[m]]) / F.shape[1]
        shares[m] = d[d.index >= C.CTX0].sum() / tot if tot else np.nan
    sh = pd.Series(shares).sort_values(ascending=False)
    print("  share of total P&L since 2007: " + ", ".join(
        f"{m} {v*100:+.0f}%" for m, v in sh.items()))
    top = sh.iloc[0]

    print("\n  ===== 4. WHICH MONTHS (full universe, avg monthly P&L) =====")
    mm = C.monthly(full[full.index >= C.CTX0])
    print("  " + "  ".join(f"{pd.Timestamp(2000, k, 1):%b} "
                           f"{mm[mm.index.month == k].mean()*1e4:+.0f}bp"
                           for k in range(1, 13)))

    ok = s1c >= 0.5 and s0c >= 0.3 and top <= 0.5
    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"  roll-cleaned Sharpe: last 5y {s1c:.2f} (need >= 0.5), 2007-21 "
          f"{s0c:.2f} (need >= 0.3); biggest single market "
          f"{sh.index[0]} {top*100:.0f}% of P&L (need <= 50%)")
    msg = ("SURVIVES -- forward paper trading next" if ok else
           "DOES NOT SURVIVE -- roll artifact or concentration")
    print(f"  SEASON {msg}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        F, E, _ = C.smoke()
        main(F, E)
    else:
        main()
