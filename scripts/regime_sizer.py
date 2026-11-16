"""
regime_sizer.py -- REGIME SIZER alone (TEST A) and with the DD2 dimmer on
top (TEST B). Each has its own verdict; pass test = judge.py.
DD2 alone is dd2_lab.py.

REGIME (decided at each month-end from the momentum strategy's own behavior,
used from the next day; Bongaerts-Kang-van Dijk 2020 conditional volatility
targeting + Ehsani-Linnainmaa 2022 factor momentum)
  calm     last-month volatility in its bottom 20% (vs own last 5 years)
           AND last-12-month return > 0                      -> 100% momentum
  stressed volatility in its top 20% OR last-12-month return <= 0 -> 35%
  normal   everything else                                    -> 70%
  (stressed beats calm; rest in T-bills; no leverage). Signal source: each
  market's own winners series; CHIMERA's book only starts 2017, so CHIMERA
  uses the Ken French US winners series (same kind of stocks, back to 1926).

DIMMER ON TOP (TEST B; final share = regime size x dimmer, traded when it
moves 5+ points; dimmer = max(0, 1 - book drop / (1.5 x book volatility)),
rolling high window picked from {126, 252} on 1927-1989)
30% yearly lock -> market until Jan 1 unchanged. 0.2% cost on what is traded.

Run (repo root; US only, nothing to download; needs dd2_lab.py, judge.py):
    python3.10 scripts/regime_sizer.py --smoke
    python3.10 scripts/regime_sizer.py | tee logs/regime_sizer.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_warning_search as CW  # noqa: E402
import dd2_lab as D2               # noqa: E402

COST, LOCK, STEP = 0.002, 0.30, 0.05
CALM, NORMAL, STRESS = 1.00, 0.70, 0.35
TEST_A = [("none",)]
TEST_B = [("BOOK", 126, 1.5), ("BOOK", 252, 1.5)]


def label(v):
    return "REGIME only" if v[0] == "none" else f"REGIME + DD2 {v[1]}d, {v[2]}x vol"


def regime_sizes(ret):
    vol = ret.rolling(21).std()
    rk = vol.rolling(1260, min_periods=756).rank(pct=True)
    yr = np.expm1(np.log1p(ret).rolling(252).sum())
    size = pd.Series(NORMAL, ret.index)
    size[(rk <= 0.20) & (yr > 0)] = CALM
    size[(rk >= 0.80) | (yr <= 0)] = STRESS
    me = np.r_[ret.index.month[1:] != ret.index.month[:-1], True]
    return size.where(me).ffill().fillna(NORMAL)


def sim(D, v=("none",), fixed=None):
    b, m, c, years = D["b"], D["m"], D["c"], D["years"]
    n = len(b)
    A, ys, yr, locked = 1.0, 1.0, years[0], False
    share = fixed if fixed is not None else NORMAL
    out = np.empty(n)
    reg = D["reg"]
    for i in range(n):
        if years[i] != yr:
            yr, ys = years[i], A
            if locked:
                A *= 1 - COST
                locked = False
        r = m[i] if locked else share * b[i] + (1 - share) * c[i]
        A *= 1 + r
        out[i] = A
        if locked:
            continue
        if A / ys - 1 >= LOCK:
            locked = True
            A *= 1 - COST
            continue
        if fixed is not None:
            continue
        mult = 1.0
        if v[0] == "BOOK":
            mult = max(0.0, 1 - D["dd"][v[1]][i] / (v[2] * D["vol"][i]))
        tgt = reg[i] * mult
        if abs(tgt - share) >= STEP:
            A *= 1 - COST * abs(tgt - share)
            share = tgt
    eq = pd.Series(out, D["idx"])
    return eq.pct_change().fillna(eq.iloc[0] - 1)


def mix(reg):
    reg = pd.Series(reg)
    return " / ".join(f"{(reg == s).mean()*100:.0f}% at {s:.0%}" for s in (CALM, NORMAL, STRESS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = CW.smoke_data() if a.smoke else CW.load_french()
    D1, tests = D2.load_tests(d, a.smoke)
    reg_us = regime_sizes(d["W"])
    D1["reg"] = reg_us.reindex(D1["idx"]).values
    for tn, D in tests.items():
        if tn in ("US 1990-2016", "CHIMERA-70"):
            D["reg"] = reg_us.reindex(D["idx"]).ffill().fillna(NORMAL).values
        else:
            D["reg"] = regime_sizes(pd.Series(D["b"], D["idx"])).values
    print("  regime mix (share of days at 100% / 70% / 35%):")
    print(f"    US 1927-1989: {mix(D1['reg'])}")
    for tn, D in tests.items():
        print(f"    {tn}: {mix(D['reg'])}")

    rule = lambda D, v: sim(D, v)               # noqa: E731
    fixed = lambda D, s: sim(D, fixed=s)        # noqa: E731
    okA, bA = D2.pick_and_judge("TEST A: REGIME ONLY", TEST_A, D1, tests, rule, fixed, label)
    okB, bB = D2.pick_and_judge("TEST B: REGIME + DD2 (combined)", TEST_B, D1, tests,
                                rule, fixed, label)
    print("\n  ===== FINAL VERDICTS =====")
    print(f"    TEST A  {label(bA) if bA else '-':<30} -> {'PASS' if okA else 'FAIL'}")
    print(f"    TEST B  {label(bB) if bB else '-':<30} -> {'PASS' if okB else 'FAIL'}")
    print("    (DD2 alone: see dd2_lab.py)")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
