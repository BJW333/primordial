"""
dd2_lab.py -- DD2: the drawdown dimmer, rebuilt. Pass test: judge.py.

Why version 1 lost, and the fix for each:
  "stuck small"   it watched the ACCOUNT's drop; once cut, the account
                  recovered slowly, so momentum stayed small after it bounced
                  (2022: -3% -> -9%).  FIX: watch the momentum BOOK itself
                  (a shadow always running at full size).
  "Jan 1 amnesia" reset to full size every Jan 1, even mid-slide.
                  FIX: drop from the book's ROLLING high (63/126/252 days).
  "all the way out" zero momentum at the bottom, missing V-rebounds.
                  FIX: optional floor at half size (35% momentum).
  "fixed numbers" a 30% zero point means different things for a calm and a
                  wild book.  FIX (dynamic): the zero point scales with the
                  book's own volatility -- zero = K x its last-year volatility
                  (K = 1.0 / 1.5 / 2.0; for US winners ~22% / 33% / 44%).

DD2: momentum share = 70% x max(FLOOR, 1 - book drop / (K x book volatility)),
traded only when the target moves 5+ points. 30% yearly lock (account-based)
-> market until Jan 1 unchanged. 0.2% cost on whatever is traded.

FAMILY (fixed before running): window {63,126,252} x K {1.0,1.5,2.0} x
FLOOR {0, 50%} = 18 versions. PICK on US 1927-1989: biggest Sharpe gain
(above T-bills) over its same-return fixed size, worst drop no worse.
Then judge.py on the holdouts.

Run (repo root; US data already on disk, nothing to download):
    python3.10 scripts/dd2_lab.py --smoke
    python3.10 scripts/dd2_lab.py | tee logs/dd2_lab.txt
"""
import argparse
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_warning_search as CW  # noqa: E402
import judge as J                  # noqa: E402

COST, MOM, LOCK, STEP = 0.002, 0.70, 0.30, 0.05
P1, P2, P3 = ("1927-01-01", "1989-12-31"), ("1990-01-01", "2016-12-31"), \
    ("2017-03-01", "2026-05-29")
WINDOWS, KS, FLOORS = (63, 126, 252), (1.0, 1.5, 2.0), (0.0, 0.5)
FAMILY = list(itertools.product(WINDOWS, KS, FLOORS))
REGIONS = []   # US only
SIZES = np.round(np.arange(0.05, 1.0001, 0.01), 2)


# ------------------------------------------------------------ data
def prep(b, m, c):
    X = (1 + b).cumprod()
    vol = (b.rolling(252, min_periods=63).std() * np.sqrt(252)).bfill()
    return {"b": b.values, "m": m.values, "c": c.values, "idx": b.index,
            "years": b.index.year.values, "vol": vol.values,
            "dd": {w: (1 - X / X.rolling(w, min_periods=1).max()).values for w in WINDOWS}}


def region_data(region, smoke, d):
    """Big-company winners + regional market + RF, daily. None if unavailable."""
    if smoke:
        rng = np.random.default_rng(len(region))
        s = d[d.index >= "1990-07-01"]
        M = s["M"] + rng.normal(0, 0.004, len(s))
        return prep(M * 1.1 + rng.normal(0.0002, 0.006, len(s)), M, s["RF"])
    names = [f"{region}_6_Portfolios_ME_Prior_12_2_Daily_CSV.zip",
             f"{region}_25_Portfolios_ME_Prior_12_2_Daily_CSV.zip"]
    port = None
    for nm in names:
        try:
            CW.FILES[nm] = nm
            port = CW.first_block(CW.fetch(nm))
            break
        except Exception:
            continue
    try:
        fac_nm = f"{region}_3_Factors_Daily_CSV.zip"
        CW.FILES[fac_nm] = fac_nm
        fac = CW.first_block(CW.fetch(fac_nm))
    except Exception:
        fac = None
    if port is None or fac is None:
        print(f"    {region}: download failed -- put {names[0]} (or {names[1]}) and "
              f"{region}_3_Factors_Daily_CSV.zip in data/french/")
        return None
    col = [c for c in port.columns if re.search(r"big", c, re.I) and re.search(r"hi", c, re.I)]
    if not col:
        print(f"    {region}: no BIG HiPRIOR column in {list(port.columns)[:8]} ...")
        return None
    x = pd.DataFrame({"W": port[col[0]], "M": fac["Mkt-RF"] + fac["RF"],
                      "RF": fac["RF"]}).dropna()
    print(f"    {region}: '{col[0]}' {x.index[0].date()} -> {x.index[-1].date()}")
    return prep(x["W"], x["M"], x["RF"])


def load_tests(d, smoke):
    def us(p):
        s = (d.index >= p[0]) & (d.index <= p[1])
        return prep(d["W"][s], d["M"][s], d["RF"][s])
    if smoke:
        s3 = (d.index >= P3[0]) & (d.index <= P3[1])
        bk = pd.DataFrame({"STEADY": d["W"][s3] * 1.5, "SPY": d["M"][s3]})
    else:
        bk = pd.read_csv(CW.BOOK, index_col=0, parse_dates=True)
        bk = bk[(bk.index >= P3[0]) & (bk.index <= P3[1])]
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    tests = {"US 1990-2016": us(P2),
             "CHIMERA-70": prep(bk["STEADY"], bk["SPY"].fillna(0), cash)}
    for rg in REGIONS:
        R = region_data(rg, smoke, d)
        if R is not None:
            tests[rg] = R
    return us(P1), tests


# ------------------------------------------------------------ simulation
def sim(D, v=None, fixed=None):
    """v = (window, K, floor) for DD2; fixed = constant momentum share."""
    b, m, c, years = D["b"], D["m"], D["c"], D["years"]
    n = len(b)
    A, ys, yr, locked = 1.0, 1.0, years[0], False
    share = fixed if fixed is not None else MOM
    out = np.empty(n)
    if v is not None:
        ddw, vol = D["dd"][v[0]], D["vol"]
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
        if v is not None:
            tgt = MOM * max(v[2], 1 - ddw[i] / (v[1] * vol[i]))
            if abs(tgt - share) >= STEP:
                A *= 1 - COST * abs(tgt - share)
                share = tgt
    eq = pd.Series(out, D["idx"])
    return eq.pct_change().fillna(eq.iloc[0] - 1)


def fixed_curve(D, simfn):
    runs = [simfn(D, s) for s in SIZES]
    return runs, np.array([J.stats(r, D["c"]) for r in runs])


def name(v):
    return f"book {v[0]}d, zero {v[1]:.1f}x vol, floor {v[2]:.0%}"


def pick_and_judge(title, cands, D1, tests, rule_fn, fixed_fn, label_fn):
    """Shared by dd2_lab and regime_sizer."""
    print(f"\n  ################ {title} ################")
    runs1, c1 = fixed_curve(D1, fixed_fn)
    b1 = J.stats(fixed_fn(D1, 0.70), D1["c"])
    print(f"  1 PICK on US 1927-1989 (BASE fixed 70%: {b1[0]*100:+.1f}%/yr, Sharpe "
          f"{b1[1]:.2f}, worst {b1[2]*100:.1f}%). Sharpe = above T-bills.")
    best, best_gain = None, -np.inf
    for v in cands:
        an, sh, wd = J.stats(rule_fn(D1, v), D1["c"])
        sz, j = J.matched(SIZES, c1, an)
        gain = sh - c1[j, 1]
        ok = wd >= c1[j, 2] - 1e-9
        print(f"    {label_fn(v):<36} {an*100:>+6.1f}%/yr Sharpe {sh:5.2f} worst {wd*100:6.1f}% | "
              f"fixed {sz:.0%}: {c1[j, 0]*100:+.1f}% {c1[j, 1]:.2f} {c1[j, 2]*100:.1f}% -> gain "
              f"{gain:+.3f}{'' if ok else ' (worse drop, excluded)'}")
        if ok and gain > best_gain:
            best, best_gain = v, gain
    if best is None:
        print("    nothing beat a same-return fixed size on 1927-1989 -> FAIL")
        return False, None
    print(f"    PICKED: {label_fn(best)}")
    print("  2 HOLDOUTS + 3 CHIMERA (picked version, unchanged):")
    rows = {}
    for tn, D in tests.items():
        rule = rule_fn(D, best)
        runs, cv = fixed_curve(D, fixed_fn)
        sz, j = J.matched(SIZES, cv, J.stats(rule, D["c"])[0])
        rows[tn] = {"rule": rule, "fixed": runs[j], "base": fixed_fn(D, 0.70),
                    "rf": D["c"], "size": sz}
    return J.judge(title, rows, "RULE"), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = CW.smoke_data() if a.smoke else CW.load_french()
    D1, tests = load_tests(d, a.smoke)
    ok, best = pick_and_judge("DD2 ALONE", FAMILY, D1, tests,
                              lambda D, v: sim(D, v=v), lambda D, s: sim(D, fixed=s), name)
    print(f"\n  FINAL: DD2 alone ({name(best) if best else '-'}) -> {'PASS' if ok else 'FAIL'}")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
