"""
dd_check.py -- two checks on DDCTRL (drawdown control) before trusting it.
DDCTRL passed buyback_lab.py's pre-stated test (1927-89 pick, 1990-2016 and
CHIMERA-70 untouched). Settings below were fixed before running these checks.

DDCTRL: momentum share = 70% x (1 - drawdown / ZERO), drawdown = account
below its high since Jan 1; rebalanced when the target moves 5+ points;
30% yearly lock -> market; Jan 1 back to 70%. 0.2% cost on what is traded.

CHECK A -- FRAGILITY: ZERO = 20%, 30% (the tested one), 40%. Each must beat
  BASE (plain 70%, no control) on Sharpe AND worst drop in ALL of 1927-1989,
  1990-2016 and CHIMERA-70. (Same idea as the lock's 25/30/35 check.)
CHECK B -- NOT JUST LESS MONEY IN STOCKS: DDCTRL holds less momentum on
  average, and simply holding less always shrinks drops. Compare DDCTRL(30%)
  with a FIXED momentum size picked to give the SAME annual return in each
  period. DDCTRL must beat it on Sharpe AND worst drop in all three periods.
PASS = A and B. Pass -> run the QuantConnect DD_CTRL switch as the final word.

Also printed (context): average momentum share, control trades per year,
losing calendar years.

Run (repo root; needs buyback_lab.py, crash_warning_search.py in scripts/):
    python3.10 scripts/dd_check.py --smoke
    python3.10 scripts/dd_check.py | tee logs/dd_check.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import buyback_lab as BL           # noqa: E402
import crash_warning_search as CW  # noqa: E402

COST, LOCK, STEP = 0.002, 0.30, 0.05
ZEROS = (0.20, 0.30, 0.40)


def sim(D, mom, zero=None):
    """Fixed size `mom`, or drawdown control to 0 at `zero`. Returns
    (daily returns, average momentum share, control trades)."""
    b, m, c, years = D["b"], D["m"], D["c"], D["years"]
    n = len(b)
    A, ys, yr, peak, share, locked = 1.0, 1.0, years[0], 1.0, mom, False
    out, shares, trades = np.empty(n), [], 0
    for i in range(n):
        if years[i] != yr:
            yr, ys, peak = years[i], A, A
            if locked:
                A *= 1 - COST
            elif share != mom:
                A *= 1 - COST * abs(mom - share)
            locked, share = False, mom
        r = m[i] if locked else share * b[i] + (1 - share) * c[i]
        A *= 1 + r
        out[i] = A
        if locked:
            continue
        shares.append(share)
        if A / ys - 1 >= LOCK:
            locked = True
            A *= 1 - COST
            continue
        peak = max(peak, A)
        if zero:
            tgt = mom * max(0.0, 1 - (1 - A / peak) / zero)
            if abs(tgt - share) >= STEP:
                A *= 1 - COST * abs(tgt - share)
                share, trades = tgt, trades + 1
    eq = pd.Series(out, D["idx"])
    return eq.pct_change().fillna(eq.iloc[0] - 1), np.mean(shares), trades


def stats(r):
    r = np.asarray(r)
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    eq = np.cumprod(1 + r)
    return ann, sh, (eq / np.maximum.accumulate(eq) - 1).min()


def losing_years(r):
    y = (1 + r).groupby(r.index.year).prod() - 1
    return int((y < 0).sum()), len(y)


def line(label, r, extra=""):
    a, s, d = stats(r)
    lo, ny = losing_years(r)
    print(f"    {label:<22} {a*100:>+6.1f}%/yr  Sharpe {s:>5.2f}  worst {d*100:>6.1f}%"
          f"  losing yrs {lo}/{ny}  {extra}")
    return s, d


def matched_size(D, target_ann):
    lo, hi = 0.05, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if stats(sim(D, mid)[0])[0] < target_ann:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = CW.smoke_data() if a.smoke else CW.load_french()
    false = pd.Series(False, d.index)

    def french(p):
        s = (d.index >= p[0]) & (d.index <= p[1])
        x = d[s]
        return BL.period_data(x["W"], x["M"], x["RF"], false[s], false[s])

    if a.smoke:
        s3 = (d.index >= BL.P3[0]) & (d.index <= BL.P3[1])
        bk = pd.DataFrame({"STEADY": d["W"][s3] * 1.5, "SPY": d["M"][s3]})
    else:
        bk = pd.read_csv(CW.BOOK, index_col=0, parse_dates=True)
        bk = bk[(bk.index >= BL.P3[0]) & (bk.index <= BL.P3[1])]
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    f3 = pd.Series(False, bk.index)
    periods = {"1927-1989": french(BL.P1), "1990-2016": french(BL.P2),
               "CHIMERA-70": BL.period_data(bk["STEADY"], bk["SPY"].fillna(0), cash, f3, f3)}

    okA, okB = True, True
    for pn, D in periods.items():
        yrs = len(set(D["years"]))
        print(f"\n  ===== {pn} =====")
        rb, _, _ = sim(D, 0.70)
        sb, db = line("BASE fixed 70%", rb)
        dd30 = None
        for z in ZEROS:
            r, avg, tr = sim(D, 0.70, z)
            s, dw = line(f"DDCTRL zero at {z:.0%}", r,
                         f"avg share {avg:.0%}, {tr/yrs:.1f} trades/yr")
            if not (s > sb and dw >= db - 1e-9):
                okA = False
            if z == 0.30:
                dd30 = r
        ann30 = stats(dd30)[0]
        msz = matched_size(D, ann30)
        rm, _, _ = sim(D, msz)
        sm, dm = line(f"FIXED {msz:.0%} (same return)", rm)
        _, s30, d30 = stats(dd30)
        good = s30 > sm and d30 >= dm - 1e-9
        okB &= good
        print(f"    check B: DDCTRL(30%) Sharpe {s30:.2f} vs {sm:.2f}, worst "
              f"{d30*100:.1f}% vs {dm*100:.1f}% -> {'pass' if good else 'FAIL'}")
        if pn == "CHIMERA-70":
            print("    by year: BASE / DDCTRL(30%) / FIXED")
            for y in sorted(set(rb.index.year)):
                g = lambda r: ((1 + r[r.index.year == y]).prod() - 1) * 100  # noqa: E731
                print(f"      {y}: {g(rb):+5.0f}%  {g(dd30):+5.0f}%  {g(rm):+5.0f}%")

    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"    A fragility 20/30/40% beats BASE everywhere -> {'pass' if okA else 'FAIL'}")
    print(f"    B beats fixed size at the same return      -> {'pass' if okB else 'FAIL'}")
    print("    " + ("PASS -- run DD_CTRL on QuantConnect" if okA and okB
                    else "FAIL -- DDCTRL is mostly just holding less; keep CHIMERA-70"))
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
