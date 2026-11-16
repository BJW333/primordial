"""
profit_guard.py -- Blake's "protect the year" idea, two parts tested separately.

  IDEA 1  FLOOR: never give back the yearly lock. Today the 30% lock moves
          the account into ETFs, which can still fall. Variants (fixed):
            FLOOR25  hold ETFs after the lock, but if the year's gain falls
                     back to +25%, move to T-bills until Jan 1
            BILLS    lock straight into T-bills (the 30% can never be lost)
  IDEA 2  GUARD: once the account is up >= 25% this year, a 10% drop from
          the account's high (since Jan 1, or since the last re-entry) moves
          momentum to T-bills. Momentum comes back when a RE-ENTRY rule fires
          (checked at the close, acted on next day). The 30% lock still works.
          RE-ENTRY rules searched (16): each single sign and every AND-pair of
            SMA50   momentum book above its 50-day average
            SMA20   momentum book above its 20-day average
            HIGH10  momentum book at a new 10-day closing high
            OFFLOW  momentum book 10%+ above its lowest close since the stop
            CALM    market's 1-month volatility NOT in its top 20% (5y rank)
          plus NEVER (stay in T-bills until Jan 1).

MODEL (same for every period; 0.2% cost on what is traded)
  momentum 70% + T-bills 30%; 30% yearly lock -> market until Jan 1
  1927-2016: Ken French top-10% momentum winners (value weighted, daily),
             market = all US stocks, T-bills = RF
  2017-03 -> 2026-05: CHIMERA-70 -- STEADY top-20 book
             (data/momentum_cap_daily.csv), market = SPY, T-bills = RF

PRE-STATED PASS RULES (no other levels -- 25 / 10 / 30 are Blake's numbers)
  IDEA 1  each variant, in ALL of 1927-1989, 1990-2016 and CHIMERA-70:
          Sharpe no more than 0.01 below BASE AND worst drop no worse.
          (It is insurance: it only has to not cost anything.)
  IDEA 2  pick the re-entry rule with the best Sharpe on 1927-1989 (the stop
          must fire >= 3 times there), then UNTOUCHED: 1990-2016 AND
          CHIMERA-70 must both show Sharpe up AND worst drop no worse.

Run (repo root; data/french and data/momentum_cap_daily.csv already exist):
    python3.10 scripts/profit_guard.py --smoke
    python3.10 scripts/profit_guard.py | tee logs/profit_guard.txt
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_warning_search as CW  # noqa: E402

COST, MOM, LOCK, ARM, DROP, FLOOR = 0.002, 0.70, 0.30, 0.25, 0.10, 0.25
P1, P2, P3 = ("1927-01-01", "1989-12-31"), ("1990-01-01", "2016-12-31"), \
    ("2017-03-01", "2026-05-29")
SIGNS = ["SMA50", "SMA20", "HIGH10", "OFFLOW", "CALM"]
RULES = [("NEVER",)] + [(s,) for s in SIGNS] + list(itertools.combinations(SIGNS, 2))


# ------------------------------------------------------------ data per period
def period_data(b, m, c, calm):
    """b book, m market (lock destination), c cash: aligned daily returns."""
    X = (1 + b).cumprod()
    return {"b": b.values, "m": m.values, "c": c.values, "X": X.values,
            "years": b.index.year.values, "idx": b.index,
            "SMA50": (X > X.rolling(50).mean()).values,
            "SMA20": (X > X.rolling(20).mean()).values,
            "HIGH10": (X >= X.rolling(10).max()).values,
            "CALM": calm.values}


def run(D, guard=None, after_lock="MKT"):
    """Account path. guard = re-entry rule tuple (None = no guard)."""
    b, m, c, X, years = D["b"], D["m"], D["c"], D["X"], D["years"]
    n = len(b)
    A, ys, yr, peak = 1.0, 1.0, years[0], 1.0
    mode, armed, low = "MOM", False, None
    out = np.empty(n)
    stops, floors, stop_days = 0, 0, []
    for i in range(n):
        if years[i] != yr:                                  # new year
            yr, ys, peak, armed = years[i], A, A, False
            if mode != "MOM":
                A *= 1 - (COST if mode in ("LOCK", "SAFE") else COST * MOM)
                mode = "MOM"
        if mode == "MOM":
            r = MOM * b[i] + (1 - MOM) * c[i]
        elif mode == "LOCK":
            r = m[i]
        else:                                               # OFF or SAFE
            r = c[i]
        A *= 1 + r
        out[i] = A
        # ---- decide tomorrow's mode at today's close
        if mode == "MOM":
            peak = max(peak, A)
            ytd = A / ys - 1
            if ytd >= ARM:
                armed = True
            if ytd >= LOCK:
                mode = "SAFE" if after_lock == "BILLS" else "LOCK"
                A *= 1 - COST
            elif guard is not None and armed and A <= peak * (1 - DROP):
                mode, low = "OFF", X[i]
                A *= 1 - COST * MOM
                stops += 1
                stop_days.append(D["idx"][i])
        elif mode == "OFF":
            low = min(low, X[i])
            if guard != ("NEVER",) and all(
                    (X[i] >= 1.10 * low) if s == "OFFLOW" else bool(D[s][i])
                    for s in guard):
                mode, peak = "MOM", A
                A *= 1 - COST * MOM
        elif mode == "LOCK" and after_lock == "FLOOR" and A / ys - 1 <= FLOOR:
            mode = "SAFE"
            A *= 1 - COST
            floors += 1
    eq = pd.Series(out, D["idx"])
    ret = eq.pct_change().fillna(eq.iloc[0] - 1)
    return ret, stops, floors, stop_days


def stats(r):
    r = np.asarray(r)
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return ann, sh, dd


def row(label, r, extra=""):
    a, s, d = stats(r)
    print(f"    {label:<26} {a*100:>+6.1f}%/yr  Sharpe {s:>5.2f}  worst {d*100:>6.1f}%  {extra}")
    return s, d


def by_year(cols):
    names = list(cols)
    base = cols[names[0]]
    for y in sorted(set(base.index.year)):
        parts = [f"{nm} {((1 + cols[nm][cols[nm].index.year == y]).prod() - 1)*100:+5.0f}%"
                 for nm in names]
        print(f"      {y}: " + "   ".join(parts))


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = CW.smoke_data() if a.smoke else CW.load_french()
    calm_all = (CW.features(d)["MKTVOL"] < 0.80).fillna(False)

    def french(p):
        s = (d.index >= p[0]) & (d.index <= p[1])
        x = d[s]
        return period_data(x["W"], x["M"], x["RF"], calm_all[s])

    if a.smoke:
        s3 = (d.index >= P3[0]) & (d.index <= P3[1])
        bk = pd.DataFrame({"STEADY": d["W"][s3] * 1.5, "SPY": d["M"][s3]})
    else:
        bk = pd.read_csv(CW.BOOK, index_col=0, parse_dates=True)
        bk = bk[(bk.index >= P3[0]) & (bk.index <= P3[1])]
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    calm3 = calm_all.reindex(bk.index).ffill().fillna(False)
    periods = {"1927-1989": french(P1), "1990-2016": french(P2),
               "CHIMERA-70": period_data(bk["STEADY"], bk["SPY"].fillna(0), cash, calm3)}
    print(f"  French {d.index[0].date()} -> {d.index[-1].date()} | CHIMERA-70 "
          f"{bk.index[0].date()} -> {bk.index[-1].date()}"
          f"{'  [SYNTHETIC]' if a.smoke else ''}")
    print(f"  model: {MOM:.0%} momentum + T-bills, {LOCK:.0%} yearly lock -> market")

    # ================= IDEA 1
    print("\n  ================= IDEA 1: FLOOR -- never give back the lock =================")
    base, ok1 = {}, {"FLOOR25": True, "BILLS": True}
    keep = {}
    for pn, D in periods.items():
        print(f"  {pn}:")
        rb, _, _, _ = run(D)
        base[pn] = rb
        sb, db = row("BASE (lock -> market)", rb)
        for v, al in (("FLOOR25", "FLOOR"), ("BILLS", "BILLS")):
            rv, _, fl, _ = run(D, after_lock=al)
            sv, dv = row(v, rv, f"floor fired {fl}x" if v == "FLOOR25" else "")
            if not (sv >= sb - 0.01 and dv >= db - 1e-9):
                ok1[v] = False
            if pn == "CHIMERA-70":
                keep[v] = rv
    print("    CHIMERA-70 by year:")
    by_year({"BASE": base["CHIMERA-70"], "FLOOR25": keep["FLOOR25"], "BILLS": keep["BILLS"]})

    # ================= IDEA 2
    print("\n  ================= IDEA 2: GUARD -- up 25%, then -10% -> T-bills until re-entry =================")
    D1 = periods["1927-1989"]
    sb1 = stats(base["1927-1989"])[1]
    res = []
    for g in RULES:
        r, st, _, _ = run(D1, guard=g)
        res.append((stats(r)[1] - sb1, st, g, r))
    print(f"  search 1927-1989 (BASE Sharpe {sb1:.2f}); re-entry rule, Sharpe gain, stops:")
    for gain, st, g, r in sorted(res, key=lambda x: -x[0]):
        a_, s_, d_ = stats(r)
        print(f"    {gain:+.3f}  stops {st:>3}  worst {d_*100:6.1f}%  {' AND '.join(g)}")
    ok_res = [x for x in res if x[1] >= 3]
    if not ok_res:
        print("  stop fired < 3 times for every rule -> FAIL (too rare to judge)")
        pass2, best = False, None
    else:
        best = max(ok_res, key=lambda x: x[0])[2]
        print(f"  fixed re-entry rule: {' AND '.join(best)}")
        pass2 = True
        for pn in ("1990-2016", "CHIMERA-70"):
            D = periods[pn]
            print(f"  {pn} (untouched):")
            sb, db = row("BASE", base[pn])
            rg, st, _, sd = run(D, guard=best)
            sg, dg = row("GUARD", rg, f"stops {st}")
            if not (sg > sb and dg >= db - 1e-9):
                pass2 = False
            if pn == "CHIMERA-70":
                print("    stop dates: " + (", ".join(f"{x:%Y-%m-%d}" for x in sd) or "none"))
                by_year({"BASE": base[pn], "GUARD": rg})

    print("\n  ===== VERDICT (pre-stated) =====")
    for v in ("FLOOR25", "BILLS"):
        print(f"    IDEA 1 {v:<8}: {'PASS -- add as a QC switch' if ok1[v] else 'FAIL'}")
    print(f"    IDEA 2 GUARD   : "
          f"{'PASS -- port to QC as a switch' if pass2 else 'FAIL'}"
          f"{'' if best is None else '  (re-entry: ' + ' AND '.join(best) + ')'}")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
