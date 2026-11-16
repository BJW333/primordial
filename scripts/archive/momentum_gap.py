"""
momentum_gap.py -- Huang (2022, Review of Financial Studies) "The Momentum
Gap": when stocks' past-year returns are unusually SPREAD OUT, momentum
stops working (paper: 2.18%/month when the gap is small, 0.04% when large;
the only variable that significantly predicted momentum crashes; holds in
1927-1971 and 1972-2016 and in 20 of 21 countries; predicts the winners
side too, which is what CHIMERA holds).

THE GAP each month = 75th percentile minus 25th percentile of stocks'
months t-12..t-2 returns, from Ken French's "Prior (2-12) Breakpoints"
(NYSE stocks, monthly since 1927; known before month t starts).

RULE (from the paper, fixed before running -- no search, no tuning):
  gap in the top 20% of its own trailing 5 years (point in time)
  -> that month, hold T-bills instead of momentum; else hold momentum.
  0.2% cost per switch.

TESTS (pre-stated)
  A 1927-2016 on Ken French's top-10% momentum stocks (value weighted, the
    long-only winners): must raise Sharpe in BOTH 1927-1971 and 1972-2016
    AND make the 1927-2016 worst drop shallower. (The paper's own sample:
    a replication check on the long-only side.)
  B 2017-03 -> 2026-05 on CHIMERA's real book with the 30% yearly lock
    (lock -> SPY until Jan 1; gap month -> T-bills): must beat LOCK30 alone
    on BOTH Sharpe and worst drop. Mostly after the paper (2022): the
    honest out-of-sample check.
  PASS only if A and B. Then it gets ported to QC as a switch.
  Context only: the gap rebuilt from Primordial's own US stock data
  2017-2026, to check it tracks French's.

Run (repo root; downloads one more small French file):
    python3.10 scripts/momentum_gap.py --smoke
    python3.10 scripts/momentum_gap.py | tee logs/momentum_gap.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_warning_search as CW   # noqa: E402

CW.FILES["bp"] = "Prior_2-12_Breakpoints_CSV.zip"
RANK_MONTHS, LEVEL = 60, 0.80
A_ALL, A1, A2 = ("1927-01-01", "2016-12-31"), ("1927-01-01", "1971-12-31"), \
    ("1972-01-01", "2016-12-31")
B = ("2017-03-01", "2026-05-29")


def load_gap():
    """Monthly 75th-25th percentile spread of prior (2-12) returns."""
    text = CW.fetch("bp")
    rows = {}
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^\d{6}\s*,", s):
            parts = [x.strip() for x in s.split(",") if x.strip() != ""]
            vals = [float(x) for x in parts[1:]]
            pct = vals[-20:]                    # 5%, 10%, ..., 100%
            rows[pd.Period(parts[0][:4] + "-" + parts[0][4:], "M")] = \
                (pct[14] - pct[4]) / 100        # 75th - 25th
        elif rows:
            break
    return pd.Series(rows).sort_index()


def own_gap(months):
    """Same spread from Primordial's US stock data (12-1 returns, price >= $5)."""
    import blowup_us as U   # noqa: E402
    px, _, _, _ = U.load()
    out = {}
    for p in months:
        t = px.index[px.index <= p.to_timestamp(how="end")]
        if len(t) < 260:
            continue
        i = px.index.get_loc(t[-1])
        win = px.iloc[i - 252:i + 1]
        ok = (win.notna().sum() >= 240) & (win.iloc[-1] >= 5)
        r = (win.iloc[-22] / win.iloc[0] - 1)[ok].dropna()
        if len(r) > 200:
            out[p + 1] = r.quantile(0.75) - r.quantile(0.25)   # used next month
    return pd.Series(out)


def signal(gap):
    rk = gap.rolling(RANK_MONTHS, min_periods=36).rank(pct=True)
    return (rk >= LEVEL).fillna(False)


def daily_flag(sig, index):
    """warn array for CW helpers: warn[i-1] = gap month flag of day i."""
    f = pd.Series(index.to_period("M").map(lambda p: bool(sig.get(p, False))),
                  index).values.astype(bool)
    return np.r_[f[1:], f[-1]]


def smoke():
    d = CW.smoke_data().copy()
    rng = np.random.default_rng(5)
    months = d.index.to_period("M").unique()
    g = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.15, len(months)))) * 0.3, months)
    rk = g.rolling(60, min_periods=36).rank(pct=True)
    bad = set(rk[rk >= 0.8].index)
    m = d.index.to_period("M")
    d.loc[m.isin(bad), "W"] -= 0.004                 # planted: high gap -> bad month
    return d, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        d, gap = smoke()
    else:
        d = CW.load_french()
        try:
            gap = load_gap()
        except Exception as e:   # noqa: BLE001
            print(f"  could not get {CW.FILES['bp']} ({e}).\n  Download the "
                  "'Prior (2-12) Return Breakpoints' CSV zip from Ken French's data "
                  f"library into data/french/ as {CW.FILES['bp']} and rerun.")
            return
    sig = signal(gap)
    print(f"  daily {d.index[0].date()} -> {d.index[-1].date()} | gap months "
          f"{gap.index[0]} -> {gap.index[-1]}{'  [SYNTHETIC]' if a.smoke else ''}")
    print(f"  gap: median {gap.median()*100:.0f} pts, rule says 'stay out' in "
          f"{sig.mean()*100:.0f}% of months")

    # A: 1927-2016 on the winners
    print("\n  ===== A  1927-2016, French top-10% momentum stocks (long only) =====")
    res = {}
    for lab, (s0, s1) in (("1927-2016", A_ALL), ("1927-1971", A1), ("1972-2016", A2)):
        sl = (d.index >= s0) & (d.index <= s1)
        W, RF = d["W"].values[sl], d["RF"].values[sl]
        w = daily_flag(sig, d.index[sl])
        r, on = CW.apply_warn(W, RF, w[:, None])
        print(f"   {lab}  (out {on.mean()*100:.0f}% of days)")
        res[lab] = (CW.line("winners always", W), CW.line("winners + gap rule", r[:, 0]))
    okA = (res["1927-1971"][1][0] > res["1927-1971"][0][0] and
           res["1972-2016"][1][0] > res["1972-2016"][0][0] and
           res["1927-2016"][1][1] > res["1927-2016"][0][1])

    # B: CHIMERA
    print(f"\n  ===== B  CHIMERA book {B[0]} -> {B[1]}, 30% lock =====")
    if a.smoke:
        sl = (d.index >= B[0]) & (d.index <= B[1])
        bk = pd.DataFrame({"STEADY": d["W"][sl] * 1.5, "SPY": d["M"][sl]})
    else:
        bk = pd.read_csv(CW.BOOK, index_col=0, parse_dates=True)
        bk = bk[(bk.index >= B[0]) & (bk.index <= B[1])]
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    if gap.index[-1] < bk.index[-1].to_period("M"):
        print(f"   note: French gap ends {gap.index[-1]}; later months count as 'stay in'")
    w = daily_flag(sig, bk.index)
    base = CW.lock_path(bk["STEADY"], bk["SPY"], cash, None)
    test = CW.lock_path(bk["STEADY"], bk["SPY"], cash, w)
    s0, d0 = CW.line("LOCK30", base.values)
    s1, d1 = CW.line("LOCK30 + gap rule", test.values)
    for y in sorted(set(bk.index.year)):
        b = (1 + base[base.index.year == y]).prod() - 1
        t = (1 + test[test.index.year == y]).prod() - 1
        print(f"      {y}: LOCK30 {b*100:+5.0f}%   + gap rule {t*100:+5.0f}%")
    out_m = [str(p) for p in sig.index if sig[p] and B[0][:7] <= str(p) <= B[1][:7]]
    print(f"   'stay out' months ({len(out_m)}): {', '.join(out_m) or 'none'}")
    okB = s1 > s0 and d1 > d0

    if not a.smoke:
        try:
            mine = own_gap(gap.index[(gap.index >= "2016-12") & (gap.index <= "2026-05")])
            j = pd.concat([gap, mine], axis=1, keys=["french", "own"]).dropna()
            print(f"\n   context: Primordial's own gap vs French's, {len(j)} months, "
                  f"correlation {j.corr().iloc[0, 1]:.2f}")
        except Exception as e:   # noqa: BLE001
            print(f"\n   context: own gap skipped ({e})")

    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"    A 1927-2016 replication -> {'pass' if okA else 'FAIL'}")
    print(f"    B CHIMERA 2017-2026     -> {'pass' if okB else 'FAIL'}")
    print("    " + ("PASS -- port the gap rule to QC as a switch" if okA and okB
                    else "FAIL -- keep LOCK30 as is"))
    if a.smoke:
        print("  smoke: synthetic data with a planted gap effect")


if __name__ == "__main__":
    main()
