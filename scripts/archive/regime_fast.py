"""
regime_fast.py -- can REGIME react faster than once a month?

WHY: REGIME checks on the 1st of the month only. Oct 2018 (-12.8%) and
Feb-Mar 2020 hit weeks before the next check. Research (Harvey et al. 2018:
volatility clusters, so measure often; Garg-Goulding-Harvey-Mazzoleni 2023:
turning points when fast and slow trends disagree) says faster reads help.

REGIME readings (same as regime_sizer.py; US momentum winners series):
  swings = last-month (21-day) volatility, ranked vs its last 5 years
  calm 100%: swings in bottom 20% AND 12-month return > 0
  stressed 35%: swings in top 20% OR 12-month return <= 0
  normal 70%: everything else      (rest in T-bills; 30% yearly lock)

THREE FAST VERSIONS (fixed before running):
  WEEKLY    same readings, checked every Friday (week's last trading day),
            both directions
  FASTDOWN  checked every Friday but a weekly check can only CUT; adding
            momentum back waits for the month-end check (less flip-flopping)
  TURN      WEEKLY + turning point: also stressed when momentum's last
            2-month (42-day) return <= 0 (fast trend down)
Reference: MONTHLY = the current REGIME.

PASS (pre-stated) -- all of:
  1 PICK the best of the three on US 1927-1989 (Sharpe gain over its
    same-return fixed size, worst drop no worse) -- judge.py
  2 judge.py: (a) US 1990-2016 beats same-return fixed size, (b) luck test
    p < 0.017, (c) CHIMERA do-no-harm
  3 it must ALSO beat the current MONTHLY REGIME on US 1990-2016: higher
    Sharpe (above T-bills) AND worst drop no worse. Otherwise faster is not
    better, keep monthly.
  Pass -> port to QuantConnect as a switch.

Run (repo root; US data on disk; needs dd2_lab.py, judge.py,
regime_sizer.py, crash_warning_search.py in scripts/):
    python3.10 scripts/regime_fast.py --smoke
    python3.10 scripts/regime_fast.py | tee logs/regime_fast.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_warning_search as CW  # noqa: E402
import dd2_lab as D2               # noqa: E402
import judge as J                  # noqa: E402
import regime_sizer as RS          # noqa: E402

CALM, NORMAL, STRESS = RS.CALM, RS.NORMAL, RS.STRESS
VERSIONS = ["WEEKLY", "FASTDOWN", "TURN"]


def all_regimes(ret):
    """Daily size series for MONTHLY and the three fast versions."""
    vol = ret.rolling(21).std()
    rk = vol.rolling(1260, min_periods=756).rank(pct=True)
    lr = np.log1p(ret)
    yr = np.expm1(lr.rolling(252).sum())
    fast = np.expm1(lr.rolling(42).sum())
    calm = (rk <= 0.20) & (yr > 0)
    stress = (rk >= 0.80) | (yr <= 0)
    s = pd.Series(NORMAL, ret.index)
    s[calm] = CALM
    s[stress] = STRESS
    st = s.copy()
    st[fast <= 0] = STRESS
    idx = ret.index
    me = np.r_[idx.month[1:] != idx.month[:-1], True]
    wk = idx.isocalendar().week.values
    we = np.r_[wk[1:] != wk[:-1], True]
    out = {"MONTHLY": s.where(me).ffill().fillna(NORMAL).values,
           "WEEKLY": s.where(we).ffill().fillna(NORMAL).values,
           "TURN": st.where(we).ffill().fillna(NORMAL).values}
    sv, fd, cur = s.values, np.empty(len(s)), NORMAL
    for i in range(len(s)):
        if me[i]:
            cur = sv[i]
        elif we[i] and sv[i] < cur:
            cur = sv[i]
        fd[i] = cur
    out["FASTDOWN"] = fd
    return out


def changes_per_year(reg, idx):
    reg = np.asarray(reg)
    return (np.abs(np.diff(reg)) > 1e-9).sum() / (len(reg) / 252)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = CW.smoke_data() if a.smoke else CW.load_french()
    D1, tests = D2.load_tests(d, a.smoke)
    regs_us = {k: pd.Series(v, d.index) for k, v in all_regimes(d["W"]).items()}
    for D in [D1] + list(tests.values()):
        D["regs"] = {k: s.reindex(D["idx"]).ffill().fillna(NORMAL).values
                     for k, s in regs_us.items()}
        D["reg"] = D["regs"]["MONTHLY"]

    print("  size changes per year (how often the dial moves):")
    for nm, D in [("US 1927-1989", D1)] + list(tests.items()):
        print(f"    {nm:<14} " + "  ".join(
            f"{k} {changes_per_year(D['regs'][k], D['idx']):.1f}"
            for k in ["MONTHLY"] + VERSIONS))

    def rule(D, v):
        D["reg"] = D["regs"][v]
        return RS.sim(D, ("none",))

    fixed = lambda D, s: RS.sim(D, fixed=s)     # noqa: E731
    ok, best = D2.pick_and_judge("FAST REGIME (best of WEEKLY / FASTDOWN / TURN)",
                                 VERSIONS, D1, tests, rule, fixed, lambda v: v)

    print("\n  ===== 3 FASTER vs CURRENT MONTHLY REGIME =====")
    ok3 = False
    if best:
        for tn, D in tests.items():
            rf, rm = rule(D, best), rule(D, "MONTHLY")
            af, sf, wf = J.stats(rf, D["c"])
            am, sm, wm = J.stats(rm, D["c"])
            good = sf > sm and wf >= wm - 1e-9
            tag = "pass" if good else "FAIL"
            if tn == "US 1990-2016":
                ok3 = good
            else:
                tag += " (context)"
            print(f"  {tn}: {best} {af*100:+.1f}%/yr Sharpe {sf:.2f} worst {wf*100:.1f}% | "
                  f"MONTHLY {am*100:+.1f}%/yr Sharpe {sm:.2f} worst {wm*100:.1f}% -> {tag}")
            if tn == "CHIMERA-70":
                print("    by year  MONTHLY / " + best)
                for y in sorted(set(rf.index.year)):
                    g = lambda r: ((1 + r[r.index.year == y]).prod() - 1) * 100  # noqa: E731
                    print(f"      {y}: {g(rm):+5.0f}% {g(rf):+5.0f}%")
                for m in ["2018-10", "2020-02", "2020-03", "2022-01", "2023-10", "2025-04"]:
                    sel = (rf.index.strftime("%Y-%m") == m)
                    if sel.any():
                        print(f"      {m}: MONTHLY {((1 + rm[sel]).prod() - 1)*100:+5.1f}%  "
                              f"{best} {((1 + rf[sel]).prod() - 1)*100:+5.1f}%")

    print("\n  ===== FINAL VERDICT (pre-stated) =====")
    print(f"    judge.py (pick + US 1990-2016 + luck + CHIMERA) -> {'pass' if ok else 'FAIL'}")
    print(f"    beats current MONTHLY REGIME on US 1990-2016   -> {'pass' if ok3 else 'FAIL'}")
    print("    " + (f"PASS -- port {best} to QuantConnect" if ok and ok3
                    else "FAIL -- keep REGIME checking monthly"))
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
