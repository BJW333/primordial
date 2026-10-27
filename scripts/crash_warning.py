"""
crash_warning.py -- MOMENTUM CRASH WARNING from the fade work (long-only).

Momentum's worst losses come from JUNK RALLIES: beaten-down losers suddenly
rip while winners drop. The attention data lets us measure that directly:
count good-news Wikipedia hype spikes among LOSER stocks.

SIGNAL (point-in-time, at each month-end):
  JUNK = good-news spikes (daily views >= 3x prior-90-day median AND stock
         up > 2% that day) in the last month among liquid US stocks (price
         >= $5, >= $5M/day) whose 12-month return is in the BOTTOM HALF of
         liquid stocks that day -- divided by the number of liquid stocks
         with a Wikipedia page (so it's a rate, not a raw count)
  WARNING = this month's JUNK rate is above the 80th percentile of its own
         history so far (expanding window, at least 24 months of history)

ACTION: the month after a WARNING, momentum (all-US steady, no stop --
momentum_v2.py STEADY, data/momentum_v2_returns.csv) is cut to 50%.
  DIAL_CASH   other 50% in cash (0% return -- conservative)  <- PRIMARY
  DIAL_SPY    other 50% in SPY
  DIAL_ZERO   momentum fully off (100% cash) during warnings

PRE-STATED: DIAL_CASH PASSES if, vs plain momentum, its Sharpe is HIGHER
AND its worst drop is SMALLER in BOTH 2017-02 -> 2021-05 and 2021-06 ->
2026-05. (Lower return is allowed -- the point is better return per unit
of risk, which can then be levered toward 30%.)
Diagnostics: momentum's return in warning vs normal months; its 10 worst
months and whether a warning was on.

Run (repo root):
    python3.10 scripts/crash_warning.py | tee logs/crash_warning.txt
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

ROOT = os.path.dirname(HERE)
MOMF = os.path.join(ROOT, "data", "momentum_v2_returns.csv")
RECENT = pd.Period("2021-06", "M")
PCT, MINHIST = 0.80, 24


def junk_rate(px, vol, views):
    U.MIN_PX = 5.0
    sets, _ = U.liquid_members(px, vol, 5e6)
    days = sorted(sets)

    def members(t):
        past = [d for d in days if d <= t]
        return sets[past[-1]] if past else set()
    AR.START = pd.Timestamp("2015-10-01")
    ev = AR.good_spikes(views, px, members)
    pxf = px.ffill()
    r12 = pxf / pxf.shift(252) - 1
    dv = (px * vol).rolling(63, min_periods=40).mean()
    liquid = (px >= 5) & (dv >= 5e6)
    med = r12.where(liquid).median(axis=1)
    loser = []
    for s, d in zip(ev["sym"], ev["d"]):
        v = r12.at[d, s]
        loser.append(bool(np.isfinite(v) and v < med.get(d, np.nan)))
    ev = ev.assign(loser=loser)
    m = ev.groupby(ev["d"].dt.to_period("M")).agg(
        all=("sym", "size"), junk=("loser", "sum"))
    page = [c for c in px.columns if c in views.columns]
    me = px.resample("ME").last().index
    den = pd.Series({d.to_period("M"): int(liquid.loc[
        px.index[px.index <= d][-1], page].sum()) for d in me
        if (px.index <= d).any()})
    m = m.reindex(den.index).fillna(0)
    m["den"] = den
    m["junk_rate"] = m["junk"] / m["den"].replace(0, np.nan)
    m["froth_rate"] = m["all"] / m["den"].replace(0, np.nan)
    return m


def warn_flags(rate):
    out = {}
    hist = []
    for p, v in rate.items():
        if len(hist) >= MINHIST and np.isfinite(v):
            out[p] = v > np.nanquantile(hist, PCT)
        else:
            out[p] = False
        if np.isfinite(v):
            hist.append(v)
    return pd.Series(out)


def stats(x):
    x = x.dropna()
    ann = (1 + x).prod() ** (12 / len(x)) - 1
    vol = x.std() * np.sqrt(12)
    eq = (1 + x).cumprod()
    return ann, vol, x.mean() * 12 / vol if vol else np.nan, \
        (eq / eq.cummax() - 1).min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, views, _ = AR.smoke()
        rng = np.random.default_rng(9)
        mi = px.resample("ME").last().index.to_period("M")
        mom = pd.DataFrame({"STEADY": rng.normal(0.02, 0.09, len(mi)),
                            "SPY": rng.normal(0.01, 0.045, len(mi))}, mi)
    else:
        px, vol, _, _ = U.load()
        px = px[[c for c in px.columns if "~" not in c]]
        vol = vol[px.columns]
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        m = pd.read_csv(MOMF, index_col=0, parse_dates=True)
        mom = m[["STEADY", "SPY"]]
        mom.index = mom.index.to_period("M")
    jr = junk_rate(px, vol, views)
    flag = warn_flags(jr["junk_rate"])
    # warning measured at month-end p applies to the NEXT month's return
    w = flag.shift(1).reindex(mom.index).fillna(False).astype(bool)
    mo, spy = mom["STEADY"], mom["SPY"]
    books = {"MOMENTUM": mo,
             "DIAL_CASH": mo.where(~w, 0.5 * mo),
             "DIAL_SPY": mo.where(~w, 0.5 * mo + 0.5 * spy),
             "DIAL_ZERO": mo.where(~w, 0.0),
             "SPY": spy}
    print(f"\n  junk-rally rate: {len(jr)} months; warnings on "
          f"{w.mean()*100:.0f}% of momentum months")
    print(f"  momentum in WARNING months: {mo[w].mean()*100:+.2f}%/mo "
          f"(n {w.sum()}, worst {mo[w].min()*100:+.1f}%) | normal months: "
          f"{mo[~w].mean()*100:+.2f}%/mo (n {(~w).sum()})")
    print(f"\n  {'book':<11} | {'2017-21 ann':>11} {'sh':>5} {'worst':>6} | "
          f"{'last5y ann':>10} {'sh':>5} {'worst':>6} | {'all ann':>8} "
          f"{'worst':>6}")
    res = {}
    for k, b in books.items():
        s0 = stats(b[b.index < RECENT])
        s1 = stats(b[b.index >= RECENT])
        sa = stats(b)
        res[k] = (s0, s1)
        print(f"  {k:<11} | {s0[0]*100:>+10.1f}% {s0[2]:>5.2f} "
              f"{s0[3]*100:>5.0f}% | {s1[0]*100:>+9.1f}% {s1[2]:>5.2f} "
              f"{s1[3]*100:>5.0f}% | {sa[0]*100:>+7.1f}% {sa[3]*100:>5.0f}%")
    print("\n  momentum's 10 worst months (warning on?):")
    for p, v in mo.nsmallest(10).items():
        print(f"    {p}  {v*100:+6.1f}%   warning {'YES' if w.get(p) else 'no'}"
              f"   junk rate prior month "
              f"{jr['junk_rate'].get(p - 1, np.nan)*100:.2f}%")
    print("\n  warning months (date: momentum that month):  " + ", ".join(
        f"{p} {mo[p]*100:+.0f}%" for p in mo.index[w]))
    (m0, m1), (d0, d1) = res["MOMENTUM"], res["DIAL_CASH"]
    ok = d0[2] > m0[2] and d1[2] > m1[2] and d0[3] > m0[3] and d1[3] > m1[3]
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    print(f"  DIAL_CASH beats momentum on Sharpe AND worst drop, both "
          f"periods  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
