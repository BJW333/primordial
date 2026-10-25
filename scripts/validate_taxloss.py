"""
validate_taxloss.py -- the two quick gates owed by the C2 tax-loss cell
(check_c2_cohorts.py) before any Norgate / forward-paper spend.

CELL (copied unchanged from check_c2_cohorts.py):
  universe   S&P 600 current membership (manifests/us_sp600.yaml)
  signal     close <= the lowest close of the prior 21 days (new 21d low)
             AND return from t-127 to t-1 <= -30% (distressed)
             AND date is the 1st-24th of the month   <- month is the variable
             events per stock at least 5 bars apart
  trade      buy next open, sell at the close 20 trading days after signal
  outcome    stock return minus SPY over the same 20 days, minus cost
  unit       one equal-weight mean per (year, month) cohort

GATE 1 -- PLACEBO MONTHS (frozen in the original C2 plan):
  run the identical cell with "December" replaced by each of the 12 months.
  PASS only if December ranks #1 of 12 by across-cohort mean.
  Why this is survivorship-resistant: dead names are missing from EVERY
  month equally, so a survivor-made edge shows up in all months, not just
  December. A real tax-deadline effect has to stand out.

GATE 2 -- COST STRESS (frozen in the original C2 plan):
  December's across-cohort mean stays > 0 at 2x and 3x cost (72 / 108 bps).

BOTH pass -> Norgate PIT re-test (sp600 with dead names), then a runner
for the calendar-windowed strategy, then forward paper.
Either fails -> C2 closed; logged to MECHANISM_MAP.md either way.

CONTEXT ONLY (decides nothing): the same December cell on S&P 400 (current
list) and S&P 500 (membership checked by date, dead names partly present).

Run (repo root):
    python3.10 scripts/validate_taxloss.py
    python3.10 scripts/validate_taxloss.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dip_features as DF  # noqa: E402  (loaders + PIT membership)

COST = 36 / 1e4
DISTRESS = -0.30
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
          "Oct", "Nov", "Dec"]


def events(data, spy_close, mem=None):
    """All cell events, any month. Returns year, month, gross excess f20."""
    rows = []
    for sym, df in data.items():
        df = df[~df.index.duplicated()]
        c, o = df["close"], df["open"]
        low21 = c.rolling(21).min().shift(1)
        trail = c.shift(1) / c.shift(127) - 1.0
        member = mem(sym, df.index) if mem else np.ones(len(df), bool)
        idxs = np.flatnonzero((c <= low21).values)
        last = -99
        for i in idxs:
            if i - last < 5 or i < 130 or i + 21 >= len(c):
                continue
            last = i                      # spacing counts every 21d low,
            t = c.index[i]                # exactly as in check_c2_cohorts
            if t.day > 24 or not member[i]:
                continue
            if not float(trail.iloc[i]) <= DISTRESS:
                continue
            si = spy_close.index.searchsorted(t)
            if si + 21 >= len(spy_close) or si < 1:
                continue
            e = float(o.iloc[i + 1])
            if not e > 0:
                continue
            f20 = float(c.iloc[i + 20]) / e - 1 - \
                (float(spy_close.iloc[si + 20]) / float(spy_close.iloc[si])
                 - 1)
            rows.append({"year": t.year, "month": t.month, "sym": sym,
                         "f20": f20})
    return pd.DataFrame(rows)


def cohort_stats(ev, cost):
    co = (ev["f20"] - cost).groupby(ev["year"]).mean()
    if len(co) < 3:
        return np.nan, np.nan, 0, 0, len(ev)
    t = co.mean() / (co.std(ddof=1) / np.sqrt(len(co)))
    return co.mean(), t, int((co > 0).sum()), len(co), len(ev)


def load(ground, a):
    if a.smoke:
        return a.smoke_data[0], a.smoke_data[1], None
    data, spy, mem, _ = DF.load(ground, a)
    return data, spy["close"], mem


def placebo_table(ev, label):
    print(f"\n  == {label}: same cell, every month (cost {COST*1e4:.0f} bps) ==")
    print(f"  {'month':<5} {'events':>7} {'cohorts':>8} {'mean/cohort':>12} "
          f"{'t':>6} {'positive':>9}")
    res = {}
    for m in range(1, 13):
        mu, t, pos, n, ne = cohort_stats(ev[ev["month"] == m], COST)
        res[m] = mu
        print(f"  {MONTHS[m-1]:<5} {ne:>7} {n:>8} {mu*100:>+11.2f}% "
              f"{t:>+6.2f} {pos:>4}/{n:<4}")
    ranked = sorted(res, key=lambda k: -np.nan_to_num(res[k], nan=-9))
    return res, ranked


def smoke_data():
    """Planted: distressed new-lows in December rebound +6% vs SPY."""
    idx = pd.bdate_range("2010-01-01", "2026-05-29")
    rng = np.random.default_rng(12)
    data = {}
    for k in range(120):
        r = rng.normal(-0.0004, 0.03, len(idx))
        c = 100 * np.exp(np.cumsum(r))
        s = pd.Series(c, idx)
        low21 = s.rolling(21).min().shift(1)
        trail = s.shift(1) / s.shift(127) - 1
        hit = (s <= low21) & (trail <= DISTRESS) & (idx.month == 12) & \
            (idx.day <= 24)
        for i in np.flatnonzero(hit.values)[::5]:
            if i + 22 < len(idx):
                r[i + 2:i + 22] += 0.003
        c = 100 * np.exp(np.cumsum(r))
        data[f"S{k}"] = pd.DataFrame({"open": c, "high": c, "low": c,
                                      "close": c, "volume": 1e6}, idx)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01,
                                                      len(idx)))), idx)
    return data, spy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.smoke_data = smoke_data()

    data, spy, mem = load("sp600", a)
    ev = events(data, spy, mem)
    res, ranked = placebo_table(ev, "S&P 600 (DECIDES)")
    dec_rank = ranked.index(12) + 1
    others = [v for m, v in res.items() if m != 12 and np.isfinite(v)]
    print(f"\n  December rank: #{dec_rank} of 12 | December "
          f"{res[12]*100:+.2f}% vs other-month average "
          f"{np.mean(others)*100:+.2f}%")

    dec = ev[ev["month"] == 12]
    print("\n  == COST STRESS (December, S&P 600) ==")
    stress = {}
    for mult in (1, 2, 3):
        mu, t, pos, n, _ = cohort_stats(dec, COST * mult)
        stress[mult] = mu
        print(f"  {mult}x cost ({COST*mult*1e4:>4.0f} bps): "
              f"{mu*100:+.2f}% per cohort, t {t:+.2f}, {pos}/{n} positive")

    if not a.smoke:
        print("\n  == CONTEXT ONLY (December cell elsewhere) ==")
        for g in ("sp400", "sp500"):
            d2, s2, m2 = load(g, a)
            e2 = events(d2, s2, m2)
            r2, rk2 = {}, None
            for m in range(1, 13):
                r2[m] = cohort_stats(e2[e2["month"] == m], COST)[0]
            rk2 = sorted(r2, key=lambda k: -np.nan_to_num(r2[k], nan=-9))
            mu, t, pos, n, ne = cohort_stats(e2[e2["month"] == 12], COST)
            tag = " (membership-gated, dead names partly present)" \
                if g == "sp500" else " (current list)"
            print(f"  {g}{tag}: Dec {mu*100:+.2f}% t {t:+.2f} "
                  f"{pos}/{n} positive, {ne} events, Dec rank "
                  f"#{rk2.index(12)+1}/12")

    ok1 = dec_rank == 1
    ok2 = stress[2] > 0 and stress[3] > 0
    print("\n  ===== VERDICT (gates frozen in the original C2 plan) =====")
    print(f"  GATE 1 placebo: December #1 of 12?  "
          f"{'YES' if ok1 else f'no (#{dec_rank})'}")
    print(f"  GATE 2 cost: positive at 2x and 3x? {'YES' if ok2 else 'no'}")
    if a.smoke:
        print("  smoke: planted December effect (pipe check)")
    if ok1 and ok2:
        print("  PASS -- tax-deadline effect stands out from every other "
              "month and survives costs. Next: Norgate PIT sp600, then a "
              "calendar runner + forward paper.")
    else:
        print("  FAIL -- C2 was survivorship/concentration, not a tax "
              "deadline. Close and log.")


if __name__ == "__main__":
    main()
