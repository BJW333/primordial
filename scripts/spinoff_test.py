"""
spinoff_test.py -- the spinoff dip: forced selling of newly spun-off
companies, then the rebound.

Needs data/spinoffs.csv from fetch_spinoffs.py.

MECHANISM: parent shareholders receive shares of a new, smaller company.
Index funds and mandate-bound funds often can't hold it and must sell in
the first weeks -- selling unrelated to value. Buy after the forced
selling, ride the recovery.

FROZEN SPEC (stated before running)
  new listing  first price bar between the first Form 10-12B filing and
               400 days after it (drops re-registrations of old names)
  day 0        first trading day; FORCED-SELLING window = days 0-20
  entry        close of trading day 20
  PRIMARY      hold 120 trading days; return minus IWM (small caps) over
               the same window, minus 40 bps round trip
  DIP version  split spinoffs by days 0-20 return vs IWM: FELL (<= -5%)
               vs DIDN'T. Reported; the "rebounding dip" read is whether
               FELL beats DIDN'T.
  unit         events averaged per calendar month of entry, then t across
               months (spinoffs cluster in hot months)

PASS: primary net excess > 0 with t > 2 over the full sample AND positive
in both halves (entries before / from 2016-01-01) AND at least 150 usable
spinoffs (else VOID: too few to say anything).
Pass -> genome/runner + forward paper. Fail -> closed, logged.

SURVIVORSHIP: spinoffs later bought out (often good outcomes) and ones
that went bust (bad) both tend to vanish from Yahoo. The script prints
how many matched spinoffs had usable prices; low coverage = read with
caution in BOTH directions.

Run (repo root):
    python3.10 scripts/spinoff_test.py
    python3.10 scripts/spinoff_test.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial.data import fetch  # noqa: E402

SPIN = os.path.join(ROOT, "data", "spinoffs.csv")
START, END = "2006-01-01", "2026-06-01"
WINDOW, HOLD, COST, FELL = 20, 120, 0.0040, -0.05
SPLIT = pd.Timestamp("2016-01-01")
MIN_EVENTS = 150


def measure(sp, data, bench):
    rows = []
    b = bench.copy()
    for _, r in sp.iterrows():
        df = data.get(r["ticker"])
        if df is None or len(df) < WINDOW + HOLD + 2:
            continue
        df = df[~df.index.duplicated()]
        d0 = df.index[0]
        f = r["first_10_12b"]
        if not (f <= d0 <= f + pd.Timedelta(days=400)):
            continue                      # not a fresh listing
        if d0 <= pd.Timestamp(START) + pd.Timedelta(days=10):
            continue                      # history predates our window
        c = df["close"].values
        bi = b.reindex(df.index, method="ffill").values
        if not np.isfinite(bi[[0, WINDOW]]).all():
            continue
        dip = (c[WINDOW] / c[0] - 1) - (bi[WINDOW] / bi[0] - 1)
        x = min(WINDOW + HOLD, len(c) - 1)
        full = x == WINDOW + HOLD
        ex = (c[x] / c[WINDOW] - 1) - (bi[x] / bi[WINDOW] - 1) - COST
        if abs(c[x] / c[WINDOW] - 1) > 3:
            continue                      # data glitch guard
        rows.append({"ticker": r["ticker"], "name": r.get("name", ""),
                     "d0": d0, "entry": df.index[WINDOW], "dip": dip,
                     "excess": ex, "full": full})
    return pd.DataFrame(rows)


def mstat(e, col="excess"):
    if len(e) < 3:
        return np.nan, np.nan, len(e)
    m = e.groupby(e["entry"].dt.to_period("M"))[col].mean()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return m.mean() * 1e4, np.nan, len(e)
    return m.mean() * 1e4, m.mean() / (m.std(ddof=1) / np.sqrt(len(m))), \
        len(e)


def report(ev):
    print(f"\n  {'slice':<34} {'n':>5} {'bps':>9} {'t':>7}")
    rows = {"ALL (primary)": ev,
            "entries before 2016": ev[ev.entry < SPLIT],
            "entries 2016+": ev[ev.entry >= SPLIT],
            "only full 120-day holds": ev[ev.full]}
    out = {}
    for k, e in rows.items():
        mu, t, n = mstat(e)
        out[k] = (mu, t, n)
        print(f"  {k:<34} {n:>5} {mu:>+9.1f} {t:>+7.2f}")
    mu, t, n = mstat(ev, "dip")
    print(f"\n  forced-selling window (days 0-20 vs IWM): {mu:+.1f} bps "
          f"t {t:+.2f} -- negative = the dip exists")
    fell, held = ev[ev.dip <= FELL], ev[ev.dip > FELL]
    a, b = mstat(fell), mstat(held)
    print(f"  DIP version: FELL (<= {FELL:.0%} in window) {a[0]:+.1f} bps "
          f"t {a[1]:+.2f} (n {a[2]}) | DIDN'T {b[0]:+.1f} bps t {b[1]:+.2f}"
          f" (n {b[2]})")
    return out


def smoke():
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2006-01-01", "2026-05-29")
    bench = pd.DataFrame({"close": 100 * np.exp(np.cumsum(
        rng.normal(0.0003, 0.012, len(idx))))}, idx)["close"]
    data, rows = {}, []
    for k in range(300):
        s = rng.integers(300, len(idx) - 400)
        n = len(idx) - s
        r = rng.normal(0.0003, 0.025, n)
        r[:20] -= 0.004                      # forced selling
        r[20:140] += 0.0008                  # planted recovery ~ +10%
        c = 100 * np.exp(np.cumsum(r))
        tk = f"SP{k}"
        data[tk] = pd.DataFrame({"open": c, "high": c, "low": c, "close": c,
                                 "volume": 1e6}, idx[s:])
        rows.append({"cik": k, "name": tk, "ticker": tk,
                     "first_10_12b": idx[s] - pd.Timedelta(days=60)})
    return pd.DataFrame(rows), data, bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        sp, data, bench = smoke()
    else:
        if not os.path.exists(SPIN):
            sys.exit("run fetch_spinoffs.py first")
        sp = pd.read_csv(SPIN, parse_dates=["first_10_12b"]).dropna(
            subset=["ticker"])
        print(f"  {len(sp)} 10-12B registrants with a trading symbol")
        data = fetch(a.source, sorted(sp["ticker"].unique()), "1d", START,
                     END, min_bars=WINDOW + HOLD + 2)
        bench = fetch(a.source, ["IWM"], "1d", START, END)["IWM"]["close"]
        print(f"  price data for {len(data)} of {sp['ticker'].nunique()} "
              f"symbols")
    ev = measure(sp, data, bench)
    print(f"  usable spinoffs (fresh listings, >= {WINDOW + HOLD} bars): "
          f"{len(ev)}")
    if len(ev):
        print(f"  entries {ev.entry.min().date()} -> {ev.entry.max().date()}")
    out = report(ev) if len(ev) >= 3 else {}

    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: planted forced-selling dip + recovery (pipe check)")
    if len(ev) < MIN_EVENTS:
        print(f"  VOID -- {len(ev)} usable spinoffs < {MIN_EVENTS}; too few "
              f"to conclude either way.")
        return
    mu, t, _ = out["ALL (primary)"]
    h1, h2 = out["entries before 2016"][0], out["entries 2016+"][0]
    if mu > 0 and t > 2 and h1 > 0 and h2 > 0:
        print(f"  PASS -- buying spinoffs after the forced-selling window "
              f"beats small caps by {mu:+.0f} bps per 120 days (t {t:+.2f}), "
              f"both halves positive. Next: runner + forward paper.")
    else:
        print(f"  FAIL -- {mu:+.0f} bps, t {t:+.2f}, halves {h1:+.0f} / "
              f"{h2:+.0f}. Spinoff rebound not tradeable on this data.")


if __name__ == "__main__":
    main()
