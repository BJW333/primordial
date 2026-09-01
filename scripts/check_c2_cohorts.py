"""
C2 TAX-LOSS COHORT CHECK -- the pre-registered gate between "passed both
looks" (train +4.73 t+6.1, tail +9.79 t+5.0) and prereg-eligibility.

Problem it addresses: the 515 events cluster inside ~16 Decembers and
December distress names crash/rebound TOGETHER, so the naive t assumes
independence it does not have. The honest unit is the DECEMBER COHORT.

FROZEN GATE (stated before running): equal-weight the cell events within
each December -> one mean per cohort. PASS requires BOTH:
  (a) >= 70% of cohorts positive (sign test), and
  (b) t >= 2.0 computed ACROSS cohort means (n ~= 16).
Fail = C2 recorded as fat-year concentration, not a real deadline effect;
no prereg. This uses the full sample: both looks are already spent, this
is a robustness diagnostic on a passed cell, not a new selection step.

Standing caveat carried regardless of outcome: sp600 current membership --
December-distress names that later died are absent; bias UP. A PIT re-test
(Norgate) is owed before deployment talk, per house protocol.

    python3.10 scripts/check_c2_cohorts.py
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primordial.data import fetch  # noqa: E402

COST = 36 / 1e4
DISTRESS = -0.30


def main():
    d = yaml.safe_load(open("manifests/us_sp600.yaml"))
    sf = os.path.join("manifests", d["symbols_file"])
    syms = [x.strip() for x in open(sf) if x.strip() and not
            x.startswith("#")]
    data = fetch(d["source"], syms, "1d", d["start"], d["end"], min_bars=300)
    spy = fetch(d["source"], ["SPY"], "1d", d["start"], d["end"],
                min_bars=300)["SPY"]["close"]

    rows = []
    for sym, df in data.items():
        c, o = df["close"], df["open"]
        low21 = c.rolling(21).min().shift(1)
        trail = c.shift(1) / c.shift(127) - 1.0
        idxs = np.flatnonzero((c <= low21).values)
        last = -99
        for i in idxs:
            if i - last < 5 or i < 130 or i + 21 >= len(c):
                continue
            last = i
            t = c.index[i]
            if not (t.month == 12 and t.day <= 24):
                continue
            if not float(trail.iloc[i]) <= DISTRESS:
                continue
            si = spy.index.searchsorted(t)
            if si + 21 >= len(spy) or si < 1:
                continue
            e = float(o.iloc[i + 1])
            if not e > 0:
                continue
            f20 = float(c.iloc[i + 20]) / e - 1 - \
                (float(spy.iloc[si + 20]) / float(spy.iloc[si]) - 1) - COST
            rows.append({"year": t.year, "sym": sym, "f20": f20})
    ev = pd.DataFrame(rows)
    co = ev.groupby("year")["f20"].agg(["mean", "count"])
    print(f"{len(ev)} events across {len(co)} December cohorts\n")
    for y, r in co.iterrows():
        bar = "#" * int(min(abs(r["mean"]) * 200, 40))
        sign = "+" if r["mean"] >= 0 else "-"
        print(f"  Dec {y}: {r['mean']*100:+7.2f}%  n={int(r['count']):>3}  "
              f"{sign}{bar}")
    m = co["mean"]
    npos = int((m > 0).sum())
    t = m.mean() / (m.std(ddof=1) / len(m) ** 0.5)
    print(f"\n  cohorts positive: {npos}/{len(m)} "
          f"({npos/len(m)*100:.0f}%, frozen pass >= 70%)")
    print(f"  across-cohort mean {m.mean()*100:+.2f}%  t = {t:+.2f} "
          f"(frozen pass >= 2.0)")
    ok = npos / len(m) >= 0.70 and t >= 2.0
    print("\n  GATE: " + ("PASS -- C2 is prereg-eligible. Next: write "
                          "TAXLOSS_PREREG.md (frozen genome via regime "
                          "tree month==12 is NOT expressible; this is a "
                          "calendar-windowed strategy needing its own "
                          "runner or a month atom -- decide there), then "
                          "PIT re-test owed before deployment talk."
                          if ok else
                          "FAIL -- concentration, not deadline flow. C2 "
                          "recorded and closed; no prereg."))


if __name__ == "__main__":
    main()
