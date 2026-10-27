"""
attention_bottom.py -- Blake's bottom-catcher: after a good-news attention
spike the stock fades (attention_rebound.py). Can a MOVING-AVERAGE
RECLAIM + OBV + ADX tell us when the slide is over, so we buy the bottom?

Same events, universes, entry/exit and benchmark as attention_rebound.py
(good-news Wikipedia spikes on non-S&P small/mid caps; enter at the close
AFTER the signal day; hold 63 trading days; return minus the equal-weight
same-size benchmark).

EVERY signal requires the stock to have DIPPED first: it closed >= 10%
below the spike-day close at some point after the spike (that's what a
"bottom" is). Checked daily from day 5 to day 252 after the spike.
  M20   RECLAIM_20   close back above its 20-day moving average
  M50   RECLAIM_50   close back above its 50-day moving average
  O     OBV_TURN     on-balance volume above its 20-day average AND higher
                     than 20 days ago (money flowing back in)
  XU    ADX_UPTREND  ADX(14) >= 20 with +DI > -DI (a new up-trend)
  XE    ADX_EXHAUST  a strong down-trend (ADX >= 25, -DI > +DI) in the last
                     20 days has flipped: now +DI > -DI
  Combos: all 31 combinations of the five.
  NOTE: small/mid caps only have closing prices in our data, so ADX uses
  high = max(today, yesterday close), low = min(...). Approximation.

PRE-STATED (same rule as attention_rebound.py)
  Per universe: pick the combo with the highest t among those with >= 30
  trades on 2017-2021-05 spikes ONLY; it PASSES if on 2021-06 -> 2026-05
  spikes its 63-day return vs the benchmark is > 0 with t > 2.5 and >= 30
  trades. t = monthly averages of trades, 3-lag Newey-West.
  Also printed (diagnostic, not judged): median trade, and the share of
  trades that beat the benchmark -- to check t vs mean sign mismatches.

Run (repo root):
    python3.10 scripts/attention_bottom.py --smoke
    python3.10 scripts/attention_bottom.py | tee logs/attention_bottom.txt
"""
import argparse
import itertools
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
import factor_portfolio as FP         # noqa: E402

SIGS = ["M20", "M50", "O", "XU", "XE"]
NAMES = {"M20": "RECLAIM_20", "M50": "RECLAIM_50", "O": "OBV_TURN",
         "XU": "ADX_UPTREND", "XE": "ADX_EXHAUST"}
COMBOS = [c for k in range(1, 6) for c in itertools.combinations(SIGS, k)]
MINN = AR.MINN


def wilder(x, n=14):
    return x.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def indicators(px, vol):
    c = px.ffill()
    prev = c.shift()
    H = np.maximum(c, prev)
    L = np.minimum(c, prev)
    up, dn = H.diff(), -L.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    ndm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = (H - L).abs()
    atr = wilder(tr).replace(0, np.nan)
    pdi = 100 * wilder(pdm) / atr
    ndi = 100 * wilder(ndm) / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx = wilder(dx)
    strong_down = ((adx >= 25) & (ndi > pdi)).astype(float)
    recent_down = strong_down.rolling(20, min_periods=1).max() > 0
    v = vol.reindex_like(c).fillna(0)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    ind = {
        "M20": c > c.rolling(20, min_periods=15).mean(),
        "M50": c > c.rolling(50, min_periods=40).mean(),
        "O": (obv > obv.rolling(20, min_periods=15).mean()) &
             (obv > obv.shift(20)),
        "XU": (adx >= 20) & (pdi > ndi),
        "XE": recent_down & (pdi > ndi),
    }
    return c, ind


def add_signals(evs, c, ind):
    cols = {s: i for i, s in enumerate(c.columns)}
    arr = {k: v.values for k, v in ind.items()}
    cv = c.values
    for e in evs:
        j = cols[e["sym"]]
        p0, w = e["p0"], e["w"]
        hi = w[-1] + 1 if len(w) else p0
        lows = np.minimum.accumulate(cv[p0:hi, j])
        dipped = (lows <= 0.9 * cv[p0, j])[w - p0]
        e["sig"] = {k: np.nan_to_num(arr[k][w, j].astype(float), nan=0)
                    .astype(bool) & dipped for k in SIGS}
    return evs


def describe(tr):
    if not len(tr):
        return np.nan, np.nan, np.nan
    return tr["r"].mean(), tr["r"].median(), (tr["r"] > 0).mean()


def run_universe(name, evs):
    old = [e for e in evs if e["d"] < AR.RECENT]
    new = [e for e in evs if e["d"] >= AR.RECENT]
    print(f"\n  ===== {name}: {len(evs)} good-news spikes "
          f"({len(old)} in 2017-21, {len(new)} last 5y) =====")
    print(f"\n  {'combo':<16} | {'2017-21: n':>10} {'mean':>7} {'med':>7} "
          f"{'win%':>5} {'t':>6} | {'last5y: n':>9} {'mean':>7} {'med':>7} "
          f"{'win%':>5} {'t':>6}")
    sel = []
    for combo in COMBOS:
        a, b = AR.trades(old, combo), AR.trades(new, combo)
        ta, tb = AR.tstat(a), AR.tstat(b)
        ma, mda, wa = describe(a)
        mb, mdb, wb = describe(b)
        lab = "+".join(combo)
        sel.append((lab, len(a), ta, len(b), mb, tb))
        if len(a) >= 10 or len(b) >= 10:
            print(f"  {lab:<16} | {len(a):>10} {ma*100:>+6.2f}% "
                  f"{mda*100:>+6.2f}% {wa*100:>4.0f}% {ta:>+6.2f} | "
                  f"{len(b):>9} {mb*100:>+6.2f}% {mdb*100:>+6.2f}% "
                  f"{wb*100:>4.0f}% {tb:>+6.2f}")
    ok = [x for x in sel if x[1] >= MINN and np.isfinite(x[2])]
    if not ok:
        print("  no combo with >= 30 trades in 2017-21")
        return None, False
    lab, n0, t0, n1, m1, t1 = max(ok, key=lambda x: x[2])
    passed = bool(n1 >= MINN and m1 > 0 and t1 > 2.5)
    print(f"\n  SELECTED on 2017-21 (highest t, >= 30 trades): {lab} "
          f"({' + '.join(NAMES[k] for k in lab.split('+'))}), t {t0:+.2f}")
    print(f"  judged on last 5y: n {n1}, mean {m1*100:+.2f}% vs benchmark, "
          f"t {t1:+.2f} -> {'PASS' if passed else 'FAIL'}")
    return lab, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, views, unis = AR.smoke()
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = AR.universes(sp_members)
    c, ind = indicators(px, vol)
    verdict = {}
    for name, (mem, bench) in unis.items():
        ev = AR.good_spikes(views, px, mem)
        evs = AR.event_arrays(ev, px, vol, views, bench, {})
        evs = add_signals(evs, c, ind)
        lab, passed = run_universe(name, evs)
        verdict[f"BOTTOM {name} ({lab})"] = passed
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    for k, v in verdict.items():
        print(f"  {k:<40} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
