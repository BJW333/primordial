"""
fade_stop_check.py -- PROVE OR BREAK the fade's 50% stop-loss, which so far
was only seen on MID caps after the squeezes were already known.

Rules are FROZEN exactly as fade_stop.py (no tuning):
  good-news attention spike -> day 21: WEAK_TREND (12m return below the
  same-size average) and in the more-liquid half of its size group ->
  short at the close of day 22, hedge with the same dollars of the equal-
  weight same-size benchmark, cover at day 252 or at the stop.

CHECKS
  1 NEW STOCKS: SMALL caps (the stop rule never saw them). Borrow 3%/yr;
    stress 6%/yr. Trading 0.2% per side.
  2 DELAYED STOP: cover at the close the day AFTER the stock first closes
    >= 50% above entry (not the same close) -- realistic execution, catches
    overnight jumps.
  3 CONTROL: same book with NO stop, to see whether the stop is what helps.
  Also printed for MID with the delayed stop (fair re-check of the original).

PRE-STATED (SMALL caps, 50% stop, DELAYED, 3% borrow):
  PASS if net return > 0 in BOTH 2017-01 -> 2021-05 and 2021-06 -> 2026-05,
  last-5y t > 2.0, AND it beats the no-stop book in BOTH periods.

Run (repo root):
    python3.10 scripts/fade_stop_check.py | tee logs/fade_stop_check.txt
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
import attention_fade_book as FB      # noqa: E402
import attention_rebound as AR        # noqa: E402
import attention_short_test as AS     # noqa: E402
import factor_portfolio as FP         # noqa: E402
import fade_predict as P              # noqa: E402

SIDE, END = 0.002, pd.Timestamp("2026-06-01")


def book(tr, px, bench, borrow, stop, delay):
    idx = px.index
    cv = px.ffill().values
    cols = {c: j for j, c in enumerate(px.columns)}
    b = bench.reindex(idx).ffill().values
    pnl, n = np.zeros(len(idx)), np.zeros(len(idx))
    stopped = 0
    for s, e, x in zip(tr["sym"], tr["e"], tr["x"]):
        c = cv[:, cols[s]]
        xx = x
        if stop is not None:
            hit = np.flatnonzero(c[e + 1:x + 1] >= c[e] * (1 + stop))
            if len(hit):
                xx = min(e + 1 + hit[0] + delay, len(idx) - 1)
                stopped += 1
        seg, bs = c[e:xx + 1], b[e:xx + 1]
        if len(seg) < 2:
            continue
        daily = np.nan_to_num(-(np.diff(seg) / seg[0]) + np.diff(bs) / bs[0])
        daily[0] -= SIDE
        daily[-1] -= SIDE
        pnl[e + 1:xx + 1] += daily - borrow / 252
        n[e + 1:xx + 1] += 1
    r = pd.Series(np.where(n > 0, pnl / np.maximum(n, 1), 0.0), idx)
    return r[(r.index >= AR.START) & (r.index < END)], stopped / max(len(tr), 1)


def line(lab, r, st):
    s0 = FB.stats(r[r.index < AR.RECENT])
    s1 = FB.stats(r[r.index >= AR.RECENT])
    print(f"  {lab:<36} stopped {st*100:>3.0f}% | 2017-21 {s0['ann']*100:>+6.1f}%"
          f" sh {s0['sh']:>+5.2f} dd {s0['dd']*100:>4.0f}% | last5y "
          f"{s1['ann']*100:>+6.1f}% sh {s1['sh']:>+5.2f} dd {s1['dd']*100:>4.0f}%"
          f" t {s1['t']:>+5.2f}")
    return s0, s1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    P.CLIP = False
    if a.smoke:
        px, vol, views, unis = AR.smoke()
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = AR.universes(sp_members)
    dv = (px * vol).rolling(63, min_periods=40).mean()
    out = {}
    for name in [u for u in ("SMALL", "MID") if u in unis]:
        mem, bench = unis[name]
        ev = AR.good_spikes(views, px, mem)
        df = P.build(name, ev, px, bench, {}, {}, mem, dv)
        tr = df[df["liq"] & df["W"]].sort_values("d")
        print(f"\n  ===== {name}: {len(tr)} trades (liquid half, WEAK_TREND) "
              f"=====")
        if name == "SMALL":
            for borrow in (0.03, 0.06):
                print(f"  -- borrow {borrow*100:.0f}%/yr --")
                rn, sn = book(tr, px, bench, borrow, None, 0)
                n0, n1 = line("no stop (control)", rn, sn)
                rs, ss = book(tr, px, bench, borrow, 0.50, 0)
                line("50% stop, same close", rs, ss)
                rd, sd = book(tr, px, bench, borrow, 0.50, 1)
                d0, d1 = line("50% stop, DELAYED 1 day", rd, sd)
                if borrow == 0.03:
                    out["SMALL"] = (n0, n1, d0, d1)
        else:
            rn, sn = book(tr, px, bench, 0.005, None, 0)
            line("no stop (control), 0.5% borrow", rn, sn)
            rd, sd = book(tr, px, bench, 0.005, 0.50, 1)
            line("50% stop, DELAYED, 0.5% borrow", rd, sd)
            rd3, sd3 = book(tr, px, bench, 0.03, 0.50, 1)
            line("50% stop, DELAYED, 3% borrow", rd3, sd3)
    print("\n  ===== VERDICT (pre-stated: SMALL, 50% stop, delayed, 3% borrow) "
          "=====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    if "SMALL" in out:
        n0, n1, d0, d1 = out["SMALL"]
        ok = bool(d0["ann"] > 0 and d1["ann"] > 0 and d1["t"] > 2.0
                  and d0["ann"] > n0["ann"] and d1["ann"] > n1["ann"])
        print(f"  fade stop-loss holds on new stocks  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
