"""
fade_stop.py -- fade_book2.py showed the liquid-MID WEAK_TREND short wins
65% of the time (median -18% / -23% vs peers) but a few squeezes (+1,198%,
+639%...) wipe it out. Fix the squeeze problem with a STOP-LOSS on each
short: cover (and drop its hedge) at the close the stock first closes
>= X% ABOVE our entry price. Same trades, same book, same costs as
fade_book2.py (0.2%/side, 0.5%/yr borrow; 3% stress shown).

Stops: 25%, 50% (PRIMARY, fixed before running), 100%.
NOTE: this idea came from seeing fade_book2.py's squeezes, so a pass here
is NOT a clean out-of-sample pass -- it means "freeze it and paper-trade
it forward". Bar is still set before running:
PASS (50% stop, 0.5% borrow): net > 0 in BOTH periods, last-5y t > 2.0.

Run (repo root):
    python3.10 scripts/fade_stop.py | tee logs/fade_stop.txt
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
import fade_book2 as F2               # noqa: E402
import fade_predict as P              # noqa: E402

STOPS = (0.25, 0.50, 1.00)
PRIMARY = 0.50


def book(tr, px, bench, borrow, stop):
    idx = px.index
    cv = px.ffill().values
    cols = {c: j for j, c in enumerate(px.columns)}
    b = bench.reindex(idx).ffill().values
    pnl = np.zeros(len(idx))
    n = np.zeros(len(idx))
    per = []
    for s, e, x in zip(tr["sym"], tr["e"], tr["x"]):
        c = cv[:, cols[s]]
        hit = np.flatnonzero(c[e + 1:x + 1] >= c[e] * (1 + stop))
        xx = e + 1 + hit[0] if len(hit) else x
        seg, bs = c[e:xx + 1], b[e:xx + 1]
        daily = np.nan_to_num(-(np.diff(seg) / seg[0]) +
                              np.diff(bs) / bs[0])
        daily[0] -= F2.SIDE
        daily[-1] -= F2.SIDE
        pnl[e + 1:xx + 1] += daily - borrow / 252
        n[e + 1:xx + 1] += 1
        per.append((len(hit) > 0, daily.sum()))
    r = pd.Series(np.where(n > 0, pnl / np.maximum(n, 1), 0.0), idx)
    return r, pd.Series(n, idx), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    P.CLIP = False
    if a.smoke:
        px, vol, views, unis = AR.smoke()
        name = "SMALL"
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = AR.universes(sp_members)
        name = "MID"
    mem, bench = unis[name]
    dv = (px * vol).rolling(63, min_periods=40).mean()
    ev = AR.good_spikes(views, px, mem)
    df = P.build(name, ev, px, bench, {}, {}, mem, dv)
    tr = df[df["liq"] & df["W"]].sort_values("d")
    print(f"\n  {len(tr)} trades (liquid {name}, WEAK_TREND)")
    verdict = None
    for stop in STOPS:
        for borrow in (0.005, 0.03):
            r, n, per = book(tr, px, bench, borrow, stop)
            r = r[r.index >= AR.START]
            stopped = np.mean([p[0] for p in per]) * 100
            worst = min(p[1] for p in per) * 100
            ss = []
            line = []
            for lab, lo, hi in (("2017-21", AR.START, AR.RECENT),
                                ("last 5y", AR.RECENT, F2.END)):
                s = FB.stats(r[(r.index >= lo) & (r.index < hi)])
                ss.append(s)
                line.append(f"{lab} {s['ann']*100:+5.1f}% sh {s['sh']:+.2f} "
                            f"dd {s['dd']*100:.0f}% t {s['t']:+.2f}")
            tag = " <- PRIMARY" if stop == PRIMARY and borrow == 0.005 else ""
            print(f"  stop +{stop*100:>3.0f}% borrow {borrow*100:.1f}% | "
                  f"stopped {stopped:>3.0f}% of trades, worst trade "
                  f"{worst:+.0f}% | " + " | ".join(line) + tag)
            if stop == PRIMARY and borrow == 0.005:
                verdict = bool(ss[0]["ann"] > 0 and ss[1]["ann"] > 0
                               and ss[1]["t"] > 2.0)
                yr = (1 + r).groupby(r.index.year).prod() - 1
                prim_years = " | ".join(f"{y} {v*100:+.0f}%"
                                        for y, v in yr.items())
    print(f"\n  primary by year: {prim_years}")
    print("\n  ===== VERDICT (bar set before running; idea is post-hoc) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    print(f"  50% stop short book  {'PASS -> paper-trade forward' if verdict else 'FAIL'}")


if __name__ == "__main__":
    main()
