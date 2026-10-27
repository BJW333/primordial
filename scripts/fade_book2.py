"""
fade_book2.py -- REALITY CHECK on the fade_predict.py PASS. The rule is
fixed exactly as it passed (no changes, no selection):
  liquid MID caps (top half of MID dollar volume at the spike), good-news
  attention spike, WEAK_TREND on day 21 (12-month return below the
  same-size average) -> short at the close of day 22, cover at day 252,
  hedge with the same dollars of the equal-weight MID benchmark.

What changes vs fade_predict.py (which averaged trades):
  1. NO trades dropped -- squeezes (stock up 3x+ vs peers) stay in.
  2. A real book: each short gets equal dollars AT ENTRY and is held
     without rebalancing; daily P&L of all open shorts / number open.
  3. Costs: 0.2% per side, borrow 0.5%/yr (liquid names) -- and a
     stress line at 3%/yr.
PRE-STATED: PASS if the book's net return (0.5% borrow) is > 0 in
2017-01 -> 2021-05 AND 2021-06 -> 2026-05, with t > 2.0 on the last five
years (monthly returns, 3-lag Newey-West).

Run (repo root):
    python3.10 scripts/fade_book2.py | tee logs/fade_book2.txt
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

SIDE = 0.002
END = pd.Timestamp("2026-06-01")


def book(tr, px, bench, borrow):
    idx = px.index
    cv = px.ffill().values
    cols = {c: j for j, c in enumerate(px.columns)}
    b = bench.reindex(idx).ffill().values
    pnl = np.zeros(len(idx))
    n = np.zeros(len(idx))
    for s, e, x in zip(tr["sym"], tr["e"], tr["x"]):
        c = cv[:, cols[s]]
        seg = c[e:x + 1]
        bs = b[e:x + 1]
        # short stock, long benchmark, both sized 1 at entry, no rebalance
        daily = -(np.diff(seg) / seg[0]) + np.diff(bs) / bs[0]
        daily = np.nan_to_num(daily)
        daily[0] -= SIDE
        daily[-1] -= SIDE
        pnl[e + 1:x + 1] += daily - borrow / 252
        n[e + 1:x + 1] += 1
    r = pd.Series(np.where(n > 0, pnl / np.maximum(n, 1), 0.0), idx)
    return r, pd.Series(n, idx)


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
    print(f"\n  {len(tr)} trades (liquid {name}, WEAK_TREND), NO trades "
          f"dropped")
    for lab, g in (("2017-21", tr[tr["d"] < AR.RECENT]),
                   ("2021-06+", tr[tr["d"] >= AR.RECENT])):
        print(f"    {lab:<9} per-trade fade {g['r'].mean()*100:+.2f}% "
              f"(median {g['r'].median()*100:+.2f}%), short wins "
              f"{(g['r'] < 0).mean()*100:.0f}%, worst 3 trades (stock vs "
              f"peers): " + ", ".join(f"{v*100:+.0f}%" for v in
                                      g['r'].nlargest(3)))
    verdict = None
    for borrow in (0.005, 0.03):
        r, n = book(tr, px, bench, borrow)
        r = r[r.index >= AR.START]
        print(f"\n  BOOK, borrow {borrow*100:.1f}%/yr: avg "
              f"{n[n > 0].mean():.0f} shorts open")
        ss = []
        for lab, lo, hi in (("2017-21", AR.START, AR.RECENT),
                            ("last 5y", AR.RECENT, END)):
            s = FB.stats(r[(r.index >= lo) & (r.index < hi)])
            ss.append(s)
            print(f"    {lab:<8} ann {s['ann']*100:>+6.1f}%  vol "
                  f"{s['vol']*100:>4.0f}%  sharpe {s['sh']:>+5.2f}  worst "
                  f"{s['dd']*100:>4.0f}%  t {s['t']:>+5.2f}")
        yr = (1 + r).groupby(r.index.year).prod() - 1
        print("    by year: " + " | ".join(f"{y} {v*100:+.0f}%"
                                         for y, v in yr.items()))
        if borrow == 0.005:
            verdict = bool(ss[0]["ann"] > 0 and ss[1]["ann"] > 0
                           and ss[1]["t"] > 2.0)
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    print(f"  liquid {name} WEAK_TREND short book, real-world  "
          f"{'PASS' if verdict else 'FAIL'}")


if __name__ == "__main__":
    main()
