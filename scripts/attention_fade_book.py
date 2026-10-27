"""
attention_fade_book.py -- turn the confirmed attention FADE into a real
portfolio, and test one new kind of "bottom": waiting a full year.

EVENTS: same as attention_rebound.py -- good-news Wikipedia spikes (daily
views >= 3x 90d median AND stock up > 2% that day) on non-S&P liquid stocks,
SMALL / MID by dollar volume. Enter at the close AFTER the spike day.

PART A  HEDGED SHORT BOOK (the trade)
  Every day, short (equal weight) every stock in its holding window after a
  good-news spike, and BUY the equal-weight same-size benchmark for the same
  dollars (so the bet is only "these stocks lag their peers", not the market).
  Holding windows: 63 and 252 trading days.
  Costs, charged daily:
    trading 0.20% per side on the short leg -> 2 x 0.20% x (252/H) a year
    borrow fee: MID 3%/yr, SMALL 6%/yr (conservative: hyped names are
    often harder to borrow)
  Daily stock returns capped at -90% / +300% (we do NOT hide squeezes).
  PRE-STATED: MID 252d book PASSES if its net return after costs is > 0 in
  2017-01 -> 2021-05 AND in 2021-06 -> 2026-05, with t > 2.0 on the last 5y
  (t = monthly returns, 3-lag Newey-West). The other three books are shown
  for information only.

PART B  YEAR-TWO REBOUND (a "bottom" that is just time)
  The fade lasts a full year. Does the stock bounce back in YEAR TWO?
  Return from day 252 to day 504 after the spike, minus the benchmark.
  PRE-STATED (per universe): PASS if > 0 with t > 2.5 on 2017-2021-05
  spikes AND > 0 on 2021-06 -> 2024-10 spikes (the newest spikes that
  already have two years of data).

Run (repo root):
    python3.10 scripts/attention_fade_book.py --smoke
    python3.10 scripts/attention_fade_book.py | tee logs/attention_fade_book.txt
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
import blowup_scan as B               # noqa: E402
import factor_portfolio as FP         # noqa: E402

RECENT, END = AR.RECENT, pd.Timestamp("2026-06-01")
BORROW = {"SMALL": 0.06, "MID": 0.03}
SIDE = 0.002
HOLDS = (63, 252)


def book(ev, px, bench, H, borrow):
    idx = px.index
    r = px.pct_change(fill_method=None).clip(-0.9, 3.0)
    rb = bench.reindex(idx).ffill().pct_change().fillna(0)
    held = np.zeros(px.shape, bool)
    cols = {c: j for j, c in enumerate(px.columns)}
    for s, d in zip(ev["sym"], ev["d"]):
        p0 = idx.get_loc(d)
        a, b = p0 + 2, min(p0 + 2 + H, len(idx))
        if a < len(idx):
            held[a:b, cols[s]] = True
    rv = np.where(held, np.nan_to_num(r.values), 0.0)
    n = held.sum(axis=1)
    longs = np.where(n > 0, rv.sum(axis=1) / np.maximum(n, 1), 0.0)
    gross = np.where(n > 0, -(longs - rb.values), 0.0)
    cost = np.where(n > 0, (2 * SIDE * 252 / H + borrow) / 252, 0.0)
    out = pd.DataFrame({"gross": gross, "net": gross - cost, "n": n}, idx)
    return out[out.index >= AR.START]


def stats(x):
    x = x.dropna()
    if len(x) < 40:
        return dict(ann=np.nan, vol=np.nan, sh=np.nan, dd=np.nan, t=np.nan)
    ann = (1 + x).prod() ** (252 / len(x)) - 1
    vol = x.std() * np.sqrt(252)
    eq = (1 + x).cumprod()
    mo = (1 + x).groupby(x.index.to_period("M")).prod() - 1
    return dict(ann=ann, vol=vol, sh=x.mean() * 252 / vol if vol else np.nan,
                dd=(eq / eq.cummax() - 1).min(), t=B.nw_t(mo, 3))


def part_a(name, ev, px, bench):
    print(f"\n  ===== PART A {name}: hedged short book "
          f"(borrow {BORROW[name]*100:.0f}%/yr, trading {SIDE*100:.1f}%/side)"
          f" =====")
    res = {}
    for H in HOLDS:
        bk = book(ev, px, bench, H, BORROW[name])
        res[H] = bk
        print(f"  hold {H}d: avg {bk['n'][bk['n'] > 0].mean():.0f} shorts "
              f"held, {(bk['n'] > 0).mean()*100:.0f}% of days invested")
        for lab, lo, hi in (("2017-21", AR.START, RECENT),
                            ("last 5y", RECENT, END)):
            for k in ("gross", "net"):
                s = stats(bk.loc[(bk.index >= lo) & (bk.index < hi), k])
                print(f"    {lab:<8} {k:<5} ann {s['ann']*100:>+6.1f}%  vol "
                      f"{s['vol']*100:>4.0f}%  sharpe {s['sh']:>+5.2f}  "
                      f"worst {s['dd']*100:>4.0f}%  t {s['t']:>+5.2f}")
    yr = pd.DataFrame({f"{H}d net": res[H]["net"] for H in HOLDS})
    yr = (1 + yr).groupby(yr.index.year).prod() - 1
    print("  by year (net): " + " | ".join(
        f"{y} " + "/".join(f"{v*100:+.0f}%" for v in row)
        for y, row in yr.iterrows()) + f"   ({'/'.join(yr.columns)})")
    return res


def part_b(name, ev, px, bench):
    idx = px.index
    pxf = px.ffill()
    b = bench.reindex(idx).ffill()
    rows = []
    for s, d in zip(ev["sym"], ev["d"]):
        p = idx.get_loc(d) + 1
        if p + 504 >= len(idx):
            continue
        c0, c1 = pxf[s].iloc[p + 252], pxf[s].iloc[p + 504]
        if not (np.isfinite(c0) and np.isfinite(c1)) or c0 <= 0:
            continue
        # a stock that stopped trading in year 2 counts at its last price
        r = c1 / c0 - b.iloc[p + 504] / b.iloc[p + 252]
        if abs(r) < 3:
            rows.append((d, r))
    tr = pd.DataFrame(rows, columns=["d", "r"])
    print(f"\n  ===== PART B {name}: year-two rebound (day 252 -> 504) =====")
    out = {}
    for lab, g in (("2017-21", tr[tr.d < RECENT]),
                   ("2021-06..", tr[tr.d >= RECENT])):
        t = AR.tstat(g)
        m = g["r"].mean() if len(g) else np.nan
        out[lab] = (m, t)
        print(f"    {lab:<10} n {len(g):>5}  mean {m*100:>+6.2f}%  median "
              f"{(g['r'].median() if len(g) else np.nan)*100:>+6.2f}%  win "
              f"{((g['r'] > 0).mean() if len(g) else np.nan)*100:>3.0f}%  "
              f"t {t:>+5.2f}")
    (m0, t0), (m1, _) = out["2017-21"], out["2021-06.."]
    return bool(m0 > 0 and t0 > 2.5 and m1 > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, views, unis = AR.smoke()
        BORROW.setdefault("SMALL", 0.06)
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = AR.universes(sp_members)
    verdict = {}
    for name, (mem, bench) in unis.items():
        ev = AR.good_spikes(views, px, mem)
        res = part_a(name, ev, px, bench)
        if name == "MID":
            bk = res[252]["net"]
            s0 = stats(bk[(bk.index >= AR.START) & (bk.index < RECENT)])
            s1 = stats(bk[(bk.index >= RECENT) & (bk.index < END)])
            verdict["A  MID 252d short book, net of costs"] = bool(
                s0["ann"] > 0 and s1["ann"] > 0 and s1["t"] > 2.0)
        verdict[f"B  year-two rebound {name}"] = part_b(name, ev, px, bench)
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    for k, v in verdict.items():
        print(f"  {k:<40} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
