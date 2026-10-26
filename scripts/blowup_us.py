"""
blowup_us.py -- the blow-up scan (blowup_scan.py, same signals, same
pre-stated tests) on EVERY liquid US stock, delisted ones included.
Needs data/us_universe/ from build_us_universe.py.

Universe at each month-end (point-in-time, nothing from the future):
  price >= $5, 63-day average dollar volume >= $5M, >= 520 trading days
  of history (signals need two years), common stock on NYSE/NASDAQ/AMEX.

Signals: ACCUM, SQUEEZE, LAGGARD (+ MOM context) -- unchanged.
PASS (pre-stated, last five years): blow-up hit-rate lift >= 1.5x with
t > 2.4 AND top-20 portfolio beats SPY at the same risk.

Run (repo root):
    python3.10 scripts/blowup_us.py --smoke
    python3.10 scripts/blowup_us.py | tee logs/blowup_us.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_scan as B        # noqa: E402
import factor_portfolio as FP  # noqa: E402
import turbo_portfolio as T    # noqa: E402

ROOT = os.path.dirname(HERE)
DIR = os.path.join(ROOT, "data", "us_universe")
OUT = os.path.join(ROOT, "data", "blowup_us_returns.csv")
MIN_PX, MIN_DV = 5.0, 5e6


def liquid_members(px, vol, min_dv):
    dv = (px * vol).rolling(63, min_periods=40).mean()
    me = px.resample("ME").last().index
    days = [px.index[px.index <= d][-1] for d in me if (px.index <= d).any()]
    ok = (px.loc[days] >= MIN_PX) & (dv.loc[days] >= min_dv)
    sets = {d: set(ok.columns[ok.loc[d].values]) for d in days}
    ever = ok.any()
    return sets, list(ever[ever].index)


def load():
    px = pd.read_parquet(os.path.join(DIR, "close.parquet"))
    vol = pd.read_parquet(os.path.join(DIR, "volume.parquet"))
    spy = pd.read_parquet(os.path.join(DIR, "spy.parquet"))["SPY"]
    feed = open(os.path.join(DIR, "feed.txt")).read().strip() \
        if os.path.exists(os.path.join(DIR, "feed.txt")) else "sip"
    return px, vol, spy, feed


def diagnose(px, res):
    """Worst/best single picks and impossible daily moves (data check)."""
    log = pd.DataFrame(B.PICKLOG, columns=["month", "signal", "sym", "ret"])
    if not len(log):
        return
    if os.path.isdir(os.path.join(ROOT, "data")):
        log.to_csv(os.path.join(ROOT, "data", "blowup_us_picks.csv"),
                   index=False)
    r = px.pct_change(fill_method=None)
    big = (r.abs() > 0.75)
    n_big = int(big.sum().sum())
    print(f"\n  DATA CHECK: {n_big} stock-days with a >75% one-day move "
          f"in {int(big.any().sum())} stocks")
    for sig in B.SIGS + ["MOM"]:
        g = log[log.signal == sig]
        if not len(g):
            continue
        rec = g[g.month >= B.RECENT]
        print(f"\n  {sig}: {len(rec)} picks last 5y | median pick month "
              f"{rec.ret.median()*100:+.1f}% | picks losing >50% in a month:"
              f" {(rec.ret < -0.5).sum()} | gaining >100%: "
              f"{(rec.ret > 1).sum()}")
        w = rec.nsmallest(6, "ret")
        print("    worst: " + ", ".join(f"{s} {m:%Y-%m} {x*100:+.0f}%"
                                         for s, m, x in zip(w.sym, w.month,
                                                            w.ret)))
        b = rec.nlargest(6, "ret")
        print("    best:  " + ", ".join(f"{s} {m:%Y-%m} {x*100:+.0f}%"
                                         for s, m, x in zip(b.sym, b.month,
                                                            b.ret)))
        yr = res.loc[res.index >= B.RECENT, sig]
        print("    by year: " + "  ".join(
            f"{y}:{((1 + v).prod() - 1)*100:+.0f}%"
            for y, v in yr.groupby(yr.index.year)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy, _ = T.smoke()
        px = px * 3                         # smoke prices clear the $5 bar
        vol = vol * 100
        feed = "sip"
    else:
        if not os.path.exists(os.path.join(DIR, "close.parquet")):
            sys.exit("run build_us_universe.py first")
        px, vol, spy, feed = load()
    min_dv = MIN_DV if feed == "sip" else MIN_DV * 0.03
    if feed != "sip":
        print("  NOTE: IEX feed -> volume is IEX-only; dollar-volume bar "
              f"scaled to ${min_dv/1e6:.2f}M")
    sets, ever = liquid_members(px, vol, min_dv)
    px, vol = px[ever], vol[ever]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731
    sizes = pd.Series({d: len(s) for d, s in sets.items()})
    print(f"  {len(ever)} stock histories ever liquid; liquid names at "
          "year-end: " + "  ".join(
              f"{d.year}:{n}" for d, n in sizes[sizes.index.month == 12]
              .items()))
    if sizes.max() < 150:
        sys.exit("  universe too small (<150 liquid names) -- did the full "
                 "build_us_universe.py run finish?")
    res, hit = B.run(px, vol, spy, members)
    if not a.smoke:
        res.to_csv(OUT)
    diagnose(px, res)
    old = res.index < B.RECENT
    if old.sum() > 6:
        B.report(res[old], hit[hit.index < B.RECENT], "CONTEXT 2018-2021")
    rec = B.report(res[~old], hit[hit.index >= B.RECENT],
                   "DECIDING: LAST FIVE YEARS (all liquid US stocks)")
    print("\n  ===== VERDICT (pre-stated: lift >= 1.5x, t > 2.4, beats "
          "same-risk SPY) =====")
    for name in B.SIGS + ["MOM"]:
        lift, t, ann, m_a = rec[name]
        ok = lift >= 1.5 and t > 2.4 and ann > m_a
        tag = " (context)" if name == "MOM" else ""
        print(f"  {name:<8} lift {lift:.2f}x  t {t:+.2f}  return "
              f"{ann*100:+.1f}% vs same-risk SPY {m_a*100:+.1f}% -> "
              f"{'PASS' if ok else 'fail'}{tag}")


if __name__ == "__main__":
    main()
