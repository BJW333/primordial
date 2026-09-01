"""
ZERO-TRIAL screen on MINUTE bars: unfinished VWAP programs.

Hypothesis (stated before this was written, 2026-09-01): a session that
CLOSES far from its own VWAP on ABNORMAL volume is an institutional order
that was worked all day and not completed. Multi-day VWAP algos resume next
session, so the move CONTINUES for 1-3 days. This is the opposite of the
band/"exhaustion" trade, which reverts to VWAP and is dead on this ledger
(short side PSR 0.008).

Feature (all daily, from the minute tape):
  vdev  = (close - session_vwap) / ATR14[t-1]      signed, ATR units
  rvol  = session_volume / mean(session_volume[t-20..t-1])
Forward return: next-session OPEN -> close of session t+h, h in {1,3,5},
in the DIRECTION of vdev (continuation is positive), net --cost-bps.

TRAIN SPLIT: first 70% of dates only, matching explore_open_gap.py. The
tail is never printed. Nothing is logged; no ledger trial spent. If a
gradient exists, the NEXT step is a pre-registered cell + one look on the
tail -- not a wider grid.

    python3.10 scripts/explore_vwap_program.py                 # pilot cache
    python3.10 scripts/explore_vwap_program.py --cost-bps 14   # midcap RT
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

VDEV_EDGES = [-99, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 99]
RVOL_EDGES = [0, 1.0, 1.5, 2.0, 3.0, 99]
HORIZONS = [1, 3, 5]


def sessions(df):
    d = df.copy()
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
        "America/New_York")
    d["date"] = ts.dt.date
    d["min"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return d[(d["min"] >= 0) & (d["min"] < 390)]


def daily_from_minutes(d):
    d = d.assign(pv=d["close"] * d["volume"])
    day = d.groupby("date").agg(open=("open", "first"), high=("high", "max"),
                                low=("low", "min"), close=("close", "last"),
                                volume=("volume", "sum"), pv=("pv", "sum"),
                                nbars=("close", "size"))
    day = day[(day["nbars"] >= 300) & (day["volume"] > 0)]   # full sessions
    day["vwap"] = day["pv"] / day["volume"]
    pc = day["close"].shift(1)
    tr = pd.concat([day["high"] - day["low"], (day["high"] - pc).abs(),
                    (day["low"] - pc).abs()], axis=1).max(axis=1)
    atr_prev = tr.rolling(14, min_periods=5).mean().shift(1)
    day["vdev"] = (day["close"] - day["vwap"]) / atr_prev.where(atr_prev > 0)
    vprev = day["volume"].rolling(20, min_periods=10).mean().shift(1)
    day["rvol"] = day["volume"] / vprev.where(vprev > 0)
    nxt_open = day["open"].shift(-1)
    for h in HORIZONS:
        day[f"fwd{h}"] = day["close"].shift(-h) / nxt_open - 1.0
    return day


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.expanduser("~/.primordial_intraday"))
    p.add_argument("--cost-bps", type=float, default=14.0,
                   help="round-trip cost; equity_midcap tier is 2 x 7 bps")
    p.add_argument("--train-frac", type=float, default=0.70)
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*_1m.parquet")))
    if not files:
        raise SystemExit(f"no parquet in {a.dir} -- run fetch_intraday.py")

    rows = []
    for f in files:
        sym = os.path.basename(f).split("_")[0]
        d = sessions(pd.read_parquet(f))
        if not len(d):
            continue
        day = daily_from_minutes(d)
        day["sym"] = sym
        rows.append(day.reset_index())
    all_ = pd.concat(rows, ignore_index=True).dropna(subset=["vdev", "rvol"])

    dates = np.sort(all_["date"].unique())
    cut = dates[int(len(dates) * a.train_frac)]
    tr_ = all_[all_["date"] < cut].copy()
    print(f"TRAIN {min(dates)} .. {cut} (tail withheld) | "
          f"{tr_['sym'].nunique()} names | {len(tr_):,} sessions | "
          f"cost {a.cost_bps:.0f} bps RT")

    sgn = np.sign(tr_["vdev"])
    cost = a.cost_bps / 1e4
    for h in HORIZONS:
        tr_[f"cont{h}"] = sgn * tr_[f"fwd{h}"] - cost
    tr_["vb"] = pd.cut(tr_["vdev"], VDEV_EDGES)
    tr_["rb"] = pd.cut(tr_["rvol"], RVOL_EDGES)

    # ---- 1. continuation by vdev bucket (all volume) --------------------
    print("\n[1] signed continuation by close-vs-VWAP distance (ATR units), "
          "net %/trade  [t]  n")
    print("      bucket        " + "  ".join(f"{'h='+str(h):>18s}"
                                             for h in HORIZONS))
    for vb, g in tr_.groupby("vb", observed=True):
        cells = []
        for h in HORIZONS:
            x = g[f"cont{h}"].dropna()
            if len(x) < 30:
                cells.append(f"{'--':>18s}")
                continue
            t = x.mean() / (x.std(ddof=1) / len(x) ** 0.5)
            cells.append(f"{x.mean()*100:+.3f}% [{t:+5.1f}] {len(x):>6,}")
        print(f"  {str(vb):>16s}  " + "  ".join(cells))

    # ---- 2. the actual claim: far from VWAP AND abnormal volume ----------
    print("\n[2] |vdev| >= 1.0 ATR, by relative volume  (the unfinished-"
          "program cell is rvol >= 2)")
    far = tr_[tr_["vdev"].abs() >= 1.0]
    print("      rvol           " + "  ".join(f"{'h='+str(h):>18s}"
                                              for h in HORIZONS))
    for rb, g in far.groupby("rb", observed=True):
        cells = []
        for h in HORIZONS:
            x = g[f"cont{h}"].dropna()
            if len(x) < 30:
                cells.append(f"{'--':>18s}")
                continue
            t = x.mean() / (x.std(ddof=1) / len(x) ** 0.5)
            cells.append(f"{x.mean()*100:+.3f}% [{t:+5.1f}] {len(x):>6,}")
        print(f"  {str(rb):>16s}  " + "  ".join(cells))

    # ---- 3. split by side: the seller story vs the buyer story ----------
    print("\n[3] |vdev| >= 1.0 AND rvol >= 2.0, by side (h=3)")
    cell = far[far["rvol"] >= 2.0]
    for name, g in (("close BELOW vwap (sell program -> short)",
                     cell[cell["vdev"] < 0]),
                    ("close ABOVE vwap (buy program -> long)",
                     cell[cell["vdev"] > 0])):
        x = g["cont3"].dropna()
        if len(x) < 30:
            print(f"  {name}: n={len(x)} -- unscorable")
            continue
        t = x.mean() / (x.std(ddof=1) / len(x) ** 0.5)
        yrs = g.assign(y=[d.year for d in g["date"]]).groupby("y")["cont3"]
        by_yr = "  ".join(f"{y}:{v.mean()*100:+.2f}" for y, v in yrs)
        print(f"  {name}: {x.mean()*100:+.3f}%/trade [t {t:+.1f}] n={len(x):,}"
              f" win {(x>0).mean()*100:.0f}%\n      by year: {by_yr}")

    print("\nREAD: continuation = positive in [1] growing with |vdev| AND "
          "larger in the high-rvol rows of [2]. Reversal = negative. A flat "
          "or reverting table closes the seam for free. A gradient earns "
          "ONE pre-registered cell on the tail, nothing else.")


if __name__ == "__main__":
    main()
