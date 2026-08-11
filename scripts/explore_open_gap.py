"""
ZERO-TRIAL screen on MINUTE bars: does the opening gap reverse intraday, and
WHEN should you enter?

The daily-bar version could only do open->close. This one sees the first
hour, where the thesis lives: the auction overshoots, then price discovery
corrects it. Nothing logged; no ledger trial spent.

TRAIN SPLIT: first 70% of dates only. The tail stays untouched so a later
gauntlet run is not scored on data this screen already saw.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ENTRIES = [0, 5, 15, 30]        # minutes after 09:30 to enter
EXITS = [30, 60, 120, 390]      # minutes after 09:30 to exit (390 = close)


def sessions(df):
    d = df.copy()
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
        "America/New_York")
    d["t"] = ts
    d["date"] = ts.dt.date
    d["min"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return d[(d["min"] >= 0) & (d["min"] < 390)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.expanduser("~/.primordial_intraday"))
    p.add_argument("--cost-bps", type=float, default=10.0)
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
        day = d.groupby("date").agg(open=("open", "first"),
                                    high=("high", "max"),
                                    low=("low", "min"),
                                    close=("close", "last"))
        pc = day["close"].shift(1)
        tr = pd.concat([day["high"] - day["low"], (day["high"] - pc).abs(),
                        (day["low"] - pc).abs()], axis=1).max(axis=1)
        atr_prev = tr.rolling(14, min_periods=5).mean().shift(1)
        gap = (day["open"] - pc) / atr_prev.where(atr_prev > 0)

        px = d.pivot_table(index="date", columns="min", values="close",
                           aggfunc="last").ffill(axis=1)
        for e in ENTRIES:
            if e not in px.columns:
                continue
            for x in EXITS:
                if x <= e or x not in px.columns:
                    continue
                ret = px[x] / px[e] - 1.0
                rows.append(pd.DataFrame(
                    {"sym": sym, "date": px.index, "gap": gap.values,
                     "entry": e, "exit": x, "ret": ret.values}
                ).dropna(subset=["gap", "ret"]))
    all_ = pd.concat(rows, ignore_index=True)

    dates = np.sort(all_["date"].unique())
    cut = dates[int(len(dates) * a.train_frac)]
    tr_ = all_[all_["date"] < cut]
    print(f"[open_gap] {all_['sym'].nunique()} names | {len(dates):,} "
          f"sessions | TRAIN to {cut} ({tr_['date'].nunique():,} sessions)")
    print(f"  cost assumption: {a.cost_bps:.0f} bps round trip\n")

    for lo, hi in [(-99, -2.0), (-2.0, -1.0), (-1.0, -0.5), (-0.5, 0.5)]:
        b = tr_[(tr_["gap"] > lo) & (tr_["gap"] <= hi)]
        if not len(b):
            continue
        print(f"  --- gap {max(lo,-9):+.1f} to {hi:+.1f} ATR "
              f"({b['date'].nunique():,} sessions) ---")
        if hi <= -1.0:
            yrs = pd.Series([d.year for d in b["date"].unique()]
                            ).value_counts().sort_index()
            print("      by year: " + "  ".join(f"{y}:{n}"
                                                for y, n in yrs.items()))
        print("   entry\\exit" + "".join(f"{x:>12}m" for x in EXITS))
        for e in ENTRIES:
            line = f"   +{e:>3}m     "
            for x in EXITS:
                s = b[(b["entry"] == e) & (b["exit"] == x)]["ret"]
                if len(s) < 30:
                    line += f"{'--':>13}"
                    continue
                line += f"{s.mean()*100 - a.cost_bps/100:>+12.3f}%"
            print(line + "   (net)")
        print()
    print("  Reversal = gap-down buckets POSITIVE net, and the best entry row")
    print("  tells you whether to buy the auction or wait it out.")
    print("  IN-SAMPLE (train dates only). A clean shape earns ONE trial.")


if __name__ == "__main__":
    main()
