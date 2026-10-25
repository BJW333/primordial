"""
dipz_screen.py -- vol-scaled dip-buy screen (successor to dip5_screen.py).

Trigger: today's drop measured in units of the stock's OWN typical daily
range, not a fixed %:
    z_t = -(close_t / close_{t-1} - 1) / atr_pct_{t-1}
    atr_pct = 14-day ATR / close, LAGGED one bar (today's crash can't
    inflate its own yardstick)
    event: z_t >= k        -> buy next open, hold H bars, exit at close

GROUND AND WHY (see prereg/DIPZ_PREREG.md):
  The S&P 500 2010-26 sample is spent -- the fixed-5% version was looked at
  there, so it can't judge a variant. Primary ground is the S&P 400, which
  only exists here as CURRENT membership. Survivorship flatters dip-buying,
  so this ground can only KILL the idea, never pass it:
    survivor-mode fail  -> CLOSED (restoring dead names only makes it worse)
    survivor-mode pass  -> NOT a pass; owes a point-in-time sp400 re-test
                           (Norgate) before anything else
  --ground sp500pit runs the PIT S&P 500 version as CONTEXT ONLY (leaked).

PRIMARY CELL (one, frozen): k = 3, H = 10, next-open entry, 20 bps RT.
  passes if net excess-vs-SPY > 0 with date-clustered t > 2 in BOTH halves
  (pre/post 2018). The k x H grid is printed as context and decides nothing.

Run (repo root; cache shared with the us_sp400 manifest / breakout runs):
    python3.10 scripts/dipz_screen.py                  # the one look
    python3.10 scripts/dipz_screen.py --ground sp500pit   # context only
    python3.10 scripts/dipz_screen.py --smoke
"""
import argparse
import csv
import os
import sys
from bisect import bisect_right

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial.data import fetch  # noqa: E402

MEMBERSHIP_CSV = os.path.join(ROOT, "manifests", "sp500_pit_membership.csv")
SP400_FILE = os.path.join(ROOT, "manifests", "sp400_symbols.txt")
END = "2026-06-01"
START = {"sp400": "2010-01-01", "sp500pit": "2009-01-01"}  # = existing caches
PRIMARY_K, PRIMARY_H = 3.0, 10
KS, HOLDS = [2.0, 3.0, 4.0], [1, 5, 10, 20]
HAIRCUT = 0.20


def load_membership(path):
    rows = [r for r in csv.reader(open(path))][1:]
    dates = [r[0] for r in rows]
    sets_ = [frozenset(t.strip() for t in r[1].split(",")) for r in rows]

    def member_on(day, sym):
        i = bisect_right(dates, str(day)) - 1
        return i >= 0 and sym in sets_[i]
    return member_on, sorted(set().union(*sets_)), sorted(sets_[-1])


def read_symbols(path):
    return [ln.strip() for ln in open(path)
            if ln.strip() and not ln.startswith("#")]


def zscore(df):
    c, h, lo = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()],
                   axis=1).max(axis=1)
    atr_pct_prev = (tr.rolling(14).mean() / c).shift(1)
    r1 = c / pc - 1
    return (-r1 / atr_pct_prev).values, r1.values


def events(data, spy, member_on, a):
    """One row per (event, k-bucket hit, H). An event with z=4.2 counts for
    k=2, 3 and 4 -- buckets are thresholds, not bins."""
    rows = []
    for sym, df in data.items():
        z, r1 = zscore(df)
        o, c, idx, n = df["open"].values, df["close"].values, df.index, len(df)
        hit = np.where((z >= min(KS)) & (r1 > -a.max_drop)
                       & (idx >= pd.Timestamp(a.sig_start)))[0]
        for i in hit:
            if i + 1 >= n or (member_on and not member_on(idx[i].date(), sym)):
                continue
            e_px = o[i + 1]
            if not np.isfinite(e_px) or e_px <= 0:
                continue
            s_in = spy["open"].asof(idx[i + 1])
            for H in HOLDS:
                x = min(i + H, n - 1)
                raw = c[x] / e_px - 1
                xs = raw - (spy["close"].asof(idx[x]) / s_in - 1)
                rows.append((idx[i], sym, z[i], r1[i], H, raw, xs,
                             i + H > n - 1, spy["r1"].asof(idx[i])))
    return pd.DataFrame(rows, columns=["date", "sym", "z", "r1", "H", "raw",
                                       "xs", "forced", "spy_r1"])


def stats(df, cost):
    if len(df) < 3:
        return dict(n=len(df), dates=0, raw=np.nan, xs=np.nan, dxs=np.nan,
                    win=np.nan, t=np.nan, forced=0, xs_hc=np.nan)
    raw, xs = df["raw"] - cost, df["xs"] - cost
    by_date = xs.groupby(df["date"]).mean()
    k, sd = len(by_date), by_date.std(ddof=1)
    t = by_date.mean() / sd * np.sqrt(k) if k > 2 and sd > 0 else np.nan
    return dict(n=len(df), dates=k, raw=raw.mean() * 1e4, xs=xs.mean() * 1e4,
                dxs=by_date.mean() * 1e4, win=(raw > 0).mean(), t=t,
                forced=int(df["forced"].sum()),
                xs_hc=xs.where(~df["forced"], xs - HAIRCUT).mean() * 1e4)


HDR = (f"  {'k':>3} {'H':>3} {'subset':<10} {'events':>7} {'dates':>6} "
       f"{'raw':>7} {'xs/evt':>7} {'xs/day':>7} {'win':>5} {'t(day)':>7} "
       f"{'forced':>6} {'xs-20%hc':>8}")


def row(k, H, name, s):
    return (f"  {k:>3.0f} {H:>3} {name:<10} {s['n']:>7} {s['dates']:>6} "
            f"{s['raw']:>7.1f} {s['xs']:>7.1f} {s['dxs']:>7.1f} "
            f"{s['win']:>5.2f} {s['t']:>7.2f} {s['forced']:>6} "
            f"{s['xs_hc']:>8.1f}")


def subsets(e, a):
    split = pd.Timestamp(a.split)
    mkt = e["spy_r1"] <= -a.mkt_crash
    return {"all": e, "train": e[e["date"] < split],
            "test": e[e["date"] >= split], "idiosync": e[~mkt],
            "mkt-crash": e[mkt]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground", choices=["sp400", "sp500pit"],
                    default="sp400")
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sig-start", default="2010-03-01")
    ap.add_argument("--split", default="2018-01-01")
    ap.add_argument("--cost-bps", type=float, default=20.0, help="round trip")
    ap.add_argument("--mkt-crash", type=float, default=0.02)
    ap.add_argument("--max-drop", type=float, default=0.60)
    a = ap.parse_args()
    cost, start = a.cost_bps / 1e4, START[a.ground]
    member_on, cov = None, None

    if a.smoke:
        data = fetch("synthetic", [f"S{i}" for i in range(12)], "1d",
                     start, END)
        spy = fetch("synthetic", ["SPYX"], "1d", start, END)["SPYX"]
    elif a.ground == "sp400":
        syms = read_symbols(SP400_FILE)
        print(f"[dipz] sp400 CURRENT membership: {len(syms)} names "
              f"(survivor-biased: this ground can only kill)")
        data = fetch(a.source, syms, "1d", start, END)
        spy = fetch(a.source, ["SPY"], "1d", start, END)["SPY"]
        print(f"  with data: {len(data)}/{len(syms)}")
    else:
        raw_m, union, cur = load_membership(MEMBERSHIP_CSV)
        union = [s.replace(".", "-") for s in union]
        print(f"[dipz] sp500 PIT: {len(union)} names -- CONTEXT ONLY "
              f"(this sample is spent; nothing here passes or fails)")
        data = fetch(a.source, union, "1d", start, END)
        spy = fetch(a.source, ["SPY"], "1d", start, END)["SPY"]
        member_on = lambda d, s: raw_m(d, s.replace("-", "."))  # noqa
        dep = [s for s in union if s not in set(cur)]
        cov = sum(s in data for s in dep) / max(1, len(dep))
        print(f"  departed-name coverage {cov:.0%}")
    spy = spy.copy()
    spy["r1"] = spy["close"].pct_change()

    ev = events(data, spy, member_on, a)
    if ev.empty:
        sys.exit("zero events -- check data/pipeline.")
    base = ev[ev["H"] == PRIMARY_H]
    for k in KS:
        b = base[base["z"] >= k]
        print(f"  k={k:.0f}: {len(b)} events | median drop "
              f"{b['r1'].median():+.1%} | share that were ALSO >5% drops "
              f"{(b['r1'] <= -0.05).mean():.0%}")

    prim = subsets(base[base["z"] >= PRIMARY_K], a)
    res = {n: stats(v, cost) for n, v in prim.items()}
    print(f"\n  == PRIMARY CELL (k={PRIMARY_K:.0f}, H={PRIMARY_H}, "
          f"{a.cost_bps:.0f} bps RT) ==\n{HDR}")
    for n, s in res.items():
        print(row(PRIMARY_K, PRIMARY_H, n, s))

    print(f"\n  == CONTEXT GRID (decides nothing) ==\n{HDR}")
    for k in KS:
        for H in HOLDS:
            e = ev[(ev["H"] == H) & (ev["z"] >= k)]
            for n in ("train", "test"):
                print(row(k, H, n, stats(subsets(e, a)[n], cost)))
        print()

    ok = all(res[n]["xs"] > 0 and res[n]["dxs"] > 0 and res[n]["t"] > 2
             for n in ("train", "test"))
    print("  ===== VERDICT (primary cell) =====")
    if a.smoke:
        print(f"  smoke run: pipeline OK ({'pass' if ok else 'fail'} on "
              f"synthetic, meaningless)")
    elif a.ground == "sp500pit":
        print(f"  CONTEXT ONLY -- spent sample, {'pass' if ok else 'fail'} "
              f"shape here changes nothing. Coverage {cov:.0%}.")
    elif ok:
        print("  SURVIVES the biased screen -- NOT a pass. Next: PIT sp400 "
              "(Norgate) under this same frozen cell, then prereg genome + "
              "gauntlet + duplication check vs deep_state.")
    else:
        print("  CLOSED on sp400 -- fails even with dead names missing; "
              "restoring them can only make it worse. Log to MECHANISM_MAP.")


if __name__ == "__main__":
    main()
