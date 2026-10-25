"""
dip5_screen.py -- quick SCREEN of Blake's idea: an S&P 500 name closes down
more than X% on the day -> buy next open -> hold H days -> exit at close.
One indicator, nothing else. Not a gauntlet verdict.

Runs the SAME event study twice, like pit_distress_sp500.py:
  SURVIVOR  current members only, no date gate (the biased, flattering view)
  PIT       every name ever in the index (manifests/sp500_pit_membership.csv),
            event counts only if the name was a member on the drop day;
            departed names' data used until it ends. If a name's data ends
            inside the hold, the trade exits at its last close (counted as
            'forced') and a -20% haircut variant is printed as stress.

Measured per event: raw return and return minus SPY over the same window
(SPY open at entry -> SPY close at exit), net of round-trip cost. t-stats are
clustered by DATE (crash days produce dozens of correlated events).

SCREEN RULE (pre-stated, PIT mode decides, survivor mode is context only):
  an H survives if PIT net excess-vs-SPY > 0 with date-clustered t > 2 in
  BOTH halves (pre/post --split). Survival = owes a prereg'd genome + gauntlet
  + duplication check vs deep_state. Four holds = four looks; pick one H
  before any follow-up.

Run (from repo root; cache shared with pit_distress_sp500.py):
    python3.10 scripts/dip5_screen.py
    python3.10 scripts/dip5_screen.py --drop 0.07 --cost-bps 20
    python3.10 scripts/dip5_screen.py --smoke
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
START, END = "2009-01-01", "2026-06-01"      # same cache keys as pit_distress
HAIRCUT = 0.20


def load_membership(path):
    rows = [r for r in csv.reader(open(path))][1:]
    dates = [r[0] for r in rows]
    sets_ = [frozenset(t.strip() for t in r[1].split(",")) for r in rows]

    def member_on(day, sym):
        i = bisect_right(dates, str(day)) - 1
        return i >= 0 and sym in sets_[i]
    return member_on, sorted(set().union(*sets_)), sorted(sets_[-1])


def events(data, spy, member_on, pit, a, holds):
    spy_o = spy["open"]
    spy_c = spy["close"]
    rows = []
    for sym, df in data.items():
        o, c = df["open"].values, df["close"].values
        idx = df.index
        r1 = np.r_[np.nan, c[1:] / c[:-1] - 1]
        hit = np.where((r1 <= -a.drop) & (r1 > -a.max_drop)
                       & (idx >= pd.Timestamp(a.sig_start)))[0]
        n = len(df)
        for i in hit:
            if i + 1 >= n:
                continue
            if pit and not member_on(idx[i].date(), sym):
                continue
            e_px = o[i + 1]
            if not np.isfinite(e_px) or e_px <= 0:
                continue
            e_day = idx[i + 1]
            s_in = spy_o.asof(e_day)
            for H in holds:
                x = min(i + H, n - 1)
                forced = (i + H > n - 1)
                if x < i + 1:
                    continue
                raw = c[x] / e_px - 1
                s_out = spy_c.asof(idx[x])
                xs = raw - (s_out / s_in - 1)
                rows.append((idx[i], sym, H, raw, xs, forced,
                             spy["r1"].asof(idx[i])))
    return pd.DataFrame(rows, columns=["date", "sym", "H", "raw", "xs",
                                       "forced", "spy_r1"])


def stats(df, cost):
    if df.empty:
        return dict(n=0, dates=0, raw=np.nan, xs=np.nan, win=np.nan,
                    t=np.nan, forced=0, xs_hc=np.nan)
    raw, xs = df["raw"] - cost, df["xs"] - cost
    by_date = xs.groupby(df["date"]).mean()
    k = len(by_date)
    sd = by_date.std(ddof=1)
    t = by_date.mean() / sd * np.sqrt(k) if k > 2 and sd > 0 else np.nan
    xs_hc = xs.where(~df["forced"], xs - HAIRCUT)
    return dict(n=len(df), dates=k, raw=raw.mean() * 1e4,
                xs=xs.mean() * 1e4, win=(raw > 0).mean(), t=t,
                forced=int(df["forced"].sum()), xs_hc=xs_hc.mean() * 1e4)


def report(name, ev, a, holds, cost):
    split = pd.Timestamp(a.split)
    print(f"\n  == {name} ==")
    print(f"  {'H':>3} {'subset':<10} {'events':>7} {'dates':>6} "
          f"{'raw bps':>8} {'xsSPY':>7} {'win':>5} {'t(date)':>8} "
          f"{'forced':>6} {'xs -20%hc':>9}")
    out = {}
    for H in holds:
        e = ev[ev["H"] == H]
        mkt = e["spy_r1"] <= -a.mkt_crash
        subs = {"all": e, "train": e[e["date"] < split],
                "test": e[e["date"] >= split],
                "idiosync": e[~mkt], "mkt-crash": e[mkt]}
        res = {k: stats(v, cost) for k, v in subs.items()}
        for k, s in res.items():
            print(f"  {H:>3} {k:<10} {s['n']:>7} {s['dates']:>6} "
                  f"{s['raw']:>8.1f} {s['xs']:>7.1f} {s['win']:>5.2f} "
                  f"{s['t']:>8.2f} {s['forced']:>6} {s['xs_hc']:>9.1f}")
        print()
        out[H] = all(res[k]["xs"] > 0 and res[k]["t"] > 2
                     for k in ("train", "test"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--drop", type=float, default=0.05)
    ap.add_argument("--sig-start", default="2010-01-01")
    ap.add_argument("--split", default="2018-01-01")
    ap.add_argument("--holds", default="1,5,10,20")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="round trip")
    ap.add_argument("--mkt-crash", type=float, default=0.02,
                    help="SPY day drop that tags a market-wide event")
    ap.add_argument("--max-drop", type=float, default=0.60,
                    help="ignore bigger 1-day drops (data glitches)")
    a = ap.parse_args()
    holds = [int(h) for h in a.holds.split(",")]
    cost = a.cost_bps / 1e4

    if a.smoke:
        syms = [f"S{i}" for i in range(12)]
        cur = syms[:8]
        member_on = lambda d, s: True  # noqa: E731
        data = fetch("synthetic", syms, "1d", START, END)
        spy = fetch("synthetic", ["SPYX"], "1d", START, END)["SPYX"]
        cov = 1.0
    else:
        member_on, union, cur = load_membership(MEMBERSHIP_CSV)
        union = [s.replace(".", "-") for s in union]
        cur = [s.replace(".", "-") for s in cur]
        print(f"[dip5] {len(union)} names ever in index | {len(cur)} current "
              f"| fetching (cache-heavy)")
        data = fetch(a.source, union, "1d", START, END)
        spy = fetch(a.source, ["SPY"], "1d", START, END)["SPY"]
        raw_member = member_on
        member_on = lambda d, s: raw_member(d, s.replace("-", "."))  # noqa
        departed = [s for s in union if s not in set(cur)]
        have = [s for s in departed if s in data]
        cov = len(have) / max(1, len(departed))
        print(f"  departed names: {len(departed)} | with data: {len(have)} "
              f"| coverage {cov:.0%}")
    spy = spy.copy()
    spy["r1"] = spy["close"].pct_change()

    print(f"  drop >{a.drop:.0%} | buy next open | cost {a.cost_bps:.0f} bps "
          f"RT | signals from {a.sig_start} | split {a.split}")
    curset = set(cur)
    surv = events({s: d for s, d in data.items() if s in curset}, spy,
                  member_on, False, a, holds)
    pit = events(data, spy, member_on, True, a, holds)
    if surv.empty or pit.empty:
        sys.exit("zero events in a mode -- check data/pipeline.")

    report("SURVIVOR (current members, no gate) -- context only",
           surv, a, holds, cost)
    verdict = report("PIT (all names, membership-gated) -- DECIDES",
                     pit, a, holds, cost)

    print("  ===== SCREEN VERDICT (PIT) =====")
    if cov < 0.50:
        print(f"  VOID -- departed coverage {cov:.0%} < 50%; PIT mode can't "
              f"speak. Survivor numbers above are upper bounds, not evidence.")
        return
    for H, ok in verdict.items():
        print(f"  H={H:>2}: {'SURVIVES SCREEN -> owes prereg genome + gauntlet + dup check vs deep_state' if ok else 'dead'}")
    if cov < 0.80:
        print(f"  note: coverage {cov:.0%} -- missing dead names still "
              f"flatter PIT; a survivor here owes the Norgate re-test.")


if __name__ == "__main__":
    main()
