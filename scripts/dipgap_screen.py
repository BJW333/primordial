"""
dipgap_screen.py -- does HOW a stock dropped decide whether it bounces?

Event: close-to-close drop > 5% (Blake's original trigger). Each event is
tagged by gap share:
    g = (open_t - close_{t-1}) / (close_t - close_{t-1})
    g <= 1/3  INTRADAY  (opened near flat, sold off in session -> flow,
                         expected to REVERT)
    g >= 2/3  GAP       (most of the drop happened overnight -> news,
                         expected to DRIFT on down)
    between   mixed     (reported, not tested)
Cutpoints are fixed numbers, not data-fitted terciles.
Buy next open, hold 10 bars, exit at close. Measured as return minus SPY
over the same window, net of cost, t-stats clustered by date.

GROUNDS (see prereg/DIPGAP_PREREG.md):
  sp600  PRIMARY (fresh sample; current membership)    cost 30 bps RT
  sp400  REPLICATION (3x-drop events already seen)     cost 20 bps RT
  sp500  REPLICATION, PIT-gated (5% events seen)       cost 10 bps RT
The bucket SPREAD is conservative under survivorship: the dead names
missing from the data would mostly have landed in the losing (GAP) bucket,
so their absence flatters GAP and shrinks the spread. The INTRADAY bucket's
own level is still survivor-flattered on sp600/sp400.

VERDICT (pre-stated):
  sp600 must show BOTH, in BOTH halves (pre/post 2018):
    (a) spread INTRADAY - GAP > 0 with t > 2
    (b) INTRADAY bucket net excess > 0 with t > 2
  AND sp400 + sp500 full-sample spreads must both be > 0 (same sign).
  (a)+(b)+replication -> SURVIVES SCREEN: owes PIT sp600 (Norgate), then a
      prereg'd genome + gauntlet + duplication check vs deep_state.
  (a) only           -> mechanism real, long side not tradeable here;
                        short-the-gap is a NEW family, not this one.
  else               -> CLOSED.

Run (repo root):
    python3.10 scripts/dipgap_screen.py            # all three grounds, one look
    python3.10 scripts/dipgap_screen.py --smoke    # planted control
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

M = os.path.join(ROOT, "manifests")
END = "2026-06-01"
GROUNDS = {  # name: (start = existing cache key, cost RT, role)
    "sp600": ("2010-01-01", 0.0030, "PRIMARY"),
    "sp400": ("2010-01-01", 0.0020, "replication"),
    "sp500": ("2009-01-01", 0.0010, "replication (PIT)"),
}
DROP, H, LO, HI = 0.05, 10, 1 / 3, 2 / 3
SIG_START, SPLIT = "2010-03-01", pd.Timestamp("2018-01-01")
MAX_FWD = 2.0    # |raw 10d return| > 200% treated as a data glitch, dropped


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


def events(data, spy, member_on):
    rows = []
    s_o, s_c = spy["open"], spy["close"]
    for sym, df in data.items():
        o, c, idx, n = df["open"].values, df["close"].values, df.index, len(df)
        pc = np.r_[np.nan, c[:-1]]
        r1 = c / pc - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (o - pc) / (c - pc)
        hit = np.where((r1 <= -DROP) & (r1 > -0.6)
                       & (idx >= pd.Timestamp(SIG_START)))[0]
        for i in hit:
            if i + 1 >= n or not np.isfinite(g[i]):
                continue
            if member_on and not member_on(idx[i].date(), sym):
                continue
            e_px = o[i + 1]
            if not np.isfinite(e_px) or e_px <= 0:
                continue
            x = min(i + H, n - 1)
            raw = c[x] / e_px - 1
            xs = raw - (s_c.asof(idx[x]) / s_o.asof(idx[i + 1]) - 1)
            b = "INTRADAY" if g[i] <= LO else "GAP" if g[i] >= HI else "mixed"
            rows.append((idx[i], sym, b, g[i], raw, xs))
    return pd.DataFrame(rows, columns=["date", "sym", "bucket", "g",
                                       "raw", "xs"])


def day_means(e, cost):
    return (e["xs"] - cost).groupby(e["date"]).mean()


def level(e, cost):
    d = day_means(e, cost)
    k = len(d)
    if k < 3 or d.std(ddof=1) == 0:
        return dict(n=len(e), days=k, xs=np.nan, t=np.nan, se=np.nan,
                    win=np.nan, raw=np.nan)
    se = d.std(ddof=1) / np.sqrt(k)
    return dict(n=len(e), days=k, xs=d.mean() * 1e4, t=d.mean() / se,
                se=se * 1e4, win=((e["raw"] - cost) > 0).mean(),
                raw=(e["raw"] - cost).mean() * 1e4)


def spread(a, b):
    if not (np.isfinite(a["xs"]) and np.isfinite(b["xs"])):
        return np.nan, np.nan
    s = a["xs"] - b["xs"]
    return s, s / np.hypot(a["se"], b["se"])


def run_ground(name, a):
    start, cost, role = GROUNDS[name]
    member_on = None
    if a.smoke:
        data, spy = a.smoke_data
    elif name == "sp500":
        raw_m, union, cur = load_membership(
            os.path.join(M, "sp500_pit_membership.csv"))
        union = [s.replace(".", "-") for s in union]
        data = fetch(a.source, union, "1d", start, END)
        member_on = lambda d, s: raw_m(d, s.replace("-", "."))  # noqa
        dep = [s for s in union if s not in set(cur)]
        print(f"  [{name}] departed coverage "
              f"{sum(s in data for s in dep) / max(1, len(dep)):.0%}")
    else:
        syms = read_symbols(os.path.join(M, f"{name}_symbols.txt"))
        data = fetch(a.source, syms, "1d", start, END)
        print(f"  [{name}] {len(data)}/{len(syms)} names with data "
              f"(current membership)")
    if not a.smoke:
        spy = fetch(a.source, ["SPY"], "1d", start, END)["SPY"]

    ev = events(data, spy, member_on)
    bad = ev["raw"].abs() > MAX_FWD
    if bad.any():
        print(f"  [{name}] dropped {int(bad.sum())} glitch events "
              f"(|10d return| > {MAX_FWD:.0%})")
    ev = ev[~bad]

    print(f"\n  == {name.upper()} -- {role} -- drop >{DROP:.0%}, H={H}, "
          f"{cost * 1e4:.0f} bps RT ==")
    print(f"  {'half':<6} {'bucket':<9} {'events':>7} {'days':>6} "
          f"{'raw':>7} {'xsSPY':>7} {'win':>5} {'t':>6}")
    out = {}
    for half, e in (("all", ev), ("train", ev[ev["date"] < SPLIT]),
                    ("test", ev[ev["date"] >= SPLIT])):
        lv = {b: level(e[e["bucket"] == b], cost)
              for b in ("INTRADAY", "mixed", "GAP")}
        for b, s in lv.items():
            print(f"  {half:<6} {b:<9} {s['n']:>7} {s['days']:>6} "
                  f"{s['raw']:>7.1f} {s['xs']:>7.1f} {s['win']:>5.2f} "
                  f"{s['t']:>6.2f}")
        sp, st = spread(lv["INTRADAY"], lv["GAP"])
        print(f"  {half:<6} SPREAD intraday-gap {sp:+.1f} bps  t {st:+.2f}\n")
        out[half] = dict(lv=lv, spread=sp, spread_t=st)
    return out


def smoke_data():
    idx = pd.bdate_range("2010-01-01", "2026-05-29")
    rng = np.random.default_rng(7)

    def mk(plant):
        r = rng.normal(0.0003, 0.02, len(idx))
        o_gap = np.zeros(len(idx))
        if plant:
            for i in rng.choice(np.arange(300, len(idx) - 30), 80,
                                replace=False):
                r[i] = -0.08
                if rng.random() < 0.5:          # intraday drop -> rebound
                    r[i + 1:i + 11] += 0.004
                else:                           # gap drop -> drift down
                    o_gap[i] = 1
                    r[i + 1:i + 11] -= 0.004
        c = 100 * np.exp(np.cumsum(r))
        pc = np.r_[100, c[:-1]]
        o = np.where(o_gap == 1, pc + 0.9 * (c - pc), pc * (1 + rng.normal(
            0, 0.002, len(idx))))
        return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005,
                             "low": np.minimum(o, c) * 0.995, "close": c},
                            idx)
    return {f"S{i}": mk(True) for i in range(40)}, mk(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    grounds = ["sp600", "sp400", "sp500"]
    if a.smoke:
        a.smoke_data, grounds = smoke_data(), ["sp600"]

    res = {g: run_ground(g, a) for g in grounds}
    p = res["sp600"]
    ok_a = all(p[h]["spread"] > 0 and p[h]["spread_t"] > 2
               for h in ("train", "test"))
    ok_b = all(p[h]["lv"]["INTRADAY"]["xs"] > 0
               and p[h]["lv"]["INTRADAY"]["t"] > 2 for h in ("train", "test"))
    rep = {g: res[g]["all"]["spread"] for g in grounds if g != "sp600"}
    ok_r = all(v > 0 for v in rep.values()) if rep else True

    print("  ===== VERDICT (pre-stated) =====")
    print(f"  (a) sp600 spread, both halves t>2: {'YES' if ok_a else 'no'}")
    print(f"  (b) sp600 INTRADAY level, both halves t>2: "
          f"{'YES' if ok_b else 'no'}")
    for g, v in rep.items():
        print(f"  replication {g}: spread {v:+.1f} bps "
              f"{'(same sign)' if v > 0 else '(WRONG sign)'}")
    if a.smoke:
        print("  smoke: planted control, verdict below is about the pipe")
    if ok_a and ok_b and ok_r:
        print("  SURVIVES SCREEN -- owes PIT sp600 (Norgate), then prereg "
              "genome + gauntlet + duplication check vs deep_state.")
    elif ok_a and ok_r:
        print("  MECHANISM REAL, LONG SIDE NOT TRADEABLE -- intraday drops do "
              "beat gap drops, but the bounce doesn't beat SPY after cost. "
              "Short-the-gap = new family.")
    else:
        print("  CLOSED -- log to MECHANISM_MAP.md.")


if __name__ == "__main__":
    main()
