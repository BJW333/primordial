"""
dipband_screen.py -- Blake's dip-buy with a DYNAMIC drop band.

Event: an S&P name closes down between 2% and X% on the day.
Bands tested (lower edge fixed at 2%): 2-3, 2-4, 2-5, 2-7, 2-10, 2-15, 2-25%
plus the narrow slices 3-5, 5-8, 8-12, 12-25% to show WHERE any edge lives.
Entry: --entry close (default; buy at the close of the drop day, i.e. the
algo buys when it sees the dip) or --entry open (next morning's open).
Hold H in {1,5,10,20} bars, exit at close. Measured as return minus SPY
over the same window, net of round-trip cost, t clustered by date.

PASS RULE (stated before running): a (band, H) cell passes only if net
excess > 0 with date-clustered t > 2 in BOTH halves (pre/post 2018) on ALL
THREE indexes. Many cells are printed; requiring every index to agree is
the guard against picking the lucky one. A pass = owes PIT data, a prereg'd
genome, the gauntlet, and a duplication check vs deep_state.

Grounds: sp500 (PIT membership-gated, 10 bps RT), sp400 (current, 20 bps),
sp600 (current, 30 bps). sp400/sp600 are survivor-flattered.

Run (repo root):
    python3.10 scripts/dipband_screen.py
    python3.10 scripts/dipband_screen.py --entry open
    python3.10 scripts/dipband_screen.py --smoke
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
GROUNDS = {"sp500": ("2009-01-01", 0.0010),
           "sp400": ("2010-01-01", 0.0020),
           "sp600": ("2010-01-01", 0.0030)}
BANDS = [(0.02, x) for x in (0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.25)] + \
        [(0.03, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 0.25)]
HOLDS = [1, 5, 10, 20]
SIG_START, SPLIT = "2010-03-01", pd.Timestamp("2018-01-01")
MAX_FWD = 2.0


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


def events(data, spy, member_on, entry):
    out = []
    for sym, df in data.items():
        idx, n = df.index, len(df)
        o, c = df["open"].values, df["close"].values
        sp = spy.reindex(idx, method="ffill")
        so, sc = sp["open"].values, sp["close"].values
        r1 = np.r_[np.nan, c[1:] / c[:-1] - 1]
        i = np.where((r1 <= -0.02) & (r1 > -0.25)
                     & (idx >= pd.Timestamp(SIG_START)))[0]
        i = i[i + 1 < n]
        if member_on is not None and len(i):
            keep = np.array([member_on(idx[j].date(), sym) for j in i])
            i = i[keep]
        if not len(i):
            continue
        if entry == "close":
            e_px, s_in = c[i], sc[i]
        else:
            e_px, s_in = o[i + 1], so[i + 1]
        ok = np.isfinite(e_px) & (e_px > 0) & np.isfinite(s_in)
        i, e_px, s_in = i[ok], e_px[ok], s_in[ok]
        for H in HOLDS:
            x = np.minimum(i + H, n - 1)
            raw = c[x] / e_px - 1
            xs = raw - (sc[x] / s_in - 1)
            out.append(pd.DataFrame({"date": idx[i], "r1": r1[i], "H": H,
                                     "raw": raw, "xs": xs}))
    ev = pd.concat(out, ignore_index=True)
    bad = ev["raw"].abs() > MAX_FWD
    return ev[~bad], int(bad.sum())


def stat(e, cost):
    d = (e["xs"] - cost).groupby(e["date"]).mean()
    k = len(d)
    if k < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan, len(e)
    return d.mean() * 1e4, d.mean() / (d.std(ddof=1) / np.sqrt(k)), len(e)


def run_ground(name, a):
    start, cost = GROUNDS[name]
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
    if not a.smoke:
        spy = fetch(a.source, ["SPY"], "1d", start, END)["SPY"]

    ev, nbad = events(data, spy, member_on, a.entry)
    print(f"\n  == {name.upper()} -- entry {a.entry}, {cost * 1e4:.0f} bps RT"
          f" -- {nbad} glitch events dropped ==")
    print(f"  {'band':<9} {'H':>3} {'events':>8} | {'train xs':>8} "
          f"{'t':>6} | {'test xs':>8} {'t':>6} | pass")
    res = {}
    for lo, hi in BANDS:
        b = ev[(ev["r1"] <= -lo) & (ev["r1"] > -hi)]
        for H in HOLDS:
            e = b[b["H"] == H]
            tr = stat(e[e["date"] < SPLIT], cost)
            te = stat(e[e["date"] >= SPLIT], cost)
            ok = all(np.isfinite(v[1]) and v[0] > 0 and v[1] > 2
                     for v in (tr, te))
            res[(lo, hi, H)] = ok
            print(f"  {lo:.0%}-{hi:.0%}".ljust(11) + f"{H:>3} {len(e):>8} | "
                  f"{tr[0]:>8.1f} {tr[1]:>6.2f} | {te[0]:>8.1f} "
                  f"{te[1]:>6.2f} | {'YES' if ok else ''}")
        print()
    return res


def smoke_data():
    idx = pd.bdate_range("2010-01-01", "2026-05-29")
    rng = np.random.default_rng(11)

    def mk(plant):
        r = rng.normal(0.0003, 0.018, len(idx))
        if plant:   # 3-5% drops get a planted 10-day rebound
            hit = np.where((r <= -0.03) & (r > -0.05))[0]
            for i in hit[hit < len(idx) - 12]:
                r[i + 1:i + 11] += 0.002
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[100, c[:-1]]
        return pd.DataFrame({"open": o, "high": np.maximum(o, c),
                             "low": np.minimum(o, c), "close": c}, idx)
    return {f"S{i}": mk(True) for i in range(30)}, mk(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--entry", choices=["close", "open"], default="close")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    grounds = list(GROUNDS)
    if a.smoke:
        a.smoke_data, grounds = smoke_data(), ["sp500"]

    res = {g: run_ground(g, a) for g in grounds}
    cells = [k for k in res[grounds[0]]
             if all(res[g][k] for g in grounds)]
    print("  ===== VERDICT (pre-stated: both halves t>2 on every index) =====")
    if a.smoke:
        print("  smoke run (one synthetic index) -- checking the pipe only")
    if cells:
        for lo, hi, H in cells:
            print(f"  SURVIVES SCREEN: drop {lo:.0%}-{hi:.0%}, hold {H} -> "
                  f"owes PIT data, prereg genome, gauntlet, dup check vs "
                  f"deep_state")
    else:
        print("  NO band/hold survives on all indexes -- dip band CLOSED.")


if __name__ == "__main__":
    main()
