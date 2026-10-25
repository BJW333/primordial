"""
dipbase_screen.py -- is the SIZE of a one-day drop a real indicator, or
just survivorship? Compares dip buys against OTHER STOCKS IN THE SAME INDEX
over the SAME window, not against SPY.

Baseline: for every day, the average H-day return (close entry -> close
exit) of every index member that day. On today's-membership lists this
baseline carries the same survivor lift as the dips, so subtracting it
cancels most of it out. On the S&P 500 it is membership-gated.

    adj = dip stock's H-day return - same-day index-average H-day return
    net = adj - round-trip cost

Bands (narrow, so the dose-response is visible): 2-3, 3-5, 5-8, 8-12,
12-25%, plus 2-25% all. Holds 5, 10, 20. Entry at the drop day's close.

PASS RULE (stated before running): a (band, H) cell passes only if net > 0
with date-clustered t > 2 in BOTH halves (pre/post 2018) on ALL THREE
indexes. Separately printed: does adj rise with drop size on each index
(the "bands are an indicator" question). Dose-response alone is not a pass.

Run (repo root):
    python3.10 scripts/dipbase_screen.py
    python3.10 scripts/dipbase_screen.py --smoke
"""
import argparse
import csv
import os
import sys

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
NARROW = [(0.02, 0.03), (0.03, 0.05), (0.05, 0.08), (0.08, 0.12),
          (0.12, 0.25)]
BANDS = NARROW + [(0.02, 0.25)]
HOLDS = [5, 10, 20]
SIG_START, SPLIT = pd.Timestamp("2010-03-01"), pd.Timestamp("2018-01-01")
MAX_FWD = 2.0


def membership(path):
    rows = [r for r in csv.reader(open(path))][1:]
    dates = pd.DatetimeIndex([r[0] for r in rows])
    sets_ = [frozenset(t.strip() for t in r[1].split(",")) for r in rows]
    return dates, sets_


def member_mask(sym, idx, mdates, sets_):
    s = sym.replace("-", ".")
    rows = np.array([s in x for x in sets_])
    k = mdates.searchsorted(idx, side="right") - 1
    return np.where(k >= 0, rows[np.clip(k, 0, None)], False)


def read_symbols(path):
    return [ln.strip() for ln in open(path)
            if ln.strip() and not ln.startswith("#")]


def build(data, gidx, mem):
    """Returns baseline[H] (Series on gidx) and an events DataFrame."""
    G = len(gidx)
    sums = {H: np.zeros(G) for H in HOLDS}
    cnts = {H: np.zeros(G) for H in HOLDS}
    evs = []
    for sym, df in data.items():
        df = df[~df.index.duplicated()]
        idx, c, n = df.index, df["close"].values, len(df)
        r1 = np.r_[np.nan, c[1:] / c[:-1] - 1]
        ok_mem = mem(sym, idx) if mem else np.ones(n, bool)
        pos = gidx.get_indexer(idx)
        for H in HOLDS:
            fwd = np.full(n, np.nan)
            if n > H:
                fwd[:-H] = c[H:] / c[:-H] - 1          # full H only
            good = ok_mem & np.isfinite(fwd) & (np.abs(fwd) <= MAX_FWD) \
                & (pos >= 0)
            np.add.at(sums[H], pos[good], fwd[good])
            np.add.at(cnts[H], pos[good], 1)
            hit = good & (r1 <= -0.02) & (r1 > -0.25) & (idx >= SIG_START)
            if hit.any():
                evs.append(pd.DataFrame({"date": idx[hit], "r1": r1[hit],
                                         "H": H, "fwd": fwd[hit]}))
    base = {H: pd.Series(np.where(cnts[H] >= 20, sums[H] /
                                  np.maximum(cnts[H], 1), np.nan), gidx)
            for H in HOLDS}
    ev = pd.concat(evs, ignore_index=True)
    ev["base"] = [base[h].get(d, np.nan) for d, h in zip(ev["date"], ev["H"])]
    ev["adj"] = ev["fwd"] - ev["base"]
    return ev.dropna(subset=["adj"])


def stat(e, cost):
    d = (e["adj"] - cost).groupby(e["date"]).mean()
    k = len(d)
    if k < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    return d.mean() * 1e4, d.mean() / (d.std(ddof=1) / np.sqrt(k))


def run_ground(name, a):
    start, cost = GROUNDS[name]
    mem = None
    if a.smoke:
        data, gidx = a.smoke_data
    else:
        if name == "sp500":
            mdates, sets_ = membership(
                os.path.join(M, "sp500_pit_membership.csv"))
            syms = sorted({s.replace(".", "-") for x in sets_ for s in x})
            mem = lambda s, idx: member_mask(s, idx, mdates, sets_)  # noqa
        else:
            syms = read_symbols(os.path.join(M, f"{name}_symbols.txt"))
        data = fetch(a.source, syms, "1d", start, END)
        gidx = fetch(a.source, ["SPY"], "1d", start, END)["SPY"].index
    ev = build(data, gidx, mem)

    print(f"\n  == {name.upper()} -- vs same-day index average, entry close, "
          f"{cost * 1e4:.0f} bps RT, {len(data)} names ==")
    print(f"  {'band':<9} {'H':>3} {'events':>8} | {'train':>7} {'t':>6} | "
          f"{'test':>7} {'t':>6} | pass")
    res, dose = {}, {}
    for lo, hi in BANDS:
        b = ev[(ev["r1"] <= -lo) & (ev["r1"] > -hi)]
        for H in HOLDS:
            e = b[b["H"] == H]
            tr = stat(e[e["date"] < SPLIT], cost)
            te = stat(e[e["date"] >= SPLIT], cost)
            ok = all(np.isfinite(v[1]) and v[0] > 0 and v[1] > 2
                     for v in (tr, te))
            res[(lo, hi, H)] = ok
            if (lo, hi) in NARROW:
                dose.setdefault(H, []).append(stat(e, 0.0)[0])
            print(f"  {lo:.0%}-{hi:.0%}".ljust(11) + f"{H:>3} {len(e):>8} | "
                  f"{tr[0]:>7.1f} {tr[1]:>6.2f} | {te[0]:>7.1f} {te[1]:>6.2f}"
                  f" | {'YES' if ok else ''}")
        print()
    print("  dose-response (gross adj bps, smallest -> biggest drop band):")
    for H, v in dose.items():
        rising = all(np.diff(v) > 0)
        print(f"    H={H:>2}: " + "  ".join(f"{x:+.0f}" for x in v) +
              f"   {'RISES every step' if rising else 'not monotone'}")
    return res


def smoke_data():
    gidx = pd.bdate_range("2010-01-01", "2026-05-29")
    rng = np.random.default_rng(5)
    data = {}
    for k in range(60):
        r = rng.normal(0.0003, 0.02, len(gidx))
        for i in rng.choice(np.arange(300, len(gidx) - 25), 40,
                            replace=False):           # planted: only ~10%
            r[i] = -0.10                              # drops rebound
            r[i + 1:i + 21] += 0.003
        c = 100 * np.exp(np.cumsum(r))
        data[f"S{k}"] = pd.DataFrame({"open": c, "high": c, "low": c,
                                      "close": c}, gidx)
    return data, gidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    grounds = list(GROUNDS)
    if a.smoke:
        a.smoke_data, grounds = smoke_data(), ["sp500"]
    res = {g: run_ground(g, a) for g in grounds}
    cells = [k for k in res[grounds[0]] if all(res[g][k] for g in grounds)]
    print("\n  ===== VERDICT (pre-stated: both halves t>2 on every index) ====")
    if a.smoke:
        print("  smoke run (one synthetic index) -- checking the pipe only")
    for lo, hi, H in cells:
        print(f"  SURVIVES: drop {lo:.0%}-{hi:.0%}, hold {H} beats its own "
              f"index -> owes PIT data, prereg genome, gauntlet, dup check "
              f"vs deep_state")
    if not cells:
        print("  NO band beats its own index on all three -- drop size is "
              "not a usable indicator. CLOSED.")


if __name__ == "__main__":
    main()
