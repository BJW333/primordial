"""
dip_combo_search.py -- explicit combo search: which COMBINATIONS of up to 3
indicators tell a dip that rebounds from a dip that keeps falling?

Uses the 17 indicators and the dip pool from dip_features.py (dips 2-25%,
outcome = H-day return minus the same-day index average).

Each indicator is cut into thirds (LOW / MID / HIGH) using S&P 500
2010-2017 dips. A combo is 1, 2 or 3 conditions joined by AND, e.g.
    close_loc=HIGH & ret5=LOW & vol_ratio=HIGH
That is 51 singles + 1,224 pairs + 18,360 triples = 19,635 combos, each at
H = 10 and 20 days: ~39,000 tests.

HOW IT AVOIDS FINDING LUCK (stated before running):
  1. DISCOVER on S&P 500 2010-2017 only. S&P 500 is membership-checked
     date by date, so the "tiny stock that later joined the index" trap
     from dip_features can't happen. A combo needs >= 300 dips on >= 100
     different days to count.
  2. Keep the 20 best REBOUND combos and the 20 best KEEPS-FALLING combos
     by discovery t-stat. Nothing else moves forward.
  3. CONFIRM those 40 on S&P 500 2018-2026 (never used in discovery).
     40 tests -> the bar is |t| > 3.2 (Bonferroni), net of cost, same sign.
  4. REPLICATE confirmed combos on S&P 400 and S&P 600 (whole period,
     same sign, |t| > 2). Those lists are survivor-biased, so this can only
     kill a combo, not crown it.
  A combo that clears 3 AND 4 is a real dip separator -> Primordial genome,
  gauntlet, forward paper. If none do, no combination of these 17
  indicators separates dips, and that's now shown explicitly.

Run (repo root; discovery is fast, building the dip tables is the slow
part, same as dip_features.py):
    python3.10 scripts/dip_combo_search.py
    python3.10 scripts/dip_combo_search.py --smoke
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dip_features as DF  # noqa: E402

HOLDS = [10, 20]
MIN_EVENTS, MIN_DAYS, KEEP = 300, 100, 20
BAR_CONFIRM, BAR_REPLICATE = 3.2, 2.0
LV = ["LOW", "MID", "HIGH"]


def levels(ev, cuts):
    """Feature -> int array 0/1/2 (or -1 if missing), using fixed cuts."""
    out = {}
    for f, (q1, q2) in cuts.items():
        x = ev[f].values
        lv = np.where(x <= q1, 0, np.where(x >= q2, 2, 1))
        out[f] = np.where(np.isfinite(x), lv, -1)
    return out


def day_t(y, day_codes, mask, n_days):
    """Date-clustered mean (bps) and t for the events in mask."""
    if mask.sum() < 3:
        return np.nan, np.nan, 0, 0
    d = day_codes[mask]
    s = np.bincount(d, weights=y[mask], minlength=n_days)
    c = np.bincount(d, minlength=n_days)
    k = c > 0
    m = s[k] / c[k]
    if k.sum() < 3 or m.std(ddof=1) == 0:
        return np.nan, np.nan, int(mask.sum()), int(k.sum())
    return (m.mean() * 1e4, m.mean() / (m.std(ddof=1) / np.sqrt(k.sum())),
            int(mask.sum()), int(k.sum()))


def combos(feats):
    for r in (1, 2, 3):
        for fs in itertools.combinations(feats, r):
            for lvs in itertools.product(range(3), repeat=r):
                yield tuple(zip(fs, lvs))


def name(combo):
    return " & ".join(f"{f}={LV[v]}" for f, v in combo)


def mask_of(combo, lv):
    m = np.ones(len(next(iter(lv.values()))), bool)
    for f, v in combo:
        m &= lv[f] == v
    return m


def prep(ev):
    codes, uniq = pd.factorize(ev["date"])
    return codes, len(uniq)


def evaluate(combo_list, ev, lv, H, cost, sub=None):
    y = (ev[f"adj{H}"] - cost).values
    codes, nd = prep(ev)
    base = np.ones(len(ev), bool) if sub is None else sub
    return [day_t(y, codes, mask_of(c, lv) & base, nd) for c in combo_list]


def run(a):
    grounds = {}
    for g in ("sp500", "sp400", "sp600"):
        data, spy, mem, cost = DF.load(g, a)
        grounds[g] = (DF.build(data, spy, mem), cost)
        print(f"  [{g}] {len(grounds[g][0])} dips")

    ev, cost5 = grounds["sp500"]
    tr, te = DF.split(ev)
    cuts = {}
    for f in DF.FEATS:
        q = tr[f].quantile([1 / 3, 2 / 3]).values
        if np.isfinite(q).all() and q[0] < q[1]:
            cuts[f] = (q[0], q[1])
    feats = sorted(cuts)
    all_c = list(combos(feats))
    lv_tr = levels(tr, cuts)
    print(f"\n  discovery: {len(all_c)} combos x {len(HOLDS)} holds = "
          f"{len(all_c) * len(HOLDS)} tests on S&P 500 2010-2017")

    verdict = []
    for H in HOLDS:
        y = tr[f"adj{H}"].values
        codes, nd = prep(tr)
        res = []
        for c in all_c:
            mu, t, n, k = day_t(y, codes, mask_of(c, lv_tr), nd)
            if n >= MIN_EVENTS and k >= MIN_DAYS and np.isfinite(t):
                res.append((t, mu, n, c))
        res.sort(key=lambda r: r[0])
        picks = [("REBOUND", r) for r in res[::-1][:KEEP]] + \
                [("KEEPS FALLING", r) for r in res[:KEEP]]
        print(f"\n  ===== H={H}: {len(res)} combos had enough dips; "
              f"top {KEEP} each way move forward =====")

        lv_te = levels(te, cuts)
        conf = evaluate([r[3] for _, r in picks], te, lv_te, H, cost5)
        reps = {}
        for g in ("sp400", "sp600"):
            evg, cg = grounds[g]
            reps[g] = evaluate([r[3] for _, r in picks], evg,
                               levels(evg, cuts), H, cg)

        print(f"  {'side':<13} {'combo':<58} {'disc t':>6} | "
              f"{'2018+ bps':>9} {'t':>6} | {'sp400 t':>7} {'sp600 t':>7} "
              f"| result")
        for (side, (t0, mu0, n0, c)), cf, r4, r6 in zip(
                picks, conf, reps["sp400"], reps["sp600"]):
            sgn = 1 if side == "REBOUND" else -1
            ok_c = np.isfinite(cf[1]) and sgn * cf[1] > BAR_CONFIRM \
                and sgn * cf[0] > 0
            ok_r = all(np.isfinite(r[1]) and sgn * r[1] > BAR_REPLICATE
                       for r in (r4, r6))
            tag = ("CONFIRMED + REPLICATED" if ok_c and ok_r else
                   "confirmed, fails replication" if ok_c else "")
            if ok_c and ok_r:
                verdict.append((H, side, name(c), cf))
            print(f"  {side:<13} {name(c)[:58]:<58} {t0:>+6.2f} | "
                  f"{cf[0]:>+9.1f} {cf[1]:>+6.2f} | {r4[1]:>+7.2f} "
                  f"{r6[1]:>+7.2f} | {tag}")

    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke run: same synthetic data on all three grounds, "
              "planted signal -- pipe check only")
    if verdict:
        for H, side, nm, cf in verdict:
            print(f"  H={H} {side}: {nm}  (2018+ {cf[0]:+.1f} bps net, "
                  f"t {cf[1]:+.2f}) -> Primordial genome, gauntlet, "
                  f"forward paper")
    else:
        print(f"  NONE of the {len(all_c) * len(HOLDS)} combos found in "
              f"discovery held up on unseen years + other indexes. No "
              f"combination of these 17 indicators separates rebounding "
              f"dips from falling ones.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.smoke_data = DF.smoke_data()
    run(a)


if __name__ == "__main__":
    main()
