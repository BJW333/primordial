"""
insider_dip_test.py -- does an INSIDER PURCHASE tell a dip that rebounds
from a dip that keeps falling?

Needs data/insider_buys.csv.gz from fetch_insider_buys.py.

FROZEN SPEC (stated before running)
  dip        on day d the stock is down >= 10% over the last 10 trading days
  INSIDER    an officer/director open-market purchase (>= $25k) was FILED
             with the SEC in the 5 trading days before d (filing <= d-1, so
             the buy was public before we act)
  CONTROL    same dip condition on day d, NO insider purchase filed in the
             20 trading days before d
  entry      close of day d. Events are de-clustered: per stock and group,
             a new event needs 20 trading days since the last one
  outcome    stock return minus the equal-weight average of every index
             member over the SAME window (S&P 500 membership checked by date)
  exits      PRIMARY: hold 60 trading days.
             context: hold 20; "gap fill" = sell at the first close back at
             or above the pre-dip price (close of d-10), max 60 days
  cost       10 / 20 / 30 bps round trip on S&P 500 / 400 / 600

TESTS
  MAIN       INSIDER minus CONTROL, 60-day hold. Day averages -> month
             averages per group, compared with a two-sample t.
  LEVEL      INSIDER dips net of cost vs the index average
  DIP ADDS?  INSIDER dips vs insider buys WITHOUT a dip (10-day return
             > -2%). If equal, the edge is just "insiders buy", a published
             effect, not the dip idea.
  CLUSTER    INSIDER events with >= 2 different insiders (context)

PASS (S&P 500, membership-gated, decides):
  MAIN spread > 0 with t > 2 in BOTH halves (2010-2017, 2018-2026)
  AND LEVEL > 0 with t > 2 over the full period
  AND MAIN spread > 0 on S&P 400 and S&P 600 (same sign)
  Pass -> prereg'd genome + gauntlet + forward paper.

SURVIVORSHIP: SEC data includes dead companies; Yahoo prices miss ~59% of
dead S&P 500 names. Missing dead names mostly belong to CONTROL (dips with
no insider confidence), which flatters CONTROL and SHRINKS the spread. A
spread that survives is conservative. The LEVEL test is not protected.

Controls (2026-10-25): planted "30% of dips get an insider buy and those
rebound +10%" -> MAIN t +4.5 / +5.1 per half, PASS. Same data, no plant
-> MAIN t +1.1, LEVEL t +0.04, FAIL. (A +4% plant was only partly
detected: per-trade noise over 60 days is ~17%, so small real effects
need many events.)

Run (repo root):
    python3.10 scripts/insider_dip_test.py
    python3.10 scripts/insider_dip_test.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dip_features as DF  # noqa: E402

ROOT = os.path.dirname(HERE)
BUYS = os.path.join(ROOT, "data", "insider_buys.csv.gz")
DIP, LOOK, RECENT, QUIET, GAP = -0.10, 10, 5, 20, 20
H_MAIN, H_ALT = 60, 20
SPLIT = pd.Timestamp("2018-01-01")
START = pd.Timestamp("2010-03-01")


def ew_index(data, mem):
    """Equal-weight daily index of members -> level series."""
    rets = []
    for sym, df in data.items():
        df = df[~df.index.duplicated()]
        r = df["close"].pct_change()
        if mem:
            r = r[mem(sym, r.index)]
        rets.append(r.clip(-0.5, 1.0).rename(sym))
    m = pd.concat(rets, axis=1).mean(axis=1).fillna(0)
    return (1 + m).cumprod()


def declutter(days, gap):
    keep, last = [], -10 ** 9
    for d in days:
        if d - last >= gap:
            keep.append(d)
            last = d
    return keep


def events(data, mem, buys, ew):
    rows = []
    by_tk = {t: g for t, g in buys.groupby("ticker")}
    for sym, df in data.items():
        df = df[~df.index.duplicated()]
        c, idx, n = df["close"].values, df.index, len(df)
        if n < 100:
            continue
        r10 = np.r_[np.full(LOOK, np.nan), c[LOOK:] / c[:-LOOK] - 1]
        member = mem(sym, idx) if mem else np.ones(n, bool)
        ok = member & (idx >= START)
        # trading-day position of each insider filing for this stock
        g = by_tk.get(sym)
        fpos = np.array([], int)
        nown = np.array([], int)
        if g is not None:
            f = g.groupby("filing_date")["owner_cik"].nunique()
            p = idx.searchsorted(f.index, side="left")
            m = p < n
            fpos, nown = p[m], f.values[m]
        filed = np.zeros(n + 1, int)
        np.add.at(filed, fpos, 1)
        cum = np.cumsum(filed)                    # filings at positions <= i
        owners = np.zeros(n)
        np.add.at(owners, fpos, nown)

        def recent(i, k):                         # filings in [i-k, i-1]
            lo = max(i - k, 0)
            return cum[i - 1] - (cum[lo - 1] if lo > 0 else 0) if i > 0 \
                else 0

        cand = {"INSIDER": [], "CONTROL": [], "INSIDER_NODIP": []}
        for i in np.where(ok & np.isfinite(r10))[0]:
            if i + 1 >= n:
                continue
            ins = recent(i, RECENT) > 0
            if r10[i] <= DIP:
                if ins:
                    cand["INSIDER"].append(i)
                elif recent(i, QUIET) == 0:
                    cand["CONTROL"].append(i)
            elif ins and r10[i] > -0.02:
                cand["INSIDER_NODIP"].append(i)
        ewv = ew.reindex(idx, method="ffill").values
        for grp, days in cand.items():
            for i in declutter(days, GAP):
                lo = max(i - RECENT, 0)
                n_ins = owners[lo:i].sum()
                pre = c[i - LOOK]
                rec = {"date": idx[i], "sym": sym, "group": grp,
                       "cluster": n_ins >= 2}
                for H in (H_ALT, H_MAIN):
                    x = min(i + H, n - 1)
                    rec[f"adj{H}"] = (c[x] / c[i] - 1) - (ewv[x] / ewv[i] - 1)
                    rec[f"full{H}"] = x == i + H
                x = i + 1
                while x < min(i + H_MAIN, n - 1) and c[x] < pre:
                    x += 1
                rec["adj_gap"] = (c[x] / c[i] - 1) - (ewv[x] / ewv[i] - 1)
                rec["gap_days"] = x - i
                rows.append(rec)
    return pd.DataFrame(rows)


def monthly(e, col):
    d = e.groupby("date")[col].mean()
    return d.groupby(d.index.to_period("M")).mean()


def level(e, col, cost):
    m = monthly(e.assign(**{col: e[col] - cost}), col)
    if len(m) < 6:
        return np.nan, np.nan, len(e)
    return m.mean() * 1e4, m.mean() / (m.std(ddof=1) / np.sqrt(len(m))), \
        len(e)


def spread(a, b, col):
    ma, mb = monthly(a, col), monthly(b, col)
    if len(ma) < 6 or len(mb) < 6:
        return np.nan, np.nan
    s = ma.mean() - mb.mean()
    return s * 1e4, s / np.hypot(ma.std(ddof=1) / np.sqrt(len(ma)),
                                 mb.std(ddof=1) / np.sqrt(len(mb)))


def run_ground(name, a, buys):
    if a.smoke:
        data, mem, cost = a.smoke_data[0], None, 0.0020
    else:
        data, _spy, mem, cost = DF.load(name, a)
    ew = ew_index(data, mem)
    ev = events(data, mem, buys, ew)
    ins, ctl = ev[ev.group == "INSIDER"], ev[ev.group == "CONTROL"]
    nod = ev[ev.group == "INSIDER_NODIP"]
    print(f"\n  ===== {name.upper()} ({cost * 1e4:.0f} bps RT) -- insider dips "
          f"{len(ins)} | control dips {len(ctl)} | insider no-dip {len(nod)} "
          f"=====")
    out = {}
    print(f"  {'test':<34} {'all':>16} {'2010-17':>16} {'2018-26':>16}")
    for label, col in (("MAIN spread, 60d", f"adj{H_MAIN}"),
                       ("spread, 20d (context)", f"adj{H_ALT}"),
                       ("spread, gap-fill exit (context)", "adj_gap")):
        cells = []
        for part in (slice(None), "tr", "te"):
            f = (lambda e: e) if part == slice(None) else \
                (lambda e, p=part: e[e.date < SPLIT] if p == "tr"
                 else e[e.date >= SPLIT])
            s, t = spread(f(ins), f(ctl), col)
            cells.append((s, t))
        out[col] = cells
        print(f"  {label:<34} " + " ".join(f"{s:+7.1f} t{t:+5.2f}"
                                           for s, t in cells))
    lv = level(ins, f"adj{H_MAIN}", cost)
    cl = level(ins[ins.cluster], f"adj{H_MAIN}", cost)
    nd = spread(ins, nod, f"adj{H_MAIN}")
    gd = ins["gap_days"].median() if len(ins) else np.nan
    print(f"  LEVEL insider dips, 60d net vs index: {lv[0]:+.1f} bps "
          f"t {lv[1]:+.2f} (n {lv[2]})")
    print(f"  CLUSTER (2+ insiders), 60d net:      {cl[0]:+.1f} bps "
          f"t {cl[1]:+.2f} (n {cl[2]})")
    print(f"  DIP ADDS? insider-dip minus insider-no-dip, 60d: "
          f"{nd[0]:+.1f} bps t {nd[1]:+.2f}")
    print(f"  gap-fill exit: median {gd:.0f} trading days to get back to "
          f"the pre-dip price (capped at {H_MAIN})")
    return out, lv


def smoke_data():
    """Planted: 30% of dips get an insider buy 2 days later, and those
    rebound +10% vs peers over 60 days; the rest don't."""
    idx = pd.bdate_range("2009-01-01", "2026-05-29")
    rng = np.random.default_rng(8)
    data, buys = {}, []
    for k in range(80):
        r = rng.normal(0.0003, 0.022, len(idx))
        c = 100 * np.exp(np.cumsum(r))
        r10 = np.r_[np.full(10, np.nan), c[10:] / c[:-10] - 1]
        i = 300
        while i < len(idx) - 80:
            if r10[i] <= -0.10 and rng.random() < 0.3:
                buys.append((idx[i + 2], f"S{k}", f"o{k}{i}"))
                r[i + 4:i + 64] += 0.10 / 60
                c = 100 * np.exp(np.cumsum(r))
                r10 = np.r_[np.full(10, np.nan), c[10:] / c[:-10] - 1]
                i += 30
            i += 1
        data[f"S{k}"] = pd.DataFrame({"open": c, "high": c, "low": c,
                                      "close": c, "volume": 1e6}, idx)
    b = pd.DataFrame(buys, columns=["filing_date", "ticker", "owner_cik"])
    return data, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.smoke_data = smoke_data()
        buys = a.smoke_data[1]
        grounds = ["sp500"]
    else:
        if not os.path.exists(BUYS):
            sys.exit("run fetch_insider_buys.py first")
        buys = pd.read_csv(BUYS, parse_dates=["filing_date"])
        print(f"  {len(buys)} insider purchases, {buys.ticker.nunique()} "
              f"tickers")
        grounds = ["sp500", "sp400", "sp600"]
    res = {g: run_ground(g, a, buys) for g in grounds}

    m = res["sp500"][0][f"adj{H_MAIN}"]
    lv = res["sp500"][1]
    ok_main = all(np.isfinite(t) and s > 0 and t > 2 for s, t in m[1:])
    ok_lvl = np.isfinite(lv[1]) and lv[0] > 0 and lv[1] > 2
    reps = {g: res[g][0][f"adj{H_MAIN}"][0][0] for g in grounds
            if g != "sp500"}
    ok_rep = all(v > 0 for v in reps.values())
    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"  MAIN spread > 0, t > 2 in both halves (S&P 500): "
          f"{'YES' if ok_main else 'no'}")
    print(f"  LEVEL > 0, t > 2 (S&P 500):                     "
          f"{'YES' if ok_lvl else 'no'}")
    for g, v in reps.items():
        print(f"  replication {g}: spread {v:+.1f} bps "
              f"{'(same sign)' if v > 0 else '(WRONG sign)'}")
    if a.smoke:
        print("  smoke: planted signal on one synthetic index (pipe check)")
    if ok_main and ok_lvl and ok_rep:
        print("  PASS -- insider buying separates rebounding dips. Next: "
              "prereg'd genome + gauntlet + forward paper.")
    else:
        print("  FAIL -- insider buying does not reliably separate dips "
              "after costs on this data.")


if __name__ == "__main__":
    main()
