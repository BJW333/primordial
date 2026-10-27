"""
fade_predict.py -- after a good-news attention spike, can we tell WHICH
stocks will fade (and which will keep running) BEFORE we bet?

EVENTS: good-news Wikipedia spikes on non-S&P liquid small + mid caps
(attention_rebound.py). DECISION DAY = 21 trading days after the spike
(lets the first insider filings and price action come in). Enter at the
close of day 22, hold to day 252. Return minus the same-size equal-weight
benchmark.

FLAGS (all known on decision day):
  S  INSIDER_SELL  an officer/director open-market sale >= $25k filed between
                   the spike and decision day ("insiders don't believe it")
  D  DILUTION      shares outstanding up >= 5% vs a year earlier (latest
                   SEC filing known on decision day) ("company sells stock")
  W  WEAK_TREND    12-month return below the same-size average
                   (hype without a trend -- tonight's momentum finding)
  G  GIVEBACK      already back below the pre-spike close (the pop failed)
  R  RUNUP         up > 50% in the 3 months BEFORE the spike (overextended)
  Rules: each flag alone and every pair (both true) = 15 rules.

PRE-STATED
  On 2017-2021-05 spikes ONLY (small + mid pooled): pick the rule whose
  flagged group has the most negative t (>= 50 trades).
  On 2021-06 -> spikes that already have 252 days of data: PASS if
    (1) flagged group mean < 0 with t < -2.5, AND
    (2) flagged fades MORE than the unflagged spikes: monthly difference
        (flagged minus unflagged) t < -2.0.
  t = monthly averages, 3-lag Newey-West.

PART 2 -- STACKED, CHEAP-TO-BORROW SHORT (ideas 1, 2, 4; fixed in advance,
no selection): MID caps in the MOST LIQUID half of MID (63-day dollar
volume at the spike >= the MID median) -- names that are almost always
easy to borrow -- with borrow 0.5%/yr + 0.4% trading. Three rules:
  WEAK_TREND (idea 1), DILUTION (idea 2), WEAK_TREND+DILUTION (stacked).
  Each PASSES if the estimated net short return per trade is > 0 in BOTH
  periods AND the fade t < -2.5 in 2021-06+.
Also printed (for put options, which pay on ABSOLUTE drops): share of
trades down >= 20% and >= 40% in absolute price over the hold; and an
estimated net short return after borrow (MID 3%, SMALL 6%/yr) + 0.4%
trading.

Run (repo root):
    python3.10 scripts/fetch_insider_sales.py --email you@example.com
    python3.10 scripts/fade_predict.py --smoke
    python3.10 scripts/fade_predict.py | tee logs/fade_predict.txt
"""
import argparse
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attention_rebound as AR        # noqa: E402
import attention_short_test as AS     # noqa: E402
import blowup_scan as B               # noqa: E402
import factor_portfolio as FP         # noqa: E402

ROOT = os.path.dirname(HERE)
SALES = os.path.join(ROOT, "data", "insider_sales.csv.gz")
SHARES = os.path.join(ROOT, "data", "shares_outstanding.csv.gz")
FLAGS = ["S", "D", "W", "G", "R"]
NAMES = {"S": "INSIDER_SELL", "D": "DILUTION", "W": "WEAK_TREND",
         "G": "GIVEBACK", "R": "RUNUP"}
RULES = [(f,) for f in FLAGS] + list(itertools.combinations(FLAGS, 2))
DEC, END = 21, 252
BORROW = {"SMALL": 0.06, "MID": 0.03}
MINN = 50
CLIP = True


def norm(s):
    return s.astype(str).str.upper().str.strip().str.replace(
        "-", ".", regex=False)


def load_sales():
    if not os.path.exists(SALES):
        print("  (no insider_sales.csv.gz -- run fetch_insider_sales.py; "
              "INSIDER_SELL will be all False)")
        return {}
    df = pd.read_csv(SALES, parse_dates=["filing_date"])
    df["ticker"] = norm(df["ticker"])
    return {k: np.sort(g["filing_date"].values) for k, g in
            df.groupby("ticker")}


def load_shares():
    if not os.path.exists(SHARES):
        print("  (no shares_outstanding.csv.gz -- DILUTION all False)")
        return {}
    df = pd.read_csv(SHARES, dtype={"ticker": str},
                     parse_dates=["filed"]).dropna(subset=["ticker",
                                                           "shares"])
    df["ticker"] = norm(df["ticker"])
    df = df[df["shares"] > 0].sort_values("filed")
    return {k: (g["filed"].values, g["shares"].values)
            for k, g in df.groupby("ticker")}


def dilution(sh, s, T):
    x = sh.get(s)
    if x is None:
        return np.nan
    f, v = x
    T = np.datetime64(T)
    now = (f <= T) & (f > T - np.timedelta64(200, "D"))
    then = (f <= T - np.timedelta64(330, "D")) & \
        (f > T - np.timedelta64(550, "D"))
    if not now.any() or not then.any():
        return np.nan
    return v[now][-1] / v[then][-1] - 1


def build(name, ev, px, bench, sales, sh, mem, dv):
    idx = px.index
    pxf = px.ffill()
    b = bench.reindex(idx).ffill().values
    cols = {c: j for j, c in enumerate(px.columns)}
    cv = pxf.values
    rows = []
    for s, d in zip(ev["sym"], ev["d"]):
        p0 = idx.get_loc(d)
        t, e, x = p0 + DEC, p0 + DEC + 1, p0 + END
        if x >= len(idx) or p0 < 64 or t < 252:
            continue
        c = cv[:, cols[s]]
        if not (np.isfinite(c[e]) and c[e] > 0 and np.isfinite(c[x])):
            continue
        T = idx[t]
        f = sales.get(s)
        sold = bool(f is not None and ((f > np.datetime64(d)) &
                                       (f <= np.datetime64(T))).any())
        dil = dilution(sh, s, T)
        liq = False
        if dv is not None:
            live = [m for m in mem(d) if m in dv.columns]
            liq = bool(dv.at[d, s] >= dv.loc[d, live].median())
        rows.append({
            "u": name, "d": d, "liq": liq, "sym": s, "e": e, "x": x,
            "S": sold,
            "D": bool(np.isfinite(dil) and dil >= 0.05),
            "has_sh": bool(np.isfinite(dil)),
            "W": bool(c[t] / c[t - 252] - 1 < b[t] / b[t - 252] - 1),
            "G": bool(c[t] <= c[p0 - 1]),
            "R": bool(c[p0 - 1] / c[p0 - 64] - 1 > 0.5),
            "r": c[x] / c[e] - b[x] / b[e],
            "abs": c[x] / c[e] - 1})
    df = pd.DataFrame(rows)
    return df[df["r"].abs() < 3] if CLIP else df


def flagged(df, rule):
    m = np.ones(len(df), bool)
    for f in rule:
        m &= df[f].values
    return m


def row_stats(g):
    if len(g) < 5:
        return len(g), np.nan, np.nan, np.nan, np.nan
    return (len(g), g["r"].mean(), AR.tstat(g[["d", "r"]]),
            (g["abs"] <= -0.2).mean(), (g["abs"] <= -0.4).mean())


def diff_t(df, m):
    a = df[m].groupby(df[m]["d"].dt.to_period("M"))["r"].mean()
    b = df[~m].groupby(df[~m]["d"].dt.to_period("M"))["r"].mean()
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    return (j["a"] - j["b"]).mean(), B.nw_t(j["a"] - j["b"], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, views, unis = AR.smoke()
        sales, sh = {}, {}
        dv = (px * vol).rolling(63, min_periods=40).mean()
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = AR.universes(sp_members)
        sales, sh = load_sales(), load_shares()
        dv = (px * vol).rolling(63, min_periods=40).mean()
        print(f"  insider-sale histories {len(sales)} tickers, share-count "
              f"histories {len(sh)} tickers")
    parts = []
    for name, (mem, bench) in unis.items():
        ev = AR.good_spikes(views, px, mem)
        parts.append(build(name, ev, px, bench, sales, sh, mem, dv))
    df = pd.concat(parts, ignore_index=True)
    old, new = df[df["d"] < AR.RECENT], df[df["d"] >= AR.RECENT]
    print(f"\n  spikes with a full year after: 2017-21 {len(old)}, "
          f"2021-06+ {len(new)} | share-count coverage "
          f"{df['has_sh'].mean()*100:.0f}%")
    print("  flag rates: " + "  ".join(
        f"{NAMES[f]} {df[f].mean()*100:.0f}%" for f in FLAGS))
    base0, base1 = row_stats(old), row_stats(new)
    print(f"  ALL spikes, day 22 -> 252 vs benchmark: 2017-21 "
          f"{base0[1]*100:+.2f}% t {base0[2]:+.2f} | 2021-06+ "
          f"{base1[1]*100:+.2f}% t {base1[2]:+.2f}")

    print(f"\n  {'rule':<26} | {'2017-21 n':>9} {'mean':>7} {'t':>6} "
          f"{'-20%':>5} | {'2021+ n':>7} {'mean':>7} {'t':>6} {'-20%':>5} "
          f"{'-40%':>5}")
    sel = []
    for rule in RULES:
        lab = "+".join(NAMES[f] for f in rule)
        s0 = row_stats(old[flagged(old, rule)])
        s1 = row_stats(new[flagged(new, rule)])
        sel.append((rule, s0))
        print(f"  {lab:<26} | {s0[0]:>9} {s0[1]*100:>+6.2f}% {s0[2]:>+6.2f} "
              f"{s0[3]*100:>4.0f}% | {s1[0]:>7} {s1[1]*100:>+6.2f}% "
              f"{s1[2]:>+6.2f} {s1[3]*100:>4.0f}% {s1[4]*100:>4.0f}%")
    ok = [x for x in sel if x[1][0] >= MINN and np.isfinite(x[1][2])]
    if not ok:
        print("  no rule with enough trades")
        return
    rule, s0 = min(ok, key=lambda x: x[1][2])
    lab = " + ".join(NAMES[f] for f in rule)
    m1 = flagged(new, rule)
    s1 = row_stats(new[m1])
    dm, dt = diff_t(new, m1)
    p = bool(s1[0] >= MINN and s1[1] < 0 and s1[2] < -2.5 and dt < -2.0)
    print(f"\n  SELECTED on 2017-21 (most negative t): {lab} "
          f"(n {s0[0]}, {s0[1]*100:+.2f}%, t {s0[2]:+.2f})")
    print(f"  2021-06+: flagged n {s1[0]}, {s1[1]*100:+.2f}% t {s1[2]:+.2f}; "
          f"flagged minus unflagged {dm*100:+.2f}%/trade t {dt:+.2f}")
    for u in ("SMALL", "MID"):
        g = new[m1 & (new["u"] == u).values]
        if len(g) >= 10:
            hold = (END - DEC) / 252
            net = -g["r"].mean() - BORROW[u] * hold - 0.004
            print(f"    {u:<5} flagged n {len(g):>4}: fade "
                  f"{g['r'].mean()*100:+.2f}% -> est. short after borrow+"
                  f"trading {net*100:+.2f}% per trade ({hold*12:.0f} months); "
                  f"down >=20% abs {(g['abs'] <= -0.2).mean()*100:.0f}%, "
                  f">=40% {(g['abs'] <= -0.4).mean()*100:.0f}%")
    verdict = {f"fade predictor ({lab})": p}

    uni = "MID" if (df["u"] == "MID").any() else df["u"].iloc[0]
    hold = (END - DEC) / 252
    print(f"\n  ===== PART 2: liquid {uni} caps only, borrow 0.5%/yr + 0.4% "
          f"trading, short {hold*12:.0f} months =====")
    L = df[(df["u"] == uni) & df["liq"]]
    print(f"  liquid {uni} spikes: {len(L)} (all of them shown first)")
    for rule in [(), ("W",), ("D",), ("W", "D"), ("S",)]:
        lab2 = "+".join(NAMES[f] for f in rule) or "ALL (no flag)"
        cells, nets, ts = [], [], []
        for per, g in (("2017-21", L[L["d"] < AR.RECENT]),
                       ("2021-06+", L[L["d"] >= AR.RECENT])):
            g = g[flagged(g, rule)] if rule else g
            n, m, tt, d20, _ = row_stats(g)
            net = -m - 0.005 * hold - 0.004
            nets.append(net)
            ts.append(tt)
            cells.append(f"{per} n {n:>4} fade {m*100:>+6.2f}% t {tt:>+5.2f}"
                         f" net short {net*100:>+6.2f}% (-20% abs "
                         f"{d20*100:>3.0f}%)")
        print(f"  {lab2:<22} " + " | ".join(cells))
        if rule in [("W",), ("D",), ("W", "D")]:
            verdict[f"liquid short {lab2}"] = bool(
                np.isfinite(nets).all() and min(nets) > 0 and ts[1] < -2.5)

    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    for k, v in verdict.items():
        print(f"  {k:<42} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
