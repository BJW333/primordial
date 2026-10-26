"""
momentum_v2.py -- momentum on ALL liquid US stocks, fixed three ways after
blowup_us.py showed raw momentum there loses money (-6.8%/yr last 5y)
while still finding 3x more blow-ups than average:
  1. NO JUNK:      price >= $10 and >= $20M/day traded (point-in-time)
  2. STEADY GAIN:  rank by 12-1 return / its daily volatility, so a stock
                   that climbed steadily beats a lottery spike
  3. RIDE & SELL:  checked every day, a stock is sold once it closes 20%
                   below its highest close since we bought it; the cash
                   waits until the next monthly rebalance

VARIANTS (top 20, equal weight, monthly rebalance, 20 bps per trade --
small/mid caps cost more to trade than the S&P 500's 10 bps)
  RAW        raw 12-1 momentum, monthly hold (old way, cleaner universe)
  STEADY     steady-gain momentum, monthly hold
  STEADY_TS  steady-gain momentum + 20% trailing stop -- PRIMARY
  HIGH52     closest to its 52-week high, monthly hold

PRE-STATED (primary STEADY_TS, last five years 2021-06 -> 2026-05)
  PASS: annual return beats SPY leveraged to the same volatility (4%/yr
  borrow) AND Sharpe above SPY's. 2017-2021 shown as context.
  Pass -> forward paper trading.

Run (repo root; needs data/us_universe/ from build_us_universe.py):
    python3.10 scripts/momentum_v2.py --smoke
    python3.10 scripts/momentum_v2.py | tee logs/momentum_v2.txt
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_scan as B      # noqa: E402
import blowup_us as U        # noqa: E402
import turbo_portfolio as T  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "momentum_v2_returns.csv")
FIRST = pd.Timestamp("2017-02-28")
RECENT = pd.Timestamp("2021-06-01")
MIN_PX, MIN_DV = 10.0, 20e6
TOP, COST, STOP, LEV_COST = 20, 0.0020, 0.20, 0.04
VARIANTS = ["RAW", "STEADY", "STEADY_TS", "HIGH52"]
PRIMARY = "STEADY_TS"


def month_path(px, pick, t, t2, peaks, stop):
    """Daily ride of each pick from t to t2. Returns (per-slot returns,
    stopped set, updated peaks). Peak is tracked since purchase."""
    seg = px.loc[t:t2, pick].ffill()
    base = seg.iloc[0]
    rets, stopped, new_peaks = {}, set(), {}
    for s in pick:
        p = seg[s].values
        if not np.isfinite(base[s]):
            rets[s] = 0.0
            continue
        pk = max(peaks.get(s, base[s]), base[s])
        out = None
        if stop:
            for v in p[1:]:
                if not np.isfinite(v):
                    continue
                pk = max(pk, v)
                if v <= pk * (1 - stop):
                    out = v
                    break
        if out is not None:
            rets[s] = out / base[s] - 1
            stopped.add(s)
        else:
            last = p[np.isfinite(p)][-1] if np.isfinite(p).any() else base[s]
            rets[s] = last / base[s] - 1
            if stop:
                pk = max(pk, np.nanmax(p))
            new_peaks[s] = pk
    return rets, stopped, new_peaks


def run(px, spy, members):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= FIRST]
    r = px.pct_change(fill_method=None)
    pxf = px.ffill()
    rows, hits = [], []
    prev = {v: set() for v in VARIANTS}
    peaks, ts_held = {}, set()
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        if t_i < 260:
            continue
        mem = members(t)
        win = px.iloc[t_i - 252:t_i + 1]
        cnt = win.notna().sum()
        elig = [s for s in mem if cnt.get(s, 0) >= 240]
        if len(elig) < 100:
            continue
        W = win[elig]
        mom = W.iloc[-22] / W.iloc[0] - 1
        vol = r[elig].iloc[t_i - 251:t_i + 1].std() * np.sqrt(252)
        steady = mom / vol.replace(0, np.nan)
        high = W.iloc[-1] / W.max()
        scores = {"RAW": mom, "STEADY": steady, "STEADY_TS": steady,
                  "HIGH52": high + steady.rank(pct=True) * 1e-6}
        spy_m = spy.asof(t2) / spy.asof(t) - 1
        row = {"month": t2, "SPY": spy_m, "n_elig": len(elig)}
        f_i = t_i + 126
        fwd = (pxf[elig].iloc[f_i] / px[elig].iloc[t_i] - 1
               if f_i < len(idx) else None)
        h = {"month": t2}
        if fwd is not None:
            h["base"] = (fwd.dropna() >= 0.5).mean()
        for v in VARIANTS:
            sc = scores[v].replace([np.inf, -np.inf], np.nan).dropna()
            pick = list(sc.sort_values(ascending=False).index[:TOP])
            stop = STOP if v == "STEADY_TS" else 0
            if stop:
                peaks = {s: peaks[s] for s in pick if s in peaks
                         and s in ts_held}
            rets, stopped, newpk = month_path(px, pick, t, t2,
                                              peaks if stop else {}, stop)
            old = prev[v]
            buys = len(set(pick) - old)
            sells = len(old - set(pick))
            trades = buys + sells + len(stopped)
            row[v] = np.mean([rets[s] for s in pick]) - \
                trades / TOP * COST
            row[v + "_stops"] = len(stopped)
            held = set(pick) - stopped
            prev[v] = held
            if stop:
                peaks, ts_held = newpk, held
            if fwd is not None:
                h[v] = (fwd[pick].dropna() >= 0.5).mean()
            B.PICKLOG.extend((t2, v, s, rets[s]) for s in pick)
        rows.append(row)
        hits.append(h)
    return (pd.DataFrame(rows).set_index("month"),
            pd.DataFrame(hits).set_index("month"))


def report(res, hit, label):
    print(f"\n  ===== {label}: {len(res)} months, {res.index[0].date()} -> "
          f"{res.index[-1].date()} | ~{res['n_elig'].mean():.0f} stocks "
          f"eligible/month =====")
    hb = hit.dropna(subset=["base"])
    print(f"  blow-up base rate (up >= 50% in 6 months): "
          f"{hb['base'].mean()*100:.1f}%")
    print(f"\n  {'variant':<10} {'annual':>7} {'vol':>6} {'sharpe':>6} "
          f"{'worst':>7} {'same-risk SPY':>13} {'blow-up lift':>12} "
          f"{'stops/mo':>8}")
    _, spy_v, spy_s, _ = B.stats(res["SPY"])
    out = {}
    for v in VARIANTS:
        a, vol, s, dd = B.stats(res[v])
        lev = vol / spy_v
        m_a = B.stats(lev * res["SPY"] - max(lev - 1, 0) * LEV_COST / 12)[0]
        lift = hb[v].mean() / hb["base"].mean() if len(hb) else np.nan
        st = f"{res[v + '_stops'].mean():>8.1f}" if v == PRIMARY else ""
        print(f"  {v:<10} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
              f"{dd*100:>6.1f}% {m_a*100:>+12.1f}% {lift:>11.2f}x {st}")
        out[v] = (a, s, m_a)
    a, vol, s, dd = B.stats(res["SPY"])
    print(f"  {'SPY':<10} {a*100:>+6.1f}% {vol*100:>5.1f}% {s:>6.2f} "
          f"{dd*100:>6.1f}%")
    print("  by year: " + " | ".join(
        f"{y}: " + " ".join(f"{v[:6]} {((1 + g[v]).prod() - 1)*100:+.0f}%"
                            for v in [PRIMARY, "SPY"])
        for y, g in res.groupby(res.index.year)))
    return out, s


def smoke():
    px, vol, spy, _ = T.smoke()
    px, vol = px * 5, vol * 1000
    # unit check of the stop: a stock that drops 30% mid-month exits ~-20%
    d = pd.bdate_range("2024-01-01", periods=21)
    p = pd.DataFrame({"X": np.r_[np.linspace(100, 110, 8),
                                 np.linspace(108, 77, 13)]}, index=d)
    rr, st, _ = month_path(p, ["X"], d[0], d[-1], {}, STOP)
    assert "X" in st and -0.15 < rr["X"] < -0.05, (rr, st)
    print(f"  stop unit check: peak 110 -> sold at {100*(1+rr['X']):.1f} "
          f"(return {rr['X']*100:+.1f}%) OK")
    return px, vol, spy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy = smoke()
        feed = "sip"
    else:
        px, vol, spy, feed = U.load()
    U.MIN_PX = MIN_PX
    min_dv = MIN_DV if feed == "sip" else MIN_DV * 0.03
    sets, ever = U.liquid_members(px, vol, min_dv)
    px = px[ever]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731
    res, hit = run(px, spy, members)
    if not a.smoke:
        res.to_csv(OUT)
    old = res.index < RECENT
    if old.sum() > 6:
        report(res[old], hit[hit.index < RECENT], "CONTEXT 2017-2021")
    rec, spy_s = report(res[~old], hit[hit.index >= RECENT],
                        "DECIDING: LAST FIVE YEARS")
    a_, s_, m_a = rec[PRIMARY]
    ok = a_ > m_a and s_ > spy_s
    print(f"\n  ===== VERDICT (pre-stated, {PRIMARY}, last 5 years) =====")
    print(f"  return {a_*100:+.1f}% vs same-risk SPY {m_a*100:+.1f}% | "
          f"Sharpe {s_:.2f} vs SPY {spy_s:.2f} -> "
          f"{'PASS -- forward paper trading next' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
