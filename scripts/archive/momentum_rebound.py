"""
momentum_rebound.py -- Blake's EXACT stop-and-buy-back idea on top of the
30% yearly lock, tested in Primordial (QuantConnect untouched).

"If it falls 20-25%, shut the algo off, monitor the price dropping, then
when volume spikes, the price comes back above its moving average, the
lowest point holds and price climbs off it, and ADX points to trending up,
buy back in and ride the rebound until the 30% limit, then the ETFs."

EVERYTHING IS MEASURED ON THE MOMENTUM BOOK ITSELF (the top 20 we sold and
keep watching), rebuilt here month by month exactly like momentum_v2.py
STEADY, with its daily returns AND its volume:

  STOP      account falls 20% (or 25%) from its highest close since it last
            bought in -> sell everything, cash (BIL).
  BUY BACK  all four on the same close:
    1 VOLUME SPIKE  at some day since the stop, the book's volume was >= 2x
                    normal (average over the 20 stocks of today's volume /
                    its own prior 50-day average)
    2 LOW HELD      the book's lowest close since the stop is at least 10
                    trading days old, and price is >= 5% above it
    3 ABOVE SMA     price above its 20-day average
    4 ADX UP        ADX(14) >= 20 and +DI > -DI. The book has closes only,
                    so ADX uses the close-to-close move as the day's range
                    (standard close-only ADX).
  LOCK      account up 30% this calendar year -> 10 SPDR sector ETFs until
            Jan 1 (unchanged from the QC winner).
  Switching costs 0.2% each time.

PRE-STATED (thresholds fixed before running; no retries)
  PRIMARY LOCK30+R20 must beat LOCK30 on BOTH Sharpe and worst drop.
  LOCK30+R25 must also beat LOCK30 on both, or the pass is called fragile.
  Note: Primordial's LOCK30 (fills at the close) runs rosier than QC's
  (25.2%, -42.9%); compare books only against each other here.

Run (repo root; ~2-4 minutes):
    python3.10 scripts/momentum_rebound.py --smoke
    python3.10 scripts/momentum_rebound.py | tee logs/momentum_rebound.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_us as U           # noqa: E402
import momentum_v2 as M         # noqa: E402

COST = 0.002
RVOL_SPIKE, RVOL_DAYS = 2.0, 50
LOW_AGE, LOW_BOUNCE = 10, 0.05
SMA_DAYS, ADX_N, ADX_MIN = 20, 14, 20
START, END = pd.Timestamp("2017-03-01"), pd.Timestamp("2026-06-01")


def build_book(px, vol, members):
    """STEADY top-20 book: daily returns and daily relative volume."""
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    pxf = px.ffill()
    vavg = vol.rolling(RVOL_DAYS, min_periods=30).mean().shift(1)
    rets, rvols, prev = [], [], set()
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
        v = r[elig].iloc[t_i - 251:t_i + 1].std() * np.sqrt(252)
        sc = (mom / v.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        pick = list(sc.index[:M.TOP])
        seg = pxf.loc[t:t2, pick]
        d = (seg / seg.iloc[0]).mean(axis=1).pct_change().iloc[1:]
        if len(d):
            d.iloc[0] -= len(set(pick) ^ prev) / M.TOP * M.COST
        prev = set(pick)
        rv = (vol.loc[t:t2, pick] / vavg.loc[t:t2, pick]).iloc[1:]
        rets.append(d)
        rvols.append(rv.mean(axis=1))
    return pd.concat(rets), pd.concat(rvols)


def close_adx(s, n=ADX_N):
    """ADX from closes only: the day's range = |close-to-close move|."""
    dlt = s.diff()
    a = 1.0 / n
    tr = dlt.abs().ewm(alpha=a, adjust=False).mean()
    pdi = 100 * dlt.clip(lower=0).ewm(alpha=a, adjust=False).mean() / tr
    mdi = 100 * (-dlt).clip(lower=0).ewm(alpha=a, adjust=False).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=a, adjust=False).mean(), pdi, mdi


def simulate(m, rvol, sec, cash, lock, stop):
    idx = m.index
    S = (1 + m).cumprod()
    sma = S.rolling(SMA_DAYS, min_periods=SMA_DAYS).mean()
    adx, pdi, mdi = close_adx(S)
    A, mode, cur_year = 1.0, "MOM", idx[0].year
    year_start, peak = 1.0, 1.0
    low = low_i = None
    spiked = False
    out, ev, pend = [], [], 0.0
    seen = np.zeros(5, int)          # days in cash, and days each signal was on
    for i, d in enumerate(idx):
        if d.year != cur_year:
            cur_year, year_start = d.year, A
            if mode == "SEC":
                mode, pend, peak = "MOM", COST, A
        rr = {"MOM": m.iloc[i], "SEC": sec.iloc[i], "CASH": cash.iloc[i]}[mode]
        A *= (1 + rr) * (1 - pend)
        pend = 0.0
        out.append(A)
        s = S.iloc[i]
        if mode == "MOM":
            peak = max(peak, A)
            if stop and A / peak - 1 <= -stop:
                mode, pend = "CASH", COST
                low, low_i, spiked, out_day = s, i, False, d
                ev.append((d, f"down {A/peak-1:.0%} from peak -> cash"))
            elif lock and A / year_start - 1 >= lock:
                mode, pend = "SEC", COST
                ev.append((d, f"up {A/year_start-1:.0%} this year -> sector ETFs"))
        elif mode == "CASH":
            if s < low:
                low, low_i = s, i
            if np.isfinite(rvol.iloc[i]) and rvol.iloc[i] >= RVOL_SPIKE:
                spiked = True
            c1 = spiked
            c2 = (i - low_i) >= LOW_AGE and s >= low * (1 + LOW_BOUNCE)
            c3 = np.isfinite(sma.iloc[i]) and s > sma.iloc[i]
            c4 = adx.iloc[i] >= ADX_MIN and pdi.iloc[i] > mdi.iloc[i]
            seen += np.array([1, c1, c2, c3, c4], int)
            if c1 and c2 and c3 and c4:
                mode, peak, pend = "MOM", A, COST
                ev.append((d, f"back in after {(d - out_day).days} days: book "
                              f"{s/low-1:+.0%} off its low, ADX {adx.iloc[i]:.0f}"))
    eq = pd.Series(out, idx)
    return eq.pct_change().fillna(eq.iloc[0] - 1), ev, (mode, seen)


def stats(r):
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    v = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    tr = dd.idxmin()
    return ann, v, r.mean() * 252 / v, dd.min(), eq.loc[:tr].idxmax(), tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy = M.smoke()
        feed = "sip"
    else:
        px, vol, spy, feed = U.load()
    U.MIN_PX = M.MIN_PX
    min_dv = M.MIN_DV if feed == "sip" else M.MIN_DV * 0.03
    sets, ever = U.liquid_members(px, vol, min_dv)
    px, vol = px[ever], vol[ever]
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731
    m, rvol = build_book(px, vol, members)
    keep = (m.index >= START) & (m.index < END)
    m, rvol = m[keep], rvol.reindex(m.index)[keep]
    if a.smoke:
        sec, cash, note = m * 0.5, pd.Series(0.0, m.index), "synthetic"
    else:
        import momentum_overlay as O   # noqa: E402
        _, sec, cash, note = O.load(False)
        sec, cash = sec.reindex(m.index).fillna(0), cash.reindex(m.index).fillna(0)
    print(f"  {m.index[0].date()} -> {m.index[-1].date()}, {len(m)} days | {note}")
    print(f"  book volume spikes (>= {RVOL_SPIKE}x normal): "
          f"{(rvol >= RVOL_SPIKE).sum()} days")

    books = {"PLAIN": (None, None), "LOCK30": (0.30, None),
             "LOCK30+R20": (0.30, 0.20), "LOCK30+R25": (0.30, 0.25)}
    R, E, END_MODE = {}, {}, {}
    for k, (lk, st) in books.items():
        R[k], E[k], END_MODE[k] = simulate(m, rvol, sec, cash, lk, st)

    print(f"\n  {'book':<11} {'annual':>7} {'vol':>6} {'sharpe':>6} {'worst':>7}"
          f"   worst drop from -> to")
    S = {}
    for k, r in R.items():
        a_, v, s, dd, pk, tr = stats(r)
        S[k] = (s, dd)
        print(f"  {k:<11} {a_*100:>+6.1f}% {v*100:>5.1f}% {s:>6.2f} {dd*100:>6.1f}%"
              f"   {pk:%Y-%m-%d} -> {tr:%Y-%m-%d}")

    cols = list(books)
    print("\n  year by year (worst drop inside the year in brackets)")
    print("  year  " + "".join(f"{c:>20}" for c in cols))
    for y in sorted(set(m.index.year)):
        cells = []
        for c in cols:
            g = R[c][R[c].index.year == y]
            eq = (1 + g).cumprod()
            cells.append(f"{(eq.iloc[-1]-1)*100:+5.0f}% ({(eq/eq.cummax()-1).min()*100:4.0f}%)")
        print(f"  {y}  " + "".join(f"{c:>20}" for c in cells))

    for c in ["LOCK30+R20", "LOCK30+R25"]:
        print(f"\n  {c} switches:")
        for d, e in E[c]:
            print(f"    {d:%Y-%m-%d}  {e}")
        mode, seen = END_MODE[c]
        if seen[0]:
            print(f"    {seen[0]} days in cash; signal on: volume spike seen {seen[1]}, "
                  f"low held {seen[2]}, above SMA {seen[3]}, ADX up {seen[4]}")
        if mode == "CASH":
            print("    (still in cash at the end -- buy-back never triggered)")

    (s0, d0), (s1, d1), (s2, d2) = S["LOCK30"], S["LOCK30+R20"], S["LOCK30+R25"]
    ok, robust = s1 > s0 and d1 > d0, s2 > s0 and d2 > d0
    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"  LOCK30+R20: Sharpe {s1:.2f} vs {s0:.2f}, worst {d1*100:.1f}% vs "
          f"{d0*100:.1f}% -> {'PASS' if ok else 'FAIL'}")
    print(f"  LOCK30+R25: Sharpe {s2:.2f}, worst {d2*100:.1f}% -> "
          f"{'also beats LOCK30' if robust else 'does NOT beat LOCK30'}")
    print("  " + ("port to QC" if ok and robust else
                  "fragile -- do not port" if ok else "keep LOCK30 as is"))
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")


if __name__ == "__main__":
    main()
