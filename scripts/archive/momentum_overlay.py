"""
momentum_overlay.py -- test CHIMERA's on/off rules on Primordial's own daily
momentum returns, without touching QuantConnect.

INPUT: data/momentum_cap_daily.csv (written by momentum_cap.py): the STEADY
momentum book's daily returns, 2017-03 -> 2026-05 (the same book running on
QC). Sector ETFs (equal weight of the 10 SPDR sectors) from the cached
histories sector_tailwind.py downloaded; cash = BIL if cached, else 0%.

THE RULES (decided at each day's close, applied from the next day; 0.2%
cost every time the account switches)
  LOCK  (Blake, passed on QC): account up 30% this calendar year -> sector
        ETFs until Jan 1, then momentum again.
  STOP  (Blake, new): while in momentum, the account falls 20% from its
        highest close since it last bought in -> sell everything, go to cash.
  BACK IN: watch the momentum book we left (what it WOULD have done). Buy
        back when it closes above its own 50-day average AND is at least 10%
        above its lowest close since the stop. Then ride; the LOCK still
        applies (+30% for the year -> sector ETFs).
  Why no ADX / volume: those need a real traded price with highs, lows and
  volume; the momentum book has only closes. They also failed in every
  rebound/bottom study in Primordial (rebound_signals, rebound2,
  attention_bottom). The SMA reclaim and "off the low" parts of the idea
  are kept.

BOOKS
  PLAIN        momentum always on            (calibration: QC 24.9%, -55.9%)
  LOCK30       the QC winner                 (calibration: QC 25.2%, -42.9%)
  LOCK30+S20   LOCK + STOP at -20%           PRIMARY
  LOCK30+S25   LOCK + STOP at -25%           fragility check

PRE-STATED (settings fixed before running, no retries with other numbers)
  PASS if LOCK30+S20 beats LOCK30 on BOTH Sharpe and worst drop over the
  full period. If LOCK30+S25 does not also beat LOCK30 on both, the pass
  is called fragile and is not ported to QC.

Run (repo root):
    python3.10 scripts/momentum_overlay.py --smoke
    python3.10 scripts/momentum_overlay.py | tee logs/momentum_overlay.txt
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
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "data", "momentum_cap_daily.csv")
SECTORS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE"]
COST, SMA, BOUNCE = 0.002, 50, 0.10


def simulate(m, sec, cash, lock, stop):
    """Daily account path. m/sec/cash: daily returns on one index."""
    idx = m.index
    shadow = (1 + m).cumprod()
    sma = shadow.rolling(SMA, min_periods=SMA).mean()
    A, mode, cur_year = 1.0, "MOM", idx[0].year
    year_start, peak, low = 1.0, 1.0, None
    out, events, pending_cost = [], [], 0.0
    for i, d in enumerate(idx):
        if d.year != cur_year:                     # new year
            cur_year, year_start = d.year, A
            if mode == "SEC":
                mode, pending_cost, peak = "MOM", COST, A
                events.append((d, "new year -> momentum"))
        r = {"MOM": m.iloc[i], "SEC": sec.iloc[i], "CASH": cash.iloc[i]}[mode]
        A *= (1 + r) * (1 - pending_cost)
        pending_cost = 0.0
        out.append(A)
        s = shadow.iloc[i]
        if mode == "MOM":
            peak = max(peak, A)
            if stop and A / peak - 1 <= -stop:
                mode, low, pending_cost = "CASH", s, COST
                events.append((d, f"down {A/peak-1:.0%} from peak -> cash"))
            elif lock and A / year_start - 1 >= lock:
                mode, pending_cost = "SEC", COST
                events.append((d, f"up {A/year_start-1:.0%} this year -> sector ETFs"))
        elif mode == "CASH":
            low = min(low, s)
            if np.isfinite(sma.iloc[i]) and s > sma.iloc[i] and s >= low * (1 + BOUNCE):
                mode, peak, pending_cost = "MOM", A, COST
                events.append((d, f"momentum {s/low-1:+.0%} off its low, above "
                                  f"{SMA}d average -> back in"))
    eq = pd.Series(out, idx)
    return eq.pct_change().fillna(eq.iloc[0] - 1), events


def stats(r):
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    v = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    tr = dd.idxmin()
    pk = eq.loc[:tr].idxmax()
    return ann, v, r.mean() * 252 / v, dd.min(), pk, tr


def load(smoke):
    if smoke:
        d = pd.bdate_range("2017-03-01", "2026-05-29")
        rng = np.random.default_rng(1)
        m = pd.Series(rng.normal(0.0009, 0.022, len(d)), d)
        m.iloc[700:740] -= 0.012                    # sharp V crash ...
        m.iloc[740:800] += 0.010                    # ... and rebound
        m.iloc[1300:1500] -= 0.003                  # slow grind down
        sec = pd.Series(rng.normal(0.0005, 0.011, len(d)), d)
        return m, sec, pd.Series(0.0, d), "synthetic"
    import factor_portfolio as FP       # noqa: E402
    import sector_tailwind as ST        # noqa: E402
    import attention_short_test as AS   # noqa: E402
    res = pd.read_csv(SRC, index_col=0, parse_dates=True)
    m = res["STEADY"].astype(float)
    etf = ST.load_etfs(m.index)
    sec = etf[[s for s in SECTORS if s in etf]].pct_change().mean(axis=1).fillna(0)
    bil = FP.alpaca("BIL")
    if bil is not None and len(bil):
        bil = AS.norm_index(bil)
        col = "close" if "close" in bil.columns else bil.columns[0]
        cash = bil[col].reindex(m.index).ffill().pct_change().fillna(0)
        note = "cash = BIL"
    else:
        cash = pd.Series(0.0, m.index)
        note = "cash = 0% (BIL not cached; slightly understates the STOP books)"
    return m, sec, cash, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    m, sec, cash, note = load(a.smoke)
    print(f"  {m.index[0].date()} -> {m.index[-1].date()}, {len(m)} days | {note}")
    books = {"PLAIN": (None, None), "LOCK30": (0.30, None),
             "LOCK30+S20": (0.30, 0.20), "LOCK30+S25": (0.30, 0.25)}
    rets, evs = {}, {}
    for name, (lk, st) in books.items():
        rets[name], evs[name] = simulate(m, sec, cash, lk, st)
    rets["SECTORS"] = sec

    print(f"\n  {'book':<11} {'annual':>7} {'vol':>6} {'sharpe':>6} {'worst':>7}"
          f"   worst drop from -> to")
    S = {}
    for k, r in rets.items():
        a_, v, s, dd, pk, tr = stats(r)
        S[k] = (s, dd)
        print(f"  {k:<11} {a_*100:>+6.1f}% {v*100:>5.1f}% {s:>6.2f} {dd*100:>6.1f}%"
              f"   {pk:%Y-%m-%d} -> {tr:%Y-%m-%d}")

    print("\n  year by year (worst drop inside the year in brackets)")
    cols = ["PLAIN", "LOCK30", "LOCK30+S20", "LOCK30+S25"]
    print("  year  " + "".join(f"{c:>20}" for c in cols))
    for y in sorted(set(m.index.year)):
        cells = []
        for c in cols:
            g = rets[c][rets[c].index.year == y]
            eq = (1 + g).cumprod()
            cells.append(f"{(eq.iloc[-1]-1)*100:+5.0f}% ({(eq/eq.cummax()-1).min()*100:4.0f}%)")
        print(f"  {y}  " + "".join(f"{c:>20}" for c in cells))
    for c in cols:
        yr = (1 + rets[c]).groupby(rets[c].index.year).prod() - 1
        full = yr[[y for y in yr.index if 2017 < y < 2026]]
        print(f"  {c:<11} losing years {(full < 0).sum()}, years under +10% "
              f"{(full < 0.10).sum()} (2018-2025)")

    for c in ["LOCK30+S20", "LOCK30+S25"]:
        print(f"\n  {c} switches:")
        for d, e in evs[c]:
            if "new year" not in e:
                print(f"    {d:%Y-%m-%d}  {e}")

    (s0, d0), (s1, d1), (s2, d2) = S["LOCK30"], S["LOCK30+S20"], S["LOCK30+S25"]
    ok = s1 > s0 and d1 > d0
    robust = s2 > s0 and d2 > d0
    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"  LOCK30+S20: Sharpe {s1:.2f} vs {s0:.2f}, worst {d1*100:.1f}% vs "
          f"{d0*100:.1f}% -> {'PASS' if ok else 'FAIL'}")
    print(f"  LOCK30+S25: Sharpe {s2:.2f}, worst {d2*100:.1f}% -> "
          f"{'also beats LOCK30' if robust else 'does NOT beat LOCK30'}")
    print("  " + ("port to QC" if ok and robust else
                  "fragile -- do not port" if ok else "keep LOCK30 as is"))
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")


if __name__ == "__main__":
    main()
