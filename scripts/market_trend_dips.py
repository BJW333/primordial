"""
market_trend_dips.py -- ONE indicator, fresh markets: do dips bought while
the market itself is below its 200-day average beat dips bought while the
market is above it?

Lead from dip_features.py (2026-10-25): spy_vs_200 was the only one of 17
features with the same sign on S&P 500 (PIT) / 400 / 600. US data is spent
for it, so this tests it on 8 markets the dip work never touched.

Frozen spec:
  dip      a stock in the market's list closes down 2-25% on the day
  outcome  20-day return from that close, MINUS the same-day average
           20-day return of every stock in that list
  regime   the market's OWN index (e.g. FTSE 250 for UK) vs its 200-day
           average on the dip day: WEAK = below, STRONG = above
           (fixed cut at 0, not fitted)
  spread   WEAK minus STRONG. Expected > 0.
  t-stat   dips are averaged by day, then by calendar MONTH, and the two
           regimes' monthly averages compared -- overlapping 20-day windows
           and slow regimes would inflate a daily t.

PASS (stated before running):
  spread > 0 in at least 6 of the 8 markets, AND the 8 market spreads
  average > 0 with t > 2 treating each market as one observation.
  Pass -> owes a prereg'd genome + gauntlet + duplication check vs
  deep_state. Fail -> the US hint was regime noise; seam CLOSED.

Caveats printed, not hidden: these are current-membership lists (survivor
lift is cancelled by the same-day baseline but not perfectly), and markets
share crises (2011, 2015-16, 2018, 2020, 2022), so 8 markets are fewer
than 8 independent votes.

Controls (2026-10-25): planted weak-market rebound -> spread +276 bps
t +16. Null (no plant) over 7 seeds: spreads -16..+29 bps, t -1.1..+2.1,
centred on zero -- one seed hit t 2.1 by chance, which is why a single
market can't pass this alone.

Run (repo root):
    python3.10 scripts/market_trend_dips.py
    python3.10 scripts/market_trend_dips.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial.data import fetch  # noqa: E402

M = os.path.join(ROOT, "manifests")
START, END = "2010-01-01", "2026-06-01"
H, COST = 20, 0.0030
MARKETS = {  # name: (symbols file, local index ticker)
    "UK FTSE250": ("ftse250_symbols.txt", "^FTMC"),
    "Japan Nikkei": ("jp_nikkei_symbols.txt", "^N225"),
    "Australia ASX200": ("asx200_symbols.txt", "^AXJO"),
    "Canada TSX": ("tsx_symbols.txt", "^GSPTSE"),
    "Germany MDAX": ("dax_mid_symbols.txt", "^MDAXI"),
    "Korea KOSPI": ("kr_kospi_symbols.txt", "^KS11"),
    "India Nifty": ("in_nifty_symbols.txt", "^NSEI"),
    "Taiwan TWSE": ("tw_twse_symbols.txt", "^TWII"),
}
SIG_START = pd.Timestamp("2011-01-01")      # 200-day average needs warmup


def read_symbols(path):
    return [ln.strip() for ln in open(path)
            if ln.strip() and not ln.startswith("#")]


def build(data, idx_close):
    trend = idx_close / idx_close.rolling(200).mean() - 1
    slim, dips = [], []
    for sym, df in data.items():
        df = df[~df.index.duplicated()]
        c = df["close"].astype(float)
        r1 = c / c.shift(1) - 1
        fwd = c.shift(-H) / c - 1
        f = pd.DataFrame({"r1": r1, "fwd": fwd}).dropna()
        f = f[f["fwd"].abs() <= 2.0]
        f.index.name = "date"
        f = f.reset_index()
        slim.append(f)
        dips.append(f[(f["r1"] <= -0.02) & (f["r1"] > -0.25)
                      & (f["date"] >= SIG_START)])
    s = pd.concat(slim, ignore_index=True)
    base = s.groupby("date").agg(base=("fwd", "mean"), n=("fwd", "count"))
    ev = pd.concat(dips, ignore_index=True).merge(base, left_on="date",
                                                  right_index=True)
    ev = ev[ev["n"] >= 20].copy()
    ev["adj"] = ev["fwd"] - ev["base"]
    ev["trend"] = trend.reindex(ev["date"], method="ffill").values
    return ev.dropna(subset=["trend"]), trend


def monthly(e):
    d = e.groupby("date")["adj"].mean()
    return d.groupby(d.index.to_period("M")).mean()


def spells(trend):
    t = trend.dropna()
    below = (t < 0).astype(int)
    return int(((below.diff() == 1)).sum() + (below.iloc[0] == 1))


def run_market(name, sym_file, ticker, a):
    if a.smoke:
        data, idx_close = a.smoke_data
    else:
        syms = read_symbols(os.path.join(M, sym_file))
        data = fetch(a.source, syms, "1d", START, END)
        ix = fetch(a.source, [ticker], "1d", START, END)
        if ticker in ix:
            idx_close = ix[ticker]["close"]
            src = ticker
        else:   # index download failed: equal-weight average of the list
            px = pd.DataFrame({s: d["close"] for s, d in data.items()})
            idx_close = (1 + px.pct_change().mean(axis=1)).cumprod()
            src = "equal-weight list (index fetch failed)"
        print(f"\n  [{name}] {len(data)}/{len(syms)} names | regime from "
              f"{src}")
    ev, trend = build(data, idx_close)
    weak, strong = ev[ev["trend"] < 0], ev[ev["trend"] >= 0]
    mw, ms = monthly(weak), monthly(strong)
    if len(mw) < 6 or len(ms) < 6:
        print(f"  [{name}] too few months in a regime -- skipped")
        return None
    spread = mw.mean() - ms.mean()
    t = spread / np.hypot(mw.std(ddof=1) / np.sqrt(len(mw)),
                          ms.std(ddof=1) / np.sqrt(len(ms)))
    w_net = monthly(weak.assign(adj=weak["adj"] - COST))
    w_t = w_net.mean() / (w_net.std(ddof=1) / np.sqrt(len(w_net)))
    print(f"  {name:<17} dips {len(ev):>7} | weak-market months {len(mw):>3} "
          f"({spells(trend)} spells below trend)")
    print(f"  {'':<17} WEAK {mw.mean() * 1e4:+7.1f} bps | STRONG "
          f"{ms.mean() * 1e4:+7.1f} bps | spread {spread * 1e4:+7.1f} "
          f"t {t:+.2f} | weak-market dips after cost "
          f"{w_net.mean() * 1e4:+.1f} bps t {w_t:+.2f}")
    return spread


def smoke_data():
    """Planted: dips rebound +3% vs peers only while the index is below
    its 200-day average."""
    idx = pd.bdate_range("2010-01-01", "2026-05-29")
    rng = np.random.default_rng(4)
    mkt = np.cumsum(np.where((np.arange(len(idx)) // 300) % 2 == 0,
                             0.0008, -0.0006) + rng.normal(0, 0.008,
                                                           len(idx)))
    ic = pd.Series(100 * np.exp(mkt), idx)
    below = (ic < ic.rolling(200).mean()).values
    data = {}
    for k in range(60):
        r = rng.normal(0, 0.02, len(idx)) + np.r_[0, np.diff(mkt)]
        for i in np.where(r <= -0.02)[0]:
            if i < len(idx) - 25 and below[i]:
                r[i + 1:i + 21] += 0.0015
        c = 100 * np.exp(np.cumsum(r))
        data[f"S{k}"] = pd.DataFrame({"open": c, "high": c, "low": c,
                                      "close": c, "volume": 1e6}, idx)
    return data, ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    markets = MARKETS
    if a.smoke:
        a.smoke_data = smoke_data()
        markets = {"SMOKE planted": ("", "")}

    print(f"  dips 2-25% | hold {H} days | outcome vs same-day list average "
          f"| cost {COST * 1e4:.0f} bps RT")
    res = {n: run_market(n, f, t, a) for n, (f, t) in markets.items()}
    sp = np.array([v for v in res.values() if v is not None])
    pos = int((sp > 0).sum())
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print(f"  smoke: spread {sp[0] * 1e4:+.1f} bps (planted, pipe check)")
        return
    t = sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp))) if len(sp) > 2 \
        else np.nan
    print(f"  markets with WEAK > STRONG: {pos} of {len(sp)} (need >= 6)")
    print(f"  average spread {sp.mean() * 1e4:+.1f} bps, t {t:+.2f} across "
          f"markets (need > 2)")
    if pos >= 6 and t > 2:
        print("  REPLICATES -- buy-dips-in-weak-markets holds outside the US. "
              "Next: prereg'd genome + gauntlet + dup check vs deep_state.")
    else:
        print("  DOES NOT REPLICATE -- the US hint was regime noise. "
              "Dip-buying seam CLOSED on price/volume data.")


if __name__ == "__main__":
    main()
