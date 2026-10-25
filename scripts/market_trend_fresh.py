"""
market_trend_fresh.py -- the out-of-sample test for "buy dips while the
market is below its 200-day average", plus the two checks owed before any
genome: does it survive an index hedge, and is it just deep_state again?

SAME frozen spec as market_trend_dips.py (commit be599a3): dips 2-25%, hold
20 days, outcome vs same-day list average, WEAK = local index below its
200-day average, months as the unit, 30 bps round trip.

DECIDING GROUND: the fresh markets from build_fresh_universes.py (never
used by any dip test). The 8 markets from be599a3 are re-run for the extra
columns only; they cannot pass or fail anything.

PASS (stated before running, same shape as be599a3):
  at least 6 fresh markets load, spread > 0 in at least 75% of them
  (rounded up), and the fresh-market spreads average > 0 with t > 2,
  each market = one observation.

EXTRA COLUMNS (reported, do not change the verdict):
  hedged   weak-market dips, LONG the stock and SHORT the local index over
           the same 20 days, net of 40 bps (stock + hedge costs). This is
           what an algo would actually earn from the edge.
  raw      weak-market dips long-only, net of 30 bps (edge + market move)
  ds%      share of weak-market dips where deep_state's entry rule was also
           true that day (MACD(12,26) spread < -0.25 ATR, sag <= 15 bars,
           close > 200-day average)
  ex-ds    the WEAK - STRONG spread with every deep_state-active dip
           removed. If the edge lives on after removing them, it's not a
           duplicate.

Run (repo root, after build_fresh_universes.py):
    python3.10 scripts/market_trend_fresh.py
    python3.10 scripts/market_trend_fresh.py --smoke
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial.data import fetch  # noqa: E402

M = os.path.join(ROOT, "manifests")
START, END = "2010-01-01", "2026-06-01"
H, COST, HEDGE_COST = 20, 0.0030, 0.0040
SIG_START = pd.Timestamp("2011-01-01")
FRESH_INDEX = {"france": "^SBF120", "sweden": "^OMX", "switzerland": "^SSMI",
               "hongkong": "^HSI", "brazil": "^BVSP", "spain": "^IBEX",
               "italy": "FTSEMIB.MI", "netherlands": "^AEX",
               "singapore": "^STI", "mexico": "^MXX"}
SEEN = {"UK FTSE250": ("ftse250_symbols.txt", "^FTMC"),
        "Japan Nikkei": ("jp_nikkei_symbols.txt", "^N225"),
        "Australia ASX200": ("asx200_symbols.txt", "^AXJO"),
        "Canada TSX": ("tsx_symbols.txt", "^GSPTSE"),
        "Germany MDAX": ("dax_mid_symbols.txt", "^MDAXI"),
        "Korea KOSPI": ("kr_kospi_symbols.txt", "^KS11"),
        "India Nifty": ("in_nifty_symbols.txt", "^NSEI"),
        "Taiwan TWSE": ("tw_twse_symbols.txt", "^TWII")}


def read_symbols(path):
    return [ln.strip() for ln in open(path)
            if ln.strip() and not ln.startswith("#")]


def deep_state_on(df):
    """deep_state's frozen entry rule, evaluated every bar."""
    c, h, lo = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    spread = (c.ewm(span=12, adjust=False).mean()
              - c.ewm(span=26, adjust=False).mean()) / atr
    neg = (spread < 0).astype(int)
    runlen = neg.groupby((neg != neg.shift()).cumsum()).cumsum() * neg
    return (spread < -0.25) & (runlen <= 15) & (c > c.rolling(200).mean())


def build(data, idx_close):
    trend = idx_close / idx_close.rolling(200).mean() - 1
    idx_fwd = idx_close.shift(-H) / idx_close - 1
    slim, dips = [], []
    for sym, df in data.items():
        df = df[~df.index.duplicated()].astype(float)
        c = df["close"]
        f = pd.DataFrame({"r1": c / c.shift(1) - 1,
                          "fwd": c.shift(-H) / c - 1,
                          "ds": deep_state_on(df)}).dropna()
        f = f[f["fwd"].abs() <= 2.0]
        f.index.name = "date"
        f = f.reset_index()
        slim.append(f[["date", "fwd"]])
        dips.append(f[(f["r1"] <= -0.02) & (f["r1"] > -0.25)
                      & (f["date"] >= SIG_START)])
    s = pd.concat(slim, ignore_index=True)
    base = s.groupby("date").agg(base=("fwd", "mean"), n=("fwd", "count"))
    ev = pd.concat(dips, ignore_index=True).merge(base, left_on="date",
                                                  right_index=True)
    ev = ev[ev["n"] >= 20].copy()
    ev["adj"] = ev["fwd"] - ev["base"]
    ev["trend"] = trend.reindex(ev["date"], method="ffill").values
    ev["ifwd"] = idx_fwd.reindex(ev["date"], method="ffill").values
    return ev.dropna(subset=["trend"])


def monthly(x, dates):
    d = x.groupby(dates).mean()
    return d.groupby(d.index.to_period("M")).mean()


def mstat(x, dates):
    m = monthly(x, dates)
    if len(m) < 3:
        return np.nan, np.nan, len(m)
    return m.mean() * 1e4, m.mean() / (m.std(ddof=1) / np.sqrt(len(m))), \
        len(m)


def spread(ev):
    w, s = ev[ev["trend"] < 0], ev[ev["trend"] >= 0]
    mw, ms = monthly(w["adj"], w["date"]), monthly(s["adj"], s["date"])
    if len(mw) < 6 or len(ms) < 6:
        return np.nan, np.nan
    sp = mw.mean() - ms.mean()
    return sp, sp / np.hypot(mw.std(ddof=1) / np.sqrt(len(mw)),
                             ms.std(ddof=1) / np.sqrt(len(ms)))


def load(sym_file, ticker, a):
    syms = read_symbols(sym_file)
    data = fetch(a.source, syms, "1d", START, END)
    ix = fetch(a.source, [ticker], "1d", START, END) if ticker else {}
    if ticker in ix:
        return data, ix[ticker]["close"], ticker, len(syms)
    px = pd.DataFrame({s: d["close"] for s, d in data.items()})
    return data, (1 + px.pct_change().mean(axis=1)).cumprod(), \
        "equal-weight list (index fetch failed)", len(syms)


def report(name, ev, src, n_names, n_have):
    sp, t = spread(ev)
    w = ev[ev["trend"] < 0]
    hedged = mstat(w["fwd"] - w["ifwd"] - HEDGE_COST, w["date"])
    raw = mstat(w["fwd"] - COST, w["date"])
    ds_share = w["ds"].mean() if len(w) else np.nan
    sp_x, t_x = spread(ev[~ev["ds"]])
    print(f"  {name:<14} {n_have:>3}/{n_names:<3} | spread "
          f"{sp * 1e4:+6.1f} t {t:+5.2f} | hedged {hedged[0]:+6.1f} "
          f"t {hedged[1]:+5.2f} | raw {raw[0]:+6.1f} t {raw[1]:+5.2f} | "
          f"ds% {ds_share:4.0%} | ex-ds {sp_x * 1e4:+6.1f} t {t_x:+5.2f}"
          + ("" if src.startswith("^") or src.endswith(".MI")
             else f"   [{src}]"))
    return sp


HDR = (f"  {'market':<14} {'names':>7} | {'WEAK-STRONG':>17} | "
       f"{'hedged (bps)':>15} | {'raw long':>15} | {'ds%':>4} | "
       f"{'ex-deep_state':>16}")


def smoke():
    idx = pd.bdate_range("2010-01-01", "2026-05-29")
    rng = np.random.default_rng(2)
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
        data[f"S{k}"] = pd.DataFrame({"open": c, "high": c * 1.01,
                                      "low": c * 0.99, "close": c,
                                      "volume": 1e6}, idx)
    return data, ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print(f"  dips 2-25% | hold {H} | vs same-day list average | WEAK = index"
          f" below 200-day avg | {COST * 1e4:.0f} bps RT ({HEDGE_COST * 1e4:.0f}"
          f" hedged)")
    if a.smoke:
        data, ic = smoke()
        print(HDR)
        report("SMOKE planted", build(data, ic), "^X", 60, 60)
        return

    files = sorted(glob.glob(os.path.join(M, "fresh_*_symbols.txt")))
    print(f"\n  == FRESH MARKETS -- DECIDE ({len(files)} lists) ==\n{HDR}")
    fresh = []
    for f in files:
        mkt = os.path.basename(f)[len("fresh_"):-len("_symbols.txt")]
        data, ic, src, n = load(f, FRESH_INDEX.get(mkt), a)
        if len(data) < 20:
            print(f"  {mkt:<14} skipped -- only {len(data)} names with data")
            continue
        ev = build(data, ic)
        sp = report(mkt, ev, src, n, len(data))
        if np.isfinite(sp):
            fresh.append(sp)

    print(f"\n  == ALREADY-SEEN MARKETS (be599a3) -- extra columns only ==\n"
          f"{HDR}")
    for name, (f, tk) in SEEN.items():
        data, ic, src, n = load(os.path.join(M, f), tk, a)
        report(name, build(data, ic), src, n, len(data))

    sp = np.array(fresh)
    need = int(np.ceil(0.75 * len(sp)))
    pos = int((sp > 0).sum())
    t = sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp))) if len(sp) > 2 \
        else np.nan
    print("\n  ===== VERDICT (pre-stated, fresh markets only) =====")
    print(f"  fresh markets loaded: {len(sp)} (need >= 6)")
    print(f"  WEAK > STRONG in {pos} of {len(sp)} (need >= {need})")
    print(f"  average spread {sp.mean() * 1e4:+.1f} bps, t {t:+.2f} (need > 2)")
    if len(sp) >= 6 and pos >= need and t > 2:
        print("  REPLICATES OUT OF SAMPLE -- read the hedged and ex-deep_state "
              "columns next: those decide whether it's a tradeable, new edge. "
              "Then prereg'd genome + gauntlet + forward paper.")
    else:
        print("  DOES NOT REPLICATE out of sample -- be599a3's pass was luck. "
              "Dip-buying seam CLOSED on price/volume data.")


if __name__ == "__main__":
    main()
