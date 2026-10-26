"""
pump_study2.py -- the honest version of pump_study.py's two verdicts.

WHY: pump_study.py (commit 73423ff) passed both FADE and RIDE, but
  * RIDE had LOOK-AHEAD: the early 5m signal was only searched inside
    hours already known to finish +10% -- the future picked the trades.
  * FADE assumed every tiny coin can be shorted at 0.3% cost, BTC-hedged,
    with no limit on how far a pump can run against you.

RIDE v2 (no look-ahead) -- rules fixed before this run:
  Random sample of 150 USDT coins (seed 7, from ALL coins, not only ones
  that pumped). Every 5-minute bar 2021-06 -> 2026-05 is scanned as it
  happens: TRIGGER = 5m dollar volume >= 10x its median over the prior
  24 hours AND the bar closes >= +5%. One trigger per coin per 6 hours.
  Buy at the trigger bar's close, sell 30 minutes later (actual coin
  return, no BTC hedge). Cost 0.5% round trip (Alpaca-level crypto fees).
  PASS: mean >= +1% net, median > 0, t > 3 (t clusters trades by day).

FADE v2 (realistic) -- rules fixed before this run:
  The pump events from pump_study.py (data/crypto_pumps/events.csv),
  restricted to coins that HAD a Binance USD-M perpetual futures market
  in the event's month (proxy for "shortable"). Short at the pump hour's
  close; cover after 24h, or at a STOP if price trades 30% above entry
  (filled at the stop or the hour's open if it gapped past it). Actual
  short P&L, no BTC hedge. Cost 1.0% round trip.
  PASS: mean >= +2% net, median > 0, t > 3 over the last five years AND
  mean > 0 in every full year 2022, 2023, 2024, 2025.
  Also printed: other costs/stops, all coins vs perp-only, worst trades.

Run (repo root; RIDE downloads ~9,000 monthly 5m files ~ 3-4 GB of
transfer, ~30-60 min, nothing big kept; FADE ~25 min):
    python3.10 scripts/pump_study2.py --smoke
    python3.10 scripts/pump_study2.py --workers 16 | tee logs/pump_study2.txt
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pump_study as P  # noqa: E402

DIR = P.DIR
RIDE_DIR = os.path.join(DIR, "ride_v2")
FADE_OUT = os.path.join(DIR, "fade_v2.csv")
RECENT, END = P.RECENT, P.END
N_COINS, SEED = 150, 7
RIDE_X, RIDE_RET, RIDE_GAP_BARS, BASE_BARS = 10.0, 0.05, 72, 288
RIDE_COST, FADE_COST, STOP = 0.005, 0.010, 0.30


def read_ohlc(content):
    z = zipfile.ZipFile(io.BytesIO(content))
    d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None,
                    usecols=[0, 1, 2, 4, 7])
    if not np.issubdtype(d[0].dtype, np.number):
        d = d.iloc[1:].astype(float)
    d.columns = ["t", "open", "high", "close", "qvol"]
    ms = np.where(d["t"] > 1e14, d["t"] / 1000, d["t"])
    d["t"] = pd.to_datetime(ms.astype("int64"), unit="ms")
    return d.set_index("t").astype(float)


def monthly(sym, tf, market="spot", start=RECENT, end=END):
    pre = (f"data/spot/monthly/klines/{sym}/{tf}/" if market == "spot"
           else f"data/futures/um/monthly/klines/{sym}/{tf}/")
    keys = [k for k in P.s3_list(pre) if k.endswith(".zip")]
    parts = []
    for k in keys:
        ym = re.findall(r"(\d{4})-(\d{2})\.zip$", k)
        if not ym:
            continue
        m = pd.Timestamp(f"{ym[0][0]}-{ym[0][1]}-01")
        if m < start - pd.DateOffset(months=1) or m >= end:
            continue
        c = P.get(f"{P.WEB}/{k}")
        if c:
            parts.append(read_ohlc(c))
    if not parts:
        return None
    d = pd.concat(parts).sort_index()
    return d[~d.index.duplicated()]


# ------------------------------------------------------------------- RIDE
def ride_triggers(d):
    q, c = d["qvol"].values, d["close"].values
    base = pd.Series(q).shift(1).rolling(BASE_BARS, min_periods=200) \
        .median().values
    r = c / np.r_[np.nan, c[:-1]] - 1
    ok = (q >= RIDE_X * base) & (base > 0) & (r >= RIDE_RET)
    out, last = [], -10 ** 9
    for i in np.flatnonzero(ok):
        if i - last < RIDE_GAP_BARS or d.index[i] < RECENT:
            continue
        last = i
        rec = {"t": d.index[i], "bar_ret": r[i], "vol_x": q[i] / base[i]}
        for k, lab in ((3, "r15m"), (6, "r30m"), (12, "r60m")):
            j = i + k
            ok_j = j < len(c) and (d.index[j] - d.index[i]) == \
                pd.Timedelta(minutes=5 * k)
            rec[lab] = c[j] / c[i] - 1 if ok_j else np.nan
        out.append(rec)
    return out


def run_ride(workers):
    os.makedirs(RIDE_DIR, exist_ok=True)
    syms = P.symbols()
    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(syms, size=min(N_COINS, len(syms)),
                               replace=False))
    todo = [s for s in sample if not os.path.exists(
        os.path.join(RIDE_DIR, s + ".csv"))]
    print(f"  RIDE: {len(sample)} random coins (of {len(syms)}), "
          f"{len(todo)} to download")

    def one(s):
        try:
            d = monthly(s, "5m")
        except Exception:  # noqa: BLE001
            return s, None
        tr = ride_triggers(d) if d is not None and len(d) > 400 else []
        pd.DataFrame(tr).assign(sym=s).to_csv(
            os.path.join(RIDE_DIR, s + ".csv"), index=False)
        return s, len(tr)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, (s, n) in enumerate(ex.map(one, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)} coins ({(time.time()-t0)/60:.0f} "
                      f"min) last: {s} {n} triggers", flush=True)
    parts = []
    for s in sample:
        try:
            parts.append(pd.read_csv(os.path.join(RIDE_DIR, s + ".csv"),
                                     parse_dates=["t"]))
        except (pd.errors.EmptyDataError, ValueError, FileNotFoundError):
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def report_ride(tr):
    print(f"\n  ===== RIDE v2 (no look-ahead): {len(tr)} triggers in "
          f"{tr['sym'].nunique() if len(tr) else 0} coins, last 5 years "
          f"=====")
    if not len(tr):
        return (np.nan,) * 3
    print(f"  {'hold':<8} {'gross mean':>10} {'median':>7} {'% down':>7}  "
          f"net @0.5%  net @1.0%")
    for lab in ("r15m", "r30m", "r60m"):
        x = tr[lab].dropna()
        print(f"  {lab[1:]:<8} {x.mean()*100:>+9.2f}% {x.median()*100:>+6.2f}%"
              f" {(x < 0).mean()*100:>6.0f}%  {(x.mean()-0.005)*100:>+7.2f}%"
              f"  {(x.mean()-0.01)*100:>+7.2f}%")
    net = tr["r30m"].dropna() - RIDE_COST
    t = P.day_t(net, tr.loc[net.index, "t"])
    by = tr.groupby(tr["t"].dt.year)["r30m"].mean()
    print("  30-min gross by year: " + "  ".join(
        f"{y}:{v*100:+.2f}%" for y, v in by.items()))
    return net.mean(), net.median(), t


# ------------------------------------------------------------------- FADE
def perp_months():
    pre = P.s3_list("data/futures/um/monthly/klines/")
    perps = sorted({p.rstrip("/").split("/")[-1] for p in pre})
    return [p for p in perps if p.endswith("USDT")]


def first_perp_month(sym):
    keys = [k for k in P.s3_list(f"data/futures/um/monthly/klines/{sym}/1h/")
            if k.endswith(".zip")]
    ms = []
    for k in keys:
        ym = re.findall(r"(\d{4})-(\d{2})\.zip$", k)
        if ym:
            ms.append(pd.Timestamp(f"{ym[0][0]}-{ym[0][1]}-01"))
    return (min(ms), max(ms)) if ms else (None, None)


def fade_trade(d, t, stop, cost):
    if t not in d.index:
        return np.nan, np.nan
    i = d.index.get_loc(t)
    entry = d["close"].iloc[i]
    win = d.iloc[i + 1:i + 25]
    if len(win) < 20:
        return np.nan, np.nan
    level = entry * (1 + stop)
    hit = np.flatnonzero(win["high"].values >= level)
    worst = win["high"].max() / entry - 1
    if len(hit):
        fill = max(level, win["open"].iloc[hit[0]])
        return -(fill / entry - 1) - cost, worst
    return -(win["close"].iloc[-1] / entry - 1) - cost, worst


def run_fade(workers):
    ev = pd.read_csv(P.EVENTS, parse_dates=["t"])
    ev = ev[(ev["t"] >= P.START) & (ev["t"] < END)]
    perps = set(perp_months())
    print(f"\n  FADE: {len(ev)} pump events; {len(perps)} USDT perp markets "
          f"ever listed")
    coins = sorted(ev["sym"].unique())

    def one(s):
        try:
            d = monthly(s, "1h", start=P.START)
        except Exception:  # noqa: BLE001
            d = None
        lo, hi = first_perp_month(s) if s in perps else (None, None)
        rows = []
        for t in ev.loc[ev["sym"] == s, "t"]:
            rec = {"sym": s, "t": t,
                   "perp": lo is not None and lo <= t.to_period("M")
                   .to_timestamp() <= hi}
            for stp in (0.30, 0.50, 9.99):
                for cst in (0.003, 0.010, 0.020):
                    pnl, worst = fade_trade(d, t, stp, cst) if d is not None \
                        else (np.nan, np.nan)
                    rec[f"s{int(stp*100)}_c{int(cst*1000)}"] = pnl
                rec["worst_24h"] = worst
            rows.append(rec)
        return rows

    out = []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, rows in enumerate(ex.map(one, coins), 1):
            out += rows
            if i % 50 == 0 or i == len(coins):
                print(f"    {i}/{len(coins)} coins ({(time.time()-t0)/60:.0f}"
                      f" min)", flush=True)
    f = pd.DataFrame(out)
    f.to_csv(FADE_OUT, index=False)
    return f


def report_fade(f):
    rec = f[f["t"] >= RECENT]
    main = "s30_c10"
    print(f"\n  ===== FADE v2 (realistic), last 5 years: {len(rec)} pumps, "
          f"{rec['perp'].mean():.0%} on coins with a perp market =====")
    print(f"  short 24h, actual P&L (mean / median):")
    print(f"  {'':<22} {'all coins':>18} {'perp-only':>18}")
    for stp, lab in ((30, "stop at +30%"), (50, "stop at +50%"),
                     (999, "no stop")):
        for cst in (3, 10, 20):
            col = f"s{stp}_c{cst}"
            a = rec[col].dropna()
            p = rec.loc[rec["perp"], col].dropna()
            print(f"  {lab:<13} cost {cst/10:.1f}%  "
                  f"{a.mean()*100:>+7.1f}% /{a.median()*100:>+6.1f}%  "
                  f"{p.mean()*100:>+7.1f}% /{p.median()*100:>+6.1f}%")
    p = rec[rec["perp"]]
    x = p[main].dropna()
    t = P.day_t(x, p.loc[x.index, "t"])
    years = {}
    for y in (2022, 2023, 2024, 2025):
        g = p.loc[p["t"].dt.year == y, main].dropna()
        years[y] = g.mean() if len(g) else np.nan
    print("  PRIMARY (perp-only, +30% stop, 1% cost) by year: " + "  ".join(
        f"{y}:{v*100:+.1f}% (n {int((p['t'].dt.year == y).sum())})"
        for y, v in years.items()))
    w = p.nlargest(5, "worst_24h")
    print("  worst runaways against a short (max rise within 24h): " +
          ", ".join(f"{s} {d:%Y-%m-%d} +{v*100:.0f}%" for s, d, v in
                    zip(w["sym"], w["t"], w["worst_24h"])))
    print(f"  stop hit in {(p['worst_24h'] >= 0.30).mean():.0%} of perp-coin "
          f"pumps")
    old = f[(f["t"] < RECENT) & f["perp"]][main].dropna()
    print(f"  context 2018-2021 (perp-only, primary): mean "
          f"{old.mean()*100:+.1f}% median {old.median()*100:+.1f}% "
          f"n {len(old)}")
    return x.mean(), x.median(), t, years


# ------------------------------------------------------------------ smoke
def smoke():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-06-01", "2022-06-01", freq="5min")
    c = np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
    q = rng.lognormal(8, 0.3, len(idx))
    spikes = rng.choice(np.arange(500, len(idx) - 20), 300, replace=False)
    for i in spikes:                       # +6% spike bar, then +1% drift
        c[i:] *= 1.06
        q[i] *= 30
        c[i + 1:i + 7] *= np.linspace(1.0, 1.01, 6)
        c[i + 7:] *= 1.01
    d = pd.DataFrame({"open": c, "high": c * 1.002, "close": c, "qvol": q},
                     idx)
    tr = pd.DataFrame(ride_triggers(d)).assign(sym="X")
    x = tr["r30m"].dropna()
    print(f"  ride smoke: {len(tr)} triggers (300 planted spikes, some "
          f"within 6h of each other), 30m gross mean {x.mean()*100:+.2f}% "
          f"(planted +1.0%)")
    h = pd.date_range("2024-01-01", periods=60, freq="h")
    p = np.r_[np.full(10, 100.0), np.linspace(100, 70, 50)]
    dd = pd.DataFrame({"open": p, "high": p * 1.01, "close": p}, h)
    pnl, worst = fade_trade(dd, h[9], 0.30, 0.01)
    print(f"  fade smoke: falling coin short P&L {pnl*100:+.1f}% "
          f"(expect ~+13%)")
    p2 = np.r_[np.full(10, 100.0), np.linspace(100, 200, 50)]
    dd2 = pd.DataFrame({"open": p2, "high": p2 * 1.01, "close": p2}, h)
    pnl2, worst2 = fade_trade(dd2, h[9], 0.30, 0.01)
    print(f"  fade smoke: runaway coin short P&L {pnl2*100:+.1f}% "
          f"(stopped, expect ~-31%), max rise +{worst2*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", choices=["ride", "fade"])
    a = ap.parse_args()
    if a.smoke:
        smoke()
        return
    res = {}
    if a.only != "fade":
        res["ride"] = report_ride(run_ride(a.workers))
    if a.only != "ride":
        f = pd.read_csv(FADE_OUT, parse_dates=["t"]) if os.path.exists(
            FADE_OUT) else run_fade(a.workers)
        res["fade"] = report_fade(f)
    print("\n  ===== VERDICT (pre-stated before this run) =====")
    if "ride" in res:
        m, md, t = res["ride"]
        ok = m >= 0.01 and md > 0 and t > 3
        print(f"  RIDE v2 (+30m, 0.5% cost): mean {m*100:+.2f}% median "
              f"{md*100:+.2f}% t {t:+.2f} -> {'PASS' if ok else 'FAIL'}")
    if "fade" in res:
        m, md, t, yrs = res["fade"]
        ok = (m >= 0.02 and md > 0 and t > 3
              and all(np.isfinite(v) and v > 0 for v in yrs.values()))
        print(f"  FADE v2 (perp-only, +30% stop, 1% cost): mean {m*100:+.1f}%"
              f" median {md*100:+.1f}% t {t:+.2f}, every year 2022-25 > 0: "
              f"{all(v > 0 for v in yrs.values())} -> "
              f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
