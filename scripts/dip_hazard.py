#!/usr/bin/env python3
"""dip_hazard.py -- measure the thing Blake described, directly:

  1. What is the distribution of dip DURATIONS (consecutive down seconds)
     in BTC at 1s resolution? Mean, median, histogram.
  2. Is duration memoryless? If dip lengths fit a geometric distribution,
     "the dip has lasted its average length" carries ZERO information and
     the strategy is dead by construction. Test: hazard curve
     P(dip ends at k | dip reached k). Flat curve = memoryless = dead.
     Rising curve = the longer the dip, the more likely it ends = alive.
  3. Simulate the actual rule: when a dip reaches the (train-window)
     average duration, buy; measure the pullback over 5/15/30/60s
     horizons on the TEST window, gross and net of costs. Plus a sweep
     over wait-times k so you can see where the curve peaks instead of
     trusting the mean.

Data: 1s bars built from Coinbase Exchange public trades (no auth), or
--csv for your own 1s file (needs columns: ts, close). Train/test split
so the average is estimated on data the evaluation never sees.

  python3 dip_hazard.py --fetch-hours 48          # pull ticks, then analyze
  python3 dip_hazard.py --csv btc_1s.csv          # your own data
  python3 dip_hazard.py --control rising          # planted rising hazard
  python3 dip_hazard.py --control memoryless      # planted null

Controls have KNOWN answers: run both first; if the report can't tell
them apart, don't believe it on real data.

Caveat that matters: 1s last-trade closes contain bid-ask bounce, which
manufactures fake reversals you cannot capture. The pullback table is an
UPPER BOUND on the real edge; anything that doesn't survive the cost
columns by a wide margin is noise. Flat seconds (no trades) end a run by
default; --flat hold lets them extend it (sensitivity check, run both).
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

COINBASE = "https://api.exchange.coinbase.com"


# ---------------------------------------------------------------- data
def fetch_trades(product: str, hours: float, out_csv: str) -> pd.DataFrame:
    import requests
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(hours=hours)
    rows, before, n_req = [], None, 0
    sess = requests.Session()
    while True:
        params = {"limit": 1000}
        if before:
            params["after"] = before          # paginate toward older trades
        r = sess.get(f"{COINBASE}/products/{product}/trades",
                     params=params, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        before = r.headers.get("CB-AFTER")
        n_req += 1
        oldest = pd.Timestamp(batch[-1]["time"])
        if n_req % 50 == 0:
            print(f"  {len(rows):>8} trades, back to {oldest}", flush=True)
        if oldest.tz_localize(None) < start.tz_localize(None) or not before:
            break
        time.sleep(0.15)                      # public rate limit headroom
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["time"])
    df["price"] = df["price"].astype(float)
    df = df.sort_values("ts").set_index("ts")
    df.to_csv(out_csv)
    print(f"  saved {len(df)} trades -> {out_csv}")
    return df


def bars_1s(trades: pd.DataFrame) -> pd.Series:
    """1s close series; seconds with no trades forward-fill (flat)."""
    px = trades["price"].resample("1s").last()
    return px.ffill().dropna()


# ------------------------------------------------------------ controls
def control_series(kind: str, n: int = 200_000, seed: int = 7) -> pd.Series:
    """Synthetic 1s closes with a KNOWN hazard structure.
    memoryless: each second down w.p. 0.5 independently -> geometric runs,
                flat hazard, strategy must show ~0 net edge.
    rising:     dips are drawn with duration ~ 4 + Poisson(2) (hazard rises
                steeply near 5-6s) and are followed by a genuine pullback
                that retraces ~60% of the dip -> strategy must light up.
    """
    rng = np.random.default_rng(seed)
    if kind == "memoryless":
        steps = rng.choice([-1.0, 1.0], size=n)
    else:
        steps = []
        while len(steps) < n:
            d = 4 + rng.poisson(2)                    # dip length ~6s mean
            dip = -np.abs(rng.normal(1.0, 0.2, d))
            pull_len = max(2, int(d * 0.8))
            pull = np.abs(rng.normal(1.0, 0.2, pull_len)) * 0.75
            drift = rng.normal(0.05, 1.0, rng.integers(3, 12))
            steps.extend(dip.tolist() + pull.tolist() + drift.tolist())
        steps = np.array(steps[:n])
    px = 60_000 + np.cumsum(steps)
    idx = pd.date_range("2026-01-01", periods=n, freq="1s")
    return pd.Series(px, index=idx)


# ------------------------------------------------------------ analysis
def down_runs(close: pd.Series, flat: str) -> np.ndarray:
    """Per-second down-run length. flat='break': unchanged second resets.
    flat='hold': unchanged second extends neither but doesn't reset."""
    d = np.sign(close.diff().to_numpy())
    run = np.zeros(len(d), dtype=int)
    for i in range(1, len(d)):
        if d[i] < 0:
            run[i] = run[i - 1] + 1
        elif d[i] == 0 and flat == "hold":
            run[i] = run[i - 1]
        else:
            run[i] = 0
    return run


def hazard_table(run: np.ndarray, kmax: int = 15) -> pd.DataFrame:
    rows = []
    for k in range(1, kmax + 1):
        reached = int((run == k).sum())   # each run counts once at k
        # dip "ends at k": run hit exactly k and next second's run is 0
        ended = int(((run[:-1] == k) & (run[1:] == 0)).sum())
        rows.append({"k_seconds": k, "n_reached": reached,
                     "P_end_next": ended / reached if reached else np.nan})
    return pd.DataFrame(rows)


def duration_stats(run: np.ndarray) -> dict:
    ends = np.where((run[:-1] > 0) & (run[1:] == 0))[0]
    durs = run[ends]
    if len(durs) == 0:
        return {}
    mean = durs.mean()
    # geometric with same mean: P(D=k) = p(1-p)^(k-1), p = 1/mean
    p = 1.0 / mean
    ks = np.arange(1, 13)
    emp = np.array([(durs == k).mean() for k in ks])
    geo = p * (1 - p) ** (ks - 1)
    return {"n_dips": len(durs), "mean_s": mean,
            "median_s": float(np.median(durs)),
            "p90_s": float(np.percentile(durs, 90)),
            "emp_vs_geo": pd.DataFrame({"k": ks, "empirical": emp,
                                        "geometric_fit": geo})}


def simulate(close: pd.Series, run: np.ndarray, wait_k: int,
             horizons=(5, 15, 30, 60)) -> dict:
    """Buy at second t+1 whenever run[t] == wait_k (first touch only).
    Forward return in bps at each horizon."""
    px = close.to_numpy()
    trig = np.where((run == wait_k) & (np.roll(run, 1) == wait_k - 1))[0]
    trig = trig[trig + max(horizons) + 1 < len(px)]
    out = {"wait_k": wait_k, "n_trades": len(trig)}
    for h in horizons:
        entry = px[trig + 1]                       # next-second fill
        exitp = px[trig + 1 + h]
        bps = (exitp / entry - 1) * 1e4
        out[f"ret{h}s_bps"] = bps.mean()
        out[f"win{h}s"] = (bps > 0).mean()
    return out


def report(close: pd.Series, flat: str, cost_rt_bps: float):
    n = len(close)
    split = int(n * 0.6)
    train, test = close.iloc[:split], close.iloc[split:]
    run_tr = down_runs(train, flat)
    run_te = down_runs(test, flat)

    st = duration_stats(run_tr)
    if not st:
        sys.exit("no dips found -- check data")
    print(f"\n== dip durations (TRAIN, {st['n_dips']} dips, flat={flat}) ==")
    print(f"mean {st['mean_s']:.2f}s  median {st['median_s']:.0f}s  "
          f"p90 {st['p90_s']:.0f}s")
    print("\nduration pmf vs geometric (same mean):")
    print(st["emp_vs_geo"].to_string(index=False,
                                     float_format=lambda x: f"{x:.4f}"))
    print("\n== hazard curve (TRAIN): P(dip ends next second | lasted k) ==")
    print("flat curve ~= memoryless ~= average carries no timing info")
    print(hazard_table(run_tr).to_string(index=False,
                                         float_format=lambda x: f"{x:.4f}"))

    kbar = max(1, int(round(st["mean_s"])))
    print(f"\n== strategy on TEST: buy when dip reaches k seconds ==")
    print(f"(train mean = {st['mean_s']:.2f}s -> Blake's rule is k={kbar}; "
          f"sweep shown for context. Gross bps; subtract "
          f"{cost_rt_bps:.0f} bps round trip for net.)")
    rows = [simulate(test, run_te, k) for k in range(1, 13)]
    df = pd.DataFrame(rows)
    df["net30s_bps"] = df["ret30s_bps"] - cost_rt_bps
    mark = df["wait_k"] == kbar
    df.insert(0, "rule", np.where(mark, ">>", ""))
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nread it as: the >> row is your stated rule (wait the average, "
          "then buy). If net30s_bps <= 0 there and everywhere nearby, the "
          "edge does not exist at taker costs; the only remaining question "
          "is whether a maker-entry version survives queue reality, which "
          "this data cannot answer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="1s data csv with ts,close columns")
    ap.add_argument("--fetch-hours", type=float,
                    help="pull Coinbase BTC-USD ticks for the last N hours")
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--control", choices=["rising", "memoryless"])
    ap.add_argument("--flat", choices=["break", "hold"], default="break")
    ap.add_argument("--cost-rt-bps", type=float, default=50.0,
                    help="round-trip cost in bps (Coinbase taker ~50-80; "
                         "maker ~10-20 incl. spread give-up)")
    a = ap.parse_args()

    if a.control:
        close = control_series(a.control)
        print(f"[control: {a.control} -- answer is planted, verify the "
              f"report recovers it]")
    elif a.csv:
        df = pd.read_csv(a.csv, parse_dates=["ts"]).set_index("ts")
        close = df["close"].astype(float)
    elif a.fetch_hours:
        trades = fetch_trades(a.product, a.fetch_hours,
                              f"{a.product}_trades.csv")
        close = bars_1s(trades)
    else:
        ap.error("one of --csv / --fetch-hours / --control required")
    report(close, a.flat, a.cost_rt_bps)


if __name__ == "__main__":
    main()
