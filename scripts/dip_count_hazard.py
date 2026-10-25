#!/usr/bin/env python3
"""dip_count_hazard.py -- test the DIP-COUNT hypothesis across timeframes.

Distinct from dip_hazard.py (which asked: within ONE dip, does age predict
its end? Answer on BTC 1s: no). This asks Blake's follow-up: across a
SEQUENCE of dips -- each a completed down-run making a lower low than the
previous dip -- does the count N predict a rebound? I.e. is "the 3rd dip"
(or 4th, ...) where buyers show up?

Per timeframe it reports:
  1. leg-count hazard: P(sequence ends at N | it reached N), where a
     sequence "ends" when the next dip FAILS to make a lower low (= the
     rebound). Flat curve = dip count carries no timing info. Rising
     curve = sequences exhaust -> mechanism exists.
  2. strategy: after the Nth lower-low dip completes (first up bar),
     buy next bar; forward return at 5/15/30/60 bars, train/test split
     (train sets nothing here -- both halves shown so a "pattern" must
     appear in BOTH to count).

VERDICT RULE (pre-stated, resist the urge to relax it after looking):
a (timeframe, N) cell is INTERESTING only if hazard rises monotonically
over >= 3 consecutive N in TRAIN, AND test-half forward return at that
same N is positive gross with n >= 100 trades. One hot cell in a 5x8
sweep is base-rate noise, not a finding.

  python3 dip_count_hazard.py --csv BTC-USD_1s.csv --tf 1 5 15 60
  python3 dip_count_hazard.py --fetch-candles 120 --tf 300 900 3600
  python3 dip_count_hazard.py --control rising      # rebound planted at N=3
  python3 dip_count_hazard.py --control memoryless

--tf values are BAR SIZES IN SECONDS, resampled from the input. From a
48h tick pull, anything >= 300s is sample-starved (warned inline); use
--fetch-candles DAYS for coarse timeframes (Coinbase public candles).
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

COINBASE = "https://api.exchange.coinbase.com"


# ---------------------------------------------------------------- data
def fetch_candles(product: str, days: float, granularity: int,
                  out_csv: str) -> pd.Series:
    """Coinbase public candles, 300/request, granularity in {60,300,900,
    3600,21600,86400}. Returns close series."""
    import requests
    assert granularity in (60, 300, 900, 3600, 21600, 86400), \
        "coinbase only serves 60/300/900/3600/21600/86400s candles"
    end = pd.Timestamp.utcnow().floor("min")
    start = end - pd.Timedelta(days=days)
    sess, rows, cur = requests.Session(), [], end
    while cur > start:
        lo = max(start, cur - pd.Timedelta(seconds=granularity * 300))
        r = sess.get(f"{COINBASE}/products/{product}/candles",
                     params={"granularity": granularity,
                             "start": lo.isoformat(), "end": cur.isoformat()},
                     timeout=20)
        r.raise_for_status()
        rows.extend(r.json())            # [t, low, high, open, close, vol]
        cur = lo
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close",
                                     "vol"]).drop_duplicates("t")
    df["ts"] = pd.to_datetime(df["t"], unit="s")
    close = df.sort_values("ts").set_index("ts")["close"].astype(float)
    close.rename("close").to_csv(out_csv)
    print(f"  saved {len(close)} {granularity}s candles -> {out_csv}")
    return close


def control_series(kind: str, n: int = 300_000, seed: int = 11) -> pd.Series:
    """memoryless: independent +-1 steps -> P(lower-low dip) constant,
    flat count-hazard, ~0 edge.
    rising: sequences of lower-low dips are drawn to die at N ~ 3
    (hazard jumps at N=3) and the 3rd dip's end is followed by a real
    multi-bar rebound -> detector must flag N=3."""
    rng = np.random.default_rng(seed)
    if kind == "memoryless":
        steps = rng.choice([-1.0, 1.0], size=n)
    else:
        steps = []
        while len(steps) < n:
            n_legs = min(1 + rng.geometric(0.45), 6)   # mode ~2-3 legs
            for leg in range(n_legs):
                down = -np.abs(rng.normal(1.0, 0.3, rng.integers(2, 6)))
                bounce = np.abs(rng.normal(0.8, 0.2,
                                           rng.integers(1, 3)))  # weak
                steps.extend(down.tolist() + bounce.tolist())
            # after final leg: real rebound, retraces most of the move
            steps.extend(np.abs(rng.normal(1.1, 0.3,
                                           rng.integers(4, 9))).tolist())
            steps.extend(rng.normal(0.0, 1.0, rng.integers(2, 8)).tolist())
        steps = np.array(steps[:n])
    px = 60_000 + np.cumsum(steps)
    idx = pd.date_range("2026-01-01", periods=n, freq="1s")
    return pd.Series(px, index=idx)


# ------------------------------------------------------------ analysis
def dip_events(close: np.ndarray):
    """Completed down-runs. Returns arrays (end_idx, low) per dip:
    end_idx = index of the last down bar; low = min close in the run."""
    d = np.sign(np.diff(close))
    ends, lows = [], []
    i, n = 1, len(close)
    while i < n:
        if d[i - 1] < 0:
            j = i
            lo = close[i]
            while j + 1 < n and d[j] < 0:
                j += 1
                lo = min(lo, close[j])
            ends.append(j)
            lows.append(lo)
            i = j + 1
        else:
            i += 1
    return np.array(ends), np.array(lows)


def leg_sequences(ends: np.ndarray, lows: np.ndarray):
    """Consecutive lower-low dip chains. Returns per-dip leg count
    (1 = first dip of a chain) and, per chain, its final length."""
    counts = np.ones(len(ends), dtype=int)
    for i in range(1, len(ends)):
        counts[i] = counts[i - 1] + 1 if lows[i] < lows[i - 1] else 1
    # chain final lengths: a chain ends where the next count resets to 1
    finals = [counts[i] for i in range(len(counts))
              if i == len(counts) - 1 or counts[i + 1] == 1]
    return counts, np.array(finals)


def count_hazard(finals: np.ndarray, counts_reached: np.ndarray,
                 nmax: int = 8) -> pd.DataFrame:
    rows = []
    for N in range(1, nmax + 1):
        reached = int((counts_reached == N).sum())     # chains reaching N
        ended = int((finals == N).sum())               # chains dying at N
        rows.append({"N_dips": N, "n_reached": reached,
                     "P_rebound_next": ended / reached if reached else np.nan})
    return pd.DataFrame(rows)


def simulate_counts(close: np.ndarray, ends: np.ndarray, counts: np.ndarray,
                    N: int, horizons=(5, 15, 30, 60)) -> dict:
    """After the Nth lower-low dip completes, buy at the next bar."""
    trig = ends[counts == N]
    trig = trig[trig + max(horizons) + 1 < len(close)]
    out = {"N": N, "n_trades": len(trig)}
    for h in horizons:
        entry = close[trig + 1]
        bps = (close[trig + 1 + h] / entry - 1) * 1e4
        out[f"ret{h}b_bps"] = bps.mean() if len(bps) else np.nan
        out[f"win{h}b"] = (bps > 0).mean() if len(bps) else np.nan
    return out


def analyze_tf(close_1s: pd.Series, bar_s: int):
    if bar_s > 1:
        close = close_1s.resample(f"{bar_s}s").last().ffill().dropna()
    else:
        close = close_1s
    px = close.to_numpy(dtype=float)
    split = int(len(px) * 0.6)
    halves = {"TRAIN": px[:split], "TEST": px[split:]}
    print(f"\n######## timeframe = {bar_s}s bars "
          f"({len(px)} bars total) ########")
    if len(px) < 5000:
        print(f"  WARNING: only {len(px)} bars -- sample-starved, "
              f"fetch more history for this timeframe")
    for name, arr in halves.items():
        ends, lows = dip_events(arr)
        if len(ends) < 50:
            print(f"  {name}: <50 dips, skipping")
            continue
        counts, finals = leg_sequences(ends, lows)
        hz = count_hazard(finals, counts)
        print(f"\n== {name}: leg-count hazard "
              f"P(rebound | N consecutive lower-low dips) ==")
        print(hz.to_string(index=False,
                           float_format=lambda x: f"{x:.4f}"))
        rows = [simulate_counts(arr, ends, counts, N) for N in range(1, 9)]
        df = pd.DataFrame(rows)
        print(f"-- {name}: buy after Nth lower-low dip completes "
              f"(gross bps, horizons in BARS) --")
        print(df.to_string(index=False,
                           float_format=lambda x: f"{x:.2f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="csv with ts,close (e.g. BTC-USD_1s.csv)")
    ap.add_argument("--fetch-candles", type=float, metavar="DAYS",
                    help="fetch DAYS of Coinbase candles instead of csv")
    ap.add_argument("--product", default="BTC-USD")
    ap.add_argument("--tf", type=int, nargs="+", default=[1, 5, 15, 60],
                    help="bar sizes in SECONDS to test")
    ap.add_argument("--control", choices=["rising", "memoryless"])
    a = ap.parse_args()

    if a.control:
        close = control_series(a.control)
        print(f"[control: {a.control} -- rising plants rebound at N=3; "
              f"verify the report recovers it]")
    elif a.fetch_candles:
        g = min(a.tf) if min(a.tf) >= 60 else 60
        close = fetch_candles(a.product, a.fetch_candles, g,
                              f"{a.product}_{g}s_candles.csv")
    elif a.csv:
        close = (pd.read_csv(a.csv, parse_dates=["ts"]).set_index("ts")
                 ["close"].astype(float))
    else:
        ap.error("one of --csv / --fetch-candles / --control required")
    for bar_s in a.tf:
        analyze_tf(close, bar_s)
    print("\nverdict rule: a (timeframe, N) cell counts ONLY if TRAIN "
          "hazard rises over >=3 consecutive N AND TEST return at that N "
          "is positive with n>=100. Anything less is sweep noise.")


if __name__ == "__main__":
    main()
