#!/usr/bin/env python3
"""kill_test_options_expression.py -- can deep-state's moves pay for calls?

FEASIBILITY SCREEN, NOT A GAUNTLET RUN. No holdout is burned: this replays
the ALREADY-VALIDATED frozen genome on its own training-era data and asks a
cost question, not an edge question. Outcome goes to MECHANISM_MAP.md as an
infrastructure decision (buy chain data / don't), never to SURVIVORS.md.

THE QUESTION: long calls only pay if realized moves over the hold exceed
what implied vol was charging at entry. No chain history exists here, so IV
is proxied as trailing realized vol x a variance-risk-premium multiplier
(IV > RV on average -- that premium is exactly the toll a call buyer pays).
Every trade is repriced two ways on identical underlying prices:

    stock  : (S1 - S0)/S0 - 2 x adverse_frac        (the delta-1 null)
    option : Black-Scholes call, buy at ask / sell at bid via a spread
             haircut, delta-equivalent sized so exposure matches the stock

DECISION RULE (pre-stated; the script enforces it, you don't interpret):
    HONEST cells = vrp >= 1.15 AND spread >= 6%   (4 cells per structure)
    ANCHOR cell  = (vrp 1.15, spread 6%) -- the realistic center for the
                   liquid names deep-state actually trades
    ALIVE  = some structure beats the stock null's mean per-$-exposure
             P&L in >= 3 of its 4 honest cells INCLUDING the anchor cell.
             The (1.30, 10%) double-worst corner is reported as a stress
             reading but holds no veto: both worst-case assumptions at
             once models an illiquid single name, not this universe.
             Note delta-equivalent sizing cancels the drift channel
             between legs by construction -- only gamma minus toll is
             being scored, which is the correct and hardest test.
    DEAD   = anything else. DEAD closes the options-expression idea for
             deep-state; next look would need a different validated signal.
    ALIVE only unlocks the NEXT decision (price ORATS/CBOE chain data for a
    real registered test) -- it is not itself evidence of an edge.

POLICY CONSTANTS (fixed ex-ante, derived from the genome, not from results):
    DTE  = 60 trading days  -- > max_hold gene (40 bars), so no forced-close
                               path exists and no early-expiry branch is hit
    ATM  : K = S0
    ITM  : K = S0 * exp(-0.30 * iv * sqrt(T))   (~0.62 delta at entry)
    r    = 0 (short-dated, rate term is noise next to spread+vrp)
    sigma: 21d close-to-close, annualized, read through the bar BEFORE entry

Underlying price convention: entry = open of entry bar (the backtester's
actual fill bar), exit = close of exit bar. Same prices feed both legs, so
the comparison cannot be biased by the convention.

Run (uses the same parquet cache the dsval runs built; missing symbols
fetch on first use):
    python3.10 scripts/kill_test_options_expression.py \
        --manifest manifests/us_sp400.yaml --manifest manifests/us_sp600.yaml
Self-test on planted synthetic worlds (run FIRST, must print both PASSes):
    python3.10 scripts/kill_test_options_expression.py --selftest
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primordial.data import fetch, BAR_SECONDS                    # noqa: E402
from primordial import atoms as A                                 # noqa: E402
from primordial.genome import Genome, run_backtest                # noqa: E402
from primordial.universe import Manifest                          # noqa: E402

# ------------------------------------------------------------ policy grid
DTE_TRADING = 60                    # trading days; > max_hold gene = 40
VRP_GRID = [1.00, 1.15, 1.30]
SPREAD_GRID = [0.03, 0.06, 0.10]    # haircut per side, frac of premium
STRUCTURES = ["atm", "itm"]
HONEST = [(v, s) for v in VRP_GRID for s in SPREAD_GRID
          if v >= 1.15 and s >= 0.06]
ANCHOR = (1.15, 0.06)
SIGMA_LOOKBACK = 21
ANN = 252.0


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S, K, iv, T):
    """Black-Scholes call, r=0. Returns (price, delta)."""
    if T <= 0 or iv <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)
    st = iv * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / st
    return S * _ncdf(d1) - K * _ncdf(d1 - st), _ncdf(d1)


def collect_trades(manifest_paths, genome_path):
    """Replay the frozen genome per symbol; emit one row per multi-bar trade:
    (sym, S0=open@entry, S1=close@exit, hold_bars, sigma_daily@entry,
     sigma_daily@exit, adverse)."""
    g = Genome.from_json(open(genome_path).read())
    if g.timeframe != "1d":
        sys.exit(f"genome timeframe {g.timeframe} != 1d -- this screen "
                 "prices daily-horizon options only.")
    rows = []
    for mp in manifest_paths:
        mf = Manifest.load(mp)
        adverse = mf.cost.adverse_frac()
        data = fetch(mf.source, mf.symbols, "1d", mf.start, mf.end)
        print(f"[{mf.name}] {len(data)}/{len(mf.symbols)} symbols cached")
        for sym, df in data.items():
            df = df.copy()          # atoms attach in place; keep cache pure
            A.compute_atoms(df, mf.adapter.atom_blocks, mf.adapter.always_open)
            lr = np.log(df["close"] / df["close"].shift(1))
            sig = (lr.rolling(SIGMA_LOOKBACK).std() * math.sqrt(ANN)) \
                .shift(1).to_numpy()                    # through PRIOR bar
            res = run_backtest(g, df, mf.cost, bar_seconds=BAR_SECONDS["1d"])
            loc = {t: i for i, t in enumerate(df.index)}
            for (t0, t1, side, _r) in res.trades:
                i0, i1 = loc[t0], loc[t1]
                if i1 <= i0 or side != "long":
                    continue                            # same-bar / defensive
                s_in, s_out = sig[i0], sig[i1]
                if not (np.isfinite(s_in) and s_in > 0
                        and np.isfinite(s_out) and s_out > 0):
                    continue
                rows.append((sym, float(df["open"].iloc[i0]),
                             float(df["close"].iloc[i1]),
                             i1 - i0, float(s_in), float(s_out), adverse))
    return rows


def price_grid(rows):
    """Reprice every trade in every (structure, vrp, spread) cell."""
    out = {}
    stock = [(S1 - S0) / S0 - 2.0 * adv for _, S0, S1, _, _, _, adv in rows]
    zs = [math.log(S1 / S0) / (sig0 / math.sqrt(ANN) * math.sqrt(h))
          for _, S0, S1, h, sig0, _, _ in rows]
    out["n_trades"] = len(rows)
    out["stock_mean"] = float(np.mean(stock))
    out["stock_win"] = float(np.mean([1.0 if x > 0 else 0.0 for x in stock]))
    out["mean_signed_z"] = float(np.mean(zs))       # >0: moves beat the vol
    out["mean_abs_z"] = float(np.mean(np.abs(zs)))  # ~0.8 = walk baseline
    out["cells"] = {}
    T0 = DTE_TRADING / ANN
    for struct in STRUCTURES:
        for vrp in VRP_GRID:
            for sp in SPREAD_GRID:
                pnl, prem_ret = [], []
                for (_, S0, S1, h, sg0, sg1, _adv) in rows:
                    iv0, iv1 = sg0 * vrp, sg1 * vrp
                    K = S0 if struct == "atm" else \
                        S0 * math.exp(-0.30 * iv0 * math.sqrt(T0))
                    p0, d0 = bs_call(S0, K, iv0, T0)
                    p1, _ = bs_call(S1, K, iv1, T0 - h / ANN)
                    buy = p0 * (1.0 + sp)
                    sell = p1 * (1.0 - sp)
                    if buy <= 0 or d0 <= 0:
                        continue
                    prem_ret.append(sell / buy - 1.0)
                    pnl.append((sell - buy) / (S0 * d0))   # per $ exposure
                out["cells"][f"{struct}|vrp{vrp:.2f}|sp{int(sp*100)}"] = {
                    "mean_pnl_delta_eq": float(np.mean(pnl)),
                    "mean_ret_per_premium": float(np.mean(prem_ret)),
                    "median_ret_per_premium": float(np.median(prem_ret)),
                    "win": float(np.mean([1.0 if x > 0 else 0.0
                                          for x in prem_ret])),
                }
    return out


def verdict(res):
    alive = []
    for struct in STRUCTURES:
        beats = {(v, s): res["cells"][f"{struct}|vrp{v:.2f}|sp{int(s*100)}"]
                 ["mean_pnl_delta_eq"] > res["stock_mean"]
                 for v, s in HONEST}
        if sum(beats.values()) >= 3 and beats[ANCHOR]:
            alive.append(struct)
    res["verdict"] = ("ALIVE:" + ",".join(alive)) if alive else "DEAD"
    return res


def report(res):
    print(f"\ntrades={res['n_trades']}  stock mean/trade="
          f"{res['stock_mean']*100:+.2f}%  win={res['stock_win']*100:.0f}%")
    print(f"signed z={res['mean_signed_z']:+.3f}  |z|={res['mean_abs_z']:.3f}"
          "   (|z| ~0.80 is what a pure random walk gives)")
    print(f"\n{'cell':>22} {'d-eq pnl':>9} {'/premium':>9} {'win':>5}"
          f"   vs stock {res['stock_mean']*100:+.2f}%")
    for k, c in res["cells"].items():
        tag = " HONEST" if any(f"vrp{v:.2f}|sp{int(s*100)}" in k
                               for v, s in HONEST) else ""
        beat = "+" if c["mean_pnl_delta_eq"] > res["stock_mean"] else " "
        print(f"{k:>22} {c['mean_pnl_delta_eq']*100:+8.2f}% "
              f"{c['mean_ret_per_premium']*100:+8.1f}% "
              f"{c['win']*100:4.0f}% {beat}{tag}")
    print(f"\nVERDICT: {res['verdict']}")
    print("DEAD  -> options expression closed for deep-state; log to "
          "MECHANISM_MAP.md.\nALIVE -> next step is pricing real chain data "
          "(ORATS / CBOE DataShop);\n         nothing is tradeable off this "
          "proxy alone.")


# ----------------------------------------------------------------- selftest
def _synth_rows(n, sigma_ann, jump, seed):
    """Planted world: entry sigma is known truth; realized move is a GBM step
    over `h` bars, optionally scaled by `jump`. jump=1 -> realized == priced
    (calls must lose the toll). jump=2 -> realized doubles what IV charges
    (calls must win)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        S0 = 100.0 * float(rng.uniform(0.5, 2.0))
        h = int(rng.integers(5, 35))
        sd = sigma_ann / math.sqrt(ANN)
        s = sd * math.sqrt(h) * jump
        # -s^2/2 drift => E[S1/S0] = 1 exactly: the planted stock null is
        # driftless in ARITHMETIC terms, so no Jensen term leaks an edge
        # into (or against) either leg. jump only widens the distribution.
        step = float(rng.normal(-0.5 * s * s, s))
        rows.append(("SYN", S0, S0 * math.exp(step), h,
                     sigma_ann, sigma_ann, 0.0005))
    return rows


def selftest():
    dead = verdict(price_grid(_synth_rows(6000, 0.30, 1.0, 1)))
    print(f"planted NULL (realized==priced): |z|={dead['mean_abs_z']:.3f} "
          f"-> {dead['verdict']}")
    assert dead["verdict"] == "DEAD", "harness failed: null world came ALIVE"
    assert 0.70 < dead["mean_abs_z"] < 0.90, "z calibration off in null world"
    live = verdict(price_grid(_synth_rows(6000, 0.30, 2.0, 2)))
    print(f"planted 2x-move world:           |z|={live['mean_abs_z']:.3f} "
          f"-> {live['verdict']}")
    assert live["verdict"].startswith("ALIVE"), \
        "harness failed: 2x-move world came DEAD"
    print("SELFTEST PASS: overlay recovers both planted answers.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--genome", default="genomes/deepstate_frozen.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    mans = a.manifest or ["manifests/us_sp400.yaml", "manifests/us_sp600.yaml"]
    rows = collect_trades(mans, a.genome)
    if len(rows) < 30:
        sys.exit(f"only {len(rows)} usable trades -- not enough to decide; "
                 "check the price cache covers these manifests.")
    res = verdict(price_grid(rows))
    report(res)
    os.makedirs("out", exist_ok=True)
    with open("out/options_killtest.json", "w") as f:
        json.dump(res, f, indent=1)
    print("\nwritten: out/options_killtest.json")


if __name__ == "__main__":
    main()
