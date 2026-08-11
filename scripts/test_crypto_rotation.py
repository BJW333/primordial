#!/usr/bin/env python3
"""
Test the CryptoTrifecta rotation as written.

    python3.10 scripts/test_crypto_rotation.py
    COINS=BTC,ETH,SOL,LINK,LTC,ADA python3.10 scripts/test_crypto_rotation.py
    LOOKBACK=72 REBAL=12 THRESH=0.01 python3.10 scripts/test_crypto_rotation.py

THE RULE, ported exactly:
  every REBAL hours, score each coin by ROC over the last LOOKBACK hours
  qualify if roc > THRESH AND last price > mean(window)
  hold the single highest-scoring coin; if none qualify, sit in cash

FOUR THINGS THIS SEPARATES, because the headline number confuses all of them:

1. FEES. 730 rebalances a year concentrated into one name at Coinbase taker
   rates. The run prints gross and net side by side. If net is deeply
   negative while gross is positive, the signal is irrelevant -- you cannot
   pay for it.

2. THE SELF-REFERENTIAL FILTER. `sma = closes.mean()` uses the SAME window
   as roc. If a series rose over the window, its last price is almost
   mechanically above the window's mean, so the "trend confirmation" largely
   re-tests what roc already tested. SMA=off reruns without it. If nothing
   changes, the filter was never doing work.

3. SELECTION vs BEING IN CRYPTO AT ALL. 2020-2026 contains the largest bull
   run in the asset's history. Buying and holding the basket is the baseline;
   BTC alone is the other. Rotation has to beat both.

4. SELECTION vs LUCK. The null picks at random from the coins that QUALIFIED
   on that bar, and is turnover-matched -- it rotates as often as the rule
   does. That isolates "picking the best of the qualifiers" from "being in
   a qualifying coin at all", which is the actual claim.

SURVIVORSHIP: XRP, DOT, AVAX, ETH, BTC were chosen in 2026 for a 2020 start.
AVAX and DOT did not trade until late 2020. Thousands of coins existed in
2020 and most are gone. This universe is five winners picked with hindsight,
and no null here repairs that -- rule and null draw from the same five.
A negative result is real evidence; a positive one is not.
"""
from __future__ import annotations

import math
import os
import statistics
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primordial import data as D                      # noqa: E402
from primordial.universe import COST_TIERS            # noqa: E402

COINS = os.environ.get("COINS", "XRP,DOT,AVAX,ETH,BTC").split(",")
LOOKBACK = int(os.environ.get("LOOKBACK", "72"))       # hours
REBAL = int(os.environ.get("REBAL", "12"))             # hours
THRESH = float(os.environ.get("THRESH", "0.01"))
USE_SMA = os.environ.get("SMA", "on") == "on"
START = os.environ.get("START", "2021-01-01")
END = os.environ.get("END", "2026-06-01")
N_NULL = int(os.environ.get("N_NULL", "500"))
TIER = os.environ.get("TIER", "crypto_spot_taker")
HOLDOUT_FRAC = float(os.environ.get("HOLDOUT_FRAC", "0.3"))

COST = COSTS = COST_TIERS[TIER]
RT_BPS = COST.adverse_frac() * 2 * 1e4
PPY = 365 * 24 / REBAL          # rebalances per year


def luck_bar(var_sr: float, n: int) -> float:
    if n < 2 or var_sr <= 0:
        return 0.0
    e = 0.5772156649
    z = statistics.NormalDist()
    return math.sqrt(var_sr) * ((1 - e) * z.inv_cdf(1 - 1.0 / n)
                                + e * z.inv_cdf(1 - 1.0 / (n * math.e)))


def load() -> pd.DataFrame:
    syms = [f"{c}-USD" for c in COINS]
    raw = D.fetch("coinbase", syms, "1h", START, END)
    if not raw:
        sys.exit("no data -- check symbols/date range")
    px = pd.DataFrame({s.split("-")[0]: d["close"] for s, d in raw.items()})
    return px.sort_index().ffill(limit=3)


def qualifiers(px: pd.DataFrame, t: int):
    """(scores, qualifying coins) at bar index t, using data through t only."""
    win = px.iloc[max(0, t - LOOKBACK + 1): t + 1]
    if len(win) < max(4, LOOKBACK // 2):
        return {}, []
    first, last, mean = win.iloc[0], win.iloc[-1], win.mean()
    roc = (last - first) / first.replace(0, np.nan)
    ok = []
    for c in px.columns:
        r = roc.get(c, np.nan)
        if not np.isfinite(r) or r <= THRESH:
            continue
        if USE_SMA and not (last[c] > mean[c]):
            continue
        ok.append(c)
    return roc.to_dict(), ok


def run(px: pd.DataFrame, chooser, seed=None):
    """Returns (per-period net returns, gross returns, rotation count)."""
    rng = np.random.default_rng(seed)
    idx = list(range(LOOKBACK, len(px) - REBAL, REBAL))
    net, gross, held, rot = [], [], None, 0
    for t in idx:
        roc, ok = qualifiers(px, t)
        pick = chooser(roc, ok, rng, held)
        if pick is None:
            g = 0.0
        else:
            p0 = px[pick].iloc[t]
            p1 = px[pick].iloc[min(t + REBAL, len(px) - 1)]
            g = float(p1 / p0 - 1.0) if np.isfinite(p0) and p0 > 0 else 0.0
        cost = 0.0
        if pick != held:
            rot += 1
            # full rotation = sell the old, buy the new
            cost = RT_BPS / 1e4 if (held and pick) else RT_BPS / 2e4
        gross.append(g)
        net.append(g - cost)
        held = pick
    return np.array(net), np.array(gross), rot


def rule_chooser(roc, ok, rng, held):
    return max(ok, key=lambda c: roc[c]) if ok else None


def make_null_chooser(p_rot):
    """Random pick among QUALIFIERS, rotating at the rule's own rate."""
    def f(roc, ok, rng, held):
        if not ok:
            return None
        if held in ok and rng.random() > p_rot:
            return held
        return str(rng.choice(ok))
    return f


def sharpe(r):
    r = np.asarray(r, float)
    return (float(r.mean() / r.std(ddof=1) * math.sqrt(PPY))
            if len(r) > 8 and r.std(ddof=1) > 0 else float("nan"))


def main() -> int:
    print(f"CryptoTrifecta | {','.join(COINS)} | lookback {LOOKBACK}h | "
          f"rebal {REBAL}h | thresh {THRESH} | SMA {'on' if USE_SMA else 'off'}")
    print(f"  {TIER}: {RT_BPS:.0f} bps round trip | {PPY:.0f} rebalances/yr")
    print(f"  !! universe is 5 coins chosen in 2026 for a {START[:4]} start. "
          f"Survivorship flatters; a POSITIVE result proves nothing.\n")

    px = load()
    print(f"loaded {px.shape[1]} coins, {len(px)} hourly bars "
          f"({px.index[0].date()} -> {px.index[-1].date()})")
    first = {c: px[c].first_valid_index() for c in px.columns}
    late = {c: str(v.date()) for c, v in first.items()
            if v is not None and v > px.index[0] + pd.Timedelta(days=30)}
    if late:
        print(f"  entering late: {late}")

    net, gross, rot = run(px, rule_chooser)
    n = len(net)
    rot_rate = rot / max(n, 1)
    print(f"\n{n} rebalance periods | rotated {rot} times ({rot_rate:.0%})")
    print(f"  fee drag: {rot_rate * RT_BPS * PPY / 100:.0f}%/yr at this "
          f"rotation rate")

    print("\nFULL PERIOD")
    print(f"  rule GROSS (no fees)     SR {sharpe(gross):+.2f} | "
          f"total {(np.prod(1 + gross) - 1) * 100:+.0f}%")
    print(f"  rule NET                 SR {sharpe(net):+.2f} | "
          f"total {(np.prod(1 + net) - 1) * 100:+.0f}%")
    eq = px.pct_change(REBAL).iloc[LOOKBACK::REBAL].mean(axis=1).dropna().values
    print(f"  equal-weight all (hold)  SR {sharpe(eq):+.2f} | "
          f"total {(np.prod(1 + eq) - 1) * 100:+.0f}%")
    btc = px["BTC"].pct_change(REBAL).iloc[LOOKBACK::REBAL].dropna().values \
        if "BTC" in px else np.array([])
    if len(btc):
        print(f"  BTC buy & hold           SR {sharpe(btc):+.2f} | "
              f"total {(np.prod(1 + btc) - 1) * 100:+.0f}%")

    cut = int(n * (1 - HOLDOUT_FRAC))
    sr_tr, sr_ho = sharpe(net[:cut]), sharpe(net[cut:])
    print(f"\nSPLIT  train {cut} / holdout {n - cut}")
    print(f"  rule NET train {sr_tr:+.2f} | holdout {sr_ho:+.2f}")

    print(f"\nNULL: {N_NULL} draws, random pick among QUALIFIERS, "
          f"rotating at the rule's own {rot_rate:.0%} rate")
    ns_tr, ns_ho = [], []
    for k in range(N_NULL):
        nk, _, _ = run(px, make_null_chooser(rot_rate), seed=k)
        if len(nk) == n:
            a, b = sharpe(nk[:cut]), sharpe(nk[cut:])
            if np.isfinite(a):
                ns_tr.append(a)
            if np.isfinite(b):
                ns_ho.append(b)
    ns_tr, ns_ho = np.array(ns_tr), np.array(ns_ho)
    if len(ns_ho) < 30:
        print("  null too small")
        return 1

    mad = float(np.median(np.abs(ns_ho - np.median(ns_ho))))
    var_sr = float(np.clip((1.4826 * mad) ** 2, 1e-6, 25.0))
    bar = luck_bar(var_sr, 12)
    print(f"  TRAIN   rule {sr_tr:+.2f} | null mean {ns_tr.mean():+.2f} "
          f"sd {ns_tr.std(ddof=1):.2f} | "
          f"{(ns_tr < sr_tr).mean() * 100:5.1f}th pct | "
          f"{(sr_tr - ns_tr.mean()) / ns_tr.std(ddof=1):+.2f} sd")
    print(f"  HOLDOUT rule {sr_ho:+.2f} | null mean {ns_ho.mean():+.2f} "
          f"sd {ns_ho.std(ddof=1):.2f} | "
          f"{(ns_ho < sr_ho).mean() * 100:5.1f}th pct | "
          f"{(sr_ho - ns_ho.mean()) / ns_ho.std(ddof=1):+.2f} sd")
    print(f"  luck bar over 12 attempted trials = {bar:+.2f}")

    print("\n" + "=" * 62)
    print("Read in this order: (1) is NET positive at all? (2) does it beat "
          "equal-weight\nand BTC? (3) is it above its null in BOTH windows? "
          "A yes to 1 and 2 but not 3\nmeans you are being paid for holding "
          "crypto, not for choosing which one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
