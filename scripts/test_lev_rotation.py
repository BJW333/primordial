#!/usr/bin/env python3
"""
Test TheOmniscientParadox: single-asset daily momentum rotation.

    python3.10 scripts/test_lev_rotation.py                  # 3x as written
    LEV=1x python3.10 scripts/test_lev_rotation.py           # unleveraged parents
    LEV=1x START=2005-01-01 python3.10 scripts/test_lev_rotation.py

THE RULE, ported:
  score = (0.5*roc9 + 0.3*roc21 + 0.2*roc63) / std21
          * (1.0 if price > sma50 else 0.5)
          * (0.9 if rsi14 > 85 or rsi14 < 30 else 1.0)
  hold the single best; rotate only when best > current * (1 + 0.10);
  fall to BIL when the current score < -0.02; when SPY < sma200, prefer UUP
  or cash. Size to an 80% annualized vol target, capped at 1.0.

WHY THE 1x RUN IS THE ACTUAL TEST
---------------------------------
The universe is SOXL, TECL, TQQQ, FAS, ERX -- five 3x ETFs, chosen in 2026
for a 2019 start, over the largest tech/semi run in history. Ranking those by
momentum will produce a spectacular curve no matter what the ranking does,
because the UNIVERSE is the bet.

LEV=1x swaps in the unleveraged parents (SOXX, XLK, QQQ, XLF, XLE) and
changes nothing else. Same signal, same rules, same dates.

  edge survives at 1x  -> the RANKING has content
  edge only at 3x      -> you found leverage and vol decay, not selection

Leveraged ETFs reset daily, so choppy periods bleed value mechanically. A
momentum filter that holds them only during smooth trends is a device for
harvesting the good half of a product whose bad half is structural. That is
a real effect and a known one -- it is why these backtests look extraordinary
and live results do not.

NULL: turnover-matched. It picks at random among the SAME candidates and
rotates at the rule's own rate, so selection is compared to selection rather
than to churn. Also printed: equal-weight-hold and SPY, because a strategy
that cannot beat holding the basket has not earned its turnover.
"""
from __future__ import annotations

import math
import os
import statistics
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEV = os.environ.get("LEV", "3x")
# The algorithm scores EVERY symbol except self.safe. tickers = [SOXL, TECL,
# TQQQ, FAS, ERX, UUP, TMF, BIL] and safe = BIL, so UUP and TMF are ranking
# candidates in their own right, not just a defensive overlay. Porting them
# as overlay-only understates how often the book sits in bonds or dollars.
# FAIR: every SPDR sector fund (all 9 that existed from 1998, plus the two
# added later) with TLT and UUP for the defensive legs. Chosen by COVERAGE --
# one fund per GICS sector, whatever it did afterwards. Includes the sectors
# that went nowhere for a decade: XLE, XLF, XLU, XLB, XLP.
#
# This is the universe someone could have written down in 2005 without
# knowing that semis and tech would win. The 3x/1x lists could not be: SOXL
# launched 2010, TECL 2008, and picking exactly the five that ran is the
# selection the null cannot see past, because rule and null draw from the
# same five.
UNIVERSE = {
    "3x": ["SOXL", "TECL", "TQQQ", "FAS", "ERX", "UUP", "TMF"],
    "1x": ["SOXX", "XLK", "QQQ", "XLF", "XLE", "UUP", "TLT"],
    "fair": ["XLE", "XLF", "XLK", "XLI", "XLV", "XLP", "XLU", "XLB", "XLY",
             "TLT", "UUP"],
}[LEV]
DEFENSIVE = ["UUP"]
SAFE = "BIL"
START = os.environ.get("START", "2019-01-01")
END = os.environ.get("END", "2026-06-01")
N_NULL = int(os.environ.get("N_NULL", "500"))
COST_BPS = float(os.environ.get("COST_BPS", "4.0"))     # ETF round trip
TARGET_VOL = float(os.environ.get("TARGET_VOL", "0.80"))
CONF = float(os.environ.get("CONF", "0.10"))
HOLDOUT_FRAC = float(os.environ.get("HOLDOUT_FRAC", "0.3"))


def luck_bar(var_sr, n):
    if n < 2 or var_sr <= 0:
        return 0.0
    e = 0.5772156649
    z = statistics.NormalDist()
    return math.sqrt(var_sr) * ((1 - e) * z.inv_cdf(1 - 1.0 / n)
                                + e * z.inv_cdf(1 - 1.0 / (n * math.e)))


def load():
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("pip install yfinance")
    tick = UNIVERSE + DEFENSIVE + [SAFE, "SPY"]
    raw = yf.download(tick, start=START, end=END, auto_adjust=True,
                      progress=False, group_by="column")
    close = raw["Close"].dropna(how="all")
    return close, raw["Open"].reindex(close.index)


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_scores(close):
    """
    Composite score as written. TIMING: the scheduled rebalance fires 5
    minutes before the close, when QC's DAILY indicators still hold
    YESTERDAY's consolidated values, while securities[s].close is TODAY's
    price. So momentum/vol/rsi/sma are shifted one day; only the
    price-vs-sma comparison uses today. That is what the code does, and it
    is not look-ahead -- both are knowable at 15:55.
    """
    out = {}
    for c in set(UNIVERSE + DEFENSIVE):
        if c not in close.columns:
            continue
        p = close[c]
        mom = (0.5 * p.pct_change(9) + 0.3 * p.pct_change(21)
               + 0.2 * p.pct_change(63)).shift(1)
        vol = p.pct_change().rolling(21).std().shift(1).replace(0, np.nan)
        sma = p.rolling(50).mean().shift(1)
        trend = np.where(p > sma, 1.0, 0.5)          # TODAY's price vs prior sma
        r = rsi(p).shift(1)
        pen = np.where((r > 85) | (r < 30), 0.9, 1.0)
        out[c] = (mom / vol) * trend * pen
    return pd.DataFrame(out)


def run(close, open_, score, spy_ok, chooser, seed=None):
    """Daily rotation. Signal from close of t, filled at open of t+1."""
    rng = np.random.default_rng(seed)
    idx = close.index
    rets, held, rot, n = [], None, 0, 0
    realized = close.pct_change().rolling(20).std() * math.sqrt(252)

    # Fills at day i's CLOSE (the algorithm trades 5 min before it), held to
    # day i+1's close. Every input to score.iloc[i] is knowable at 15:55.
    for i in range(70, len(idx) - 1):
        row = score.iloc[i].dropna()
        cands = [c for c in UNIVERSE if c in row.index]
        if not cands:
            continue
        pick = chooser(row, cands, rng, held)

        # SPY regime overlay
        if not spy_ok.iloc[i] and pick != SAFE:
            u = DEFENSIVE[0]
            if u in row.index and row[u] > 0 and row[u] > row.get(pick, -999):
                pick = u
            elif row.get(pick, -999) < 0:
                pick = SAFE

        w = 1.0
        if pick not in (SAFE, None):
            cv = (realized[pick].iloc[i - 1] if pick in realized
                  else np.nan)          # self.history() excludes the live bar
            w = min(1.0, TARGET_VOL / cv) if np.isfinite(cv) and cv > 0 else 1.0

        if pick != held:
            rot += 1
        n += 1
        cost = (COST_BPS / 1e4) if pick != held else 0.0

        r = 0.0
        for sym, wt in ((pick, w), (SAFE, 1.0 - w) if w < 1.0 else (None, 0)):
            if sym and sym in close.columns and wt > 0:
                o0, o1 = close[sym].iloc[i], close[sym].iloc[i + 1]
                if np.isfinite(o0) and np.isfinite(o1) and o0 > 0:
                    r += wt * float(o1 / o0 - 1.0)
        rets.append(r - cost)
        held = pick
    return np.array(rets), rot / max(n, 1)


def rule_chooser(row, cands, rng, held):
    ranked = sorted(cands, key=lambda c: row[c], reverse=True)
    best, bs = ranked[0], row[ranked[0]]
    if held is None:
        return best if bs > 0 else SAFE
    if held == SAFE:
        return best if bs > 0.02 else SAFE
    cs = row.get(held, -999)
    if bs > cs * (1 + CONF):
        return best
    if cs < -0.02:
        return SAFE
    return held


def make_null(p_rot):
    def f(row, cands, rng, held):
        if held in cands and rng.random() > p_rot:
            return held
        return str(rng.choice(cands))
    return f


def tail(r, label, ppy=252):
    """
    Sharpe is the wrong lens for a concentrated single-asset leveraged book.
    Vol targeting budgets VOLATILITY, not loss, and it measures vol trailing
    20 days -- so position size is largest right before a regime turn, when
    realized vol is still low. Momentum's known failure mode is a crash at
    reversals, because the book is maximally concentrated in whatever just
    ran. These numbers show that; the Sharpe hides it.
    """
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return {}
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    dd = 1 - eq / peak
    mdd = float(dd.max())
    under = int((dd > 0.05).sum())
    cagr = float(eq[-1] ** (ppy / len(r)) - 1)
    worst = float(np.sort(r)[:max(1, len(r) // 20)].mean())   # 5% CVaR
    print(f"  {label:24s} maxDD {mdd * 100:5.1f}% | Calmar "
          f"{(cagr / mdd if mdd > 0 else float('nan')):5.2f} | "
          f"worst day {r.min() * 100:6.1f}% | CVaR5 {worst * 100:5.2f}% | "
          f"days >5% underwater {under / len(r) * 100:4.0f}%")
    return {"mdd": mdd, "calmar": cagr / mdd if mdd > 0 else np.nan}


def sr(r, ppy=252):
    r = np.asarray(r, float)
    return (float(r.mean() / r.std(ddof=1) * math.sqrt(ppy))
            if len(r) > 20 and r.std(ddof=1) > 0 else float("nan"))


def main():
    print(f"OmniscientParadox | LEV={LEV} | {','.join(UNIVERSE)} | "
          f"{START} -> {END} | {COST_BPS:.0f} bps | vol target {TARGET_VOL}")
    if LEV == "3x":
        print("  !! 3x universe picked in 2026 for a 2019 start. Run LEV=1x --")
        print("     that is the test of whether the RANKING has content.\n")
    else:
        print("  1x parents: same signal, same rules, leverage removed.\n")

    close, open_ = load()
    close = close.ffill(limit=3)
    print(f"loaded {close.shape[1]} tickers, {len(close)} sessions")
    spy_ok = close["SPY"] > close["SPY"].rolling(200).mean()

    score = build_scores(close)
    r_rule, rot = run(close, open_, score, spy_ok, rule_chooser)
    n = len(r_rule)
    print(f"  {n} sessions traded | rotated {rot:.0%} of days "
          f"| fee drag {rot * COST_BPS * 252 / 100:.1f}%/yr\n")

    print("FULL PERIOD")
    print(f"  rule                     SR {sr(r_rule):+.2f} | "
          f"total {(np.prod(1 + r_rule) - 1) * 100:+.0f}%")
    eqc = [c for c in UNIVERSE if c in close.columns]
    eq = close[eqc].pct_change().mean(axis=1).iloc[71:71 + n].values
    print(f"  equal-weight all (hold)  SR {sr(eq):+.2f} | "
          f"total {(np.prod(1 + eq) - 1) * 100:+.0f}%")
    spy = close["SPY"].pct_change().iloc[71:71 + n].values
    print(f"  SPY buy & hold           SR {sr(spy):+.2f} | "
          f"total {(np.prod(1 + spy) - 1) * 100:+.0f}%")

    print("\nDRAWDOWN / TAIL  (a concentrated leveraged book lives or dies "
          "here, not on Sharpe)")
    d_rule = tail(r_rule, "rule")
    tail(eq, "equal-weight all (hold)")
    tail(spy, "SPY buy & hold")

    cut = int(n * (1 - HOLDOUT_FRAC))
    tr, ho = sr(r_rule[:cut]), sr(r_rule[cut:])
    print(f"\nSPLIT  train {cut} / holdout {n - cut}")
    print(f"  rule train {tr:+.2f} | holdout {ho:+.2f}")

    print(f"\nNULL: {N_NULL} draws, random pick among the same candidates, "
          f"rotating at the rule's own {rot:.0%} rate")
    ntr, nho, ndd = [], [], []
    for k in range(N_NULL):
        rk, _ = run(close, open_, score, spy_ok, make_null(rot), seed=k)
        if len(rk) == n:
            e = np.cumprod(1 + rk)
            ndd.append(float((1 - e / np.maximum.accumulate(e)).max()))
            a, b = sr(rk[:cut]), sr(rk[cut:])
            if np.isfinite(a):
                ntr.append(a)
            if np.isfinite(b):
                nho.append(b)
    ntr, nho = np.array(ntr), np.array(nho)
    if len(nho) < 30:
        print("  null too small")
        return 1
    if ndd:
        nd = np.array(ndd)
        pct = float((nd > d_rule.get("mdd", np.nan)).mean() * 100)
        print(f"  null maxDD: median {np.median(nd) * 100:.1f}% | "
              f"best {nd.min() * 100:.1f}% | worst {nd.max() * 100:.1f}% "
              f"| rule is better than {pct:.0f}% of random picks")

    mad = float(np.median(np.abs(nho - np.median(nho))))
    bar = luck_bar(float(np.clip((1.4826 * mad) ** 2, 1e-6, 25.0)), 12)
    print(f"  TRAIN   rule {tr:+.2f} | null {ntr.mean():+.2f} "
          f"sd {ntr.std(ddof=1):.2f} | {(ntr < tr).mean() * 100:5.1f}th pct "
          f"| {(tr - ntr.mean()) / ntr.std(ddof=1):+.2f} sd")
    print(f"  HOLDOUT rule {ho:+.2f} | null {nho.mean():+.2f} "
          f"sd {nho.std(ddof=1):.2f} | {(nho < ho).mean() * 100:5.1f}th pct "
          f"| {(ho - nho.mean()) / nho.std(ddof=1):+.2f} sd")
    print(f"  luck bar over 12 attempted trials = {bar:+.2f}")

    print("\n" + "=" * 64)
    print("The comparison that decides it is 3x vs 1x. If the ranking only")
    print("works on leveraged products, it is harvesting the smooth half of")
    print("daily-reset decay -- known, and not what the strategy claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
