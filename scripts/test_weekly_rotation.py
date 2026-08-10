#!/usr/bin/env python3
"""
Weekly (or monthly) cross-sectional rotation on S&P 400 / 500 / 600 names.

    UNIV=sp400 REBAL=weekly  python3.10 scripts/test_weekly_rotation.py
    UNIV=sp400 REBAL=monthly python3.10 scripts/test_weekly_rotation.py
    UNIV=sp500 REBAL=weekly  python3.10 scripts/test_weekly_rotation.py

Same machinery that killed the ETF rotation: rank by composite ROC, hold the
top K, inverse-vol weight, score against a RANDOM-PICK NULL with identical
mechanics. If the ranking cannot beat picking K names at random from the same
universe with the same weighting and the same costs, the ranking is
decoration.

READ THIS BEFORE BELIEVING A POSITIVE RESULT
--------------------------------------------
The symbol lists are TODAY'S membership. Names delisted, acquired or demoted
are missing. For small caps that bias is severe and it points ONE WAY: it
flatters. Every name in sp600_symbols.txt survived to 2026.

So only one direction is informative:

  NEGATIVE result -> REAL EVIDENCE. It failed despite the tailwind.
  POSITIVE result -> PROVES NOTHING until re-run on point-in-time membership.

The random-pick null does NOT fix this -- rule and null draw from the same
survivor pool, so the comparison is fair but the pool is not. Same trap the
ETF test had, except there is no easy hindsight-free equity universe to fall
back on.

COSTS DOMINATE. 52 rebalances a year at small-cap spreads is 7-15%/yr of drag
before you are right about anything. Always run REBAL=monthly on the same
signal to see how much of a result is just paying the spread less often.
"""
from __future__ import annotations

import math
import os
import statistics
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ------------------------------------------------------------------ config
UNIV = os.environ.get("UNIV", "sp400")
REBAL = os.environ.get("REBAL", "weekly")

_FILES = {"sp400": ("sp400_symbols.txt", "equity_midcap"),
          "sp500": ("sp500_symbols.txt", "equity_liquid"),
          "sp600": ("sp600_symbols.txt", "equity_smallcap"),
          "sp600top150": ("sp600_symbols_top150.txt", "equity_smallcap")}
_TIER_BPS = {"equity_liquid": 4.0, "equity_midcap": 14.0,
             "equity_smallcap": 36.0}


def _load_symbols(name):
    fn, tier = _FILES[name]
    p = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "manifests", fn)
    syms = [ln.strip() for ln in open(p)
            if ln.strip() and not ln.startswith("#")]
    return syms, tier


TICKERS, COST_TIER = _load_symbols(UNIV)
ROC_PERIODS = [5, 21, 63, 126, 252]
VOL_PERIOD = 63
TOP_N = int(os.environ.get("TOP_N", "20"))

START = os.environ.get("START", "2010-01-01")
END = os.environ.get("END", "2026-06-01")
N_NULL = int(os.environ.get("N_NULL", "500"))
COST_BPS = float(os.environ.get("COST_BPS", _TIER_BPS[COST_TIER]))
HOLDOUT_FRAC = float(os.environ.get("HOLDOUT_FRAC", "0.3"))


def luck_bar(var_sr: float, n: int) -> float:
    """Expected max Sharpe among n null trials (Bailey/Lopez de Prado)."""
    if n < 2 or var_sr <= 0:
        return 0.0
    e = 0.5772156649
    z = statistics.NormalDist()
    return math.sqrt(var_sr) * ((1 - e) * z.inv_cdf(1 - 1.0 / n)
                                + e * z.inv_cdf(1 - 1.0 / (n * math.e)))


def psr(returns: np.ndarray, benchmark: float = 0.0) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 8 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    m3 = float(((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3)
    m4 = float(((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4)
    den = 1.0 - m3 * sr + ((m4 - 1.0) / 4.0) * sr ** 2
    if den <= 0:
        return float("nan")
    z = (sr - benchmark) * math.sqrt(T - 1) / math.sqrt(den)
    return float(statistics.NormalDist().cdf(z))


def load_prices() -> pd.DataFrame:
    """Daily adjusted closes AND opens. Fills happen at the next open."""
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("pip install yfinance  (Alpaca equity history starts ~2016; "
                 "this test needs 2005)")
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True,
                      progress=False, group_by="column")
    close = raw["Close"].dropna(how="all")
    open_ = raw["Open"].reindex(close.index)
    keep = [t for t in TICKERS if t in close.columns]
    return close[keep], open_[keep]


def composite_score(close: pd.DataFrame) -> pd.DataFrame:
    """Mean of ROC over the five lookbacks -- the algorithm's own score."""
    parts = [close.pct_change(p) for p in ROC_PERIODS]
    return sum(parts) / len(parts)


PPY = 52 if REBAL == "weekly" else 12


def rebalance_dates(idx: pd.DatetimeIndex) -> list:
    """First trading day of each week (weekly) or month (monthly)."""
    ser = pd.Series(idx, index=idx)
    if REBAL == "weekly":
        iso = idx.isocalendar()
        return list(ser.groupby([iso.year.values,
                                 iso.week.values]).first().values)
    return list(ser.groupby([idx.year, idx.month]).first().values)


def run_strategy(close, open_, score, vol, dates, picker, seed=None,
                 allowed=None):
    """
    picker(scores_row, eligible) -> list of tickers.
    Returns (monthly_returns, avg_n_positions, turnover_per_rebal).
    """
    rng = np.random.default_rng(seed)
    rets, weights_prev, n_pos, turns = [], {}, [], []
    swaps = []                       # names replaced per rebalance

    for i in range(len(dates) - 1):
        d0, d1 = pd.Timestamp(dates[i]), pd.Timestamp(dates[i + 1])
        # signal uses data through the PREVIOUS session; fill at d0's open
        prior = close.index[close.index < d0]
        if len(prior) < max(ROC_PERIODS) + 5:
            continue
        t_sig = prior[-1]
        s_row = score.loc[t_sig]
        v_row = vol.loc[t_sig]
        pool = allowed if allowed is not None else close.columns
        elig = [t for t in pool
                if np.isfinite(s_row.get(t, np.nan))
                and np.isfinite(v_row.get(t, np.nan))
                and v_row.get(t, 0) > 0]
        if len(elig) < TOP_N:
            continue

        picks = picker(s_row, elig, rng)
        if not picks:                      # cash month
            rets.append(0.0)
            turns.append(sum(abs(w) for w in weights_prev.values()))
            weights_prev, _ = {}, n_pos.append(0)
            continue

        iv = np.array([1.0 / float(v_row[t]) for t in picks])
        w = iv / iv.sum()
        wmap = dict(zip(picks, w))

        swaps.append(len(set(picks) - set(weights_prev)))
        # turnover vs previous book -> cost
        names = set(wmap) | set(weights_prev)
        turn = sum(abs(wmap.get(t, 0.0) - weights_prev.get(t, 0.0))
                   for t in names)
        # hold from d0 open to d1 open
        try:
            px0 = open_.loc[d0, picks].astype(float)
            px1 = open_.loc[d1, picks].astype(float)
        except KeyError:
            continue
        if not np.isfinite(px0).all() or not np.isfinite(px1).all():
            continue
        leg = float(((px1 / px0 - 1.0) * w).sum())
        rets.append(leg - turn * COST_BPS / 1e4)
        turns.append(turn)
        n_pos.append(len(picks))
        weights_prev = wmap

    return (np.array(rets, float),
            float(np.mean(n_pos)) if n_pos else 0.0,
            float(np.mean(turns)) if turns else 0.0,
            swaps)


# ATTRIBUTION SWITCHES.
# The strategy has TWO mechanisms and they make different claims:
#   RANKING   - relative momentum persists across assets
#   CASH      - absolute momentum; get out when everything is falling
# The original null had neither, which quietly credited the ranking with the
# crash protection. CASH=off turns the rule's switch off; NULLCASH=on gives
# the null the same switch. Run the four corners to separate them.
RULE_CASH = os.environ.get("CASH", "on") == "on"
NULL_CASH = os.environ.get("NULLCASH", "off") == "on"


CASH_FIRES = [0]


def real_picker(s_row, elig, rng):
    ranked = sorted(elig, key=lambda t: float(s_row[t]), reverse=True)
    top = ranked[:TOP_N]
    if all(float(s_row[t]) < 0 for t in top):
        CASH_FIRES[0] += 1                 # counted even when disabled
        if RULE_CASH:
            return []
    return top


def random_picker(s_row, elig, rng):
    """Unmatched null: rebuilds the whole book every rebalance."""
    picks = list(rng.choice(elig, size=min(TOP_N, len(elig)), replace=False))
    if NULL_CASH and picks and all(float(s_row[t]) < 0 for t in picks):
        return []
    return picks


class MatchedPicker:
    """
    TURNOVER-MATCHED null. Momentum is sticky, so the rule keeps most of its
    book each period. A null that re-draws all TOP_N names every time trades
    ~4x as much and eats ~4x the spread -- that is a cost handicap, not worse
    stock picking. This one holds a book and replaces exactly as many names
    as the rule replaced at the same rebalance, chosen at random. Selection
    vs selection, with cost held equal.
    """

    def __init__(self, swaps):
        self.swaps = list(swaps)
        self.i = 0
        self.book = []

    def __call__(self, s_row, elig, rng):
        elig = list(elig)
        if not elig:
            return []
        keep = [t for t in self.book if t in elig]
        if not self.book:
            n = min(TOP_N, len(elig))
            self.book = list(rng.choice(elig, size=n, replace=False))
        else:
            k = self.swaps[self.i] if self.i < len(self.swaps) else 0
            k = min(k, len(keep))
            if k > 0:
                drop = set(rng.choice(keep, size=k, replace=False))
                keep = [t for t in keep if t not in drop]
            pool = [t for t in elig if t not in set(keep)]
            need = min(TOP_N, len(elig)) - len(keep)
            add = (list(rng.choice(pool, size=min(need, len(pool)),
                                   replace=False)) if need > 0 else [])
            self.book = keep + add
        self.i += 1
        return list(self.book)


def stats(r, label, ppy=None):
    ppy = ppy or PPY
    if len(r) < 8:
        return None
    sr = r.mean() / r.std(ddof=1) * math.sqrt(ppy)
    cagr = (np.prod(1 + r) ** (ppy / len(r)) - 1) * 100
    dd = 1 - (np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r)))
    print(f"  {label:26s} SR {sr:+.2f} | CAGR {cagr:+6.2f}% | "
          f"maxDD {dd.max() * 100:5.1f}% | n={len(r)}")
    return sr


def main() -> int:
    drag = PPY * 0.6 * COST_BPS / 100
    print(f"WeeklyEquityRotation | {UNIV} ({len(TICKERS)} names, {COST_TIER}) "
          f"| {REBAL} | top {TOP_N} | {START} -> {END} "
          f"| {COST_BPS:.0f} bps round trip")
    print(f"  cost hurdle ~{drag:.2f}%/yr at 0.6 turnover per rebalance -- "
          f"gross alpha clears this before anything else")
    print(f"  !! {UNIV} is TODAY'S membership. Survivorship flatters, one way."
          f" A POSITIVE result proves nothing;")
    print(f"     a NEGATIVE result is real evidence.\n")
    close, open_ = load_prices()
    print(f"loaded {close.shape[1]} tickers, {len(close)} sessions")
    first = {t: close[t].first_valid_index() for t in close.columns}
    late = {t: str(v.date()) for t, v in first.items()
            if v is not None and v > pd.Timestamp(START) + pd.Timedelta(days=40)}
    if late:
        print(f"  {len(late)} of {close.shape[1]} names have no data at "
              f"START (listed later, or the list is post-hoc)")

    score = composite_score(close)
    vol = close.pct_change().rolling(VOL_PERIOD).std()
    dates = rebalance_dates(close.index)
    print(f"  {len(dates)} {REBAL} rebalances\n")

    r_real, npos, turn, swaps = run_strategy(close, open_, score, vol, dates,
                                             real_picker)
    print("FULL PERIOD")
    sr_real = stats(r_real, "rotation (composite ROC)")
    # benchmark: whatever broad-equity proxy this universe actually has
    bench = next((t for t in (os.environ.get("BENCH"), "SPY", "IVV", "VFINX")
                  if t and t in close.columns), None)
    if bench:
        b = close[bench].reindex(
            pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))
        stats(b.pct_change().dropna().values, f"{bench} buy & hold")
    else:
        print("  (no broad-equity benchmark in this universe)")
    eq_r, _, _, _ = run_strategy(close, open_, score, vol, dates,
                                 lambda s, e, g: list(e))
    stats(eq_r, "equal-weight all (no rank)")
    print(f"  avg positions {npos:.1f} | avg turnover/rebal {turn:.2f}")
    print(f"  cash rule would fire on {CASH_FIRES[0]} of {len(dates) - 1} "
          f"rebalances"
          + ("  <- never fires: the top slice is essentially never all"
             "-negative in this universe, so the cash rule is inert here."
             if CASH_FIRES[0] == 0 else ""))

    # ---- holdout: last HOLDOUT_FRAC of time, never used for anything ----
    cut = int(len(r_real) * (1 - HOLDOUT_FRAC))
    print(f"\nSPLIT  (train {cut}mo / holdout {len(r_real) - cut}mo)")
    sr_tr = stats(r_real[:cut], "rotation TRAIN")
    sr_ho = stats(r_real[cut:], "rotation HOLDOUT")

    # ---- the null: random top-5 from the same universe ----
    print(f"\nNULL: {N_NULL} draws, TURNOVER-MATCHED (same names swapped "
          f"per rebalance as the rule)")
    null_sr, null_tr, unm_sr, null_turn = [], [], [], []
    for k in range(N_NULL):
        # matched: replaces exactly as many names as the rule did
        rk, _, tk, _ = run_strategy(close, open_, score, vol, dates,
                                    MatchedPicker(swaps), seed=k)
        null_turn.append(tk)
        if len(rk) >= 8 and rk.std(ddof=1) > 0:
            h, t = rk[cut:], rk[:cut]
            if h.std(ddof=1) > 0:
                null_sr.append(h.mean() / h.std(ddof=1) * math.sqrt(PPY))
            if t.std(ddof=1) > 0:
                null_tr.append(t.mean() / t.std(ddof=1) * math.sqrt(PPY))
        # unmatched: the old, cost-handicapped null, kept for contrast
        ru, _, _, _ = run_strategy(close, open_, score, vol, dates,
                                   random_picker, seed=k)
        if len(ru) > cut + 8 and ru[cut:].std(ddof=1) > 0:
            unm_sr.append(ru[cut:].mean() / ru[cut:].std(ddof=1)
                          * math.sqrt(PPY))
    null_sr = np.array([x for x in null_sr if np.isfinite(x)])
    null_tr = np.array([x for x in null_tr if np.isfinite(x)])
    if len(null_sr) < 30:
        print(f"  null too small ({len(null_sr)}) -- raise N_NULL")
        return 1

    # ---- THE REGIME TEST -------------------------------------------------
    # Both universes showed train SR well BELOW holdout SR. Overfitting runs
    # the other way, so the suspicion is regime: the holdout window is
    # roughly 2020+, which is exactly the tape momentum-with-a-cash-switch is
    # built for. Score the rule against a null computed on the SAME window it
    # is being judged in. If the rule sits at the null median in TRAIN but
    # high in HOLDOUT, the effect is one regime, not an edge.
    if len(null_tr) >= 30 and sr_tr is not None:
        pct_tr = float((null_tr < sr_tr).mean() * 100)
        z_tr = ((sr_tr - null_tr.mean()) / null_tr.std(ddof=1)
                if null_tr.std(ddof=1) > 0 else float("nan"))
        pct_h = float((null_sr < sr_ho).mean() * 100)
        z_h = ((sr_ho - null_sr.mean()) / null_sr.std(ddof=1)
               if null_sr.std(ddof=1) > 0 else float("nan"))
        print(f"\nREGIME TEST (rule vs its own null, window by window)")
        print(f"  TRAIN   rule {sr_tr:+.2f} | null mean {null_tr.mean():+.2f} "
              f"sd {null_tr.std(ddof=1):.2f} | {pct_tr:5.1f}th pct | "
              f"{z_tr:+.2f} sd")
        print(f"  HOLDOUT rule {sr_ho:+.2f} | null mean {null_sr.mean():+.2f} "
              f"sd {null_sr.std(ddof=1):.2f} | {pct_h:5.1f}th pct | "
              f"{z_h:+.2f} sd")
        if pct_tr < 60 and pct_h > 85:
            print("  VERDICT: at the null median in train, high in holdout ->")
            print("  the effect lives in ONE REGIME. This is a bet that the")
            print("  next years resemble the last ones, not a persistent edge.")
        elif pct_tr > 75 and pct_h > 75:
            print("  VERDICT: above its null in BOTH windows -> weak but")
            print("  persistent. Worth pursuing.")
        else:
            print("  VERDICT: mixed. Neither window is decisive on its own.")
    mad = float(np.median(np.abs(null_sr - np.median(null_sr))))
    # var_sr is the SPREAD OF THE NULL, measured. Do NOT clip it up to the
    # pipeline's 0.25 floor -- that floor is calibrated for per-name-fold
    # TRADE Sharpes, which are far noisier than a portfolio's monthly-return
    # Sharpe. Clipping a null with sd 0.20 up to sd 0.50 invents a bar 2.5x
    # too strict.
    var_sr = float(np.clip((1.4826 * mad) ** 2, 1e-6, 25.0))
    # N is HOW MANY THINGS YOU TRIED, not how big the null sample is. The
    # null size is a precision parameter; using it as the trial count
    # punishes you for measuring the null more carefully.
    n_trials = int(os.environ.get("TRIALS", "12"))
    bar = luck_bar(var_sr, n_trials)
    print(f"  rule turnover/rebal {turn:.2f} | matched-null turnover "
          f"{np.mean(null_turn):.2f}  <- these must be close, or the "
          f"comparison is a cost handicap")
    if unm_sr:
        unm = np.array(unm_sr)
        print(f"  UNMATCHED null (rebuilds book each time) holdout SR "
              f"{unm.mean():+.2f} -- the difference between this and the "
              f"matched null is pure trading cost, not selection")
    print(f"  matched null holdout SR: mean {null_sr.mean():+.2f} "
          f"sd {null_sr.std(ddof=1):.2f} | 95th pct {np.percentile(null_sr, 95):+.2f}")
    print(f"  measured var_sr {var_sr:.4f} | luck bar over {n_trials} "
          f"attempted trials = {bar:+.2f}   (set TRIALS= to your real count)")

    # ---------------- ROBUSTNESS: is the edge in the RULE or the TICKERS? --
    def holdout_sr(allowed):
        rk, _, _, _ = run_strategy(close, open_, score, vol, dates,
                                   real_picker, allowed=list(allowed))
        if len(rk) <= cut + 8:
            return None
        h = rk[cut:]
        if h.std(ddof=1) == 0:
            return None
        return float(h.mean() / h.std(ddof=1) * math.sqrt(PPY))

    cols = list(close.columns)
    print(f"\nLEAVE-ONE-OUT (drop each ticker, rerun the real rule)")
    loo = {}
    for t in cols:
        v = holdout_sr([c for c in cols if c != t])
        if v is not None:
            loo[t] = v
    if loo:
        ser = pd.Series(loo).sort_values()
        for t, v in list(ser.items())[:3]:
            print(f"  worst without: drop {t:5s} -> holdout SR {v:+.2f} "
                  f"({v - sr_ho:+.2f} vs full)")
        print(f"  best  without: drop {ser.index[-1]:5s} -> "
              f"{ser.iloc[-1]:+.2f} ({ser.iloc[-1] - sr_ho:+.2f})")
        print(f"  range across drops: {ser.min():+.2f} .. {ser.max():+.2f}")
        if ser.min() < bar:
            print(f"  !! dropping ONE ticker takes it below the bar -- the "
                  f"result leans on {ser.index[0]}")

    k = max(5, int(len(cols) * 0.7))
    n_sub = int(os.environ.get("N_SUB", "200"))
    print(f"\nSUBSAMPLE ({n_sub} random {k}-of-{len(cols)} universes, real rule)")
    rng2 = np.random.default_rng(99)
    subs = []
    for _ in range(n_sub):
        v = holdout_sr(rng2.choice(cols, size=k, replace=False))
        if v is not None:
            subs.append(v)
    if len(subs) >= 20:
        subs = np.array(subs)
        frac = float((subs > bar).mean() * 100)
        print(f"  holdout SR: mean {subs.mean():+.2f} sd {subs.std(ddof=1):.2f} "
              f"| 10th pct {np.percentile(subs, 10):+.2f} "
              f"| {frac:.0f}% clear the bar")
        print("  a rule that only works on the full hindsight list will show a")
        print("  wide spread here and a low percentage clearing.")

    print("\n" + "=" * 66)
    pct = float((null_sr < sr_ho).mean() * 100)
    beat = int((null_sr < sr_ho).sum())
    print(f"rotation HOLDOUT SR {sr_ho:+.2f} | beat {beat}/{len(null_sr)} "
          f"random picks ({pct:.0f}th pct) | bar {bar:+.2f} -> "
          f"{'CLEARS' if sr_ho > bar else 'FAILS'}")
    z_null = ((sr_ho - null_sr.mean()) / null_sr.std(ddof=1)
              if null_sr.std(ddof=1) > 0 else float("nan"))
    print(f"  {z_null:+.2f} sd above the random-pick mean "
          f"(random picking already earns {null_sr.mean():+.2f} here -- that "
          f"part is diversification, not skill)")
    print(f"PSR(holdout, vs 0) = {psr(r_real[cut:]):.3f}")
    if sr_tr is not None and sr_ho is not None and sr_tr > 0 > sr_ho:
        print("WARNING: positive train, negative holdout -- overfit signature.")
    print("=" * 66)
    print("If the rotation sits inside the random-pick distribution, the")
    print("momentum ranking is decoration: you are paid for diversification")
    print("and inverse-vol sizing, which cost nothing to obtain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
