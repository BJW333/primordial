"""
combo_search.py -- every combination of the momentum gap rule and Blake's
ideas (yearly lock, stop at -X%, buy-back signals), searched on 1927-1989,
judged against luck with a block-bootstrap permutation test, then checked
untouched on 1990-2016 and on CHIMERA's own book 2017-2026.

PIECES (each can be off)
  GAP     momentum gap (Huang 2022) in the top 30/20/10% of its trailing 5
          years -> T-bills that month                         4 choices
  STOP    account down 15/20/25% from its high since last buy-in -> T-bills
                                                               4 choices
  BUYBACK (only with a stop) on the momentum book we left:     5 choices
          A  above its 50-day average and +10% off its low
          B  low held 10 days, +5% off it, above its 20-day average
          C  B + new 10-day closing high
          D  B + ADX(14) >= 20 trending up (close-only ADX)
          E  new 10-day closing high and +5% off its low
  LOCK    account up 25/30/35% this calendar year -> the market until Jan 1
                                                               4 choices
  -> 4 x (1 + 3 x 5) x 4 = 256 combinations. 0.2% cost per switch.
  Not testable on 100 years: the volume spike (Ken French portfolios have no
  volume) and the ride rule (failed on QC).

DATA  Ken French top-10% momentum stocks (long only, value weighted) +
      market + T-bills, daily since 1926; gap from French's Prior (2-12)
      breakpoints. CHIMERA = data/momentum_cap_daily.csv (STEADY + SPY).

STEPS (pre-stated; nothing changes after seeing results)
  1 SEARCH 1927-1989: pick the combination with the biggest Sharpe gain over
    always holding the winners.
  2 LUCK  redo the whole 256-combination search on 200 fake histories:
    the 1927-1989 returns reshuffled in ~3-month blocks (keeps their
    volatility and fat tails, destroys any real timing), gap months slid
    >= 1 year. p = share of fake searches whose best gain >= the real best.
  3 LATER 1990-2016 untouched: must raise Sharpe AND shrink the worst drop
    vs always holding the winners.
  4 CHIMERA 2017-03 -> 2026-05 on the real book: must beat LOCK30 (the QC
    winner) on BOTH Sharpe and worst drop.
  PASS only if p < 0.05 AND 3 AND 4.

Run (repo root; uses all CPU cores, ~3-10 minutes):
    python3.10 scripts/combo_search.py --smoke
    python3.10 scripts/combo_search.py | tee logs/combo_search.txt
"""
import argparse
import itertools
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_warning_search as CW   # noqa: E402
import momentum_gap as MG           # noqa: E402

COST, BLOCK = 0.002, 63
GAPS, STOPS, LOCKS = (None, 0.70, 0.80, 0.90), (None, 0.15, 0.20, 0.25), \
    (None, 0.25, 0.30, 0.35)
BUYBACKS = ("A", "B", "C", "D", "E")
P1, P2, P3 = ("1927-01-01", "1989-12-31"), ("1990-01-01", "2016-12-31"), \
    ("2017-03-01", "2026-05-29")


def combos():
    out = []
    for g, s, lk in itertools.product(GAPS, STOPS, LOCKS):
        for b in (BUYBACKS if s else (None,)):
            out.append((g, s, b, lk))
    return out


def name(c):
    g, s, b, lk = c
    parts = []
    if g:
        parts.append(f"gap top {round((1-g)*100)}%")
    if s:
        parts.append(f"stop {round(s*100)}% buyback {b}")
    if lk:
        parts.append(f"lock {round(lk*100)}%")
    return " + ".join(parts) or "always in"


# ---------------------------------------------------------------- inputs
def gap_flags(gap, dates):
    """{level: bool array per day} -- month flagged by the gap rule."""
    per = pd.Index(dates).to_period("M")
    out = {}
    for g in GAPS[1:]:
        rk = gap.rolling(MG.RANK_MONTHS, min_periods=36).rank(pct=True)
        flag = (rk >= g)
        out[g] = per.map(lambda p, f=flag: bool(f.get(p, False))).values.astype(bool)
    return out


def features(W):
    S = np.cumprod(1 + W)
    s = pd.Series(S)
    d = s.diff()
    a = 1 / 14
    tr = d.abs().ewm(alpha=a, adjust=False).mean()
    pdi = 100 * d.clip(lower=0).ewm(alpha=a, adjust=False).mean() / tr
    mdi = 100 * (-d).clip(lower=0).ewm(alpha=a, adjust=False).mean() / tr
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=a, adjust=False).mean()
    return {"S": S, "sma20": s.rolling(20).mean().values,
            "sma50": s.rolling(50).mean().values,
            "hi10": s.shift(1).rolling(10).max().values,
            "adx": adx.values, "pdi": pdi.values, "mdi": mdi.values}


# ------------------------------------------------------------ simulation
def simulate(c, W, M, RF, years, F, G):
    g, stop, bb, lock = c
    gf = G[g] if g else None
    S, s20, s50, hi, adx, pdi, mdi = (F["S"], F["sma20"], F["sma50"], F["hi10"],
                                      F["adx"], F["pdi"], F["mdi"])
    n = len(W)
    out = np.empty(n)
    A, mode, prev, yr, ys, peak = 1.0, 0, 0, years[0], 1.0, 1.0   # 0 MOM 1 SEC 2 CASH 3 GAP
    low, lowi = 0.0, 0
    for i in range(n):
        if years[i] != yr:
            yr, ys = years[i], A
            if mode == 1:
                mode, peak = 0, A
        if gf is not None and (mode == 0 or mode == 3):
            if gf[i]:
                mode = 3
            elif mode == 3:
                mode, peak = 0, A
        r = W[i] if mode == 0 else (M[i] if mode == 1 else RF[i])
        if mode != prev:
            A *= 1 - COST
        A *= 1 + r
        prev = mode
        out[i] = A
        if mode == 0:
            if A > peak:
                peak = A
            if stop and A / peak - 1 <= -stop:
                mode, low, lowi = 2, S[i], i
            elif lock and A / ys - 1 >= lock:
                mode = 1
        elif mode == 2:
            s = S[i]
            if s < low:
                low, lowi = s, i
            held = (i - lowi) >= 10 and s >= low * 1.05 and s > s20[i]
            if bb == "A":
                ok = s > s50[i] and s >= low * 1.10
            elif bb == "B":
                ok = held
            elif bb == "C":
                ok = held and s > hi[i]
            elif bb == "D":
                ok = held and adx[i] >= 20 and pdi[i] > mdi[i]
            else:
                ok = s > hi[i] and s >= low * 1.05
            if ok:
                mode, peak = 0, A
    r = np.empty(n)
    r[0] = out[0] - 1
    r[1:] = out[1:] / out[:-1] - 1
    return r


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def worst(r):
    eq = np.cumprod(1 + r)
    return (eq / np.maximum.accumulate(eq) - 1).min()


def search(W, M, RF, years, G, cs):
    F = features(W)
    base = sharpe(W)
    return np.array([sharpe(simulate(c, W, M, RF, years, F, G)) - base for c in cs])


# ------------------------------------------------------------- the null
def fake_search(args):
    seed, W, M, RF, years, months, gap_by_month = args
    rng = np.random.default_rng(seed)
    n = len(W)
    idx = np.empty(n, int)
    idx[0] = rng.integers(n)
    jump = rng.random(n) < 1 / BLOCK
    starts = rng.integers(n, size=n)
    for i in range(1, n):
        idx[i] = starts[i] if jump[i] else (idx[i - 1] + 1) % n
    k = int(rng.integers(12, len(gap_by_month) - 12))
    gap = pd.Series(np.roll(gap_by_month.values, k), gap_by_month.index)
    G = gap_flags(gap, months)
    return np.max(search(W[idx], M[idx], RF[idx], years, G, combos()))


# ------------------------------------------------------------------ main
def stats_line(label, r):
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    print(f"    {label:<44} {ann*100:>+6.1f}%/yr  Sharpe {sharpe(r):>5.2f}  "
          f"worst {worst(r)*100:>6.1f}%")
    return sharpe(r), worst(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    if a.smoke:
        d, gap = MG.smoke()
    else:
        d, gap = CW.load_french(), MG.load_gap()
    cs = combos()
    print(f"  {len(cs)} combinations | daily {d.index[0].date()} -> {d.index[-1].date()}"
          f"{'  [SYNTHETIC]' if a.smoke else ''}")

    def cut(lo, hi):
        sl = (d.index >= lo) & (d.index <= hi)
        x = d[sl]
        return (x["W"].values, x["M"].values, x["RF"].values,
                x.index.year.values, x.index)

    # 1 SEARCH
    W1, M1, R1, Y1, I1 = cut(*P1)
    G1 = gap_flags(gap, I1)
    gain = search(W1, M1, R1, Y1, G1, cs)
    order = np.argsort(-gain)
    best = cs[int(order[0])]
    print("\n  ===== 1 SEARCH 1927-1989 (Sharpe gain vs always holding winners) =====")
    for j in order[:15]:
        print(f"    {gain[j]:+.3f}  {name(cs[j])}")

    # 2 LUCK
    print(f"\n  ===== 2 LUCK: {a.perms} fake histories, full search each "
          f"({a.jobs} cores) =====")
    jobs = [(s, W1, M1, R1, Y1, I1, gap) for s in range(a.perms)]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        null = np.array(list(ex.map(fake_search, jobs)))
    p = (np.sum(null >= gain[order[0]]) + 1) / (a.perms + 1)
    print(f"    real best {gain[order[0]]:+.3f} | fake searches: median best "
          f"{np.median(null):+.3f}, 95th pct {np.percentile(null, 95):+.3f} | p = {p:.3f}")

    # 3 LATER
    print(f"\n  ===== 3 LATER 1990-2016, fixed: {name(best)} =====")
    W2, M2, R2, Y2, I2 = cut(*P2)
    G2, F2 = gap_flags(gap, I2), features(W2)
    sb, db = stats_line("winners always", W2)
    sw, dw = stats_line(name(best), simulate(best, W2, M2, R2, Y2, F2, G2))
    ok3 = sw > sb and dw > db
    print("    top-15 search picks here (context only):")
    for j in order[:15]:
        r = simulate(cs[j], W2, M2, R2, Y2, F2, G2)
        print(f"      Sharpe {sharpe(r):.2f} vs {sb:.2f}, worst {worst(r)*100:.0f}% "
              f"vs {db*100:.0f}%  {name(cs[j])}")

    # 4 CHIMERA
    print(f"\n  ===== 4 CHIMERA {P3[0]} -> {P3[1]} =====")
    if a.smoke:
        sl = (d.index >= P3[0]) & (d.index <= P3[1])
        bk = pd.DataFrame({"STEADY": d["W"][sl] * 1.5, "SPY": d["M"][sl]})
    else:
        bk = pd.read_csv(CW.BOOK, index_col=0, parse_dates=True)
        bk = bk[(bk.index >= P3[0]) & (bk.index <= P3[1])]
    W3, M3 = bk["STEADY"].values, bk["SPY"].values
    R3 = d["RF"].reindex(bk.index).ffill().fillna(0).values
    Y3, G3, F3 = bk.index.year.values, gap_flags(gap, bk.index), features(W3)
    lock30 = (None, None, None, 0.30)
    r0 = simulate(lock30, W3, M3, R3, Y3, F3, G3)
    r1 = simulate(best, W3, M3, R3, Y3, F3, G3)
    s0, d0 = stats_line("LOCK30", r0)
    s1, d1 = stats_line(name(best), r1)
    ok4 = s1 > s0 and d1 > d0
    for y in sorted(set(Y3)):
        m = Y3 == y
        print(f"      {y}: LOCK30 {(np.prod(1+r0[m])-1)*100:+5.0f}%   best "
              f"{(np.prod(1+r1[m])-1)*100:+5.0f}%")
    print("    Blake's named versions on CHIMERA (context only, NOT a decision):")
    for c in [(0.80, None, None, None), (0.80, None, None, 0.30),
              (None, 0.20, "C", 0.30), (0.80, 0.20, "C", 0.30),
              (0.80, 0.20, "D", 0.30), (0.80, 0.25, "B", 0.30)]:
        stats_line(name(c), simulate(c, W3, M3, R3, Y3, F3, G3))

    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"    luck test p = {p:.3f} -> {'pass' if p < 0.05 else 'FAIL'}")
    print(f"    1990-2016 untouched -> {'pass' if ok3 else 'FAIL'}")
    print(f"    CHIMERA vs LOCK30   -> {'pass' if ok4 else 'FAIL'}")
    print("    " + ("PASS -- port to QC as switches" if p < 0.05 and ok3 and ok4
                    else "FAIL -- keep LOCK30 as is"))
    if a.smoke:
        print("  smoke: synthetic data with a planted gap effect")


if __name__ == "__main__":
    main()
