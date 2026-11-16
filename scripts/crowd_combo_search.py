"""
crowd_combo_search.py -- CROWD and WBETA combined with every warning sign
already tested (6 price + 3 economic), judged against luck by redoing the
WHOLE search on signals slid in time.

ACTION ON A WARNING (both parts): HALVE momentum, never exit.
  French winners 100% -> 50% (rest T-bills); CHIMERA-70 momentum 70% -> 35%.
  0.2% cost on the share of the account traded. CHIMERA-70 = STEADY book
  (crowding_signal.py) at 70% + 30% T-bills, 30% yearly lock -> SPY.

SIGNS (each ranked against its own trailing history, point in time;
3 levels each: top/bottom 20%, 10%, 5%)
  price    MOMVOL BEAR MKT21 MKTVOL HOT RS        (crash_warning_search.py)
  economic CREDIT CURVE JOBS                      (macro_combo_search.py)
  new      WBETA  winners' 126-day beta to the market, vs last 5 years (high)
           CROWD  avg pairwise correlation of the current 20 holdings'
                  market-removed returns, 63 days, vs last 3 years (high)
                  -- stock data, so 2018-2026 only

PART A -- 100 YEARS (WBETA + the 9 old signs; CROWD can't go back that far)
  every 1-, 2-, 3-sign combination at every level (3,675 rules)
  1 SEARCH 1927-1989: biggest Sharpe gain over always holding the winners
           (warning on 1%-50% of days)
  2 LUCK   whole search redone 200x on signs slid >= 1 year: p < 0.05
  3 LATER  1990-2016 untouched: Sharpe up AND worst drop shallower
  4 CHIMERA-70 2017-03 -> 2026-05: Sharpe up AND worst drop shallower

PART B -- CROWD COMBOS (2017-03 -> 2026-05 only)
  CROWD alone, and CROWD AND any ONE other sign (all 10), every level
  (93 rules). Searched directly on CHIMERA-70 -- there is no untouched
  period, so the bar is higher:
  1 best rule beats CHIMERA-70 on Sharpe AND worst drop
  2 LUCK  whole 93-rule search redone 200x with all signs slid >= 63 days
          against the returns: p < 0.05
  3 HALVES best rule's Sharpe beats CHIMERA-70 in BOTH 2017-03 -> 2021-05
          and 2021-06 -> 2026-05
  PASS only if 1, 2 and 3 all pass.

Each part passes or fails on its own. Nothing is re-run with other levels,
windows or actions after seeing results.

Run (repo root; needs crowding_signal.py, crash_warning_search.py,
macro_combo_search.py in scripts/ and data/french, data/fred on disk):
    python3.10 scripts/crowd_combo_search.py --smoke
    python3.10 scripts/crowd_combo_search.py | tee logs/crowd_combo_search.txt
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
import blowup_us as U              # noqa: E402
import crash_warning_search as CW  # noqa: E402
import crowding_signal as CS       # noqa: E402
import macro_combo_search as MC    # noqa: E402
import momentum_v2 as M            # noqa: E402

COST, LOCK, MOM, CUT = 0.002, 0.30, 0.70, 0.35
N_PERM, CHUNK = 200, 256
P1, P2, P3 = ("1927-01-01", "1989-12-31"), ("1990-01-01", "2016-12-31"), \
    ("2017-03-01", "2026-05-29")
SPLIT = pd.Timestamp("2021-06-01")


# ------------------------------------------------------------ returns
def half_returns(W, RF, warn):
    """warn (days x rules) at close t -> winners at 50% on day t+1."""
    on = np.vstack([np.zeros((1, warn.shape[1]), bool), warn[:-1]])
    e = 1.0 - 0.5 * on
    de = np.abs(np.diff(np.vstack([np.ones((1, warn.shape[1])), e]), axis=0))
    return e * W[:, None] + (1 - e) * RF[:, None] - de * COST, on


def score_half(W, RF, warn):
    r, on = half_returns(W, RF, warn)
    gain = CW.sharpe(r) - CW.sharpe(W[:, None])[0]
    frac = on.mean(axis=0)
    gain[(frac < 0.01) | (frac > 0.50)] = -np.inf
    return gain


def score_all(W, RF, C, rules):
    out = np.empty(len(rules))
    for i in range(0, len(rules), CHUNK):
        out[i:i + CHUNK] = score_half(W, RF, CW.warn_matrix(C, rules[i:i + CHUNK]))
    return out


def slid_best(args):
    seed, W, RF, C, rules = args
    k = int(np.random.default_rng(seed).integers(CW.MIN_SHIFT, len(W) - CW.MIN_SHIFT))
    return float(np.max(score_all(W, RF, np.roll(C, k, axis=0), rules)))


def chimera_many(book, spy, cash, warn):
    """CHIMERA-70 for many warning columns at once (days x rules) -> returns."""
    b, s, c = book.values, spy.values, cash.values
    years = book.index.year
    n, k = warn.shape
    A, ys = np.ones(k), np.ones(k)
    locked, fresh = np.zeros(k, bool), np.zeros(k, bool)
    e_prev = np.full(k, MOM)
    out = np.empty((n, k))
    yr = years[0]
    for i in range(n):
        cost = np.zeros(k)
        if years[i] != yr:
            yr, ys = years[i], A.copy()
            fresh = locked.copy()
            cost[locked] = COST                     # sell SPY, back to momentum
            locked[:] = False
        e = np.where(warn[i - 1], CUT, MOM) if i > 0 else np.full(k, MOM)
        ch = ~locked & ~fresh
        cost[ch] += COST * np.abs(e[ch] - e_prev[ch])
        e_prev = np.where(~locked, e, e_prev)
        fresh[:] = False
        r = np.where(locked, s[i], e * b[i] + (1 - e) * c[i])
        A = A * (1 + r) * (1 - cost)
        out[i] = A
        new = ~locked & (A / ys - 1 >= LOCK)
        A = np.where(new, A * (1 - COST), A)        # sell momentum, buy SPY
        locked |= new
    ret = np.vstack([out[:1] - 1, out[1:] / out[:-1] - 1])
    return ret


def sh(r):
    return r.mean(axis=0) / r.std(axis=0) * np.sqrt(252)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--perms", type=int, default=N_PERM)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    perms = 20 if a.smoke else a.perms

    # ---------------- French + macro + WBETA signs
    d = CW.smoke_data() if a.smoke else CW.load_french()
    Cp, names_p, feat_p = CW.conditions(CW.features(d))
    Cm, names_m, feat_m = MC.macro_conditions(MC.macro_monthly(a.smoke), d.index,
                                              max(feat_p) + 1)
    _, beta = CS.wbeta_warn(d)
    brk = beta.rolling(CS.BETA_RANK, min_periods=CS.BETA_MIN).rank(pct=True)
    fb = max(feat_m) + 1
    Cb = np.column_stack([(brk >= q).fillna(False).values for q in CW.LEVELS])
    names_b = [f"WBETA top {round((1 - q) * 100)}%" for q in CW.LEVELS]
    C = np.hstack([Cp, Cm, Cb])
    names = names_p + names_m + names_b
    feat_of = feat_p + feat_m + [fb] * 3
    rules = CW.all_rules(feat_of)
    rnA = lambda j: " AND ".join(names[i] for i in rules[j])  # noqa: E731

    # ---------------- stock book + CROWD
    if a.smoke:
        px, vol, spy = M.smoke()
        feed = "sip"
    else:
        px, vol, spy, feed = U.load()
    U.MIN_PX = M.MIN_PX
    min_dv = M.MIN_DV if feed == "sip" else M.MIN_DV * 0.03
    sets, ever = U.liquid_members(px, vol, min_dv)
    px = px[ever]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)
    members = lambda t: sets[max(x for x in days if x <= t)]  # noqa: E731
    book, picks = CS.steady_book(px, spy, members)
    p3 = (str(book.index[0].date()), str(book.index[-1].date())) if a.smoke else P3
    bk = book[(book.index >= p3[0]) & (book.index <= p3[1])]
    sp = spy.pct_change().reindex(bk.index).fillna(0)
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    base = chimera_many(bk, sp, cash, np.zeros((len(bk), 1), bool))[:, 0]
    s0, d0 = CS.sharpe(base), CS.worst(base)
    print(f"  {len(rules)} 100-year rules (10 signs x 3 levels) | CHIMERA-70 "
          f"{bk.index[0].date()} -> {bk.index[-1].date()}"
          f"{'  [SYNTHETIC]' if a.smoke else ''}")

    # ================= PART A
    print("\n  ================= PART A: 100 years, WBETA + 9 old signs =================")
    s1 = (d.index >= P1[0]) & (d.index <= P1[1])
    W1, RF1, C1 = d["W"].values[s1], d["RF"].values[s1], C[s1]
    g = score_all(W1, RF1, C1, rules)
    order = np.argsort(-g)
    best = int(order[0])
    print("  1 SEARCH 1927-1989 (Sharpe gain, winners halved on warning):")
    for j in order[:12]:
        print(f"    {g[j]:+.3f}  {rnA(j)}")
    wb = [j for j in order if any(feat_of[i] == fb for i in rules[j])]
    if wb:
        print(f"    best rule using WBETA (context): {g[wb[0]]:+.3f}  {rnA(wb[0])}")
    jobs = [(sd, W1, RF1, C1, rules) for sd in range(perms)]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        null = np.array(list(ex.map(slid_best, jobs)))
    pA = (np.sum(null >= g[best]) + 1) / (perms + 1)
    print(f"  2 LUCK: real best {g[best]:+.3f} | slid: median {np.median(null):+.3f}, "
          f"95th pct {np.percentile(null, 95):+.3f} | p = {pA:.3f}")
    s2 = (d.index >= P2[0]) & (d.index <= P2[1])
    W2, RF2 = d["W"].values[s2], d["RF"].values[s2]
    r2, on2 = half_returns(W2, RF2, CW.warn_matrix(C[s2], [rules[best]]))
    print(f"  3 LATER 1990-2016, fixed: {rnA(best)}")
    sb, db = CS.line("winners always", W2)
    sw, dw = CS.line("winners, halved on warning", r2[:, 0])
    print(f"      warning on {on2.mean()*100:.0f}% of days")
    okA3 = sw > sb and dw > db
    wA = pd.DataFrame(CW.warn_matrix(C, [rules[best]]), index=d.index) \
        .reindex(bk.index).ffill().fillna(False).values.astype(bool)
    tA = chimera_many(bk, sp, cash, wA)[:, 0]
    print("  4 CHIMERA-70:")
    CS.line("CHIMERA-70", base)
    sA, dA = CS.line("CHIMERA-70 + warning", tA)
    print(f"      warning on {wA.mean()*100:.0f}% of days")
    CS.by_year(pd.Series(base, bk.index), pd.Series(tA, bk.index), "warning")
    okA4 = sA > s0 and dA > d0
    passA = pA < 0.05 and okA3 and okA4

    # ================= PART B
    print("\n  ================= PART B: CROWD alone and CROWD + one other sign =================")
    crowd = CS.crowd_series(px, spy, picks)
    crk = crowd.rolling(CS.CROWD_RANK, min_periods=CS.CROWD_MIN).rank(pct=True) \
        .reindex(bk.index)
    Cc = np.column_stack([(crk >= q).fillna(False).values for q in CW.LEVELS])
    names_c = [f"CROWD top {round((1 - q) * 100)}%" for q in CW.LEVELS]
    Co = pd.DataFrame(C, index=d.index).reindex(bk.index).ffill().fillna(False) \
        .values.astype(bool)
    CB = np.hstack([Cc, Co])
    namesB = names_c + names
    rnB = lambda j: " AND ".join(namesB[i] for i in rulesB[j])  # noqa: E731
    nc = len(names_c)
    rulesB = [(i,) for i in range(nc)] + \
        [(i, nc + o) for i in range(nc) for o in range(Co.shape[1])]
    n = len(bk)

    def search(Cmat):
        warn = CW.warn_matrix(Cmat, rulesB)
        r = chimera_many(bk, sp, cash, warn)
        gain = sh(r) - CS.sharpe(base)
        frac = warn.mean(axis=0)
        gain[(frac < 0.01) | (frac > 0.50)] = -np.inf
        return gain, r

    gB, rB = search(CB)
    ordB = np.argsort(-gB)
    bB = int(ordB[0])
    print("  search on CHIMERA-70 (Sharpe gain):")
    for j in ordB[:12]:
        print(f"    {gB[j]:+.3f}  worst {CS.worst(rB[:, j])*100:6.1f}%  {rnB(j)}")
    print(f"    CROWD alone (context): " + " | ".join(
        f"{names_c[i]} {gB[i]:+.3f}" for i in range(nc)))
    print(f"  1 best rule fixed: {rnB(bB)}")
    CS.line("CHIMERA-70", base)
    sB, dB = CS.line("CHIMERA-70 + warning", rB[:, bB])
    wB = CW.warn_matrix(CB, [rulesB[bB]])[:, 0]
    print(f"      warning on {wB.mean()*100:.0f}% of days")
    CS.by_year(pd.Series(base, bk.index), pd.Series(rB[:, bB], bk.index), "warning")
    print(f"      warning periods {CS.periods(bk.index, wB)}")
    okB1 = sB > s0 and dB > d0
    rng = np.random.default_rng(11)
    nullB = []
    for _ in range(perms):
        k = int(rng.integers(63, n - 63))
        nullB.append(np.max(search(np.roll(CB, k, axis=0))[0]))
    nullB = np.array(nullB)
    pB = (np.sum(nullB >= gB[bB]) + 1) / (perms + 1)
    print(f"  2 LUCK: real best {gB[bB]:+.3f} | slid: median {np.median(nullB):+.3f}, "
          f"95th pct {np.percentile(nullB, 95):+.3f} | p = {pB:.3f}")
    h1 = bk.index < SPLIT
    hs = [(CS.sharpe(base[m]), CS.sharpe(rB[m, bB])) for m in (h1, ~h1)]
    print(f"  3 HALVES: 2017-03 -> 2021-05 Sharpe {hs[0][1]:.2f} vs {hs[0][0]:.2f} | "
          f"2021-06 -> 2026-05 Sharpe {hs[1][1]:.2f} vs {hs[1][0]:.2f}")
    okB3 = all(t > b for b, t in hs)
    passB = okB1 and pB < 0.05 and okB3

    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"    PART A ({rnA(best)}): luck p = {pA:.3f} "
          f"{'pass' if pA < 0.05 else 'FAIL'} | 1990-2016 {'pass' if okA3 else 'FAIL'}"
          f" | CHIMERA-70 {'pass' if okA4 else 'FAIL'} -> {'PASS' if passA else 'FAIL'}")
    print(f"    PART B ({rnB(bB)}): CHIMERA-70 {'pass' if okB1 else 'FAIL'}"
          f" | luck p = {pB:.3f} {'pass' if pB < 0.05 else 'FAIL'} | halves "
          f"{'pass' if okB3 else 'FAIL'} -> {'PASS' if passB else 'FAIL'}")
    print("    " + ("PASS -> port the passing rule to QuantConnect as a switch"
                    if passA or passB else "FAIL -- keep CHIMERA-70 as is"))
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
