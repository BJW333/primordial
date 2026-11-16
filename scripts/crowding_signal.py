"""
crowding_signal.py -- TWO research-based crash signals for a winners-only
momentum book, each tested ONCE with settings fixed before running.

WHY THESE TWO: published momentum-crash fixes (Daniel-Moskowitz, Barroso-
Santa-Clara, Dierkes-Krupski) mostly fix the SHORT side (losers rebounding).
CHIMERA only owns winners. Research ties winners-only drops to:
  CROWD  crowding (Lou & Polk "comomentum"): the stocks you own start moving
         together -> too much money in one trade -> it unwinds together.
         (Plain momentum's -56% began Feb 2021 with the top 20 all one bet.)
  WBETA  winners' market beta (Daniel & Moskowitz): after long bull runs
         the winners become high-beta and fall hardest when the market turns.
Neither was among the 9 signs already searched.

SIGNALS (point in time; warning at a close acts from the next day)
  CROWD  each day: average pairwise correlation of the CURRENT 20 holdings'
         market-removed daily returns (stock minus SPY) over the last 63
         days; ranked against its own last 756 days (min 252). Warning when
         in its top 20%. Stock data -> 2017-2026 only (ranks start ~2018).
  WBETA  beta of Ken French's top-10% momentum winners to the market over
         the last 126 days; ranked against its own last 1260 days (min 756).
         Warning when in its top 20%. Ken French data -> 1927-2026.

ACTION ON A WARNING: halve momentum, never exit (full exits died in the
V-rebounds of 2020 and 2025). French winners 100% -> 50% (rest T-bills).
CHIMERA-70: momentum 70% -> 35% of the account (rest T-bills). 0.2% cost
on the share of the account traded.

CHIMERA-70 here = STEADY top-20 book (same code as momentum_cap.py) at 70%
+ 30% T-bills, 30% yearly lock -> SPY until Jan 1 (the QC winner, 30/70).

PRE-STATED PASS RULES (no other levels, windows or actions will be tried)
  WBETA  (a) luck: on 1927-1989, Sharpe gain beats >= 95% of 1,000 copies
             of the warning slid in time (circular shift >= 1 year)
         (b) 1990-2016 untouched: Sharpe up AND worst drop shallower
         (c) CHIMERA-70 2017-03 -> 2026-05: Sharpe up AND worst drop shallower
  CROWD  (a) CHIMERA-70 2017-03 -> 2026-05: Sharpe up AND worst drop shallower
         (b) luck: Sharpe gain beats >= 95% of 1,000 slid copies (>= 63 days)
  Each signal passes or fails on its own. A pass -> port that signal to
  QuantConnect as a switch and confirm there before any paper trading.

Run (repo root; uses data/us_universe/ and data/french/ already on disk):
    python3.10 scripts/crowding_signal.py --smoke
    python3.10 scripts/crowding_signal.py | tee logs/crowding_signal.txt
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blowup_us as U              # noqa: E402
import crash_warning_search as CW  # noqa: E402
import momentum_v2 as M            # noqa: E402

ROOT = os.path.dirname(HERE)
BOOK_CSV = os.path.join(ROOT, "data", "momentum_cap_daily.csv")
COST, LOCK, MOM, CUT, Q = 0.002, 0.30, 0.70, 0.35, 0.80
BETA_DAYS, BETA_RANK, BETA_MIN = 126, 1260, 756
CROWD_DAYS, CROWD_RANK, CROWD_MIN = 63, 756, 252
N_PERM = 1000
P1, P2, P3 = ("1927-01-01", "1989-12-31"), ("1990-01-01", "2016-12-31"), \
    ("2017-03-01", "2026-05-29")


# ------------------------------------------------------------ stats
def sharpe(r):
    r = np.asarray(r)
    return r.mean() / r.std() * np.sqrt(252)


def worst(r):
    eq = np.cumprod(1 + np.asarray(r))
    return (eq / np.maximum.accumulate(eq) - 1).min()


def line(label, r):
    r = np.asarray(r)
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    print(f"    {label:<30} {ann*100:>+6.1f}%/yr  Sharpe {sharpe(r):>5.2f}"
          f"  worst {worst(r)*100:>6.1f}%")
    return sharpe(r), worst(r)


def periods(dates, warn):
    on = pd.Series(warn, dates)
    st = on[on & ~on.shift(1, fill_value=False)].index
    return f"({len(st)}): " + ", ".join(f"{s:%Y-%m-%d}" for s in st[:30]) + \
        (" ..." if len(st) > 30 else "")


# ------------------------------------------------------------ WBETA
def wbeta_warn(d):
    b = d["W"].rolling(BETA_DAYS).cov(d["M"]) / d["M"].rolling(BETA_DAYS).var()
    rk = b.rolling(BETA_RANK, min_periods=BETA_MIN).rank(pct=True)
    return (rk >= Q).fillna(False).values, b


def halve(W, RF, warn):
    """French winners: 100% normally, 50% (rest T-bills) the day after a warning."""
    e = np.where(np.r_[False, warn[:-1]], 0.5, 1.0)
    de = np.abs(np.diff(np.r_[1.0, e]))
    return e * W + (1 - e) * RF - de * COST


# ------------------------------------------------------------ CHIMERA-70
def chimera(book, spy, cash, warn=None):
    """70% book + 30% T-bills; 30% yearly lock -> SPY; warning -> 35% book."""
    A, ys, yr, locked, e_prev = 1.0, 1.0, book.index[0].year, False, MOM
    out = np.empty(len(book))
    b, s, c = book.values, spy.values, cash.values
    for i, d in enumerate(book.index):
        cost = 0.0
        if d.year != yr:
            yr, ys = d.year, A
            if locked:
                locked, cost = False, COST          # sell SPY, back to momentum
                e_prev = None
        if locked:
            r = s[i]
        else:
            e = CUT if (warn is not None and i > 0 and warn[i - 1]) else MOM
            if e_prev is not None:
                cost += COST * abs(e - e_prev)
            e_prev = e
            r = e * b[i] + (1 - e) * c[i]
        A *= (1 + r) * (1 - cost)
        out[i] = A
        if not locked and A / ys - 1 >= LOCK:
            locked, A = True, A * (1 - COST)        # sell momentum, buy SPY
    eq = pd.Series(out, book.index)
    return eq.pct_change().fillna(eq.iloc[0] - 1)


def by_year(base, test, name):
    for y in sorted(set(base.index.year)):
        b = (1 + base[base.index.year == y]).prod() - 1
        t = (1 + test[test.index.year == y]).prod() - 1
        print(f"      {y}: CHIMERA-70 {b*100:+5.0f}%   + {name} {t*100:+5.0f}%")


def chimera_luck(book, spy, cash, warn, min_shift, seed):
    base = sharpe(chimera(book, spy, cash).values)
    real = sharpe(chimera(book, spy, cash, warn).values) - base
    rng = np.random.default_rng(seed)
    n = len(warn)
    null = np.array([sharpe(chimera(book, spy, cash, np.roll(
        warn, int(rng.integers(min_shift, n - min_shift)))).values) - base
        for _ in range(N_PERM)])
    p = (np.sum(null >= real) + 1) / (N_PERM + 1)
    return real, null, p


# ------------------------------------------------------------ STEADY book + CROWD
def steady_book(px, spy, members):
    """Same picks and daily returns as momentum_cap.py STEADY; also holdings."""
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= M.FIRST]
    r = px.pct_change(fill_method=None)
    pxf = px.ffill()
    daily, prev, picks = [], set(), {}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        if t_i < 260:
            continue
        mem = members(t)
        win = px.iloc[t_i - 252:t_i + 1]
        cnt = win.notna().sum()
        elig = [s for s in mem if cnt.get(s, 0) >= 240]
        if len(elig) < 100:
            continue
        W = win[elig]
        mom = W.iloc[-22] / W.iloc[0] - 1
        vol = r[elig].iloc[t_i - 251:t_i + 1].std() * np.sqrt(252)
        sc = (mom / vol.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        pick = list(sc.index[:M.TOP])
        picks[t] = pick
        seg = pxf.loc[t:t2, pick]
        path = (seg / seg.iloc[0]).mean(axis=1)
        dd = path.pct_change().iloc[1:]
        if len(dd):
            dd.iloc[0] -= len(set(pick) ^ prev) / M.TOP * M.COST
        daily.append(dd)
        prev = set(pick)
    return pd.concat(daily), picks


def crowd_series(px, spy, picks):
    """Avg pairwise correlation of current holdings' (stock - SPY) returns, 63d."""
    R = px.pct_change(fill_method=None).sub(spy.pct_change(), axis=0)
    reb = sorted(picks)
    idx = px.index
    out = {}
    for j, t in enumerate(reb):
        end = reb[j + 1] if j + 1 < len(reb) else idx[-1]
        days = idx[(idx >= t) & (idx < end)] if j + 1 < len(reb) else idx[idx >= t]
        cols = picks[t]
        for d in days:
            i = idx.get_loc(d)
            if i < CROWD_DAYS:
                continue
            X = R.iloc[i - CROWD_DAYS + 1:i + 1][cols]
            C = X.corr(min_periods=40).values
            iu = np.triu_indices_from(C, k=1)
            out[d] = np.nanmean(C[iu])
    return pd.Series(out).sort_index()


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global N_PERM
    if a.smoke:
        N_PERM = 50

    # ---------------- data
    d = CW.smoke_data() if a.smoke else CW.load_french()
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
    print(f"  French {d.index[0].date()} -> {d.index[-1].date()} | stocks "
          f"{px.index[0].date()} -> {px.index[-1].date()}, {px.shape[1]} names"
          f"{'  [SYNTHETIC]' if a.smoke else ''}")

    book, picks = steady_book(px, spy, members)
    if not a.smoke and os.path.exists(BOOK_CSV):
        old = pd.read_csv(BOOK_CSV, index_col=0, parse_dates=True)["STEADY"]
        both = pd.concat([book, old], axis=1, join="inner").dropna()
        print(f"  data check: rebuilt STEADY vs momentum_cap_daily.csv "
              f"correlation {both.corr().iloc[0, 1]:.4f} on {len(both)} days")

    if a.smoke:   # smoke: make the windows fit the synthetic stock dates
        P3s = (str(book.index[0].date()), str(book.index[-1].date()))
    else:
        P3s = P3
    bk = book[(book.index >= P3s[0]) & (book.index <= P3s[1])]
    sp = spy.pct_change().reindex(bk.index).fillna(0)
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    base = chimera(bk, sp, cash)

    # ================= WBETA
    print("\n  ================= WBETA: winners' market beta (126d) top 20% =================")
    wb, beta = wbeta_warn(d)
    s1 = (d.index >= P1[0]) & (d.index <= P1[1])
    W1, RF1, w1 = d["W"].values[s1], d["RF"].values[s1], wb[s1]
    real1 = sharpe(halve(W1, RF1, w1)) - sharpe(W1)
    rng = np.random.default_rng(2026)
    n1 = len(W1)
    null1 = np.array([sharpe(halve(W1, RF1, np.roll(w1, int(rng.integers(
        252, n1 - 252))))) - sharpe(W1) for _ in range(N_PERM)])
    p1 = (np.sum(null1 >= real1) + 1) / (N_PERM + 1)
    print(f"  (a) 1927-1989: Sharpe gain {real1:+.3f} | slid copies: median "
          f"{np.median(null1):+.3f}, 95th pct {np.percentile(null1, 95):+.3f} | "
          f"p = {p1:.3f} | warning on {w1.mean()*100:.0f}% of days")
    line("winners always", W1)
    line("winners, halved on warning", halve(W1, RF1, w1))

    s2 = (d.index >= P2[0]) & (d.index <= P2[1])
    W2, RF2, w2 = d["W"].values[s2], d["RF"].values[s2], wb[s2]
    print("  (b) 1990-2016 untouched:")
    sb, db = line("winners always", W2)
    sw, dw = line("winners, halved on warning", halve(W2, RF2, w2))
    print(f"      warning on {w2.mean()*100:.0f}% of days")
    ok_b = sw > sb and dw > db

    wb3 = pd.Series(wb, d.index).reindex(bk.index).ffill().fillna(False) \
        .astype(bool).values
    test_b = chimera(bk, sp, cash, wb3)
    print(f"  (c) CHIMERA-70 {bk.index[0].date()} -> {bk.index[-1].date()}:")
    s0, d0 = line("CHIMERA-70", base.values)
    s3, d3 = line("CHIMERA-70 + WBETA", test_b.values)
    print(f"      warning on {wb3.mean()*100:.0f}% of days")
    by_year(base, test_b, "WBETA")
    print(f"      warning periods {periods(bk.index, wb3)}")
    ok_c = s3 > s0 and d3 > d0
    wbeta_pass = p1 < 0.05 and ok_b and ok_c

    # ================= CROWD
    print("\n  ================= CROWD: holdings' co-movement (63d) top 20% =================")
    crowd = crowd_series(px, spy, picks)
    rk = crowd.rolling(CROWD_RANK, min_periods=CROWD_MIN).rank(pct=True)
    wc = (rk >= Q).reindex(bk.index).fillna(False).astype(bool).values
    first = rk.dropna().index[0] if rk.notna().any() else None
    print(f"  CROWD ranks usable from {first.date() if first is not None else '-'}"
          f" (no warnings before that) | avg correlation: median "
          f"{crowd.median():.2f}, 90th pct {crowd.quantile(0.9):.2f}")
    test_c = chimera(bk, sp, cash, wc)
    print(f"  (a) CHIMERA-70 {bk.index[0].date()} -> {bk.index[-1].date()}:")
    line("CHIMERA-70", base.values)
    s4, d4 = line("CHIMERA-70 + CROWD", test_c.values)
    print(f"      warning on {wc.mean()*100:.0f}% of days")
    by_year(base, test_c, "CROWD")
    print(f"      warning periods {periods(bk.index, wc)}")
    ok_ca = s4 > s0 and d4 > d0
    real_c, null_c, p_c = chimera_luck(bk, sp, cash, wc, CROWD_DAYS, 7)
    print(f"  (b) luck: Sharpe gain {real_c:+.3f} | slid copies: median "
          f"{np.median(null_c):+.3f}, 95th pct {np.percentile(null_c, 95):+.3f}"
          f" | p = {p_c:.3f}")
    crowd_pass = ok_ca and p_c < 0.05

    # context: what each signal said going into the worst stretches
    print("\n  context: signal ranks (0-1) at the start of each month, bad stretches")
    brk = beta.rolling(BETA_RANK, min_periods=BETA_MIN).rank(pct=True)
    for m in ["2018-01", "2018-09", "2020-01", "2020-02", "2021-01", "2021-02",
              "2021-11", "2022-01", "2025-02", "2025-03"]:
        ts = pd.Timestamp(m + "-01")
        cv = rk[rk.index < ts]
        bv = brk[brk.index < ts]
        if len(cv) or len(bv):
            print(f"    {m}: CROWD {cv.iloc[-1] if len(cv) and pd.notna(cv.iloc[-1]) else float('nan'):.2f}"
                  f"   WBETA {bv.iloc[-1] if len(bv) else float('nan'):.2f}")

    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"    WBETA: luck p = {p1:.3f} {'pass' if p1 < 0.05 else 'FAIL'} | "
          f"1990-2016 {'pass' if ok_b else 'FAIL'} | CHIMERA-70 "
          f"{'pass' if ok_c else 'FAIL'}  -> {'PASS' if wbeta_pass else 'FAIL'}")
    print(f"    CROWD: CHIMERA-70 {'pass' if ok_ca else 'FAIL'} | luck p = "
          f"{p_c:.3f} {'pass' if p_c < 0.05 else 'FAIL'}  -> "
          f"{'PASS' if crowd_pass else 'FAIL'}")
    if not (wbeta_pass or crowd_pass):
        print("    FAIL -- keep CHIMERA-70 as is")
    else:
        print("    PASS -> port the passing signal(s) to QuantConnect as a switch")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
