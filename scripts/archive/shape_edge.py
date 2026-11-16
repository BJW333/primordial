"""
shape_edge.py -- do chart SHAPES (alone or combined with an indicator)
predict what commodity ETFs do next?  With a permutation test that charges
for every combination searched.

DATA: data/commod_rth_1h/<SYM>.csv (scripts/fetch_commod_rth.py), 16 commodity
ETFs, regular-hours hourly bars. UUP (dollar) and SPY (stocks) as context.
TRAIN 2016-01 -> 2021-12 (everything is chosen here), TEST 2022-01 -> now.

STAGE 1  shapes alone
  Shapes = k-means of the last W hourly closes (W = 14 ~2 days, 35 ~1 week),
  rescaled so only the shape counts, fitted on TRAIN only (12 shapes per W).
  Every hour, each ETF's last W bars get the nearest shape. Table: what price
  did over the next H bars (7 ~1 day, 14 ~2 days, 35 ~1 week), train vs test.

STAGE 2  shape + one indicator condition (the search)
  conditions: none | RSI(14) <30 / 30-70 / >70 | price above / below its
  140-bar (~20-day) average | hourly volatility above / below its 500-bar
  median | volume spike (z > 1) / no spike | dollar (UUP) above / below its
  140-bar average | stocks (SPY) above / below theirs        = 14 conditions
  rules = 2 W x 12 shapes x 14 conditions x 3 H x long/short = 2,016
  score = average forward return per signal after 14 bps round-trip costs,
  as a t-stat (adjusted for overlapping holds); needs >= 200 train signals.

STAGE 3  permutation test (Masters-style, on TRAIN)
  The whole of stages 1-2 -- refit the shapes, recompute indicators, search
  all 2,016 rules, keep the best -- is repeated on P copies of the train data
  whose hourly moves are shuffled (same shuffle for every ETF, so cross-asset
  links survive; any time pattern is destroyed). p-value = share of shuffled
  copies whose best rule scores >= the real best rule.

TEST  the 10 best train rules traded on 2022 -> now as overlapping slices
  (each signal opens a 1/H position for H bars), 14 bps per round trip.

PASS (fixed before any real-data run): ALL of
  (a) permutation p <= 0.05
  (b) the best train rule's TEST Sharpe >= 0.5 after costs
  (c) at least 7 of the top-10 train rules have positive TEST Sharpe

Run (repo root):
    python3.10 scripts/shape_edge.py --smoke     # planted edge must pass, noise must fail
    python3.10 scripts/shape_edge.py | tee logs/shape_edge.txt
    (--perms 200 for a sharper p-value; default 100, ~1-3 s each)
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data", "commod_rth_1h")
SPLIT = pd.Timestamp("2022-01-01")
WS, HS, K = (14, 35), (7, 14, 35), 12
COST = 0.0014
MIN_N = 200
CONTEXT = ("UUP", "SPY")
CONDS = ["none", "rsi<30", "rsi30-70", "rsi>70", "above avg", "below avg",
         "vol high", "vol low", "volume spike", "no spike",
         "dollar up", "dollar down", "stocks up", "stocks down"]


# ===================================================================== data
def load():
    px, vol = {}, {}
    for p in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        s = os.path.basename(p)[:-4]
        if s == "TLT":
            continue
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        px[s], vol[s] = d["close"], d["volume"]
    if not px:
        raise SystemExit(f"  no CSVs in {DATA}: run scripts/fetch_commod_rth.py first")
    P = pd.DataFrame(px).sort_index().ffill().dropna(how="all")
    V = pd.DataFrame(vol).reindex(P.index).fillna(0.0)
    P = P.dropna()                      # common start
    return P, V.loc[P.index]


def permute(P, V, rng):
    """shuffle hourly moves (same order for every column); first price kept."""
    lr = np.log(P).diff().iloc[1:].values
    idx = rng.permutation(len(lr))
    newp = np.exp(np.log(P.values[0]) + np.cumsum(lr[idx], axis=0))
    Pn = pd.DataFrame(np.vstack([P.values[0], newp]), P.index, P.columns)
    Vn = pd.DataFrame(np.vstack([V.values[0], V.values[1:][idx]]), V.index, V.columns)
    return Pn, Vn


# ================================================================ features
def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def features(P, V, syms):
    """per symbol: condition masks (14 x n), log closes. All causal."""
    ctx = {}
    for c in CONTEXT:
        if c in P:
            ctx[c] = (P[c] > P[c].rolling(140).mean()).values
    out = {}
    for s in syms:
        c, v = P[s], V[s]
        r = rsi(c).values
        avg = c.rolling(140).mean().values
        hv = np.log(c).diff().rolling(14).std()
        hvm = hv.rolling(500, min_periods=200).median()
        vz = ((v - v.rolling(140).mean()) / v.rolling(140).std()).values
        cv = c.values
        n = len(cv)
        m = np.zeros((len(CONDS), n), bool)
        m[0] = True
        m[1], m[2], m[3] = r < 30, (r >= 30) & (r <= 70), r > 70
        m[4], m[5] = cv > avg, cv < avg
        m[6], m[7] = (hv > hvm).values, (hv <= hvm).values & hvm.notna().values
        m[8], m[9] = vz > 1, vz <= 1
        if "UUP" in ctx:
            m[10], m[11] = ctx["UUP"], ~ctx["UUP"]
        if "SPY" in ctx:
            m[12], m[13] = ctx["SPY"], ~ctx["SPY"]
        m[:, :500] = False                                  # warm-up
        out[s] = (m, np.log(cv))
    return out


def norm_windows(lc, w):
    X = swv(lc, w)
    X = X - X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    return np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)


def kmeans(X, k, seed=0, iters=40, n_init=2):
    best = None
    for r in range(n_init):
        rng = np.random.default_rng(seed + r)
        C = X[rng.integers(len(X))][None]
        for _ in range(1, k):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1).min(1)
            C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            new = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                            for j in range(k)])
            if np.allclose(new, C):
                break
            C = new
        inert = ((X - C[lab]) ** 2).sum()
        if best is None or inert < best[0]:
            best = (inert, C)
    return best[1]


def fit_shapes(F, w):
    """centroids from non-overlapping windows of every symbol (TRAIN data)."""
    X = np.vstack([norm_windows(lc, w)[::w] for (_, lc) in F.values()])
    C = kmeans(X, K)
    order = np.argsort(-np.bincount(((X[:, None] - C[None]) ** 2).sum(-1).argmin(1),
                                    minlength=K))
    return C[order]


def labels(lc, C, w):
    X = norm_windows(lc, w)
    lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    out = np.full(len(lc), -1)
    out[w - 1:] = lab
    return out


def describe(c):
    w, rngv = len(c), c.max() - c.min()
    pts, mv = [c[0], c[w // 3], c[2 * w // 3], c[-1]], []
    for a, b in zip(pts, pts[1:]):
        d = b - a
        x = "rise" if d > 0.25 * rngv else ("fall" if d < -0.25 * rngv else "flat")
        if not mv or mv[-1] != x:
            mv.append(x)
    return "sideways" if mv == ["flat"] else ", then ".join(mv)


# ==================================================================== search
def grid(F, shapes):
    """stats for every (W, shape, cond, H): n, mean, std of forward log return."""
    res = {}
    for w in WS:
        C = shapes[w]
        Ls, Ms, FW = [], [], {h: [] for h in HS}
        for (m, lc) in F.values():
            Ls.append(labels(lc, C, w))
            Ms.append(m)
            for h in HS:
                f = np.full(len(lc), np.nan)
                f[:-h] = lc[h:] - lc[:-h]
                FW[h].append(f)
        L, M = np.concatenate(Ls), np.concatenate(Ms, axis=1)
        for h in HS:
            f = np.concatenate(FW[h])
            ok = (L >= 0) & np.isfinite(f)
            for ci in range(len(CONDS)):
                sel = ok & M[ci]
                lab, x = L[sel], f[sel]
                n = np.bincount(lab, minlength=K).astype(float)
                s1 = np.bincount(lab, weights=x, minlength=K)
                s2 = np.bincount(lab, weights=x * x, minlength=K)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mu = s1 / n
                    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
                res[(w, ci, h)] = (n, mu, sd)
    return res


def rank_rules(res):
    rows = []
    for (w, ci, h), (n, mu, sd) in res.items():
        for k in range(K):
            if n[k] < MIN_N or not sd[k] > 0:
                continue
            for d in (1, -1):
                net = d * mu[k] - COST
                t = net / sd[k] * np.sqrt(n[k] / h)          # overlap-adjusted
                rows.append((t, w, k, ci, h, d, n[k], net))
    rows.sort(key=lambda r: -r[0])
    return rows


def best_t(P, V, syms):
    F = features(P, V, syms)
    shapes = {w: fit_shapes(F, w) for w in WS}
    rows = rank_rules(grid(F, shapes))
    return (rows[0][0] if rows else -9.0), F, shapes, rows


# ===================================================================== test
def trade(F, shapes, rule):
    """overlapping-slice backtest of one rule; returns per-bar portfolio P&L."""
    _, w, k, ci, h, d, _, _ = rule
    pnl = None
    for (m, lc) in F.values():
        lab = labels(lc, shapes[w], w)
        trig = ((lab == k) & m[ci]).astype(float)
        pos = pd.Series(trig).rolling(h, min_periods=1).sum().values / h * d
        r = np.r_[0.0, np.diff(lc)]
        entries = pd.Series(trig).values
        exits = np.r_[np.zeros(h), entries[:-h]] if len(entries) > h else np.zeros_like(entries)
        p = np.r_[0.0, pos[:-1]] * r - (entries + exits) * (COST / 2) / h
        pnl = p if pnl is None else pnl + p
    return pnl / len(F)


def sharpe_daily(pnl, index):
    s = pd.Series(pnl, index).groupby(index.normalize()).sum()
    return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0, s


def rule_name(r, shapes):
    _, w, k, ci, h, d, n, net = r
    return (f"{'LONG ' if d > 0 else 'SHORT'} {w}h shape #{k + 1} "
            f"({describe(shapes[w][k])}) + {CONDS[ci]}, hold {h}h")


# ==================================================================== main
def run(P, V, perms, label):
    syms = [c for c in P.columns if c not in CONTEXT]
    tr, te = P.index < SPLIT, P.index >= SPLIT
    print(f"\n  ===== {label}: {len(syms)} ETFs, train {tr.sum():,} hourly bars, "
          f"test {te.sum():,}")
    t0 = time.time()
    real, F, shapes, rows = best_t(P[tr], V[tr], syms)
    res = grid(F, shapes)
    # ---- stage 1: shapes alone, long, train vs test
    Ft = features(P, V, syms)                       # full history for test labels
    Fte = {s: (m[:, te], lc[te]) for s, (m, lc) in Ft.items()}
    rte = grid(Fte, shapes)
    print("  STAGE 1 shapes alone: average move after the shape (bp), train | test")
    for w in WS:
        print(f"   {w}-hour shapes        " + "".join(f"  next {h:>2}h tr|te " for h in HS))
        for k in range(K):
            line = f"   #{k + 1:<2} {describe(shapes[w][k])[:22]:<22}"
            for h in HS:
                a = res[(w, 0, h)][1][k] * 1e4
                b = rte[(w, 0, h)][1][k] * 1e4
                line += f"  {a:+6.1f} | {b:+6.1f}"
            print(line)
    # ---- stage 2
    print(f"  STAGE 2: {len(rows):,} rules with >= {MIN_N} train signals; top 10 (train):")
    for r in rows[:10]:
        print(f"   t {r[0]:+5.2f}  net {r[7] * 1e4:+5.1f}bp/signal  n {int(r[6]):>6}  "
              f"{rule_name(r, shapes)}")
    # ---- stage 3: permutations
    rng = np.random.default_rng(7)
    null = []
    for i in range(perms):
        Pn, Vn = permute(P[tr], V[tr], rng)
        null.append(best_t(Pn, Vn, syms)[0])
        if (i + 1) % 25 == 0:
            print(f"    permutations {i + 1}/{perms} ({time.time() - t0:.0f}s)")
    null = np.array(null)
    p = (np.sum(null >= real) + 1) / (len(null) + 1)
    print(f"  STAGE 3: real best t {real:+.2f} | shuffled-data best t: median "
          f"{np.median(null):+.2f}, 95th pct {np.percentile(null, 95):+.2f} | p = {p:.3f}")
    # ---- test
    print("  TEST 2022 -> now (top-10 train rules, overlapping slices, 14bps RT):")
    idx = P.index[te]
    sh = []
    for r in rows[:10]:
        s, daily = sharpe_daily(trade(Fte, shapes, r), idx)
        sh.append(s)
        print(f"   Sharpe {s:+5.2f}  ann {daily.mean() * 252 * 100:+5.1f}%  {rule_name(r, shapes)}")
    ok = {"a permutation p <= 0.05": p <= 0.05,
          "b best rule test Sharpe >= 0.5": bool(sh) and sh[0] >= 0.5,
          "c >= 7/10 top rules positive in test": sum(x > 0 for x in sh) >= 7}
    for kk, v in ok.items():
        print(f"    {kk:<38} {'yes' if v else 'NO'}")
    print(f"  VERDICT: {'PASS' if all(ok.values()) else 'FAIL'}   ({time.time() - t0:.0f}s)")


def synth(planted, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", "2026-10-30").repeat(7) + \
        pd.to_timedelta(np.tile(np.arange(7) * 60 + 870, len(pd.bdate_range(
            "2016-01-04", "2026-10-30"))), unit="m")
    n, names = len(idx), [f"S{i}" for i in range(8)] + list(CONTEXT)
    sig = 0.004
    P = {}
    for s in names:
        r = rng.normal(0, sig, n)
        if planted and s not in CONTEXT:          # plant: after a 14h V-bottom, drift up
            drift = 0.0
            for t in range(14, n):
                a, b = r[t - 14:t - 7].sum(), r[t - 7:t].sum()
                if a < -2.0 * sig * np.sqrt(7) and b > 1.0 * sig * np.sqrt(7):
                    drift = 14
                if drift > 0:
                    r[t] += 0.6 * sig
                    drift -= 1
        P[s] = 100 * np.exp(np.cumsum(r))
    P = pd.DataFrame(P, idx)
    V = pd.DataFrame(rng.integers(1000, 5000, P.shape).astype(float), idx, P.columns)
    return P, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--perms", type=int, default=100)
    a = ap.parse_args()
    if a.smoke:
        for lab, pl in (("SMOKE NOISE (must FAIL)", False), ("SMOKE PLANTED (must PASS)", True)):
            P, V = synth(pl)
            run(P, V, min(a.perms, 40), lab)
        return
    P, V = load()
    run(P, V, a.perms, "REAL DATA")


if __name__ == "__main__":
    main()
