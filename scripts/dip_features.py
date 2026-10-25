"""
dip_features.py -- can ANY of ~17 price/volume measurements tell a dip
that rebounds from a dip that keeps falling? Step 1 before any combo search.

Event pool: every close-to-close drop of 2-25% on S&P 500 (PIT-gated),
S&P 400 and S&P 600 (current lists), 2010-03 onward.
Target: H-day return (close of drop day -> close H days later) MINUS the
same-day average H-day return of every index member. Survivor lift common
to all names cancels; the uneven part (big droppers delist most) does not.

Features (all known at the drop day's close):
  drop_pct      size of the drop in %
  drop_atr      drop in units of the prior 14-day average range
  gap_share     share of the drop that happened overnight (0=intraday, 1=gap)
  close_loc     where it closed in the day's range (0=at low, 1=at high)
  vol_ratio     volume vs prior 20-day average
  ret5/20/60    return over the 5/20/60 days BEFORE the drop day
  dist_52wh     distance below the 52-week high
  vs_200dma     distance from the 200-day average
  down_streak   consecutive down closes before the drop day
  atr_rank      how volatile the stock is vs its own past year (0-1)
  log_dollarvol size / liquidity proxy (20-day avg dollar volume)
  spy_ret       SPY's move the same day
  spy_vs_200    SPY's distance from its 200-day average
  breadth       share of index members down >= 3% the same day
  past_dips     average target of THIS stock's earlier dips that had
                fully finished before today (its rebound "habit")

TESTS (pre-stated; split 2018-01-01, train events within 30 trading days
of the split dropped so no target overlaps it):
  A. Each feature alone: train-set terciles; top-vs-bottom third spread on
     train and on test (test uses TRAIN cutpoints). A feature REPLICATES on
     an index if the test spread has the train sign with |t| > 2.
     ~17 features x 2 holds x 3 indexes ~ 100 looks: expect ~5 lucky ones.
  B. All features at once: gradient-boosted model per index, fit on train,
     scored on test by daily rank correlation (prediction vs outcome among
     that day's dips; t across days) and by the top-fifth's net return vs
     the index average.
  VERDICT: SIGNAL if, at H=10, the model's test rank-correlation t > 3 on
  ALL THREE indexes AND its top fifth beats the index average after cost
  with t > 2 on all three. One model per index = one look each, so luck
  can't fake this. SIGNAL -> step 2 (Primordial search over the features
  that replicated). Otherwise no combo of these features will work.

Controls (verified 2026-10-25): planted "dips that close near the day's
high rebound" -> model t +14.9, close_loc ranked #1; same data with the
plant removed -> model t +0.9, NO SIGNAL.

Run (repo root; several minutes, the model step is the slow part):
    python3.10 scripts/dip_features.py
    python3.10 scripts/dip_features.py --smoke
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial.data import fetch  # noqa: E402

M = os.path.join(ROOT, "manifests")
END = "2026-06-01"
GROUNDS = {"sp500": ("2009-01-01", 0.0010),
           "sp400": ("2010-01-01", 0.0020),
           "sp600": ("2010-01-01", 0.0030)}
HOLDS = [10, 20]
PRIMARY_H = 10
SIG_START, SPLIT = pd.Timestamp("2010-03-01"), pd.Timestamp("2018-01-01")
EMBARGO = 30
FEATS = ["drop_pct", "drop_atr", "gap_share", "close_loc", "vol_ratio",
         "ret5", "ret20", "ret60", "dist_52wh", "vs_200dma", "down_streak",
         "atr_rank", "log_dollarvol", "spy_ret", "spy_vs_200", "breadth",
         "past_dips"]
DAY_LEVEL = {"spy_ret", "spy_vs_200", "breadth"}   # tested across days


# ---------------------------------------------------------------- data
def membership(path):
    rows = [r for r in csv.reader(open(path))][1:]
    return (pd.DatetimeIndex([r[0] for r in rows]),
            [frozenset(t.strip() for t in r[1].split(",")) for r in rows])


def member_mask(sym, idx, mdates, sets_):
    s = sym.replace("-", ".")
    rows = np.array([s in x for x in sets_])
    k = mdates.searchsorted(idx, side="right") - 1
    return np.where(k >= 0, rows[np.clip(k, 0, None)], False)


def read_symbols(path):
    return [ln.strip() for ln in open(path)
            if ln.strip() and not ln.startswith("#")]


def down_streak(c):
    d = np.r_[False, c[1:] < c[:-1]]
    out = np.zeros(len(c))
    run = 0
    for i in range(len(c)):
        out[i] = run              # streak BEFORE today
        run = run + 1 if d[i] else 0
    return out


def per_symbol(df, spy):
    """All features + forward returns for every bar of one symbol."""
    df = df[~df.index.duplicated()].copy()
    o, h, lo, c, v = (df[k].astype(float) for k in
                      ("open", "high", "low", "close", "volume"))
    pc = c.shift(1)
    r1 = c / pc - 1
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(1)
    atr_pct = tr.rolling(14).mean() / c
    rng = (h - lo).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["r1"] = r1
    f["drop_pct"] = -r1
    f["drop_atr"] = -r1 / atr_pct.shift(1)
    f["gap_share"] = ((o - pc) / (c - pc)).clip(-1, 2)
    f["close_loc"] = (c - lo) / rng
    f["vol_ratio"] = v / v.rolling(20).mean().shift(1).replace(0, np.nan)
    for n in (5, 20, 60):
        f[f"ret{n}"] = pc / c.shift(n + 1) - 1
    f["dist_52wh"] = c / h.rolling(252).max() - 1
    f["vs_200dma"] = c / c.rolling(200).mean() - 1
    f["down_streak"] = down_streak(c.values)
    f["atr_rank"] = atr_pct.rolling(252).rank(pct=True)
    f["log_dollarvol"] = np.log1p((c * v).rolling(20).mean())
    sp = spy.reindex(df.index, method="ffill")
    f["spy_ret"] = sp["close"].pct_change()
    f["spy_vs_200"] = sp["close"] / sp["close"].rolling(200).mean() - 1
    for H in HOLDS:
        f[f"fwd{H}"] = c.shift(-H) / c - 1
    return f


def build(data, spy, mem):
    slim, dips = [], []          # slim = every member-day (for baselines)
    for sym, df in data.items():
        if len(df) < 300:
            continue
        f = per_symbol(df, spy)
        f = f[mem(sym, f.index)] if mem else f
        f.index.name = "date"
        f = f.reset_index()
        f["sym"] = sym
        slim.append(f[["date", "r1"] + [f"fwd{H}" for H in HOLDS]])
        dips.append(f[(f["r1"] <= -0.02) & (f["r1"] > -0.25)
                      & (f["date"] >= SIG_START)])
    s = pd.concat(slim, ignore_index=True)
    g = s.groupby("date")
    day = pd.DataFrame({"n": g["r1"].count(),
                        "breadth": g["r1"].apply(lambda x: (x <= -0.03)
                                                 .mean())})
    for H in HOLDS:
        x = s[f"fwd{H}"].where(s[f"fwd{H}"].abs() <= 2.0)
        day[f"base{H}"] = x.groupby(s["date"]).mean()
    del s, slim
    ev = pd.concat(dips, ignore_index=True).merge(
        day, left_on="date", right_index=True, how="left")
    ev = ev[ev["n"] >= 20].copy()
    for H in HOLDS:
        ev[f"adj{H}"] = ev[f"fwd{H}"] - ev[f"base{H}"]
    for H in HOLDS:
        ev = ev[ev[f"fwd{H}"].abs() <= 2.0]
    ev = ev.dropna(subset=[f"adj{H}" for H in HOLDS])
    ev["past_dips"] = past_dips(ev)
    return ev.reset_index(drop=True)


def past_dips(ev):
    """Expanding mean of a stock's earlier dips' adj10, counting a dip only
    once its 10-day window has closed (no peeking)."""
    out = pd.Series(np.nan, index=ev.index)
    for _, e in ev.sort_values("date").groupby("sym"):
        d = e["date"].values
        y = e[f"adj{PRIMARY_H}"].values
        done = d + np.timedelta64(PRIMARY_H * 7 // 5 + 2, "D")  # ~H bars
        order = np.argsort(done)
        cs = np.r_[0, np.cumsum(y[order])]
        k = np.searchsorted(done[order], d, side="left")
        vals = np.where(k >= 3, cs[k] / np.maximum(k, 1), np.nan)
        out.loc[e.index] = vals
    return out


# ---------------------------------------------------------------- tests
def clustered(y, dates):
    d = y.groupby(dates).mean()
    if len(d) < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    return d.mean() * 1e4, d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))


def daily_ic(pred, y, dates, min_n=5):
    df = pd.DataFrame({"p": pred, "y": y, "d": dates})
    df = df[df.groupby("d")["y"].transform("count") >= min_n]
    df["pr"] = df.groupby("d")["p"].rank()
    df["yr"] = df.groupby("d")["y"].rank()
    ic = df.groupby("d")[["pr", "yr"]].corr().unstack().iloc[:, 1].dropna()
    if len(ic) < 3:
        return np.nan, np.nan, 0
    return ic.mean(), ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))), len(ic)


def split(ev):
    cut = ev.loc[ev["date"] < SPLIT, "date"].drop_duplicates().sort_values()
    emb = cut.iloc[-EMBARGO] if len(cut) > EMBARGO else SPLIT
    return ev[ev["date"] < emb], ev[ev["date"] >= SPLIT]


def feature_table(ev, H):
    tr, te = split(ev)
    y = f"adj{H}"
    rows = []
    for f in FEATS:
        a, b = tr[[f, y, "date"]].dropna(), te[[f, y, "date"]].dropna()
        if len(a) < 300 or a[f].nunique() < 3:
            continue
        q1, q2 = a[f].quantile([1 / 3, 2 / 3])
        if q1 == q2:
            continue

        def spread(x):
            top = x[x[f] >= q2]
            bot = x[x[f] <= q1]
            if f in DAY_LEVEL:   # same value for every dip that day:
                a_ = top.groupby("date")[y].mean()   # compare across days
                b_ = bot.groupby("date")[y].mean()
                if len(a_) < 3 or len(b_) < 3:
                    return np.nan, np.nan
                se = np.hypot(a_.std(ddof=1) / np.sqrt(len(a_)),
                              b_.std(ddof=1) / np.sqrt(len(b_)))
                return (a_.mean() - b_.mean()) * 1e4, \
                    (a_.mean() - b_.mean()) / se
            dt = top.groupby("date")[y].mean() - \
                bot.groupby("date")[y].mean()
            dt = dt.dropna()
            if len(dt) < 3:
                return np.nan, np.nan
            return dt.mean() * 1e4, dt.mean() / (dt.std(ddof=1) /
                                                  np.sqrt(len(dt)))
        s_tr, s_te = spread(a), spread(b)
        rep = (np.isfinite(s_te[1]) and np.sign(s_te[0]) == np.sign(s_tr[0])
               and abs(s_te[1]) > 2)
        rows.append((f, s_tr, s_te, rep))
    return rows


def model_test(ev, H, cost):
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr, te = split(ev)
    y = f"adj{H}"
    ytr = tr[y].clip(-0.5, 0.5)
    mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                        max_leaf_nodes=31,
                                        min_samples_leaf=200,
                                        l2_regularization=1.0,
                                        random_state=0)
    mdl.fit(tr[FEATS], ytr)
    p = pd.Series(mdl.predict(te[FEATS]), index=te.index)
    ic, ic_t, ndays = daily_ic(p, te[y], te["date"])
    q = p.groupby(te["date"]).rank(pct=True)
    top = te[q >= 0.8]
    bot = te[q <= 0.2]
    top_net = clustered(top[y] - cost, top["date"])
    dt = (top.groupby("date")[y].mean()
          - bot.groupby("date")[y].mean()).dropna()
    ls = ((dt.mean() * 1e4, dt.mean() / (dt.std(ddof=1) / np.sqrt(len(dt))))
          if len(dt) > 2 else (np.nan, np.nan))
    imp = importance(mdl, te, y)
    return dict(ic=ic, ic_t=ic_t, days=ndays, top_net=top_net, ls=ls,
                n_tr=len(tr), n_te=len(te), imp=imp)


def importance(mdl, te, y, n=40000):
    """How much test rank-correlation drops when each feature is shuffled."""
    s = te.sample(min(n, len(te)), random_state=0)
    base = daily_ic(pd.Series(mdl.predict(s[FEATS]), index=s.index),
                    s[y], s["date"])[0]
    rng = np.random.default_rng(0)
    out = {}
    for f in FEATS:
        X = s[FEATS].copy()
        X[f] = rng.permutation(X[f].values)
        out[f] = base - daily_ic(pd.Series(mdl.predict(X), index=s.index),
                                 s[y], s["date"])[0]
    return sorted(out.items(), key=lambda kv: -kv[1])[:6]


# ---------------------------------------------------------------- driver
def load(name, a):
    start, cost = GROUNDS[name]
    if a.smoke:
        return a.smoke_data[0], a.smoke_data[1], None, cost
    mem = None
    if name == "sp500":
        mdates, sets_ = membership(os.path.join(M, "sp500_pit_membership.csv"))
        syms = sorted({s.replace(".", "-") for x in sets_ for s in x})
        mem = lambda s, idx: member_mask(s, idx, mdates, sets_)  # noqa
    else:
        syms = read_symbols(os.path.join(M, f"{name}_symbols.txt"))
    data = fetch(a.source, syms, "1d", start, END)
    spy = fetch(a.source, ["SPY"], "1d", start, END)["SPY"]
    return data, spy, mem, cost


def run_ground(name, a):
    data, spy, mem, cost = load(name, a)
    ev = build(data, spy, mem)
    tr, te = split(ev)
    print(f"\n  ===== {name.upper()} -- {len(ev)} dips "
          f"({len(tr)} train / {len(te)} test), {cost * 1e4:.0f} bps RT =====")
    reps = {}
    for H in HOLDS:
        print(f"\n  A. each feature alone, H={H}: top third minus bottom "
              f"third (bps, vs index avg)")
        print(f"  {'feature':<14} {'train':>8} {'t':>6} | {'test':>8} "
              f"{'t':>6} | replicates")
        rows = feature_table(ev, H)
        for f, s_tr, s_te, rep in rows:
            print(f"  {f:<14} {s_tr[0]:>8.1f} {s_tr[1]:>6.2f} | "
                  f"{s_te[0]:>8.1f} {s_te[1]:>6.2f} | "
                  f"{'YES' if rep else ''}")
        reps[H] = [f for f, *_, r in rows if r]
    res = {}
    for H in HOLDS:
        m = model_test(ev, H, cost)
        res[H] = m
        print(f"\n  B. all features, one model, H={H} "
              f"(fit 2010-17, scored 2018+):")
        print(f"     daily rank-correlation {m['ic']:+.4f}  t {m['ic_t']:+.2f}"
              f"  over {m['days']} days")
        print(f"     top fifth vs index avg, after cost: "
              f"{m['top_net'][0]:+.1f} bps  t {m['top_net'][1]:+.2f}")
        print(f"     top fifth minus bottom fifth: {m['ls'][0]:+.1f} bps  "
              f"t {m['ls'][1]:+.2f}")
        print("     features the model leans on most: " + ", ".join(
            f"{f} ({d:+.4f})" for f, d in m["imp"]))
    return res, reps


def smoke_data():
    """Planted: dips that CLOSE NEAR THE DAY'S HIGH rebound, others don't."""
    idx = pd.bdate_range("2010-01-01", "2026-05-29")
    rng = np.random.default_rng(9)
    data = {}
    for k in range(60):
        r = rng.normal(0.0003, 0.02, len(idx))
        hi_close = rng.random(len(idx)) < 0.5
        for i in np.where(r <= -0.02)[0]:
            if i < len(idx) - 12 and hi_close[i]:
                r[i + 1:i + 11] += 0.002
        c = 100 * np.exp(np.cumsum(r))
        pc = np.r_[100, c[:-1]]
        o = pc * (1 + rng.normal(0, 0.003, len(idx)))
        span = np.abs(c - pc) + c * 0.01
        lo = np.where(hi_close, c - 0.9 * span, c - 0.1 * span)
        hi = np.maximum(np.maximum(o, c), lo + span)
        lo = np.minimum(lo, np.minimum(o, c))
        data[f"S{k}"] = pd.DataFrame({"open": o, "high": hi, "low": lo,
                                      "close": c,
                                      "volume": rng.uniform(1e5, 1e6,
                                                            len(idx))}, idx)
    spy_r = rng.normal(0.0003, 0.01, len(idx))
    sc = 100 * np.exp(np.cumsum(spy_r))
    spy = pd.DataFrame({"open": sc, "high": sc, "low": sc, "close": sc,
                        "volume": 1e7}, idx)
    return data, spy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    grounds = list(GROUNDS)
    if a.smoke:
        a.smoke_data, grounds = smoke_data(), ["sp400"]

    out = {g: run_ground(g, a) for g in grounds}
    print("\n  ===== VERDICT (pre-stated, H=10) =====")
    ok = True
    for g in grounds:
        m = out[g][0][PRIMARY_H]
        g_ok = m["ic_t"] > 3 and m["top_net"][0] > 0 and m["top_net"][1] > 2
        ok &= g_ok
        print(f"  {g}: rank-corr t {m['ic_t']:+.2f}, top fifth net "
              f"{m['top_net'][0]:+.1f} bps t {m['top_net'][1]:+.2f} -> "
              f"{'signal' if g_ok else 'no'}")
    if a.smoke:
        print("  smoke: planted signal on one synthetic index (pipe check)")
    common = set.intersection(*[set(out[g][1][PRIMARY_H]) for g in grounds])
    print(f"  features replicating on every index at H=10: "
          f"{', '.join(sorted(common)) or 'none'}")
    if ok:
        print("  SIGNAL -- dips ARE separable. Step 2: Primordial search over "
              "the replicating features; winners still owe sp500-PIT/Norgate "
              "and the dup check vs deep_state.")
    else:
        print("  NO SIGNAL -- these 17 measurements can't tell bouncers from "
              "fallers out of sample on every index. A combo search over "
              "them would only find noise. Next lever is NEW data (earnings "
              "dates, news, short interest), not more combinations.")


if __name__ == "__main__":
    main()
