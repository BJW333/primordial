"""
blowup_scan.py -- three lesser-known "before it blows up" signals on the
survivorship-free S&P 500 machine (PIT membership + Alpaca prices incl.
delisted). DECIDED ON THE LAST FIVE YEARS.

SIGNALS (scored at every month-end t, from data up to t only)
  ACCUM    quiet accumulation: price flat (|63-day return| < 10%) but
           over the last 50 days, volume on up days / volume on down days
           is high. Score = that ratio; top 20 flat stocks.
  SQUEEZE  squeeze then breakout: the stock's 20-day volatility hit its
           lowest 10% of the past two years within the last 20 days, AND
           it set a 52-week closing high in the last 5 days, AND 5-day
           volume >= 1.5x its 63-day average. Flagged stocks ranked by
           volume surge, up to 20; unused slots held in SPY.
  LAGGARD  catch-up: its 10 most correlated stocks (daily returns, past
           year) rose last month, it didn't. Score = peers' 21-day return
           minus its own; top 20 where peers were up.
  MOM      plain 12-1 momentum top 20 -- context only (the one to beat).

TWO TESTS, PRE-STATED, last five years (2021-06 -> 2026-05)
  1. BLOWUP HIT RATE: share of picks up >= 50% six months (126 trading
     days) later vs the share of ALL eligible stocks that did (base rate).
     Monthly difference, Newey-West t (6 lags, windows overlap).
  2. PORTFOLIO: top 20 bought monthly, held a month, 10 bps/side, vs SPY
     and vs SPY leveraged to the same volatility (4%/yr borrow).
  PASS (per signal): hit-rate lift >= 1.5x with t > 2.4 (3 signals,
  Bonferroni) AND portfolio return beats same-risk SPY.
  2017-2021 printed as context.

Run (repo root, Alpaca keys exported; prices cached):
    python3.10 scripts/blowup_scan.py --smoke
    python3.10 scripts/blowup_scan.py > logs/blowup_scan.txt
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
import factor_portfolio as FP  # noqa: E402
import turbo_portfolio as T   # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "blowup_scan_returns.csv")
RECENT = pd.Timestamp("2021-06-01")
SIGS = ["ACCUM", "SQUEEZE", "LAGGARD"]
FWD, JUMP, LEV_COST = 126, 0.50, 0.04
PICKLOG = []          # (month, signal, symbol, month return) for diagnostics


def signals(px, vol, r, sd20, t_i, elig):
    P = px[elig]
    out = {}
    # ACCUM
    r50 = r[elig].iloc[t_i - 49:t_i + 1]
    v50 = vol[elig].iloc[t_i - 49:t_i + 1]
    up = v50.where(r50 > 0).sum()
    dn = v50.where(r50 < 0).sum()
    ud = up / dn.replace(0, np.nan)
    flat = (P.iloc[t_i] / P.iloc[t_i - 63] - 1).abs() < 0.10
    out["ACCUM"] = ud[flat].dropna()
    # SQUEEZE
    s = sd20[elig].iloc[t_i - 503:t_i + 1]
    q10 = s.quantile(0.10)
    squeezed = s.iloc[-20:].min() <= q10
    hi252 = P.iloc[t_i - 251:t_i + 1].max()
    brk = P.iloc[t_i - 4:t_i + 1].max() >= hi252 * 0.9999
    V = vol[elig]
    surge = V.iloc[t_i - 4:t_i + 1].mean() / \
        V.iloc[t_i - 67:t_i - 4].mean().replace(0, np.nan)
    flag = squeezed & brk & (surge >= 1.5)
    out["SQUEEZE"] = surge[flag].dropna()
    # LAGGARD
    R = r[elig].iloc[t_i - 251:t_i + 1]
    R = R.loc[:, R.notna().sum() >= 240]
    C = R.corr(min_periods=200).values.copy()
    np.fill_diagonal(C, -np.inf)
    C = np.nan_to_num(C, nan=-np.inf)
    names = R.columns
    r21 = (P[names].iloc[t_i] / P[names].iloc[t_i - 21] - 1).values
    top = np.argsort(-C, axis=1)[:, :10]
    peer = np.nanmean(r21[top], axis=1)
    gap = pd.Series(peer - r21, names)
    out["LAGGARD"] = gap[(peer > 0) & np.isfinite(gap.values)]
    return out


def nw_t(x, lags=6):
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = (e @ e) / n
    for L in range(1, lags + 1):
        s += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return x.mean() / np.sqrt(s / n) if s > 0 else np.nan


def run(px, vol, spy, members):
    idx = px.index
    me = [d for d in px.resample("ME").last().index if d >= FP.FIRST]
    r = px.pct_change(fill_method=None)
    sd20 = r.rolling(20, min_periods=15).std()
    pxf = px.ffill()
    rows, hits, prev = [], [], {}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        if t_i < 520:
            continue
        mom, _lv, c = FP.scores(px, t_i)
        if mom is None:
            continue
        mem = members(t)
        hist = px.iloc[:t_i + 1].notna().sum()
        elig = [s for s in px.columns
                if s in mem and np.isfinite(c.get(s, np.nan))
                and hist[s] >= 520 and np.isfinite(mom.get(s, np.nan))]
        if len(elig) < 150:
            continue
        sig = signals(px, vol, r, sd20, t_i, elig)
        sig["MOM"] = mom[elig]
        seg = px.loc[t:t2, elig]
        mret = seg.ffill().iloc[-1] / seg.iloc[0] - 1
        spy_m = spy.asof(t2) / spy.asof(t) - 1
        row = {"month": t2, "SPY": spy_m, "EW": mret.mean()}
        # forward six-month blow-ups
        f_i = t_i + FWD
        fwd = (pxf[elig].iloc[f_i] / px[elig].iloc[t_i] - 1
               if f_i < len(idx) else None)
        h = {"month": t2}
        if fwd is not None:
            h["base"] = (fwd.dropna() >= JUMP).mean()
        for name, sc in sig.items():
            pick = list(sc.sort_values(ascending=False).index[:FP.TOP])
            n = len(pick)
            w = {s: 1 / FP.TOP for s in pick}
            old = prev.get(name, {})
            turn = sum(abs(w.get(s, 0) - old.get(s, 0))
                       for s in set(w) | set(old))
            prev[name] = w
            gross = (mret[pick].sum() / FP.TOP if n else 0) + \
                (FP.TOP - n) / FP.TOP * spy_m
            row[name] = gross - turn * FP.COST
            row[name + "_n"] = n
            PICKLOG.extend((t2, name, s, mret[s]) for s in pick)
            if fwd is not None and n:
                h[name] = (fwd[pick].dropna() >= JUMP).mean()
        rows.append(row)
        hits.append(h)
    return (pd.DataFrame(rows).set_index("month"),
            pd.DataFrame(hits).set_index("month"))


def stats(x):
    x = x.dropna()
    ann = (1 + x).prod() ** (12 / len(x)) - 1
    v = x.std() * np.sqrt(12)
    eq = (1 + x).cumprod()
    return ann, v, x.mean() * 12 / v, (eq / eq.cummax() - 1).min()


def report(res, hit, label):
    print(f"\n  ===== {label}: {len(res)} months, {res.index[0].date()} -> "
          f"{res.index[-1].date()} =====")
    hb = hit.dropna(subset=["base"])
    print(f"  blow-up base rate (any eligible stock up >= 50% in 6 months): "
          f"{hb['base'].mean()*100:.1f}%  ({len(hb)} months with 6m data)")
    print(f"\n  {'signal':<9} {'picks/mo':>8} {'hit rate':>9} {'lift':>6} "
          f"{'t':>6} | {'annual':>7} {'vol':>6} {'sharpe':>6} {'worst':>7}"
          f" {'same-risk SPY':>13}")
    _, spy_v, _, _ = stats(res["SPY"])
    out = {}
    for name in SIGS + ["MOM"]:
        hh = hb[[name, "base"]].dropna() if name in hb else pd.DataFrame()
        hr = hh[name].mean() if len(hh) else np.nan
        base = hh["base"].mean() if len(hh) else np.nan
        t = nw_t(hh[name] - hh["base"]) if len(hh) else np.nan
        a, v, s, dd = stats(res[name])
        lev = v / spy_v
        m_a = stats(lev * res["SPY"] - max(lev - 1, 0) * LEV_COST / 12)[0]
        print(f"  {name:<9} {res[name + '_n'].mean():>8.1f} {hr*100:>8.1f}% "
              f"{hr/base:>5.2f}x {t:>+6.2f} | {a*100:>+6.1f}% {v*100:>5.1f}% "
              f"{s:>6.2f} {dd*100:>6.1f}% {m_a*100:>+12.1f}%")
        out[name] = (hr / base, t, a, m_a)
    a, v, s, dd = stats(res["SPY"])
    print(f"  {'SPY':<9} {'':>8} {'':>9} {'':>6} {'':>6} | {a*100:>+6.1f}% "
          f"{v*100:>5.1f}% {s:>6.2f} {dd*100:>6.1f}%")
    return out


def smoke():
    px, vol, spy, members = T.smoke()
    rng = np.random.default_rng(11)
    px, vol = px.copy(), vol.copy()
    # planted: some names get quiet up-volume for 50 days, then jump 60%
    cols = list(px.columns)
    for j, s in enumerate(cols[1::7]):
        for start in range(600 + j * 37, len(px) - 200, 400):
            seg = slice(start, start + 50)
            r = np.log(px[s]).diff()
            up = r.iloc[seg] > 0
            vol.iloc[start:start + 50, vol.columns.get_loc(s)] *= \
                np.where(up, 3.0, 0.5)
            ramp = np.ones(len(px))
            ramp[start + 50:start + 176] = np.linspace(1, 1.6, 126)
            ramp[start + 176:] = 1.6
            px[s] = px[s] * ramp
    return px, vol, spy, members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, spy, members = smoke()
    else:
        if not os.environ.get("ALPACA_API_KEY"):
            sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
        members, union = FP.membership()
        px, spy = FP.load_prices(union, members)
        vol = T.volumes(px)
    res, hit = run(px, vol, spy, members)
    if not a.smoke:
        res.to_csv(OUT)
    old = res.index < RECENT
    if old.sum() > 6:
        report(res[old], hit[hit.index < RECENT], "CONTEXT 2017-2021")
    rec = report(res[~old], hit[hit.index >= RECENT],
                 "DECIDING: LAST FIVE YEARS")
    print("\n  ===== VERDICT (pre-stated: lift >= 1.5x, t > 2.4, beats "
          "same-risk SPY) =====")
    if a.smoke:
        print("  smoke: planted quiet-accumulation-then-jump (ACCUM should "
              "pass)")
    for name in SIGS:
        lift, t, ann, m_a = rec[name]
        ok = lift >= 1.5 and t > 2.4 and ann > m_a
        print(f"  {name:<8} lift {lift:.2f}x  t {t:+.2f}  return "
              f"{ann*100:+.1f}% vs same-risk SPY {m_a*100:+.1f}% -> "
              f"{'PASS' if ok else 'fail'}")


if __name__ == "__main__":
    main()
