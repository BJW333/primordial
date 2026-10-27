"""
adline_obv.py -- Advance-Decline line (breadth) + ADX (trend strength) +
On-Balance Volume (volume flow), on STOCKS and COMMODITIES.

  A/D LINE  market level: cumulative (# rising - # falling) each day.
            "Healthy" = above its 100-day average.
  ADX       per asset: ADX > 25 = strong trend; +DI vs -DI = direction.
  OBV       per asset: cumulative volume, + on up days, - on down days.
            "Flowing in" = OBV above its 50-day average.

STOCKS: S&P 500 point-in-time members, Alpaca daily OHLCV incl. delisted
(the factor_portfolio cache -- needs the high/low that ADX uses).
  S_BREADTH  hold SPY while the S&P 500 A/D line is healthy, else cash
             (checked daily, 2 bps per switch)
  S_ADX_OBV  each month-end: stocks with ADX > 25, +DI > -DI and OBV
             above its 50-day avg; top 20 by ADX, equal weight, held a
             month, 10 bps per trade; empty slots in SPY
  S_COMBO    S_ADX_OBV, but the whole book sits in cash for any month
             that starts with breadth unhealthy
  Select the best Sharpe on 2017-02 -> 2021-05; that one PASSES if over
  the last five years it beats SPY leveraged to the same volatility AND
  has Sharpe above SPY's.

COMMODITIES: the 14 ETFs of commodity_adx.py (real prices and volumes).
  C_BREADTH  hold the equal-risk long basket while commodity breadth (A/D
             line across the 14 ETFs) is healthy, else cash
  C_ADX_OBV  long when ADX > 25, +DI > -DI and OBV above its 50d avg;
             short when ADX > 25, -DI > +DI and OBV below; else flat
  C_COMBO    C_ADX_OBV, longs only while breadth healthy, shorts only
             while unhealthy
  Select the best Sharpe on 2008 -> 2021-05; that one PASSES if its last-
  five-year Sharpe >= 0.5 AND a 50/50 blend with stock momentum
  (momentum_v2 STEADY) beats momentum alone.

Run (repo root; caches already warm):
    python3.10 scripts/adline_obv.py --smoke
    python3.10 scripts/adline_obv.py | tee logs/adline_obv.txt
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
import blowup_scan as B       # noqa: E402
import commodities_test as C  # noqa: E402
import commodity_adx as A     # noqa: E402
import factor_portfolio as FP  # noqa: E402

S_RECENT = pd.Timestamp("2021-06-01")
S_FIRST = pd.Timestamp("2017-02-28")


# ------------------------------------------------------------ indicators
def adline(Cl, members=None):
    r = Cl.pct_change(fill_method=None)
    if members is not None:
        r = r.where(members)
    adv = (r > 0).sum(axis=1) - (r < 0).sum(axis=1)
    ad = adv.cumsum()
    return ad > ad.rolling(100, min_periods=80).mean()


def obv_up(Cl, V):
    step = np.sign(Cl.diff()).fillna(0) * V.fillna(0)
    o = step.cumsum()
    return o > o.rolling(50, min_periods=40).mean()


# --------------------------------------------------------------- stocks
def load_stocks():
    members, union = FP.membership()
    px, spy = FP.load_prices(union, members)
    pix = pd.to_datetime(px.index)
    if pix.tz is not None:
        pix = pix.tz_convert(None)
    px.index = pix.normalize()
    six = pd.to_datetime(spy.index)
    if six.tz is not None:
        six = six.tz_convert(None)
    spy.index = six.normalize()
    H, L, V = {}, {}, {}
    for s in px.columns:
        p = FP.D._cache_path("alpaca", s, "1d", FP.A_START, FP.END)
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d.columns = [c.lower() for c in d.columns]
        ix = pd.to_datetime(d.index)
        if ix.tz is not None:                 # cache may be UTC-stamped
            ix = ix.tz_convert(None)
        d.index = ix.normalize()
        d = d[~d.index.duplicated(keep="last")]
        ok = px[s].notna()
        for tgt, col in ((H, "high"), (L, "low"), (V, "volume")):
            tgt[s] = d[col].reindex(px.index).where(ok)
    H, L, V = (pd.DataFrame(x).reindex(columns=px.columns) for x in (H, L, V))
    cov = H.notna().sum().sum() / max(px.notna().sum().sum(), 1)
    print(f"  high/low/volume matched to {cov:.0%} of price days")
    if cov < 0.8:
        sys.exit("  high/low data does not line up with prices -- paste "
                 "this output so the date alignment can be fixed")
    return px, H, L, V, spy, members


def stock_books(px, H, L, V, spy, members):
    idx = px.index
    mem_mask = pd.DataFrame(False, idx, px.columns)
    me_all = px.resample("ME").last().index
    for d in me_all:                      # membership mask, month by month
        rows = (idx > d - pd.offsets.MonthEnd(1)) & (idx <= d)
        m = members(d)
        cols = [c for c in px.columns if c in m]
        mem_mask.loc[rows, cols] = True
    healthy = adline(px, mem_mask)
    spy = spy.reindex(idx).ffill()
    spy_r = spy.pct_change().fillna(0)
    sig = healthy.astype(float)
    daily = sig.shift(1).fillna(0) * spy_r - sig.diff().abs().fillna(0) \
        * 0.0002
    breadth_m = (1 + daily).resample("ME").prod() - 1

    a, pdi, ndi = A.adx(H, L, px)
    obv = obv_up(px, V)
    me = [d for d in me_all if d >= S_FIRST]
    rows, prev = [], {"S_ADX_OBV": set(), "S_COMBO": set()}
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        m = members(t)
        ok = [s for s in px.columns if s in m and np.isfinite(a.at[t, s])
              and a.at[t, s] > 25 and pdi.at[t, s] > ndi.at[t, s]
              and bool(obv.at[t, s])]
        pick = list(a.loc[t, ok].sort_values(ascending=False).index[:20])
        seg = px.loc[t:t2, pick]
        mret = (seg.ffill().iloc[-1] / seg.iloc[0] - 1).fillna(0) \
            if pick else pd.Series(dtype=float)
        spy_m = spy.asof(t2) / spy.asof(t) - 1
        row = {"month": t2, "SPY": spy_m}
        for name in ("S_ADX_OBV", "S_COMBO"):
            if name == "S_COMBO" and not bool(healthy.loc[t]):
                hold, ret = set(), 0.0
            else:
                hold = set(pick)
                ret = (mret.sum() + (20 - len(pick)) * spy_m) / 20
            trades = len(hold - prev[name]) + len(prev[name] - hold)
            row[name] = ret - trades / 20 * 0.0010
            prev[name] = hold
        row["n_picks"] = len(pick)
        rows.append(row)
    res = pd.DataFrame(rows).set_index("month")
    bm = breadth_m.copy()
    bm.index = [idx[idx <= d][-1] for d in bm.index]
    res["S_BREADTH"] = bm.reindex(res.index)
    return res


def stock_report(res):
    out = {}
    print(f"\n  STOCKS (S&P 500 PIT): avg qualifying ADX+OBV picks "
          f"{res['n_picks'].mean():.1f}/20")
    print(f"  {'variant':<10} | {'2017-21 ann':>11} {'sh':>5} | "
          f"{'last5y ann':>10} {'vol':>4} {'sh':>5} {'worst':>6} | "
          f"{'same-risk SPY':>13} {'SPY sh':>6}")
    for k in ("S_BREADTH", "S_ADX_OBV", "S_COMBO", "SPY"):
        old = res.loc[res.index < S_RECENT, k].dropna()
        new = res.loc[res.index >= S_RECENT, k].dropna()
        a0, _, s0, _ = B.stats(old)
        a1, v1, s1, dd = B.stats(new)
        sp = res.loc[new.index, "SPY"]
        _, sv, ss, _ = B.stats(sp)
        lev = v1 / sv
        m_a = B.stats(lev * sp - max(lev - 1, 0) * 0.04 / 12)[0]
        print(f"  {k:<10} | {a0*100:>+10.1f}% {s0:>5.2f} | {a1*100:>+9.1f}% "
              f"{v1*100:>3.0f}% {s1:>5.2f} {dd*100:>5.0f}% | "
              f"{m_a*100:>+12.1f}% {ss:>6.2f}")
        out[k] = (s0, a1, s1, m_a, ss)
    cand = max(("S_BREADTH", "S_ADX_OBV", "S_COMBO"), key=lambda k: out[k][0])
    s0, a1, s1, m_a, ss = out[cand]
    ok = a1 > m_a and s1 > ss
    print(f"  CANDIDATE (best 2017-21 Sharpe): {cand} -> last 5y "
          f"{a1*100:+.1f}% vs same-risk SPY {m_a*100:+.1f}%, Sharpe {s1:.2f} "
          f"vs SPY {ss:.2f} -> {'PASS' if ok else 'FAIL'}")
    return cand, ok


# ----------------------------------------------------------- commodities
def comm_books(H, L, Cl, V):
    R = Cl.pct_change(fill_method=None).clip(-0.25, 0.25)
    vol = R.rolling(60, min_periods=40).std() * np.sqrt(252)
    healthy = adline(Cl)
    a, pdi, ndi = A.adx(H, L, Cl)
    up = obv_up(Cl, V)
    d = np.sign(pdi - ndi)
    strong = a > 25
    sig = pd.DataFrame(0.0, Cl.index, Cl.columns)
    sig[strong & (d > 0) & up] = 1.0
    sig[strong & (d < 0) & ~up] = -1.0
    sig = sig.where(a.notna())
    hb = pd.DataFrame(np.repeat(healthy.values[:, None], Cl.shape[1], 1),
                      Cl.index, Cl.columns)
    combo = sig.where(((sig > 0) & hb) | ((sig < 0) & ~hb), 0.0) \
        .where(sig.notna())
    basket = pd.DataFrame(1.0, Cl.index, Cl.columns).where(Cl.notna()) \
        .where(hb, 0.0).where(Cl.notna())
    return {k: A.book(W, R, vol) for k, W in
            (("C_BREADTH", basket), ("C_ADX_OBV", sig), ("C_COMBO", combo))}


def comm_report(books, mom_m):
    print(f"\n  COMMODITIES (14 ETFs)")
    print(f"  {'variant':<10} | {'2008-21 ann':>11} {'sh':>5} | "
          f"{'last5y ann':>10} {'sh':>5} {'worst':>6} {'2022':>5} | "
          f"{'mom':>5} {'blend':>5}")
    out = {}
    for k, d in books.items():
        a0, s0, a1, v1, s1, dd, y22 = A.split(d)
        msh, bsh, _ = C.blend_sharpe(mom_m, C.monthly(d), C.RECENT, C.DEC1)
        print(f"  {k:<10} | {a0*100:>+10.1f}% {s0:>5.2f} | {a1*100:>+9.1f}% "
              f"{s1:>5.2f} {dd*100:>5.0f}% {y22*100:>+4.0f}% | {msh:>5.2f} "
              f"{bsh:>5.2f}")
        out[k] = (s0, s1, bsh > msh)
    cand = max(out, key=lambda k: out[k][0])
    s0, s1, helps = out[cand]
    ok = s1 >= 0.5 and helps
    print(f"  CANDIDATE (best 2008-21 Sharpe): {cand} -> last 5y Sharpe "
          f"{s1:.2f} (need >= 0.5), improves blend {helps} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return cand, ok


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", choices=["stocks", "commodities"])
    a = ap.parse_args()
    verdicts = {}
    if a.only != "commodities":
        if a.smoke:
            import turbo_portfolio as T
            px, vol, spy, members = T.smoke()
            H, L = px * 1.01, px * 0.99
            res = stock_books(px, H, L, vol, spy, members)
        else:
            if not os.environ.get("ALPACA_API_KEY"):
                sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first "
                         "(prices are cached; keys are only checked)")
            res = stock_books(*load_stocks())
        if res["n_picks"].mean() < 3:
            print("  WARNING: almost no stocks qualified -- result invalid")
        verdicts["stocks"] = stock_report(res)
    if a.only != "stocks":
        if a.smoke:
            H, L, Cl, V = A.smoke()
            mom_m = C.monthly(pd.Series(np.random.default_rng(9).normal(
                0.0005, 0.01, len(Cl)), Cl.index))
        else:
            H, L, Cl, V = A.load()
            mom_m, _ = C.stock_momentum()
        verdicts["commodities"] = comm_report(comm_books(H, L, Cl, V), mom_m)
    print("\n  ===== VERDICT (pre-stated: chosen on the early period, judged "
          "once on the last five years) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    for u, (cand, ok) in verdicts.items():
        print(f"  {u:<12} {cand:<10} {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
