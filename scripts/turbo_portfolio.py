"""
turbo_portfolio.py -- momentum, turbocharged two ways, on the same
survivorship-free S&P 500 machine (factor_portfolio.py: PIT membership +
Alpaca prices incl. delisted, monthly, top 20, 10 bps/side).

LEVERS
  BIG  pick only from the 100 largest eligible stocks that month (size =
       63-day average dollar volume, close x volume -- a market-cap proxy
       available from price data). Removes the small-stock drag that cost
       every equal-weight portfolio ~3%/yr vs SPY in 2017-2026.
  VM   volatility management (Barroso & Santa-Clara 2015): each month scale
       exposure to  min(15% / realised vol of the strategy's own last 126
       daily returns, 1.5). Calm momentum -> up to 1.5x; wild -> cut.
       Leverage above 1x is charged 4%/yr; un-invested cash earns 0.

VARIANTS (all reported)
  MOM          baseline momentum, whole list (as in factor_portfolio)
  MOM_BIG      momentum within the 100 largest
  MOM_VM       vol-managed momentum, whole list
  MOM_BIG_VM   both levers -- PRIMARY

PRE-STATED (primary MOM_BIG_VM, benchmark SPY -- the bar every portfolio
lost to on 2026-10-25)
  PASS: monthly excess vs SPY > 0 with t > 2 over 2017-2026, positive in
  both halves (2017-2021, 2022-2026), and Sharpe above SPY's.
  Note: 2017-2026 was already used by factor_portfolio/quality_portfolio,
  so this ranks the levers; a pass earns forward paper trading, which is
  the final judge.

Run (repo root, Alpaca keys exported; prices are cached):
    python3.10 scripts/turbo_portfolio.py
    python3.10 scripts/turbo_portfolio.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import factor_portfolio as FP  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "turbo_portfolio_returns.csv")
BIG_N, TARGET_VOL, MAX_LEV, LEV_COST, VOL_WIN = 100, 0.15, 1.5, 0.04, 126
VARIANTS = ["MOM", "MOM_BIG", "MOM_VM", "MOM_BIG_VM"]


def volumes(px):
    """Alpaca volume for each loaded name, aligned to its close series."""
    vols = {}
    for s in px.columns:
        p = FP.D._cache_path("alpaca", s, "1d", FP.A_START, FP.END)
        if os.path.exists(p):
            v = pd.read_parquet(p)["volume"]
            v = v[~v.index.duplicated()]
            vols[s] = v.reindex(px.index).where(px[s].notna())
    return pd.DataFrame(vols).reindex(columns=px.columns)


def run(px, vol, spy, members):
    me = [d for d in px.resample("ME").last().index if d >= FP.FIRST]
    idx = px.index
    dv = (px * vol).rolling(63, min_periods=40).mean()
    rows, prev = [], {}
    daily = {"MOM": [], "MOM_BIG": []}           # unscaled daily histories
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        t_i = idx.get_loc(t)
        mom, _lv, c = FP.scores(px, t_i)
        if mom is None:
            continue
        mem = members(t)
        hist = px.iloc[:t_i + 1].notna().sum()
        elig = [s for s in px.columns
                if s in mem and np.isfinite(c.get(s, np.nan))
                and hist[s] >= 253 and np.isfinite(mom.get(s, np.nan))]
        if len(elig) < 150:
            continue
        size = dv.loc[t, elig].dropna()
        big = list(size.sort_values(ascending=False).index[:BIG_N])
        seg = px.loc[t:t2, elig]
        mret = seg.ffill().iloc[-1] / seg.iloc[0] - 1
        dret = seg.pct_change(fill_method=None).iloc[1:]
        row = {"month": t2, "SPY": FP_spy(spy, t, t2), "EW": mret.mean()}
        for base, pool in (("MOM", elig), ("MOM_BIG", big)):
            pick = list(mom[pool].sort_values(ascending=False).index[:FP.TOP])
            w = {s: 1 / FP.TOP for s in pick}
            old = prev.get(base, {})
            turn = sum(abs(w.get(s, 0) - old.get(s, 0))
                       for s in set(w) | set(old))
            prev[base] = w
            gross = mret[pick].mean()
            row[base] = gross - turn * FP.COST
            row[base + "_turn"] = turn
            # vol-managed: scale set from history BEFORE this month
            hist_d = pd.concat(daily[base]) if daily[base] else pd.Series()
            if len(hist_d) >= VOL_WIN:
                rv = hist_d.iloc[-VOL_WIN:].std() * np.sqrt(252)
                scale = min(TARGET_VOL / rv, MAX_LEV) if rv > 0 else 1.0
            else:
                scale = 1.0
            lev_cost = max(scale - 1, 0) * LEV_COST / 12
            row[base + "_VM"] = scale * (gross - turn * FP.COST) - lev_cost
            row[base + "_VM_scale"] = scale
            daily[base].append(dret[pick].mean(axis=1))
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def FP_spy(spy, t, t2):
    return spy.asof(t2) / spy.asof(t) - 1


def report(res):
    print(f"\n  {len(res)} monthly rebalances, {res.index[0].date()} -> "
          f"{res.index[-1].date()}\n")
    print(f"  {'portfolio':<12} {'annual':>8} {'vol':>7} {'sharpe':>7} "
          f"{'max DD':>8} {'vs SPY/yr':>10} {'t':>6} {'avg exposure':>13}")
    for col in VARIANTS + ["SPY", "EW"]:
        a, v, s, dd = FP.stats(res[col])
        ex = res[col] - res["SPY"]
        exs = (f"{ex.mean()*1200:>+9.1f}% {FP.tstat(ex):>+6.2f}"
               if col != "SPY" else f"{'':>10} {'':>6}")
        sc = (f"{res[col + '_scale'].mean():>12.2f}x"
              if col + "_scale" in res else "")
        print(f"  {col:<12} {a*100:>+7.1f}% {v*100:>6.1f}% {s:>7.2f} "
              f"{dd*100:>7.1f}% {exs} {sc}")
    print("\n  halves (MOM_BIG_VM excess vs SPY, per year):")
    for lab, part in (("2017-2021", res[res.index < FP.SPLIT]),
                      ("2022-2026", res[res.index >= FP.SPLIT])):
        ex = part["MOM_BIG_VM"] - part["SPY"]
        print(f"    {lab}: {ex.mean()*1200:+.1f}%/yr  t {FP.tstat(ex):+.2f}")


def smoke():
    (px, spy), members, _ = FP.smoke_data()
    rng = np.random.default_rng(6)
    vol = pd.DataFrame(rng.uniform(1e5, 1e6, px.shape), px.index, px.columns)
    # planted: half the names are "big" and carry the momentum drift
    big = px.columns[::2]
    px = px.copy()
    trend = np.exp(np.cumsum(np.full(len(px), 0.0004)))
    for i, s in enumerate(big):
        if i % 3 == 0:
            px[s] = px[s] * trend
        vol[s] *= 20
    common = rng.normal(0, 0.01, len(px))       # market-wide swings
    spy = (px[big].pct_change().mean(axis=1).fillna(0) + common).add(
        1).cumprod() * 100
    px = px.mul(np.exp(np.cumsum(common)), axis=0)
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
        vol = volumes(px)
    res = run(px, vol, spy, members)
    report(res)
    if not a.smoke:
        res.to_csv(OUT)
        print(f"\n  monthly returns -> {OUT}")
    ex = res["MOM_BIG_VM"] - res["SPY"]
    h1, h2 = ex[ex.index < FP.SPLIT].mean(), ex[ex.index >= FP.SPLIT].mean()
    t = FP.tstat(ex)
    s_p, s_spy = FP.stats(res["MOM_BIG_VM"])[2], FP.stats(res["SPY"])[2]
    print("\n  ===== VERDICT (pre-stated, primary MOM_BIG_VM vs SPY) =====")
    if a.smoke:
        print("  smoke: planted big-cap momentum (pipe check)")
    if ex.mean() > 0 and t > 2 and h1 > 0 and h2 > 0 and s_p > s_spy:
        print(f"  PASS -- {ex.mean()*1200:+.1f}%/yr over SPY (t {t:+.2f}), "
              f"Sharpe {s_p:.2f} vs {s_spy:.2f}. Next: forward paper.")
    else:
        print(f"  FAIL -- {ex.mean()*1200:+.1f}%/yr vs SPY, t {t:+.2f}, "
              f"halves {h1*1200:+.1f}% / {h2*1200:+.1f}%, Sharpe {s_p:.2f} "
              f"vs SPY {s_spy:.2f}.")


if __name__ == "__main__":
    main()
