"""
fx_rotation.py -- Blake's "trade currency to currency, harvest small gains, almost every country".
PRE-REGISTERED 2026-11-15, before any real-data run. No tuning after results.

Idea: every week, across ~20 currencies, BUY the ones that fell most against the dollar over the
past week and SELL the ones that rose most -- small snap-back gains, harvested weekly, many currencies.

Universe (USD per unit): the 8 from fx_screen (AUD NZD CAD NOK EUR GBP JPY CHF) plus
  SEK PLN HUF CZK MXN ZAR TRY SGD CNH, and BRL KRW INR if Dukascopy has them (skipped if not).
  A currency joins once it has 60 days of prices.
Data: Dukascopy hourly (same cache/folder as fx_screen: data/fx_1h/). The new currencies are a
  one-time download (~20-40 min).

Rule (weekly)
  signal : last trading day of each week, 16:00 New York close; trade at the 17:00 close
  score  : the currency's return vs USD over the last 5 trading days
  book   : long the bottom third (biggest losers), short the top third (biggest winners),
           weighted by 1 / 60-day volatility inside each side, $1 long + $1 short; hold one week
  needs  : at least 8 currencies with prices that week, else flat
Costs per side (bps): majors as fx_screen; SEK 2, SGD 3, CNH 3, PLN 3, MXN 3, HUF 4, CZK 4,
  ZAR 5, BRL 6, KRW 6, INR 6, TRY 10. Every row also at 2x.
Known bias, stated now: spot prices only -- the interest earned or paid for holding a currency
  (carry) is NOT included. That matters most for TRY/ZAR/BRL/MXN (high rates). The G10-only row,
  where carry is small, is the cleaner read.
Rows: ALL (judged), G10 only and EM only (shown, not separately judged).
Control (must FAIL): RANDOM -- same book, currencies ranked at random each week.
Pass bar (same as always): Sharpe >= 0.5 2012-07 -> 2021-10, >= 0.7 2021-11 -> end, 50/50
  equal-risk monthly blend with CHIMERA > 1.42, whole-period Sharpe > 0 at 2x costs.

Run (repo root; fx_screen.py must be in scripts/ too):
    python3.10 scripts/fx_rotation.py --smoke
    python3.10 scripts/fx_rotation.py | tee logs/fx_rotation.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fx_screen as fx  # noqa: E402

EXTRA = {"SEK": "USDSEK", "PLN": "USDPLN", "HUF": "USDHUF", "CZK": "USDCZK", "MXN": "USDMXN",
         "ZAR": "USDZAR", "TRY": "USDTRY", "SGD": "USDSGD", "CNH": "USDCNH", "BRL": "USDBRL",
         "KRW": "USDKRW", "INR": "USDINR"}
EXTRA_COST = {"SEK": 2, "SGD": 3, "CNH": 3, "PLN": 3, "MXN": 3, "HUF": 4, "CZK": 4, "ZAR": 5,
              "BRL": 6, "KRW": 6, "INR": 6, "TRY": 10}
fx.COST_BP.update(EXTRA_COST)
G10 = ["AUD", "NZD", "CAD", "NOK", "EUR", "GBP", "JPY", "CHF", "SEK"]
MIN_N = 8


# =================================================================== data
def download_extra():
    os.makedirs(fx.DATA, exist_ok=True)
    sess = fx.session()
    for cur, inst in EXTRA.items():
        p = os.path.join(fx.DATA, f"{cur}.csv")
        skip = os.path.join(fx.DATA, f"{cur}.none")
        if os.path.exists(p) or os.path.exists(skip):
            continue
        try:
            probe_dir = os.path.join(fx.RAW, inst)
            if not os.path.isdir(probe_dir):     # quick probe: does Dukascopy carry it at all?
                if fx.fetch_month(sess, inst, 2024, 1) is None:
                    print(f"  {inst}: not available -> skipped")
                    open(skip, "w").close()
                    continue
            df = fx.get_instrument(sess, inst)
        except fx.Throttled:
            raise SystemExit(fx.THROTTLE_MSG)
        if df is None:
            open(skip, "w").close()
            continue
        fx.write_csv(df, inst, p, 1.0)            # raw scale: returns don't care


def load_all():
    P = fx.load_real()
    download_extra()
    px = {}
    for cur in EXTRA:
        p = os.path.join(fx.DATA, f"{cur}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df = df[(df.volume > 0) & (df.close > 0)]
        px[cur] = 1 / df.close                                 # all USDxxx -> USD per unit
    X = pd.DataFrame(px)
    X = X[(X.index >= fx.START) & (X.index < fx.END)]
    X.index = X.index.tz_localize("UTC")
    P = P.join(X, how="outer").sort_index().ffill(limit=3)
    return P


# =================================================================== strategy
def weekly_book(C, E, cols, cmult, random_rank=False, seed=17):
    C, E = C[cols], E[cols]
    score = C - C.shift(5)
    vol = E.diff().rolling(60, min_periods=60).std()
    week = pd.Series(C.index.to_period("W-FRI"), index=C.index)
    last = week != week.shift(-1)                        # last trading day of each week
    rng = np.random.default_rng(seed)
    pos = pd.DataFrame(np.nan, index=C.index, columns=cols)
    for t in C.index[last.values]:
        s = score.loc[t].dropna()
        v = vol.loc[t].reindex(s.index)
        ok = s.index[v.notna() & (v > 0)]
        s, v = s[ok], v[ok]
        if len(s) < MIN_N:
            pos.loc[t] = 0.0
            continue
        if random_rank:
            s = pd.Series(rng.permutation(s.values), index=s.index)
        q = len(s) // 3
        order = s.sort_values()
        lo, hi = order.index[:q], order.index[-q:]
        w = pd.Series(0.0, index=cols)
        w[lo] = (1 / v[lo]) / (1 / v[lo]).sum()
        w[hi] = -(1 / v[hi]) / (1 / v[hi]).sum()
        pos.loc[t] = w
    pos = pos.ffill().fillna(0.0)                         # hold until next week's rebalance
    return fx.book(pos, E.diff(), cmult), pos


def report(P):
    C, E = fx.daily(P)
    have = [c for c in C.columns if C[c].notna().sum() > 250]
    print(f"\n  data {P.index[0].date()} -> {P.index[-1].date()}, {len(have)} currencies: "
          f"{' '.join(have)}")
    first = {c: C[c].first_valid_index().date() for c in have}
    late = {c: d for c, d in first.items() if d > pd.Timestamp("2005-01-01").date()}
    if late:
        print("  joined later: " + ", ".join(f"{c} {d}" for c, d in late.items()))
    em = [c for c in have if c not in G10]
    rows = {"ALL (judged)": (have, False), "  G10 only": ([c for c in have if c in G10], False),
            "  EM only": (em, False), "control RANDOM": (have, True)}
    print(f"  {'book':<15}| 2012-21 ann  shp | last5y ann  vol  shp  worst | whole shp 2x-cost |"
          f" corrCHIM CHIM blend badM | turnover/yr | verdict")
    res = {}
    for name, (cols, rnd) in rows.items():
        if len(cols) < MIN_N and not name.startswith("control"):
            if len(cols) < 3:
                print(f"  {name:<15}| too few currencies ({len(cols)})")
                continue
        x, pos = weekly_book(C, E, cols, 1.0, rnd)
        x2, _ = weekly_book(C, E, cols, 2.0, rnd)
        a0, _, s0, _ = fx.stats(fx.win(x, fx.EARLY))
        a1, v1, s1, d1 = fx.stats(fx.win(x, fx.RECENT))
        _, _, sw, _ = fx.stats(x)
        _, _, sw2, _ = fx.stats(x2)
        csh, bsh, corr, bad = fx.blend(x)
        to = pos.diff().abs().sum(axis=1).sum() / (len(pos) / 252)
        ok = (s0 >= 0.5) and (s1 >= 0.7) and (bsh > csh) and (sw2 > 0)
        res[name] = ok
        judged = name.startswith("ALL") or name.startswith("control")
        verdict = ("PASS" if ok else "fail") if judged else ("(info)")
        print(f"  {name:<15}| {a0*100:+6.1f}% {s0:5.2f} | {a1*100:+6.1f}% {v1*100:4.1f}% {s1:5.2f}"
              f" {d1*100:5.0f}% | {sw:6.2f} {sw2:7.2f} | {corr:+6.2f} {csh:5.2f} {bsh:5.2f}"
              f" {bad*100:+5.1f}% | {to:10.0f}x | {verdict}")
    print("  control: " + ("fails (good)" if not res.get("control RANDOM") else
                           "PASSED -> the test setup is suspect"))
    print("  ann = average yearly return (log) on $1 long + $1 short, spot only (no carry). "
          "turnover = $ traded per $1 per year.\n  pass = 2012-21 >= 0.5, last5y >= 0.7, "
          "blend > CHIM, 2x-cost > 0.  badM = avg month when CHIMERA < -3%.")


# =================================================================== smoke
def synth(planted, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2004-01-04 22:00", "2026-10-30 20:00", freq="h", tz="UTC")
    wd, hr = idx.dayofweek, idx.hour
    idx = idx[~((wd == 5) | ((wd == 6) & (hr < 22)) | ((wd == 4) & (hr >= 21)))]
    n = len(idx)
    curs = list(fx.INST) + [c for c in EXTRA if c not in ("BRL", "KRW", "INR")]
    com = rng.normal(0, 0.0012, n)
    data = {}
    for c in curs:
        vol = 0.0013 if c in G10 else 0.002
        r = 0.5 * com + rng.normal(0, vol, n)
        if planted:                       # each currency snaps back over about a week
            u = np.zeros(n)
            e = rng.normal(0, 0.0015, n)
            for i in range(1, n):
                u[i] = u[i - 1] * (1 - 1 / 60) + e[i]
            r = r + np.diff(u, prepend=0)
        data[c] = np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        for planted in (False, True):
            print(f"\n===== SMOKE {'PLANTED weekly snap-back (ALL should PASS, control fail)' if planted else 'NOISE (everything should fail)'}")
            report(synth(planted))
        return
    report(load_all())


if __name__ == "__main__":
    main()
