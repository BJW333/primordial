"""
crypto_sleeve.py -- ADD RETURN to momentum, long-only, with a second engine:
crypto with a trend filter (sit in cash during crypto crashes).

SLEEVE (daily, Yahoo prices, rules fixed before running):
  BTC and ETH, half each (BTC alone until ETH has 200 days of history).
  A coin is HELD while yesterday's close is above its 200-day average
  (checked every day, acting next day -- no look-ahead); otherwise that
  half sits in cash (T-bills, BIL). Cost 0.25% per trade (crypto fees +
  slippage).

PORTFOLIOS (monthly, momentum = all-US steady, the ~26.6%/yr version):
  CARVE-OUT (no leverage): momentum 90/80/70% + crypto sleeve 10/20/30%
  ON TOP (leverage): momentum 100% + crypto 10/20/30%, financed at cash
                     + 1%/yr
  References: BTC buy-and-hold, the sleeve alone, momentum levered 1.15x.

PRE-STATED: the sleeve ADDS RETURN if MOM 80% + CRYPTO 20% (carve-out, no
leverage) beats momentum alone on BOTH annual return AND Sharpe in BOTH
2017-02 -> 2021-05 and 2021-06 -> 2026-05.

Run (repo root, Alpaca keys exported for BIL):
    python3.10 scripts/crypto_sleeve.py | tee logs/crypto_sleeve.txt
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
import attention_short_test as AS     # noqa: E402
import factor_portfolio as FP         # noqa: E402
from primordial import data as D      # noqa: E402

ROOT = os.path.dirname(HERE)
MOMF = os.path.join(ROOT, "data", "momentum_v2_returns.csv")
RECENT = pd.Period("2021-06", "M")
COINS = ["BTC-USD", "ETH-USD"]
COST, SPREAD, MA = 0.0025, 0.01, 200
SIZES = (0.10, 0.20, 0.30)


def coin(sym):
    p = os.path.join(ROOT, "data", f"yahoo_{sym}.parquet")
    if os.path.exists(p):
        return pd.read_parquet(p)["close"]
    df = D._fetch_one("yfinance", sym, "1d", "2014-09-01", "2026-06-01")
    df = AS.norm_index(df)
    df.to_parquet(p)
    return df["close"]


def sleeve_daily():
    px = pd.DataFrame({c: coin(c) for c in COINS}).sort_index()
    px = px[~px.index.duplicated()]
    bil = AS.norm_index(FP.alpaca("BIL"))["close"]
    cash_d = bil.pct_change().reindex(px.index).fillna(0.0)
    r = px.pct_change()
    ma = px.rolling(MA, min_periods=MA).mean()
    on = (px > ma).shift(1).fillna(False)            # act the next day
    avail = ma.shift(1).notna()
    n = avail.sum(axis=1).replace(0, np.nan)
    w = avail.div(n, axis=0).fillna(0.0)             # half each if both exist
    held = w * on
    trade = held.diff().abs().sum(axis=1).fillna(0.0)
    ret = (held * r.fillna(0)).sum(axis=1) + \
        (1 - held.sum(axis=1)) * cash_d - COST * trade
    btc = r["BTC-USD"].fillna(0)
    return ret, btc, cash_d, held


def monthly(x):
    return (1 + x).groupby(x.index.to_period("M")).prod() - 1


def stats(x):
    x = x.dropna()
    ann = (1 + x).prod() ** (12 / len(x)) - 1
    vol = x.std() * np.sqrt(12)
    eq = (1 + x).cumprod()
    return ann, vol, x.mean() * 12 / vol if vol else np.nan, \
        (eq / eq.cummax() - 1).min()


def main():
    argparse.ArgumentParser().parse_args()
    m = pd.read_csv(MOMF, index_col=0, parse_dates=True)
    mom = m["STEADY"]
    mom.index = mom.index.to_period("M")
    sd, btc, cash_d, held = sleeve_daily()
    cs, bt, cash = monthly(sd), monthly(btc), monthly(cash_d)
    idx = mom.index.intersection(cs.index)
    mom, cs, bt, cash = (s.reindex(idx) for s in (mom, cs, bt, cash))
    print(f"\n  {len(idx)} months {idx[0]} -> {idx[-1]} | crypto held "
          f"{(held.sum(axis=1) > 0).mean()*100:.0f}% of days | correlation "
          f"sleeve vs momentum {pd.concat([mom, cs], axis=1).corr().iloc[0, 1]:+.2f}")
    books = {"MOM": mom,
             "MOM x1.15 lev": 1.15 * mom - 0.15 * (cash + SPREAD / 12),
             "BTC buy & hold": bt,
             "CRYPTO sleeve": cs}
    for k in SIZES:
        books[f"MOM {1-k:.0%} + CRYPTO {k:.0%}"] = (1 - k) * mom + k * cs
    for k in SIZES:
        books[f"MOM 100% + CRYPTO {k:.0%}"] = mom + k * (cs - cash) - \
            k * SPREAD / 12
    print(f"\n  {'book':<22} | {'all ann':>8} {'vol':>5} {'sh':>5} "
          f"{'worst':>6} | {'2017-21':>8} {'sh':>5} | {'last 5y':>8} {'sh':>5}"
          f" {'worst':>6}")
    res = {}
    for k, b in books.items():
        a, s0, s1 = stats(b), stats(b[b.index < RECENT]), \
            stats(b[b.index >= RECENT])
        res[k] = (a, s0, s1)
        flag = "  <- 30%+" if a[0] >= 0.30 and s1[0] >= 0.30 else ""
        print(f"  {k:<22} | {a[0]*100:>+7.1f}% {a[1]*100:>4.0f}% "
              f"{a[2]:>5.2f} {a[3]*100:>5.0f}% | {s0[0]*100:>+7.1f}% "
              f"{s0[2]:>5.2f} | {s1[0]*100:>+7.1f}% {s1[2]:>5.2f} "
              f"{s1[3]*100:>5.0f}%{flag}")
    cols = ["MOM", "BTC buy & hold", "CRYPTO sleeve", "MOM 80% + CRYPTO 20%",
            "MOM 100% + CRYPTO 20%"]
    yr = pd.DataFrame({c: books[c] for c in cols})
    yr = (1 + yr).groupby(yr.index.year).prod() - 1
    print("\n  by year:  " + "".join(f"{c[:20]:>21}" for c in cols))
    for y, row in yr.iterrows():
        print(f"  {y:<8} " + "".join(f"{x*100:>+20.1f}%" for x in row))
    (_, m0, m1), (_, c0, c1) = res["MOM"], res["MOM 80% + CRYPTO 20%"]
    ok = c0[0] > m0[0] and c1[0] > m1[0] and c0[2] > m0[2] and c1[2] > m1[2]
    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"  crypto trend sleeve adds return to momentum (80/20 carve-out) "
          f" {'PASS' if ok else 'FAIL'}")
    print("  (backtest; expect live results below these numbers)")


if __name__ == "__main__":
    main()
