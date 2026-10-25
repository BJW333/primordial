"""
dip_combo_classic.py -- SECOND combo search (recorded as such): the same
search as dip_combo_search.py (commit dce4200) with 8 classic trading
indicators added to the 17:

  rsi2          2-day RSI (very short-term oversold)
  rsi14         14-day RSI
  bb_pctb       where the close sits in its 20-day Bollinger Bands
                (0 = lower band, 1 = upper band)
  bb_width      Bollinger band width (how stretched volatility is)
  macd_hist     MACD histogram (12/26/9) in units of 14-day ATR
  stoch14       14-day stochastic %K
  vs_20dma      distance from the 20-day average
  vs_50dma      distance from the 50-day average

25 indicators x thirds, 1-3 condition ANDs = 64,875 combos x 2 holds
= 129,750 tests. Same rules as dce4200: discover S&P 500 2010-2017 ->
top 20 each way -> confirm S&P 500 2018+ at |t| > 3.2 after cost ->
replicate S&P 400 and 600 at |t| > 2, same sign.

HONESTY NOTE: S&P 500 2010-17 was already searched in dce4200, and 40
combos were already scored on 2018+. This is look #2 on that data; a pass
here is weaker evidence than a pass would have been the first time, and
the MECHANISM_MAP entry must say so. Most classic indicators overlap the
original 17 (RSI ~ recent returns, Bollinger ~ distance from average and
volatility, MACD ~ 20-day trend), so expect little new information.

Run (repo root):
    python3.10 scripts/dip_combo_classic.py
    python3.10 scripts/dip_combo_classic.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dip_features as DF       # noqa: E402
import dip_combo_search as CS   # noqa: E402

CLASSIC = ["rsi2", "rsi14", "bb_pctb", "bb_width", "macd_hist", "stoch14",
           "vs_20dma", "vs_50dma"]
_orig_per_symbol = DF.per_symbol


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def per_symbol_classic(df, spy):
    f = _orig_per_symbol(df, spy)
    df = df[~df.index.duplicated()].astype(float)
    c, h, lo = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    macd = c.ewm(span=12, adjust=False).mean() - \
        c.ewm(span=26, adjust=False).mean()
    hh, ll = h.rolling(14).max(), lo.rolling(14).min()
    f["rsi2"] = rsi(c, 2)
    f["rsi14"] = rsi(c, 14)
    f["bb_pctb"] = (c - (sma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    f["bb_width"] = 4 * sd20 / sma20
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / atr
    f["stoch14"] = (c - ll) / (hh - ll).replace(0, np.nan)
    f["vs_20dma"] = c / sma20 - 1
    f["vs_50dma"] = c / c.rolling(50).mean() - 1
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    DF.per_symbol = per_symbol_classic
    DF.FEATS = DF.FEATS + CLASSIC
    print(f"  SECOND SEARCH on S&P 500 2010-17 (first: dce4200) -- "
          f"{len(DF.FEATS)} indicators")
    if a.smoke:
        a.smoke_data = DF.smoke_data()
    CS.run(a)


if __name__ == "__main__":
    main()
