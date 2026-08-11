"""
Fetch Alpaca minute bars for a small pilot universe and cache to parquet.

    source ~/.alpaca_env
    python3.10 scripts/fetch_intraday.py

Pilot-sized on purpose: 15 midcaps x 10y is ~15M rows, enough to see whether
the opening-gap reversal exists before committing to a 50-name fetch.
SIP feed (full consolidated tape) -- IEX-only would distort opening prints.
"""
import argparse
import os
import sys
import time

import pandas as pd

PILOT = ["TCBI", "KEX", "WWD", "CVLT", "ETR", "WMB", "MU", "MGM", "GEV",
         "AEIS", "SANM", "ARW", "AVT", "PLAB", "COHU"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs=2, type=int, default=[2016, 2026])
    p.add_argument("--out", default=os.path.expanduser("~/.primordial_intraday"))
    p.add_argument("--symbols", nargs="*", default=PILOT)
    a = p.parse_args()

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    key = os.environ.get("APCA_API_KEY_ID")
    sec = os.environ.get("APCA_API_SECRET_KEY")
    if not key:
        sys.exit("no APCA_API_KEY_ID -- source ~/.alpaca_env first")
    cli = StockHistoricalDataClient(key, sec)
    os.makedirs(a.out, exist_ok=True)

    y0, y1 = a.years
    for i, sym in enumerate(a.symbols, 1):
        path = os.path.join(a.out, f"{sym}_1m.parquet")
        if os.path.exists(path):
            n = len(pd.read_parquet(path))
            print(f"  [{i:2d}/{len(a.symbols)}] {sym:6s} cached {n:>9,} bars")
            continue
        frames = []
        for yr in range(y0, y1 + 1):
            for attempt in range(3):
                try:
                    r = cli.get_stock_bars(StockBarsRequest(
                        symbol_or_symbols=[sym], timeframe=TimeFrame.Minute,
                        start=pd.Timestamp(f"{yr}-01-01", tz="UTC"),
                        end=pd.Timestamp(f"{yr}-12-31", tz="UTC"),
                        feed=DataFeed.SIP))
                    df = r.df
                    if len(df):
                        frames.append(df.reset_index())
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"      {sym} {yr}: FAILED {str(e)[:60]}")
                    time.sleep(2 * (attempt + 1))
        if not frames:
            print(f"  [{i:2d}/{len(a.symbols)}] {sym:6s} NO DATA")
            continue
        out = pd.concat(frames, ignore_index=True)
        out.to_parquet(path)
        print(f"  [{i:2d}/{len(a.symbols)}] {sym:6s} {len(out):>9,} bars -> "
              f"{os.path.basename(path)}")
    print(f"\ndone -> {a.out}")


if __name__ == "__main__":
    main()
