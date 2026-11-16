"""
fetch_commod_rth.py -- builds REGULAR-HOURS 1h bars for the commodities_1h
manifest from Alpaca SIP 1-minute bars, split/dividend adjusted.

Why not `source: alpaca` directly: Primordial's _alpaca() hourly bars include
pre/post-market, which is near-empty for CORN/WEAT/SOYB/CPER etc. -- thin
prints there would read as fake moves. Here: 9:30-16:00 ET only, hourly bars
starting 9:30, 10:30 ... 15:30 (the last is the 30-minute 15:30-16:00 bar),
timestamps written as tz-naive UTC like the rest of Primordial.
adjustment=all: USO (1:8, 2020) and UNG reverse splits would otherwise look
like +700% bars.

Writes data/commod_rth_1h/<SYM>.csv  (index time, open, high, low, close, volume)
Run (repo root):  python3.10 scripts/fetch_commod_rth.py
Needs ALPACA_API_KEY / ALPACA_SECRET_KEY in the environment. One-time, a few
minutes per symbol; re-running skips symbols already written (--force redoes).
"""
import argparse
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "commod_rth_1h")
SYMBOLS = ["USO", "BNO", "UNG", "UGA", "GLD", "SLV", "PPLT", "PALL", "CPER", "DBB",
           "CORN", "WEAT", "SOYB", "CANE", "DBA", "DBC"]
CONTEXT = ["UUP", "TLT", "SPY"]
START, END = "2016-01-01", "2026-11-01"


def minute_bars(sym, key, sec):
    import requests
    url = f"https://data.alpaca.markets/v2/stocks/{sym}/bars"
    hd = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    frames = []
    for y in range(int(START[:4]), int(END[:4]) + 1):
        a = max(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(START))
        b = min(pd.Timestamp(f"{y + 1}-01-01"), pd.Timestamp(END))
        if a >= b:
            continue
        prm = {"timeframe": "1Min", "start": a.tz_localize("UTC").isoformat(),
               "end": b.tz_localize("UTC").isoformat(), "limit": 10000,
               "adjustment": "all", "feed": "sip"}
        rows, tok = [], None
        while True:
            if tok:
                prm["page_token"] = tok
            for k in range(6):
                r = requests.get(url, headers=hd, params=prm, timeout=60)
                if r.status_code == 429:
                    time.sleep(3 + 3 * k)
                    continue
                break
            r.raise_for_status()
            j = r.json()
            rows += j.get("bars") or []
            tok = j.get("next_page_token")
            if not tok:
                break
        if rows:
            frames.append(pd.DataFrame(rows))
        print(f"    {sym} {y}: {len(rows):,} minute bars")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df.index = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                              "v": "volume"})[["open", "high", "low", "close", "volume"]]


def to_rth_1h(m):
    """ET-aware minute bars -> RTH hourly bars (9:30, 10:30 ... 15:30), UTC-naive."""
    m = m[~m.index.duplicated(keep="last")].sort_index()
    mod = m.index.hour * 60 + m.index.minute
    m = m[(mod >= 570) & (mod < 960)]
    mod = m.index.hour * 60 + m.index.minute
    start = m.index.normalize() + pd.to_timedelta(570 + (mod - 570) // 60 * 60, unit="m")
    g = m.groupby(start)
    h = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                      "low": g["low"].min(), "close": g["close"].last(),
                      "volume": g["volume"].sum()}).dropna(subset=["open", "close"])
    h.index = pd.DatetimeIndex(h.index).tz_convert("UTC").tz_localize(None)
    h.index.name = "time"
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("  export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
    os.makedirs(OUT, exist_ok=True)
    for sym in SYMBOLS + CONTEXT:
        p = os.path.join(OUT, f"{sym}.csv")
        if os.path.exists(p) and not a.force:
            print(f"  {sym}: exists, skipped")
            continue
        m = minute_bars(sym, key, sec)
        if m is None or m.empty:
            print(f"  {sym}: NO DATA")
            continue
        h = to_rth_1h(m)
        h.to_csv(p)
        days = h.index.normalize().nunique()
        print(f"  {sym}: {len(h):,} hourly bars, {days} days "
              f"({h.index[0].date()} -> {h.index[-1].date()}), {len(h) / days:.1f}/day")
    print(f"  done -> {OUT}")


if __name__ == "__main__":
    main()
