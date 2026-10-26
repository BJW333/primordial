"""
build_us_universe.py -- every US-listed common stock Alpaca has ever
carried, INCLUDING delisted ones, with daily prices 2016-2026. One-time
download (~30-60 min), resumable.

Steps
  1. Alpaca asset list, status active AND inactive (inactive = delisted /
     renamed / acquired -- keeping them is what removes survivorship).
     Kept: NYSE / NASDAQ / AMEX, plain symbols (no warrants, units,
     preferreds), names without ETF/fund words.
  2. Daily bars (split+dividend adjusted), 100 symbols per request,
     saved per chunk in data/us_universe/chunks/ (re-run resumes).
  3. Wide close/volume tables. A ticker with a >40-day gap is split into
     separate stocks (Alpaca recycles symbols): SYM (latest) and SYM~1,
     SYM~2 ... for earlier owners of the symbol.

Liquidity is NOT filtered here -- the scanner filters point-in-time
(price >= $5 and >= $5M/day traded, as of each month-end).

Run (repo root, keys exported):
    python3.10 scripts/build_us_universe.py
    python3.10 scripts/build_us_universe.py --limit 300     # quick trial
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIR = os.path.join(ROOT, "data", "us_universe")
CHUNKS = os.path.join(DIR, "chunks")
START, END = "2016-01-01", "2026-06-01"
TRADE = "https://paper-api.alpaca.markets/v2/assets"
DATA = "https://data.alpaca.markets/v2/stocks/bars"
EXCH = {"NYSE", "NASDAQ", "AMEX"}
SYM_OK = re.compile(r"^[A-Z]{1,5}(\.[AB])?$")
FUNDY = re.compile(r"\b(ETF|ETN|FUND|ISHARES|SPDR|PROSHARES|DIREXION|"
                   r"VANGUARD|WISDOMTREE|VANECK|INDEX|LEVERAGED|ULTRA|"
                   r"WARRANT|UNITS?|RIGHTS?|PREFERRED|NOTES?|DEPOSITARY "
                   r"SHARES REPRESENTING|ACQUISITION CORP)\b")
CHUNK = 100


def hdr():
    k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get(
        "ALPACA_SECRET_KEY")
    if not k or not s:
        sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def get(url, params, h, tries=6):
    for i in range(tries):
        r = requests.get(url, params=params, headers=h, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def assets(h):
    p = os.path.join(DIR, "assets.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    rows = []
    for status in ("active", "inactive"):
        js = get(TRADE, {"status": status, "asset_class": "us_equity"}, h)
        for a in js:
            rows.append({"symbol": a["symbol"], "name": a.get("name") or "",
                         "exchange": a.get("exchange"), "status": status})
    df = pd.DataFrame(rows)
    n0 = len(df)
    df = df[df["exchange"].isin(EXCH)
            & df["symbol"].str.match(SYM_OK)
            & ~df["name"].str.upper().str.contains(FUNDY)]
    df = df.drop_duplicates("symbol").sort_values("symbol")
    os.makedirs(DIR, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"  assets: {n0} listed -> {len(df)} common stocks kept "
          f"({(df.status == 'inactive').sum()} delisted/inactive)")
    return df


def fetch_chunk(syms, h, feed):
    out, token = [], None
    while True:
        prm = {"symbols": ",".join(syms), "timeframe": "1Day",
               "start": START, "end": END, "limit": 10000,
               "adjustment": "all", "feed": feed}
        if token:
            prm["page_token"] = token
        js = get(DATA, prm, h)
        for s, bars in (js.get("bars") or {}).items():
            for b in bars:
                out.append((s, b["t"][:10], b["c"], b["v"]))
        token = js.get("next_page_token")
        if not token:
            break
        time.sleep(0.25)
    return pd.DataFrame(out, columns=["symbol", "date", "close", "volume"])


def cut_gaps(s):
    s = s[~s.index.duplicated()].dropna()
    if s.empty:
        return []
    br = np.flatnonzero(np.diff(s.index.values).astype("timedelta64[D]")
                        > np.timedelta64(40, "D"))
    cuts = [0] + list(br + 1) + [len(s)]
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 20]


def build_wide():
    files = sorted(f for f in os.listdir(CHUNKS) if f.endswith(".parquet"))
    long = pd.concat([pd.read_parquet(os.path.join(CHUNKS, f))
                      for f in files], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    close = long.pivot_table("close", "date", "symbol", aggfunc="last")
    vol = long.pivot_table("volume", "date", "symbol", aggfunc="last")
    spy = close.pop("SPY") if "SPY" in close else None
    vol = vol.drop(columns=["SPY"], errors="ignore")
    C, V = {}, {}
    for s in close.columns:
        c = close[s].dropna()
        segs = cut_gaps(c)
        for k, (a, b) in enumerate(reversed(segs)):
            name = s if k == 0 else f"{s}~{k}"
            seg = c.iloc[a:b]
            C[name] = seg
            V[name] = vol[s].reindex(seg.index)
    close = pd.DataFrame(C).sort_index()
    vol = pd.DataFrame(V).reindex(close.index)
    close.to_parquet(os.path.join(DIR, "close.parquet"))
    vol.to_parquet(os.path.join(DIR, "volume.parquet"))
    if spy is not None:
        spy.dropna().to_frame("SPY").to_parquet(os.path.join(DIR,
                                                             "spy.parquet"))
    print(f"  wide tables: {close.shape[1]} stock histories, "
          f"{close.shape[0]} days -> {DIR}")
    yr = close.notna().groupby(close.index.year).any().sum(axis=1)
    print("  stocks with prices per year: " +
          "  ".join(f"{y}:{n}" for y, n in yr.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--feed", default="sip")
    a = ap.parse_args()
    h = hdr()
    os.makedirs(CHUNKS, exist_ok=True)
    df = assets(h)
    syms = list(df["symbol"])
    if a.limit:
        syms = syms[:a.limit]
    syms = ["SPY"] + [s for s in syms if s != "SPY"]
    groups = [syms[i:i + CHUNK] for i in range(0, len(syms), CHUNK)]
    feed = a.feed
    t0 = time.time()
    for i, g in enumerate(groups):
        p = os.path.join(CHUNKS, f"c{i:04d}_{g[0]}.parquet")
        if os.path.exists(p):
            continue
        try:
            d = fetch_chunk(g, h, feed)
        except requests.HTTPError as e:
            if feed == "sip" and e.response is not None and \
                    e.response.status_code == 403:
                print("  SIP feed not allowed on this key -> using IEX")
                feed = "iex"
                d = fetch_chunk(g, h, feed)
            else:
                raise
        d.to_parquet(p)
        el = time.time() - t0
        print(f"  chunk {i+1}/{len(groups)}  {d.symbol.nunique()} symbols "
              f"with data  ({el/60:.1f} min)", flush=True)
    with open(os.path.join(DIR, "feed.txt"), "w") as f:
        f.write(feed)
    build_wide()


if __name__ == "__main__":
    main()
