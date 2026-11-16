"""
fetch_diversifiers.py -- download daily prices (dividend- and split-adjusted)
for ~40 diversifier candidates from Alpaca, 2016-12 -> 2026-06, and save
data/diversifier_probe.csv (one column per ticker). Crypto (BTC, ETH) is
added if Alpaca's crypto history reaches back far enough.

Needs ALPACA_API_KEY and ALPACA_SECRET_KEY in the environment.
Run (repo root):
    python3.10 scripts/fetch_diversifiers.py
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

TICKERS = ["GLD", "SLV", "TLT", "IEF", "SHY", "TIP", "DBMF", "KMLM", "WTMF", "CTA",
           "BTAL", "MNA", "QAI", "RPAR", "USMV", "SPLV", "XLU", "XLP", "XLV", "VTV",
           "RPV", "QUAL", "DBC", "PDBC", "UUP", "FXY", "FXF", "TAIL", "SH", "SVXY",
           "BIL", "SPY", "EFA", "EEM", "VNQ", "MTUM"]
CRYPTO = ["BTC/USD", "ETH/USD"]
START, END = "2016-12-01T00:00:00Z", "2026-06-02T00:00:00Z"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "diversifier_probe.csv")


def get(url, params):
    key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("export ALPACA_API_KEY and ALPACA_SECRET_KEY first")
    req = urllib.request.Request(url + "?" + urllib.parse.urlencode(params),
                                 headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("Alpaca request kept failing")


def bars(url, symbols, extra):
    out, token = {}, None
    while True:
        p = dict(symbols=",".join(symbols), timeframe="1Day", start=START, end=END,
                 limit=10000, **extra)
        if token:
            p["page_token"] = token
        j = get(url, p)
        for s, rows in (j.get("bars") or {}).items():
            out.setdefault(s, []).extend(rows)
        token = j.get("next_page_token")
        if not token:
            return out


def to_frame(raw):
    cols = {}
    for s, rows in raw.items():
        if rows:
            ix = pd.to_datetime([r["t"] for r in rows]).tz_convert(None).normalize()
            cols[s.replace("/USD", "USD")] = pd.Series([r["c"] for r in rows], index=ix)
    return pd.DataFrame(cols)


def main():
    frames = []
    for feed in ("sip", "iex"):
        try:
            raw = {}
            for i in range(0, len(TICKERS), 10):
                raw.update(bars("https://data.alpaca.markets/v2/stocks/bars",
                                TICKERS[i:i + 10], {"adjustment": "all", "feed": feed}))
            frames.append(to_frame(raw))
            print(f"  stocks/ETFs: {len(raw)} tickers from feed={feed}")
            break
        except Exception as e:
            print(f"  feed={feed} failed ({e}); trying next")
    try:
        raw = bars("https://data.alpaca.markets/v1beta3/crypto/us/bars", CRYPTO, {})
        frames.append(to_frame(raw))
        print(f"  crypto: {list(raw)}")
    except Exception as e:
        print(f"  crypto skipped ({e})")
    px = pd.concat(frames, axis=1).sort_index()
    px = px.groupby(level=0).last()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    px.to_csv(OUT)
    first = px.apply(lambda c: c.first_valid_index())
    print(f"  saved {OUT}: {px.shape[1]} columns, {px.index[0].date()} -> {px.index[-1].date()}")
    print("  first date per ticker: " + ", ".join(f"{k} {v.date()}" for k, v in first.items()
                                                  if v is not None))
    missing = [t for t in TICKERS if t not in px.columns]
    if missing:
        print(f"  missing: {missing}")


if __name__ == "__main__":
    main()
