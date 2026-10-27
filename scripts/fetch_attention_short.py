"""
fetch_attention_short.py -- two non-price datasets for the S&P 500 stocks
in the factor_portfolio universe (point-in-time members incl. delisted).

ATTENTION: daily English-Wikipedia page views per company, 2015-07 -> now.
  Ticker -> Wikipedia article via Wikidata (ticker symbols listed on NYSE /
  NASDAQ). Page views via the Wikimedia REST API (no key; User-Agent with
  your email required).
  -> data/attention/views.parquet (date x ticker), mapping.csv

SHORT INTEREST: FINRA consolidated short interest (exchange-listed + OTC),
  twice monthly, via FINRA's public Query API (no key). Saved for S&P names.
  If the API refuses or has little history, falls back to FINRA's free
  daily short-sale VOLUME files (cdn.finra.org, Reg SHO) -- a different
  but related measure (shorted share of daily volume).
  -> data/short/short_interest.parquet  and/or  data/short/short_volume.parquet

Run (repo root; ~20-40 min first time, resumable):
    python3.10 scripts/fetch_attention_short.py --email you@example.com
"""
import argparse
import io
import json
import os
import sys
import time
import urllib.parse

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import factor_portfolio as FP  # noqa: E402

ROOT = os.path.dirname(HERE)
ATT = os.path.join(ROOT, "data", "attention")
SHO = os.path.join(ROOT, "data", "short")
WD = "https://query.wikidata.org/sparql"
PV = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
      "en.wikipedia/all-access/user/{a}/daily/20150701/{end}")
FINRA = ("https://api.finra.org/data/group/otcMarket/name/"
         "consolidatedShortInterest")
REGSHO = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"


def sess(email):
    s = requests.Session()
    s.headers["User-Agent"] = f"primordial-research/1.0 ({email})"
    return s


# -------------------------------------------------------------- attention
def wikidata_map(s):
    q = """
    SELECT ?ticker ?article WHERE {
      ?item p:P414 ?st . ?st ps:P414 ?ex . ?st pq:P249 ?ticker .
      VALUES ?ex { wd:Q13677 wd:Q82059 }
      ?article schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> .
    }"""
    r = s.get(WD, params={"query": q, "format": "json"}, timeout=180)
    r.raise_for_status()
    rows = [(b["ticker"]["value"].upper().strip(),
             urllib.parse.unquote(b["article"]["value"].rsplit("/", 1)[-1]))
            for b in r.json()["results"]["bindings"]]
    m = pd.DataFrame(rows, columns=["ticker", "article"]).drop_duplicates()
    m["ticker"] = m["ticker"].str.replace("-", ".", regex=False)
    return m


def fetch_attention(s, tickers):
    os.makedirs(os.path.join(ATT, "views"), exist_ok=True)
    mp = os.path.join(ATT, "mapping.csv")
    if os.path.exists(mp):
        m = pd.read_csv(mp)
    else:
        m = wikidata_map(s)
        m.to_csv(mp, index=False)
    m = m[m["ticker"].isin(tickers)]
    # one article per ticker: if several, keep the one with most views later
    print(f"  Wikipedia articles found for {m['ticker'].nunique()} of "
          f"{len(tickers)} S&P tickers")
    end = pd.Timestamp.today().strftime("%Y%m%d")
    series = {}
    for i, (tk, g) in enumerate(m.groupby("ticker")):
        p = os.path.join(ATT, "views", tk.replace("/", "_") + ".csv")
        if not os.path.exists(p):
            best = None
            for art in g["article"]:
                r = s.get(PV.format(a=urllib.parse.quote(
                    art.replace(" ", "_"), safe=""), end=end), timeout=60)
                if r.status_code != 200:
                    continue
                it = r.json().get("items", [])
                v = pd.Series({pd.Timestamp(x["timestamp"][:8]): x["views"]
                               for x in it}, dtype=float)
                if best is None or v.sum() > best.sum():
                    best = v
                time.sleep(0.05)
            (best if best is not None else pd.Series(dtype=float)) \
                .rename("views").to_csv(p)
        try:
            v = pd.read_csv(p, index_col=0, parse_dates=True)["views"]
            if len(v):
                series[tk] = v
        except (pd.errors.EmptyDataError, KeyError):
            pass
        if i % 100 == 0:
            print(f"    page views {i}/{m['ticker'].nunique()}", flush=True)
    df = pd.DataFrame(series).sort_index()
    df.to_parquet(os.path.join(ATT, "views.parquet"))
    print(f"  page views: {df.shape[1]} companies, {df.index[0]:%Y-%m-%d} -> "
          f"{df.index[-1]:%Y-%m-%d}")


# ---------------------------------------------------------- short interest
def finra_window(s, start, end):
    out, off = [], 0
    while True:
        body = {"limit": 5000, "offset": off,
                "dateRangeFilters": [{"fieldName": "settlementDate",
                                      "startDate": start, "endDate": end}]}
        r = s.post(FINRA, data=json.dumps(body), timeout=120,
                   headers={"Accept": "application/json",
                            "Content-Type": "application/json"})
        if r.status_code != 200:
            raise RuntimeError(f"FINRA HTTP {r.status_code}: {r.text[:200]}")
        rows = r.json() if r.text.strip() else []
        out += rows
        if len(rows) < 5000:
            return out
        off += 5000
        time.sleep(0.2)


def fetch_short_interest(s, tickers):
    os.makedirs(os.path.join(SHO, "si"), exist_ok=True)
    frames = []
    for m in pd.period_range("2016-01", pd.Timestamp.today().to_period("M"),
                             freq="M"):
        p = os.path.join(SHO, "si", f"{m}.parquet")
        if not os.path.exists(p):
            rows = finra_window(s, f"{m.start_time:%Y-%m-%d}",
                                f"{m.end_time:%Y-%m-%d}")
            d = pd.DataFrame(rows)
            if len(d):
                cols = {c.lower(): c for c in d.columns}
                pick = lambda *ks: next(cols[c] for c in cols  # noqa: E731
                                        if all(k in c for k in ks))
                d = pd.DataFrame({
                    "ticker": d[pick("symbol")].astype(str).str.upper(),
                    "date": pd.to_datetime(d[pick("settlement")]),
                    "short": pd.to_numeric(d[pick("currentshort")],
                                           errors="coerce"),
                    "adv": pd.to_numeric(d[pick("averagedaily")],
                                         errors="coerce")})
                d["ticker"] = d["ticker"].str.replace("-", ".", regex=False)
                d = d[d["ticker"].isin(tickers)]
            d.to_parquet(p)
        d = pd.read_parquet(p)
        if len(d):
            frames.append(d)
            print(f"    short interest {m}: {len(d)} S&P rows", flush=True)
    si = pd.concat(frames, ignore_index=True)
    si.to_parquet(os.path.join(SHO, "short_interest.parquet"))
    print(f"  short interest: {si['ticker'].nunique()} stocks, "
          f"{si['date'].min():%Y-%m-%d} -> {si['date'].max():%Y-%m-%d}")
    return si


def fetch_short_volume(s, tickers):
    os.makedirs(os.path.join(SHO, "sv"), exist_ok=True)
    days = pd.bdate_range("2017-01-01", pd.Timestamp.today())
    frames, got = [], 0
    for d in days:
        p = os.path.join(SHO, "sv", f"{d:%Y%m%d}.parquet")
        if not os.path.exists(p):
            r = s.get(REGSHO.format(d=f"{d:%Y%m%d}"), timeout=60)
            if r.status_code != 200:
                pd.DataFrame().to_parquet(p)
                continue
            x = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str)
            x = x[x["Symbol"].isin(tickers)]
            x = pd.DataFrame({"ticker": x["Symbol"], "date": d,
                              "short": pd.to_numeric(x["ShortVolume"],
                                                     errors="coerce"),
                              "total": pd.to_numeric(x["TotalVolume"],
                                                     errors="coerce")})
            x.to_parquet(p)
        x = pd.read_parquet(p)
        if len(x):
            frames.append(x)
            got += 1
        if d.day == 1 or d == days[-1]:
            print(f"    short volume through {d:%Y-%m}: {got} days", flush=True)
    sv = pd.concat(frames, ignore_index=True)
    sv.to_parquet(os.path.join(SHO, "short_volume.parquet"))
    print(f"  short volume: {sv['ticker'].nunique()} stocks, "
          f"{sv['date'].min():%Y-%m-%d} -> {sv['date'].max():%Y-%m-%d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--only", choices=["attention", "short"])
    a = ap.parse_args()
    s = sess(a.email)
    _, union = FP.membership()
    tickers = sorted({t.upper() for t in union})
    if a.only != "short":
        fetch_attention(s, tickers)
    if a.only != "attention":
        try:
            si = fetch_short_interest(s, tickers)
            years = (si["date"].max() - si["date"].min()).days / 365
            if years < 6:
                print(f"  short interest history only {years:.1f} years -> "
                      "also fetching daily short volume as a fallback")
                fetch_short_volume(s, tickers)
        except Exception as e:  # noqa: BLE001
            print(f"  FINRA short interest unavailable ({e}) -> using daily "
                  "short volume instead")
            fetch_short_volume(s, tickers)


if __name__ == "__main__":
    main()
