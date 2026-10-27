"""
fetch_rebound_data.py -- data for rebound2.py (both free, resumable):

1. FINRA Reg SHO daily short-sale volume for EVERY symbol (the earlier
   download kept only S&P 500 names). 2018-08 -> now, one file a day,
   cached in data/short/sv_all/. -> data/short/short_volume_all.parquet
   (columns: date, ticker, short, total)
2. Activist stakes: every NEW Schedule 13D (5%+ owner who may push for
   change; amendments dropped) from the SEC's quarterly form.idx (cached
   in data/edgar_form_idx/, shared with fetch_spinoffs.py), CIK -> ticker
   by year. -> data/activist_13d.csv (ticker, filed)

Run (repo root, ~20-40 min first time):
    python3.10 scripts/fetch_rebound_data.py --email you@example.com
"""
import argparse
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_sec_fundamentals as SF  # noqa: E402
import fetch_spinoffs as SP          # noqa: E402

ROOT = os.path.dirname(HERE)
SV = os.path.join(ROOT, "data", "short", "sv_all")
SV_OUT = os.path.join(ROOT, "data", "short", "short_volume_all.parquet")
AD_OUT = os.path.join(ROOT, "data", "activist_13d.csv")
REGSHO = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
FORMS = re.compile(r"^(SC 13D|SCHEDULE 13D)\s{2,}")


def one_day(s, d):
    p = os.path.join(SV, f"{d:%Y%m%d}.parquet")
    if os.path.exists(p):
        return
    for i in range(4):
        try:
            r = s.get(REGSHO.format(d=f"{d:%Y%m%d}"), timeout=60)
            break
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    else:
        return
    if r.status_code != 200:
        pd.DataFrame(columns=["ticker", "short", "total"]).to_parquet(p)
        return
    x = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str)
    x = x[x["Symbol"].notna() & x["TotalVolume"].notna()]
    pd.DataFrame({"ticker": x["Symbol"].str.upper(),
                  "short": pd.to_numeric(x["ShortVolume"], errors="coerce"),
                  "total": pd.to_numeric(x["TotalVolume"], errors="coerce")}
                 ).dropna().to_parquet(p)


def short_volume(email, tickers):
    os.makedirs(SV, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = f"primordial-research/1.0 ({email})"
    days = pd.bdate_range("2018-08-01", pd.Timestamp.today())
    with ThreadPoolExecutor(6) as ex:
        for i, _ in enumerate(ex.map(lambda d: one_day(s, d), days)):
            if i % 250 == 0:
                print(f"    short volume {i}/{len(days)} days", flush=True)
    frames = []
    for d in days:
        p = os.path.join(SV, f"{d:%Y%m%d}.parquet")
        if not os.path.exists(p):
            continue
        x = pd.read_parquet(p)
        if len(x):
            x = x[x["ticker"].isin(tickers)]
            frames.append(x.assign(date=d))
    sv = pd.concat(frames, ignore_index=True)
    sv.to_parquet(SV_OUT)
    print(f"  short volume: {sv['ticker'].nunique()} stocks, "
          f"{sv['date'].nunique()} days -> {SV_OUT}")


def activist(email):
    os.makedirs(SP.IDX_CACHE, exist_ok=True)
    rows = []
    for y, q in SP.quarters(2015):
        txt = SP.get_index(y, q, email)
        if txt is None:
            continue
        for line in txt.splitlines():
            if not FORMS.match(line):
                continue
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 5 or parts[0].endswith("/A"):
                continue
            rows.append((parts[2].lstrip("0"), parts[3]))
    df = pd.DataFrame(rows, columns=["cik", "filed"]).drop_duplicates()
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["year"] = df["filed"].dt.year
    ct = SF.cik_tickers()
    ct["cik"] = ct["cik"].astype(str).str.lstrip("0")
    ct["year"] = pd.to_numeric(ct["year"], errors="coerce")
    m = df.merge(ct, on=["cik", "year"], how="inner")
    m["ticker"] = m["ticker"].str.upper().str.replace("-", ".", regex=False)
    m = m[["ticker", "filed"]].drop_duplicates().sort_values("filed")
    m.to_csv(AD_OUT, index=False)
    print(f"  activist 13D: {len(df)} new filings -> {len(m)} matched to "
          f"{m['ticker'].nunique()} traded companies -> {AD_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    a = ap.parse_args()
    close = pd.read_parquet(os.path.join(ROOT, "data", "us_universe",
                                         "close.parquet"))
    tickers = {c for c in close.columns if "~" not in c}
    activist(a.email)
    short_volume(a.email, tickers)


if __name__ == "__main__":
    main()
