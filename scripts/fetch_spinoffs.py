"""
fetch_spinoffs.py -- list US spinoffs since 2006 from SEC EDGAR, free.

A spun-off company registers its shares on Form 10 (form type "10-12B")
before it starts trading. This script:
  1. downloads EDGAR's quarterly form index (full-index/<year>/QTR<n>/
     form.idx) and keeps every 10-12B / 10-12B/A filing -> first filing
     date per registrant CIK
  2. maps each CIK to its trading symbol using the insider-filing data
     already cached by fetch_insider_buys.py (data/sec_form345/*.zip:
     every listed company's Form 4s carry ISSUERCIK + ISSUERTRADINGSYMBOL)
  3. writes data/spinoffs.csv

Not every 10-12B is a spinoff (some are re-registrations or holding-company
reorganisations); spinoff_test.py keeps only names whose price history
STARTS within ~13 months after the filing, i.e. genuinely new listings.

    python3.10 scripts/fetch_spinoffs.py --email you@example.com
"""
import argparse
import glob
import io
import os
import re
import time
import zipfile

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX_CACHE = os.path.join(ROOT, "data", "edgar_form_idx")
F345 = os.path.join(ROOT, "data", "sec_form345")
OUT = os.path.join(ROOT, "data", "spinoffs.csv")
URL = "https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"


def quarters(start=2006):
    now = pd.Timestamp.today()
    for y in range(start, now.year + 1):
        for q in range(1, 5):
            if pd.Timestamp(y, 3 * q - 2, 1) <= now:
                yield y, q


def get_index(y, q, email):
    path = os.path.join(IDX_CACHE, f"{y}q{q}_form.idx")
    if not os.path.exists(path):
        r = requests.get(URL.format(y=y, q=q), timeout=120,
                         headers={"User-Agent": f"primordial research "
                                                f"{email}"})
        time.sleep(0.3)
        if r.status_code != 200:
            return None
        open(path, "wb").write(r.content)
    return open(path, "rb").read().decode("latin-1")


def parse_index(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("10-12B"):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5:
            continue
        form, name, cik, date = parts[0], parts[1], parts[2], parts[3]
        rows.append((form, name, cik.lstrip("0"), date))
    return rows


def cik_tickers():
    """CIK -> list of (filing_date, ticker) from cached Form 4 data."""
    frames = []
    for z in sorted(glob.glob(os.path.join(F345, "*_form345.zip"))):
        zf = zipfile.ZipFile(z)
        name = [n for n in zf.namelist()
                if os.path.basename(n).upper().startswith("SUBMISSION")][0]
        s = pd.read_csv(io.BytesIO(zf.read(name)), sep="\t", dtype=str,
                        usecols=["FILING_DATE", "ISSUERCIK",
                                 "ISSUERTRADINGSYMBOL"],
                        on_bad_lines="skip")
        frames.append(s.dropna())
    if not frames:
        raise SystemExit("no cached Form 4 zips in data/sec_form345 -- run "
                         "fetch_insider_buys.py first")
    s = pd.concat(frames, ignore_index=True)
    s["cik"] = s["ISSUERCIK"].str.lstrip("0")
    s["date"] = pd.to_datetime(s["FILING_DATE"], format="%d-%b-%Y",
                               errors="coerce")
    s["ticker"] = (s["ISSUERTRADINGSYMBOL"].str.upper().str.strip()
                   .str.replace(".", "-", regex=False))
    s = s[s["ticker"].str.fullmatch(r"[A-Z0-9\-]{1,7}", na=False)]
    return s[["cik", "date", "ticker"]].dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    a = ap.parse_args()
    os.makedirs(IDX_CACHE, exist_ok=True)
    rows, missing = [], []
    for y, q in quarters():
        t = get_index(y, q, a.email)
        if t is None:
            missing.append(f"{y}q{q}")
            continue
        got = parse_index(t)
        rows += got
        print(f"  {y}q{q}: {len(got):>3} Form 10-12B filings")
    f = pd.DataFrame(rows, columns=["form", "name", "cik", "date"])
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    first = (f.sort_values("date").groupby("cik")
             .agg(name=("name", "first"), first_10_12b=("date", "first"))
             .reset_index())
    print(f"\n  {len(first)} registrants filed a 10-12B")

    m = cik_tickers()
    m = m.merge(first[["cik", "first_10_12b"]], on="cik")
    m = m[(m["date"] >= m["first_10_12b"])
          & (m["date"] <= m["first_10_12b"] + pd.Timedelta(days=730))]
    tk = (m.groupby(["cik", "ticker"]).size().reset_index(name="n")
          .sort_values("n").groupby("cik").tail(1)[["cik", "ticker"]])
    out = first.merge(tk, on="cik", how="left")
    out.to_csv(OUT, index=False)
    print(f"  {out['ticker'].notna().sum()} matched to a trading symbol "
          f"-> {OUT}")
    if missing:
        print(f"  index quarters unavailable: {', '.join(missing)}")


if __name__ == "__main__":
    main()
