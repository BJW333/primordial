"""
fetch_insider_buys.py -- build a clean list of insider OPEN-MARKET PURCHASES
(Form 4, transaction code P) by officers and directors, 2006 -> now, from
the SEC's free quarterly "Insider Transactions Data Sets".

    python3.10 scripts/fetch_insider_buys.py --email you@example.com
    -> data/sec_form345/<yyyy>q<q>_form345.zip  (cached; re-runs skip them)
    -> data/insider_buys.csv.gz

SEC requires a User-Agent with a contact email and caps traffic at 10
requests/second; this script sends one request at a time with a pause.

Kept: TRANS_CODE == 'P', acquired ('A'), reporter is an officer or director
(10%-owner-only funds dropped), value >= $25,000 (token buys dropped).
Each row keeps FILING_DATE -- the day the buy became public, which is the
only date a strategy may act on.
"""
import argparse
import io
import os
import sys
import time
import zipfile

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "sec_form345")
OUT = os.path.join(ROOT, "data", "insider_buys.csv.gz")
FOLDERS = ["structureddata", "datastandardsinnovation"]
URL = "https://www.sec.gov/files/{f}/data/insider-transactions-data-sets/{q}_form345.zip"
MIN_VALUE = 25_000


def quarters(start=2006):
    now = pd.Timestamp.today()
    for y in range(start, now.year + 1):
        for q in range(1, 5):
            if pd.Timestamp(y, 3 * q - 2, 1) <= now:
                yield f"{y}q{q}"


def download(q, email):
    path = os.path.join(CACHE, f"{q}_form345.zip")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    hdr = {"User-Agent": f"primordial research {email}"}
    for f in FOLDERS:
        r = requests.get(URL.format(f=f, q=q), headers=hdr, timeout=120)
        time.sleep(0.3)
        if r.status_code == 200 and r.content[:2] == b"PK":
            open(path, "wb").write(r.content)
            return path
    return None


def table(z, name):
    hit = [n for n in z.namelist()
           if os.path.basename(n).upper().startswith(name)]
    if not hit:
        raise KeyError(f"{name} not in zip ({z.namelist()[:6]})")
    return pd.read_csv(io.BytesIO(z.read(hit[0])), sep="\t", dtype=str,
                       on_bad_lines="skip", low_memory=False)


def to_date(s):
    return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce").fillna(
        pd.to_datetime(s, errors="coerce"))


def parse(path):
    z = zipfile.ZipFile(path)
    sub = table(z, "SUBMISSION")[["ACCESSION_NUMBER", "FILING_DATE",
                                  "ISSUERCIK", "ISSUERTRADINGSYMBOL"]]
    own = table(z, "REPORTINGOWNER")
    tr = table(z, "NONDERIV_TRANS")
    tr = tr[(tr["TRANS_CODE"].str.upper() == "P")
            & (tr["TRANS_ACQUIRED_DISP_CD"].str.upper() == "A")]
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("").str.upper()
    own = own[rel.str.contains("OFFICER") | rel.str.contains("DIRECTOR")]
    own = own.groupby("ACCESSION_NUMBER").agg(
        owner_cik=("RPTOWNERCIK", "first"),
        relationship=("RPTOWNER_RELATIONSHIP", "first"),
        title=("RPTOWNER_TITLE", "first")).reset_index()
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own,
                                                    on="ACCESSION_NUMBER")
    df["shares"] = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    df["price"] = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    df["value"] = df["shares"] * df["price"]
    df = df[df["value"] >= MIN_VALUE]
    out = pd.DataFrame({
        "filing_date": to_date(df["FILING_DATE"]),
        "trans_date": to_date(df["TRANS_DATE"]),
        "ticker": df["ISSUERTRADINGSYMBOL"].str.upper().str.strip()
                    .str.replace(".", "-", regex=False),
        "issuer_cik": df["ISSUERCIK"], "owner_cik": df["owner_cik"],
        "relationship": df["relationship"], "title": df["title"],
        "shares": df["shares"], "price": df["price"], "value": df["value"]})
    return out.dropna(subset=["filing_date", "ticker"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True,
                    help="contact email for the SEC User-Agent header")
    ap.add_argument("--start", type=int, default=2006)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    frames, missing = [], []
    for q in quarters(a.start):
        p = download(q, a.email)
        if not p:
            missing.append(q)
            continue
        try:
            d = parse(p)
        except Exception as e:  # noqa: BLE001
            print(f"  {q}: parse failed ({e})")
            missing.append(q)
            continue
        frames.append(d)
        print(f"  {q}: {len(d):>6} qualifying purchases")
    if not frames:
        sys.exit("nothing downloaded -- check the email/User-Agent and "
                 "network")
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df.to_csv(OUT, index=False)
    print(f"\n  {len(df)} purchases, {df['ticker'].nunique()} tickers, "
          f"{df['filing_date'].min().date()} -> "
          f"{df['filing_date'].max().date()} -> {OUT}")
    if missing:
        print(f"  quarters not available yet / failed: {', '.join(missing)}")
    lag = (df["filing_date"] - df["trans_date"]).dt.days
    print(f"  filing lag (trade -> public): median {lag.median():.0f} days, "
          f"90th pct {lag.quantile(0.9):.0f} days")


if __name__ == "__main__":
    main()
