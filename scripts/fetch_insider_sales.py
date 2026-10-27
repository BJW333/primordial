"""
fetch_insider_sales.py -- insider OPEN-MARKET SALES (Form 4, code S,
disposed) by officers and directors, value >= $25,000, from the same SEC
quarterly zips fetch_insider_buys.py already cached in data/sec_form345/.
Quarters already downloaded are NOT downloaded again.

    python3.10 scripts/fetch_insider_sales.py --email you@example.com
    -> data/insider_sales.csv.gz   (filing_date = the day it became public)
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_insider_buys as FB  # noqa: E402

OUT = os.path.join(FB.ROOT, "data", "insider_sales.csv.gz")


def parse(path):
    import zipfile
    z = zipfile.ZipFile(path)
    sub = FB.table(z, "SUBMISSION")[["ACCESSION_NUMBER", "FILING_DATE",
                                     "ISSUERTRADINGSYMBOL"]]
    own = FB.table(z, "REPORTINGOWNER")
    tr = FB.table(z, "NONDERIV_TRANS")
    tr = tr[(tr["TRANS_CODE"].str.upper() == "S")
            & (tr["TRANS_ACQUIRED_DISP_CD"].str.upper() == "D")]
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("").str.upper()
    own = own[rel.str.contains("OFFICER") | rel.str.contains("DIRECTOR")]
    own = own.groupby("ACCESSION_NUMBER").agg(
        owner_cik=("RPTOWNERCIK", "first")).reset_index()
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER")
    v = pd.to_numeric(df["TRANS_SHARES"], errors="coerce") * \
        pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    df = df.assign(value=v)[v >= FB.MIN_VALUE]
    return pd.DataFrame({
        "filing_date": FB.to_date(df["FILING_DATE"]),
        "ticker": df["ISSUERTRADINGSYMBOL"].str.upper().str.strip()
                    .str.replace("-", ".", regex=False),
        "owner_cik": df["owner_cik"], "value": df["value"]}).dropna(
            subset=["filing_date", "ticker"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--start", type=int, default=2015)
    a = ap.parse_args()
    os.makedirs(FB.CACHE, exist_ok=True)
    frames = []
    for q in FB.quarters(a.start):
        p = FB.download(q, a.email)
        if not p:
            continue
        try:
            d = parse(p)
        except Exception as e:  # noqa: BLE001
            print(f"  {q}: parse failed ({e})")
            continue
        frames.append(d)
        print(f"  {q}: {len(d):>6} qualifying sales")
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df.to_csv(OUT, index=False)
    print(f"\n  {len(df)} sales, {df['ticker'].nunique()} tickers -> {OUT}")


if __name__ == "__main__":
    main()
