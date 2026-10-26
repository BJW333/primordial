"""
fetch_earnings.py -- every US company's quarterly net income, stamped with
the day it was FIRST filed with the SEC (point-in-time), 2014 -> today.

Source: SEC companyfacts.zip (every XBRL fact every company has filed;
~1.3 GB, one download). For each company and quarter we keep the value
from the FIRST 10-Q/10-K that reported it -- later restatements are
ignored, so nothing from the future leaks in. Q4 = annual (10-K) minus
Q1+Q2+Q3. Net income, not EPS, so stock splits can't fake a surprise.

Tickers: CIK -> ticker by year from the Form 4 cache (data/sec_form345/,
includes delisted companies), via fetch_sec_fundamentals.cik_tickers().

Output: data/earnings_quarterly.csv.gz
  cik, ticker, end (quarter end), filed (first filing date), ni

Run (repo root):
    python3.10 scripts/fetch_earnings.py --email you@example.com
    python3.10 scripts/fetch_earnings.py --email ... --delete-zip
"""
import argparse
import io
import json
import os
import sys
import zipfile

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
ZIP = os.path.join(ROOT, "data", "sec", "companyfacts.zip")
OUT = os.path.join(ROOT, "data", "earnings_quarterly.csv.gz")
TAGS = ["NetIncomeLoss", "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic"]
FORMS = {"10-Q", "10-K"}
MIN_END = pd.Timestamp("2013-01-01")


def download(email):
    if os.path.exists(ZIP) and os.path.getsize(ZIP) > 5e8:
        print(f"  using cached {ZIP}")
        return
    os.makedirs(os.path.dirname(ZIP), exist_ok=True)
    hdr = {"User-Agent": f"primordial research {email}"}
    tmp = ZIP + ".part"
    with requests.get(URL, headers=hdr, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got, step = 0, 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if got >= step:
                    print(f"  downloaded {got/1e6:,.0f} / {total/1e6:,.0f} MB",
                          flush=True)
                    step += 100e6
    os.replace(tmp, ZIP)


def parse_company(js):
    """First-reported quarterly net income for one company."""
    gaap = (js.get("facts") or {}).get("us-gaap") or {}
    items = None
    for tag in TAGS:
        u = ((gaap.get(tag) or {}).get("units") or {}).get("USD")
        if u:
            items = u
            break
    if not items:
        return []
    df = pd.DataFrame(items)
    need = {"start", "end", "val", "form", "filed"}
    if not need <= set(df.columns):
        return []
    df = df[df["form"].isin(FORMS)].copy()
    if df.empty:
        return []
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["start", "end", "filed", "val"])
    df = df[df["end"] >= MIN_END]
    df["days"] = (df["end"] - df["start"]).dt.days
    first = (df.sort_values("filed")
             .groupby(["start", "end"], as_index=False).first())
    q = first[first["days"].between(80, 100)]
    a = first[first["days"].between(350, 380)]
    rows = {(e): (f, v) for e, f, v in zip(q["end"], q["filed"], q["val"])}
    for _, ar in a.iterrows():
        if ar["end"] in rows:
            continue
        inside = q[(q["start"] >= ar["start"] - pd.Timedelta(days=7))
                   & (q["end"] < ar["end"] - pd.Timedelta(days=20))]
        if len(inside) == 3:
            rows[ar["end"]] = (ar["filed"], ar["val"] - inside["val"].sum())
    return [(e, f, v) for e, (f, v) in rows.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--delete-zip", action="store_true")
    a = ap.parse_args()
    download(a.email)
    zf = zipfile.ZipFile(ZIP)
    names = [n for n in zf.namelist() if n.endswith(".json")]
    print(f"  {len(names)} companies in companyfacts")
    out = []
    for i, n in enumerate(names):
        try:
            js = json.loads(zf.read(n))
        except Exception:  # noqa: BLE001
            continue
        cik = str(js.get("cik", "")).lstrip("0")
        for e, f, v in parse_company(js):
            out.append((cik, e, f, v))
        if i % 2000 == 0:
            print(f"    parsed {i}/{len(names)}  ({len(out)} quarters)",
                  flush=True)
    df = pd.DataFrame(out, columns=["cik", "end", "filed", "ni"])
    import fetch_sec_fundamentals as FS  # noqa: E402
    tk = FS.cik_tickers()
    df["year"] = df["filed"].dt.year
    df = df.merge(tk, on=["cik", "year"], how="left")
    # fill years with no Form 4 from the nearest year for the same company
    df = df.sort_values(["cik", "filed"])
    df["ticker"] = df.groupby("cik")["ticker"].transform(
        lambda s: s.ffill().bfill())
    df = df.drop(columns="year")
    df.to_csv(OUT, index=False)
    print(f"\n  {len(df)} company-quarters, {df['cik'].nunique()} companies, "
          f"{df['ticker'].notna().mean():.0%} with a ticker -> {OUT}")
    if a.delete_zip:
        os.remove(ZIP)
        print("  deleted companyfacts.zip")


if __name__ == "__main__":
    main()
