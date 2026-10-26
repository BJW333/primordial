"""
fetch_shares.py -- every company's shares outstanding as reported on the
cover of each 10-Q/10-K (dei:EntityCommonStockSharesOutstanding), stamped
with the filing date (point-in-time). For the buyback test.

Re-uses the SEC companyfacts.zip download from fetch_earnings.py
(~1.4 GB; re-downloaded if it was deleted).

Output: data/shares_outstanding.csv.gz  cik, ticker, filed, asof, shares

Run (repo root):
    python3.10 scripts/fetch_shares.py --email you@example.com --delete-zip
"""
import argparse
import json
import os
import sys
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_earnings as FE  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "shares_outstanding.csv.gz")


def parse(js):
    dei = (js.get("facts") or {}).get("dei") or {}
    u = ((dei.get("EntityCommonStockSharesOutstanding") or {})
         .get("units") or {}).get("shares")
    if not u:
        return []
    out = []
    for x in u:
        if x.get("form") in ("10-Q", "10-K") and x.get("filed") \
                and x.get("val"):
            out.append((x["filed"], x.get("end"), x.get("accn"), x["val"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--delete-zip", action="store_true")
    a = ap.parse_args()
    FE.download(a.email)
    zf = zipfile.ZipFile(FE.ZIP)
    names = [n for n in zf.namelist() if n.endswith(".json")]
    rows = []
    for i, n in enumerate(names):
        try:
            js = json.loads(zf.read(n))
        except Exception:  # noqa: BLE001
            continue
        cik = str(js.get("cik", "")).lstrip("0")
        for f, e, acc, v in parse(js):
            rows.append((cik, f, e, acc, v))
        if i % 4000 == 0:
            print(f"    parsed {i}/{len(names)}", flush=True)
    df = pd.DataFrame(rows, columns=["cik", "filed", "asof", "accn",
                                     "shares"])
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["asof"] = pd.to_datetime(df["asof"], errors="coerce")
    # multiple share classes on one cover -> total; one row per filing
    df = df.groupby(["cik", "accn"], as_index=False).agg(
        filed=("filed", "first"), asof=("asof", "max"),
        shares=("shares", "sum"))
    df = df.dropna(subset=["filed"]).drop(columns="accn")
    import fetch_sec_fundamentals as FS  # noqa: E402
    tk = FS.cik_tickers()
    df["year"] = df["filed"].dt.year
    df = df.merge(tk, on=["cik", "year"], how="left").sort_values(
        ["cik", "filed"])
    df["ticker"] = df.groupby("cik")["ticker"].transform(
        lambda s: s.ffill().bfill())
    df = df.drop(columns="year")
    df.to_csv(OUT, index=False)
    print(f"\n  {len(df)} share counts, {df['cik'].nunique()} companies, "
          f"{df['ticker'].notna().mean():.0%} with a ticker -> {OUT}")
    if a.delete_zip:
        os.remove(FE.ZIP)
        print("  deleted companyfacts.zip")


if __name__ == "__main__":
    main()
