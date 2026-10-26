"""
fetch_sec_fundamentals.py -- annual quality metrics for every SEC filer,
point-in-time, from the SEC's free Financial Statement Data Sets
(quarterly zips of every 10-K/10-Q XBRL filing, dead companies included).

    python3.10 scripts/fetch_sec_fundamentals.py --email you@example.com
    -> data/sec_fsds/<yyyy>q<q>.zip   (cached; BIG: ~50-100 MB each)
    -> data/fundamentals.csv.gz

Only 10-K filings are used (annual, audited). Each row is usable from the
day AFTER its filing date -- never earlier -- so no look-ahead.

Per 10-K:
  roa      net income / total assets
  gpa      gross profit / total assets   (gross profit = GrossProfit, or
           revenue minus cost of revenue; missing for banks -> NaN)
  lev      total liabilities / total assets   (lower = better)
  roa_std  std-dev of this company's last 3 annual ROAs (lower = steadier),
           computed only from 10-Ks already filed at that point

Tickers come from the Form 4 data cached by fetch_insider_buys.py
(CIK -> trading symbol around the filing date).

--start defaults to 2013 so 3-year stability exists by the 2017 backtest
start. Pass --delete-zips to drop each zip after parsing (saves disk).
"""
import argparse
import glob
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "sec_fsds")
F345 = os.path.join(ROOT, "data", "sec_form345")
OUT = os.path.join(ROOT, "data", "fundamentals.csv.gz")
URLS = ["https://www.sec.gov/files/dera/data/financial-statement-data-sets/"
        "{q}.zip",
        "https://www.sec.gov/files/dera/data/financial-statement-data-sets-"
        "zip/{q}.zip",
        "https://www.sec.gov/data/financial-statements/{q}.zip"]

TAGS = {
    "ni": ["NetIncomeLoss", "ProfitLoss",
           "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "assets": ["Assets"],
    "liab": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableTo"
               "NoncontrollingInterest"],
    "gp": ["GrossProfit"],
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet", "RevenueFromContractWithCustomerIncluding"
            "AssessedTax"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold",
             "CostOfGoodsSold"],
}
FLOW = {"ni", "gp", "rev", "cogs"}           # need qtrs == 4
ALL_TAGS = {t for v in TAGS.values() for t in v}


def quarters(start):
    now = pd.Timestamp.today()
    for y in range(start, now.year + 1):
        for q in range(1, 5):
            if pd.Timestamp(y, 3 * q - 2, 1) <= now:
                yield f"{y}q{q}"


def download(q, email):
    path = os.path.join(CACHE, f"{q}.zip")
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    hdr = {"User-Agent": f"primordial research {email}"}
    for u in URLS:
        r = requests.get(u.format(q=q), headers=hdr, timeout=600)
        time.sleep(0.3)
        if r.status_code == 200 and r.content[:2] == b"PK":
            open(path, "wb").write(r.content)
            return path
    return None


def parse(path):
    z = zipfile.ZipFile(path)
    sub = pd.read_csv(io.BytesIO(z.read("sub.txt")), sep="\t", dtype=str,
                      usecols=["adsh", "cik", "form", "period", "filed"],
                      on_bad_lines="skip")
    sub = sub[sub["form"].isin(["10-K", "10-K/A"])]
    keep = set(sub["adsh"])
    rows = []
    with z.open("num.txt") as fh:
        for ch in pd.read_csv(fh, sep="\t", dtype=str, chunksize=1_000_000,
                              on_bad_lines="skip", encoding="latin-1"):
            ch = ch[ch["adsh"].isin(keep) & ch["tag"].isin(ALL_TAGS)
                    & (ch["uom"] == "USD")]
            for col in ("coreg", "segments"):        # whole-company only
                if col in ch:
                    ch = ch[ch[col].isna()]
            rows.append(ch[["adsh", "tag", "ddate", "qtrs", "value"]])
    num = pd.concat(rows, ignore_index=True)
    num = num.merge(sub, on="adsh")
    num = num[num["ddate"] == num["period"]]          # this fiscal year only
    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    out = []
    for adsh, g in num.groupby("adsh"):
        rec = {"adsh": adsh, "cik": g["cik"].iloc[0].lstrip("0"),
               "period": g["period"].iloc[0], "filed": g["filed"].iloc[0]}
        for key, tags in TAGS.items():
            want_q = "4" if key in FLOW else "0"
            gg = g[g["qtrs"] == want_q]
            rec[key] = np.nan
            for t in tags:                            # first tag that exists
                v = gg.loc[gg["tag"] == t, "value"]
                if len(v):
                    rec[key] = float(v.iloc[0])
                    break
        out.append(rec)
    return pd.DataFrame(out)


def cik_tickers():
    frames = []
    for zp in sorted(glob.glob(os.path.join(F345, "*_form345.zip"))):
        zf = zipfile.ZipFile(zp)
        n = [x for x in zf.namelist()
             if os.path.basename(x).upper().startswith("SUBMISSION")][0]
        frames.append(pd.read_csv(io.BytesIO(zf.read(n)), sep="\t",
                                  dtype=str, on_bad_lines="skip",
                                  usecols=["FILING_DATE", "ISSUERCIK",
                                           "ISSUERTRADINGSYMBOL"]).dropna())
    if not frames:
        raise SystemExit("no Form 4 cache -- run fetch_insider_buys.py first")
    s = pd.concat(frames, ignore_index=True)
    s["cik"] = s["ISSUERCIK"].str.lstrip("0")
    s["year"] = pd.to_datetime(s["FILING_DATE"], format="%d-%b-%Y",
                               errors="coerce").dt.year
    s["ticker"] = s["ISSUERTRADINGSYMBOL"].str.upper().str.strip()
    s = s[s["ticker"].str.fullmatch(r"[A-Z0-9.\-]{1,7}", na=False)]
    return (s.groupby(["cik", "year", "ticker"]).size().reset_index(name="n")
            .sort_values("n").groupby(["cik", "year"]).tail(1)
            [["cik", "year", "ticker"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--start", type=int, default=2013)
    ap.add_argument("--delete-zips", action="store_true")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    parts, missing = [], []
    for q in quarters(a.start):
        pq = os.path.join(CACHE, f"{q}.parsed.csv")
        if os.path.exists(pq):
            parts.append(pd.read_csv(pq, dtype={"cik": str}))
            continue
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
        d.to_csv(pq, index=False)
        parts.append(d)
        if a.delete_zips:
            os.remove(p)
        print(f"  {q}: {len(d):>5} 10-K filings")
    f = pd.concat(parts, ignore_index=True).drop_duplicates("adsh")
    f["filed"] = pd.to_datetime(f["filed"].astype(str), format="%Y%m%d",
                                errors="coerce")
    f["period"] = pd.to_datetime(f["period"].astype(str), format="%Y%m%d",
                                 errors="coerce")
    f = f.dropna(subset=["filed", "assets"])
    f = f[f["assets"] > 0]
    gp = f["gp"].fillna(f["rev"] - f["cogs"])
    f["roa"] = f["ni"] / f["assets"]
    f["gpa"] = gp / f["assets"]
    f["lev"] = f["liab"] / f["assets"]
    f = f.sort_values(["cik", "filed"])
    f["roa_std"] = (f.groupby("cik")["roa"]
                    .transform(lambda s: s.rolling(3, min_periods=3).std()))
    f["year"] = f["filed"].dt.year
    f = f.merge(cik_tickers(), on=["cik", "year"], how="left")
    keep = ["ticker", "cik", "filed", "period", "roa", "gpa", "lev",
            "roa_std"]
    f[keep].to_csv(OUT, index=False)
    print(f"\n  {len(f)} annual filings, {f['ticker'].notna().sum()} with a "
          f"ticker, {f['cik'].nunique()} companies -> {OUT}")
    if missing:
        print(f"  quarters unavailable: {', '.join(missing)}")


if __name__ == "__main__":
    main()
