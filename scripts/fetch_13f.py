"""
fetch_13f.py -- hedge-fund "best ideas" from SEC Form 13F holdings, 2015+.

Sources (free, SEC):
  * Form 13F data sets (every 13F-HR filing's holdings table), quarterly
    zips listed on sec.gov/data-research/sec-markets-data/form-13f-data-sets
  * Fails-to-deliver files (cnsfailsYYYYMMa/b.zip): CUSIP + ticker by date,
    delisted names included -> the CUSIP -> ticker map

BEST NEW IDEA (Cohen, Polk & Silli 2010, simplified): for each manager
with 20-500 stock positions, at each quarter: its largest-weight position
that it did NOT hold the previous quarter. Signal date = 13F filing date
(up to 45 days after quarter end -- that lag is real and kept).
Index-sized managers (> 500 positions) and tiny ones (< 20) are dropped.
Options (put/call rows) are dropped. Amendments are ignored.

Output
  data/13f_best_ideas.csv.gz   mgr, period, filed, cusip, ticker, weight
  data/cusip_map.csv.gz        cusip, date, ticker
Cache: data/sec_13f/ (per-zip compact holdings; zips deleted with
--delete-zips), data/sec_ftd/

Run (repo root, ~1-2 hours first time, resumable):
    python3.10 scripts/fetch_13f.py --email you@example.com --delete-zips
"""
import argparse
import glob
import io
import os
import re
import time
import warnings
import zipfile

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D13 = os.path.join(ROOT, "data", "sec_13f")
DFTD = os.path.join(ROOT, "data", "sec_ftd")
OUT = os.path.join(ROOT, "data", "13f_best_ideas.csv.gz")
MAP = os.path.join(ROOT, "data", "cusip_map.csv.gz")
PAGES = ["https://www.sec.gov/data-research/sec-markets-data/"
         "form-13f-data-sets",
         "https://www.sec.gov/dera/data/form-13f-data-sets"]
FTD = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{}.zip"
MIN_POS, MAX_POS, FROM_YEAR = 20, 500, 2015
warnings.filterwarnings("ignore")


def sess(email):
    s = requests.Session()
    s.headers["User-Agent"] = f"primordial research {email}"
    return s


def get(s, url, tries=4):
    for i in range(tries):
        r = s.get(url, timeout=120)
        if r.status_code in (429, 503):
            time.sleep(5 * (i + 1))
            continue
        time.sleep(0.15)
        return r
    return r


def zip_links(s):
    links = set()
    for p in PAGES:
        try:
            r = get(s, p)
            if r.ok:
                for h in re.findall(r'href="([^"]+_form13f\.zip)"', r.text):
                    links.add(h if h.startswith("http")
                              else "https://www.sec.gov" + h)
        except requests.RequestException:
            pass
    if not links:                       # fallback: old quarterly naming
        for y in range(FROM_YEAR, 2025):
            for q in range(1, 5):
                links.add("https://www.sec.gov/files/structureddata/data/"
                          f"form-13f-data-sets/{y}q{q}_form13f.zip")
    keep = []
    for u in sorted(links):
        yr = re.findall(r"(20\d\d)", os.path.basename(u))
        if yr and int(yr[0]) >= FROM_YEAR:
            keep.append(u)
    return keep


def table(z, name, cols):
    hit = [n for n in z.namelist()
           if os.path.basename(n).upper().startswith(name)]
    return pd.read_csv(io.BytesIO(z.read(hit[0])), sep="\t", dtype=str,
                       usecols=lambda c: c.upper() in cols,
                       on_bad_lines="skip", low_memory=False)


def to_date(x):
    d = pd.to_datetime(x, format="%d-%b-%Y", errors="coerce")
    return d.fillna(pd.to_datetime(x, errors="coerce"))


def compact(path):
    """One 13F zip -> per-filing holdings weights (20-500 positions)."""
    z = zipfile.ZipFile(path)
    sub = table(z, "SUBMISSION", {"ACCESSION_NUMBER", "FILING_DATE",
                                  "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"})
    sub.columns = [c.upper() for c in sub.columns]
    sub = sub[sub["SUBMISSIONTYPE"].str.upper() == "13F-HR"]
    it = table(z, "INFOTABLE", {"ACCESSION_NUMBER", "CUSIP", "VALUE",
                                "PUTCALL"})
    it.columns = [c.upper() for c in it.columns]
    if "PUTCALL" in it:
        it = it[it["PUTCALL"].isna() | (it["PUTCALL"].str.strip() == "")]
    it["VALUE"] = pd.to_numeric(it["VALUE"], errors="coerce")
    it = it.dropna(subset=["VALUE", "CUSIP"])
    it["CUSIP"] = it["CUSIP"].str.upper().str.strip().str[:9]
    it = it[it["ACCESSION_NUMBER"].isin(sub["ACCESSION_NUMBER"])]
    h = it.groupby(["ACCESSION_NUMBER", "CUSIP"], as_index=False)["VALUE"] \
        .sum()
    n = h.groupby("ACCESSION_NUMBER")["CUSIP"].transform("size")
    h = h[(n >= MIN_POS) & (n <= MAX_POS)]
    tot = h.groupby("ACCESSION_NUMBER")["VALUE"].transform("sum")
    h["weight"] = (h["VALUE"] / tot).astype("float32")
    h = h.merge(sub, on="ACCESSION_NUMBER")
    out = pd.DataFrame({"mgr": h["CIK"].str.lstrip("0"),
                        "period": to_date(h["PERIODOFREPORT"]),
                        "filed": to_date(h["FILING_DATE"]),
                        "cusip": h["CUSIP"], "weight": h["weight"]})
    return out.dropna(subset=["period", "filed"])


def fetch_13f(s, delete):
    os.makedirs(D13, exist_ok=True)
    links = zip_links(s)
    print(f"  {len(links)} 13F data-set zips from {FROM_YEAR}")
    for u in links:
        name = os.path.basename(u)
        pq = os.path.join(D13, name.replace(".zip", ".parquet"))
        if os.path.exists(pq):
            continue
        zp = os.path.join(D13, name)
        if not os.path.exists(zp):
            r = get(s, u)
            if not r.ok:
                print(f"    {name}: HTTP {r.status_code}, skipped")
                continue
            open(zp, "wb").write(r.content)
        try:
            c = compact(zp)
        except Exception as e:  # noqa: BLE001
            print(f"    {name}: parse failed ({e})")
            continue
        c.to_parquet(pq)
        print(f"    {name}: {c['mgr'].nunique()} managers, "
              f"{len(c)} positions", flush=True)
        if delete:
            os.remove(zp)


def best_new_ideas():
    parts = [pd.read_parquet(p) for p in sorted(glob.glob(
        os.path.join(D13, "*.parquet")))]
    h = pd.concat(parts, ignore_index=True)
    # first filing per manager-period only (later ones = re-filings)
    first = h.groupby(["mgr", "period"])["filed"].transform("min")
    h = h[h["filed"] == first]
    h = h.sort_values(["mgr", "period"])
    periods = h[["mgr", "period"]].drop_duplicates()
    periods["prev"] = periods.groupby("mgr")["period"].shift(1)
    h = h.merge(periods, on=["mgr", "period"])
    prev_hold = h[["mgr", "period", "cusip"]].rename(
        columns={"period": "prev"})
    prev_hold["held"] = True
    h = h.merge(prev_hold, on=["mgr", "prev", "cusip"], how="left")
    gap = (h["period"] - h["prev"]).dt.days
    new = h[h["held"].isna() & gap.between(80, 100)]
    best = new.sort_values("weight").groupby(["mgr", "period"]).tail(1)
    return best[["mgr", "period", "filed", "cusip", "weight"]]


def fetch_ftd(s):
    os.makedirs(DFTD, exist_ok=True)
    rows = []
    for m in pd.period_range(f"{FROM_YEAR}-01", pd.Timestamp.today()
                             .to_period("M"), freq="M"):
        for half in ("a", "b"):
            tag = f"{m.year}{m.month:02d}{half}"
            p = os.path.join(DFTD, f"{tag}.csv.gz")
            if not os.path.exists(p):
                r = get(s, FTD.format(tag))
                if not r.ok:
                    continue
                try:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    raw = z.read(z.namelist()[0]).decode("latin-1")
                    d = pd.read_csv(io.StringIO(raw), sep="|", dtype=str,
                                    on_bad_lines="skip")
                    d.columns = [c.strip().upper() for c in d.columns]
                    d = d[["SETTLEMENT DATE", "CUSIP", "SYMBOL"]].dropna()
                    d = d.drop_duplicates(["CUSIP", "SYMBOL"])
                    d.to_csv(p, index=False)
                except Exception:  # noqa: BLE001
                    continue
            rows.append(pd.read_csv(p, dtype=str))
    m = pd.concat(rows, ignore_index=True)
    m["date"] = pd.to_datetime(m["SETTLEMENT DATE"], format="%Y%m%d",
                               errors="coerce")
    m = m.dropna(subset=["date"])
    m["cusip"] = m["CUSIP"].str.upper().str.strip().str[:9]
    m["ticker"] = m["SYMBOL"].str.upper().str.strip()
    m = m.groupby(["cusip", "ticker"], as_index=False)["date"].min()
    m.to_csv(MAP, index=False)
    print(f"  CUSIP map: {m['cusip'].nunique()} CUSIPs, "
          f"{m['ticker'].nunique()} tickers")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--delete-zips", action="store_true")
    a = ap.parse_args()
    s = sess(a.email)
    fetch_13f(s, a.delete_zips)
    best = best_new_ideas()
    m = fetch_ftd(s)
    best = best.sort_values("filed")
    m = m.sort_values("date")
    best = pd.merge_asof(best, m.rename(columns={"date": "filed"}),
                         on="filed", by="cusip", direction="backward")
    miss = best["ticker"].isna()
    if miss.any():                         # first seen after the filing
        fwd = pd.merge_asof(best[miss].drop(columns="ticker"),
                            m.rename(columns={"date": "filed"}),
                            on="filed", by="cusip", direction="forward")
        best.loc[miss, "ticker"] = fwd["ticker"].values
    best.to_csv(OUT, index=False)
    print(f"\n  {len(best)} best new ideas, {best['mgr'].nunique()} managers,"
          f" {best['ticker'].notna().mean():.0%} with a ticker, "
          f"{best['filed'].min().date()} -> {best['filed'].max().date()}"
          f" -> {OUT}")


if __name__ == "__main__":
    main()
