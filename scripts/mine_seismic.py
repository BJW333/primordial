"""
mine_seismic.py -- Blake's idea (2026-11-15): use seismograph data around coal and gravel mines to
measure mine output, then trade on it. STAGE 1 only: does the seismic signal measure output at all?
(No trading test until this passes -- a signal that doesn't track output can't predict earnings.)

Data (all free, downloaded into data/mine_seismic/ on first run, ~330 MB):
  USGS earthquake catalog, events typed "quarry blast", "mining explosion", "explosion", lower 48,
    2010 -> now (analysts at regional seismic networks label these; ~34,000 events)
  MSHA Mines.txt (every US mine: location, owner, type) and MinesProdQuarterly.txt (hours worked
    and coal tons, by mine and quarter -- the public ground truth)
Method: each blast -> nearest active surface mine (within 8 km; 15 km for the big Wyoming/Montana
  coal pits). Per mine with 60+ blasts and 16+ quarters: correlation between blasts per quarter and
  MSHA output (coal tons for coal, hours worked for quarries/metal), in levels, year-over-year
  changes and quarter-over-quarter changes.
PRE-STATED read: the signal "measures output" only if the MEDIAN year-over-year correlation across
  mines is >= 0.30 and >= 75% of mines are positive. Below that the catalog is not a production meter.

Run (repo root):  python3.10 scripts/mine_seismic.py | tee logs/mine_seismic.txt
"""
import argparse
import glob
import os
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
USGS = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={y}-01-01&endtime={y2}-01-01"
        "&eventtype={t}&minlatitude=24&maxlatitude=50&minlongitude=-125&maxlongitude=-66")
MSHA = "https://arlweb.msha.gov/OpenGovernmentData/DataSets/{}.zip"


def download(d):
    import requests
    os.makedirs(d, exist_ok=True)
    for name in ("Mines", "MinesProdQuarterly"):
        if not os.path.exists(os.path.join(d, name + ".txt")):
            print(f"  downloading MSHA {name} ...", flush=True)
            z = os.path.join(d, name + ".zip")
            with requests.get(MSHA.format(name), stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(z, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            zipfile.ZipFile(z).extractall(d)
    for y in range(2010, pd.Timestamp.now().year + 1):
        for t in ("quarry blast", "mining explosion", "explosion"):
            p = os.path.join(d, f"ev_{y}_{t.split()[0]}.csv")
            if not os.path.exists(p):
                r = requests.get(USGS.format(y=y, y2=y + 1, t=t.replace(" ", "%20")), timeout=120)
                open(p, "w").write(r.text if r.status_code == 200 else "")


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def nearest(ev, mines):
    ml, mo, mid = mines.lat.values, mines.lon.values, mines.index.values
    idx, dist = [], []
    for la, lo in zip(ev.latitude.values, ev.longitude.values):
        d = hav(la, lo, ml, mo)
        i = d.argmin()
        idx.append(mid[i])
        dist.append(d[i])
    return idx, dist


def per_mine(ev, qq, radius, col):
    e = ev[ev.dist <= radius]
    g = (e.groupby(["mine", "yr", "qtr"]).size().rename("n").reset_index()
         .rename(columns={"mine": "MINE_ID", "yr": "CAL_YR", "qtr": "CAL_QTR"}))
    keep = g.groupby("MINE_ID").n.sum()
    keep = keep[keep >= 60].index
    p = qq[qq.MINE_ID.isin(keep)].merge(g, on=["MINE_ID", "CAL_YR", "CAL_QTR"], how="left")
    p["n"] = p["n"].fillna(0)
    first = g.groupby("MINE_ID").CAL_YR.min()
    p = p[p.CAL_YR >= p.MINE_ID.map(first)].sort_values(["MINE_ID", "CAL_YR", "CAL_QTR"])
    last_full = pd.Timestamp.now() - pd.Timedelta(days=150)
    p = p[(p.CAL_YR < last_full.year) | ((p.CAL_YR == last_full.year) & (p.CAL_QTR <= last_full.quarter))]
    rows = []
    for mid, x in p.groupby("MINE_ID"):
        x = x[x[col] > 0]
        if len(x) < 16:
            continue
        lv = np.corrcoef(x.n, x[col])[0, 1]
        dn, dv = x.n.diff(4), x[col].pct_change(4)
        ok = dn.notna() & np.isfinite(dv)
        yy = np.corrcoef(dn[ok], dv[ok])[0, 1] if ok.sum() > 8 and dn[ok].std() > 0 else np.nan
        d1, v1 = x.n.diff().iloc[1:], x[col].pct_change().iloc[1:]
        qo = np.corrcoef(d1, v1)[0, 1] if d1.std() > 0 and np.isfinite(v1).all() else np.nan
        rows.append((mid, len(x), int(x.n.sum()), lv, yy, qo))
    return pd.DataFrame(rows, columns=["mine", "qtrs", "blasts", "level", "yoy", "qoq"]).set_index("mine")


def summary(name, r, mines):
    r = r.dropna(subset=["yoy"])
    med, pos = r.yoy.median(), (r.yoy > 0).mean()
    ok = med >= 0.30 and pos >= 0.75
    print(f"\n  ===== {name}: {len(r)} mines =====")
    print(f"  median correlation with MSHA output: levels {r.level.median():+.2f} | year-over-year "
          f"{med:+.2f} | quarter-over-quarter {r.qoq.median():+.2f} | {pos*100:.0f}% of mines positive (yoy)")
    t = r.join(mines[["CURRENT_MINE_NAME", "STATE", "CURRENT_CONTROLLER_NAME"]]).sort_values("blasts", ascending=False)
    for mid, x in t.head(12).iterrows():
        print(f"    {str(x.CURRENT_MINE_NAME)[:30]:<30} {x.STATE} {str(x.CURRENT_CONTROLLER_NAME)[:28]:<28} "
              f"blasts {x.blasts:5.0f}  level {x.level:+.2f}  yoy {x.yoy:+.2f}  qoq {x.qoq:+.2f}")
    print(f"  measures output (median yoy >= 0.30 and >= 75% positive)? {'YES' if ok else 'NO'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "mine_seismic"))
    a = ap.parse_args()
    download(a.dir)
    fs = [f for f in glob.glob(os.path.join(a.dir, "ev_*.csv")) if os.path.getsize(f) > 200]
    ev = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    ev["time"] = pd.to_datetime(ev["time"])
    ev["yr"], ev["qtr"] = ev.time.dt.year, ev.time.dt.quarter
    states = ev.place.str.extract(r",\s*([A-Za-z ]+)$")[0].value_counts()
    print(f"  USGS blasts: {len(ev):,} events {ev.time.min().date()} -> {ev.time.max().date()}; "
          f"by type {ev.type.value_counts().to_dict()}")
    print("  where the catalog has blasts (top areas): " + ", ".join(f"{k} {v}" for k, v in states.head(10).items()))
    m = pd.read_csv(os.path.join(a.dir, "Mines.txt"), sep="|", encoding="latin-1", low_memory=False)
    m = m.dropna(subset=["LONGITUDE", "LATITUDE"])
    m["lon"], m["lat"] = -m.LONGITUDE.abs(), m.LATITUDE
    m = m[(m.lat > 24) & (m.lat < 50) & (m.lon < -66) & (m.lon > -125)]
    q = pd.read_csv(os.path.join(a.dir, "MinesProdQuarterly.txt"), sep="|", encoding="latin-1",
                    usecols=["MINE_ID", "CAL_YR", "CAL_QTR", "HOURS_WORKED", "COAL_PRODUCTION"], low_memory=False)
    q = q[q.CAL_YR >= 2010]
    qq = q.groupby(["MINE_ID", "CAL_YR", "CAL_QTR"], as_index=False)[["HOURS_WORKED", "COAL_PRODUCTION"]].sum()
    tot = qq.groupby("MINE_ID").HOURS_WORKED.sum()
    m = m[m.MINE_ID.isin(tot[tot > 100000].index) & (m.CURRENT_MINE_TYPE == "Surface")].set_index("MINE_ID")
    coal, rock = m[m.COAL_METAL_IND == "C"], m[m.COAL_METAL_IND == "M"]
    print(f"  MSHA: {len(coal)} active surface coal mines, {len(rock):,} surface quarries / metal mines")

    ce = ev[ev.type == "mining explosion"].copy()
    ce["mine"], ce["dist"] = nearest(ce, coal)
    ok1 = summary("COAL (blasts vs coal tons)", per_mine(ce, qq, 15, "COAL_PRODUCTION"), coal)
    re_ = ev.copy()
    re_["mine"], re_["dist"] = nearest(re_, rock)
    ok2 = summary("QUARRIES and METAL (blasts vs hours worked)", per_mine(re_, qq, 8, "HOURS_WORKED"), rock)
    print(f"\n  STAGE 1 VERDICT: coal {'PASS' if ok1 else 'FAIL'}, quarries {'PASS' if ok2 else 'FAIL'}"
          " -> a trading test is only worth building for a group that passes")


if __name__ == "__main__":
    main()
