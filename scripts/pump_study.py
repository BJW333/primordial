"""
pump_study.py -- crypto pump-and-dump event study: after a coin suddenly
spikes on huge volume, does it keep going (ride) or collapse (fade), and
when? Answer this BEFORE building any bot.

DATA (free): Binance public archive data.binance.vision, every USDT spot
pair incl. DELISTED ones, 2018 -> now.
  Stage 1: hourly bars per coin. Downloaded, scanned, then thrown away --
           only the pump events + daily closes are kept (disk-friendly).
  Stage 2: 5-minute bars ONLY for the days a pump happened (zoom-in).
Stablecoins, fiat pairs and leveraged tokens (UP/DOWN/BULL/BEAR) dropped.
A coin's first 14 days after listing are skipped (listings always spike).

PUMP (pre-stated, known at the close of the hour):
  hour's trading volume ($) >= 10x the coin's median hourly volume over
  the previous 7 days, AND the hour's close is >= +10% vs the prior close.
  One event per coin per 72 hours.

WHAT HAPPENED NEXT (close-to-close, minus BTC over the same window so a
market-wide move doesn't count): +1h, +4h, +24h, +72h, +7d. Plus, from 5m
data: how many minutes into the hour the peak came, and an EARLY signal --
the first 5-minute bar in the pump hour with >= 10x normal 5m volume and
>= +5% -- then its return 15 and 30 minutes later.

PRE-STATED VERDICTS (deciding period = last five years 2021-06 -> 2026-05;
2018 -> 2021-05 shown as context; t-stats cluster events by day)
  FADE  short at the detection close, cover 24h later, 0.30% round trip:
        PASS if mean >= +2% net, median > 0, t > 3, and the same sign in
        the context period.
  RIDE  buy at the EARLY 5m signal, sell 30 minutes later, 0.20% round
        trip: PASS if mean >= +1% net, median > 0, t > 3.
  Results also split by coin size (median daily $ volume before the
  pump): tiny < $1M, small $1-10M, large > $10M.

Run (repo root; stage 1 ~30-60 min first time, resumable):
    python3.10 scripts/pump_study.py --smoke
    python3.10 scripts/pump_study.py | tee logs/pump_study.txt
Output: data/crypto_pumps/ (events.csv, daily_close.parquet, done/ markers)
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "crypto_pumps")
DONE = os.path.join(DIR, "done")
EVENTS = os.path.join(DIR, "events.csv")
FINE = os.path.join(DIR, "events_5m.csv")
DAILY = os.path.join(DIR, "daily")
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
WEB = "https://data.binance.vision"
START = pd.Timestamp("2018-01-01")
RECENT, END = pd.Timestamp("2021-06-01"), pd.Timestamp("2026-06-01")
VOL_X, RET_MIN, BASE_H, GAP_H, SKIP_NEW_D = 10.0, 0.10, 168, 72, 14
HORIZONS = {"1h": 1, "4h": 4, "24h": 24, "72h": 72, "7d": 168}
FADE_COST, RIDE_COST = 0.0030, 0.0020
BAD = re.compile(r"(UP|DOWN|BULL|BEAR)USDT$|^(USDC|BUSD|TUSD|FDUSD|USDP|"
                 r"DAI|PAX|EUR|GBP|AUD|BRL|TRY|UST|USDS|SUSD|AEUR|EURI|"
                 r"XUSD|USD1|BFUSD|RLUSD|PYUSD)USDT$")
SESSION = requests.Session()


# ------------------------------------------------------------- download
def s3_list(prefix):
    """All sub-prefixes (folders) under a prefix of the public bucket."""
    out, marker = [], ""
    while True:
        r = SESSION.get(S3, params={"prefix": prefix, "delimiter": "/",
                                    "marker": marker}, timeout=60)
        r.raise_for_status()
        out += re.findall(r"<Prefix>([^<]+)</Prefix>", r.text)[1:]
        keys = re.findall(r"<Key>([^<]+)</Key>", r.text)
        out += keys
        if "<IsTruncated>true</IsTruncated>" not in r.text:
            break
        nm = re.findall(r"<NextMarker>([^<]+)</NextMarker>", r.text)
        marker = nm[0] if nm else (out[-1] if out else "")
    return out


def symbols():
    pre = s3_list("data/spot/monthly/klines/")
    syms = sorted({p.rstrip("/").split("/")[-1] for p in pre})
    return [s for s in syms if s.endswith("USDT") and not BAD.search(s)]


def read_klines(content):
    z = zipfile.ZipFile(io.BytesIO(content))
    raw = z.read(z.namelist()[0])
    d = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 2, 4, 7])
    if not np.issubdtype(d[0].dtype, np.number):     # header row present
        d = d.iloc[1:].astype(float)
    d.columns = ["t", "high", "close", "qvol"]
    ms = np.where(d["t"] > 1e14, d["t"] / 1000, d["t"])  # 2025+ files: us
    d["t"] = pd.to_datetime(ms.astype("int64"), unit="ms")
    return d


def get(url, tries=4):
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return None


def hourly(sym):
    keys = [k for k in s3_list(f"data/spot/monthly/klines/{sym}/1h/")
            if k.endswith(".zip")]
    parts = []
    for k in keys:
        ym = re.findall(r"(\d{4})-(\d{2})\.zip$", k)
        if not ym or pd.Timestamp(f"{ym[0][0]}-{ym[0][1]}-01") < START:
            continue
        c = get(f"{WEB}/{k}")
        if c:
            parts.append(read_klines(c))
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates("t").set_index("t").sort_index()
    return d.astype("float64")


# ---------------------------------------------------------------- detect
def detect(d, btc):
    """Pump events in one coin's hourly series (+ forward returns)."""
    d = d[d.index >= d.index[0] + pd.Timedelta(days=SKIP_NEW_D)]
    if len(d) < BASE_H + 10:
        return []
    q = d["qvol"].values
    c = d["close"].values
    h = d["high"].values
    base = pd.Series(q).shift(1).rolling(BASE_H, min_periods=120).median() \
        .values
    ret = c / np.r_[np.nan, c[:-1]] - 1
    ok = (q >= VOL_X * base) & (base > 0) & (ret >= RET_MIN)
    bclose = btc.reindex(d.index).ffill().values
    daily_vol = pd.Series(q).shift(1).rolling(24 * 30, min_periods=24 * 7) \
        .median().values * 24
    out, last = [], -10 ** 9
    for i in np.flatnonzero(ok):
        if i - last < GAP_H:
            continue
        last = i
        rec = {"t": d.index[i], "ret_hour": ret[i],
               "peak": h[i] / c[i - 1] - 1, "vol_x": q[i] / base[i],
               "dollar_vol_day": daily_vol[i]}
        for name, k in HORIZONS.items():
            j = i + k
            if j < len(c) and (d.index[j] - d.index[i]) <= pd.Timedelta(
                    hours=k + 6):
                coin = c[j] / c[i] - 1
                b = bclose[j] / bclose[i] - 1 if np.isfinite(
                    bclose[j] * bclose[i]) else 0.0
                rec[name] = coin - b
            else:
                rec[name] = np.nan            # delisted / gap: unknown
        out.append(rec)
    return out


def stage1(workers, limit=0):
    os.makedirs(DONE, exist_ok=True)
    os.makedirs(DAILY, exist_ok=True)
    syms = symbols()
    if limit:
        syms = syms[:limit]
    print(f"  {len(syms)} USDT spot pairs in the Binance archive "
          f"(incl. delisted)")
    bp = os.path.join(DIR, "btc_hourly.parquet")
    if os.path.exists(bp):
        btc = pd.read_parquet(bp)["close"]
    else:
        btc = hourly("BTCUSDT")["close"]
        btc.to_frame().to_parquet(bp)
    todo = [s for s in syms if not os.path.exists(
        os.path.join(DONE, s + ".csv"))]
    print(f"  {len(syms) - len(todo)} already scanned, {len(todo)} to go")

    def one(s):
        try:
            d = hourly(s)
        except Exception:  # noqa: BLE001
            return s, None
        ev = detect(d, btc) if d is not None else []
        pd.DataFrame(ev).assign(sym=s).to_csv(
            os.path.join(DONE, s + ".csv"), index=False)
        if d is not None:
            d["close"].resample("D").last().to_frame(s).to_parquet(
                os.path.join(DAILY, s + ".parquet"))
        return s, len(ev)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, (s, n) in enumerate(ex.map(one, todo), 1):
            if i % 25 == 0 or i == len(todo):
                print(f"    scanned {i}/{len(todo)} ({(time.time()-t0)/60:.0f}"
                      f" min) last: {s} {n} pumps", flush=True)
    parts = []
    for f in os.listdir(DONE):
        try:
            parts.append(pd.read_csv(os.path.join(DONE, f),
                                     parse_dates=["t"]))
        except (pd.errors.EmptyDataError, ValueError):
            continue
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    ev.to_csv(EVENTS, index=False)
    return ev


# ------------------------------------------------------------ 5m zoom-in
def five_min(sym, day):
    c = get(f"{WEB}/data/spot/daily/klines/{sym}/5m/"
            f"{sym}-5m-{day:%Y-%m-%d}.zip")
    return read_klines(c).set_index("t") if c else None


def zoom(row, btc5):
    t0 = pd.Timestamp(row["t"])                  # start of the pump hour
    parts = [five_min(row["sym"], d) for d in
             (t0.normalize() - pd.Timedelta(days=1), t0.normalize(),
              t0.normalize() + pd.Timedelta(days=1))]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    m = pd.concat(parts).sort_index()
    m = m[~m.index.duplicated()]
    hour = m[(m.index >= t0) & (m.index < t0 + pd.Timedelta(hours=1))]
    before = m[(m.index < t0) & (m.index >= t0 - pd.Timedelta(days=1))]
    if len(hour) < 6 or len(before) < 100:
        return None
    base = before["qvol"].median()
    prev = before["close"].iloc[-1]
    peak_i = hour["high"].values.argmax()
    rec = {"peak_min": int(peak_i * 5)}
    r5 = hour["close"] / hour["close"].shift(1).fillna(prev) - 1
    hit = np.flatnonzero((hour["qvol"].values >= VOL_X * base)
                         & (r5.values >= 0.05))
    if len(hit):
        i = hit[0]
        ts = hour.index[i]
        rec["early_min"] = int(i * 5)
        p0 = hour["close"].iloc[i]
        for k, lab in ((3, "e15m"), (6, "e30m")):
            tt = ts + pd.Timedelta(minutes=5 * k)
            if tt in m.index:
                rec[lab] = m.at[tt, "close"] / p0 - 1
    end = t0 + pd.Timedelta(minutes=55)          # last 5m bar of the hour
    if end in m.index:
        p1 = m.at[end, "close"]
        for k, lab in ((3, "d15m"), (6, "d30m")):
            tt = end + pd.Timedelta(minutes=5 * k)
            if tt in m.index:
                rec[lab] = m.at[tt, "close"] / p1 - 1
    return rec


def stage2(ev, workers):
    have = pd.read_csv(FINE, parse_dates=["t"]) if os.path.exists(FINE) \
        else pd.DataFrame(columns=["sym", "t"])
    key = set(zip(have["sym"], have["t"]))
    rows = [r for r in ev[["sym", "t"]].to_dict("records")
            if (r["sym"], r["t"]) not in key]

    def one(r):
        try:
            z = zoom(r, None)
        except Exception:  # noqa: BLE001
            z = None
        return {**r, **(z or {})}

    print(f"  stage 2: 5-minute zoom-in on {len(rows)} pumps")
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, rows))
    f = pd.concat([have, pd.DataFrame(res)], ignore_index=True)
    f.to_csv(FINE, index=False)
    return ev.merge(f, on=["sym", "t"], how="left")


# ---------------------------------------------------------------- report
def day_t(x, t):
    s = pd.Series(np.asarray(x, float), index=pd.to_datetime(t)).dropna()
    g = s.groupby(s.index.normalize()).mean()
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 5 \
        else np.nan


def summarize(ev, label):
    print(f"\n  ===== {label}: {len(ev)} pumps, {ev['sym'].nunique()} coins "
          f"=====")
    if not len(ev):
        return {}
    print(f"  typical pump hour: +{ev['ret_hour'].median()*100:.0f}% close, "
          f"+{ev['peak'].median()*100:.0f}% at the peak, "
          f"{ev['vol_x'].median():.0f}x normal volume")
    if "peak_min" in ev and ev["peak_min"].notna().any():
        em = ev["early_min"] if "early_min" in ev else pd.Series(dtype=float)
        share = em.notna().mean()
        print(f"  peak comes {ev['peak_min'].median():.0f} min into the hour"
              f" (median); early 5m signal fires at {em.median():.0f} min "
              f"({share:.0%} of pumps)")
    print(f"\n  after detection (coin minus BTC):  {'mean':>7} {'median':>7}"
          f" {'% down':>7} {'t':>6} {'n':>6}")
    cols = [("d15m", "+15 min"), ("d30m", "+30 min")] + \
        [(k, "+" + k) for k in HORIZONS]
    for c, lab in cols:
        if c not in ev:
            continue
        x = ev[c].dropna()
        if len(x) < 10:
            continue
        tt = day_t(x, ev.loc[x.index, "t"])
        print(f"    {lab:<29} {x.mean()*100:>+6.1f}% {x.median()*100:>+6.1f}%"
              f" {(x < 0).mean()*100:>6.0f}% {tt:>+6.2f} {len(x):>6}")
    out = {}
    f = -ev["24h"].dropna() - FADE_COST
    out["fade"] = (f.mean(), f.median(), day_t(f, ev.loc[f.index, "t"]))
    if "e30m" in ev:
        r = ev["e30m"].dropna() - RIDE_COST
        out["ride"] = (r.mean(), r.median(), day_t(r, ev.loc[r.index, "t"])) \
            if len(r) > 10 else (np.nan, np.nan, np.nan)
        if len(r) > 10:
            print(f"    ride from EARLY 5m signal +30m {r.mean()*100:>+6.1f}% "
                  f"{r.median()*100:>+6.1f}% (net of {RIDE_COST*100:.1f}%), "
                  f"t {out['ride'][2]:+.2f}, n {len(r)}")
    print(f"    fade 24h (short, net {FADE_COST*100:.1f}%)   "
          f"{out['fade'][0]*100:>+6.1f}% {out['fade'][1]*100:>+6.1f}%  "
          f"t {out['fade'][2]:+.2f}")
    print("  by coin size (fade 24h net, mean / median / n):")
    for lab, lo, hi in (("tiny <$1M/day", 0, 1e6), ("small $1-10M", 1e6,
                                                    1e7),
                        ("large >$10M", 1e7, np.inf)):
        g = ev[(ev["dollar_vol_day"] >= lo) & (ev["dollar_vol_day"] < hi)]
        x = -g["24h"].dropna() - FADE_COST
        if len(x) > 5:
            print(f"    {lab:<15} {x.mean()*100:+.1f}% / "
                  f"{x.median()*100:+.1f}% / {len(x)}")
    by = ev.groupby(ev["t"].dt.year).size()
    print("  pumps per year: " + "  ".join(f"{y}:{n}" for y, n in
                                            by.items()))
    return out


# ------------------------------------------------------------------ smoke
def smoke():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2018-01-01", "2026-05-31", freq="h")
    btc = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.006, len(idx)))) * 1e4,
                    idx)
    evs = []
    for k in range(60):
        c = np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        q = rng.lognormal(10, 0.3, len(idx))
        for i in range(800 + k * 13, len(idx) - 400, 1500):
            c[i:] *= 1.4                           # pump +40% ...
            q[i] *= 40
            c[i + 1:] *= 0.92                      # ... gives back 8% in 1h
            c[i + 5:] *= 0.85                      # ... and 15% more by 5h
        h = c * 1.01
        h[q > 1e6] *= 1.15
        d = pd.DataFrame({"high": h, "close": c, "qvol": q}, idx)
        evs += [dict(e, sym=f"C{k}") for e in detect(d, btc)]
    return pd.DataFrame(evs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="scan only the first N coins (quick trial)")
    a = ap.parse_args()
    if a.smoke:
        ev = smoke()
        print(f"  smoke: {len(ev)} pumps detected (planted 2,880); "
              "planted: -8% at 1h, ~-22% by 24h")
    else:
        try:
            SESSION.get(WEB, timeout=20)
        except requests.RequestException as e:
            sys.exit(f"cannot reach {WEB}: {e}")
        ev = stage1(a.workers, a.limit)      # skips coins already scanned
        if not len(ev):
            sys.exit("no pumps found -- check the download")
        ev = ev[(ev["t"] >= START) & (ev["t"] < END)]
        if not a.no_zoom:
            ev = stage2(ev, a.workers)
    old = ev[ev["t"] < RECENT]
    new = ev[ev["t"] >= RECENT]
    ctx = summarize(old, "CONTEXT 2018 -> 2021-05")
    dec = summarize(new, "DECIDING: LAST FIVE YEARS 2021-06 -> 2026-05")
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: planted collapse after each pump (FADE should pass)")
    fm, fmed, ft = dec.get("fade", (np.nan,) * 3)
    same = ctx.get("fade", (np.nan,))[0]
    ok = fm >= 0.02 and fmed > 0 and ft > 3 and (np.isnan(same) or same > 0)
    print(f"  FADE 24h: mean {fm*100:+.1f}% median {fmed*100:+.1f}% t {ft:+.2f}"
          f" (context mean {same*100:+.1f}%) -> {'PASS' if ok else 'FAIL'}")
    rm, rmed, rt = dec.get("ride", (np.nan,) * 3)
    ok2 = rm >= 0.01 and rmed > 0 and rt > 3
    print(f"  RIDE early +30m: mean {rm*100:+.1f}% median {rmed*100:+.1f}% "
          f"t {rt:+.2f} -> {'PASS' if ok2 else 'FAIL (or no 5m data)'}")


if __name__ == "__main__":
    main()
