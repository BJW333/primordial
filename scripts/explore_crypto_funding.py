"""
ZERO-TRIAL screen: crypto FORCED-DELEVERAGING absorption, daily hold.

Mechanism: perp liquidation engines are the most literal forced seller in
any market -- leveraged longs are market-sold into a falling book by an
algorithm with a deadline of "now". Historical per-order liquidation tapes
are NOT freely available; the free, honest proxy is FUNDING: a violently
negative funding print after a down move marks the aftermath of long
liquidations / crowded shorts paying to stay short. Trade the absorption
at DAILY hold with MAKER fills -- the sub-hour seam stays closed.

Data: perp funding history from a free public REST API (--venue binance
fapi, fallback bybit if geo-blocked) + spot daily bars via the primordial
cache (coinbase source, already warmed). Funding is archived to
out/funding_{sym}.csv on first fetch so this survives API changes.

PRIMARY CELL (frozen before any table prints): per symbol, 3-day funding
sum <= its TRAIN p10 AND 3-day spot return < 0 -- buy spot at that day's
close, hold 5 sessions, net --cost-bps (default 46 = maker 23 x 2).
PASS: mean > 0 AND t >= 2 on train (first 70% of dates), pooled BTC+ETH.
Context rows (deciles, h=3/10, positive-funding mirror) cannot be
promoted. --tail-look is the one look. Nothing logged; 0 trials.

    python3.10 scripts/explore_crypto_funding.py            # fetch + train
    python3.10 scripts/explore_crypto_funding.py --tail-look
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primordial.data import fetch  # noqa: E402

TRAIN_FRAC = 0.70
PAIRS = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "primordial"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def fetch_funding(perp, venue):
    path = f"out/funding_{perp}.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["time"])
    rows, start = [], int(pd.Timestamp("2019-09-01").timestamp() * 1000)
    while True:
        if venue == "binance":
            url = ("https://fapi.binance.com/fapi/v1/fundingRate?symbol="
                   f"{perp}&startTime={start}&limit=1000")
            batch = _get(url)
            if not batch:
                break
            rows += [{"time": int(b["fundingTime"]),
                      "rate": float(b["fundingRate"])} for b in batch]
            start = rows[-1]["time"] + 1
            if len(batch) < 1000:
                break
        else:                                   # bybit v5
            url = ("https://api.bybit.com/v5/market/funding/history?"
                   f"category=linear&symbol={perp}&startTime={start}&limit=200")
            r = _get(url)["result"]["list"]
            if not r:
                break
            r = sorted(r, key=lambda b: int(b["fundingRateTimestamp"]))
            rows += [{"time": int(b["fundingRateTimestamp"]),
                      "rate": float(b["fundingRate"])} for b in r]
            start = rows[-1]["time"] + 1
            if len(r) < 200:
                break
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    os.makedirs("out", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  {perp}: {len(df)} funding prints -> {path}")
    return df


def tstat(x):
    x = pd.Series(x).dropna()
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return len(x), x.mean(), (x.mean() / (sd / len(x) ** 0.5)
                              if sd > 0 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--venue", choices=["binance", "bybit"], default="binance")
    p.add_argument("--cost-bps", type=float, default=46.0)
    p.add_argument("--tail-look", action="store_true")
    a = p.parse_args()
    cost = a.cost_bps / 1e4

    frames = []
    for perp, spot_sym in PAIRS.items():
        try:
            f = fetch_funding(perp, a.venue)
        except Exception as e:
            sys.exit(f"funding fetch failed on {a.venue} ({str(e)[:60]}) -- "
                     f"retry with --venue bybit")
        daily = f.set_index("time")["rate"].resample("1D").sum()
        spot = fetch("coinbase", [spot_sym], "1d", "2019-09-01",
                     "2026-08-31", min_bars=200).get(spot_sym)
        if spot is None:
            sys.exit(f"no spot bars for {spot_sym} in the coinbase cache")
        c = spot["close"]
        idx = c.index.normalize()
        fr = daily.reindex(idx).to_numpy()
        d = pd.DataFrame({"date": c.index, "close": c.to_numpy(),
                          "fund1": fr})
        d["fund3"] = d["fund1"].rolling(3).sum()
        d["ret3"] = d["close"].pct_change(3)
        for h in (3, 5, 10):
            d[f"f{h}"] = d["close"].shift(-h) / d["close"] - 1.0
        d["sym"] = perp
        frames.append(d.dropna(subset=["fund3", "ret3"]))
    all_ = pd.concat(frames, ignore_index=True).sort_values("date")
    cut = all_["date"].iloc[int(len(all_) * TRAIN_FRAC)]
    tr = all_[all_["date"] < cut]
    p10 = tr.groupby("sym")["fund3"].quantile(0.10)      # TRAIN thresholds
    p90 = tr.groupby("sym")["fund3"].quantile(0.90)
    for fr_ in (all_,):
        fr_["p10"] = fr_["sym"].map(p10)
        fr_["p90"] = fr_["sym"].map(p90)
    cellmask = (all_["fund3"] <= all_["p10"]) & (all_["ret3"] < 0)

    if a.tail_look:
        seg = all_[(all_["date"] >= cut) & cellmask]
        n, m, t = tstat(seg["f5"] - cost)
        print(f"===== TAIL LOOK (one look, frozen cell) | >= {cut.date()} "
              f"=====")
        print(f"  funding<=train-p10 AND ret3<0, buy close, +5d net: "
              f"{m*100:+.2f}% [t {t:+.1f}] n={n}")
        print("  VERDICT: " + ("PASS -- next: gauntlet family on the crypto "
                               "ground, charged normally."
                               if n >= 30 and m > 0 and t >= 2
                               else "DEAD. Seam closed."))
        return

    trc = tr[cellmask.loc[tr.index]]
    print(f"TRAIN {tr['date'].min().date()} .. {cut.date()} (tail withheld)"
          f" | {len(tr):,} sym-days | cost {a.cost_bps:.0f} bps RT")
    n, m, t = tstat(trc["f5"] - cost)
    print(f"\n  PRIMARY: fund3<=p10 AND ret3<0, +5d net: {m*100:+.2f}% "
          f"[t {t:+.1f}] n={n}   frozen pass: >0, t>=2")
    for h in (3, 10):
        n, m, t = tstat(trc[f"f{h}"])
        print(f"  context +{h}d gross: {m*100:+.2f}% [t {t:+.1f}]")
    mir = tr[(tr["fund3"] >= tr["sym"].map(p90)) & (tr["ret3"] > 0)]
    n, m, t = tstat(-(mir["f5"]) - cost)
    print(f"  context mirror (short crowded longs) +5d net: {m*100:+.2f}% "
          f"[t {t:+.1f}] n={n}")
    dec = tr.assign(b=pd.qcut(tr["fund3"], 10, labels=False,
                              duplicates="drop"))
    print("  context funding-decile +5d gross: " + " ".join(
        f"{int(b)}:{g.mean()*100:+.1f}" for b, g in dec.groupby("b")["f5"]))
    print("\nREAD: primary only. Pass earns ONE --tail-look. The mirror row "
          "is context; your short-side result says it will be nothing.")


if __name__ == "__main__":
    main()
