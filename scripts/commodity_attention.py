"""
commodity_attention.py -- does the FADE carry over to futures-type markets?
Wikipedia hype spikes on commodities / currencies / the S&P 500, and what
the market does next. ETF prices stand in for futures.

MARKETS -> Wikipedia article(s) (summed):
  GLD Gold | SLV Silver | PPLT Platinum | PALL Palladium | CPER Copper
  USO Petroleum, Price_of_oil | UNG Natural_gas | URA Uranium
  WEAT Wheat | CORN Maize | UUP United_States_dollar | FXY Japanese_yen
  FXE Euro | SPY S&P_500

HYPE SPIKE (fixed before running): daily views >= 3x prior-90-day median
AND price up >= 10% over the last 30 days; first per market per 60 days;
act at the NEXT day's close (views have a 1-day lag).

TEST 1  FADE: 30 / 90 / 180-day return after the spike vs that market's
        typical (average log) return over the same horizon.
TEST 2  SELL THE HYPE inside the two-way trend sleeve (trend_ls.py): a market
        with a hype spike in the last 90 days is set to ZERO at each
        month-end; compare MOM + TREND 100% with and without.

PRE-STATED (2017-01 -> 2021-05 vs 2021-06 -> 2026-05):
  FADE : pooled 90-day excess < 0 in BOTH periods
  SELL : overlay version has a higher Sharpe in BOTH periods
Small samples -- a pass is a hint, not proof.

Run (repo root, Alpaca keys exported):
    python3.10 scripts/commodity_attention.py --email you@example.com \
        | tee logs/commodity_attention.txt
"""
import argparse
import os
import sys
import time
import urllib.parse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_attention_short as FA    # noqa: E402
import trend_ls as TL                 # noqa: E402
import trend_sleeve as TS             # noqa: E402

ROOT = os.path.dirname(HERE)
OUTD = os.path.join(ROOT, "data", "attention", "markets")
ARTS = {"GLD": ["Gold"], "SLV": ["Silver"], "PPLT": ["Platinum"],
        "PALL": ["Palladium"], "CPER": ["Copper"],
        "USO": ["Petroleum", "Price_of_oil"], "UNG": ["Natural_gas"],
        "URA": ["Uranium"], "WEAT": ["Wheat"], "CORN": ["Maize"],
        "UUP": ["United_States_dollar"], "FXY": ["Japanese_yen"],
        "FXE": ["Euro"], "SPY": ["S&P_500"]}
START, RECENT = pd.Timestamp("2017-01-01"), pd.Timestamp("2021-06-01")
HOR = (30, 90, 180)


def views(s, sym):
    os.makedirs(OUTD, exist_ok=True)
    p = os.path.join(OUTD, sym + ".csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True)["views"]
    end = pd.Timestamp.today().strftime("%Y%m%d")
    tot = None
    for art in ARTS[sym]:
        r = s.get(FA.PV.format(a=urllib.parse.quote(art, safe=""), end=end),
                  timeout=60)
        time.sleep(0.1)
        if r.status_code != 200:
            continue
        v = pd.Series({pd.Timestamp(x["timestamp"][:8]): x["views"]
                       for x in r.json().get("items", [])}, dtype=float)
        tot = v if tot is None else tot.add(v, fill_value=0)
    tot = (tot if tot is not None else pd.Series(dtype=float)).rename("views")
    tot.to_csv(p)
    return tot


def spikes(px, v):
    v = v.asfreq("D").fillna(0)
    base = v.shift(1).rolling(90, min_periods=60).median()
    ratio = (v / base.replace(0, np.nan)).reindex(px.index)
    r30 = px / px.shift(21) - 1                      # ~30 calendar days
    hit = px.index[(ratio >= 3) & (r30 >= 0.10)]
    out, last = [], None
    for d in hit:
        if d < START or (last is not None and (d - last).days < 60):
            continue
        last = d
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    a = ap.parse_args()
    s = FA.sess(a.email)
    px = TS.load(list(ARTS) + [m for m in TL.MKTS if m not in ARTS])
    rows, sp_all = [], {}
    tdays = {30: 21, 90: 63, 180: 126}                # trading-day horizons
    for sym in ARTS:
        if sym not in px:
            continue
        p = px[sym].dropna()
        p = p[p > 0]
        v = views(s, sym)
        if not len(v):
            print(f"  {sym}: no Wikipedia views")
            continue
        sp = spikes(p, v)
        sp_all[sym] = sp
        lp = np.log(p)
        normal = {h: (lp.shift(-tdays[h]) - lp).mean() for h in HOR}
        print(f"  {sym:<5} {len(sp)} hype spikes " +
              ", ".join(f"{d:%Y-%m-%d}" for d in sp))
        for d in sp:
            i = p.index.get_loc(d) + 1
            row = {"sym": sym, "d": d}
            for h in HOR:
                k = tdays[h]
                row[h] = (np.exp(lp.iloc[i + k] - lp.iloc[i] - normal[h]) - 1
                          if i + k < len(p) else np.nan)
            rows.append(row)
    ev = pd.DataFrame(rows)
    verdict = {}
    print("\n  ===== TEST 1: FADE after hype (vs the market's normal drift) =====")
    for lab, g in (("2017-21", ev[ev.d < RECENT]), ("2021-06+", ev[ev.d >= RECENT])):
        cells = [f"+{h}d {g[h].mean()*100:+.1f}% (med {g[h].median()*100:+.1f}%,"
                 f" {(g[h] < 0).mean()*100:.0f}% neg)" for h in HOR]
        print(f"    {lab:<9} n {len(g):>3}  " + "  ".join(cells))
    f0, f1 = ev[ev.d < RECENT][90].mean(), ev[ev.d >= RECENT][90].mean()
    verdict["FADE after market hype (90d < 0 both periods)"] = bool(
        f0 < 0 and f1 < 0)
    print("    by market (90d, both periods):  " + "  ".join(
        f"{k} {g[90].mean()*100:+.0f}% (n {len(g)})"
        for k, g in ev.groupby("sym")))

    print("\n  ===== TEST 2: SELL THE HYPE inside the two-way trend sleeve =====")
    base = TL.sleeve(px)
    orig = TL.TARGET
    blocked = {}
    me = px.resample("ME").last().index
    for t in me:
        blocked[t.to_period("M")] = {m for m, sp in sp_all.items()
                                     if any(t - pd.Timedelta(days=90) < d <= t
                                            for d in sp)}
    names = [n for n in TL.MKTS if n in px.columns]
    r = px.pct_change()
    vol = r.rolling(63, min_periods=50).std() * np.sqrt(252)
    out, prev = [], pd.Series(0.0, index=names)
    for k in range(len(me) - 1):
        t = px.index[px.index <= me[k]][-1]
        t2 = px.index[px.index <= me[k + 1]][-1]
        i = px.index.get_loc(t)
        if i < 252:
            continue
        r12 = px.iloc[i][names] / px.iloc[i - 252][names] - 1
        cash12 = px.iloc[i]["BIL"] / px.iloc[i - 252]["BIL"] - 1
        sign = np.where(r12 > cash12, 1.0, -1.0)
        w = (orig / vol.iloc[i][names] / len(names)).clip(upper=TL.CAP)
        w = (w * sign).fillna(0.0)
        for m in blocked.get(me[k].to_period("M"), set()):
            if m in w.index:
                w[m] = 0.0
        mm = px.loc[t2] / px.loc[t] - 1
        out.append({"p": t2.to_period("M"),
                    "ret": (w * (mm[names] - mm["BIL"])).sum()
                    - TL.COST * (w - prev).abs().sum()})
        prev = w
    over = pd.DataFrame(out).set_index("p")["ret"]
    mom = pd.read_csv(TS.MOMF, index_col=0, parse_dates=True)["STEADY"]
    mom.index = mom.index.to_period("M")
    idx = mom.index.intersection(base.index)
    books = {"MOM + TREND": mom.reindex(idx) + base["ret"].reindex(idx)
             - TL.SPREAD / 12,
             "MOM + TREND sell-hype": mom.reindex(idx) + over.reindex(idx)
             - TL.SPREAD / 12}
    res = {}
    for k_, b in books.items():
        s0, s1, al = TS.stats(b[b.index < TS.RECENT]), \
            TS.stats(b[b.index >= TS.RECENT]), TS.stats(b)
        res[k_] = (s0, s1)
        print(f"    {k_:<24} all {al[0]*100:+6.1f}% sh {al[2]:.2f} worst "
              f"{al[3]*100:.0f}% | 2017-21 {s0[0]*100:+6.1f}% sh {s0[2]:.2f} | "
              f"last5y {s1[0]*100:+6.1f}% sh {s1[2]:.2f}")
    (b0, b1), (o0, o1) = res["MOM + TREND"], res["MOM + TREND sell-hype"]
    verdict["SELL THE HYPE improves trend sleeve (Sharpe, both)"] = bool(
        o0[2] > b0[2] and o1[2] > b1[2])
    print("\n  ===== VERDICT (pre-stated; small samples) =====")
    for k_, v in verdict.items():
        print(f"  {k_:<52} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
