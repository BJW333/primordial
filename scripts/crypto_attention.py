"""
crypto_attention.py -- the FADE idea on crypto, long-only.
Crypto tops are retail-hype tops: does a Wikipedia attention spike during a
big rally mark a top, and does buying back after the hype crash pay?

COINS (Wikipedia article(s) -> Yahoo price; several titles are summed so
renamed articles keep their history): Bitcoin, Ethereum, Dogecoin,
Litecoin, Cardano, Solana, XRP, Shiba Inu.

HYPE SPIKE (rules fixed before running): daily views >= 3x the prior-90-day
median AND the coin up >= 20% over the last 30 days; first per coin per 60
days; views used with a 1-day lag (act at the NEXT day's close).

TEST 1  FADE: coin return 30 / 90 / 180 days after the spike, relative to
        that coin's typical 30 / 90 / 180-day return (average LOG return, so
        a meme coin's 1,000,000x run can't distort "normal").
TEST 2  SELL THE HYPE (overlay on the crypto_sleeve.py BTC/ETH trend sleeve):
        same sleeve, but a coin is forced to cash for 90 days after its hype
        spike. Then 80% momentum / 20% sleeve, as in crypto_sleeve.py.
TEST 3  REBOUND (Blake's idea): after a spike, buy the first day the coin is
        >= 40% below the spike-day close AND 7-day views are back <= 1.5x
        normal; hold 180 days; vs the coin's normal 180-day return.

PRE-STATED (2017-01 -> 2021-05 vs 2021-06 -> 2026-05):
  FADE  : 90-day excess return < 0 in BOTH periods (pooled over coins)
  SELL  : overlay sleeve beats the plain sleeve on Sharpe in BOTH periods
  REBOUND: 180-day excess return > 0 in BOTH periods, >= 5 trades each
Small samples (a few dozen spikes) -- treat any pass as promising, not proven.

Run (repo root, Alpaca keys exported for BIL):
    python3.10 scripts/crypto_attention.py --email you@example.com \
        | tee logs/crypto_attention.txt
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
import crypto_sleeve as CS            # noqa: E402
import fetch_attention_short as FA    # noqa: E402

ROOT = os.path.dirname(HERE)
OUTD = os.path.join(ROOT, "data", "attention", "crypto")
COINS = {
    "BTC-USD": ["Bitcoin"],
    "ETH-USD": ["Ethereum"],
    "DOGE-USD": ["Dogecoin"],
    "LTC-USD": ["Litecoin"],
    "ADA-USD": ["Cardano_(blockchain_platform)", "Cardano_(cryptocurrency)"],
    "SOL-USD": ["Solana_(blockchain_platform)", "Solana_(cryptocurrency)"],
    "XRP-USD": ["XRP_Ledger", "Ripple_(payment_protocol)", "XRP"],
    "SHIB-USD": ["Shiba_Inu_(cryptocurrency)"],
}
RECENT = pd.Timestamp("2021-06-01")
START = pd.Timestamp("2017-01-01")
HOR = (30, 90, 180)


def views(s, coin):
    os.makedirs(OUTD, exist_ok=True)
    p = os.path.join(OUTD, coin + ".csv")
    if os.path.exists(p):
        return pd.read_csv(p, index_col=0, parse_dates=True)["views"]
    end = pd.Timestamp.today().strftime("%Y%m%d")
    tot = None
    for art in COINS[coin]:
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
    r30 = px / px.shift(30) - 1
    hit = px.index[(ratio >= 3) & (r30 >= 0.20)]
    out, last = [], None
    for d in hit:
        if d < START or (last is not None and (d - last).days < 60):
            continue
        last = d
        out.append(d)
    return out, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    a = ap.parse_args()
    s = FA.sess(a.email)
    ev_rows, reb_rows, spike_days = [], [], {}
    for coin in COINS:
        try:
            px = CS.coin(coin)
        except Exception as e:  # noqa: BLE001
            print(f"  {coin}: no price ({e})")
            continue
        px = px[~px.index.duplicated()].asfreq("D").ffill()
        px = px[px > 0]
        lp = np.log(px)
        v = views(s, coin)
        if not len(v):
            print(f"  {coin}: no Wikipedia views")
            continue
        sp, base = spikes(px, v)
        spike_days[coin] = sp
        # normal drift in LOG terms (a typical-period return, not a mean
        # blown up by 1,000,000x meme-coin runs)
        normal = {h: (lp.shift(-h) - lp).mean() for h in HOR}

        def exc(i, h):
            return np.exp(lp.iloc[i + h] - lp.iloc[i] - normal[h]) - 1
        v7 = v.asfreq("D").fillna(0).rolling(7).mean()
        print(f"  {coin}: {len(sp)} hype spikes " +
              ", ".join(f"{d:%Y-%m-%d}" for d in sp))
        for d in sp:
            i = px.index.get_loc(d) + 1                 # act next day
            row = {"coin": coin, "d": d}
            for h in HOR:
                row[h] = exc(i, h) if i + h < len(px) else np.nan
            ev_rows.append(row)
            c0 = px.iloc[i - 1]
            for j in range(i, min(i + 540, len(px) - 181)):
                dj = px.index[j]
                b = base.get(d, np.nan)
                if px.iloc[j] <= 0.6 * c0 and np.isfinite(b) and \
                        v7.get(dj, np.inf) <= 1.5 * b:
                    k = j + 1
                    reb_rows.append({"coin": coin, "d": d, "buy": px.index[k],
                                     "r": exc(k, 180)})
                    break
    ev = pd.DataFrame(ev_rows)
    rb = pd.DataFrame(reb_rows)
    verdict = {}

    print("\n  ===== TEST 1: FADE after hype (vs the coin's normal drift) =====")
    for lab, g in (("2017-21", ev[ev.d < RECENT]), ("2021-06+", ev[ev.d >= RECENT])):
        cells = [f"+{h}d {g[h].mean()*100:+.1f}% (med {g[h].median()*100:+.1f}%,"
                 f" {(g[h] < 0).mean()*100:.0f}% neg)" for h in HOR]
        print(f"    {lab:<9} n {len(g):>3}  " + "  ".join(cells))
    f0 = ev[ev.d < RECENT][90].mean()
    f1 = ev[ev.d >= RECENT][90].mean()
    verdict["FADE after crypto hype (90d < 0 both periods)"] = bool(
        f0 < 0 and f1 < 0)

    print("\n  ===== TEST 2: SELL THE HYPE (BTC/ETH trend sleeve overlay) =====")
    sd, btc, cash_d, held = CS.sleeve_daily()
    pxs = pd.DataFrame({c: CS.coin(c) for c in CS.COINS}).sort_index()
    pxs = pxs[~pxs.index.duplicated()].reindex(held.index)
    block = pd.DataFrame(False, held.index, held.columns)
    for c in held.columns:
        for d in spike_days.get(c, []):
            block.loc[(block.index > d) & (block.index <= d +
                                           pd.Timedelta(days=90)), c] = True
    held2 = held.where(~block, 0.0)
    r = pxs.pct_change().fillna(0)
    trade = held2.diff().abs().sum(axis=1).fillna(0)
    sd2 = (held2 * r).sum(axis=1) + (1 - held2.sum(axis=1)) * cash_d - \
        CS.COST * trade
    m = pd.read_csv(CS.MOMF, index_col=0, parse_dates=True)["STEADY"]
    m.index = m.index.to_period("M")
    s1, s2, cm = CS.monthly(sd), CS.monthly(sd2), CS.monthly(cash_d)
    idx = m.index.intersection(s1.index)
    rows = {"sleeve (plain)": s1.reindex(idx),
            "sleeve SELL-THE-HYPE": s2.reindex(idx),
            "MOM 80 + plain 20": 0.8 * m.reindex(idx) + 0.2 * s1.reindex(idx),
            "MOM 80 + SELL 20": 0.8 * m.reindex(idx) + 0.2 * s2.reindex(idx)}
    rp = pd.Period("2021-06", "M")
    res = {}
    for k, b in rows.items():
        a0, s0_ = CS.stats(b[b.index < rp]), CS.stats(b[b.index >= rp])
        al = CS.stats(b)
        res[k] = (a0, s0_)
        print(f"    {k:<22} all {al[0]*100:+6.1f}% sh {al[2]:.2f} worst "
              f"{al[3]*100:.0f}% | 2017-21 {a0[0]*100:+6.1f}% sh {a0[2]:.2f} "
              f"| last5y {s0_[0]*100:+6.1f}% sh {s0_[2]:.2f} worst "
              f"{s0_[3]*100:.0f}%")
    (p0, p1), (q0, q1) = res["sleeve (plain)"], res["sleeve SELL-THE-HYPE"]
    verdict["SELL THE HYPE beats plain sleeve (Sharpe, both)"] = bool(
        q0[2] > p0[2] and q1[2] > p1[2])

    print("\n  ===== TEST 3: REBOUND -- buy after the hype crash, hold 180d =====")
    if len(rb):
        for _, x in rb.iterrows():
            print(f"    {x.coin:<9} spike {x.d:%Y-%m-%d} -> bought "
                  f"{x.buy:%Y-%m-%d}: {x.r*100:+.0f}% vs normal 180d")
        g0, g1 = rb[rb.d < RECENT], rb[rb.d >= RECENT]
        print(f"    2017-21 n {len(g0)} mean {g0.r.mean()*100:+.1f}% | "
              f"2021-06+ n {len(g1)} mean {g1.r.mean()*100:+.1f}%")
        verdict["REBOUND after hype crash (>0 both, >=5 each)"] = bool(
            len(g0) >= 5 and len(g1) >= 5 and g0.r.mean() > 0 and
            g1.r.mean() > 0)
    else:
        print("    no rebound entries")
        verdict["REBOUND after hype crash (>0 both, >=5 each)"] = False

    print("\n  ===== VERDICT (pre-stated; small samples) =====")
    for k, v in verdict.items():
        print(f"  {k:<50} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
