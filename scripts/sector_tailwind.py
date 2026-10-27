"""
sector_tailwind.py -- rebound idea: faded stocks come back when their WHOLE
INDUSTRY catches a tailwind (OVV/TRGP with oil in 2021, CLF with steel),
not because of anything company-specific.

INDUSTRY MATCH (no outside data): for each spike, the stock is matched to
the industry ETF its daily returns moved with most closely over the 252
trading days BEFORE the spike (correlation >= 0.40 required, else no
industry). 26 ETFs: the 10 SPDR sectors (XLE XLB XLI XLK XLF XLV XLY XLP
XLU XLRE) + industries (XOP OIH XME SLX GDX KRE XBI SMH ITB IYT TAN XRT
JETS URA LIT KIE).

EVENTS: good-news attention spikes on non-S&P liquid small + mid caps;
FADE DETECTED = >= 10% behind the same-size benchmark since the spike;
checked daily from day 42 to day 378 (closes up to that day only).

SIGNALS (while faded):
  S  TAILWIND    the matched ETF beat SPY by >= 5% over the last 63 days
                 AND is above its 200-day average
  F  FOLLOWING   S, and the stock itself beat its benchmark over the last
                 21 days (it has started to move with its industry)
Buy at the close after the first signal day; hold 126 trading days;
return minus the same-size benchmark; trades beyond +/-300% dropped.

PRE-STATED: pick S or F by higher t (>= 30 trades) on 2017-2021-05
spikes; PASS if on 2021-06+ spikes its mean > 0 with t > 3.0.
Diagnostics (not judged): results by matched ETF.

Run (repo root, Alpaca keys exported -- downloads 27 ETF histories once):
    python3.10 scripts/sector_tailwind.py | tee logs/sector_tailwind.txt
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attention_rebound as AR        # noqa: E402
import attention_short_test as AS     # noqa: E402
import factor_portfolio as FP         # noqa: E402
import rebound_signals as RS          # noqa: E402

ETFS = ["XLE", "XLB", "XLI", "XLK", "XLF", "XLV", "XLY", "XLP", "XLU", "XLRE",
        "XOP", "OIH", "XME", "SLX", "GDX", "KRE", "XBI", "SMH", "ITB", "IYT",
        "TAN", "XRT", "JETS", "URA", "LIT", "KIE"]
LO, HI, HOLD = RS.LO, RS.HI, RS.HOLD
MINN, BAR, MINCORR = 30, 3.0, 0.40


def load_etfs(idx):
    out = {}
    for s in ETFS + ["SPY"]:
        df = FP.alpaca(s)
        if df is None or not len(df):
            print(f"  (no data for {s})")
            continue
        df = AS.norm_index(df)
        col = "close" if "close" in df.columns else df.columns[0]
        out[s] = df[col]
    e = pd.DataFrame(out).reindex(idx).ffill()
    print(f"  ETFs loaded: {e.shape[1] - 1} industries + SPY, "
          f"{e.index[0]:%Y-%m-%d} -> {e.index[-1]:%Y-%m-%d}")
    return e


def events(ev, px, bench, etf):
    idx = px.index
    cv = px.ffill().values
    cols = {c: j for j, c in enumerate(px.columns)}
    b = bench.reindex(idx).ffill().values
    names = [c for c in etf.columns if c != "SPY"]
    E = etf[names].values
    spy = etf["SPY"].values
    er = np.diff(np.log(E), axis=0)
    rel63 = (E[63:] / E[:-63]) - (spy[63:] / spy[:-63])[:, None]
    rel63 = np.vstack([np.full((63, len(names)), np.nan), rel63])
    ma200 = etf[names].rolling(200, min_periods=180).mean().values
    tail = (np.nan_to_num(rel63, nan=-1) >= 0.05) & (E > ma200)
    out = []
    for s, d in zip(ev["sym"], ev["d"]):
        p0 = idx.get_loc(d)
        if p0 < 260 or p0 + LO + HOLD + 2 >= len(idx):
            continue
        c = cv[:, cols[s]]
        sr = np.diff(np.log(c[p0 - 253:p0]))
        er_w = er[p0 - 253:p0 - 1]
        ok = np.isfinite(sr)
        if ok.sum() < 200:
            continue
        cors = []
        for k in range(len(names)):
            x = er_w[:, k]
            m = ok & np.isfinite(x)
            cors.append(np.corrcoef(sr[m], x[m])[0, 1] if m.sum() > 150
                        else np.nan)
        cors = np.array(cors)
        if not np.isfinite(cors).any() or np.nanmax(cors) < MINCORR:
            k = -1
        else:
            k = int(np.nanargmax(cors))
        w = np.arange(p0 + LO, min(p0 + HI, len(idx) - HOLD - 2))
        if not len(w) or not np.isfinite(c[p0 + 1]):
            continue
        rel = c[w] / c[p0 + 1] - b[w] / b[p0 + 1]
        faded = np.nan_to_num(rel, nan=0) <= -0.10
        S = tail[w, k] & faded if k >= 0 else np.zeros(len(w), bool)
        r21 = c[w] / c[w - 21] - b[w] / b[w - 21]
        F = S & (np.nan_to_num(r21, nan=-1) > 0)

        def fwd(i, c=c):
            if not (np.isfinite(c[i]) and np.isfinite(c[i + HOLD])):
                return np.nan
            return c[i + HOLD] / c[i] - b[i + HOLD] / b[i]
        out.append({"d": d, "w": w, "faded": faded, "fwd": fwd,
                    "sig": {"S": S, "F": F},
                    "etf": names[k] if k >= 0 else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, views, unis = AR.smoke()
        rng = np.random.default_rng(3)
        base = px.iloc[:, :27].values
        etf = pd.DataFrame(base * np.exp(rng.normal(0, 0.002, base.shape)),
                           px.index, ETFS + ["SPY"])
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = AR.universes(sp_members)
        etf = load_etfs(px.index)
    evs = []
    for name, (mem, bench) in unis.items():
        evs += events(AR.good_spikes(views, px, mem), px, bench, etf)
    old = [e for e in evs if e["d"] < AR.RECENT]
    new = [e for e in evs if e["d"] >= AR.RECENT]
    m = pd.Series([e["etf"] or "none" for e in evs]).value_counts()
    print(f"\n  {len(evs)} spikes; {len(old)} in 2017-21, {len(new)} 2021-06+")
    print("  industry matches: " + ", ".join(f"{k} {v}" for k, v in m.items()))
    rows = []
    print(f"\n  {'rule':<28} | {'2017-21 n':>9} {'mean':>7} {'med':>7} "
          f"{'win':>4} {'t':>6} | {'2021+ n':>7} {'mean':>7} {'med':>7} "
          f"{'win':>4} {'t':>6}")
    for rule, lab in ((None, "BASELINE: fade detected"),
                      (("S",), "TAILWIND"), (("F",), "TAILWIND+FOLLOWING")):
        s0, s1 = RS.line(RS.trades(old, rule)), RS.line(RS.trades(new, rule))
        if rule:
            rows.append((lab, s0, s1))

        def f(s):
            return (f"{s[0]:>7} {s[1]*100:>+6.2f}% {s[2]*100:>+6.2f}% "
                    f"{s[3]*100:>3.0f}% {s[4]:>+6.2f}")
        print(f"  {lab:<28} | {f(s0):>40} | {f(s1)}")
    print("\n  by matched industry (TAILWIND trades, both periods, diagnostic):")
    for k in m.index:
        sub = [e for e in evs if (e["etf"] or "none") == k]
        tr = RS.trades(sub, ("S",))
        if len(tr) >= 10:
            print(f"    {k:<5} n {len(tr):>4} mean {tr['r'].mean()*100:+6.2f}%"
                  f" median {tr['r'].median()*100:+6.2f}% win "
                  f"{(tr['r'] > 0).mean()*100:.0f}%")
    ok = [x for x in rows if x[1][0] >= MINN and np.isfinite(x[1][4])]
    if not ok:
        print("\n  not enough trades in 2017-21 -> FAIL")
        return
    lab, s0, s1 = max(ok, key=lambda x: x[1][4])
    p = bool(s1[0] >= MINN and s1[1] > 0 and s1[4] > BAR)
    print(f"\n  SELECTED on 2017-21: {lab} (n {s0[0]}, {s0[1]*100:+.2f}%, "
          f"t {s0[4]:+.2f})")
    print(f"  2021-06+: n {s1[0]}, {s1[1]*100:+.2f}% vs benchmark, t "
          f"{s1[4]:+.2f} (bar t > {BAR}) -> {'PASS' if p else 'FAIL'}")
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    print(f"  sector tailwind rebound ({lab})  {'PASS' if p else 'FAIL'}")


if __name__ == "__main__":
    main()
