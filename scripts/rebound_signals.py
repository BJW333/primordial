"""
rebound_signals.py -- the INVERSE of the fade. After a good-news attention
spike most stocks fade, but about a third of the faded ones later rip
(FCEL +1,198%, OVV +639%, STMP +453%, CVNA, COHR, CLF...). Is there a
sign, AFTER the fade is detected, that a stock is about to go back up?

EVENTS: all good-news spikes, non-S&P liquid small + mid caps
(attention_rebound.py). FADE DETECTED = the stock is >= 10% behind its
same-size benchmark since the spike. Checked daily from day 42 to day 378
after the spike. Prices/flags use closes up to that day only.

REBOUND SIGNALS (on top of "fade detected"):
  K  EARN_BEAT    the latest quarterly earnings filed after the spike was a
                  big positive surprise (seasonal net-income surprise >= +1
                  std) -- the business is improving under the hype hangover
  X  MKT_WASHOUT  SPY is >= 15% below its 1-year high (the stock fell WITH
                  a market crash -- the March-2020 squeezers)
  T  TREND_BACK   6-month return above the same-size average AND close above
                  its 50-day average (strength is returning)
  C  CAPITULATE   down >= 50% from the spike-day close (fully washed out)
  Rules: every combination of K, X, T, C (15). Buy at the close AFTER the
  first day a rule is true; hold 126 trading days (6 months); return minus
  the same-size benchmark. Returns beyond +/-300% dropped (conservative
  for a long, and protects against stitched-ticker data errors).

PRE-STATED: pick the rule with the highest t among those with >= 30
trades on 2017-2021-05 spikes ONLY; PASS if on 2021-06+ spikes its mean
is > 0 with t > 2.5 (>= 30 trades). Baseline shown: buying every stock on
the first day the fade is detected.

Run (repo root; needs data/earnings_quarterly.csv.gz from fetch_earnings.py):
    python3.10 scripts/rebound_signals.py --smoke
    python3.10 scripts/rebound_signals.py | tee logs/rebound_signals.txt
"""
import argparse
import itertools
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
import blowup_us as U                 # noqa: E402
import factor_portfolio as FP         # noqa: E402

ROOT = os.path.dirname(HERE)
EARN = os.path.join(ROOT, "data", "earnings_quarterly.csv.gz")
SIGS = ["K", "X", "T", "C"]
NAMES = {"K": "EARN_BEAT", "X": "MKT_WASHOUT", "T": "TREND_BACK",
         "C": "CAPITULATE"}
RULES = [c for k in range(1, 5) for c in itertools.combinations(SIGS, k)]
LO, HI, HOLD, MINN = 42, 378, 126, 30


def load_earn():
    if not os.path.exists(EARN):
        print("  (no earnings file -- EARN_BEAT all False)")
        return {}
    import earnings_test as E
    e = pd.read_csv(EARN, dtype={"cik": str, "ticker": str},
                    parse_dates=["end", "filed"])
    s = E.surprises(e).dropna(subset=["ticker"])
    s["ticker"] = s["ticker"].astype(str).str.upper().str.replace(
        "-", ".", regex=False)
    s = s.sort_values("filed")
    return {k: (g["filed"].values, g["sue"].values) for k, g in
            s.groupby("ticker")}


def events(name, ev, px, bench, spy_dd, earn):
    idx = px.index
    cv = px.ffill().values
    cols = {c: j for j, c in enumerate(px.columns)}
    b = bench.reindex(idx).ffill().values
    ma50 = px.ffill().rolling(50, min_periods=40).mean().values
    dd = spy_dd.reindex(idx).ffill().values
    out = []
    for s, d in zip(ev["sym"], ev["d"]):
        p0 = idx.get_loc(d)
        if p0 + LO + HOLD + 2 >= len(idx):
            continue
        j = cols[s]
        c = cv[:, j]
        w = np.arange(p0 + LO, min(p0 + HI, len(idx) - HOLD - 2))
        if not len(w) or not np.isfinite(c[p0 + 1]):
            continue
        rel = c[w] / c[p0 + 1] - b[w] / b[p0 + 1]
        faded = np.nan_to_num(rel, nan=0) <= -0.10
        sig = {}
        f = earn.get(s)
        if f is not None:
            fd, su = f
            k = np.searchsorted(fd, idx[w].values, side="right") - 1
            ok = k >= 0
            kk = np.where(ok, k, 0)
            sig["K"] = ok & (fd[kk] > np.datetime64(d)) & (su[kk] >= 1)
        else:
            sig["K"] = np.zeros(len(w), bool)
        sig["X"] = np.nan_to_num(dd[w], nan=0) <= -0.15
        r6 = c[w] / c[w - 126] - 1 - (b[w] / b[w - 126] - 1)
        sig["T"] = (np.nan_to_num(r6, nan=-1) > 0) & (c[w] > ma50[w, j])
        sig["C"] = c[w] <= 0.5 * c[p0]
        for k_ in sig:
            sig[k_] = sig[k_] & faded

        def fwd(i):
            if not (np.isfinite(c[i]) and np.isfinite(c[i + HOLD])):
                return np.nan
            return c[i + HOLD] / c[i] - b[i + HOLD] / b[i]
        out.append({"d": d, "w": w, "sig": sig, "faded": faded, "fwd": fwd})
    return out


def trades(evs, rule):
    rows = []
    for e in evs:
        if rule is None:
            m = e["faded"]
        else:
            m = np.ones(len(e["w"]), bool)
            for k in rule:
                m &= e["sig"][k]
        hit = np.flatnonzero(m)
        if not len(hit):
            continue
        r = e["fwd"](e["w"][hit[0]] + 1)
        if np.isfinite(r) and abs(r) < 3:
            rows.append((e["d"], r))
    return pd.DataFrame(rows, columns=["d", "r"])


def line(tr):
    if len(tr) < 5:
        return len(tr), np.nan, np.nan, np.nan, np.nan
    return (len(tr), tr["r"].mean(), tr["r"].median(), (tr["r"] > 0).mean(),
            AR.tstat(tr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        px, vol, views, unis = AR.smoke()
        spy = px.mean(axis=1)
        earn = {}
    else:
        views = AS.norm_index(pd.read_parquet(AS.ATT))
        sp_members, _ = FP.membership()
        px, vol, unis = AR.universes(sp_members)
        spy = AS.norm_index(pd.read_parquet(
            os.path.join(U.DIR, "spy.parquet")))["SPY"]
        earn = load_earn()
        print(f"  earnings-surprise histories for {len(earn)} tickers")
    spy_dd = spy / spy.rolling(252, min_periods=200).max() - 1
    evs = []
    for name, (mem, bench) in unis.items():
        evs += events(name, AR.good_spikes(views, px, mem), px, bench,
                      spy_dd, earn)
    old = [e for e in evs if e["d"] < AR.RECENT]
    new = [e for e in evs if e["d"] >= AR.RECENT]
    print(f"\n  {len(evs)} spikes (small + mid); {len(old)} in 2017-21, "
          f"{len(new)} 2021-06+")
    print(f"\n  {'rule':<36} | {'2017-21 n':>9} {'mean':>7} {'med':>7} "
          f"{'win':>4} {'t':>6} | {'2021+ n':>7} {'mean':>7} {'med':>7} "
          f"{'win':>4} {'t':>6}")
    sel = []
    for rule in [None] + RULES:
        lab = "BASELINE: fade detected" if rule is None else \
            "+".join(NAMES[k] for k in rule)
        s0, s1 = line(trades(old, rule)), line(trades(new, rule))
        if rule is not None:
            sel.append((rule, s0, s1))

        def f(s):
            return (f"{s[0]:>7} {s[1]*100:>+6.2f}% {s[2]*100:>+6.2f}% "
                    f"{s[3]*100:>3.0f}% {s[4]:>+6.2f}")
        print(f"  {lab:<36} | {f(s0):>40} | {f(s1)}")
    ok = [x for x in sel if x[1][0] >= MINN and np.isfinite(x[1][4])]
    if not ok:
        print("  no rule with >= 30 trades in 2017-21")
        return
    rule, s0, s1 = max(ok, key=lambda x: x[1][4])
    lab = " + ".join(NAMES[k] for k in rule)
    p = bool(s1[0] >= MINN and s1[1] > 0 and s1[4] > 2.5)
    print(f"\n  SELECTED on 2017-21 (highest t): {lab} (n {s0[0]}, "
          f"{s0[1]*100:+.2f}%, t {s0[4]:+.2f})")
    print(f"  2021-06+: n {s1[0]}, {s1[1]*100:+.2f}% vs benchmark, t "
          f"{s1[4]:+.2f} -> {'PASS' if p else 'FAIL'}")
    print("\n  ===== VERDICT (pre-stated) =====")
    if a.smoke:
        print("  smoke: plumbing check on synthetic data")
    print(f"  rebound signal ({lab})  {'PASS' if p else 'FAIL'}")


if __name__ == "__main__":
    main()
