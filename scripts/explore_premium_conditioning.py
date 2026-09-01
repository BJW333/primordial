"""
ZERO-TRIAL conditioning screen: WHEN is the dip-absorption premium fat?

The premium itself is settled: six measurements on this ledger, +0.2 to
+0.6, never enough to clear an honest bar UNCONDITIONALLY. Three calendar
mechanisms predict the premium concentrates when a specific non-price
seller/missing-buyer is active. One base event set, three frozen cells.

BASE EVENT (fixed): fresh 21-day closing low on a us_sp600 manifest name
(daily cache). Entry next open; fwd10/fwd20 = close 10/20 sessions later,
SPY-adjusted, net --cost-bps. DISTRESS = trailing 126d return <= -30%.

FROZEN CELLS (before any table prints; everything else is context):
  C1 BUYBACK BLACKOUT (proxy): corporates -- the largest standing bid --
     are sidelined ~5 weeks pre-earnings. Proxy windows, stated crude and
     frozen: Mar16-Apr15, Jun16-Jul15, Sep16-Oct15, Dec16-Jan15.
     PASS: fwd10(blackout) - fwd10(outside) > 0 with Welch t >= 2.
  C2 TAX-LOSS DEADLINE: distress events dated Dec 1-24, fwd20 (spans the
     January re-entry). PASS: mean net > 0, t >= 2, AND mean exceeds
     non-December distress fwd20.
  C3 QUARTER-END DRESSING: distress events in the LAST 5 SESSIONS of
     Mar/Jun/Sep/Dec (reporting-date optics selling).
     PASS: fwd10 - (distress fwd10 elsewhere) > 0 with Welch t >= 2.

Train = first 70% of event dates; tail withheld; --tail-look C1|C2|C3 is
the one look each. Nothing logged; no ledger trial spent.

    python3.10 scripts/explore_premium_conditioning.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primordial.data import fetch  # noqa: E402

TRAIN_FRAC = 0.70
DISTRESS = -0.30


def load_syms(manifest):
    d = yaml.safe_load(open(manifest))
    sf = os.path.join(os.path.dirname(manifest), d["symbols_file"])
    syms = [x.strip() for x in open(sf) if x.strip() and not
            x.startswith("#")]
    return syms, d["source"], d["start"], d["end"]


def welch(x, y):
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    if len(x) < 20 or len(y) < 20:
        return len(x), len(y), np.nan, np.nan
    d = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return len(x), len(y), d, d / se if se > 0 else np.nan


def tstat(x):
    x = pd.Series(x).dropna()
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return len(x), x.mean(), (x.mean() / (sd / len(x) ** 0.5)
                              if sd > 0 else 0.0)


def in_blackout(ts):
    m, day = ts.month, ts.day
    return ((m in (3, 6, 9, 12) and day >= 16) or
            (m in (1, 4, 7, 10) and day <= 15))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="manifests/us_sp600.yaml")
    p.add_argument("--cost-bps", type=float, default=36.0,
                   help="RT; equity_smallcap adverse is 18bps x 2")
    p.add_argument("--max-names", type=int, default=0)
    p.add_argument("--tail-look", choices=["C1", "C2", "C3"])
    a = p.parse_args()
    cost = a.cost_bps / 1e4

    syms, source, start, end = load_syms(a.manifest)
    if a.max_names:
        syms = syms[:a.max_names]
    data = fetch(source, syms, "1d", start, end, min_bars=300)
    spy = fetch(source, ["SPY"], "1d", start, end, min_bars=300)["SPY"]["close"]
    print(f"{a.manifest}: {len(data)}/{len(syms)} names | cost "
          f"{a.cost_bps:.0f} bps RT")

    rows = []
    for sym, df in data.items():
        c, o = df["close"], df["open"]
        low21 = c.rolling(21).min().shift(1)
        trail = c.shift(1) / c.shift(127) - 1.0
        sig = c <= low21
        idxs = np.flatnonzero(sig.values)
        last = -99
        for i in idxs:
            if i - last < 5:            # one event per 5 sessions per name
                continue
            if i < 130 or i + 21 >= len(c):
                continue
            last = i
            t = c.index[i]
            si = spy.index.searchsorted(t)
            if si + 21 >= len(spy) or si < 1:
                continue
            e = float(o.iloc[i + 1])
            if not e > 0:
                continue
            f10 = float(c.iloc[i + 10]) / e - 1 - \
                (float(spy.iloc[si + 10]) / float(spy.iloc[si]) - 1)
            f20 = float(c.iloc[i + 20]) / e - 1 - \
                (float(spy.iloc[si + 20]) / float(spy.iloc[si]) - 1)
            rows.append({"sym": sym, "date": t, "trail": float(trail.iloc[i]),
                         "f10": f10 - cost, "f20": f20 - cost})
    ev = pd.DataFrame(rows).dropna(subset=["trail"])
    ev["distress"] = ev["trail"] <= DISTRESS
    ev["bo"] = [in_blackout(t) for t in ev["date"]]
    ev["dec"] = [(t.month == 12 and t.day <= 24) for t in ev["date"]]
    # last-5-sessions-of-quarter flag from each name's own trading calendar
    qflags = {}
    for sym, df in data.items():
        idx = pd.DatetimeIndex(df.index)
        per = idx.to_period("Q")
        lastpos = pd.Series(np.arange(len(idx)), index=idx).groupby(
            per).transform("max")
        pos = pd.Series(np.arange(len(idx)), index=idx)
        qflags[sym] = ((lastpos - pos) < 5) & idx.month.isin(
            [3, 6, 9, 12]).astype(bool)
    ev["qe"] = [bool(qflags[s].get(t, False))
                for s, t in zip(ev["sym"], ev["date"])]

    ev = ev.sort_values("date").reset_index(drop=True)
    cut = ev["date"].iloc[int(len(ev) * TRAIN_FRAC)]
    tr = ev[ev["date"] < cut]
    seg = ev[ev["date"] >= cut]

    def report(fr, tag):
        print(f"\n===== {tag}: {len(fr):,} events "
              f"({int(fr['distress'].sum()):,} distress) =====")
        # C1
        n1, n0, d, t = welch(fr.loc[fr["bo"], "f10"], fr.loc[~fr["bo"], "f10"])
        print(f"  [C1 blackout] fwd10 in {fr.loc[fr['bo'],'f10'].mean()*100:+.2f}% "
              f"(n={n1:,}) vs out {fr.loc[~fr['bo'],'f10'].mean()*100:+.2f}% "
              f"(n={n0:,}) | diff {d*100:+.2f}% [Welch t {t:+.1f}]"
              f"  frozen pass: diff>0, t>=2")
        # C2
        dd = fr[fr["distress"]]
        cell = dd[dd["dec"]]
        n, m, tt = tstat(cell["f20"])
        base = dd.loc[~dd["dec"], "f20"].mean()
        print(f"  [C2 tax-loss] Dec1-24 distress fwd20 {m*100:+.2f}% "
              f"[t {tt:+.1f}] n={n} vs non-Dec distress {base*100:+.2f}%"
              f"  frozen pass: mean>0, t>=2, > non-Dec")
        # C3
        n1, n0, d, t = welch(dd.loc[dd["qe"], "f10"], dd.loc[~dd["qe"], "f10"])
        print(f"  [C3 qtr-end]  distress fwd10 last-5-sess "
              f"{dd.loc[dd['qe'],'f10'].mean()*100 if n1 else float('nan'):+.2f}% (n={n1:,}) vs "
              f"else {dd.loc[~dd['qe'],'f10'].mean()*100:+.2f}% (n={n0:,}) | "
              f"diff {d*100:+.2f}% [Welch t {t:+.1f}]  frozen pass: diff>0, t>=2")

    if a.tail_look:
        print(f"===== TAIL LOOK {a.tail_look} (one look) | events >= "
              f"{cut.date()} =====")
        report(seg, "TAIL (frozen cells only -- read ONLY your cell)")
        return
    print(f"TRAIN < {cut.date()} (tail withheld)")
    report(tr, "TRAIN")
    print("\nREAD: each cell judged ONLY by its frozen rule. A pass earns "
          "ONE --tail-look for that cell. Context numbers cannot be "
          "promoted. All three flat = the premium is thin everywhere, "
          "conditioning seam closed.")


if __name__ == "__main__":
    main()
