"""
ZERO-TRIAL event study: index DELETION flow.

Mechanism: on the effective date every fund tracking the index MUST sell
the deleted name -- size known, date known, price-insensitive, and (unlike
additions) the literature on the deletion side in smaller caps is thin
because PIT membership is hard to get. We have it.

Events come from a PIT membership CSV (date,tickers -- the
manifests/sp500_pit_membership.csv format). A deletion = present on one
membership date, absent on the next; effective date = the date it is first
absent. Works unchanged on an sp600/sp400 PIT export from Norgate later --
that is the real target; sp500 is the free first pass.

CONTAMINATION HANDLED, stated up front:
  - M&A: most large-cap deletions are acquisitions; price is pinned at the
    deal price and there is nothing to revert. EXCLUDED by requiring the
    name to keep trading >= 55 sessions after the effective date.
  - SURVIVORSHIP (the one that cannot be fixed for free): names must still
    be fetchable from yfinance TODAY to enter the study, so bankrupt/
    delisted-after-deletion paths are missing and every post-deletion
    number is biased UP. Coverage is printed; treat the level as an upper
    bound. The Norgate export (delisted histories included) removes this
    bias and is required before any gauntlet talk.

Design mirrors the house screen pattern: events split 70/30 by effective
date; the tail is withheld; ONE frozen primary cell; --tail-look is the
one look. Nothing logged; no ledger trial spent.

PRIMARY CELL (frozen before any table prints): distress deletions --
trailing 126d total return <= -20% at T-1 -- BUY the close of the
effective day T, hold 60 sessions, SPY-adjusted, net --cost-bps.
PASS: mean abnormal return > 0 AND t >= 2 on train events.
Everything else printed (the full CAR path, the non-distress rows, the
pre-event window) is descriptive context and cannot be promoted.

    python3.10 scripts/explore_index_deletion.py \\
        --membership manifests/sp500_pit_membership.csv
    # later, ONLY if train passed, ONE invocation ever:
    python3.10 scripts/explore_index_deletion.py \\
        --membership manifests/sp500_pit_membership.csv --tail-look
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primordial.data import fetch  # noqa: E402

TRAIN_FRAC = 0.70
HOLD = 60                # sessions, frozen
DISTRESS = -0.20         # trailing 126d total return cutoff, frozen
OFFSETS = [-20, -10, -5, -1, 0, 5, 10, 20, 40, 60]


def deletions(path):
    m = pd.read_csv(path)
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date").reset_index(drop=True)
    ev = []
    prev = None
    for _, r in m.iterrows():
        cur = set(str(r["tickers"]).split(","))
        if prev is not None:
            for sym in prev - cur:
                ev.append((r["date"], sym.strip()))
        prev = cur
    return pd.DataFrame(ev, columns=["eff", "sym"])


def tstat(x):
    x = pd.Series(x).dropna()
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return (len(x), x.mean(),
            x.mean() / (sd / len(x) ** 0.5) if sd > 0 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--membership", required=True)
    p.add_argument("--source", default="yfinance")
    p.add_argument("--start", default="2009-01-01")
    p.add_argument("--end", default="2026-06-01")
    p.add_argument("--cost-bps", type=float, default=14.0)
    p.add_argument("--tail-look", action="store_true",
                   help="THE one look: frozen primary cell on the withheld "
                        "last 30%% of events. One invocation ever.")
    a = p.parse_args()
    cost = a.cost_bps / 1e4

    ev = deletions(a.membership)
    if not len(ev):
        raise SystemExit("no deletions found in membership file")
    ev = ev.sort_values("eff").reset_index(drop=True)
    cut = ev["eff"].iloc[int(len(ev) * TRAIN_FRAC)]
    print(f"membership {a.membership}: {len(ev)} deletion events "
          f"{ev['eff'].min().date()} .. {ev['eff'].max().date()} | "
          f"train < {cut.date()} | cost {a.cost_bps:.0f} bps")

    spy = fetch(a.source, ["SPY"], "1d", a.start, a.end, min_bars=300).get(
        "SPY")
    if spy is None:
        raise SystemExit("no SPY series")
    spyc = spy["close"]

    data = fetch(a.source, sorted(set(ev["sym"])), "1d", a.start, a.end,
                 min_bars=200)
    print(f"price coverage: {len(data)}/{ev['sym'].nunique()} names fetched "
          f"-- missing names are mostly delisted-after-deletion paths; "
          f"every number below is biased UP by their absence.")

    rows = []
    for eff, sym in ev.itertuples(index=False):
        df = data.get(sym)
        if df is None:
            rows.append({"eff": eff, "sym": sym, "ok": False})
            continue
        c = df["close"]
        idx = c.index.searchsorted(eff)
        # effective date = first date OUT; T = last session at/after eff-1
        # trading day boundary: use the first session index >= eff as T0,
        # entry at close of T0 (the forced-sell day close).
        if idx >= len(c):
            rows.append({"eff": eff, "sym": sym, "ok": False})
            continue
        t0 = idx
        if t0 < 130 or t0 + HOLD + 1 > len(c):
            rows.append({"eff": eff, "sym": sym, "ok": False})
            continue
        trail = c.iloc[t0 - 1] / c.iloc[t0 - 127] - 1.0
        si = spyc.index.searchsorted(c.index[t0])
        if si + HOLD + 1 > len(spyc) or si < 25:
            rows.append({"eff": eff, "sym": sym, "ok": False})
            continue
        r = {"eff": eff, "sym": sym, "ok": True, "trail": trail}
        for off in OFFSETS:
            if off == 0:
                continue
            if off < 0:
                r[f"car{off}"] = (c.iloc[t0] / c.iloc[t0 + off] - 1.0) - \
                                 (spyc.iloc[si] / spyc.iloc[si + off] - 1.0)
            else:
                r[f"car{off}"] = (c.iloc[t0 + off] / c.iloc[t0] - 1.0) - \
                                 (spyc.iloc[si + off] / spyc.iloc[si] - 1.0)
        rows.append(r)
    d = pd.DataFrame(rows)
    ok = d[d["ok"] == True].copy()          # noqa: E712
    print(f"scorable events (traded >= {HOLD} sessions post, full windows): "
          f"{len(ok)}/{len(d)}")
    if not len(ok):
        raise SystemExit("nothing scorable")
    ok["distress"] = ok["trail"] <= DISTRESS

    if a.tail_look:
        seg = ok[ok["eff"] >= cut]
        cell = seg[seg["distress"]]
        n, m, t = tstat(cell["car60"] - cost)
        print(f"===== TAIL LOOK (one look, frozen cell) | events >= "
              f"{cut.date()} =====")
        if n < 15:
            print(f"  n={n} -- unscorable at this event count. Record; the "
                  f"sp600 Norgate export is the real sample.")
            return
        print(f"  distress deletions, buy close T, hold 60d, SPY-adj: "
              f"{m*100:+.2f}% [t {t:+.1f}] n={n}")
        print("  VERDICT vs frozen rule: "
              + ("PASS -- next: Norgate PIT export (delisted histories) to "
                 "kill the survivorship bias, THEN a gauntlet family."
                 if m > 0 and t >= 2 else "DEAD. Seam closed."))
        return

    tr = ok[ok["eff"] < cut]
    print(f"\nTRAIN: {len(tr)} events ({int(tr['distress'].sum())} distress "
          f"at trail<= {DISTRESS:.0%}, {len(tr) - int(tr['distress'].sum())}"
          f" other)")

    def block(name, g):
        print(f"\n  [{name}] n={len(g)}  SPY-adjusted CAR from close(T), "
              f"mean [t]")
        pre = " ".join(f"T{o:+d}:{tstat(g[f'car{o}'])[1]*100:+.1f}%"
                       for o in OFFSETS if o < 0)
        print(f"    into the event: {pre}")
        for o in [5, 10, 20, 40, 60]:
            n, m, t = tstat(g[f"car{o}"])
            tag = "  <= PRIMARY (net shown)" if (o == 60 and
                                                 name == "distress") else ""
            if o == 60 and name == "distress":
                n, m, t = tstat(g["car60"] - cost)
            print(f"    T -> +{o:<3d} {m*100:+.2f}% [t {t:+5.1f}]{tag}")

    block("distress", tr[tr["distress"]])
    block("non-distress", tr[~tr["distress"]])
    print("\nREAD (frozen in header): PRIMARY = distress row, T->+60 net, "
          "mean > 0 AND t >= 2. Pass earns ONE --tail-look. Fail = closed "
          "on this ground; the sp600 Norgate export may still be run ONCE "
          "as its own pre-stated study (different index, different flow "
          "size), not as a retry of this one.")


if __name__ == "__main__":
    main()
