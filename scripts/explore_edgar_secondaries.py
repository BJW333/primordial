"""
ZERO-TRIAL event study: SECONDARY OFFERINGS from EDGAR (free, timestamped).

Mechanism: a follow-on offering is a contractual supply shock -- size
known, priced overnight at a discount because a book must clear NOW. The
discount overshoots, then digests. Events come from SEC EDGAR full-index
files (424B4/424B5 prospectuses), a free federal timestamped event log --
this also builds the event-time machinery the analyst-target leg needs.

Two subcommands:
  fetch  -- download quarterly form indexes 2010->present from sec.gov,
            keep 424B4/424B5, map CIK->ticker (sec company_tickers.json),
            intersect with sp400+sp600 symbol files, dedupe to one event
            per (ticker, date), write out/edgar_secondaries.csv.
            SEC requires a User-Agent with contact info: --ua "name email".
            ~68 index files; be patient, it sleeps between requests.
  study  -- SPY-adjusted CAR around the filing. Entry = close of the FIRST
            SESSION AFTER the filing date (T+1 close; filings are often
            after hours and the pricing day itself is not reliably
            tradeable at the modeled cost). Hold 20 sessions.

KNOWN DIRT, stated up front: 424B5 includes some shelf takedowns/ATM
supplements that are not marketed deals (noise, biases effect toward 0);
survivorship via yfinance as in the deletion study (coverage printed,
numbers biased UP); filing date can lag pricing by a day.

PRIMARY CELL (frozen): all matched events, BUY T+1 close, hold 20
sessions, SPY-adjusted, net --cost-bps. PASS: mean > 0 AND t >= 2 on
train (first 70% of events). Distress split and the pre-event window are
context and cannot be promoted. --tail-look is the one look.

    python3.10 scripts/explore_edgar_secondaries.py fetch --ua "Blake W bjw333@..."
    python3.10 scripts/explore_edgar_secondaries.py study
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
HOLD = 20
OUT = "out/edgar_secondaries.csv"
FORMS = {"424B4", "424B5"}


def _get(url, ua):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    return urllib.request.urlopen(req, timeout=60).read()


def cmd_fetch(a):
    syms = set()
    for f in ("manifests/sp400_symbols.txt", "manifests/sp600_symbols.txt"):
        syms |= {x.strip() for x in open(f) if x.strip()
                 and not x.startswith("#")}
    tick = json.loads(_get("https://www.sec.gov/files/company_tickers.json",
                           a.ua))
    cik2sym = {}
    for v in tick.values():
        if v["ticker"] in syms:
            cik2sym[int(v["cik_str"])] = v["ticker"]
    print(f"{len(cik2sym)} manifest names matched to CIKs "
          f"(current-ticker mapping -- renamed/delisted names are missed; "
          f"stated survivorship dirt)")
    rows = []
    yq = [(y, q) for y in range(2010, 2027) for q in (1, 2, 3, 4)]
    for y, q in yq:
        url = f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx"
        try:
            raw = _get(url, a.ua).decode("latin-1")
        except Exception as e:
            print(f"  {y} Q{q}: skip ({str(e)[:50]})")
            continue
        n0 = len(rows)
        import re
        pat = re.compile(r"(\d{4,10})\s+(\d{4}-\d{2}-\d{2})\s+edgar/")
        for line in raw.splitlines():
            form = line.split()[0] if line.split() else ""
            if form not in FORMS:
                continue
            m = pat.search(line)
            if not m:
                continue
            cik, date = int(m.group(1)), m.group(2)
            if cik in cik2sym:
                rows.append({"sym": cik2sym[cik], "date": date, "form": form})
        print(f"  {y} Q{q}: +{len(rows) - n0}")
        time.sleep(0.4)
    df = pd.DataFrame(rows).drop_duplicates(["sym", "date"])
    os.makedirs("out", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} events -> {OUT}")


def tstat(x):
    x = pd.Series(x).dropna()
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return len(x), x.mean(), (x.mean() / (sd / len(x) ** 0.5)
                              if sd > 0 else 0.0)


def cmd_study(a):
    cost = a.cost_bps / 1e4
    ev = pd.read_csv(a.events, parse_dates=["date"]).sort_values("date")
    print(f"{a.events}: {len(ev)} filings {ev['date'].min().date()} .. "
          f"{ev['date'].max().date()} | cost {a.cost_bps:.0f} bps")
    data = fetch(a.source, sorted(set(ev["sym"])), "1d",
                 "2009-06-01", "2026-06-01", min_bars=250)
    spy = fetch(a.source, ["SPY"], "1d", "2009-06-01", "2026-06-01",
                min_bars=250)["SPY"]["close"]
    print(f"price coverage: {len(data)}/{ev['sym'].nunique()} names -- "
          f"missing = delisted paths, numbers biased UP")
    rows = []
    for _, r in ev.iterrows():
        df = data.get(r["sym"])
        if df is None:
            continue
        c = df["close"]
        i = c.index.searchsorted(r["date"], side="right")   # first session AFTER filing
        if i < 140 or i + HOLD + 1 >= len(c):
            continue
        si = spy.index.searchsorted(c.index[i])
        if si + HOLD + 1 >= len(spy) or si < 25:
            continue
        e = float(c.iloc[i])
        trail = float(c.iloc[i - 1] / c.iloc[i - 127] - 1.0)
        pre5 = float(c.iloc[i - 1] / c.iloc[i - 6] - 1.0)
        out = {"sym": r["sym"], "date": c.index[i], "trail": trail,
               "pre5": pre5}
        for h in (5, 10, HOLD):
            out[f"f{h}"] = float(c.iloc[i + h]) / e - 1 - \
                (float(spy.iloc[si + h]) / float(spy.iloc[si]) - 1)
        rows.append(out)
    d = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"scorable: {len(d)}/{len(ev)}")
    if len(d) < 60:
        print("too thin to read; record and stop.")
        return
    cut = d["date"].iloc[int(len(d) * TRAIN_FRAC)]

    def block(fr, tag):
        print(f"\n===== {tag}: {len(fr):,} events =====")
        n, m, t = tstat(fr[f"f{HOLD}"] - cost)
        print(f"  PRIMARY: buy T+1 close, +{HOLD}d SPY-adj net: "
              f"{m*100:+.2f}% [t {t:+.1f}] n={n:,}   frozen pass: >0, t>=2")
        for h in (5, 10):
            n, m, t = tstat(fr[f"f{h}"])
            print(f"  context +{h}d gross: {m*100:+.2f}% [t {t:+.1f}]")
        n, m, t = tstat(fr["pre5"])
        print(f"  context into-event 5d: {m*100:+.2f}% [t {t:+.1f}]")
        dd = fr[fr["trail"] <= -0.30]
        n, m, t = tstat(dd[f"f{HOLD}"] - cost)
        print(f"  context distress-only +{HOLD}d net: {m*100:+.2f}% "
              f"[t {t:+.1f}] n={n:,}")

    if a.distress_tail:
        seg = d[(d["date"] >= cut) & (d["trail"] <= -0.30)]
        n, m, t = tstat(seg[f"f{HOLD}"] - cost)
        print(f"===== DISTRESS-ISSUER TAIL LOOK (the study's ONLY look) "
              f"| >= {cut.date()} =====")
        print(f"  trail<=-30%, buy T+1 close, +{HOLD}d SPY-adj net: "
              f"{m*100:+.2f}% [t {t:+.1f}] n={n}")
        print("  VERDICT vs frozen rule (mean>0 AND t>=2): "
              + ("PASS -- next: cohort/cluster check, then prereg."
                 if n >= 30 and m > 0 and t >= 2 else "DEAD. Seam closed."))
        return
    if a.tail_look:
        print(f"===== TAIL LOOK (one look, frozen cell) | >= {cut.date()} "
              f"=====")
        block(d[d["date"] >= cut], "TAIL -- read ONLY the primary line")
        return
    print(f"TRAIN < {cut.date()} (tail withheld)")
    block(d[d["date"] < cut], "TRAIN")
    print("\nREAD: primary line only. Pass earns ONE --tail-look. Context "
          "cannot be promoted; a juicy distress row here becomes, at most, "
          "a NEW pre-stated study.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--ua", required=True,
                   help='SEC-required User-Agent, e.g. "Blake W email@x.com"')
    s = sub.add_parser("study")
    s.add_argument("--events", default=OUT)
    s.add_argument("--source", default="yfinance")
    s.add_argument("--cost-bps", type=float, default=25.0)
    s.add_argument("--tail-look", action="store_true")
    s.add_argument("--distress-tail", action="store_true",
                   help="ONE look: distress-issuer cell, tail only")
    a = p.parse_args()
    (cmd_fetch if a.cmd == "fetch" else cmd_study)(a)


if __name__ == "__main__":
    main()
