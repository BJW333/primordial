"""
POINT-IN-TIME survivorship test for the distress rule -- THE one test that
could reopen the idea (DISTRESS_PREREG.md RESULT block) or confirm the
mechanism of its death.

Base rule, unchanged (sha b62a9753 family):
    signal_t = atr_pct 252d-percentile > 0.9  AND  close < SMA200
    enter next open | exit first close > EMA26 after entry day | 40-bar max

Two runs of the SAME engine on the SAME prices, 2011-01-01 onward:
  SURVIVOR mode: universe = names in the index TODAY, no date gating.
                 (replicates the biased design of every prior run)
  PIT mode:      universe = every name in the index at any point since 2010
                 (fja05680/sp500 point-in-time record, incl. departed);
                 entry allowed only when the name was a member on signal
                 date; departed names' data used until it ends.

Ground caveat, stated up front: this is S&P 500, not sp600 -- the free PIT
record that exists. Large caps have fewer deep-distress names; a kill here
transfers imperfectly and a pass here would still owe the Norgate sp600
version. This is the cheap decisive-in-one-direction test: if the edge
DISAPPEARS when dead names are restored, survivorship is confirmed as the
driver and the idea stays closed on all grounds.

REGISTERED THRESHOLDS (commit this file BEFORE running; one look):
  VOID        departed-name data coverage < 50% -- yfinance kept too few
              dead names; verdict impossible without Norgate. Not a pass.
  CONFIRMED   PIT SR <= max(0, 0.5 x survivor SR)  -> survivorship was the
              edge; distress stays dead everywhere, mechanism identified.
  REOPEN      PIT SR >= 0.8 x survivor SR AND PIT SR > 0 AND per-trade
              t >= 2 -> survivorship objection weakened; justifies the
              Norgate sp600 purchase and a NEW prereg inheriting the
              family's ledger debt. Nothing trades off this result alone.
  else        INCONCLUSIVE -- Norgate decision falls back to the sp400
              rotation's merits alone.

Conventions (fixed here so nobody argues with the output):
  costs 20 bps round trip; max 20 concurrent positions, first-come;
  portfolio SR = annualized mean/std of equal-weight daily book returns on
  invested days only (exposure reported beside it); positions still open
  when a departed name's data ends are closed at its last close, and a
  -20% terminal-haircut variant is reported as stress (verdict uses the
  unhaircut number -- the haircut only ever makes PIT worse).

Run:
    python3.10 scripts/pit_distress_sp500.py            # the one look
    python3.10 scripts/pit_distress_sp500.py --smoke    # synthetic pipe test
"""
import argparse
import csv
import os
import sys
from bisect import bisect_right

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial.data import fetch  # noqa: E402

MEMBERSHIP_CSV = os.path.join(ROOT, "manifests", "sp500_pit_membership.csv")
START, SIG_START = "2009-01-01", "2011-01-01"
COST_RT, MAX_POS, MAX_HOLD, HAIRCUT = 0.0020, 20, 40, 0.20


def load_membership(path):
    rows = [r for r in csv.reader(open(path))][1:]
    dates = [r[0] for r in rows]
    sets_ = [frozenset(t.strip() for t in r[1].split(",")) for r in rows]
    def member_on(day, sym):
        i = bisect_right(dates, str(day)) - 1
        return i >= 0 and sym in sets_[i]
    union = set().union(*sets_)
    current = sets_[-1]
    return member_on, sorted(union), sorted(current)


def indicators(df):
    c, h, l = df["close"], df["high"], df["low"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_pct = tr.rolling(14).mean() / c
    rank = atr_pct.rolling(252).rank(pct=True)
    return pd.DataFrame({"open": df["open"], "close": c,
                         "rank": rank, "sma200": c.rolling(200).mean(),
                         "ema26": c.ewm(span=26, adjust=False).mean()})


def run_mode(data, member_on, pit):
    """One pass. Returns dict of stats. pit=False -> survivor mode
    (universe already filtered to current members; no date gate)."""
    sigs = []           # (signal_date, sym)
    frames = {}
    for sym, df in data.items():
        ind = indicators(df)
        frames[sym] = ind
        ok = (ind["rank"] > 0.9) & (ind["close"] < ind["sma200"])
        for t in ind.index[ok]:
            if str(t.date()) < SIG_START:
                continue
            if pit and not member_on(t.date(), sym):
                continue
            sigs.append((t, sym))
    sigs.sort()
    exits, trades, daily = [], [], {}
    forced = 0
    for t, sym in sigs:
        ind = frames[sym]
        i = ind.index.get_loc(t)
        if i + 1 >= len(ind):
            continue
        entry_date = ind.index[i + 1]
        exits = [x for x in exits if x > entry_date]
        if len(exits) >= MAX_POS:
            continue
        e_i, e_px = i + 1, float(ind["open"].iloc[i + 1])
        if not np.isfinite(e_px) or e_px <= 0:
            continue
        # walk the position now (frames are static; concurrency cap is
        # approximate first-come by signal order, matching engine spirit)
        x_i, x_px, why = None, None, "maxhold"
        last = min(e_i + MAX_HOLD, len(ind) - 1)
        for j in range(e_i + 1, last + 1):
            if float(ind["close"].iloc[j]) > float(ind["ema26"].iloc[j]):
                x_i, x_px, why = j, float(ind["close"].iloc[j]), "anchor"
                break
        if x_i is None:
            x_i, x_px = last, float(ind["close"].iloc[last])
            if last < e_i + MAX_HOLD and last == len(ind) - 1:
                why = "data_end"; forced += 1
        ret = x_px / e_px - 1 - COST_RT
        trades.append({"sym": sym, "entry": ind.index[e_i],
                       "exit": ind.index[x_i], "ret": ret, "why": why})
        # daily book contributions (close-to-close; entry day open->close)
        px = ind["close"].iloc[e_i:x_i + 1].values
        days = ind.index[e_i:x_i + 1]
        r0 = px[0] / e_px - 1
        rs = np.concatenate([[r0], px[1:] / px[:-1] - 1]) if len(px) > 1 \
            else np.array([r0])
        for d, r in zip(days, rs):
            daily.setdefault(d, []).append(r)
        exits.append(ind.index[x_i])
    tr = pd.DataFrame(trades)
    if not len(tr):
        return None
    book = pd.Series({d: float(np.mean(v)) for d, v in daily.items()}
                     ).sort_index()
    sr = book.mean() / book.std(ddof=1) * np.sqrt(252) if book.std(ddof=1) \
        else 0.0
    t_pt = tr["ret"].mean() / (tr["ret"].std(ddof=1) / np.sqrt(len(tr)))
    hc = tr["ret"].where(tr["why"] != "data_end", tr["ret"] - HAIRCUT)
    span_days = set()
    for f in frames.values():
        span_days.update(d for d in f.index
                         if book.index.min() <= d <= book.index.max())
    return {"sr": sr, "n": len(tr), "mean": tr["ret"].mean(),
            "t": t_pt, "win": (tr["ret"] > 0).mean(),
            "forced": forced, "mean_hc": hc.mean(),
            "exposure": len(book) / max(1, len(span_days)),
            "by_year": tr.groupby(tr["entry"].dt.year)["ret"]
                         .agg(["mean", "count"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="yfinance")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        syms = [f"S{i}" for i in range(12)]
        cur = syms[:8]
        member_on = lambda d, s: s in syms  # noqa: E731
        data = fetch("synthetic", syms, "1d", START, "2026-06-01")
    else:
        member_on, union, cur = load_membership(MEMBERSHIP_CSV)
        union = [s.replace(".", "-") for s in union]
        cur = [s.replace(".", "-") for s in cur]
        print(f"[pit] {len(union)} names ever in index 2009+ | "
              f"{len(cur)} current | fetching (cache-heavy, be patient)")
        data = fetch(a.source, union, "1d", START, "2026-06-01")
        raw_member = member_on
        member_on = lambda d, s: raw_member(d, s.replace("-", "."))  # noqa

    # ---- coverage of DEPARTED names (gates validity) ----
    if not a.smoke:
        departed = [s for s in union if s not in cur]
        have = [s for s in departed if s in data and len(data[s]) >= 252]
        cov = len(have) / max(1, len(departed))
        print(f"  departed names: {len(departed)} | with >=1y data: "
              f"{len(have)} | coverage {cov:.0%}")
    else:
        cov, departed = 1.0, []

    surv = run_mode({s: d for s, d in data.items() if s in set(cur)},
                    member_on, pit=False)
    pit = run_mode(data, member_on, pit=True)
    if surv is None or pit is None:
        sys.exit("a mode produced zero trades -- check data/pipeline.")

    for name, m in (("SURVIVOR (current members, no gate)", surv),
                    ("PIT (all names, membership-gated)", pit)):
        print(f"\n  == {name} ==")
        print(f"  SR {m['sr']:+.2f} (invested days, exposure "
              f"{m['exposure']:.0%}) | {m['n']} trades | "
              f"mean {m['mean']*100:+.3f}%/trade | t {m['t']:+.2f} | "
              f"win {m['win']:.0%}")
        if m["forced"]:
            print(f"  forced closes at data end: {m['forced']} | mean with "
                  f"-20% haircut on those: {m['mean_hc']*100:+.3f}%")
        print("  by entry year: " + "  ".join(
            f"{y}:{r['mean']*100:+.2f}%({int(r['count'])})"
            for y, r in m["by_year"].iterrows()))

    print("\n  ===== REGISTERED VERDICT =====")
    if cov < 0.50:
        print(f"  VOID -- departed coverage {cov:.0%} < 50%. yfinance kept "
              f"too few dead names; only Norgate can run this test.")
    elif pit["sr"] <= max(0.0, 0.5 * surv["sr"]):
        print(f"  CONFIRMED -- PIT SR {pit['sr']:+.2f} vs survivor "
              f"{surv['sr']:+.2f}: the edge lives in the dead names' "
              f"absence. Distress stays closed on every ground; "
              f"survivorship is the identified mechanism.")
    elif pit["sr"] >= 0.8 * surv["sr"] and pit["sr"] > 0 and pit["t"] >= 2:
        print(f"  REOPEN SIGNAL -- PIT SR {pit['sr']:+.2f} holds up vs "
              f"survivor {surv['sr']:+.2f} (t {pit['t']:+.2f}). The "
              f"objection is weakened ON LARGE CAPS. Next: Norgate sp600 "
              f"PIT + a NEW prereg inheriting the family's debt. Nothing "
              f"trades off this result.")
    else:
        print(f"  INCONCLUSIVE -- PIT {pit['sr']:+.2f} vs survivor "
              f"{surv['sr']:+.2f}. Norgate decision falls back to the "
              f"sp400 rotation's merits alone.")


if __name__ == "__main__":
    main()
