"""
ZERO-TRIAL screen, three VWAP-mechanism hypotheses, ONE pass over the
minute cache. Train dates only (first 70%); the tail is never printed.
Nothing logged; no ledger trial spent.

MULTIPLICITY, priced now: three hypotheses ride one dataset. Each gets
EXACTLY ONE primary cell, frozen below BEFORE any table prints. All other
rows are descriptive context. A hypothesis whose primary cell fails on
train is CLOSED. One that passes earns ONE look at the withheld tail
(--tail-look A|B), nothing else. No re-bucketing, no added cells, no
"the adjacent row looks better". C has no trade cell; it is a feature
study and can only graduate to an atom candidate.

  A. PERSISTENCE (cross-count). A session that crosses its own running
     VWAP few times is one party working size all day; many crossings is
     two-sided noise. Claim: low-cross sessions closing at their extreme
     CONTINUE next session; high-cross sessions mean nothing.
     PRIMARY CELL A (frozen): ncross <= per-name train p20 AND (clv >= 0.8
     or <= 0.2), direction = sign(close - open), enter next open, exit
     next close, net --cost-bps. PASS: net mean > 0 AND t >= 2 AND mean
     exceeds the high-cross (>= p80) extreme-close control row.

  B. DEADLINE FLOW (MOC pressure). Benchmark flow must finish by the
     close; a big final-30-minute push on outsized closing volume is a
     price-insensitive party hitting a deadline. Claim: it REVERTS
     overnight. PRIMARY CELL B (frozen): last30 volume share >= per-name
     train p90 AND |close - px(15:30)| >= 0.25 * ATR14[t-1], trade
     AGAINST the last-30m move, close -> next open, net --cost-bps.
     PASS: net mean > 0 AND t >= 2.

  C. VOLUME-CLOCK SURPRISE (feature study, no trade). VWAP schedulers
     plan on the historical intraday volume curve; when the first hour
     prints far above its expected share, every scheduler is off-plan.
     Claim: surprise >= 2 predicts (i) elevated day range and (ii)
     elevated next-day volume. PASS (frozen): range/ATR median ratio
     >= 1.25 vs quiet days AND next-day rvol median > 1.20. Pass =
     candidate atom for future families, not a strategy.

    python3.10 scripts/explore_vwap_mechanisms.py
    python3.10 scripts/explore_vwap_mechanisms.py --tail-look A   # later,
        # only if A passed train; ONE invocation ever per hypothesis.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

TRAIN_FRAC = 0.70


def sessions(df):
    d = df.copy()
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
        "America/New_York")
    d["date"] = ts.dt.date
    d["min"] = (ts.dt.hour - 9) * 60 + ts.dt.minute - 30
    return d[(d["min"] >= 0) & (d["min"] < 390)].sort_values(["date", "min"])


def per_session(d):
    """One row per session with everything all three hypotheses need."""
    d = d.assign(pv=d["close"] * d["volume"])
    g = d.groupby("date", sort=True)
    cum_pv = g["pv"].cumsum()
    cum_v = g["volume"].cumsum()
    runvwap = cum_pv / cum_v.where(cum_v > 0)
    above = np.sign(d["close"].values - runvwap.values)
    newday = g.cumcount().values == 0
    flip = np.zeros(len(d), bool)
    flip[1:] = (above[1:] != above[:-1]) & (above[1:] != 0) & ~newday[1:]
    d = d.assign(flip=flip,
                 v_last30=d["volume"].where(d["min"] >= 360, 0.0),
                 v_first60=d["volume"].where(d["min"] < 60, 0.0),
                 px1030=d["close"].where(d["min"] == 60),
                 px1530=d["close"].where(d["min"] == 359))
    day = d.groupby("date").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        ncross=("flip", "sum"), v_last30=("v_last30", "sum"),
        v_first60=("v_first60", "sum"), px1030=("px1030", "max"),
        px1530=("px1530", "max"), nbars=("close", "size"))
    day = day[(day["nbars"] >= 300) & (day["volume"] > 0)]

    pc = day["close"].shift(1)
    tr = pd.concat([day["high"] - day["low"], (day["high"] - pc).abs(),
                    (day["low"] - pc).abs()], axis=1).max(axis=1)
    day["atr_prev"] = tr.rolling(14, min_periods=5).mean().shift(1)
    rng = (day["high"] - day["low"]).where(lambda s: s > 0)
    day["clv"] = (day["close"] - day["low"]) / rng
    day["daydir"] = np.sign(day["close"] - day["open"])
    day["sh30"] = day["v_last30"] / day["volume"]
    day["ret30_atr"] = (day["close"] - day["px1530"]) \
        / day["atr_prev"].where(day["atr_prev"] > 0)
    day["sh60"] = day["v_first60"] / day["volume"]
    exp60 = day["sh60"].rolling(60, min_periods=30).median().shift(1)
    day["surprise"] = day["sh60"] / exp60.where(exp60 > 0)
    vprev = day["volume"].rolling(20, min_periods=10).mean().shift(1)
    day["rvol"] = day["volume"] / vprev.where(vprev > 0)
    day["range_atr"] = (day["high"] - day["low"]) \
        / day["atr_prev"].where(day["atr_prev"] > 0)
    day["fwd_oc"] = day["close"].shift(-1) / day["open"].shift(-1) - 1.0
    day["fwd_on"] = day["open"].shift(-1) / day["close"] - 1.0
    day["rvol_next"] = day["rvol"].shift(-1)
    return day


def tstat(x):
    x = x.dropna()
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return (len(x), x.mean(),
            x.mean() / (sd / len(x) ** 0.5) if sd > 0 else 0.0)


def row(label, x, cost=0.0):
    n, m, t = tstat(x - cost)
    if n < 30:
        print(f"  {label:46s} n={n} -- unscorable")
        return None
    print(f"  {label:46s} {m*100:+.3f}%/trade [t {t:+5.1f}] n={n:>6,}")
    return (n, m, t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.expanduser("~/.primordial_intraday"))
    p.add_argument("--cost-bps", type=float, default=14.0)
    p.add_argument("--tail-look", choices=["A", "B"],
                   help="THE one look for a hypothesis that PASSED train. "
                        "One invocation ever per hypothesis; scores ONLY the "
                        "frozen primary cell on the withheld dates.")
    a = p.parse_args()
    cost = a.cost_bps / 1e4

    files = sorted(glob.glob(os.path.join(a.dir, "*_1m.parquet")))
    if not files:
        raise SystemExit(f"no parquet in {a.dir} -- run fetch_intraday.py")

    frames = []
    for f in files:
        sym = os.path.basename(f).split("_")[0]
        d = sessions(pd.read_parquet(f))
        if not len(d):
            continue
        day = per_session(d)
        day["sym"] = sym
        frames.append(day.reset_index())
    all_ = pd.concat(frames, ignore_index=True)

    dates = np.sort(all_["date"].unique())
    cut = dates[int(len(dates) * TRAIN_FRAC)]
    tr_ = all_[all_["date"] < cut].copy()

    # per-name thresholds from TRAIN dates only (applied unchanged to any
    # tail look -- the tail never informs its own cut)
    thr = tr_.groupby("sym").agg(nc20=("ncross", lambda s: s.quantile(0.20)),
                                 nc80=("ncross", lambda s: s.quantile(0.80)),
                                 sh90=("sh30", lambda s: s.quantile(0.90)))

    def enrich(fr):
        fr = fr.copy()
        fr["nc20"] = fr["sym"].map(thr["nc20"])
        fr["nc80"] = fr["sym"].map(thr["nc80"])
        fr["sh90"] = fr["sym"].map(thr["sh90"])
        fr["extreme"] = (fr["clv"] >= 0.8) | (fr["clv"] <= 0.2)
        fr["contA"] = fr["daydir"] * fr["fwd_oc"]
        fr["fadeB"] = -np.sign(fr["ret30_atr"]) * fr["fwd_on"]
        return fr

    tr_ = enrich(tr_)

    if a.tail_look:
        seg = enrich(all_[all_["date"] >= cut])
        print(f"===== TAIL LOOK {a.tail_look} (one look, frozen cell) "
              f"| dates >= {cut} | cost {a.cost_bps:.0f} bps =====")
        if a.tail_look == "A":
            c = seg[(seg["ncross"] <= seg["nc20"]) & seg["extreme"]]
            r = row("A: low-cross extreme-close continuation", c["contA"],
                    cost)
        else:
            c = seg[(seg["sh30"] >= seg["sh90"])
                    & (seg["ret30_atr"].abs() >= 0.25)]
            r = row("B: fade the deadline push, overnight", c["fadeB"], cost)
        if r:
            n, m, t = r
            print("  VERDICT vs frozen rule: "
                  + ("PASS -- next step is a gauntlet family, charged to "
                     "the ledger like anything else."
                     if m > 0 and t >= 2 else "DEAD. Seam closed."))
        return

    print(f"TRAIN {min(dates)} .. {cut} (tail withheld) | "
          f"{tr_['sym'].nunique()} names | {len(tr_):,} sessions | "
          f"cost {a.cost_bps:.0f} bps RT")

    # ================= A: persistence =================
    print("\n[A] cross-count persistence -- signed next-day continuation "
          "(next open -> next close)")
    lo = tr_["ncross"] <= tr_["nc20"]
    hi = tr_["ncross"] >= tr_["nc80"]
    row("PRIMARY: low-cross AND close at extreme",
        tr_.loc[lo & tr_["extreme"], "contA"], cost)
    row("control: high-cross AND close at extreme",
        tr_.loc[hi & tr_["extreme"], "contA"], cost)
    row("context: low-cross, any close", tr_.loc[lo, "contA"], cost)
    row("context: all sessions", tr_["contA"], cost)
    if (lo & tr_["extreme"]).sum():
        print(f"  (median crossings in primary cell: "
              f"{tr_.loc[lo & tr_['extreme'], 'ncross'].median():.0f}; "
              f"all-session median: {tr_['ncross'].median():.0f})")

    # ================= B: deadline flow =================
    print("\n[B] MOC deadline pressure -- fade last-30m push, overnight "
          "(close -> next open)")
    big30 = tr_["ret30_atr"].abs() >= 0.25
    hot30 = tr_["sh30"] >= tr_["sh90"]
    pb = hot30 & big30
    row("PRIMARY: sh30>=p90 AND |ret30|>=0.25 ATR", tr_.loc[pb, "fadeB"],
        cost)
    row("control: sh30>=p90, small ret30", tr_.loc[hot30 & ~big30, "fadeB"],
        cost)
    row("control: normal sh30, |ret30|>=0.25", tr_.loc[~hot30 & big30,
        "fadeB"], cost)
    if pb.sum() >= 30:
        g = tr_.loc[pb].assign(y=[d.year for d in tr_.loc[pb, "date"]])
        by = "  ".join(f"{y}:{(v - cost).mean()*100:+.2f}"
                       for y, v in g.groupby("y")["fadeB"])
        print(f"      by year: {by}")

    # ================= C: volume-clock surprise =================
    print("\n[C] volume-clock surprise (feature study, no trade cell)")
    s = tr_.dropna(subset=["surprise", "range_atr"])
    hot = s[s["surprise"] >= 2.0]
    quiet = s[s["surprise"] < 1.25]
    if len(hot) >= 30 and len(quiet) >= 30:
        ratio = hot["range_atr"].median() / quiet["range_atr"].median()
        print(f"  day range/ATR median: surprise>=2 "
              f"{hot['range_atr'].median():.2f} vs quiet "
              f"{quiet['range_atr'].median():.2f} (ratio {ratio:.2f}, "
              f"frozen pass >= 1.25) | n={len(hot):,}")
        print(f"  next-day rvol median on surprise days: "
              f"{hot['rvol_next'].median():.2f} (frozen pass > 1.20)")
        m1030 = (hot["close"] / hot["px1030"] - 1) * np.sign(
            hot["px1030"] / hot["open"] - 1)
        n, m, t = tstat(m1030)
        print(f"  descriptive: 10:30->close continuation of the morning "
              f"move on surprise days {m*100:+.3f}% [t {t:+.1f}] n={n:,}")
    else:
        print(f"  surprise>=2: n={len(hot)} -- unscorable")

    print("\nREAD (frozen above): A passes on its cell + beats its "
          "high-cross control; B passes on its cell; C passes on both "
          "medians. Fail = closed. Pass = ONE --tail-look, then gauntlet "
          "or nothing.")


if __name__ == "__main__":
    main()
