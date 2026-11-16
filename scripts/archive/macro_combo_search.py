"""
macro_combo_search.py -- can an early warning predict momentum's drops if we
add ECONOMIC warning signs (not just prices) to the 6 price signs already
tested? Every 1-, 2- and 3-sign combination, searched on 1927-1989, judged
against luck with a permutation test, then checked untouched on 1990-2016
and on CHIMERA-70.

WARNING SIGNS (each ranked against its own trailing 5 years, point in time;
3 levels each: top/bottom 20%, 10%, 5%)
  price (from crash_warning_search.py, daily, Ken French data since 1926)
    MOMVOL  winners' last-month volatility / their last-year volatility (high)
    BEAR    market's last-2-year return                               (low)
    MKT21   market's last-month return                                (high)
    MKTVOL  market's last-month volatility                            (high)
    HOT     winners' last-month return                                (high)
    RS      winners' minus market's last-month return                 (low)
  economic (monthly, Federal Reserve FRED, used only AFTER publication:
  rates/spreads 1 month later, unemployment 2 months later)
    CREDIT  Baa minus Aaa corporate bond yield, change vs 12 months ago
            -- lenders demanding more from risky companies     (high) 1919+
    CURVE   10-year Treasury minus 3-month T-bill -- inverted
            yield curve                                         (low)  1953+
    JOBS    3-month average unemployment minus its lowest 3-month
            average of the prior 12 months (Sahm rule)          (high) 1948+
  RULES: every combination of 1, 2 or 3 DIFFERENT signs, AND-ed, at every
  level -> 2,619 rules. Warning on at a close -> T-bills the next day; 0.2%
  per switch. (Gold isn't testable over 100 years: its price was fixed by
  law until the 1970s.)

STEPS (pre-stated; nothing changes after seeing results)
  1 SEARCH  1927-1989: rule with the biggest Sharpe gain over always holding
            Ken French's top-10% momentum stocks (warning on 1%-50% of days).
  2 LUCK    redo the whole 2,619-rule search 200 times with all signs slid
            in time against the returns (>= 1 year); p = share of slid
            searches whose best gain >= the real best. Must be < 0.05.
  3 LATER   1990-2016 untouched: must raise Sharpe AND shrink the worst drop.
  4 CHIMERA 2017-03 -> 2026-05: CHIMERA-70 (70% momentum book + 30% T-bills,
            35% yearly lock -> SPY) with the warning (-> all T-bills) must
            beat CHIMERA-70 alone on BOTH Sharpe and worst drop.
  PASS only if 2, 3 and 4 all pass. Context only: the best ECONOMIC-ONLY
  rule is also shown.

Run (repo root; downloads 6 small FRED files once; uses all cores):
    python3.10 scripts/macro_combo_search.py --smoke
    python3.10 scripts/macro_combo_search.py | tee logs/macro_combo_search.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import io
import itertools
import os
import sys
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_warning_search as CW   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
FRED_DIR = os.path.join(ROOT, "data", "fred")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SERIES = ["BAA", "AAA", "GS10", "TB3MS", "UNRATE"]
MACRO = [("CREDIT", "high"), ("CURVE", "low"), ("JOBS", "high")]
N_PERM, CHUNK = 200, 256
P1, P2, P3 = ("1927-01-01", "1989-12-31"), ("1990-01-01", "2016-12-31"), \
    ("2017-03-01", "2026-05-29")


# ------------------------------------------------------------- macro data
def fred(sid):
    os.makedirs(FRED_DIR, exist_ok=True)
    p = os.path.join(FRED_DIR, f"{sid}.csv")
    if not os.path.exists(p):
        print(f"  downloading FRED {sid} ...")
        req = urllib.request.Request(FRED.format(sid), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
            f.write(r.read())
    df = pd.read_csv(p)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0]).dt.to_period("M")
    return s.dropna()


def macro_monthly(smoke=False):
    if smoke:
        per = pd.period_range("1919-01", "2026-09", freq="M")
        rng = np.random.default_rng(9)
        x = {k: pd.Series(np.cumsum(rng.normal(0, 0.1, len(per))), per) for k in SERIES}
        x["UNRATE"] = 5 + x["UNRATE"].abs()
    else:
        x = {k: fred(k) for k in SERIES}
    credit = (x["BAA"] - x["AAA"]).diff(12)
    curve = x["GS10"] - x["TB3MS"]
    u3 = x["UNRATE"].rolling(3).mean()
    jobs = u3 - u3.shift(1).rolling(12).min()
    m = pd.DataFrame({"CREDIT": credit.shift(1), "CURVE": curve.shift(1),
                      "JOBS": jobs.shift(2)})          # usable only after publication
    return m.rolling(60, min_periods=36).rank(pct=True)


def macro_conditions(rank_m, dates, start_feat):
    per = pd.Index(dates).to_period("M")
    cols, names, feat_of = [], [], []
    for fi, (f, side) in enumerate(MACRO):
        r = rank_m[f].reindex(per).values
        for q in CW.LEVELS:
            c = (r >= q) if side == "high" else (r <= 1 - q)
            cols.append(np.nan_to_num(c, nan=0).astype(bool))
            names.append(f"{f} {'top' if side == 'high' else 'bottom'} {round((1 - q) * 100)}%")
            feat_of.append(start_feat + fi)
    return np.column_stack(cols), names, feat_of


# ---------------------------------------------------------------- search
def score_all(W, RF, C, rules):
    out = np.empty(len(rules))
    for i in range(0, len(rules), CHUNK):
        out[i:i + CHUNK] = CW.score(W, RF, CW.warn_matrix(C, rules[i:i + CHUNK]))
    return out


def slid_best(args):
    seed, W, RF, C, rules = args
    k = int(np.random.default_rng(seed).integers(CW.MIN_SHIFT, len(W) - CW.MIN_SHIFT))
    return float(np.max(score_all(W, RF, np.roll(C, k, axis=0), rules)))


def chimera70(m, rf):
    return 0.7 * m + 0.3 * rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--perms", type=int, default=N_PERM)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    d = CW.smoke_data() if a.smoke else CW.load_french()
    rank_m = macro_monthly(a.smoke)
    Cp, names_p, feat_p = CW.conditions(CW.features(d))
    Cm, names_m, feat_m = macro_conditions(rank_m, d.index, max(feat_p) + 1)
    C = np.hstack([Cp, Cm])
    names, feat_of = names_p + names_m, feat_p + feat_m
    rules = CW.all_rules(feat_of)
    macro_ids = set(range(len(names_p), len(names)))
    rname = lambda j: " AND ".join(names[i] for i in rules[j])  # noqa: E731
    print(f"  French {d.index[0].date()} -> {d.index[-1].date()} | "
          f"{len(rules)} rules from {len(set(feat_of))} signs x 3 levels"
          f"{'  [SYNTHETIC]' if a.smoke else ''}")

    s1 = (d.index >= P1[0]) & (d.index <= P1[1])
    W1, RF1, C1 = d["W"].values[s1], d["RF"].values[s1], C[s1]

    # 1 SEARCH
    g = score_all(W1, RF1, C1, rules)
    order = np.argsort(-g)
    best = int(order[0])
    print("\n  ===== 1 SEARCH 1927-1989 (Sharpe gain vs always holding winners) =====")
    for j in order[:15]:
        print(f"    {g[j]:+.3f}  {rname(j)}")
    mac = [j for j in order if set(rules[j]) <= macro_ids]
    if mac:
        print(f"    best ECONOMIC-ONLY rule (context): {g[mac[0]]:+.3f}  {rname(mac[0])}")

    # 2 LUCK
    jobs = [(s, W1, RF1, C1, rules) for s in range(a.perms)]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        null = np.array(list(ex.map(slid_best, jobs)))
    p = (np.sum(null >= g[best]) + 1) / (a.perms + 1)
    print(f"\n  ===== 2 LUCK: {a.perms} searches with the signs slid in time =====")
    print(f"    real best {g[best]:+.3f} | slid: median best {np.median(null):+.3f}, "
          f"95th pct {np.percentile(null, 95):+.3f} | p = {p:.3f}")

    # 3 LATER
    print(f"\n  ===== 3 LATER 1990-2016, fixed: {rname(best)} =====")
    s2 = (d.index >= P2[0]) & (d.index <= P2[1])
    W2, RF2 = d["W"].values[s2], d["RF"].values[s2]
    r2, on2 = CW.apply_warn(W2, RF2, CW.warn_matrix(C[s2], [rules[best]]))
    sb, db = CW.line("winners always", W2)
    sw, dw = CW.line("winners + warning", r2[:, 0])
    print(f"    warning on {on2.mean()*100:.0f}% of days")
    ok3 = sw > sb and dw > db
    for j in order[:10]:
        rj, _ = CW.apply_warn(W2, RF2, CW.warn_matrix(C[s2], [rules[j]]))
        print(f"      context: Sharpe {CW.sharpe(rj)[0]:.2f} vs {sb:.2f}, worst "
              f"{CW.worst(rj)[0]*100:.0f}% vs {db*100:.0f}%  {rname(j)}")

    # 4 CHIMERA-70
    print(f"\n  ===== 4 CHIMERA-70 {P3[0]} -> {P3[1]} (70% momentum, 35% lock) =====")
    if a.smoke:
        s3 = (d.index >= P3[0]) & (d.index <= P3[1])
        bk = pd.DataFrame({"STEADY": d["W"][s3] * 1.5, "SPY": d["M"][s3]})
    else:
        bk = pd.read_csv(CW.BOOK, index_col=0, parse_dates=True)
        bk = bk[(bk.index >= P3[0]) & (bk.index <= P3[1])]
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    wfull = pd.Series(CW.warn_matrix(C, [rules[best]])[:, 0], d.index)
    w3 = wfull.reindex(bk.index).ffill().fillna(False).astype(bool).values
    CW.LOCK = 0.35
    m70 = chimera70(bk["STEADY"], cash)
    base = CW.lock_path(m70, bk["SPY"], cash, None)
    test = CW.lock_path(m70, bk["SPY"], cash, w3)
    s0, d0 = CW.line("CHIMERA-70", base.values)
    s4, d4 = CW.line("CHIMERA-70 + warning", test.values)
    print(f"    warning on {w3.mean()*100:.0f}% of days")
    for y in sorted(set(bk.index.year)):
        b = (1 + base[base.index.year == y]).prod() - 1
        t = (1 + test[test.index.year == y]).prod() - 1
        print(f"      {y}: CHIMERA-70 {b*100:+5.0f}%   + warning {t*100:+5.0f}%")
    on = pd.Series(w3, bk.index)
    starts = on[on & ~on.shift(1, fill_value=False)].index
    print(f"    warning periods ({len(starts)}): " + ", ".join(
        f"{s:%Y-%m-%d}" for s in starts[:25]) + (" ..." if len(starts) > 25 else ""))
    ok4 = s4 > s0 and d4 > d0

    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"    luck test p = {p:.3f} -> {'pass' if p < 0.05 else 'FAIL'}")
    print(f"    1990-2016 untouched -> {'pass' if ok3 else 'FAIL'}")
    print(f"    CHIMERA-70          -> {'pass' if ok4 else 'FAIL'}")
    print("    " + ("PASS -- port the warning to QC as a switch"
                    if p < 0.05 and ok3 and ok4 else "FAIL -- keep CHIMERA-70 as is"))
    if a.smoke:
        print("  smoke: synthetic data (planted price warning; macro is random)")


if __name__ == "__main__":
    main()
