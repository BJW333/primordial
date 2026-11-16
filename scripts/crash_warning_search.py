"""
crash_warning_search.py -- can an EARLY WARNING spot momentum crashes before
the account is down 20-25%? Searches combinations of warning signs on ~100
years of data, judges the winner against luck with a permutation test, then
checks it untouched on later data and on CHIMERA's own momentum book.

DATA (downloaded once to data/french/, Ken French data library)
  winners  "Hi PRIOR" = top 10% of US stocks by 12-month return (skipping
           the last month), value weighted, daily since 1926 -- the long-only
           version of the momentum book
  market   Mkt-RF + RF (all US stocks), daily;  cash = RF (T-bills)
  book     data/momentum_cap_daily.csv (STEADY, the book running on QC) and
           its SPY column, 2017-03 -> 2026-05

WARNING SIGNS (computed from the French winners + market only, so they exist
for every period; each ranked against its own trailing 5 years, point in
time, so levels from 1930 and 2020 are comparable)
  MOMVOL  winners' last-month volatility / their last-year volatility  (high)
  BEAR    market's last-2-year return                                  (low)
  MKT21   market's last-month return (the "rebound")                   (high)
  MKTVOL  market's last-month volatility                               (high)
  HOT     winners' last-month return (overheated)                      (high)
  RS      winners' minus market's last-month return (leadership cracks)(low)
  Each at 3 levels: top/bottom 20%, 10%, 5% of its own 5-year history.
  RULES: every single sign, every pair, every triple of DIFFERENT signs,
  combined with AND -> 693 rules. Warning on at a close -> hold T-bills
  the next day; 0.2% cost per switch.

STEPS (pre-stated; nothing below may be changed after seeing results)
  1 SEARCH   1927-1989 only: pick the rule with the biggest Sharpe gain over
             always holding the winners (warning on 1%-50% of days).
  2 LUCK     redo the whole 693-rule search 200 times with the warning signs
             slid in time against the returns (circular shift >= 1 year);
             p = share of slid searches whose best gain >= the real best.
  3 LATER    1990-2016, same rule, untouched: must improve BOTH Sharpe and
             worst drop vs always holding the winners.
  4 CHIMERA  2017-03 -> 2026-05 on the real book with the 30% yearly lock
             (lock -> SPY until Jan 1; warning -> T-bills while on): must
             beat LOCK30 alone on BOTH Sharpe and worst drop.
  PASS only if p < 0.05 AND step 3 AND step 4. Then (and only then) it gets
  ported to QuantConnect as a switch.

Run (repo root; downloads ~2 MB once, then ~2-5 minutes):
    python3.10 scripts/crash_warning_search.py --smoke
    python3.10 scripts/crash_warning_search.py | tee logs/crash_warning_search.txt
"""
import argparse
import io
import itertools
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FDIR = os.path.join(ROOT, "data", "french")
BOOK = os.path.join(ROOT, "data", "momentum_cap_daily.csv")
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"mom": "10_Portfolios_Prior_12_2_Daily_CSV.zip",
         "ff": "F-F_Research_Data_Factors_daily_CSV.zip"}

COST, LOCK, RANK_WIN = 0.002, 0.30, 1260
LEVELS = (0.80, 0.90, 0.95)
N_PERM, MIN_SHIFT = 200, 252
P1 = ("1927-01-01", "1989-12-31")
P2 = ("1990-01-01", "2016-12-31")
P3 = ("2017-03-01", "2026-05-29")
FEATS = [("MOMVOL", "high"), ("BEAR", "low"), ("MKT21", "high"),
         ("MKTVOL", "high"), ("HOT", "high"), ("RS", "low")]


# ------------------------------------------------------------------ data
def fetch(key):
    os.makedirs(FDIR, exist_ok=True)
    path = os.path.join(FDIR, FILES[key])
    if not os.path.exists(path):
        print(f"  downloading {FILES[key]} ...")
        req = urllib.request.Request(BASE + FILES[key],
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def first_block(text):
    """First table in a French CSV: header line + YYYYMMDD rows."""
    header, rows = None, []
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"^\d{8}\s*,", s):
            rows.append([x.strip() for x in s.split(",")])
        elif rows:
            break
        elif "," in s:
            header = [x.strip() for x in s.split(",")]
    df = pd.DataFrame(rows).set_index(0)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.columns = header[1:len(df.columns) + 1]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.where(df > -99) / 100


def load_french():
    mom = first_block(fetch("mom"))
    ff = first_block(fetch("ff"))
    hi = [c for c in mom.columns if c.lower().startswith("hi")][0]
    d = pd.DataFrame({"W": mom[hi], "M": ff["Mkt-RF"] + ff["RF"],
                      "RF": ff["RF"]}).dropna()
    return d


def smoke_data():
    """Synthetic: a volatility spike in the winners comes before each crash."""
    d = pd.bdate_range("1926-11-01", "2026-06-30")
    rng = np.random.default_rng(7)
    n = len(d)
    M = rng.normal(0.0004, 0.010, n)
    W = M * 1.2 + rng.normal(0.0003, 0.008, n)
    for s in rng.choice(np.arange(1500, n - 60), 40, replace=False):
        W[s:s + 15] += rng.normal(0, 0.035, 15)          # warning: wild swings
        W[s + 15:s + 35] -= 0.012                         # then the crash
    return pd.DataFrame({"W": W, "M": M, "RF": 0.0001}, index=d)


# -------------------------------------------------------------- features
def cumret(x, n):
    return np.expm1(np.log1p(x).rolling(n).sum())


def features(d):
    W, M = d["W"], d["M"]
    raw = {"MOMVOL": W.rolling(21).std() / W.rolling(252).std(),
           "BEAR": cumret(M, 504), "MKT21": cumret(M, 21),
           "MKTVOL": M.rolling(21).std(), "HOT": cumret(W, 21)}
    raw["RS"] = raw["HOT"] - raw["MKT21"]
    return pd.DataFrame({k: v.rolling(RANK_WIN, min_periods=756).rank(pct=True)
                         for k, v in raw.items()})


def conditions(rank):
    """18 yes/no columns: each sign at each level."""
    cols, names, feat_of = [], [], []
    for fi, (f, side) in enumerate(FEATS):
        for q in LEVELS:
            c = rank[f] >= q if side == "high" else rank[f] <= 1 - q
            cols.append(c.fillna(False).values)
            names.append(f"{f} {'top' if side == 'high' else 'bottom'} "
                         f"{round((1 - q) * 100)}%")
            feat_of.append(fi)
    return np.column_stack(cols), names, feat_of


def all_rules(feat_of):
    idx = range(len(feat_of))
    rules = []
    for k in (1, 2, 3):
        for combo in itertools.combinations(idx, k):
            if len({feat_of[i] for i in combo}) == k:
                rules.append(combo)
    return rules


def warn_matrix(C, rules):
    out = np.ones((C.shape[0], len(rules)), bool)
    for j, combo in enumerate(rules):
        for i in combo:
            out[:, j] &= C[:, i]
    return out


# ------------------------------------------------------------ evaluation
def sharpe(r):
    return r.mean(axis=0) / r.std(axis=0) * np.sqrt(252)


def worst(r):
    eq = np.cumprod(1 + r, axis=0)
    return (eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0)


def apply_warn(Wr, RF, warn):
    """warn at close t -> T-bills on day t+1; 0.2% per switch."""
    on = np.vstack([np.zeros((1, warn.shape[1]), bool), warn[:-1]])
    r = np.where(on, RF[:, None], Wr[:, None])
    sw = np.vstack([np.zeros((1, warn.shape[1]), bool), on[1:] != on[:-1]])
    return r - sw * COST, on


def score(Wr, RF, warn):
    r, on = apply_warn(Wr, RF, warn)
    gain = sharpe(r) - sharpe(Wr[:, None])[0]
    frac = on.mean(axis=0)
    gain[(frac < 0.01) | (frac > 0.50)] = -np.inf
    return gain


def lock_path(m, sec, cash, warn):
    """CHIMERA: 30% yearly lock (-> sec until Jan 1) + warning (-> cash)."""
    A, mode, yr, ys = 1.0, "MOM", m.index[0].year, 1.0
    out, prev_mode = [], "MOM"
    for i, d in enumerate(m.index):
        if d.year != yr:
            yr, ys = d.year, A
            if mode == "SEC":
                mode = "MOM"
        if mode in ("MOM", "CASH") and warn is not None:
            mode = "CASH" if (i > 0 and warn[i - 1]) else "MOM"
        r = {"MOM": m.iloc[i], "SEC": sec.iloc[i], "CASH": cash.iloc[i]}[mode]
        A *= (1 + r) * (1 - (COST if mode != prev_mode else 0.0))
        prev_mode = mode
        out.append(A)
        if mode == "MOM" and A / ys - 1 >= LOCK:
            mode = "SEC"
    eq = pd.Series(out, m.index)
    return eq.pct_change().fillna(eq.iloc[0] - 1)


def line(label, r):
    r = np.asarray(r)
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    print(f"    {label:<28} {ann*100:>+6.1f}%/yr  Sharpe {sharpe(r[:, None])[0]:>5.2f}"
          f"  worst {worst(r[:, None])[0]*100:>6.1f}%")
    return sharpe(r[:, None])[0], worst(r[:, None])[0]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = smoke_data() if a.smoke else load_french()
    print(f"  French data {d.index[0].date()} -> {d.index[-1].date()} "
          f"({len(d)} days){'  [SYNTHETIC]' if a.smoke else ''}")
    rank = features(d)
    C, names, feat_of = conditions(rank)
    rules = all_rules(feat_of)
    rname = lambda j: " AND ".join(names[i] for i in rules[j])  # noqa: E731
    print(f"  {len(rules)} rules from {len(FEATS)} warning signs x {len(LEVELS)} levels")

    s1 = (d.index >= P1[0]) & (d.index <= P1[1])
    W1, RF1, C1 = d["W"].values[s1], d["RF"].values[s1], C[s1]

    # 1 SEARCH
    g = score(W1, RF1, warn_matrix(C1, rules))
    order = np.argsort(-g)
    best = int(order[0])
    print("\n  ===== 1 SEARCH 1927-1989 (Sharpe gain vs always holding winners) =====")
    for j in order[:10]:
        print(f"    {g[j]:+.3f}  {rname(j)}")

    # 2 LUCK
    rng = np.random.default_rng(2026)
    n1 = len(W1)
    null = []
    for _ in range(N_PERM):
        k = int(rng.integers(MIN_SHIFT, n1 - MIN_SHIFT))
        null.append(np.max(score(W1, RF1, warn_matrix(np.roll(C1, k, axis=0), rules))))
    null = np.array(null)
    p = (np.sum(null >= g[best]) + 1) / (N_PERM + 1)
    print(f"\n  ===== 2 LUCK: {N_PERM} searches with the signs slid in time =====")
    print(f"    real best gain {g[best]:+.3f} | slid searches: median best "
          f"{np.median(null):+.3f}, 95th pct {np.percentile(null, 95):+.3f} | p = {p:.3f}")

    # 3 LATER
    print(f"\n  ===== 3 LATER 1990-2016, rule fixed: {rname(best)} =====")
    s2 = (d.index >= P2[0]) & (d.index <= P2[1])
    W2, RF2 = d["W"].values[s2], d["RF"].values[s2]
    r2, on2 = apply_warn(W2, RF2, warn_matrix(C[s2], [rules[best]]))
    sb, db = line("winners always", W2)
    sw, dw = line("winners + warning", r2[:, 0])
    print(f"    warning on {on2.mean()*100:.0f}% of days")
    ok3 = sw > sb and dw > db
    print("    top-10 search rules here (context only):")
    for j in order[:10]:
        rj, _ = apply_warn(W2, RF2, warn_matrix(C[s2], [rules[j]]))
        print(f"      Sharpe {sharpe(rj)[0]:.2f} vs {sb:.2f}, worst "
              f"{worst(rj)[0]*100:.0f}% vs {db*100:.0f}%  {rname(j)}")

    # 4 CHIMERA
    print(f"\n  ===== 4 CHIMERA book {P3[0]} -> {P3[1]}, 30% lock =====")
    if a.smoke:
        s3 = (d.index >= P3[0]) & (d.index <= P3[1])
        bk = pd.DataFrame({"STEADY": d["W"][s3] * 1.5, "SPY": d["M"][s3]})
    else:
        bk = pd.read_csv(BOOK, index_col=0, parse_dates=True)
        bk = bk[(bk.index >= P3[0]) & (bk.index <= P3[1])]
    wcol = pd.Series(warn_matrix(C, [rules[best]])[:, 0], d.index)
    cover = d.index[-1]
    w3 = wcol.reindex(bk.index).ffill().fillna(False).astype(bool).values
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    if cover < bk.index[-1]:
        print(f"    note: French data ends {cover.date()}; warning held at its last "
              f"value after that")
    base = lock_path(bk["STEADY"], bk["SPY"], cash, None)
    test = lock_path(bk["STEADY"], bk["SPY"], cash, w3)
    s0, d0 = line("LOCK30", base.values)
    s4, d4 = line("LOCK30 + warning", test.values)
    print(f"    warning on {w3.mean()*100:.0f}% of days")
    for y in sorted(set(bk.index.year)):
        b = (1 + base[base.index.year == y]).prod() - 1
        t = (1 + test[test.index.year == y]).prod() - 1
        print(f"      {y}: LOCK30 {b*100:+5.0f}%   + warning {t*100:+5.0f}%")
    on = pd.Series(w3, bk.index)
    starts = on[on & ~on.shift(1, fill_value=False)].index
    print(f"    warning periods ({len(starts)}): " + ", ".join(
        f"{s:%Y-%m-%d}" for s in starts[:25]) + (" ..." if len(starts) > 25 else ""))
    ok4 = s4 > s0 and d4 > d0

    print("\n  ===== VERDICT (pre-stated) =====")
    print(f"    luck test p = {p:.3f} -> {'pass' if p < 0.05 else 'FAIL'}")
    print(f"    1990-2016 untouched -> {'pass' if ok3 else 'FAIL'}")
    print(f"    CHIMERA 2017-2026   -> {'pass' if ok4 else 'FAIL'}")
    print("    " + ("PASS -- port the warning to QC as a switch"
                    if p < 0.05 and ok3 and ok4 else "FAIL -- keep LOCK30 as is"))
    if a.smoke:
        print("  smoke: synthetic data with a planted warning (MOMVOL should win)")


if __name__ == "__main__":
    main()
