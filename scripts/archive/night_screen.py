"""
night_screen.py -- the NIGHT idea on cheap, liquid funds: a STOCK version and a COMMODITY version.
PRE-REGISTERED 2026-11-15, before any run on this data. No tuning after results.

Background: structural_screen.py showed CHIMERA's momentum picks earn everything overnight but
10,000 trades a year made costs decisive. A quick look at 2016-2026 showed gold and most commodity
funds also earn overnight. This screen tests the same rule on funds that cost almost nothing to
trade, including years that have NOT been looked at (everything before 2016).

Rule: buy at the close, sell at the next open, every day (closing / opening auction orders).
Data: Yahoo daily open and close, dividend-adjusted, each fund from its first day.
Costs per side: SPY QQQ GLD 1 bp; IWM SLV USO 2 bp; DBC 3 bp. Every row also at 2x.

Judged books (three, equal weight across the funds that exist that day):
  STOCK_NIGHT   SPY + QQQ + IWM
  COMMOD_NIGHT  GLD + SLV + USO + DBC
  ALL_NIGHT     half STOCK_NIGHT + half COMMOD_NIGHT
PASS (all of): Sharpe >= 0.5 in 2012-07 -> 2021-10; >= 0.7 in 2021-11 -> end; 50/50 equal-risk blend
  with CHIMERA > 1.42; whole-period Sharpe > 0 at 2x costs; whole-period Sharpe above simply holding
  the same funds all day; and in the OLD years (start -> 2012-06, never looked at) Sharpe > 0 and
  above holding all day.
Controls (must NOT pass): each book with every fund-day randomly taking the night OR the day leg.
Info rows: the day-only mirror, buy-and-hold, and each fund's night leg on its own.

Run (repo root):
    python3.10 scripts/night_screen.py --smoke
    python3.10 scripts/night_screen.py | tee logs/night_screen.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx_screen as fx  # noqa: E402
import structural_screen as ss  # noqa: E402

STOCK = {"SPY": 1.0, "QQQ": 1.0, "IWM": 2.0}
COMMOD = {"GLD": 1.0, "SLV": 2.0, "USO": 2.0, "DBC": 3.0}
PRE = ("1990-01-01", "2012-07-01")
WINS = (PRE, fx.EARLY, fx.RECENT)


def load():
    import yfinance as yf
    t = list(STOCK) + list(COMMOD)
    d = yf.download(t, start="1993-01-01", auto_adjust=False, progress=False, group_by="column")
    k = d["Adj Close"] / d["Close"]
    O, C = (d["Open"] * k), d["Adj Close"]
    O.index = C.index = pd.to_datetime(C.index).tz_localize(None) if getattr(C.index, "tz", None) else pd.to_datetime(C.index)
    return O, C


def legs(O, C):
    night = O / C.shift(1) - 1
    day = C / O - 1
    bad = (night.abs() > 0.3) | (day.abs() > 0.3) | (O <= 0)
    return night.where(~bad), day.where(~bad)


def book(night, day, costs, cmult=1.0, mode="night", seed=0):
    """Equal weight across funds with data that day. mode: night | day | hold | shuf."""
    cols = [c for c in costs if c in night]
    n, d = night[cols], day[cols]
    c = pd.Series(costs)[cols] * 1e-4 * cmult
    if mode == "hold":
        x = (1 + n) * (1 + d) - 1
    elif mode == "day":
        x = d - 2 * c
    elif mode == "shuf":
        flip = np.random.default_rng(seed).random(n.shape) < 0.5
        x = n.where(~flip, d) - 2 * c
    else:
        x = n - 2 * c
    cnt = x.notna().sum(axis=1)
    out = x.mean(axis=1)
    return out[cnt >= min(2, len(cols))].dropna()


def half(a, b):
    j = pd.concat([a, b], axis=1).dropna()
    return 0.5 * j.iloc[:, 0] + 0.5 * j.iloc[:, 1]


def row(name, x, extra=""):
    p = [(ss.ann_d(ss.w(x, wn)), ss.sh_d(ss.w(x, wn))) for wn in WINS]
    b, c, bad = ss.blend_d(x)
    print(f"  {name:<19}| {p[0][0]*100:+6.1f}% {p[0][1]:5.2f} | {p[1][0]*100:+6.1f}% {p[1][1]:5.2f} |"
          f" {p[2][0]*100:+6.1f}% {p[2][1]:5.2f} | {ss.ann_d(x)*100:+6.1f}% {ss.sh_d(x):5.2f} {ss.dd(x)*100:5.0f}% |"
          f" {c:+5.2f} {b:5.2f} {bad*100:+5.1f}% | {extra}")
    return dict(pre=p[0][1], early=p[1][1], recent=p[2][1], whole=ss.sh_d(x), blend=b)


def verdict(N, N2, H):
    r = dict(pre=ss.sh_d(ss.w(N, PRE)), early=ss.sh_d(ss.w(N, fx.EARLY)), recent=ss.sh_d(ss.w(N, fx.RECENT)))
    b, _, _ = ss.blend_d(N)
    hp = ss.sh_d(ss.w(H.reindex(N.index), PRE))
    return bool(r["early"] >= 0.5 and r["recent"] >= 0.7 and b > ss.CHIM_SH and ss.sh_d(N2) > 0
                and ss.sh_d(N) > ss.sh_d(H.reindex(N.index)) and r["pre"] > 0 and r["pre"] > hp)


def report(O, C):
    night, day = legs(O, C)
    HDR = (f"  {'row':<19}| old yrs ann  shp | 2012-21 ann  shp | last5y ann   shp |  whole ann  shp worst |"
           f" corr  blend  badM |")
    books = {}
    for name, costs in (("STOCK", STOCK), ("COMMOD", COMMOD)):
        books[name] = {m: book(night, day, costs, mode=m) for m in ("night", "day", "hold", "shuf")}
        books[name]["night2"] = book(night, day, costs, cmult=2.0)
        books[name]["night0"] = book(night, day, costs, cmult=0.0)
        books[name]["shuf2"] = book(night, day, costs, cmult=2.0, mode="shuf")
    books["ALL"] = {k: half(books["STOCK"][k], books["COMMOD"][k]) for k in books["STOCK"]}
    res = {}
    for name in ("STOCK", "COMMOD", "ALL"):
        b = books[name]
        start = b["night"].index[0].date()
        print(f"\n  ===== {name}_NIGHT (from {start}) =====")
        print(HDR)
        ok = verdict(b["night"], b["night2"], b["hold"])
        okc = verdict(b["shuf"], b["shuf2"], b["hold"])
        row("NIGHT net", b["night"], "PASS" if ok else "fail")
        row("NIGHT at 2x cost", b["night2"])
        row("NIGHT before cost", b["night0"])
        row("DAY net", b["day"], "(info)")
        row("hold all day", b["hold"].reindex(b["night"].index), "(reference)")
        row("ctl SHUFFLED", b["shuf"], "PASS (bad!)" if okc else "fails (good)")
        res[name] = (ok, okc)
    print("\n  ===== each fund, night leg only, net of its cost (info, not judged) =====")
    print(HDR)
    for t, c in {**STOCK, **COMMOD}.items():
        if t in night:
            row(t + " night", (night[t] - 2 * c * 1e-4).dropna())
            row(t + " day", (day[t] - 2 * c * 1e-4).dropna())
    print("\n  pass = 2012-21 >= 0.5, last5y >= 0.7, blend with CHIMERA > 1.42, 2x cost > 0, Sharpe above"
          " holding all day,\n         and in the old years (never looked at) Sharpe > 0 and above holding all day.")
    print("  VERDICTS: " + ", ".join(f"{k}_NIGHT {'PASS' if v[0] else 'fail'}" for k, v in res.items())
          + " | controls: " + ("all fail (good)" if not any(v[1] for v in res.values()) else "A CONTROL PASSED"))
    return res


def smoke():
    rng = np.random.default_rng(2)
    days = pd.bdate_range("2000-01-03", "2026-10-30")
    n = len(days)
    for planted in (False, True):
        cols = list(STOCK) + list(COMMOD)
        mk = rng.normal(0, 0.006, (n, 1))
        nt = rng.normal(0.0012 if planted else 0.00015, 0.006, (n, 7)) + mk * 0.6
        dy = rng.normal(-0.0002 if planted else 0.00015, 0.009, (n, 7)) + rng.normal(0, 0.005, (n, 1))
        C = pd.DataFrame(50 * np.exp(np.cumsum(nt + dy, 0)), index=days, columns=cols)
        O = C.shift(1) * (1 + nt)
        print(f"\n===== SMOKE {'PLANTED (books should PASS, controls fail)' if planted else 'NOISE (everything should fail)'}")
        report(O, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        return smoke()
    O, C = load()
    print("  first day per fund: " + ", ".join(f"{t} {C[t].first_valid_index().date()}" for t in C.columns))
    report(O, C)


if __name__ == "__main__":
    main()
