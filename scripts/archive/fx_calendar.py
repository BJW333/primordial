"""
fx_calendar.py -- the last FX screen: three CALENDAR-FLOW ideas (the TSY_CAL family).
PRE-REGISTERED 2026-11-15, before any real-data run. No tuning after results.

Why these: research (2026-11-15) says the classic FX factors are decayed or crash with stocks
(carry ~0 after publication; carry/value/momentum/dollar Sharpe ~0.3-0.55), and our hourly ideas
lost to costs. What's left is forced flows on a known clock -- few trades, so costs matter less.

Data (already on the Mac): data/fx_1h/ (8 majors, Dukascopy hourly 2004 -> 2026-10, via fx_screen),
  data/fomc_dates.csv (scheduled FOMC decision days 2003-2026, from federalreserve.gov),
  data/fx_rates_3m.csv (OECD 3-month interbank rates by month, FRED IR3TIB01xxM156N),
  SPY + EFA daily closes (Yahoo through primordial.data.fetch).
Costs: fx_screen.COST_BP per side (CME futures), every row also at 2x.

  1 MONTH_END  Two trading days before the last trading day of each month, at the 16:00 New York
               close: US stocks' month-to-date return (SPY) minus foreign stocks' (EFA).
               US ahead -> foreign holders of US stocks must sell dollars to re-hedge -> SHORT USD
               vs all 8 majors (1/8 each, $1 total); US behind -> LONG USD. Enter at the 17:00 NY
               close that day, exit at the 4pm London fix on the last trading day (Melvin-Prins 2015).
  2 FOMC_CARRY Day before each scheduled FOMC decision, 17:00 NY: long the 3 highest-rate, short the
               3 lowest-rate of the 8 majors (rates = last month's 3-month interbank, lagged one
               month), $1 / $1; exit at 17:00 NY on decision day (Mueller-Tahbaz-Salehi-Vedolin 2017).
  3 GOTOBI     Tokyo dates 5/10/15/20/25/30 (Feb: last day; weekend -> previous Friday; Japanese
               holidays ignored): long USD/JPY from 08:00 to 10:00 Tokyo, through the 9:55 fix
               (Ito-Yamada 2017). $1.
Controls (must FAIL): MONTH_END_RANDOM (random sign each month), FOMC_RANDOM (same carry book on
  random non-FOMC days, same count per year), GOTOBI_SHIFT (same trade on 3/8/13/18/23/28).
Pass bar (same as always): Sharpe >= 0.5 2012-07 -> 2021-10, >= 0.7 2021-11 -> end, 50/50
  equal-risk monthly blend with CHIMERA > 1.42, whole-period Sharpe > 0 at 2x costs.
  Three ideas judged -> one pass counts for less than it looks.

Run (repo root):
    python3.10 scripts/fx_calendar.py --smoke
    python3.10 scripts/fx_calendar.py | tee logs/fx_calendar.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fx_screen as fx  # noqa: E402

MAJ = ["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "NOK"]
FOMC_CSV = os.path.join(ROOT, "data", "fomc_dates.csv")
RATES_CSV = os.path.join(ROOT, "data", "fx_rates_3m.csv")


# =================================================================== data
def load_equities(start, end):
    from primordial.data import fetch
    d = fetch("yfinance", ["SPY", "EFA"], "1d", start, end, min_bars=200)
    px = pd.DataFrame({s: d[s]["close"] for s in ("SPY", "EFA")})
    px.index = pd.to_datetime(px.index).normalize()
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px[~px.index.duplicated(keep="last")].sort_index()


def load_fomc():
    return sorted(pd.to_datetime(pd.read_csv(FOMC_CSV)["date"]).dt.date)


def load_rates():
    r = pd.read_csv(RATES_CSV, index_col=0, parse_dates=True).ffill()
    return r


# =================================================================== helpers
def hour_index_tz(P, tz):
    return P.index.tz_convert(tz)


def daily_month_end_days(days):
    """last trading day of each month, from a DatetimeIndex of trading days."""
    s = pd.Series(days, index=days)
    return s.groupby(days.to_period("M")).max().values


# =================================================================== 1 MONTH_END
def month_end_positions(P, eq, random_sign=False, seed=21):
    """hourly position frame (USD-per-unit currencies)."""
    C, _ = fx.daily(P)
    days = C.index
    ny = P.index.tz_convert("America/New_York")
    lon = P.index.tz_convert("Europe/London")
    pos = pd.DataFrame(0.0, index=P.index, columns=MAJ)
    rng = np.random.default_rng(seed)
    eqd = eq.reindex(days).ffill()
    n_tr = 0
    for last in daily_month_end_days(days):
        i = days.get_loc(last)
        if i < 3:
            continue
        sig_day = days[i - 2]
        prev_me = days[days < pd.Timestamp(last).to_period("M").start_time]
        if not len(prev_me) or sig_day not in eqd.index:
            continue
        base = eqd.loc[prev_me[-1]]
        now = eqd.loc[sig_day]
        if base.isna().any() or now.isna().any():
            continue
        us_minus_for = (now["SPY"] / base["SPY"]) - (now["EFA"] / base["EFA"])
        sign = 1.0 if us_minus_for > 0 else -1.0          # +1 = short USD = long foreign
        if random_sign:
            sign = rng.choice([-1.0, 1.0])
        # entry: bar starting 16:00 NY on sig_day (closes 17:00 NY); exit: bar starting 15:00 London on `last`
        start_mask = (ny.normalize().tz_localize(None) == sig_day) & (ny.hour == 16)
        end_mask = (lon.normalize().tz_localize(None) == pd.Timestamp(last)) & (lon.hour == 15)
        if not start_mask.any() or not end_mask.any():
            continue
        b0 = np.flatnonzero(start_mask)[0]
        b1 = np.flatnonzero(end_mask)[0]
        if b1 <= b0:
            continue
        pos.iloc[b0:b1, :] = sign / len(MAJ)
        n_tr += 1
    return pos, n_tr


# =================================================================== 2 FOMC_CARRY
def carry_weights(rates, day):
    m = (pd.Timestamp(day).to_period("M") - 1).start_time        # last month's value
    r = rates[rates.index <= m]
    if not len(r):
        return None
    r = r.iloc[-1]
    diff = (r[MAJ] - r["USD"]).dropna()
    if len(diff) < 6:
        return None
    order = diff.sort_values()
    w = pd.Series(0.0, index=MAJ)
    w[order.index[-3:]] = 1 / 3
    w[order.index[:3]] = -1 / 3
    return w


def fomc_daily_positions(C, rates, fomc, random_days=False, seed=31):
    """daily position frame: set on the trading day BEFORE each event day (book earns t -> t+1)."""
    days = C.index
    pos = pd.DataFrame(0.0, index=days, columns=MAJ)
    ev = [pd.Timestamp(d) for d in fomc if pd.Timestamp(d) in days]
    if random_days:
        rng = np.random.default_rng(seed)
        evset, picks = set(ev), []
        for y in sorted({d.year for d in ev}):
            n = sum(1 for d in ev if d.year == y)
            cand = [d for d in days if d.year == y and d not in evset]
            picks += list(rng.choice(cand, size=min(n, len(cand)), replace=False))
        ev = sorted(pd.Timestamp(d) for d in picks)
    n_tr = 0
    for d in ev:
        i = days.get_loc(d)
        if i < 1:
            continue
        w = carry_weights(rates, d)
        if w is None:
            continue
        pos.iloc[i - 1] = w.values
        n_tr += 1
    return pos, n_tr


# =================================================================== 3 GOTOBI
def gotobi_dates(start, end, days_of_month=(5, 10, 15, 20, 25, 30)):
    out = []
    for p in pd.period_range(start, end, freq="M"):
        last = p.days_in_month
        for d in days_of_month:
            dd = min(d, last) if d >= 28 else d          # 30 -> month end in February
            if dd > last:
                continue
            x = dt.date(p.year, p.month, dd)
            while x.weekday() >= 5:                       # weekend -> previous Friday
                x -= dt.timedelta(days=1)
            out.append(x)
    return sorted(set(out))


def gotobi_positions(P, days_of_month=(5, 10, 15, 20, 25, 30)):
    tok = P.index.tz_convert("Asia/Tokyo")
    pos = pd.DataFrame(0.0, index=P.index, columns=["JPY"])
    dates = set(gotobi_dates(str(P.index[0].date()), str(P.index[-1].date()), days_of_month))
    tdate = np.array([x.date() for x in tok])
    hr = np.asarray(tok.hour)
    in_date = np.array([d in dates for d in tdate])
    # earn the bars 08:00-09:00 and 09:00-10:00 Tokyo -> hold from the close of the 07:00 bar
    hold = in_date & ((hr == 7) | (hr == 8))
    pos.loc[hold, "JPY"] = -1.0                           # short yen = long USD/JPY
    n_tr = int(((hr == 7) & in_date).sum())
    return pos, n_tr


# =================================================================== judging
def judge_row(name, x, x2, n_tr, years, judged=True):
    a0, _, s0, _ = fx.stats(fx.win(x, fx.EARLY))
    a1, v1, s1, d1 = fx.stats(fx.win(x, fx.RECENT))
    _, _, sw, _ = fx.stats(x)
    _, _, sw2, _ = fx.stats(x2)
    csh, bsh, corr, bad = fx.blend(x)
    ok = (s0 >= 0.5) and (s1 >= 0.7) and (bsh > csh) and (sw2 > 0)
    v = ("PASS" if ok else "fail") if judged else "(info)"
    print(f"  {name:<19}| {a0*100:+6.2f}% {s0:5.2f} | {a1*100:+6.2f}% {v1*100:4.1f}% {s1:5.2f}"
          f" {d1*100:5.0f}% | {sw:6.2f} {sw2:7.2f} | {corr:+6.2f} {csh:5.2f} {bsh:5.2f}"
          f" {bad*100:+5.2f}% | {n_tr / years:8.0f} | {v}")
    return ok


def run(P, eq, fomc, rates):
    R = fx.hourly_returns(P)
    C, E = fx.daily(P)
    years = len(C) / 252
    print(f"\n  data {P.index[0].date()} -> {P.index[-1].date()}, {len(C):,} days, "
          f"{len(fomc)} FOMC dates")
    print(f"  {'idea':<19}| 2012-21 ann   shp | last5y ann   vol  shp  worst | whole shp 2x-cost |"
          f" corrCHIM CHIM blend  badM | trades/yr | verdict")
    res = {}
    for name, f, judged in [
        ("1 MONTH_END", lambda: month_end_positions(P, eq), True),
        ("2 FOMC_CARRY", None, True),
        ("3 GOTOBI", lambda: gotobi_positions(P), True),
        ("control ME_RANDOM", lambda: month_end_positions(P, eq, random_sign=True), True),
        ("control FOMC_RANDOM", None, True),
        ("control GOTOBI_SHIFT", lambda: gotobi_positions(P, (3, 8, 13, 18, 23, 28)), True)]:
        if "FOMC" in name:
            pos, n = fomc_daily_positions(C, rates, fomc, random_days="RANDOM" in name)
            x, x2 = fx.book(pos, E[MAJ].diff(), 1.0), fx.book(pos, E[MAJ].diff(), 2.0)
        else:
            pos, n = f()
            x, x2 = fx.hourly_book(pos, R, 1.0), fx.hourly_book(pos, R, 2.0)
        res[name] = judge_row(name, x, x2, n, years, judged)
    bad = [k for k, v in res.items() if k.startswith("control") and v]
    print("  controls: " + ("all fail (good)" if not bad else
                            f"WARNING {', '.join(bad)} passed -> the test setup is suspect"))
    print("  ann = average yearly return (log) at $1 exposure while in a trade (flat otherwise);"
          " pass = 2012-21 >= 0.5, last5y >= 0.7,\n  blend > CHIM, 2x-cost > 0. badM = avg month"
          " when CHIMERA < -3%.")


# =================================================================== smoke
def synth(planted, seed=4):
    P, _ = fx.synth(False, seed=seed)
    P = P[["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "NOK"]].copy()
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2004-01-01", "2026-10-30")
    eq = pd.DataFrame({"SPY": np.exp(np.cumsum(rng.normal(0.0004, 0.011, len(days)))),
                       "EFA": np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(days))))}, index=days)
    rates = pd.DataFrame(rng.normal(2, 1.5, (300, 9)).cumsum(0) * 0.05 + 2,
                         index=pd.date_range("2002-01-01", periods=300, freq="MS"),
                         columns=["USD"] + MAJ)
    fomc = []
    for y in range(2004, 2027):
        for m in (1, 3, 4, 6, 7, 9, 10, 12):
            d = pd.Timestamp(y, m, 15) + pd.offsets.Week(weekday=2)
            if d <= pd.Timestamp("2026-10-28"):
                fomc.append(d.date())
    if planted:
        L = np.log(P)
        R = L.diff().fillna(0)
        me, _ = month_end_positions(P, eq)
        R = R + me.shift(1).fillna(0) * len(MAJ) * 0.0010          # 10 bp/hour when the signal is on
        gp, _ = gotobi_positions(P)
        R["JPY"] = R["JPY"] + gp["JPY"].shift(1).fillna(0) * 0.0015  # 15 bp/hour into the fix
        P = np.exp(R.cumsum())
        C, E = fx.daily(P)
        fp, _ = fomc_daily_positions(C, rates, fomc)
        bump = (fp * 0.008)                                           # 80 bp/day on FOMC days
        hours = P.index.tz_convert("America/New_York")
        hd = pd.DatetimeIndex(hours.normalize().tz_localize(None))
        add = pd.DataFrame(0.0, index=P.index, columns=MAJ)
        nxt = bump.shift(1).fillna(0)                                 # earned on the event day
        for d in nxt.index[(nxt != 0).any(axis=1)]:
            m = (hd == d) & (hours.hour == 12)
            add.loc[m] = nxt.loc[d].values
        P = np.exp(np.log(P).diff().fillna(0).add(add).cumsum())
    return P, eq, fomc, rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        for planted in (False, True):
            print(f"\n===== SMOKE {'PLANTED (rows 1-3 should PASS, controls fail)' if planted else 'NOISE (everything should fail)'}")
            run(*synth(planted))
        return
    P = fx.load_real([c for c in fx.INST if c in MAJ])
    eq = load_equities(fx.START, fx.END)
    run(P, eq, load_fomc(), load_rates())


if __name__ == "__main__":
    main()
