"""
structural_screen.py -- four STRUCTURAL ideas from the 2026-11-15 list that can be tested on data
we can get for free. PRE-REGISTERED 2026-11-15, before any real-data run. No tuning after results.
(Blake's brief: no reliance on big funds' forced flows, no calendar events, stocks long-only.)

  1 NIGHT     CHIMERA's momentum picks (momentum_v2 STEADY, top 20, monthly) held ONLY overnight:
              buy at the close, sell at the next open, every day (auction orders, 2 bp per side).
              Reference rows: FULL (the same picks held normally) and DAY (open -> close only).
              PASS = in BOTH windows NIGHT's Sharpe beats FULL's and it keeps >= 75% of FULL's
              annual return, AND its whole-period Sharpe still beats FULL's at 2x costs.
              Controls: NIGHT_SHUF (each stock-day randomly takes the night or the day leg, same
              costs; must NOT pass). Info row: NIGHT_RANDOM20 (night-only on 20 random liquid stocks).
  2 LOWVOL    Each month hold the 50 least volatile liquid US stocks (252-day vol), equal weight,
              10 bp per trade. PASS = Sharpe above SPY's in both windows AND the 50/50 equal-risk
              blend with CHIMERA beats CHIMERA's 1.42. Control RANDOM50 must NOT pass.
  3 VIXCARRY  When the VIX curve slopes up (3-month VIX / VIX > 1.05) be short 25% VIXY (front VIX
              futures) the next day; otherwise flat. 10 bp per switch. Reference: ALWAYS short 25%.
              PASS = standard bar (Sharpe >= 0.5 2012-07 -> 2021-10, >= 0.7 2021-11 -> end, blend
              > 1.42, > 0 at 2x costs) AND Sharpe above ALWAYS AND its average month when CHIMERA
              loses more than 3% is no worse than -1%.
  4 CEF       ~110 closed-end funds, price and NAV from Yahoo. Discount = price / NAV - 1,
              z-scored against its own last 252 days. Each week buy funds with z <= -2 (widest
              first, max 15, 1/15 each), sell when z >= 0 or after 126 days. Signal at the close,
              trade the next close, 35 bp per trade. Reference EW_ALL (all funds, equal weight).
              PASS = standard bar AND whole-period Sharpe above EW_ALL.
              Control CEF_PERMUTED (each fund trades on ANOTHER fund's discount signal; if this
              passes too, the edge is "buy funds in panics", not the discount) must NOT pass.
Windows: stock ideas (1, 2) 2017-02 -> 2021-10 and 2021-11 -> 2026-05 (data/us_universe starts
2016); VIXCARRY and CEF the usual 2012-07 -> 2021-10 and 2021-11 -> end.

Run (repo root):
    python3.10 scripts/structural_screen.py --smoke
    export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...      # NIGHT needs daily opens (one-time fetch)
    python3.10 scripts/structural_screen.py | tee logs/structural_screen.txt
    python3.10 scripts/structural_screen.py --only vix cef      # run a subset
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fx_screen as fx  # noqa: E402  (CHIM table + blend only)

UNI = os.path.join(ROOT, "data", "us_universe")
CACHE = os.path.join(ROOT, "data", "structural")
S_EARLY = ("2017-02-01", "2021-11-01")
S_RECENT = ("2021-11-01", "2100-01-01")
CHIM_SH = 1.42

CEFS = """ADX GAM CET USA ASG GDV GAB GUT GGT BST BME BDJ BGY BOE CII BGR BCX ETV ETW ETY EXG ETG
EOS EOI ETB ETJ EVT NFJ CSQ CHI CHY CHW CGO HTD PDT HPI HPS HPF PDI PTY PCN PFN PFL PHK PCM DSL DBL
EVV EFR EFT FRA BGT JQC JFR NZF NEA NAD NVG NUV BLE MQY MYI MUI IIM VMO VKQ VGM KTF LEO EIM UTG UTF
DNP RQI RNP RFI NRO JRS AWP IGR KYN TYG GOF GBAB AVK BTZ HYT HIO HIX EAD ERC FAX FCO IAF JOF JEQ TWN
IFN MXF TDF CHN KF EEA SWZ RVT RMT SPE CLM CRF TY MCI MPV PPT PMM PIM BHK BKT BLW BIT DSU FFC FLC
FPF JPC JPS NCV NCZ ACP AOD THQ HQH HQL BMEZ""".split()


# =================================================================== shared stats
def sh_d(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 60 and x.std() > 0 else np.nan


def ann_d(x):
    x = x.dropna()
    return (1 + x).prod() ** (252 / len(x)) - 1 if len(x) > 60 else np.nan


def dd(x):
    q = (1 + x.fillna(0)).cumprod()
    return (q / q.cummax() - 1).min()


def w(x, win):
    return x[(x.index >= win[0]) & (x.index < win[1])]


def blend_d(x):
    """(blend Sharpe, corr, avg month when CHIMERA < -3%) from DAILY simple returns."""
    m = (1 + x.fillna(0)).groupby(x.index.to_period("M")).prod() - 1
    return blend_m(m)


def blend_m(m):
    m = m.copy()
    m.index = [str(p)[:7] for p in m.index]
    j = pd.concat([pd.Series(fx.CHIM), m], axis=1, keys=["c", "s"]).dropna()
    if len(j) < 36 or j["s"].std() == 0:
        return np.nan, np.nan, np.nan
    shm = lambda v: v.mean() / v.std() * np.sqrt(12)  # noqa: E731
    b = 0.5 * j["c"] + 0.5 * j["s"] * (j["c"].std() / j["s"].std())
    return shm(b), j["c"].corr(j["s"]), j.loc[j["c"] < -0.03, "s"].mean()


def line(name, x, wins, extra=""):
    a0, s0 = ann_d(w(x, wins[0])), sh_d(w(x, wins[0]))
    a1, s1 = ann_d(w(x, wins[1])), sh_d(w(x, wins[1]))
    b, c, bad = blend_d(x)
    print(f"  {name:<17}| {a0*100:+6.1f}% {s0:5.2f} | {a1*100:+6.1f}% {s1:5.2f} | {ann_d(x)*100:+6.1f}%"
          f" {sh_d(x):5.2f} {dd(x)*100:5.0f}% | {c:+5.2f} {b:5.2f} {bad*100:+5.1f}% | {extra}")
    return dict(a0=a0, s0=s0, a1=a1, s1=s1, sw=sh_d(x), blend=b, bad=bad)


HDR = (f"  {'row':<17}| early ann   shp | recent ann  shp |  whole ann  shp worst | corr  blend  badM | ")


# =================================================================== 1 NIGHT
def night_engine(O, C, windows, c_side, seed=0, shuffle=False):
    """O, C: adjusted open/close (days x symbols). windows: list of (start_day, end_day, picks).
    Returns daily simple-return Series: FULL, NIGHT, DAY (net of costs)."""
    on = O / C.shift(1) - 1
    idr = C / O - 1
    rng = np.random.default_rng(seed)
    full, night, day = {}, {}, {}
    prev = set()
    for t, t2, picks in windows:
        picks = [s for s in picks if s in C.columns]
        days = C.index[(C.index > t) & (C.index <= t2)]
        if not len(days) or not picks:
            continue
        a, b = on.loc[days, picks], idr.loc[days, picks]
        ok = a.notna() & b.notna() & (a.abs() < 0.5) & (b.abs() < 0.5)
        a, b = a.where(ok), b.where(ok)
        if shuffle:
            flip = rng.random(a.shape) < 0.5
            a, b = a.where(~flip, b), b.where(~flip, a)
        f = ((1 + a) * (1 + b) - 1).mean(axis=1)
        turn = (len(set(picks) - prev) + len(prev - set(picks))) / max(len(picks), 1)
        f.iloc[0] -= turn * 0.0020                      # monthly rebalance cost, as momentum_v2
        prev = set(picks)
        for d in days:
            full[d] = f[d]
            night[d] = a.loc[d].mean() - 2 * c_side
            day[d] = b.loc[d].mean() - 2 * c_side
    return pd.Series(full), pd.Series(night), pd.Series(day)


def night_verdict(F, N, N2x):
    ok = True
    for win in (S_EARLY, S_RECENT):
        ok &= sh_d(w(N, win)) > sh_d(w(F, win)) and ann_d(w(N, win)) >= 0.75 * ann_d(w(F, win))
    return bool(ok and sh_d(N2x) > sh_d(F))


def _alpaca_bars(chunk, start, end, h):
    import requests
    rows, tok = [], None
    while True:
        prm = {"symbols": ",".join(chunk), "timeframe": "1Day", "start": start, "end": end,
               "limit": 10000, "adjustment": "all", "feed": "sip"}
        if tok:
            prm["page_token"] = tok
        for a in range(6):
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars", params=prm,
                             headers=h, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (a + 1))
                continue
            break
        if r.status_code == 400 and len(chunk) > 1:      # one bad symbol: go one by one
            for s_ in chunk:
                try:
                    rows += _alpaca_bars([s_], start, end, h)
                except Exception:  # noqa: BLE001
                    pass
            return rows
        r.raise_for_status()
        js = r.json()
        for sym, bars in (js.get("bars") or {}).items():
            rows += [(sym, b["t"][:10], b["o"], b["c"]) for b in bars]
        tok = js.get("next_page_token")
        if not tok:
            return rows


def fetch_open_close(syms, start, end):
    """Adjusted daily open/close for syms, cached; saved every 200 symbols so a rerun resumes."""
    import re
    os.makedirs(CACHE, exist_ok=True)
    po, pc = os.path.join(CACHE, "night_open.parquet"), os.path.join(CACHE, "night_close.parquet")
    O = C = None
    if os.path.exists(po):
        O, C = pd.read_parquet(po), pd.read_parquet(pc)
    have = set(O.columns) if O is not None else set()
    need = sorted(s for s in set(syms) - have if re.match(r"^[A-Z]{1,5}(\.[AB])?$", s))
    if need:
        k, s = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not k or not s:
            sys.exit("  NIGHT needs daily opens: export ALPACA_API_KEY and ALPACA_SECRET_KEY first "
                     "(or run with --only lowvol vix cef)")
        h = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}
        for j in range(0, len(need), 200):
            batch, rows = need[j:j + 200], []
            for i in range(0, len(batch), 40):
                rows += _alpaca_bars(batch[i:i + 40], start, end, h)
            df = pd.DataFrame(rows, columns=["s", "d", "o", "c"])
            df["d"] = pd.to_datetime(df["d"])
            o2 = df.pivot(index="d", columns="s", values="o")
            c2 = df.pivot(index="d", columns="s", values="c")
            for s_ in batch:                             # remember symbols with no data too
                if s_ not in o2:
                    o2[s_], c2[s_] = np.nan, np.nan
            O = o2 if O is None else O.join(o2, how="outer")
            C = c2 if C is None else C.join(c2, how="outer")
            O.sort_index().to_parquet(po)
            C.sort_index().to_parquet(pc)
            print(f"    opens: {min(j + 200, len(need))}/{len(need)} symbols", flush=True)
    return O.sort_index(), C.sort_index()


def run_night():
    import blowup_scan as B
    import blowup_us as U
    import momentum_v2 as M
    px, vol, spy, feed = U.load()
    U.MIN_PX = M.MIN_PX
    sets, ever = U.liquid_members(px, vol, M.MIN_DV)
    px = px[ever]
    spy = spy.reindex(px.index).ffill()
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731
    B.PICKLOG.clear()
    M.run(px, spy, members)
    log = pd.DataFrame([r for r in B.PICKLOG if r[1] == "STEADY"], columns=["t2", "v", "s", "r"])
    ends = sorted(log.t2.unique())
    idx = px.index
    windows, rnd_windows = [], []
    rng = np.random.default_rng(3)
    for t2 in ends:
        t2 = pd.Timestamp(t2)
        prior = idx[idx < t2.to_period("M").start_time]
        if not len(prior):
            continue
        t = prior[-1]
        windows.append((t, t2, list(log[log.t2 == t2].s)))
        mem = sorted(members(t))
        rnd_windows.append((t, t2, list(rng.choice(mem, size=min(20, len(mem)), replace=False))))
    syms = sorted({s for _, _, p in windows + rnd_windows for s in p})
    print(f"  NIGHT: {len(windows)} months, {len(syms)} symbols need daily opens")
    O, C = fetch_open_close(syms, "2016-12-01", "2026-06-02")
    F, N, D = night_engine(O, C, windows, 0.0002)
    _, N2, _ = night_engine(O, C, windows, 0.0004)
    _, N0, D0 = night_engine(O, C, windows, 0.0)
    _, NS, _ = night_engine(O, C, windows, 0.0002, shuffle=True)
    FR, NR, _ = night_engine(O, C, rnd_windows, 0.0002)
    wins = (S_EARLY, S_RECENT)
    print("\n  ===== 1 NIGHT (momentum picks held overnight only) =====")
    print(HDR)
    line("FULL (normal)", F, wins)
    line("NIGHT net", N, wins, "PASS" if night_verdict(F, N, N2) else "fail")
    line("NIGHT at 2x cost", N2, wins)
    line("NIGHT before cost", N0, wins)
    line("DAY before cost", D0, wins)
    line("ctl NIGHT_SHUF", NS, wins, "PASS (bad!)" if night_verdict(F, NS, NS) else "fails (good)")
    line("RANDOM20 full", FR, wins)
    line("RANDOM20 night", NR, wins, "(info)")
    print("  pass = NIGHT net Sharpe > FULL and >= 75% of FULL's return in both windows, and Sharpe > FULL at 2x cost")


# =================================================================== 2 LOWVOL
def lowvol_engine(px, members, n=50, cost=0.0010, random=False, seed=5):
    r = px.pct_change(fill_method=None)
    pxf = px.ffill()
    me = [d for d in px.resample("ME").last().index]
    idx = px.index
    rng = np.random.default_rng(seed)
    out, prev = {}, set()
    for k in range(len(me) - 1):
        t = idx[idx <= me[k]][-1]
        t2 = idx[idx <= me[k + 1]][-1]
        ti = idx.get_loc(t)
        if ti < 260:
            continue
        mem = [s for s in members(t) if s in px.columns]
        cnt = px.iloc[ti - 252:ti + 1][mem].notna().sum()
        elig = [s for s in mem if cnt[s] >= 240]
        if len(elig) < 200:
            continue
        if random:
            pick = list(rng.choice(elig, size=n, replace=False))
        else:
            v = r[elig].iloc[ti - 251:ti + 1].std()
            pick = list(v.sort_values().index[:n])
        ret = (pxf.loc[t2, pick] / px.loc[t, pick] - 1).fillna(0).mean()
        turn = (len(set(pick) - prev) + len(prev - set(pick))) / n
        out[t2] = ret - turn * cost
        prev = set(pick)
    return pd.Series(out)


def m_stats(m, win=None):
    if win:
        m = m[(m.index >= win[0]) & (m.index < win[1])]
    return (1 + m).prod() ** (12 / len(m)) - 1, m.mean() / m.std() * np.sqrt(12)


def lowvol_report(L, R, SPYm):
    print("\n  ===== 2 LOWVOL (50 least volatile liquid stocks, monthly) =====")
    print(f"  {'row':<17}| early ann   shp | recent ann  shp |  whole ann  shp worst | corr  blend  badM |")
    res = {}
    for name, m in [("LOWVOL", L), ("SPY", SPYm.reindex(L.index)), ("ctl RANDOM50", R)]:
        a0, s0 = m_stats(m, S_EARLY)
        a1, s1 = m_stats(m, S_RECENT)
        aw, sw = m_stats(m)
        q = (1 + m).cumprod()
        b, c, bad = blend_m(m.copy().rename(lambda d: d.to_period("M")))
        res[name] = (s0, s1, b)
        print(f"  {name:<17}| {a0*100:+6.1f}% {s0:5.2f} | {a1*100:+6.1f}% {s1:5.2f} | {aw*100:+6.1f}%"
              f" {sw:5.2f} {(q / q.cummax() - 1).min()*100:5.0f}% | {c:+5.2f} {b:5.2f} {bad*100:+5.1f}% |")
    def ok(k):
        return res[k][0] > res["SPY"][0] and res[k][1] > res["SPY"][1] and res[k][2] > CHIM_SH
    print(f"  LOWVOL: {'PASS' if ok('LOWVOL') else 'fail'}   control RANDOM50: "
          f"{'PASS (bad!)' if ok('ctl RANDOM50') else 'fails (good)'}")
    print("  pass = Sharpe above SPY in both windows AND blend with CHIMERA > 1.42")
    return ok("LOWVOL")


def run_lowvol():
    import blowup_us as U
    px, vol, spy, feed = U.load()
    U.MIN_PX = 10.0
    sets, ever = U.liquid_members(px, vol, 20e6)
    px = px[ever]
    days = sorted(sets)
    members = lambda t: sets[max(d for d in days if d <= t)]  # noqa: E731
    L = lowvol_engine(px, members)
    R = lowvol_engine(px, members, random=True)
    spy = spy.reindex(px.index).ffill()
    SPYm = spy.loc[[px.index[px.index <= d][-1] for d in L.index]].pct_change()
    SPYm.index = L.index
    SPYm.iloc[0] = spy.loc[L.index[0]] / spy.loc[px.index[px.index < L.index[0].to_period("M").start_time][-1]] - 1
    lowvol_report(L, R, SPYm)


# =================================================================== 3 VIXCARRY
def vix_engine(vix, vix3m, vixy_ret, cmult=1.0, always=False):
    ratio = (vix3m / vix).reindex(vixy_ret.index).ffill()
    pos = pd.Series(np.where(ratio > 1.05, -0.25, 0.0), index=vixy_ret.index)
    if always:
        pos[:] = -0.25
    held = pos.shift(1).fillna(0)
    return held * vixy_ret - pos.diff().abs().fillna(0) * 0.0010 * cmult, (pos != 0).mean()


def vix_report(vix, vix3m, vixy_ret):
    x, frac = vix_engine(vix, vix3m, vixy_ret)
    x2, _ = vix_engine(vix, vix3m, vixy_ret, cmult=2.0)
    al, _ = vix_engine(vix, vix3m, vixy_ret, always=True)
    wins = (fx.EARLY, fx.RECENT)
    print(f"\n  ===== 3 VIXCARRY (short 25% front VIX futures when the curve slopes up; on {frac*100:.0f}% of days) =====")
    print(HDR)
    r = line("VIXCARRY", x, wins)
    line("VIXCARRY 2x cost", x2, wins)
    ra = line("ALWAYS short", al, wins, "(reference)")
    ok = (r["s0"] >= 0.5 and r["s1"] >= 0.7 and r["blend"] > CHIM_SH and sh_d(x2) > 0
          and r["sw"] > ra["sw"] and r["bad"] >= -0.01)
    print(f"  VIXCARRY: {'PASS' if ok else 'fail'}")
    print("  pass = early >= 0.5, recent >= 0.7, blend > 1.42, 2x cost > 0, Sharpe > ALWAYS, badM >= -1%")
    return ok


def yf_close(tickers, start="2009-01-01", adj=False):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 50):
        d = yf.download(tickers[i:i + 50], start=start, auto_adjust=False, progress=False,
                        group_by="column", threads=True)
        col = "Adj Close" if adj else "Close"
        if col in d:
            for t in d[col].columns:
                out[t] = d[col][t]
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    return df.sort_index()


def run_vix():
    c = yf_close(["^VIX", "^VIX3M"])
    v = yf_close(["VIXY"], adj=True)["VIXY"].dropna()
    vix_report(c["^VIX"], c["^VIX3M"], v.pct_change().dropna())


# =================================================================== 4 CEF
def cef_engine(price, nav, tr, cmult=1.0, permute=False, seed=9, max_n=15, cost=0.0035):
    """price, nav: unadjusted closes; tr: total-return (adjusted) closes. Daily simple returns."""
    d = (price / nav - 1).where(lambda z: z.abs() < 0.6)
    z = (d - d.rolling(252, min_periods=200).mean()) / d.rolling(252, min_periods=200).std()
    if permute:
        cols = list(z.columns)
        perm = list(np.random.default_rng(seed).permutation(cols))
        while any(a == b for a, b in zip(cols, perm)):
            perm = perm[1:] + perm[:1]
        z = z[perm].set_axis(cols, axis=1)
    ret = tr.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 0.5)
    days = z.index
    wk = pd.Series(days.to_period("W-FRI"), index=days)
    is_wk_end = (wk != wk.shift(-1)).values
    held, age = {}, {}
    W = np.zeros((len(days), z.shape[1]))
    ci = {c: i for i, c in enumerate(z.columns)}
    trades = 0
    for i, t in enumerate(days):
        zt = z.iloc[i]
        for s in list(held):
            age[s] += 1
            if (np.isfinite(zt[s]) and zt[s] >= 0) or age[s] >= 126:
                del held[s], age[s]
                trades += 1
        if is_wk_end[i]:
            cand = zt[(zt <= -2)].sort_values()
            for s in cand.index:
                if len(held) >= max_n:
                    break
                if s not in held:
                    held[s], age[s] = 1, 0
                    trades += 1
        for s in held:
            W[i, ci[s]] = 1 / max_n
    Wd = pd.DataFrame(W, index=days, columns=z.columns)
    eff = Wd.shift(2).fillna(0)                 # signal at close t, trade at close t+1, earn from t+2
    pnl = (eff * ret).sum(axis=1) - Wd.diff().abs().fillna(0).sum(axis=1).shift(1).fillna(0) * cost * cmult
    return pnl, trades, (Wd.sum(axis=1) > 0).mean(), Wd.sum(axis=1).mean()


def cef_report(price, nav, tr):
    wins = (fx.EARLY, fx.RECENT)
    x, n, inm, avgw = cef_engine(price, nav, tr)
    x2, _, _, _ = cef_engine(price, nav, tr, cmult=2.0)
    xp, _, _, _ = cef_engine(price, nav, tr, permute=True)
    ew = tr.pct_change(fill_method=None).where(lambda r: r.abs() < 0.5).mean(axis=1)
    yrs = len(x) / 252
    print(f"\n  ===== 4 CEF discount snap-back ({price.shape[1]} funds, {n / yrs:.0f} trades/yr, "
          f"invested {inm*100:.0f}% of days, avg {avgw*100:.0f}% of capital) =====")
    print(HDR)
    r = line("CEF", x, wins)
    line("CEF 2x cost", x2, wins)
    re = line("EW_ALL funds", ew, wins, "(reference)")
    rp = line("ctl CEF_PERMUTED", xp, wins)
    def std(rr, xx2):
        return rr["s0"] >= 0.5 and rr["s1"] >= 0.7 and rr["blend"] > CHIM_SH and sh_d(xx2) > 0
    ok = std(r, x2) and r["sw"] > re["sw"]
    okp = std(rp, xp) and rp["sw"] > re["sw"]
    print(f"  CEF: {'PASS' if ok else 'fail'}   control PERMUTED: {'PASS (bad!)' if okp else 'fails (good)'}")
    print("  pass = early >= 0.5, recent >= 0.7, blend > 1.42, 2x cost > 0, Sharpe > EW_ALL")
    return ok, okp


def run_cef():
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, "cef.parquet")
    if os.path.exists(f):
        raw = pd.read_parquet(f)
    else:
        navs = ["X" + t + "X" for t in CEFS]
        p = yf_close(CEFS)
        a = yf_close(CEFS, adj=True)
        n = yf_close(navs)
        raw = pd.concat({"price": p, "tr": a, "nav": n.rename(columns=lambda c: c[1:-1])}, axis=1)
        raw.to_parquet(f)
    price, tr, nav = raw["price"], raw["tr"], raw["nav"]
    ok = [t for t in CEFS if t in price and t in nav
          and (price[t].notna() & nav[t].notna()).sum() >= 1250]
    print(f"  CEF: {len(ok)} of {len(CEFS)} funds have 5+ years of price and NAV on Yahoo")
    nav = nav[ok].ffill(limit=5)
    cef_report(price[ok], nav, tr[ok])


# =================================================================== smoke
def smoke():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2017-01-02", "2026-05-29")
    n = len(days)
    # NIGHT
    syms = [f"S{i}" for i in range(40)]
    for planted in (False, True):
        mk = rng.normal(0, 0.006, (n, 1))                 # common market moves, split night/day
        on = rng.normal(0.0015 if planted else 0.0002, 0.007, (n, 40)) + mk * 0.5
        idr = rng.normal(-0.0005 if planted else 0.0002, 0.010, (n, 40)) + rng.normal(0, 0.006, (n, 1))
        C = pd.DataFrame(100 * np.exp(np.cumsum(on + idr, 0)), index=days, columns=syms)
        O = C.shift(1) * (1 + on)
        me = [d for d in pd.Series(days, index=days).resample("ME").last().values]
        wins_ = [(pd.Timestamp(me[k]), pd.Timestamp(me[k + 1]), list(rng.choice(syms, 20, replace=False)))
                 for k in range(1, len(me) - 1)]
        F, N, _ = night_engine(O, C, wins_, 0.0002)
        _, N2, _ = night_engine(O, C, wins_, 0.0004)
        _, NS, _ = night_engine(O, C, wins_, 0.0002, shuffle=True)
        v, vs = night_verdict(F, N, N2), night_verdict(F, NS, NS)
        print(f"  NIGHT {'planted' if planted else 'noise  '}: FULL shp {sh_d(F):5.2f} NIGHT {sh_d(N):5.2f} "
              f"-> {'PASS' if v else 'fail'} | shuffled control {'PASS' if vs else 'fail'}"
              f"   (want {'PASS / fail' if planted else 'fail / fail'})")
    # LOWVOL
    for planted in (False, True):
        vols = np.linspace(0.008, 0.04, 400)
        drift = (0.0008 - vols * 0.02) if planted else np.zeros(400)
        mkt = rng.normal(0.0004, 0.01, n)
        r = mkt[:, None] * (vols / 0.02) + rng.normal(0, 1, (n, 400)) * vols * 0.6 + drift
        px = pd.DataFrame(50 * np.exp(np.cumsum(r, 0)), index=days, columns=[f"L{i}" for i in range(400)])
        L = lowvol_engine(px, lambda t: set(px.columns))
        R = lowvol_engine(px, lambda t: set(px.columns), random=True)
        spy = pd.Series(100 * np.exp(np.cumsum(mkt)), index=days)
        SPYm = spy.loc[L.index].pct_change().fillna(0)
        s = [m_stats(m, win)[1] for m in (L, SPYm) for win in (S_EARLY, S_RECENT)]
        beats = s[0] > s[2] and s[1] > s[3]
        sr = [m_stats(R, win)[1] for win in (S_EARLY, S_RECENT)]
        print(f"  LOWVOL {'planted' if planted else 'noise  '}: LOWVOL shp {s[0]:.2f}/{s[1]:.2f} vs market "
              f"{s[2]:.2f}/{s[3]:.2f} -> {'beats' if beats else 'no'} | random {sr[0]:.2f}/{sr[1]:.2f}")
    # VIX
    d2 = pd.bdate_range("2011-01-03", "2026-10-30")
    for planted in (False, True):
        vix = pd.Series(18 + 6 * np.sin(np.arange(len(d2)) / 60) + rng.normal(0, 1.5, len(d2)), index=d2)
        v3 = pd.Series(19.0, index=d2) + rng.normal(0, 0.5, len(d2))
        up = (v3 / vix > 1.05).shift(1).fillna(False).values
        vr = pd.Series(rng.normal(0, 0.04, len(d2)) - (0.004 * up if planted else 0), index=d2)
        x, _ = vix_engine(vix, v3, vr)
        al, _ = vix_engine(vix, v3, vr, always=True)
        print(f"  VIX {'planted' if planted else 'noise  '}: switch shp {sh_d(x):5.2f}  always {sh_d(al):5.2f}"
              f"   (want {'switch high, above always' if planted else 'both near 0'})")
    # CEF
    d3 = pd.bdate_range("2009-01-02", "2026-10-30")
    m_ = len(d3)
    for planted in (False, True):
        navr = rng.normal(0.0002, 0.008, (m_, 1)) + rng.normal(0, 0.004, (m_, 60))   # shared market
        nav = pd.DataFrame(20 * np.exp(np.cumsum(navr, 0)), index=d3, columns=[f"C{i}" for i in range(60)])
        if planted:
            u = np.zeros((m_, 60))
            e = rng.normal(0, 0.006, (m_, 60))
            for i in range(1, m_):
                u[i] = u[i - 1] * 0.98 + e[i]
            disc = -0.07 + u
        else:
            disc = -0.07 + np.cumsum(rng.normal(0, 0.003, (m_, 60)), 0)
        price = nav * (1 + np.clip(disc, -0.5, 0.5))
        x, _, _, _ = cef_engine(price, nav, price)
        xp, _, _, _ = cef_engine(price, nav, price, permute=True)
        print(f"  CEF {'planted' if planted else 'noise  '}: CEF shp {sh_d(x):5.2f}  permuted control {sh_d(xp):5.2f}"
              f"   (want {'CEF high, control lower' if planted else 'both low'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=["night", "lowvol", "vix", "cef"])
    a = ap.parse_args()
    if a.smoke:
        return smoke()
    for k, f in [("night", run_night), ("lowvol", run_lowvol), ("vix", run_vix), ("cef", run_cef)]:
        if k in a.only:
            try:
                f()
            except SystemExit as e:
                print(e)
            except Exception as e:  # noqa: BLE001
                print(f"\n  {k.upper()} could not run: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
