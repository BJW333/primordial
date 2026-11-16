"""
buyback_lab.py -- creative BUY-BACK switches for a momentum stop, each built
to fix a reason the old buy-backs (moving averages, 10-day highs) failed.

THE STOP (Blake's number, any time of year): the account falls 10% from its
high since Jan 1 (or since the last re-entry) -> momentum goes to T-bills.
The 30% yearly lock still works (-> market until Jan 1). Jan 1: back to full.

RE-ENTRY SWITCHES (all settings fixed before running)
  NEVER     stay in T-bills until Jan 1                        (reference)
  HIGH10    momentum book at a new 10-day high        (old style, reference)
  THRUST    BREADTH THRUST (Zweig 1986): the 10-day average share of the 49
            Ken French industries that rose that day jumps from below 40% to
            above 61.5% within 10 days -- the classic "everything is being
            bought at once" signal that marks the END of sell-offs. Fixes
            re-entering too late after V-shaped bottoms.
  PANICFADE the market's 1-month volatility hit its top 5% (5-year rank)
            during the stop and has now fallen back below its top 20% --
            panic peaked and is fading. Re-enter on the fade, not on price.
  LADDER    no prediction: buy back 1/3 of momentum 10 trading days after
            the stop, 2/3 after 20, all after 30. Spreads the bet so a
            wrong guess never costs the whole rebound.
  THRUST>BRIDGE / PANICFADE>BRIDGE
            same signals, but re-enter through the MARKET (70% market +
            30% T-bills) until the next month-end rebalance, then momentum.
            Rebounds after crashes are led by last year's LOSERS (Daniel &
            Moskowitz), so buying back the old winners catches the wrong
            part of the rally; the market catches it, and the next monthly
            ranking picks up the new leaders.
  DDCTRL    no stop at all: momentum share = 70% x (1 - drawdown / 30%),
            drawdown = account below its high since Jan 1 (Grossman & Zhou
            drawdown control). Shrinks smoothly as losses grow and buys back
            automatically as the account recovers. Rebalanced when the
            target moves 5+ points.
  0.2% cost on whatever share of the account is traded.

MODEL: momentum 70% + T-bills 30%, 30% yearly lock -> market.
  1927-2016 Ken French top-10% momentum winners / all-market / RF.
  2017-03 -> 2026-05 CHIMERA-70 STEADY book (data/momentum_cap_daily.csv),
  SPY, RF. Breadth from Ken French 49 industries (daily, 1926+).

PRE-STATED PASS RULE
  1 pick the switch with the best Sharpe on 1927-1989 (switches with a stop
    must have stopped >= 3 times there)
  2 UNTOUCHED: 1990-2016 AND CHIMERA-70 must both show Sharpe up AND worst
    drop no worse than BASE (no stop). Pass -> port to QC as a switch.
  Every switch's results are printed for all periods as context only.

Run (repo root; downloads the 49-industry file once, ~3 MB):
    python3.10 scripts/buyback_lab.py --smoke
    python3.10 scripts/buyback_lab.py | tee logs/buyback_lab.txt
  If the download fails, get 49_Industry_Portfolios_daily_CSV.zip from
  Ken French's data library in Safari and move it into data/french/.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crash_warning_search as CW  # noqa: E402

CW.FILES["ind49"] = "49_Industry_Portfolios_daily_CSV.zip"
COST, MOM, LOCK, DROP = 0.002, 0.70, 0.30, 0.10
P1, P2, P3 = ("1927-01-01", "1989-12-31"), ("1990-01-01", "2016-12-31"), \
    ("2017-03-01", "2026-05-29")
SWITCHES = ["NEVER", "HIGH10", "THRUST", "PANICFADE", "LADDER",
            "THRUST>BRIDGE", "PANICFADE>BRIDGE", "DDCTRL"]


# ------------------------------------------------------------ signals
def breadth_thrust(ind):
    up = (ind > 0).sum(axis=1) / ind.notna().sum(axis=1).replace(0, np.nan)
    ema = up.ewm(span=10, adjust=False).mean()
    return ((ema > 0.615) & (ema.rolling(11).min() < 0.40)).fillna(False)


def panic_fade(d):
    rk = CW.features(d)["MKTVOL"]
    hit = (rk >= 0.95).astype(float).rolling(21).max() > 0
    return (hit & (rk < 0.80)).fillna(False)


def period_data(b, m, c, thrust, fade):
    X = (1 + b).cumprod()
    return {"b": b.values, "m": m.values, "c": c.values, "idx": b.index,
            "years": b.index.year.values,
            "month_end": np.r_[b.index.month[1:] != b.index.month[:-1], True],
            "HIGH10": (X >= X.rolling(10).max()).values,
            "THRUST": thrust.values, "PANICFADE": fade.values}


# ------------------------------------------------------------ simulation
def run(D, sw=None):
    """sw None = BASE (no stop). Returns daily returns and number of stops."""
    b, m, c, years = D["b"], D["m"], D["c"], D["years"]
    n = len(b)
    A, ys, yr, peak = 1.0, 1.0, years[0], 1.0
    mode = "MOM"            # MOM, OUT, BRIDGE, LOCK
    share = MOM             # momentum share while MOM (LADDER / DDCTRL vary it)
    out_days, stops = 0, 0
    out = np.empty(n)
    sig = sw.split(">")[0] if sw else None
    bridge = bool(sw) and sw.endswith(">BRIDGE")
    for i in range(n):
        if years[i] != yr:                                     # new year
            yr, ys, peak = years[i], A, A
            if mode != "MOM" or share != MOM:
                A *= 1 - COST * (1.0 if mode == "LOCK" else MOM)
            mode, share = "MOM", MOM
        if mode == "MOM":
            r = share * b[i] + (1 - share) * c[i]
        elif mode == "BRIDGE":
            r = MOM * m[i] + (1 - MOM) * c[i]
        elif mode == "LOCK":
            r = m[i]
        else:
            r = c[i]
        A *= 1 + r
        out[i] = A
        # ---------------- decide tomorrow at today's close
        if mode == "LOCK":
            continue
        if A / ys - 1 >= LOCK:
            mode = "LOCK"
            A *= 1 - COST
            continue
        peak = max(peak, A)
        if sw == "DDCTRL":
            dd = 1 - A / peak
            tgt = MOM * max(0.0, 1 - dd / 0.30)
            if abs(tgt - share) >= 0.05:
                A *= 1 - COST * abs(tgt - share)
                share = tgt
            continue
        if sw is None:
            continue
        if mode in ("MOM", "BRIDGE") and A <= peak * (1 - DROP):
            A *= 1 - COST * (share if mode == "MOM" else MOM)
            mode, share, out_days = "OUT", MOM, 0
            stops += 1
            continue
        if mode == "BRIDGE" and D["month_end"][i]:
            mode = "MOM"                                       # market -> momentum
            A *= 1 - COST * MOM
            continue
        if mode == "OUT":
            out_days += 1
            if sig == "NEVER":
                continue
            if sig == "LADDER":
                if out_days in (10, 20, 30):
                    tgt = MOM * out_days / 30
                    A *= 1 - COST * MOM / 3
                    mode, share, peak = "MOM", tgt, A
                continue
            if D[sig][i]:
                A *= 1 - COST * MOM
                mode, peak = ("BRIDGE" if bridge else "MOM"), A
                share = MOM
        elif mode == "MOM" and sig == "LADDER" and share < MOM - 1e-9:
            out_days += 1                                      # keep laddering in
            if out_days in (20, 30):
                A *= 1 - COST * MOM / 3
                share = MOM * out_days / 30
    eq = pd.Series(out, D["idx"])
    return eq.pct_change().fillna(eq.iloc[0] - 1), stops


def stats(r):
    r = np.asarray(r)
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(252)
    eq = np.cumprod(1 + r)
    return ann, sh, (eq / np.maximum.accumulate(eq) - 1).min()


def show(label, r, extra=""):
    a, s, d = stats(r)
    print(f"    {label:<18} {a*100:>+6.1f}%/yr  Sharpe {s:>5.2f}  worst {d*100:>6.1f}%  {extra}")
    return s, d


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        d = CW.smoke_data()
        rng = np.random.default_rng(3)
        ind = pd.DataFrame(d["M"].values[:, None] + rng.normal(0, 0.012, (len(d), 49)),
                           index=d.index)
    else:
        d = CW.load_french()
        ind = CW.first_block(CW.fetch("ind49"))
    thrust = breadth_thrust(ind).reindex(d.index).fillna(False)
    fade = panic_fade(d)

    def french(p):
        s = (d.index >= p[0]) & (d.index <= p[1])
        x = d[s]
        return period_data(x["W"], x["M"], x["RF"], thrust[s], fade[s])

    if a.smoke:
        s3 = (d.index >= P3[0]) & (d.index <= P3[1])
        bk = pd.DataFrame({"STEADY": d["W"][s3] * 1.5, "SPY": d["M"][s3]})
    else:
        bk = pd.read_csv(CW.BOOK, index_col=0, parse_dates=True)
        bk = bk[(bk.index >= P3[0]) & (bk.index <= P3[1])]
    cash = d["RF"].reindex(bk.index).ffill().fillna(0)
    D3 = period_data(bk["STEADY"], bk["SPY"].fillna(0), cash,
                     thrust.reindex(bk.index).ffill().fillna(False),
                     fade.reindex(bk.index).ffill().fillna(False))
    periods = {"1927-1989": french(P1), "1990-2016": french(P2), "CHIMERA-70": D3}
    th_days = thrust[thrust & ~thrust.shift(1, fill_value=False)]
    print(f"  French {d.index[0].date()} -> {d.index[-1].date()}"
          f"{'  [SYNTHETIC]' if a.smoke else ''} | breadth thrusts: {len(th_days)} "
          f"(e.g. {', '.join(f'{x:%Y-%m-%d}' for x in th_days.index[-6:])})")
    print(f"  stop: account -{DROP:.0%} from its high | {MOM:.0%} momentum | "
          f"{LOCK:.0%} lock -> market")

    res = {}
    for pn, D in periods.items():
        print(f"\n  ===== {pn} =====")
        rb, _ = run(D)
        res[(pn, "BASE")] = (rb, 0)
        show("BASE (no stop)", rb)
        for sw in SWITCHES:
            r, st = run(D, sw)
            res[(pn, sw)] = (r, st)
            show(sw, r, "" if sw == "DDCTRL" else f"stops {st}")

    sb1 = stats(res[("1927-1989", "BASE")][0])[1]
    elig = [sw for sw in SWITCHES if sw == "DDCTRL" or res[("1927-1989", sw)][1] >= 3]
    best = max(elig, key=lambda sw: stats(res[("1927-1989", sw)][0])[1]) if elig else None
    print("\n  ===== PICK (1927-1989) + UNTOUCHED CHECK =====")
    ok = best is not None
    if best:
        print(f"    best switch 1927-1989: {best} (Sharpe "
              f"{stats(res[('1927-1989', best)][0])[1]:.2f} vs BASE {sb1:.2f})")
        for pn in ("1990-2016", "CHIMERA-70"):
            _, s0, d0 = stats(res[(pn, "BASE")][0])
            _, s1, d1 = stats(res[(pn, best)][0])
            good = s1 > s0 and d1 >= d0 - 1e-9
            ok &= good
            print(f"    {pn}: Sharpe {s1:.2f} vs {s0:.2f}, worst {d1*100:.1f}% vs "
                  f"{d0*100:.1f}% -> {'pass' if good else 'FAIL'}")
        rb, rs = res[("CHIMERA-70", "BASE")][0], res[("CHIMERA-70", best)][0]
        print(f"    CHIMERA-70 by year (BASE vs {best}):")
        for y in sorted(set(rb.index.year)):
            print(f"      {y}: {((1 + rb[rb.index.year == y]).prod() - 1)*100:+5.0f}%   "
                  f"{((1 + rs[rs.index.year == y]).prod() - 1)*100:+5.0f}%")
    print("\n  ===== VERDICT (pre-stated) =====")
    print("    " + (f"PASS -- port {best} to QuantConnect as a switch" if ok
                    else "FAIL -- keep CHIMERA-70 as is"))
    if a.smoke:
        print("  smoke: plumbing check on synthetic data (results meaningless)")


if __name__ == "__main__":
    main()
