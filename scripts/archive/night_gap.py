"""
night_gap.py -- "trade less often": skip the DAY only when it is most likely to lose.
PRE-REGISTERED 2026-11-15, before looking at any gap-conditional numbers. No tuning after results.

Background (structural_screen.py): CHIMERA's momentum picks earn everything overnight and lose
~6%/yr during the day, but stepping out EVERY day means ~10,000 trades a year. On an average day
the day leg loses only ~0.02%, less than the ~0.04% it costs to step out and back in.
Idea (Berkman et al. 2012, Lou-Polk-Skouras): the daytime loss should be concentrated in stocks
that just GAPPED UP at the open (overnight buyers overpay; the price sags during the day).

Rule GAP1 (judged): hold the monthly momentum picks as usual. Each morning, for each stock, sell at
  the opening auction ONLY IF it opens more than 1.0% above yesterday's close (a limit-on-open sell
  order placed before the open does exactly this), and buy it back at that day's close.
  2 bp per side on those trades; the usual 20 bp per trade on the monthly rebalance.
Info rows: thresholds 0.5% and 2.0%; NIGHT (step out every day); FULL (never step out).
Control GAP1_RANDOM (must NOT pass): step out on a random set of stock-days of the same size.
PASS = in BOTH windows (2017-02 -> 2021-10, 2021-11 -> 2026-05) GAP1 beats FULL on annual return AND
  on Sharpe, and its whole-period Sharpe still beats FULL's at 2x step-out costs.

Run (repo root; uses the cached opens from structural_screen.py):
    python3.10 scripts/night_gap.py | tee logs/night_gap.txt
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import structural_screen as ss  # noqa: E402


def windows_and_data():
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
    idx, wins = px.index, []
    for t2 in sorted(log.t2.unique()):
        t2 = pd.Timestamp(t2)
        prior = idx[idx < t2.to_period("M").start_time]
        if len(prior):
            wins.append((prior[-1], t2, list(log[log.t2 == t2].s)))
    syms = sorted({s for _, _, p in wins for s in p})
    O, C = ss.fetch_open_close(syms, "2016-12-01", "2026-06-02")
    return O, C, wins


def engine(O, C, wins, thr, c_side=0.0002, mode="gap", seed=0, frac=None):
    """mode: gap (step out when overnight gap > thr) | all (every day) | none | random (frac of stock-days)."""
    on, idr = O / C.shift(1) - 1, C / O - 1
    rng = np.random.default_rng(seed)
    out, prev, n_out, n_all = {}, set(), 0, 0
    for t, t2, picks in wins:
        picks = [s for s in picks if s in C.columns]
        days = C.index[(C.index > t) & (C.index <= t2)]
        if not len(days) or not picks:
            continue
        a, b = on.loc[days, picks], idr.loc[days, picks]
        ok = a.notna() & b.notna() & (a.abs() < 0.5) & (b.abs() < 0.5)
        a, b = a.where(ok), b.where(ok)
        if mode == "gap":
            skip = a > thr
        elif mode == "all":
            skip = ok
        elif mode == "random":
            skip = pd.DataFrame(rng.random(a.shape) < frac, index=a.index, columns=a.columns) & ok
        else:
            skip = ok & False
        r = ((1 + a) * (1 + b) - 1).where(~skip, a - 2 * c_side)
        f = r.mean(axis=1)
        turn = (len(set(picks) - prev) + len(prev - set(picks))) / max(len(picks), 1)
        f.iloc[0] -= turn * 0.0020
        prev = set(picks)
        n_out += int(skip.sum().sum())
        n_all += int(ok.sum().sum())
        out.update(f.to_dict())
    return pd.Series(out).sort_index(), n_out / max(n_all, 1)


def main():
    O, C, wins = windows_and_data()
    W = (ss.S_EARLY, ss.S_RECENT)
    F, _ = engine(O, C, wins, 0, mode="none")
    G, frac = engine(O, C, wins, 0.01)
    G2, _ = engine(O, C, wins, 0.01, c_side=0.0004)
    R, _ = engine(O, C, wins, 0, mode="random", frac=frac)
    R2, _ = engine(O, C, wins, 0, mode="random", frac=frac, c_side=0.0004)

    def passed(x, x2):
        ok = all(ss.ann_d(ss.w(x, w_)) > ss.ann_d(ss.w(F, w_)) and ss.sh_d(ss.w(x, w_)) > ss.sh_d(ss.w(F, w_))
                 for w_ in W)
        return bool(ok and ss.sh_d(x2) > ss.sh_d(F))

    yrs = len(F) / 252
    print(f"\n  ===== NIGHT_GAP: step out of the day only after a gap up ({len(wins)} months) =====")
    print(ss.HDR + "stock-days stepped out")
    ss.line("FULL (never)", F, W, "0%")
    ss.line("GAP1 (>1.0%)", G, W, f"{frac*100:.0f}%  " + ("PASS" if passed(G, G2) else "fail"))
    ss.line("GAP1 at 2x cost", G2, W)
    for thr in (0.005, 0.02):
        x, fr = engine(O, C, wins, thr)
        ss.line(f"gap >{thr*100:.1f}% (info)", x, W, f"{fr*100:.0f}%")
    x, fr = engine(O, C, wins, 0, mode="all")
    ss.line("NIGHT (every day)", x, W, "100%")
    ss.line("ctl GAP1_RANDOM", R, W, f"{frac*100:.0f}%  " + ("PASS (bad!)" if passed(R, R2) else "fails (good)"))
    on, idr = O / C.shift(1) - 1, C / O - 1
    print(f"\n  round trips per year at GAP1: about {frac * 20 * 252:.0f} (vs {20*252} stepping out every day)")
    print("  pass = beats FULL on return AND Sharpe in both windows, and on Sharpe at 2x step-out costs")


if __name__ == "__main__":
    main()
