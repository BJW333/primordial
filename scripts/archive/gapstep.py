"""
gapstep.py -- GAPSTEP: a stand-alone momentum strategy (its own picks, its own account; nothing to do
with the live CHIMERA algorithm) that steps out of the day after a gap up, WITH a switch that turns
the step-out off when gap-ups have stopped fading.
PRE-REGISTERED 2026-11-15, before this rule was run. No tuning after results.

HONEST NOTE written before running: night_gap.py already showed the plain rule (GAP1) fails
2017-21, and a by-year look showed 2020 is the reason (gap-ups kept running all day in that rally).
The switch below was designed AFTER seeing that. The sample holds one runaway rally, so a pass here
is weak evidence; forward paper trading is the real test.

Strategy
  Picks : momentum_v2 STEADY top 20, monthly (steady 12-month winners, liquid US stocks).
  Daily : a held stock that opens > 1.0% above yesterday's close is sold at the opening auction
          (limit-on-open order) and bought back at the close.
  Switch: the step-out is ON today only if, over the previous 63 trading days, the strategy's own
          stocks that gapped up > 1% LOST money on average between open and close (measured whether
          or not we stepped out). Fewer than 30 such stock-days in the window -> ON.
          In a rally where gap-ups keep running, that average turns positive and the rule sits out.
  Costs : 2 bp per side on step-out trades (also 2x); 20 bp per trade on the monthly rebalance.
Reference rows: FULL (same picks, never step out), GAP1 (step out with no switch).
Control (must NOT pass): GAPSTEP_RANDOM -- same switch days, but step out of random stocks, same count.
PASS = in BOTH windows (2017-02 -> 2021-10, 2021-11 -> 2026-05) GAPSTEP beats FULL on annual return AND
  on Sharpe, and its whole-period Sharpe beats FULL's at 2x step-out costs.

Run (repo root; uses the cached opens):  python3.10 scripts/gapstep.py | tee logs/gapstep.txt
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
import night_gap as ng  # noqa: E402
import structural_screen as ss  # noqa: E402

THR, LOOK, MIN_OBS = 0.01, 63, 30


def engine(O, C, wins, c_side=0.0002, switch=True, random=False, seed=0):
    on, idr = O / C.shift(1) - 1, C / O - 1
    rng = np.random.default_rng(seed)
    out, prev = {}, set()
    hist = []                      # (day index, sum of day returns after gap-ups, count) per past day
    state, n_skip, n_ok, on_days, all_days = {}, 0, 0, 0, 0
    k = 0
    for t, t2, picks in wins:
        picks = [s for s in picks if s in C.columns]
        days = C.index[(C.index > t) & (C.index <= t2)]
        if not len(days) or not picks:
            continue
        a, b = on.loc[days, picks], idr.loc[days, picks]
        ok = a.notna() & b.notna() & (a.abs() < 0.5) & (b.abs() < 0.5)
        a, b = a.where(ok), b.where(ok)
        turn = (len(set(picks) - prev) + len(prev - set(picks))) / max(len(picks), 1)
        prev = set(picks)
        for j, d in enumerate(days):
            recent = hist[-LOOK:]
            cnt = sum(h[1] for h in recent)
            enabled = True
            if switch and cnt >= MIN_OBS:
                enabled = sum(h[0] for h in recent) / cnt < 0
            av, bv = a.loc[d].values, b.loc[d].values
            gap = av > THR                                   # NaN compares False
            skip = gap & enabled
            if random and enabled:
                valid = np.flatnonzero(~np.isnan(av))
                skip = np.zeros(len(av), bool)
                if len(valid) and gap.sum():
                    skip[rng.choice(valid, size=min(int(gap.sum()), len(valid)), replace=False)] = True
            r = np.where(skip, av - 2 * c_side, (1 + av) * (1 + bv) - 1)
            f = np.nanmean(r) if np.isfinite(r).any() else 0.0
            if j == 0:
                f -= turn * 0.0020
            out[d] = f
            state[d] = enabled
            hist.append((np.nansum(bv[gap]), int(gap.sum())))
            n_skip += int(skip.sum())
            n_ok += int((~np.isnan(av)).sum())
            on_days += int(enabled)
            all_days += 1
            k += 1
    return (pd.Series(out).sort_index(), pd.Series(state).sort_index(),
            n_skip / max(n_ok, 1), on_days / max(all_days, 1), n_skip)


def main():
    O, C, wins = ng.windows_and_data()
    W = (ss.S_EARLY, ss.S_RECENT)
    F, _ = ng.engine(O, C, wins, 0, mode="none")
    G1, st1, fr1, _, _ = engine(O, C, wins, switch=False)
    G, st, fr, on_frac, n = engine(O, C, wins)
    G2, _, _, _, _ = engine(O, C, wins, c_side=0.0004)
    R, _, _, _, _ = engine(O, C, wins, random=True)
    R2, _, _, _, _ = engine(O, C, wins, random=True, c_side=0.0004)

    def passed(x, x2):
        ok = all(ss.ann_d(ss.w(x, w_)) > ss.ann_d(ss.w(F, w_)) and ss.sh_d(ss.w(x, w_)) > ss.sh_d(ss.w(F, w_))
                 for w_ in W)
        return bool(ok and ss.sh_d(x2) > ss.sh_d(F))

    yrs = len(F) / 252
    print(f"\n  ===== GAPSTEP (stand-alone momentum + gap step-out with switch), {len(wins)} months =====")
    print(f"  switch ON {on_frac*100:.0f}% of days; stepped out of {fr*100:.0f}% of stock-days "
          f"(~{n / yrs:.0f} round trips a year)")
    print(ss.HDR)
    ss.line("FULL (never)", F, W)
    ss.line("GAP1 (no switch)", G1, W)
    ss.line("GAPSTEP", G, W, "PASS" if passed(G, G2) else "fail")
    ss.line("GAPSTEP 2x cost", G2, W)
    ss.line("ctl GAPSTEP_RANDOM", R, W, "PASS (bad!)" if passed(R, R2) else "fails (good)")
    print("\n  by year (return %):   FULL    GAP1  GAPSTEP   switch ON % of days")
    for y in sorted(set(F.index.year)):
        m = F.index.year == y
        f_ = ((1 + F[m]).prod() - 1) * 100
        g1 = ((1 + G1[G1.index.year == y]).prod() - 1) * 100
        g_ = ((1 + G[G.index.year == y]).prod() - 1) * 100
        print(f"    {y}            {f_:+7.1f} {g1:+7.1f} {g_:+7.1f}   {st[st.index.year == y].mean()*100:5.0f}%")
    print("  pass = beats FULL on return AND Sharpe in both windows, and on Sharpe at 2x step-out costs")


if __name__ == "__main__":
    main()
