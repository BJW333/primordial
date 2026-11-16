"""
judge.py -- the shared, pre-stated pass test for exposure rules on CHIMERA
(used by dd2_lab.py and regime_sizer.py). Written BEFORE any of them ran.

WHAT A RULE MUST PROVE: that it beats simply holding a smaller FIXED amount
of momentum. Holding less always shrinks drops, so a rule only counts if it
gives the SAME return with better risk than a fixed size.

  Sharpe here = return ABOVE T-bills / volatility. (Raw return / volatility
  rises just by holding more T-bills -- size 30% shows 0.92 vs 0.45 at 100%
  on the same stocks -- which would reward any rule that holds less.)

  "Same-return fixed size": the fixed momentum share (1% steps, same 30%
  lock, same costs) whose annual return is closest to the rule's.

PASS -- ALL of:
  1 PICK      done by each script on US 1927-1989 only.
  2 HOLDOUT   US 1990-2016 (27 years the pick never saw; US only):
              a) higher Sharpe AND worst drop no worse than the same-return
                 fixed size, and
              b) the Sharpe gain is real, not luck: block bootstrap (3-month
                 blocks, 2,000 resamples, rule and fixed resampled on the same
                 days) puts it below zero less than 1.7% of the time (5% / 3,
                 because three tests -- DD2, REGIME, REGIME+DD2 -- share it).
  3 CHIMERA   2017-03 -> 2026-05, the real book (only 9 years and already
              seen, so it can't prove anything): DO NO HARM -- Sharpe no more
              than 0.05 below the same-return fixed size AND worst drop no
              more than 2 points worse.
"""
import numpy as np
import pandas as pd

LONG = ["US 1990-2016"]
CHIM = "CHIMERA-70"
P_MAX, BLOCK, N_BOOT = 0.05 / 3, 63, 2000


def stats(r, rf):
    r, rf = np.asarray(r), np.asarray(rf)
    ann = np.prod(1 + r) ** (252 / len(r)) - 1
    ex = r - rf
    sh = ex.mean() / ex.std() * np.sqrt(252)
    eq = np.cumprod(1 + r)
    return ann, sh, (eq / np.maximum.accumulate(eq) - 1).min()


def losing(r):
    y = (1 + r).groupby(r.index.year).prod() - 1
    return int((y < 0).sum()), len(y)


def matched(sizes, curve, ann):
    j = int(np.argmin(np.abs(curve[:, 0] - ann)))
    return sizes[j], j


def _sh(x):
    return x.mean(axis=-1) / x.std(axis=-1) * np.sqrt(252)


def bootstrap_p(pairs, seed=7):
    """pairs: list of (rule_excess, fixed_excess). p = share of resampled
    average Sharpe gains <= 0."""
    rng = np.random.default_rng(seed)
    gains = np.zeros(N_BOOT)
    for a, b in pairs:
        n = len(a)
        nb = int(np.ceil(n / BLOCK))
        starts = rng.integers(0, n - BLOCK, size=(N_BOOT, nb))
        idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :n]
        gains += _sh(a[idx]) - _sh(b[idx])
    gains /= len(pairs)
    return float(np.mean(gains <= 0))


def judge(title, rows, label):
    """rows: {test name: dict(rule=Series, fixed=Series, base=Series, rf=array,
    size=float)}. Prints the table, returns True/False."""
    print(f"  {'':<24}{'annual':>8}{'Sharpe':>8}{'worst':>8}  losing yrs")
    point, pairs = {}, []
    for tn, x in rows.items():
        rf = x["rf"]
        ab, sb, wb = stats(x["base"], rf)
        ar, sr, wr = stats(x["rule"], rf)
        af, sf, wf = stats(x["fixed"], rf)
        lb, ny = losing(x["base"])
        lr, _ = losing(x["rule"])
        lf, _ = losing(x["fixed"])
        if tn == CHIM:
            ok = sr >= sf - 0.05 and wr >= wf - 0.02
            tag = "do-no-harm " + ("pass" if ok else "FAIL")
        else:
            ok = sr > sf and wr >= wf - 1e-9
            tag = "pass" if ok else "FAIL"
            pairs.append((np.asarray(x["rule"]) - rf, np.asarray(x["fixed"]) - rf))
        point[tn] = ok
        idx = x["rule"].index
        print(f"  {tn} ({idx[0].date()} -> {idx[-1].date()}):")
        print(f"    {'BASE fixed 70%':<22}{ab*100:>+7.1f}%{sb:>8.2f}{wb*100:>7.1f}%  {lb}/{ny}")
        print(f"    {label:<22}{ar*100:>+7.1f}%{sr:>8.2f}{wr*100:>7.1f}%  {lr}/{ny}")
        print(f"    {'FIXED ' + format(x['size'], '.0%') + ' same return':<22}{af*100:>+7.1f}%"
              f"{sf:>8.2f}{wf*100:>7.1f}%  {lf}/{ny}   -> {tag}")
        if tn == CHIM:
            print("    by year  BASE / RULE / FIXED")
            for y in sorted(set(idx.year)):
                g = lambda r: ((1 + r[r.index.year == y]).prod() - 1) * 100  # noqa: E731
                print(f"      {y}: {g(x['base']):+5.0f}% {g(x['rule']):+5.0f}% {g(x['fixed']):+5.0f}%")
    have = [t for t in LONG if t in point]
    wins = sum(point[t] for t in have)
    p = bootstrap_p(pairs) if pairs else 1.0
    complete = len(have) == len(LONG) and CHIM in point
    ok_a = complete and wins == len(LONG)
    ok_b = complete and p < P_MAX
    ok_c = point.get(CHIM, False)
    print(f"  VERDICT {title}:")
    print(f"    a) US 1990-2016 beats same-return fixed size -> {'pass' if ok_a else 'FAIL'}")
    print(f"    b) Sharpe gain is real: bootstrap p = {p:.3f} (need < {P_MAX:.3f}) -> "
          f"{'pass' if ok_b else 'FAIL'}")
    print(f"    c) CHIMERA do-no-harm -> {'pass' if ok_c else 'FAIL'}")
    res = ok_a and ok_b and ok_c
    print(f"    => {'PASS' if res else 'FAIL'}")
    return res
