#!/usr/bin/env python3
"""apply_judge_reporting_fix.py -- two reporting fixes surfaced by the
2026-09-09 mh_vote runs. NO strategy logic changes; verdicts on past runs
stand (pooled PSR, which drove them, is untouched).

  FIX 1 (judge.py): trade_sharpe printed -923.53 when a cell was all
  near-identical stop-outs -- gap-through fills differ by ~1e-3 R, so
  std ~ 0.003 and mean/std explodes. Guarded only for std == 0 exactly.
  Fix: clip the annualized cell Sharpe to [-50, +50]. Sign and ordering
  preserved; medians stay readable; the exact-zero legacy return of 0.0
  is unchanged.

  FIX 2 (pipeline.py + validate_genome.py): when no (name x fold) cell
  reaches 4 trades, the median was silently reported as +0.00 --
  indistinguishable from measured breakeven. Fix: carry the eligible
  cell count into the report row (holdout_cells) and print
  'n/a (0 cells)' / 'NO ELIGIBLE CELLS' instead of a fake zero.

  Plus: tests/test_judge_reporting.py (5 guards) and a CHANGELOG entry.

    python3.10 patches/apply_judge_reporting_fix.py    # from repo root
    python3.10 -m pytest tests/ -q                     # all green, then commit

Idempotent; anchored; exits loudly on missing anchors.
"""
import os
import sys

if not os.path.exists("primordial/judge.py"):
    sys.exit("run from the repo root (primordial/judge.py not found)")


def sub_once(path, old, new, label):
    s = open(path).read()
    if new in s:
        print(f"  = {path}: {label} (already)")
        return
    n = s.count(old)
    if n != 1:
        sys.exit(f"REFUSING {path}: anchor for '{label}' found {n} times "
                 "(need exactly 1). Diff it; nothing further applied.")
    open(path, "w").write(s.replace(old, new, 1))
    print(f"  ~ {path}: {label}")


def write_new(path, content, label):
    if os.path.exists(path):
        if open(path).read() == content:
            print(f"  = {path}: {label} (already)")
            return
        sys.exit(f"REFUSING {path}: exists with different content. Diff it; "
                 "nothing further applied.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write(content)
    print(f"  + {path}: {label}")


# ---------------------------------------------------------------- FIX 1
OLD_TS = '''def trade_sharpe(r_mults, trades_per_year):
    r = np.asarray(r_mults, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(trades_per_year))'''

NEW_TS = '''def trade_sharpe(r_mults, trades_per_year):
    """Annualized per-trade Sharpe, CLIPPED to [-50, +50]. The clip exists
    because a cell of near-identical stop-outs (gap-through fills differ by
    ~1e-3 R) has std ~ 0.003, and mean/std then prints -923 -- a degeneracy
    artifact, not a measurement (mhv_gate2, 2026-09-09). |SR| 50 at these
    trade frequencies is already beyond any real edge, and the cheat
    detector's leaked strategy still trips its >5 assert at the cap. Sign
    and ordering preserved. std == 0 exactly keeps the legacy 0.0 (no
    dispersion, no evidence)."""
    r = np.asarray(r_mults, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    sr = float(r.mean() / r.std(ddof=1) * math.sqrt(trades_per_year))
    return float(np.clip(sr, -50.0, 50.0))'''

# ---------------------------------------------------------------- FIX 2
OLD_EVAL = '''        med = float(np.median(cells)) if cells else 0.0
        evals.append((fit_train, g, tf, hs, data_h, med, pooled))'''
NEW_EVAL = '''        med = float(np.median(cells)) if cells else 0.0
        evals.append((fit_train, g, tf, hs, data_h, med, pooled, len(cells)))'''

OLD_UNPACK = '    for fit_train, g, tf, hs, data_h, med, pooled in evals:'
NEW_UNPACK = ('    for fit_train, g, tf, hs, data_h, med, pooled, n_cells '
              'in evals:')

OLD_ROW = '''        row.update(holdout_median_sharpe=med, holdout_trades=len(pooled),
                   holdout_psr=psr(pooled), luck_bar=bar)'''
NEW_ROW = '''        row.update(holdout_median_sharpe=med, holdout_trades=len(pooled),
                   holdout_psr=psr(pooled), luck_bar=bar,
                   holdout_cells=n_cells)'''

OLD_TAG = '''            tag = "SURVIVED" if ok else "rejected"
            print(f"  [{tag}] {tf} holdout SR {med:+.2f} vs bar {bar:+.2f} | "
                  f"PSR {row['holdout_psr']:.2f} | {g.describe()[:60]}")'''
NEW_TAG = '''            tag = "SURVIVED" if ok else "rejected"
            sr_txt = f"{med:+.2f}" if n_cells else "n/a (0 cells)"
            print(f"  [{tag}] {tf} holdout SR {sr_txt} vs bar {bar:+.2f} | "
                  f"PSR {row['holdout_psr']:.2f} | {g.describe()[:60]}")'''

OLD_GATE = '''    print(f"  holdout SR {r.get('holdout_median_sharpe',0):+.2f} vs bar "
          f"{r.get('luck_bar',0):+.2f} | PSR {r.get('holdout_psr',0):.3f} | "
          f"trades {r.get('holdout_trades')}")'''
NEW_GATE = '''    if r.get("holdout_cells") == 0:
        print(f"  holdout SR: NO ELIGIBLE CELLS -- every name x fold cell "
              f"had < 4 trades (old runs printed +0.00 here) | pooled PSR "
              f"{r.get('holdout_psr',0):.3f} | trades "
              f"{r.get('holdout_trades')}")
    else:
        print(f"  holdout SR {r.get('holdout_median_sharpe',0):+.2f} vs bar "
              f"{r.get('luck_bar',0):+.2f} | PSR {r.get('holdout_psr',0):.3f}"
              f" | trades {r.get('holdout_trades')} | cells "
              f"{r.get('holdout_cells', '?')}")'''

# ---------------------------------------------------------------- tests
TESTS = '''"""Guards for the 2026-09 judge reporting fixes.

FIX 1: trade_sharpe clip -- a cell of near-identical stop-outs must read
as a capped bad number, never -923. FIX 2 is covered structurally: the
row key holdout_cells distinguishes 'no eligible cells' from breakeven.
"""
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from primordial.judge import trade_sharpe


def test_near_identical_stopouts_clip_not_explode():
    # the exact mhv_gate2 pathology: ~-1R stop-outs, std ~ 0.003
    rng = np.random.default_rng(0)
    rs = list(-1.0 + rng.normal(0, 0.003, 14))
    sr = trade_sharpe(rs, 14 / 1.35)
    assert sr == -50.0, f"expected clip at -50, got {sr}"


def test_exact_zero_std_keeps_legacy_zero():
    assert trade_sharpe([-1.0] * 10, 20) == 0.0
    assert trade_sharpe([0.7] * 10, 20) == 0.0


def test_positive_degenerate_clips_high():
    rng = np.random.default_rng(1)
    rs = list(2.0 + rng.normal(0, 0.001, 20))
    assert trade_sharpe(rs, 50) == 50.0


def test_normal_values_untouched():
    rng = np.random.default_rng(2)
    rs = list(rng.normal(0.1, 1.0, 200))
    r = np.asarray(rs)
    manual = float(r.mean() / r.std(ddof=1) * math.sqrt(30))
    assert abs(manual) < 50
    assert abs(trade_sharpe(rs, 30) - manual) < 1e-12


def test_ordering_preserved_below_cap():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1.0, 100)
    worse = list(-0.5 + noise)
    better = list(0.2 + noise)
    assert trade_sharpe(worse, 20) < trade_sharpe(better, 20)
'''

CHANGELOG_OLD = "# v0.1.7 \u2014 deep-state integration (Aug 2026)"
CHANGELOG_NEW = """# v0.1.8 \u2014 judge reporting fixes (Sep 2026)

Two display/measurement artifacts surfaced by the mh_vote family runs
(2026-09-09). No strategy logic changed; past verdicts stand (pooled PSR,
which drove them, is untouched).

## judge.py
- `trade_sharpe` clipped to [-50, +50]. A cell of near-identical stop-outs
  (std ~ 0.003 from gap-through fill variation) printed -923.53; the guard
  covered only std == 0 exactly. Sign and ordering preserved.

## pipeline.py / scripts/validate_genome.py
- Report rows carry `holdout_cells` (count of name x fold cells with >= 4
  trades). Zero eligible cells now prints `n/a (0 cells)` / `NO ELIGIBLE
  CELLS` instead of a `+0.00` indistinguishable from measured breakeven.
  Old runs' `+0.00` lines meant this.

## tests/test_judge_reporting.py
- 5 guards: stop-out degeneracy clips at -50, exact-zero std keeps legacy
  0.0, positive degeneracy clips at +50, normal values bit-identical,
  ordering preserved below the cap.

# v0.1.7 \u2014 deep-state integration (Aug 2026)"""


def main():
    sub_once("primordial/judge.py", OLD_TS, NEW_TS,
             "trade_sharpe clip [-50, +50]")
    sub_once("primordial/pipeline.py", OLD_EVAL, NEW_EVAL,
             "evals carry cell count")
    sub_once("primordial/pipeline.py", OLD_UNPACK, NEW_UNPACK,
             "unpack n_cells")
    sub_once("primordial/pipeline.py", OLD_ROW, NEW_ROW,
             "row holdout_cells")
    sub_once("primordial/pipeline.py", OLD_TAG, NEW_TAG,
             "rejected-line n/a on 0 cells")
    sub_once("scripts/validate_genome.py", OLD_GATE, NEW_GATE,
             "gate detail: NO ELIGIBLE CELLS")
    write_new("tests/test_judge_reporting.py", TESTS, "5 reporting guards")
    sub_once("CHANGELOG.md", CHANGELOG_OLD, CHANGELOG_NEW,
             "v0.1.8 entry")
    print("\nNext:")
    print("  python3.10 -m pytest tests/ -q     # expect all green")
    print('  git add -A && git commit -m "judge reporting fixes: clip '
          'degenerate cell Sharpe, surface zero-cell holdouts (v0.1.8)"')


if __name__ == "__main__":
    main()
