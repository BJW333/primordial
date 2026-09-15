"""Guards for the 2026-09 judge reporting fixes.

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
