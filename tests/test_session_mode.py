"""Session entry style: pinned guarantees.

1. NO FREE PROFIT: on an honestly-generated random walk (open built from
   the PRIOR close, close built from the open), same-session trades must
   earn ~zero minus costs in both gap directions. A generator that builds
   open FROM today's close manufactures reversion and once showed +12.9
   Sharpe on pure noise -- that failure lives here so it can never return.
2. LOOK-AHEAD LOCKOUT: session genomes with close-derived atoms must be
   rejected by the ENGINE itself, not just by make_genome (raw JSON bypass).
"""
import numpy as np
import pandas as pd
import pytest

from primordial import atoms as A
from primordial import genome as G
from primordial.universe import COST_TIERS


def _honest_walk(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    o = np.empty(n); c = np.empty(n); c[0] = o[0] = 100.0
    gap = rng.normal(0, 0.008, n); intr = rng.normal(0, 0.010, n)
    for t in range(1, n):
        o[t] = c[t - 1] * np.exp(gap[t])
        c[t] = o[t] * np.exp(intr[t])
    df = pd.DataFrame(
        {"open": o, "high": np.maximum(o, c) * 1.004,
         "low": np.minimum(o, c) * 0.996, "close": c,
         "volume": rng.integers(1e5, 1e6, n).astype(float)},
        index=pd.date_range("2005-01-01", periods=n, freq="D"))
    A.compute_atoms(df, list(A.BLOCKS.keys()), False)
    return df


def _session_genome(tree):
    return G.Genome(timeframe="1d", root_type="bool", entry_tree=tree,
                    regime_tree=None, direction="long",
                    entry_style="session", entry_param=1, stop_atr=99.0,
                    target_r=0.0, time_stop=0, trail_mode="none",
                    max_hold=1, anchor_ema=0)


def _term(name):
    return {"op": "term", "ch": [], "name": name}


def _const(v):
    return {"op": "const", "ch": [], "value": v}


def test_session_no_free_profit_on_random_walk():
    df = _honest_walk()
    cost = COST_TIERS["equity_liquid"]
    for tree in (
        {"op": "gt", "ch": [_term("open"), _const(0.0)]},          # always
        {"op": "lt", "ch": [_term("gap_atr_open"), _const(-1.0)]}, # gap dn
        {"op": "gt", "ch": [_term("gap_atr_open"), _const(1.0)]},  # gap up
    ):
        res = G.run_backtest(_session_genome(tree), df, cost,
                             bar_seconds=86400)
        r = np.array([t[3] for t in res.trades])   # r-multiples
        assert len(r) > 20, "session mode should trade on this walk"
        # statistical bound: on a random walk the mean return must not be
        # significantly positive. Raw-mean thresholds trip on small-n noise
        # (93 trades at 1% vol -> SE ~0.10%); a t-stat catches manufactured
        # profit (the bad generator scored t ~ +12) while tolerating noise.
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)) + 1e-12)
        assert t < 3.0, f"free profit: mean {r.mean():.5f}, t {t:.2f}"


def test_session_rejects_lookahead_atoms_at_engine_level():
    df = _honest_walk(800)
    cost = COST_TIERS["equity_liquid"]
    bad = _session_genome(
        {"op": "lt", "ch": [_term("gap_atr"), _const(-1.0)]})
    # bypass make_genome entirely (raw construction == hand-edited JSON)
    bad2 = G.Genome.from_json(bad.to_json())
    with pytest.raises(ValueError):
        G.run_backtest(bad2, df, cost, bar_seconds=86400)


def test_open_safety_is_lag_aware():
    """Yesterday's close IS knowable at 09:30; today's is not. Only an
    explicit lag(n>=1) confers safety -- a rolling window does not, because
    roll_mean(close, n=200) still contains today's bar."""
    from primordial.genome import _open_safe_violations as V

    def term(nm):
        return {"op": "term", "ch": [], "name": nm}

    def lag(ch, n):
        return {"op": "lag", "ch": [ch], "n": n}

    def roll(ch, n):
        return {"op": "roll_mean", "ch": [ch], "n": n}

    assert V(term("close")) == {"close"}                    # bare: unsafe
    assert V(lag(term("close"), 1)) == set()                # lagged: safe
    assert V(roll(lag(term("close"), 1), 200)) == set()     # sma of lagged
    assert V(roll(term("close"), 200)) == {"close"}         # sma of today
    assert V(lag(term("close"), 0)) == {"close"}            # lag 0 is today
    assert V(term("gap_atr_open")) == set()                 # whitelisted
    assert V(term("gap_atr")) == {"gap_atr"}                # ATR has today
