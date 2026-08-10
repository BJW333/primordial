"""
judge.py -- fitness, splits, per-manifest ledger, PBO, synthetic stress.

Everything that decides whether a candidate is believed lives here, in one
module, so it is hard to bypass and easy to audit.
"""
from __future__ import annotations
import json
import math
import os
import statistics
import time
import numpy as np
import pandas as pd

from .genome import run_backtest


# ---------------- metrics on a trade list ----------------------------------
def trade_sharpe(r_mults, trades_per_year):
    r = np.asarray(r_mults, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(trades_per_year))


def psr(r_mults, benchmark=0.0):
    """Probabilistic Sharpe Ratio: P(true per-trade Sharpe > benchmark) given
    n, skew, kurtosis. Replaces hard min-trade gates -- evidence scales with
    sample size."""
    r = np.asarray(r_mults, float)
    n = len(r)
    if n < 3:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = r.mean() / sd
    s = pd.Series(r)
    g3 = float(s.skew()) if n > 2 else 0.0
    g4 = float(s.kurt()) + 3.0 if n > 3 else 3.0
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if not np.isfinite(denom) or denom <= 0:
        return 0.0
    z = (sr - benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(statistics.NormalDist().cdf(z))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float((equity / equity.cummax() - 1).min())


# ---------------- splits ----------------------------------------------------
class NameTimeSplit:
    """Hold out a fraction of NAMES and the TAIL of time, with an embargo at
    the outer boundary (edgesearch applied embargo intra-fold only)."""

    def __init__(self, syms, name_frac, time_frac, embargo_bars, rng):
        syms = list(syms)
        rng.shuffle(syms)
        k = max(2, int(len(syms) * name_frac))
        self.hold_syms = syms[:k]
        self.train_syms = syms[k:]
        self.time_frac = time_frac
        self.embargo = embargo_bars

    def train(self, data):
        out = {}
        for s in self.train_syms:
            df = data[s]
            cut = int(len(df) * (1 - self.time_frac))
            out[s] = df.iloc[:cut - self.embargo].copy()
        return out

    def holdout(self, data):
        out = {}
        for s in self.hold_syms:
            df = data[s]
            cut = int(len(df) * (1 - self.time_frac))
            out[s] = df.iloc[cut:].copy()
        return out


# ---------------- k-fold robust fitness -------------------------------------
def fold_cells(genome, data, syms, cost, bar_seconds, k=3, embargo=10,
               min_trades=4):
    """Sharpe per (name x contiguous time fold) cell, plus pooled r_mults and
    mean turnover. The signal/backtest runs per fold slice; atoms were
    computed on the full frame upstream so indicators keep context."""
    cells, pooled_r, turns = [], [], []
    for s in syms:
        df = data[s]
        n = len(df)
        kk = min(k, max(1, n // 300))
        bounds = [int(n * i / kk) for i in range(kk + 1)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            a2 = a + (embargo if a > 0 else 0)
            if b - a2 < 150:
                continue
            try:
                res = run_backtest(genome, df.iloc[a2:b], cost,
                                   bar_seconds=bar_seconds)
            except Exception:
                continue
            rs = [t[3] for t in res.trades]
            pooled_r += rs
            turns.append(res.turnover)
            if len(rs) >= min_trades:
                sl = df.index[a2:b]
                span_yr = max((sl[-1] - sl[0]).total_seconds() / 31557600.0,
                              1e-6)
                cells.append(trade_sharpe(rs, len(rs) / span_yr))
    return cells, pooled_r, (float(np.mean(turns)) if turns else 0.0)


def fitness(genome, data, syms, cost, bar_seconds, lam=0.5, gamma=0.05,
            delta=0.02, min_cells=6):
    from . import tree as _T
    if not _T.has_terminal(genome.entry_tree):
        # constant entry = always/never fire; not a strategy, a switch
        return -9.99, {"n_cells": 0, "median_sharpe": 0.0,
                       "degenerate": "constant_entry"}
    # BEHAVIORAL constant check: tautologies like lte(x, x) reference data,
    # so has_terminal passes them -- but their signal never varies. Measure
    # the signal itself on one real frame; zero variance = a switch in an
    # atom costume (third disguise of the always-fire species).
    try:
        frame = data[syms[0]]
        # probe the ENTRY TREE directly, not genome.signal(): the "confirm"
        # style applies a rolling min AFTERWARDS, so a single NaN-induced 0
        # propagates forward into bars where the atoms ARE defined and a
        # tautology looks like it varies even under the finite mask below.
        probe = _T.bool_signal(genome.entry_tree, frame)
        # skip atom warmup (max rolling n is 200): NaN->0 padding makes an
        # always-true tree look like it "varies" from 0 to 1 once
        tail = probe[250:]
        # ...and look ONLY where every atom this tree reads is defined.
        # Without this, lte(x, x) escapes: comparison against NaN is False,
        # so one zero-range bar (high == low -> body_frac NaN) makes the
        # tautology flip 1 -> 0. That is how lte(body_frac, body_frac)
        # reached gen-3 best on the 5m ETF run instead of being floored.
        names = [n for n in _T.terminal_names(genome.entry_tree)
                 if n in frame.columns]
        if names:
            ok = np.ones(len(frame), dtype=bool)
            for nm in names:
                ok &= np.isfinite(frame[nm].to_numpy(dtype=float))
            tail = tail[ok[250:]]
        tail = tail[np.isfinite(tail)]
        if len(tail) > 50 and tail.std() == 0:
            return -9.99, {"n_cells": 0, "median_sharpe": 0.0,
                           "degenerate": "constant_signal"}
    except Exception:
        pass
    cells, pooled, turnover = fold_cells(genome, data, syms, cost, bar_seconds)
    # coverage-aware floor: a "winner" scored on 6 of 900 available cells is
    # a fluke, not a strategy. Require 15% of attemptable (name x fold) cells,
    # never less than the absolute floor. On the 12-name controls this equals
    # the old behaviour (floor 6); on 300-name universes it demands breadth.
    attempted = sum(min(3, max(1, len(data[s]) // 300)) for s in syms
                    if s in data)
    need = max(min_cells, int(0.15 * attempted))
    if len(cells) < need:
        return -9.99, {"n_cells": len(cells), "median_sharpe": 0.0,
                       "need_cells": need}
    med, sd = float(np.median(cells)), float(np.std(cells))
    # SCALE-RELATIVE parsimony. The old absolute per-node cost (gamma flat)
    # was wrong in both regimes: on planted controls (SR 10-20) it was
    # negligible so bloat won; on real data (SR ~0.5) 15 nodes cost more than
    # the whole signal so evolution fled to vacuous one-liners. Charging a
    # fraction of the fitness scale per node keeps the pressure identical in
    # shape at any Sharpe magnitude. Tuned on the CONTROL (planted answer
    # known), never on real data. 2026-08-08.
    scale = max(abs(med), 0.25)
    fit = (med - lam * sd
           - gamma * scale * genome.complexity()
           - delta * turnover)
    return fit, {"n_cells": len(cells), "median_sharpe": med, "std": sd,
                 "turnover": turnover, "n_trades": len(pooled)}


# ---------------- per-manifest trial ledger ---------------------------------
EFFECTIVE_EXP = 0.5      # crude correlation haircut on GP trial counts


class Ledger:
    """Persistent, keyed by manifest fingerprint. Every candidate evaluated
    and every holdout peek is recorded; the luck bar is computed from the
    cumulative EFFECTIVE trial count for THIS holdout only. On ephemeral
    compute (RunPod) this file must live on durable storage or the bar
    silently resets."""

    def __init__(self, path, fingerprint):
        self.path = path
        self.key = fingerprint

    def _load(self):
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception:
            return {}

    def log(self, engine, n_candidates, holdout_peeks=1, note=""):
        d = self._load()
        d.setdefault(self.key, []).append(
            {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "engine": engine, "n_candidates": int(n_candidates),
             "holdout_peeks": int(holdout_peeks), "note": note})
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(d, f, indent=1)

    def effective_trials(self):
        eff = 0.0
        for r in self._load().get(self.key, []):
            eff += max(1.0, r["n_candidates"] ** EFFECTIVE_EXP) \
                   + max(0, r.get("holdout_peeks", 1) - 1)
        return max(1, int(round(eff)))

    def luck_bar(self, var_sr):
        """Expected max Sharpe among N null trials (Bailey/LdP asymptotic)."""
        n = self.effective_trials()
        if n < 2 or var_sr <= 0:
            return 0.0
        e = 0.5772156649
        z = statistics.NormalDist()
        return math.sqrt(var_sr) * (
            (1 - e) * z.inv_cdf(1 - 1.0 / n) + e * z.inv_cdf(1 - 1.0 / (n * math.e)))


def _binom_tail(wins: int, names: int) -> float:
    """Exact one-sided binomial p: P(X >= wins | n=names, q=0.5)."""
    if names <= 0:
        return 1.0
    return float(sum(math.comb(names, k)
                     for k in range(wins, names + 1)) * 0.5 ** names)


def breadth(genome, data, syms, cost, bar_seconds, min_trades=4):
    """Per-NAME evidence: on how many names does the genome's own pooled
    mean r_mult come out positive? Pooled stats double-count correlated
    names -- a market-wide burst looks like 400 trades but is one event.
    Breadth counts each name once. Returns (frac, binomial_p, n_names).
    (Deep-state lesson: pooled p said yes on 3,515 trades while breadth
    said coin-flip on the same panel.)"""
    wins = names = 0
    for s in syms:
        try:
            res = run_backtest(genome, data[s], cost, bar_seconds=bar_seconds)
        except Exception:
            continue
        rs = [t[3] for t in res.trades]
        if len(rs) < min_trades:
            continue
        names += 1
        wins += int(float(np.mean(rs)) > 0)
    if names == 0:
        return 0.0, 1.0, 0
    return wins / names, _binom_tail(wins, names), names


# ---------------- PBO (CSCV, lightweight) -----------------------------------
def pbo(cell_matrix: np.ndarray, n_splits=8, rng=None) -> float:
    """Probability of Backtest Overfitting via combinatorially symmetric CV.
    cell_matrix: candidates x cells performance. For each random half/half
    split of cells: rank candidates in-sample, take the IS best, ask whether
    its OOS rank is below median. PBO = fraction of splits where it is."""
    import random as _r
    rng = rng or _r.Random(0)
    m, c = cell_matrix.shape
    if m < 3 or c < 4:
        return 0.5
    below = 0
    for _ in range(n_splits):
        cols = list(range(c))
        rng.shuffle(cols)
        a, b = cols[: c // 2], cols[c // 2:]
        is_perf = cell_matrix[:, a].mean(axis=1)
        oos_perf = cell_matrix[:, b].mean(axis=1)
        best = int(np.argmax(is_perf))
        oos_rank = (oos_perf < oos_perf[best]).mean()
        if oos_rank < 0.5:
            below += 1
    return below / n_splits


# ---------------- synthetic stress ------------------------------------------
def make_block_index(n, rng, block=48):
    """One resample index shared by every name AND every context series in a
    path -- per-name independent indices would destroy name-vs-context and
    name-vs-name relationships, executing every conditional edge (real or
    not) the same way independent volume shuffling once did."""
    ridx = np.empty(n, dtype=int)
    i = 0
    while i < n:
        j = rng.randrange(1, max(2, n - block))
        take = min(block, n - i)
        ridx[i:i + take] = np.arange(j, min(j + take, n))
        i += take
    return ridx


def block_bootstrap(df: pd.DataFrame, rng, block=48, ridx=None) -> pd.DataFrame:
    """Resample WHOLE BARS in contiguous blocks and rebuild the price path.
    Returns, ranges, and volume travel TOGETHER, so genuine within-block
    structure (e.g. a volume spike followed by drift) survives; only the
    long-range historical ordering is destroyed. A candidate that needs the
    exact historical path dies here; one exploiting local structure does not.
    (First cut shuffled volume independently -- that silently executes every
    volume-conditioned edge, including real ones.)"""
    n = len(df)
    r = df["close"].pct_change().fillna(0).to_numpy()
    o_rel = (df["open"] / df["close"].shift()).fillna(1.0).to_numpy()
    h_rel = (df["high"] / np.maximum(df["open"], df["close"])).to_numpy()
    l_rel = (df["low"] / np.minimum(df["open"], df["close"])).to_numpy()
    v = df["volume"].to_numpy()
    if ridx is None:
        ridx = make_block_index(n, rng, block)
    close = 100 * np.cumprod(1 + r[ridx])
    prev = np.roll(close, 1); prev[0] = 100.0
    o = prev * o_rel[ridx]
    h = np.maximum(o, close) * np.clip(h_rel[ridx], 1.0, None)
    l = np.minimum(o, close) * np.clip(l_rel[ridx], None, 1.0)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": close,
                         "volume": v[ridx]}, index=df.index)


def stress_survivor(genome, data, syms, cost, bar_seconds, prepare_fn, rng,
                    n_paths=3, block_override=None):
    """Median holdout-style Sharpe across bootstrapped paths. Each path
    bootstraps EVERY name, then prepare_fn(dict) recomputes the full feature
    set -- atoms, motif matches, cross-sectional ranks -- on the joint panel,
    so motif/xs-conditioned genomes are stressed fairly instead of silently
    failing feature lookup and being auto-rejected. Costs at 1x here; cost
    stress is separate."""
    from statistics import median
    outs = []
    # common timestamp grid so ONE block index aligns every name + context
    common = None
    for s in syms:
        common = data[s].index if common is None else common.intersection(
            data[s].index)
    # block must EXCEED the strategy's claimed event horizon (entry context +
    # max_hold), else the resampler cuts through the mechanism it is testing
    # (fixed 48 was splitting a 15+40-bar dip->recovery object). Floor 48,
    # cap n//6 so shuffling still destroys long-range ordering. Recalibrated
    # 2026-08-05; negative/machinery/deepstate controls re-verified after.
    blk = block_override if block_override is not None else \
        int(np.clip(2 * getattr(genome, "max_hold", 24), 48,
                    max(48, len(common) // 6)))
    for p in range(n_paths):
        ridx = make_block_index(len(common), rng, block=blk)
        boots = {s: block_bootstrap(
                     data[s][["open", "high", "low", "close", "volume"]]
                     .reindex(common).ffill().copy(), rng, ridx=ridx)
                 for s in syms}
        prepare_fn(boots, ridx=ridx, common=common)
        cells = []
        for s in syms:
            try:
                res = run_backtest(genome, boots[s], cost,
                                   bar_seconds=bar_seconds)
            except Exception:
                continue
            rs = [t[3] for t in res.trades]
            if len(rs) >= 4:
                idx = boots[s].index
                span_yr = max((idx[-1] - idx[0]).total_seconds()
                              / 31557600.0, 1e-6)
                cells.append(trade_sharpe(rs, len(rs) / span_yr))
        outs.append(median(cells) if cells else 0.0)
    return (median(outs) if outs else 0.0), outs
