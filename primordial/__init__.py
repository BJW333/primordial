"""
PRIMORDIAL -- a prior-free strategy synthesizer.

Point it at an asset class (equities, global indexes, crypto spot, Kalshi
perps) and a date range; it evolves complete strategy genomes -- entry tree,
regime gate, timeframe, direction, exits -- from a toolkit of measurement
atoms, judged by a gauntlet it cannot fool: next-open fills, honest costs,
k-fold name x time fitness, a per-manifest trial ledger with a deflated luck
bar, PSR, PBO, cost stress, and bootstrap path stress.

The expected output of most runs is "nothing survived." That is the system
working. See SPEC.md for the full design.
"""
from .universe import Manifest, COST_TIERS, ADAPTERS, PRESETS
from .genome import Genome, run_backtest, random_genome
from .pipeline import run
from .runtime import Strategy
from . import tree, atoms, motifs, judge, engine, data

__version__ = "0.1.9"
