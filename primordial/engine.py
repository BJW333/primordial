"""
engine.py -- island GP over full Genomes.

Islands: N populations evolving independently with occasional migration --
single-population GP reliably collapses into one basin. Each island can carry a
DIFFERENT terminal subset (meta-gene lite): productive corners of the toolkit
get discovered instead of guessed. No module globals anywhere -- every island
owns its own terminal list, so parallelism is safe.

Novelty pressure (lite): a candidate whose train-signal correlates > 0.95 with
a behavior archive entry takes a fitness haircut -- pushes the search into
unmapped regions instead of polishing one optimum.

Selection is tournament on scalar fitness; NSGA-II is a marked seam
(select() is the only thing to replace).
"""
from __future__ import annotations
import random
import numpy as np

from .genome import random_genome, mutate_genome, crossover_genomes
from .judge import fitness
from .hof import HallOfFame
from .data import BAR_SECONDS


class Island:
    def __init__(self, terminals, allowed_timeframes, rng, pop_size,
                 root_type="bool", allow_short=True, max_hold_hours=0.0):
        self.terminals = terminals
        self.tfs = allowed_timeframes
        self.rng = rng
        self.max_hold_hours = max_hold_hours
        self.pop = [random_genome(rng, terminals, allowed_timeframes,
                                  root_type=root_type, allow_short=allow_short,
                                  max_hold_hours=max_hold_hours)
                    for _ in range(pop_size)]


def _tournament(scored, rng, k=3):
    picks = rng.sample(scored, min(k, len(scored)))
    return max(picks, key=lambda x: x[0])[1]


def evolve(mtf_data: dict, train_syms, cost, terminals, allowed_timeframes,
           generations=12, pop_size=30, n_islands=3, elite=3, seed=42,
           migrate_every=4, hof_size=6, hof_max_corr=0.7,
           root_type="bool", allow_short=True, novelty_corr=0.95,
           verbose=True, on_generation=None, max_hold_hours=0.0):
    """mtf_data: {timeframe: {sym: df-with-atoms}}. Returns (best, hall,
    n_evaluated). Every genome evaluation is counted and must be logged to
    the ledger by the caller (pipeline does)."""
    rng = random.Random(seed)
    islands = []
    for i in range(n_islands):
        # island terminal subsets: island 0 sees everything; others drop a
        # random third -- crude meta-gene, lets fertile blocks reveal themselves
        terms = list(terminals)
        irng = random.Random(seed + 100 + i)
        if i > 0 and len(terms) > 12:
            irng.shuffle(terms)
            terms = terms[: max(12, int(len(terms) * 0.66))]
        islands.append(Island(terms, allowed_timeframes, irng, pop_size,
                              root_type, allow_short, max_hold_hours))

    hall = HallOfFame(max_size=hof_size, max_corr=hof_max_corr)
    behavior_archive = []          # concatenated signals of past elites
    best = None
    n_eval = 0

    def _sig_concat(g):
        data = mtf_data[g.timeframe]
        parts = []
        for s in train_syms:
            if s in data:
                try:
                    parts.append(np.nan_to_num(g.signal(data[s])))
                except Exception:
                    parts.append(np.zeros(len(data[s])))
        return np.concatenate(parts) if parts else np.zeros(1)

    def _novel_haircut(g, f):
        if f <= -9.0 or not behavior_archive:
            return f
        sig = _sig_concat(g)
        for b in behavior_archive[-30:]:
            m = min(len(sig), len(b))
            if m > 100 and sig[:m].std() > 0 and b[:m].std() > 0:
                if abs(np.corrcoef(sig[:m], b[:m])[0, 1]) > novelty_corr:
                    return f - 0.15
        return f

    for gen in range(generations):
        gen_best = None
        for isl in islands:
            scored = []
            for g in isl.pop:
                data = mtf_data.get(g.timeframe)
                if data is None:
                    scored.append((-9.99, g, {}))
                    continue
                f, info = fitness(g, data, [s for s in train_syms if s in data],
                                  cost, BAR_SECONDS[g.timeframe])
                n_eval += 1
                f = _novel_haircut(g, f)
                scored.append((f, g, info))
            scored.sort(key=lambda x: x[0], reverse=True)
            if gen_best is None or scored[0][0] > gen_best[0]:
                gen_best = scored[0]
            for f, g, info in scored[:3]:
                if f > -9.0:
                    data = mtf_data[g.timeframe]
                    hall.consider(f, g, lambda df, g=g: g.signal(df),
                                  data, [s for s in train_syms if s in data])
                    behavior_archive.append(_sig_concat(g))
            newpop = [scored[i][1] for i in range(min(elite, len(scored)))]
            while len(newpop) < len(isl.pop):
                p1 = _tournament(scored, isl.rng)
                p2 = _tournament(scored, isl.rng)
                c1, c2 = crossover_genomes(p1, p2, isl.rng)
                newpop.append(mutate_genome(c1, isl.rng, isl.terminals,
                                            isl.tfs, isl.max_hold_hours))
                if len(newpop) < len(isl.pop):
                    newpop.append(mutate_genome(c2, isl.rng, isl.terminals,
                                                isl.tfs, isl.max_hold_hours))
            isl.pop = newpop
        if best is None or gen_best[0] > best[0]:
            best = gen_best
        # RANDOM IMMIGRANTS: replace the worst 15% of each island with fresh
        # random genomes every migration cycle. Three separate production runs
        # stagnated for their final ~10 generations because the population
        # collapsed into one lineage; ring migration alone just spreads that
        # lineage. Elites at the front of pop are untouched. 2026-08-08.
        if gen and gen % migrate_every == 0:
            for isl in islands:
                k_new = max(1, int(0.15 * len(isl.pop)))
                for i in range(1, k_new + 1):   # back of pop = non-elite
                    isl.pop[-i] = random_genome(
                        isl.rng, isl.terminals, isl.tfs,
                        max_hold_hours=isl.max_hold_hours)
        if gen and gen % migrate_every == 0 and len(islands) > 1:
            for i, isl in enumerate(islands):        # ring migration
                nxt = islands[(i + 1) % len(islands)]
                nxt.pop[-1] = isl.pop[0]
        if verbose:
            bf, bg, bi = gen_best
            print(f"  gen {gen:2d} | best {bf:+.3f} | med SR "
                  f"{bi.get('median_sharpe', 0):+.2f} over {bi.get('n_cells', 0)} "
                  f"cells | {bg.describe()[:70]}", flush=True)
        if on_generation:
            on_generation(gen, gen_best)
    return best, hall, n_eval
