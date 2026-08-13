"""
pipeline.py -- the one entry point. Owns the honest sequence so it cannot be
bypassed: resolve -> fetch/cache -> split -> mine motifs (train only) -> atoms
+ dedupe -> evolve (islands) -> gauntlet (holdout once + cost stress + PSR +
deflated bar + PBO + bootstrap stress) -> persist artifacts -> log ledger ->
export survivors.

Run: python -m primordial run --manifest manifests/global_indexes.yaml
"""
from __future__ import annotations
import json
import os
import random
import time
import numpy as np

from . import atoms as A
from . import motifs as M
from .data import fetch, multi_timeframe, BAR_SECONDS
from .engine import evolve
from .genome import run_backtest, Genome
from .judge import (NameTimeSplit, Ledger, fitness, fold_cells, psr,
                    trade_sharpe, pbo, stress_survivor, max_drawdown, breadth)
from .universe import Manifest

LEDGER_PATH = os.environ.get("PRIMORDIAL_LEDGER",
                             os.path.expanduser("~/.primordial/ledger.json"))
PSR_GATE = 0.95
PBO_CEIL = 0.5
COST_STRESS = (1.0, 2.0, 3.0)
STRESS_MIN_FRAC = 0.5      # bootstrap-path Sharpe must keep >= 50% of holdout
BREADTH_MIN_NAMES = 10     # breadth gate engages only with real cross-section
BREADTH_MAX_P = 0.05       # per-name binomial: edge must exist across names


def apply_exclusions(kept: list, all_terminals: list, exclude: list) -> list:
    """Filter kept terminals by name. Validates against the PRE-dedupe list
    so an atom that was merged away doesn't read as a typo; a name that never
    existed raises (silent no-op exclusions are how the mult/mul class of bug
    ships). Returns a new list."""
    if not exclude:
        return kept
    unknown = [a for a in exclude if a not in all_terminals]
    if unknown:
        raise ValueError(f"exclude_atoms not in terminal set: {unknown} -- "
                         f"check spelling against atoms.py")
    ex = set(exclude)
    return [k for k in kept if k not in ex]


def _prepare(mtf, adapter, train_syms, hold_syms, motif_defs,
             benchmarks=None):
    """Compute atoms + motif matches + context per timeframe. Cross-sectional
    and universe-context atoms are computed WITHIN each name-set separately
    (train features never embed holdout names); benchmark context is an
    external data stream shared by both. Returns terminal list (train side)."""
    terminals = None
    for tf, data in mtf.items():
        cols = None
        for s, df in data.items():
            cols = A.compute_atoms(df, adapter.atom_blocks, adapter.always_open)
        mcols = M.add_motif_atoms(data, motif_defs) if motif_defs else []
        ts = [s for s in train_syms if s in data]
        hs = [s for s in hold_syms if s in data]
        xs_t = A.add_cross_sectional(data, ts) if len(ts) >= 5 else []
        _ = A.add_cross_sectional(data, hs) if len(hs) >= 5 else []
        b = (benchmarks or {}).get(tf, {})
        ctx_t = A.add_context(data, ts, b) if ts else []
        _ = A.add_context(data, hs, b) if hs else []
        if terminals is None:
            terminals = (cols or []) + mcols + xs_t + ctx_t
    return terminals


def run(manifest_path: str, generations=12, pop_size=30, n_islands=3,
        seed=None, verbose=True, out_dir="runs"):
    mf = Manifest.load(manifest_path)
    seed = seed if seed is not None else mf.seed
    rng = random.Random(seed)
    adapter = mf.adapter
    t0 = time.time()

    # 1. fetch base timeframe once; resample up
    base = fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    if len(base) < 6:
        raise RuntimeError(f"only {len(base)} symbols fetched -- not enough")
    mtf = multi_timeframe(base, mf.base_timeframe, mf.allowed_timeframes)
    tfs = list(mtf.keys())
    # benchmark/context reference series (SPY, TLT, ...) -- data streams, not
    # universe members: fetched separately, never searched, never held out
    bench_mtf = {tf: {} for tf in tfs}
    ctx_syms = [s for s in mf.context_symbols if s not in mf.symbols]
    if ctx_syms:
        braw = fetch(mf.source, ctx_syms, mf.base_timeframe, mf.start, mf.end)
        bm = multi_timeframe(braw, mf.base_timeframe, mf.allowed_timeframes)
        for tf in tfs:
            bench_mtf[tf] = bm.get(tf, {})

    # 2. split names + time with outer embargo
    split = NameTimeSplit(list(base.keys()), mf.holdout_name_frac,
                          mf.holdout_time_frac, mf.embargo_bars, rng)
    mtf_train = {tf: split.train(d) for tf, d in mtf.items()}
    mtf_hold = {tf: split.holdout(d) for tf, d in mtf.items()}

    # 3. motifs mined on TRAIN slice only, at base tf; frozen for the run
    motif_defs = M.mine(mtf_train[tfs[0]], split.train_syms,
                        mf.motif_scales, mf.n_motifs, seed=seed)

    # 4. atoms everywhere; dedupe decided on train
    terminals = _prepare(mtf_train, adapter, split.train_syms, [], motif_defs,
                         bench_mtf)
    _ = _prepare(mtf_hold, adapter, [], split.hold_syms, motif_defs,
                 bench_mtf)
    kept, merges = A.dedupe(mtf_train[tfs[0]], terminals)
    kept = apply_exclusions(kept, terminals, mf.exclude_atoms)
    if verbose:
        print(f"[{mf.name}] {len(base)} syms | tfs {tfs} | atoms "
              f"{len(terminals)} -> {len(kept)} after |rho|>0.95 dedupe "
              f"({len(merges)} merged) | motifs {len(motif_defs)}")

    ledger = Ledger(LEDGER_PATH, mf.fingerprint())
    allow_short = adapter.shortable

    # 5. evolve
    best, hall, n_eval = evolve(
        mtf_train, split.train_syms, mf.cost, kept, tfs,
        generations=generations, pop_size=pop_size, n_islands=n_islands,
        seed=seed, allow_short=allow_short, verbose=verbose,
        max_hold_hours=mf.max_hold_hours)

    # 6. gauntlet the HOF on the holdout -- each peek is a logged trial
    candidates = hall.payloads() or ([( best[0], best[1] )] if best else [])
    ledger.log("evolve", n_candidates=n_eval, holdout_peeks=len(candidates))
    survivors, report = _gauntlet(candidates, mtf_train, mtf_hold, split, mf,
                                  ledger, rng, verbose, motif_defs, bench_mtf)

    # 7. persist artifacts
    run_dir = os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    art = {
        "manifest": manifest_path, "fingerprint": mf.fingerprint(),
        "seed": seed, "n_evaluated": n_eval,
        "effective_trials_alltime": ledger.effective_trials(),
        "runtime_sec": round(time.time() - t0, 1),
        "atom_merges": merges, "report": report,
        # audit: exactly which names landed where, and which never arrived --
        # a silent fetch drop must be visible in the artifact, not inferred
        # from a trade-count discrepancy two runs later.
        "split": split.membership(),
        "symbols_missing": sorted(set(mf.symbols) - set(base.keys())),
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(art, f, indent=1)
    for i, s in enumerate(survivors):
        with open(os.path.join(run_dir, f"survivor_{i}.json"), "w") as f:
            f.write(export_survivor(s["genome"], mf, motif_defs, s))
    if verbose:
        print(f"\n{'='*66}\n{len(survivors)} survivor(s) of {len(candidates)} "
              f"candidates | all-time effective trials on this holdout: "
              f"{ledger.effective_trials()}\nartifacts: {run_dir}")
        if not survivors:
            print("Nothing survived. That is a real answer -- the bar is doing "
                  "its job. Do NOT loosen a threshold to change it.")
    try:
        from .report import generate
        print("report:", generate(run_dir))
    except Exception as e:
        print(f"(report generation failed: {e})")
    return survivors, report, run_dir


def _gauntlet(candidates, mtf_train, mtf_hold, split, mf, ledger, rng,
              verbose, motif_defs=None, bench_mtf=None):
    motif_defs = motif_defs or {}
    bench_mtf = bench_mtf or {}
    """Holdout (once) -> luck bar -> PSR -> cost stress -> PBO -> bootstrap."""
    report = []
    survivors = []
    # pass 1: holdout stats for every candidate
    evals = []
    for fit_train, g in candidates:
        tf = g.timeframe
        data_h = mtf_hold.get(tf, {})
        hs = [s for s in split.hold_syms if s in data_h]
        cells, pooled, _ = fold_cells(g, data_h, hs, mf.cost, BAR_SECONDS[tf],
                                      k=2)
        med = float(np.median(cells)) if cells else 0.0
        evals.append((fit_train, g, tf, hs, data_h, med, pooled))
    # null spread estimated ROBUSTLY from the candidates' own holdout Sharpes:
    # MAD ignores the outliers that real edges are, so junk sets the scale.
    # (First cut used variance of train FITNESS -- one degenerate overfit
    # poisoned it and the bar went to +164.)
    meds = [e[5] for e in evals]
    if len(meds) >= 3:
        mad = float(np.median(np.abs(np.array(meds) - np.median(meds))))
        var_sr = float(np.clip((1.4826 * mad) ** 2, 0.25, 25.0))
    else:
        var_sr = 0.25
    bar = ledger.luck_bar(var_sr)
    for fit_train, g, tf, hs, data_h, med, pooled in evals:
        row = {"genome": g.describe(), "genome_json": g.to_json(),
               "timeframe": tf, "train_fit": fit_train}
        row.update(holdout_median_sharpe=med, holdout_trades=len(pooled),
                   holdout_psr=psr(pooled), luck_bar=bar)
        # pooled holdout equity curve (downsampled) for the report
        try:
            eqs = []
            for s in hs:
                res = run_backtest(g, data_h[s], mf.cost,
                                   bar_seconds=BAR_SECONDS[tf])
                if len(res.equity) > 10:
                    eqs.append(res.equity)
            if eqs:
                import pandas as _pd
                pooled_eq = _pd.concat(eqs, axis=1).ffill().mean(axis=1)
                step = max(1, len(pooled_eq) // 300)
                pe = pooled_eq.iloc[::step]
                row["holdout_equity"] = [[str(t), float(v)]
                                         for t, v in pe.items()]
        except Exception:
            pass
        ok = med > bar and row["holdout_psr"] > PSR_GATE
        # breadth: pooled stats double-count correlated names; an "edge" that
        # lives on 2-3 tickers is a stock pick, not a structure. Cheap, so it
        # runs before the expensive stress gates.
        if ok and len(hs) >= BREADTH_MIN_NAMES:
            bfrac, bp, bn = breadth(g, data_h, hs, mf.cost, BAR_SECONDS[tf])
            row["breadth"] = {"frac": round(bfrac, 3), "p": round(bp, 4),
                              "names": bn}
            if bn >= BREADTH_MIN_NAMES:
                ok = bfrac > 0.5 and bp < BREADTH_MAX_P
        # cost stress: dies at 2x costs => spread-capture illusion
        if ok:
            stress_meds = []
            for m in COST_STRESS[1:]:
                cs, _, _ = fold_cells(g, data_h, hs, mf.cost.stressed(m),
                                      BAR_SECONDS[tf], k=2)
                stress_meds.append(float(np.median(cs)) if cs else 0.0)
            row["cost_stress_sharpes"] = stress_meds
            ok = all(s > 0 for s in stress_meds)
        # PBO on the train cell matrix of all candidates x this tf
        if ok and len(candidates) >= 3:
            rows = []
            data_t = mtf_train[tf]
            ts = [s for s in split.train_syms if s in data_t]
            for _, gg in candidates:
                if gg.timeframe != tf:
                    continue
                cc, _, _ = fold_cells(gg, data_t, ts, mf.cost, BAR_SECONDS[tf])
                if cc:
                    rows.append(cc)
            # truncate to the common minimum -- zero-padding fabricates cells
            mat = []
            if len(rows) >= 3:
                m = min(len(r) for r in rows)
                if m >= 4:
                    mat = [r[:m] for r in rows]
            if len(mat) >= 3:
                row["pbo"] = pbo(np.array(mat), rng=rng)
                ok = row["pbo"] < PBO_CEIL
        # bootstrap stress: structure must survive resampled paths
        if ok:
            adapter = mf.adapter
            btf = bench_mtf.get(tf, {})
            def prepare_fn(boots, ridx=None, common=None, _m=motif_defs,
                           _hs=hs, _b=btf):
                for _s, _df in boots.items():
                    A.compute_atoms(_df, adapter.atom_blocks,
                                    adapter.always_open)
                if _m:
                    M.add_motif_atoms(boots, _m)
                names = [s for s in _hs if s in boots]
                if len(names) >= 5:
                    A.add_cross_sectional(boots, names)
                # benchmark series ride the SAME block index so the
                # name-vs-context relationship survives the resample
                bb = {}
                for _bn, _bdf in _b.items():
                    al = _bdf.reindex(common).ffill() if common is not None                          else _bdf
                    if ridx is not None and len(al) == len(common):
                        al = al.iloc[ridx].set_axis(common)
                    bb[_bn] = al
                A.add_context(boots, names, bb)
            sm, paths = stress_survivor(g, mtf_hold[tf], hs, mf.cost,
                                        BAR_SECONDS[tf], prepare_fn, rng)
            row["bootstrap_sharpes"] = paths
            # TRANSPARENCY (dual-block): also record fixed-48 blocks. Gate
            # decides on mechanism-scaled paths above (pre-committed
            # 2026-08-05); this second set makes any pass that exists ONLY
            # under the recalibrated sizing visible on its face.
            _, paths48 = stress_survivor(g, mtf_hold[tf], hs, mf.cost,
                                         BAR_SECONDS[tf], prepare_fn, rng,
                                         block_override=48)
            row["bootstrap_sharpes_blk48"] = paths48
            # 50%-of-holdout for normal-range edges, PLUS an absolute escape:
            # a very strong true edge (holdout SR 30+) loses adjacency at
            # block boundaries and can't retain 50%, but resampled paths that
            # STILL beat max(1.0, luck bar) have proven the structure
            # survives resampling (control caught the relative-only rule
            # rejecting a planted SR-33 edge whose bootstrap held SR ~3.2).
            ok = (sm > STRESS_MIN_FRAC * med or sm > max(1.0, bar)) \
                 if med > 0 else False
        row["survived"] = bool(ok)
        report.append(row)
        if ok:
            survivors.append({"genome": g, **{k: v for k, v in row.items()
                                              if k != "genome"}})
        if verbose:
            tag = "SURVIVED" if ok else "rejected"
            print(f"  [{tag}] {tf} holdout SR {med:+.2f} vs bar {bar:+.2f} | "
                  f"PSR {row['holdout_psr']:.2f} | {g.describe()[:60]}")
    return survivors, report


def export_survivor(g: Genome, mf: Manifest, motif_defs: dict, row: dict) -> str:
    """Self-contained deployable artifact: genome + only the motifs it uses
    (FROZEN centroids) + cost tier + validation record. runtime.py loads this
    with no search-framework imports."""
    used = {k: v for k, v in motif_defs.items()
            if f"match_{k}" in json.dumps(g.entry_tree) +
               json.dumps(g.regime_tree or {})}
    return json.dumps({
        "genome": json.loads(g.to_json()),
        "motifs": used,
        "asset_class": mf.asset_class,
        "cost_tier": mf.cost_tier,
        "atom_blocks": list(mf.adapter.atom_blocks),
        "always_open": mf.adapter.always_open,
        "validation": {k: v for k, v in row.items() if k != "genome"},
        "manifest_fingerprint": mf.fingerprint(),
    }, indent=1)
