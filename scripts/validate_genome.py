"""
Run ANY hand-built genome through the FULL gauntlet on a real manifest --
the generic version of validate_deepstate_real.py, for testing your ideas.

    1) write the idea as a genome JSON (template: --example > idea.json)
    2) python3.10 scripts/validate_genome.py --manifest manifests/us_sp400.yaml \
           --genome idea.json

Every run logs 1 candidate + 1 holdout peek on that manifest's fingerprint.
Testing five variants of one idea = five logged trials = a higher bar for all
of them. Decide the variant list BEFORE running any.
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from primordial import atoms as A, motifs as M       # noqa: E402
from primordial.data import fetch, multi_timeframe   # noqa: E402
from primordial.genome import Genome                 # noqa: E402
from primordial.judge import NameTimeSplit, Ledger   # noqa: E402
from primordial.universe import Manifest             # noqa: E402
import primordial.pipeline as P                      # noqa: E402

EXAMPLE = {
    "timeframe": "1d", "root_type": "bool",
    "entry_tree": {"op": "and", "ch": [
        {"op": "lt", "ch": [{"op": "term", "name": "rsi_m", "ch": []},
                             {"op": "const", "value": 0.3, "ch": []}]},
        {"op": "gt", "ch": [{"op": "term", "name": "dist_sma_s", "ch": []},
                             {"op": "const", "value": 0.0, "ch": []}]}]},
    "regime_tree": None, "direction": "long", "entry_style": "market",
    "entry_param": 1, "stop_atr": 3.0, "target_r": 0.0, "time_stop": 0,
    "trail_mode": "none", "max_hold": 40, "anchor_ema": 26,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest")
    p.add_argument("--genome", help="path to genome JSON")
    p.add_argument("--example", action="store_true",
                   help="print a template genome JSON and exit")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    if a.example:
        print(json.dumps(EXAMPLE, indent=1))
        return
    if not a.manifest or not a.genome:
        sys.exit("need --manifest and --genome (or --example)")

    with open(a.genome) as f:
        g = Genome.from_json(f.read())
    mf = Manifest.load(a.manifest)
    seed = a.seed if a.seed is not None else mf.seed
    rng = random.Random(seed)
    print(f"[idea gauntlet] {mf.name}")
    print(f"  genome: {g.describe()}")

    base = fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    if len(base) < 6:
        raise RuntimeError(f"only {len(base)} symbols fetched")
    mtf = multi_timeframe(base, mf.base_timeframe, mf.allowed_timeframes)
    if g.timeframe not in mtf:
        sys.exit(f"genome timeframe {g.timeframe} not in manifest "
                 f"timeframes {list(mtf)}")
    split = NameTimeSplit(list(base.keys()), mf.holdout_name_frac,
                          mf.holdout_time_frac, mf.embargo_bars, rng)
    mtf_train = {tf: split.train(d) for tf, d in mtf.items()}
    mtf_hold = {tf: split.holdout(d) for tf, d in mtf.items()}
    tfs = list(mtf.keys())
    motif_defs = M.mine(mtf_train[tfs[0]], split.train_syms, mf.motif_scales,
                        mf.n_motifs, seed=seed)
    bench = {tf: {} for tf in tfs}
    ctx = [s for s in mf.context_symbols if s not in mf.symbols]
    if ctx:
        braw = fetch(mf.source, ctx, mf.base_timeframe, mf.start, mf.end)
        bm = multi_timeframe(braw, mf.base_timeframe, mf.allowed_timeframes)
        for tf in tfs:
            bench[tf] = bm.get(tf, {})
    P._prepare(mtf_train, mf.adapter, split.train_syms, [], motif_defs, bench)
    P._prepare(mtf_hold, mf.adapter, [], split.hold_syms, motif_defs, bench)

    ledger = Ledger(P.LEDGER_PATH, mf.fingerprint())
    # NOTE: charge deferred until after the run -- a genome that fires ZERO
    # trades revealed nothing about the holdout and must not raise the bar
    # for the next idea. (2026-08-08: an unknown atom name produced a
    # never-firing genome that still cost a trial.)
    survivors, report = P._gauntlet([(5.0, g)], mtf_train, mtf_hold, split,
                                    mf, ledger, rng, verbose=True,
                                    motif_defs=motif_defs, bench_mtf=bench)
    r = report[0]
    if int(r.get("holdout_trades") or 0) > 0:
        ledger.log("idea_validate", n_candidates=1, holdout_peeks=1,
                   note=f"hand genome {g.describe()[:60]}")
    else:
        print("\n  !! ZERO TRADES -- genome never fired on the holdout.")
        print("     Nothing was learned; NO ledger trial charged.")
        print("     Check atom names / thresholds, then re-run.")
    import statistics as _st
    import time as _time
    print("\n  --- gate detail ---")
    print(f"  holdout SR {r.get('holdout_median_sharpe',0):+.2f} vs bar "
          f"{r.get('luck_bar',0):+.2f} | PSR {r.get('holdout_psr',0):.3f} | "
          f"trades {r.get('holdout_trades')}")
    if r.get("breadth") is not None:
        b = r["breadth"]
        print(f"  breadth: {b['frac']:.2f} on {b['names']} names, p={b['p']}")
    else:
        print("  breadth: skipped (fewer than 10 holdout names)")
    if r.get("cost_stress_sharpes") is not None:
        print(f"  cost stress 2x/3x: "
              f"{[round(x,2) for x in r['cost_stress_sharpes']]} (need >0)")
    else:
        print("  cost stress: not reached")
    if r.get("bootstrap_sharpes") is not None:
        m = _st.median(r["bootstrap_sharpes"])
        med = r.get("holdout_median_sharpe", 0)
        bar = r.get("luck_bar", 0)
        print(f"  bootstrap paths: {[round(x,2) for x in r['bootstrap_sharpes']]}"
              f" -> median {m:+.2f} (need > {0.5*med:+.2f} OR > "
              f"{max(1.0, bar):+.2f})")
        b48 = r.get("bootstrap_sharpes_blk48")
        if b48 is not None:
            print(f"  bootstrap (fixed-48 blocks, informational): "
                  f"{[round(x,2) for x in b48]} -> median "
                  f"{_st.median(b48):+.2f}")
    else:
        print("  bootstrap: not reached")
    run_dir = os.path.join("runs", _time.strftime("%Y%m%d_%H%M%S") + "_idea")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"manifest": a.manifest, "fingerprint": mf.fingerprint(),
                   "report": report,
                   "effective_trials_alltime": ledger.effective_trials()},
                  f, indent=1, default=str)
    try:
        from primordial.report import generate
        print("  report:", generate(run_dir))
    except Exception as e:
        print(f"  (report generation failed: {e})")
    print("\n" + ("IDEA SURVIVED THE FULL GAUNTLET." if survivors else
                  "Idea did NOT survive. The gate table says which claim "
                  "failed. Do not loosen a threshold; change the idea."))
    print(f"  all-time effective trials on this holdout: "
          f"{ledger.effective_trials()}")


if __name__ == "__main__":
    main()
