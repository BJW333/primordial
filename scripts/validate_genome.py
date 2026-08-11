"""
Run ANY hand-built genome through the FULL gauntlet on a real manifest --
the generic version of validate_deepstate_real.py, for testing your ideas.

    1) write the idea as a genome JSON (template: --example > idea.json)
    2) python3.10 scripts/validate_genome.py --manifest manifests/us_sp400.yaml \
           --genome idea.json --splits 5

Every run logs 1 candidate + one holdout peek PER SPLIT on that manifest's
fingerprint. Testing five variants of one idea = five logged trials = a higher
bar for all of them. Decide the variant list BEFORE running any.

--splits N (default 1): evaluate the same genome across N different
name/time holdout splits (seed, seed+1, ...). One split is an anecdote --
the 2026-08-10 record shows the same genome printing +0.92 and +0.31 on two
draws of the same universe. The number to believe is the MEDIAN across
splits; the range is the error bar. Each split that fires trades is charged
to the ledger as a peek, because each one genuinely looked at holdout data.
"""
import argparse
import json
import os
import random
import statistics as st
import sys
import time

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


def _gate_detail(r):
    print(f"  holdout SR {r.get('holdout_median_sharpe',0):+.2f} vs bar "
          f"{r.get('luck_bar',0):+.2f} | PSR {r.get('holdout_psr',0):.3f} | "
          f"trades {r.get('holdout_trades')}")
    if r.get("breadth") is not None:
        b = r["breadth"]
        print(f"  breadth: {b['frac']:.2f} on {b['names']} names, p={b['p']}")
    else:
        print("  breadth: skipped (not reached, or fewer than 10 holdout "
              "names)")
    if r.get("cost_stress_sharpes") is not None:
        print(f"  cost stress 2x/3x: "
              f"{[round(x,2) for x in r['cost_stress_sharpes']]} (need >0)")
    else:
        print("  cost stress: not reached")
    if r.get("bootstrap_sharpes") is not None:
        m = st.median(r["bootstrap_sharpes"])
        med = r.get("holdout_median_sharpe", 0)
        bar = r.get("luck_bar", 0)
        print(f"  bootstrap paths: "
              f"{[round(x,2) for x in r['bootstrap_sharpes']]}"
              f" -> median {m:+.2f} (need > {0.5*med:+.2f} OR > "
              f"{max(1.0, bar):+.2f})")
        b48 = r.get("bootstrap_sharpes_blk48")
        if b48 is not None:
            print(f"  bootstrap (fixed-48 blocks, informational): "
                  f"{[round(x,2) for x in b48]} -> median "
                  f"{st.median(b48):+.2f}")
    else:
        print("  bootstrap: not reached")


def _eval_one_split(g, mf, mtf, tfs, bench, ledger, split_seed):
    """Build one split, mine motifs on ITS train slice only, prepare atoms,
    run the gauntlet once. Returns (report_row, membership)."""
    rng = random.Random(split_seed)
    keys = list(mtf[tfs[0]].keys())
    split = NameTimeSplit(keys, mf.holdout_name_frac,
                          mf.holdout_time_frac, mf.embargo_bars, rng)
    mtf_train = {tf: split.train(d) for tf, d in mtf.items()}
    mtf_hold = {tf: split.holdout(d) for tf, d in mtf.items()}
    motif_defs = M.mine(mtf_train[tfs[0]], split.train_syms, mf.motif_scales,
                        mf.n_motifs, seed=split_seed)
    P._prepare(mtf_train, mf.adapter, split.train_syms, [], motif_defs, bench)
    P._prepare(mtf_hold, mf.adapter, [], split.hold_syms, motif_defs, bench)
    _, report = P._gauntlet([(5.0, g)], mtf_train, mtf_hold, split,
                            mf, ledger, rng, verbose=True,
                            motif_defs=motif_defs, bench_mtf=bench)
    return report[0], split.membership()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest")
    p.add_argument("--genome", help="path to genome JSON")
    p.add_argument("--example", action="store_true",
                   help="print a template genome JSON and exit")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--splits", type=int, default=1,
                   help="number of independent holdout splits (seed, seed+1, "
                        "...); the median across splits is the result")
    a = p.parse_args()
    if a.example:
        print(json.dumps(EXAMPLE, indent=1))
        return
    if not a.manifest or not a.genome:
        sys.exit("need --manifest and --genome (or --example)")
    if a.splits < 1:
        sys.exit("--splits must be >= 1")

    with open(a.genome) as f:
        g = Genome.from_json(f.read())
    mf = Manifest.load(a.manifest)
    seed = a.seed if a.seed is not None else mf.seed
    print(f"[idea gauntlet] {mf.name} | splits {a.splits} (seed {seed}..)")
    print(f"  genome: {g.describe()}")

    base = fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    if len(base) < 6:
        raise RuntimeError(f"only {len(base)} symbols fetched")
    missing = sorted(set(mf.symbols) - set(base.keys()))
    if missing:
        print(f"  !! {len(missing)} manifest symbols did NOT fetch: "
              f"{', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    mtf = multi_timeframe(base, mf.base_timeframe, mf.allowed_timeframes)
    if g.timeframe not in mtf:
        sys.exit(f"genome timeframe {g.timeframe} not in manifest "
                 f"timeframes {list(mtf)}")
    tfs = list(mtf.keys())
    bench = {tf: {} for tf in tfs}
    ctx = [s for s in mf.context_symbols if s not in mf.symbols]
    if ctx:
        braw = fetch(mf.source, ctx, mf.base_timeframe, mf.start, mf.end)
        bm = multi_timeframe(braw, mf.base_timeframe, mf.allowed_timeframes)
        for tf in tfs:
            bench[tf] = bm.get(tf, {})

    ledger = Ledger(P.LEDGER_PATH, mf.fingerprint())
    # NOTE: charge deferred until after all splits -- splits that fire ZERO
    # trades revealed nothing about the holdout and are not charged.
    reports, memberships = [], []
    for i in range(a.splits):
        print(f"\n  --- split {i + 1}/{a.splits} (seed {seed + i}) ---")
        r, mem = _eval_one_split(g, mf, mtf, tfs, bench, ledger, seed + i)
        _gate_detail(r)
        reports.append(r)
        memberships.append(mem)

    fired = [r for r in reports
             if int(r.get("holdout_trades") or 0) > 0]
    if fired:
        ledger.log("idea_validate", n_candidates=1,
                   holdout_peeks=len(fired),
                   note=f"hand genome x{len(fired)} splits "
                        f"{g.describe()[:48]}")
    else:
        print("\n  !! ZERO TRADES on every split -- genome never fired.")
        print("     Nothing was learned; NO ledger trial charged.")
        print("     Check atom names / thresholds, then re-run.")

    summary = None
    if fired:
        srs = [r["holdout_median_sharpe"] for r in fired]
        bars = [r.get("luck_bar", 0) for r in fired]
        beat = sum(1 for r in fired
                   if r["holdout_median_sharpe"] > r.get("luck_bar", 0))
        summary = {
            "splits_run": a.splits, "splits_fired": len(fired),
            "sr_median": st.median(srs),
            "sr_min": min(srs), "sr_max": max(srs),
            "bar_max": max(bars), "splits_beating_bar": beat,
        }
        print(f"\n  ===== across {len(fired)} split(s) =====")
        print(f"  SR median {summary['sr_median']:+.2f} | range "
              f"[{summary['sr_min']:+.2f}, {summary['sr_max']:+.2f}] | "
              f"beat bar on {beat}/{len(fired)}")
        if a.splits == 1:
            print("  (single split -- treat as an anecdote; re-run with "
                  "--splits 5 before believing it)")
        elif summary["sr_max"] - summary["sr_min"] > 0.4:
            print("  !! split variance is large -- any single-split verdict "
                  "on this universe is noise")

    run_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S") + "_idea")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"manifest": a.manifest, "fingerprint": mf.fingerprint(),
                   "seed": seed, "splits": a.splits,
                   "report": reports, "split_summary": summary,
                   "split_memberships": memberships,
                   "symbols_missing": missing,
                   "effective_trials_alltime": ledger.effective_trials()},
                  f, indent=1, default=str)
    try:
        from primordial.report import generate
        print("\n  report:", generate(run_dir))
    except Exception as e:
        print(f"\n  (report generation failed: {e})")
    n_surv = sum(1 for r in reports if r.get("survived"))
    if fired:
        print("\n" + (f"IDEA SURVIVED THE FULL GAUNTLET on {n_surv}/"
                      f"{len(fired)} split(s)." if n_surv else
                      "Idea did NOT survive on any split. The gate table "
                      "says which claim failed. Do not loosen a threshold; "
                      "change the idea."))
        if 0 < n_surv < len(fired):
            print("  Mixed splits are NOT a pass -- a real mechanism should "
                  "not care which names were held out.")
    print(f"  all-time effective trials on this holdout: "
          f"{ledger.effective_trials()}")
    print(f"  artifacts: {run_dir}")


if __name__ == "__main__":
    main()
