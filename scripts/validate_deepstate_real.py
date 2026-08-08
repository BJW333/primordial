"""
Run YOUR deep-state genome -- hand-built, not evolved -- through the FULL
gauntlet on REAL data. This is the honest yes/no on the strategy itself:
holdout names x holdout years, luck bar from the seeded ledger, PSR, breadth,
cost stress, PBO skipped (single candidate), bootstrap paths.

    python3.10 scripts/validate_deepstate_real.py --manifest manifests/us_equities.yaml

Run scripts/seed_ledger_deepstate.py FIRST (once, ever) so the bar remembers
your prior manual research on this data. Every run of THIS script also logs
one candidate + one holdout peek -- rerunning it to fish is self-defeating by
design.

Tunable genes via flags if you want to test variants (each variant = another
logged trial):
    --spread-thresh -0.25   entry: ema_spread_atr below this (sag depth; spec)
    --max-runlen 15         entry: sag no older than this many bars
    --anchor 26             exit at close crossing back over this EMA
    --stop 6.0              disaster stop, ATR units (wide on purpose)
"""
import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from primordial import tree as T                      # noqa: E402
from primordial import atoms as A, motifs as M        # noqa: E402
from primordial.data import fetch, multi_timeframe    # noqa: E402
from primordial.genome import Genome                  # noqa: E402
from primordial.judge import NameTimeSplit, Ledger    # noqa: E402
from primordial.universe import Manifest              # noqa: E402
import primordial.pipeline as P                       # noqa: E402


def build_genome(a) -> Genome:
    return Genome(
        timeframe="1d", root_type="bool",
        entry_tree=T.node("and", [
            T.node("lt", [T.term("ema_spread_atr"), T.const(a.spread_thresh)]),
            T.node("and", [
                T.node("lte", [T.term("spread_runlen"), T.const(a.max_runlen)]),
                # spec gate is SMA200 (dist_sma_s is SMA100 -- close, not it):
                T.node("gt", [T.term("close"),
                              T.node("roll_mean", [T.term("close")], n=200)]),
            ]),
        ]),
        regime_tree=None, direction="long", entry_style="market",
        entry_param=1, stop_atr=a.stop, target_r=0.0, time_stop=0,
        trail_mode="none", max_hold=40, anchor_ema=a.anchor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    # defaults = the FROZEN spec in deep_state.py, verified against the repo:
    #   MACD(12,26) < 0 with depth >= 0.25 ATR; close > SMA200; runlen <= 15;
    #   exit first close >= EMA26; max_hold 40; NO STOP ("every stop tested
    #   destroyed the edge"). --stop 99 keeps the stop unreachable while
    #   preserving R-unit accounting (Sharpe is scale-invariant in R).
    p.add_argument("--spread-thresh", type=float, default=-0.25)
    p.add_argument("--max-runlen", type=int, default=15)
    p.add_argument("--anchor", type=int, default=26)
    p.add_argument("--stop", type=float, default=99.0)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    mf = Manifest.load(a.manifest)
    seed = a.seed if a.seed is not None else mf.seed
    rng = random.Random(seed)
    g = build_genome(a)
    print(f"[deep-state real-data gauntlet] {mf.name}")
    print(f"  genome: {g.describe()}")

    # replicate the pipeline's data path exactly (same split seed => the same
    # holdout the ledger's trials are booked against)
    base = fetch(mf.source, mf.symbols, mf.base_timeframe, mf.start, mf.end)
    if len(base) < 6:
        raise RuntimeError(f"only {len(base)} symbols fetched")
    mtf = multi_timeframe(base, mf.base_timeframe, mf.allowed_timeframes)
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
    ledger.log("deepstate_validate", n_candidates=1, holdout_peeks=1,
               note=f"hand genome {g.describe()[:60]}")
    survivors, report = P._gauntlet([(5.0, g)], mtf_train, mtf_hold, split,
                                    mf, ledger, rng, verbose=True,
                                    motif_defs=motif_defs, bench_mtf=bench)
    r = report[0]
    import json as _json, time as _time
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
        import statistics as _st
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
    run_dir = os.path.join("runs", _time.strftime("%Y%m%d_%H%M%S") + "_dsval")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        _json.dump({"manifest": a.manifest, "fingerprint": mf.fingerprint(),
                    "report": report,
                    "effective_trials_alltime": ledger.effective_trials()},
                   f, indent=1, default=str)
    try:
        from primordial.report import generate
        print("  report:", generate(run_dir))
    except Exception as e:
        print(f"  (report generation failed: {e})")
    print()
    if survivors:
        print("DEEP-STATE SURVIVED THE FULL GAUNTLET ON REAL DATA.")
        print("Next: NARRS the parameter neighborhood, then paper-forward.")
    else:
        print("Deep-state did NOT survive on real data. The gate table above "
              "says which claim failed -- that is the finding. Do not loosen "
              "a threshold; change the hypothesis.")
    print(f"  all-time effective trials on this holdout: "
          f"{ledger.effective_trials()}")


if __name__ == "__main__":
    main()
