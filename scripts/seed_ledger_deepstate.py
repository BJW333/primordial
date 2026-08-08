"""
Seed the ledger with the deep-state research session's manual trials.

The Jul-Aug 2026 deep-state campaign examined ~25 rule variants against US
large-cap daily data by hand, plus ~6 holdout-style peeks (gate v1/v2, the
selection walk-forward, the midcap gate, the sizing frontier). Those trials
happened outside PRIMORDIAL, but they were trials on the same ground -- if
this fingerprint's holdout is ever searched, the luck bar must remember them
or the deflation is a lie.

  python scripts/seed_ledger_deepstate.py --manifest manifests/us_equities.yaml
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from primordial.judge import Ledger
from primordial.pipeline import LEDGER_PATH
from primordial.universe import Manifest

NOTE = ("deep-state session Jul-Aug 2026: ~25 variants examined by hand on US "
        "large-cap daily; gates run: largecap breadth 68/100 p=2e-4, index OOS "
        "2019-26, midcap 273/385 p<1e-5, selection walk-forward, sizing "
        "frontier. Seeded so future searches on this ground inherit the debt.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifests/us_equities.yaml"))
    ap.add_argument("--ledger", default=LEDGER_PATH)
    a = ap.parse_args()
    mf = Manifest.load(a.manifest)
    led = Ledger(a.ledger, mf.fingerprint())
    before = led.effective_trials()
    led.log("manual_research", n_candidates=25, holdout_peeks=6, note=NOTE)
    after = led.effective_trials()
    print(f"fingerprint {mf.fingerprint()}")
    print(f"effective trials: {before} -> {after}   (ledger: {a.ledger})")
    print("luck bar for this ground now carries the deep-state session's debt.")
