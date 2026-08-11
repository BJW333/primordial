"""
Seed a manifest's ledger with the DISTRESS family's cross-universe debt.

The 2026-08-10 distress campaign tried 5 hand variants (base, stabilized,
volconf, deep, deep_short) across 14 validation runs on 10 universes
(sp400 x1, sp600 x5, uk ca de jp kr tw au in x1 each). The selection event
was FAMILY-WIDE: the same idea kept being re-asked on fresh ground until
some ground said yes. A fingerprint that has only ever been asked once
shows a 0.00 luck bar -- which is exactly how every international "pass"
was produced. Before believing any result on such ground, charge it the
family's history:

  python3.10 scripts/seed_ledger_distress.py --manifest manifests/jp_nikkei.yaml

Idempotent: refuses to double-charge a fingerprint it already seeded.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from primordial.judge import Ledger
from primordial.pipeline import LEDGER_PATH
from primordial.universe import Manifest

NOTE = ("distress family debt 2026-08-10: 5 hand variants, 14 holdout peeks "
        "across 10 universes (sp400, sp600 x5, uk, ca, de, jp, kr, tw, au, "
        "in). Family-wide selection; every ground this idea touches inherits "
        "the full history or its bar is a lie.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ledger", default=LEDGER_PATH)
    a = ap.parse_args()
    mf = Manifest.load(a.manifest)
    led = Ledger(a.ledger, mf.fingerprint())
    if any(r.get("note", "").startswith("distress family debt")
           for r in led._load().get(led.key, [])):
        print(f"fingerprint {mf.fingerprint()} already carries the distress "
              f"debt -- nothing charged.")
        sys.exit(0)
    before = led.effective_trials()
    led.log("manual_research", n_candidates=5, holdout_peeks=14, note=NOTE)
    after = led.effective_trials()
    print(f"fingerprint {mf.fingerprint()}")
    print(f"effective trials: {before} -> {after}   (ledger: {a.ledger})")
    print("luck bar on this ground now remembers the whole campaign.")
