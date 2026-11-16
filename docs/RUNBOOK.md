# RUNBOOK — how this framework is actually used

## Test a new idea (the standard loop)
1. Say the hypothesis in one sentence; write the prediction BEFORE running.
2. Free screen if one fits (train-only, logs nothing):
   scripts/explore_xs.py (cross-sectional) / scripts/explore_vol.py (vol).
   Flat table => family closed for free. Shape => it has earned ONE trial.
3. Fix the variant list. Five variants = five logged trials on that ground.
4. Build + judge:
       python3.10 scripts/make_genome.py "<expr>" --tf 1d ... > genomes/idea_universe.json
       python3.10 scripts/validate_genome.py --manifest manifests/<m>.yaml --genome genomes/idea_universe.json
5. Read the autopsy. The gate that killed it IS the finding. Never loosen a
   threshold to change a verdict.
6. python3.10 scripts/list_survivors.py --write, then commit.

## Open a new universe (ground protocol)
- HAND IDEAS FIRST, evolution LAST — a search burns ~50-60 effective trials
  into the fingerprint and raises the bar for every later hand idea there.
- Book any prior manual research as debt BEFORE the first look
  (Ledger.log("manual_research", n_candidates=X, holdout_peeks=Y)).
- Foreign/new lists: scripts/build_universe.py (US) or build_foreign.py
  (FTSE250/TSX/ASX/MDAX), sed a manifest from a neighbour, fetch, go.

## A pass is a candidate, not an edge
PASS => replicate on fresh ground => point-in-time universe re-test =>
pre-registered forward test. Only forward data upgrades "not rejected" to
"real". (Deep-state: PREREGISTRATION.md in deep-state-algo, 60 closed trades.)

## Health & honesty
- After ANY code change: python3.10 -m primordial doctor must say OK.
- Before trusting a search null: python3.10 scripts/run_controls.py —
  negative 0 survivors, machinery passes, search recovers the planted edge.
- python3.10 -m primordial ledger = every trial ever spent, per ground.
- Controls are the ONLY place scoring logic may ever be tuned (planted answer
  known). Never on real data. Gate changes ship with controls re-verified and
  a disclosure in CHANGELOG.md.

## Where things are written down
- SURVIVORS.md      — every verdict ever, auto-generated
- MECHANISM_MAP.md  — findings + the caveats against them
- genomes/          — every hypothesis exactly as asked
- runs/*/run.json   — raw artifacts (tracked); report.html per run (local)
- CHANGELOG.md      — how the tool itself changed, and why
- ~/.primordial/ledger.json — the honesty account (view: primordial ledger)
