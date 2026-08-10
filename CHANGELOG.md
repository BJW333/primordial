# v0.1.7 — deep-state integration (Aug 2026)

The deep-state campaign's validated primitives, folded in as VOCABULARY and
JUDGE, never as doctrine. All 21 tests green; negative + machinery + search
controls re-pass; new equity machinery control passes.

## atoms.py
- `ema_spread_atr` — 12/26 EMA spread in ATR units, **sign-preserving**
  (`macd_line` is z-scored, which destroys below-zero semantics). Depth of a
  sag = −this.
- `spread_runlen` — bars the current sag has lasted (0 when spread ≥ 0;
  first sag bar counts 1). Freshness was the deep-state driver: fresh ≤5d
  +1.10%/tr vs stale >15d +0.10%/tr, replicated backward to 1962.

## genome.py
- New gene `anchor_ema: int = 0` — exit-at-reference family. When on: exit at
  the CLOSE of the first bar after the entry bar where close recrosses the
  reference EMA (long: ≥, short: ≤). Close-based, so it cannot harvest the
  bar's range; entry-bar doctrine (stop only) preserved. Fires after the stop
  check, before target/time. Stops/targets still coexist — the search decides
  whether they help (on mean reversion they should not: stop added back onto
  the anchor exit cost the deep-state sleeve Sharpe 1.27 → 0.92).
- Default 0 ⇒ every pre-0.1.7 survivor JSON loads unchanged. Reachable via
  random init (mostly-off), mutation ("anchor"), crossover carry; describe()
  shows `anchorN`.

## judge.py
- `breadth(genome, data, syms, cost, bar_seconds)` → (frac, exact binomial p,
  n_names): per-NAME evidence. Pooled stats double-count correlated names —
  400 trades in one market-wide burst is one event. Deep-state lesson: pooled
  p said yes on 3,515 trades while breadth said coin-flip on the same panel.
- `_binom_tail` — exact, dependency-free.

## pipeline.py
- Breadth gate in the gauntlet (after PSR, before cost stress): engages when
  the holdout has ≥ `BREADTH_MIN_NAMES` (10) names; requires frac > 0.5 and
  p < `BREADTH_MAX_P` (0.05). An "edge" on 2-3 tickers is a stock pick, not
  a structure. Row records `breadth: {frac, p, names}`.

## scripts
- `seed_ledger_deepstate.py` — logs the deep-state session's ~25 manual
  trials + 6 holdout peeks against the us_equities fingerprint, so any future
  search on that ground inherits the luck-bar debt.
- `control_deepstate.py` — equity machinery control: planted fresh-sag/
  anchor-revert edge on synthetic daily names, hand-built genome using the
  new atoms + anchor exit, must survive the full gauntlet. Is to the equity
  adapter what the volz plant is to crypto.

## tests
- `tests/test_deepstate.py` — 6 guards: atom semantics (sign preservation,
  runlen counting), anchor exit timing/price, never-on-entry-bar, genome
  backcompat + variation reach, exact binomial values, breadth on planted vs
  narrow.

## Migration
Drop-in. Old ledgers, manifests, and survivor JSONs load unchanged. Run
`scripts/seed_ledger_deepstate.py` once per equity ledger, and
`scripts/control_deepstate.py` + `scripts/run_controls.py` after pulling.

## v0.1.8 (2026-08-05)
- Behavioural constant-signal guard (tautology entries floored past warmup).
- Deterministic control seeding (hash() footgun removed).
- Bootstrap absolute escape: median > max(1.0, bar) also passes.

## v0.1.9 (2026-08-05..08)
- Bootstrap block scaled to the genome's own event horizon
  (clip(2 x max_hold, 48, n/6)); fixed-48 recorded alongside on every stress.
- CLI: version, ledger, doctor. Version banner on every command.
- validate_genome.py, make_genome.py, explore_xs.py / explore_vol.py,
  build_foreign.py, list_survivors.py (SURVIVORS.md registry).
- SEARCHER FIX (2026-08-08): scale-relative parsimony; coverage-aware
  min-cells floor (15% of attemptable); random immigrants (15%/cycle);
  search-control budget 10x24 -> 20x48 with CLI flags. Controls 3/3 green;
  search control recovers the planted mechanism. Judge unmodified throughout.

## Guards (2026-08-09)
- make_genome validates BOTH atom names and operator names against the real
  vocabularies (atoms.BLOCKS / tree.OPS). Root cause: `mult` is not an
  operator (`mul` is); the parser accepted it, evaluation raised per-symbol,
  the error was swallowed by the per-name try/except, and the run reported
  ZERO TRADES -- a silent null that reads exactly like a real finding.
- validate_genome defers the ledger charge until after the run; a genome that
  fires zero trades observed nothing and no longer raises the bar.
