# PRIMORDIAL

A prior-free strategy synthesizer. Point it at an asset class — global stock
indexes, US equities, Coinbase spot, Kalshi perps — and a date range; it
evolves **complete strategy genomes** (entry logic, regime gate, timeframe,
direction, entry style, exits) from a toolkit of ~55 measurement atoms, and
judges them with a gauntlet built to say **no**.

The expected output of most runs is `0 survivors`. That is the system
working, not failing. The framework's primary product is honest negatives
delivered in hours; a survivor is a rare event that has earned real money's
attention. Full design rationale: `SPEC.md`.

## Quick start

```bash
pip install -r requirements.txt

# negative control first — random data, correct answer is zero survivors
python -m primordial run --manifest manifests/smoke_synthetic.yaml

# search all major global stock indexes (US, Europe, Korea, India, Japan...)
# on daily bars back to 2005
python -m primordial run --manifest manifests/global_indexes.yaml \
    --generations 20 --pop 40 --islands 3
```

Artifacts land in `runs/<timestamp>/`: `run.json` (everything about the run)
and `survivor_i.json` (self-contained deployable strategies, if any).

## What a candidate is

A `Genome`: timeframe (5m–1w, mutates to *adjacent* frames only), a typed GP
entry tree over measurement atoms, an optional regime tree, direction,
entry style (market / limit-at-k·ATR with honest non-fills / confirm-n-bars),
and exit genes (ATR stop, R target, time stop, trailing mode, max hold).
Everything a human would have defaulted is a gene. Position sizing is
permanently out of scope — that's capital allocation, not signal discovery.

## The vocabulary rule

Indicators appear as **measurements** (`rsi(n)` is a series the tree may
compare to anything), never as **strategies** (`RSI < 30` as a unit does not
exist). Human trading knowledge enters as vocabulary; conclusions must be
rediscovered against the judge or not used. Motifs (k-means centroids of
z-normed price windows, mined from **train data only**) enter the same way:
as `match_*` similarity series, frozen into any survivor that uses them.

## The judge

Signal at `close[t]`, fill at `open[t+1]`. Stops that gap through fill at the
open. Equity marked to market every bar. Funding/borrow carry charged per
held bar from the cost tier. Fitness is median-minus-λ·std of Sharpe across
name×time fold cells, penalized for complexity and turnover.

Survivors then face, in order:
1. **Holdout** (held-out names × held-out tail, outer embargo) — peeked once,
   and the peek is a logged trial
2. **Deflated luck bar** — expected max Sharpe of N null trials, where N is
   the *all-time* effective trial count against this manifest fingerprint,
   from a persistent ledger (`PRIMORDIAL_LEDGER`, default
   `~/.primordial/ledger.json`). On RunPod, put this on durable storage or
   the bar silently resets.
3. **PSR > 0.95** — evidence scales with trade count, no arbitrary minimums
4. **Cost stress** — must stay profitable at 2× and 3× costs
5. **PBO < 0.5** (CSCV) on the candidate pool
6. **Bootstrap paths** — whole bars resampled in contiguous blocks (returns,
   ranges, and volume travel together), must retain ≥50% of holdout Sharpe

## Controls — run these before believing anything

```bash
python -m pytest tests/          # fast unit controls (~3s)
python scripts/run_controls.py   # full-stack, slow (~10 min):
                                 #   negative  -> must be 0 survivors
                                 #   machinery -> planted true genome must survive
                                 #   search    -> evolution should find the plant
```

If the negative control ever produces a survivor, the judge leaks — stop
trusting every result until it's fixed. If machinery fails, a gate is killing
real edges. If only search fails, that's budget, not truth: raise
`--generations`/`--pop`, never a threshold.

## Downloading data (do this before searching)

```bash
python3.10 scripts/fetch_data.py                        # warm cache, all manifests
python3.10 scripts/fetch_data.py manifests/coinbase_spot.yaml
python3.10 scripts/fetch_data.py --source omnifeed      # route through omnifeed
```

Prefetch once (slow -- Coinbase pages 300 candles/request), then every run
reads parquet instantly. Set `source: omnifeed` in a manifest (or use the
override above) to route all downloads through the omnifeed package -- one
interface for every venue with pagination/retries/rate-limits handled;
install it first from your omnifeed repo (`pip install -e .`). Cache
entries are keyed by source: pick one per manifest and stick with it.

## Manifests

A run is fully specified by a YAML manifest: universe (explicit list or a
preset like `global_indexes` / `us_megacap` / `coinbase_liquid`), source
(`yfinance` / `coinbase` / `alpaca` / `csv:<dir>` / `synthetic`), asset class
(picks the adapter: session shape, valid atom blocks, shortability), cost
tier, timeframes, dates, holdout geometry. The manifest's **fingerprint**
(symbols + dates + holdout geometry) keys the trial ledger, so searching
crypto never deflates your index results.

Data is fetched once at the base timeframe, resampled up in memory, and
cached to parquet (`PRIMORDIAL_CACHE`). Note: the cache key does not include
a generator version — if you monkeypatch or change a source, use a fresh
cache dir.

## Deploying a survivor

```python
from primordial.runtime import Strategy
s = Strategy.load("runs/.../survivor_0.json")
df = ...                      # OHLCV at s.timeframe
fire = s.signal(df)           # 0/1 per bar — fill at NEXT bar open
spec = s.exit_spec()          # stop_atr / target_r / trail / max_hold
```

`runtime.py` imports only pure evaluation code — no engine, no judge — so the
deploy path stays thin. Survivor artifacts embed their motif centroids and
validation record.

## Honest odds

Read SPEC.md §22. A year of disciplined runs has maybe a 15–25% chance of
producing one deployable survivor. The number this framework cannot inflate
is the one that matters: what survives *after* the machine has been prevented
from lying to you. Use it to (1) falsify your own hypotheses fast, (2)
explore inside structurally-motivated regions, (3) kill ideas cheaply —
not as an oracle.
