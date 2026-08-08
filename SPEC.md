# PRIMORDIAL — a prior-free strategy synthesizer

*Working name. Full build specification, v2.*
*Successor to and consumer of: `edgesearch`, `pattern_scanner`.*

---

# 0. What this document is

A complete build spec: what the system is, what every part does, why it exists,
how it works mechanically, what could go wrong with it, and in what order to
build it. Written to be buildable without further conversation.

---

# 1. The idea

Point the system at an asset class and a date range. Give it a deep toolkit of
**measurements** and one **incorruptible judge**. It evolves complete strategy
specifications — its own indicators, patterns, entry logic, regime conditions,
exits, timeframe, and cross-sectional factors — and only what survives
out-of-sample validation, cost stress, adversarial attack, and multiple-testing
deflation is ever shown to you.

    HUMAN PROVIDES:  a data pointer + a toolkit of measurements + the judge
    MACHINE GROWS:   everything between raw bars and a deployable strategy
    NEVER PROVIDED:  named strategies, preset thresholds, chart patterns —
                     no pre-written OPINION about what works

## 1.1 The governing distinction: measurement vs instruction

This is the single rule that decides what may enter the system.

A **measurement** describes what the market did or is doing. `rsi(14)` is a
measurement: a number at every bar. `hurst(200)` is a measurement: it says
whether the series has been trending or reverting. `funding_rate` is a
measurement.

An **instruction** tells the machine what to do about it. "Buy when RSI < 30"
is an instruction. "Double bottoms are bullish" is an instruction. "Fade the
extreme" is an instruction wearing an indicator costume.

**Measurements go in the toolkit. Instructions are what the machine is
supposed to discover.** When evaluating any proposed addition, ask: does it
measure something, or does it tell you to do something? If the latter, it is a
prior and it stays out.

This resolves the apparent contradiction in "prior-free but with RSI." RSI is
not a prior — it is a transformation of price. "RSI oversold means buy" is the
prior. Evolution is allowed to discover that RSI works best as a crossover
input, a volatility filter, a ranking score, or not at all. It is never told.

## 1.2 What "prior-free" does NOT mean

Three priors remain, deliberately, and should be understood as load-bearing:

1. **Mathematical priors.** The operator set (add, rolling mean, compare) is a
   human choice. It constrains what is expressible. This is unavoidable and
   fine — arithmetic is not an opinion about markets.
2. **The judge.** Next-open fills, cost models, k-fold validation, deflation
   thresholds. These encode a strong opinion: *what counts as evidence*. That
   opinion is the most valuable thing in the system.
3. **Scope.** Bar-based signals with stop-based exits. The system cannot
   discover, say, a market-making strategy or a latency arbitrage, because
   its harness cannot express them. Know this limit; do not mistake "found
   nothing" for "nothing exists" outside the expressible space.

---

# 2. Inheritance: what dies, what lives, what's born

## 2.1 Deleted — domain priors

| Source | Why it dies |
|---|---|
| `edgesearch/hypotheses.py` | Seven named strategies (momentum, MA cross, RSI mean-rev, Donchian, Bollinger reversion, vol contraction, gap fade). Each is a pre-written human opinion. |
| `evolve.py` CONDITIONS' **thresholds** | The `< 30`, `> 70`, band-touch comparisons are opinions. The *indicator computations underneath survive* as Tier-1 atoms. |
| `pattern_scanner` `named_patterns()` | W/M, head & shoulders, triangles — chartist priors, and mostly statistically hollow ones. |
| `pattern_scanner --template` mode | Hand-drawn shapes are the purest form of "I already know what to look for." |

`discover.py` and `evolve.py` may remain in the repo as a legacy tier for
sanity comparison, but are not part of this system's search path.

## 2.2 Kept as-is — the judge and the substrate

| Module | Role |
|---|---|
| `backtest.py` | Next-open fills, ATR risk unit, pluggable exits, gap-through fill realism. |
| `xsbt.py` | Cross-sectional rank harness — dollar-neutral top/bottom portfolios. |
| `costs.py` | Per-asset-class cost tiers + a stress multiplier. |
| `fitness.py` | K-fold (name × time) robust fitness. |
| `hof.py` | Decorrelated survivor set. |
| `ledger.py` | Cumulative trial count → deflated luck bar. |
| `metrics.py` | Sharpe, PSR, drawdown. |
| `data.py` | DataSource zoo (Alpaca, Coinbase, CSV, omnifeed). |

## 2.3 Kept as the core

`synth.py` — typed-tree GP over raw atoms. **This system is essentially synth
promoted from deepest tier to only tier, given a vastly richer atom supply, a
full strategy genome instead of just a tree, and a harder validation gauntlet.**

## 2.4 Born — new modules

| Module | Purpose |
|---|---|
| `universe.py` | The "pointer": a manifest fully specifying a run. |
| `normalize.py` | Scale-free exposure of every atom. |
| `atoms.py` | The Tier-1 measurement catalog + correlation dedupe. |
| `motifs.py` | pattern_scanner's discovery refactored into an atom generator. |
| `genome.py` | Full strategy specification and its mutation operators. |
| `pipeline.py` | One object owning the honest sequence end to end. |
| `runtime.py` | Load a survivor and emit signals live, framework-free. |
| `archive.py` | Persistent cross-run memory of everything ever evolved. |
| `adversary.py` | Population that hunts survivors' failure modes. |
| `stress.py` | Synthetic-data resampling gauntlet. |
| `explain.py` | Ablation, perturbation, and NL description of survivors. |
| `forward.py` | Paper-forward queue and backtest-vs-forward tracking. |

---

# 3. The universe manifest — "the pointer"

A run is fully specified by a manifest. This is the only thing you write by
hand to start a search.

```yaml
name: coinbase_liquid_top60
asset_class: crypto
resolver: coinbase_top_n_by_volume     # or an explicit symbol list
n_symbols: 60
base_timeframe: 5m                     # finest fetch; all others resampled up
allowed_timeframes: [5m, 15m, 30m, 1h, 4h]
start: 2021-01-01
end:   2026-06-01
cost_tier: crypto_taker
holdout:
  name_frac: 0.4                       # fraction of symbols locked away
  time_frac: 0.3                       # fraction of the tail locked away
  embargo_bars: 50
external_streams: [funding_rate, basis_spot, open_interest]
```

**Why a manifest matters beyond convenience:** its hash keys the trial ledger.
Trials burned searching a crypto universe must not deflate an equities holdout
they never touched. Without manifest keying, a single global trial count makes
every search progressively and wrongly harder for unrelated data.

**Resolvers** turn intent into symbols: `coinbase_top_n_by_volume`,
`alpaca_liquid_equities`, `csv_directory`. A resolver must be **deterministic
given a date** — resolving "top 60 by volume" using *today's* volume when
backtesting 2021 is survivorship bias, and a nasty one. Resolve the universe as
of the run's start date, or better, re-resolve at each rebalance and accept the
churn.

---

# 4. The toolkit — three tiers

## 4.1 Tier 0 — pure math operators

The structural glue. No market content whatsoever.

```
arithmetic    add  sub  mul  div  abs  neg  min  max
temporal      lag(n)  diff(n)  roll_mean(n)  roll_std(n)  roll_min(n)
              roll_max(n)  roll_rank(n)  roll_sum(n)  ema(n)
comparison    gt  lt  gte  lte  cross_above  cross_below
logic         and  or  not  xor
conditional   if_then_else(cond, a, b)
scaling       zscore(n)  pctile_rank(n)  clip(lo, hi)
```

`div` must be protected (return 0 or NaN-safe on zero denominators) or the GP
will find division-by-near-zero as a free volatility spike detector — a classic
GP pathology that produces beautiful, meaningless backtests.

## 4.2 Tier 1 — measurement atoms

Every atom is a **series** (a value at every bar), exposed **scale-free**, with
periods drawn from a coarse fast/medium/slow menu. Series, not events: a series
can be compared, ranked, differenced, or fed into another operator; an event
("RSI is oversold") is a dead end that can only be ANDed.

Scale-free matters because a tree must mean the same thing on BTC at $60,000
and a $4 small-cap. Anything in raw units (OBV in share counts, MACD in
dollars) is z-scored or percentile-ranked before entering the pool.

### Price / trend
| Atom | What it measures |
|---|---|
| `dist_sma(n)` | Distance from n-bar mean, in % or ATR units |
| `ema_slope(n)` | Rate of change of the EMA — trend direction and strength |
| `roc(n)` | Simple n-bar rate of change |
| `macd_line(f,s)`, `macd_hist(f,s)` | Fast/slow EMA divergence, normalized |
| `adx(n)` | Trend strength irrespective of direction |
| `aroon(n)` | Bars since the highest high / lowest low — trend freshness |
| `linreg_slope(n)` | Least-squares slope over n bars, in sigma units |
| `price_pctile(n)` | Where price sits in its own trailing distribution |
| `vwap_dist` | Distance from VWAP — the institutional reference price |

### Oscillators / stretch
| Atom | What it measures |
|---|---|
| `rsi(n)` | Ratio of average gains to average losses |
| `stoch_k(n)`, `stoch_d(n)` | Position in the trailing high-low range |
| `cci(n)` | Deviation from typical price, mean-deviation scaled |
| `williams_r(n)` | Inverse stochastic |
| `bb_z(n)` | Distance from rolling mean in rolling-sigma units |
| `keltner_z(n)` | Same, but ATR-scaled — behaves differently in gaps |

### Volatility structure — *highest-value block*
Most real edges are conditional on volatility state; nothing in the oscillator
block can express that.

| Atom | What it measures |
|---|---|
| `rvol(n)` | Realized volatility, close-to-close |
| `vol_ratio(fast,slow)` | Short vol ÷ long vol — expansion vs contraction |
| `parkinson(n)`, `garman_klass(n)` | Range-based vol; ~5x more statistically efficient than close-to-close |
| `vol_of_vol(n)` | Volatility of volatility — regime instability |
| `vol_pctile(n)` | Current vol vs its own trailing distribution |
| `atr(n)`, `atr_pct(n)` | Average true range, absolute and as % of price |
| `bb_width(n)` | Band width — squeeze and release |

### Distribution shape
Describes *how* an asset has been moving, which no oscillator captures.

| Atom | What it measures |
|---|---|
| `skew(n)`, `kurt(n)` | Asymmetry and tail-heaviness of recent returns |
| `up_bar_frac(n)` | Fraction of recent bars that closed up |
| `up_down_ratio(n)` | Mean up-move size ÷ mean down-move size |
| `max_dd_window(n)` | Worst drawdown inside the trailing window |
| `ret_pctile(n)` | Today's return vs its own history |

### Persistence / memory — *the honest form of "mean reversion"*
These measure regime rather than instructing a fade.

| Atom | What it measures |
|---|---|
| `autocorr(n, lag)` | Serial correlation of returns — reverting vs trending |
| `hurst(n)` | Long-memory exponent; <0.5 reverting, >0.5 trending |
| `variance_ratio(short,long)` | Random-walk deviation |
| `efficiency_ratio(n)` | Net move ÷ total path travelled — trend purity |

### Bar microstructure
Every bar carries more information than its close.

| Atom | What it measures |
|---|---|
| `close_loc` | Where close sits in the bar range (0–1) |
| `body_frac`, `upper_wick_frac`, `lower_wick_frac` | Bar anatomy — rejection and absorption |
| `gap_atr` | Gap size in ATR units |
| `overnight_ret`, `intraday_ret` | Two genuinely different return streams that close-to-close collapses together |
| `true_range_z(n)` | Today's range vs typical |

### Volume / liquidity
On crypto especially, liquidity state is often the whole story.

| Atom | What it measures |
|---|---|
| `obv`, `obv_slope(n)` | Cumulative signed volume and its trend |
| `dollar_volume`, `volz(n)` | Activity level, normalized |
| `turnover_ratio(n)` | Volume vs its trailing average |
| `amihud(n)` | Return per unit of volume — illiquidity |
| `vol_price_corr(n)` | Do moves come with volume? |
| `mfi(n)`, `cmf(n)` | Volume-weighted flow measures |
| `vwap_dist_z(n)` | Normalized VWAP deviation |

### Cross-sectional / relative — *edges often live in the relationship*
| Atom | What it measures |
|---|---|
| `xs_rank(<any atom>)` | Rank within the universe on any atom above |
| `rel_ret(n)` | Return vs universe median |
| `beta_uni(n)`, `corr_uni(n)` | Sensitivity/correlation to the universe (or BTC) |
| `xs_dispersion(n)` | Spread of the cross-section itself — a regime measure |
| `rel_vol(n)` | Own vol vs universe vol |

**Leakage note:** cross-sectional atoms are computed over the full universe,
including holdout names. That is not lookahead in time, but it does mean train
features embed contemporaneous holdout-name information. Either compute
cross-sectional atoms within the train-name set only, or accept and document
the impurity. Recommendation: compute within-split.

### Calendar
`hod`, `dow`, `dom`, `bars_since_session_open`, `dist_to_weekend`.
Exposed as plain numerics so the GP can build its own conditioning
(`gt(hod, const(14))`) rather than being handed session rules.

### Crypto-specific — *prioritize these*
Real structural asymmetry, not another transformation of price.

`funding_rate`, `funding_pctile(n)`, `basis_spot`, `basis_z(n)`,
`oi_change(n)`, `long_short_ratio`, `liquidations_z(n)`.

### Equity-specific
`iv_atm`, `iv_skew`, `iv_rank`, `short_interest`, `borrow_cost`, `etf_flow`,
`days_to_earnings`.

## 4.3 Tier 2 — data-derived atoms

**Mined motifs** (see §6) and **external streams** via `add_external`.

## 4.4 Redundancy control

Many atoms are near-duplicates: `roc(20)` ≈ `dist_sma(20)` ≈ `ret20`. Without
handling, evolution burns generations rediscovering one tree in three costumes,
and the HOF's decorrelation gate is fooled by cosmetic differences.

**Procedure:** at startup, compute every atom on train data, build the
correlation matrix, hierarchically cluster at |ρ| > 0.95, keep one
representative per cluster (prefer the cheapest to compute and the most
interpretable), and **log the merges** so you can see what the system considers
equivalent.

---

# 5. What a candidate IS — the genome

A candidate is not a formula. It is a full strategy specification, and every
part of it evolves. Anywhere a human default currently sits, it becomes a gene.

```python
@dataclass
class Genome:
    timeframe:       str          # one of manifest.allowed_timeframes
    root_type:       str          # "bool" -> timing | "series" -> xs score
    entry_tree:      dict         # the structure
    regime_tree:     dict | None  # gate: when may entry_tree trade?
    direction:       str          # "long" | "short" | "both"
    entry_style:     dict         # market_next_open | limit(k*ATR) | confirm(n)
    exit_params:     dict         # stop_atr, target_r, time_stop, trail_mode
    max_hold_bars:   int
    universe_filter: dict | None  # liquidity/vol slice of the pool (P7)
```

## 5.1 The regime tree — the highest-value gene

A second boolean tree that decides **when the entry tree is allowed to trade at
all**. Most edges are conditional; without a separate gate, a single tree must
contort itself to express "only when volatility is in the bottom third," and
usually fails to.

Mechanically: `final_signal = entry_tree(df) AND regime_tree(df)`. Evolved
jointly, but with a bias toward simple regime trees (a 12-node regime gate is
usually overfitting a few good months).

## 5.2 Timeframe as a gene

Edges live at specific horizons; forcing everything to 1h is an arbitrary human
choice. Implementation: fetch once at the manifest's `base_timeframe` (5m),
resample up to each allowed timeframe in memory, and cache to parquet.

Three hazards, each with a mitigation:

1. **Trial multiplication.** Five timeframes ≈ 5× the trials. The ledger absorbs
   this automatically; expect *fewer* survivors, not more.
2. **Thin samples at the slow end.** 4h bars over three years is ~6,500 bars —
   thin for k-fold. Enforce a minimum bars-per-fold or drop the timeframe.
3. **Drift toward the fastest timeframe.** If costs are even slightly
   optimistic, evolution will *always* converge on 5m, because that is where
   fake edge is densest. Mitigations: honest per-class cost tiers, an explicit
   turnover penalty in fitness, and cost-stress validation (§8.3).

Mutation may shift timeframe only to an **adjacent** one — a structure that
half-works at 30m often works at 1h, whereas 5m→4h is effectively a new random
candidate.

## 5.3 Entry style and exits

**Entry style:** market at next open; limit at `k × ATR` offset (better fills,
some signals never fill — the harness must model non-fills honestly);
wait-for-confirmation (signal must persist n bars).

**Exit params:** stop distance in ATR multiples, target in R multiples,
time-stop in bars, trailing behavior (none / breakeven-at-1R / trail-1R /
scale-out). **How you exit is often more of the edge than how you enter**, and
it is currently the largest block of unexamined human defaults in `edgesearch`.

---

# 6. Motif mining — pattern_scanner as an atom generator

`pattern_scanner`'s `discover` mode currently finds recurring shapes and draws
them on a PDF. Here it becomes a **parts supplier**.

**Procedure:**
1. Slice the **train data only** into overlapping windows at multiple scales
   (e.g. 20, 40, 80 bars).
2. Z-normalize each window (scale- and level-invariant).
3. Cluster (KMeans or similar) into k motifs.
4. Each motif becomes a terminal `match(motif_i, scale_s)` — a **similarity
   series**: at every bar, the correlation/distance between the trailing window
   and that motif.

Now the GP can build `and(gt(match(motif_3, 40), 0.8), lt(vol_pctile(100),
0.3))` — "this shape, in a low-vol regime" — a structure nobody named.

**Discipline:**
- Motifs are mined on train slices only. The holdout has never been clustered.
  Verify with an explicit leak test.
- Motif definitions are **frozen** into any survivor that uses them (same class
  of bug as the library-key reuse found in the `edgesearch` audit — a motif
  index that means something different at validation time silently invalidates
  the result).
- Motif count is a search parameter and multiplies trials; log it.

---

# 7. The evolution loop

```
1  resolve manifest -> symbol list, date range, cost tier
2  fetch base timeframe once; resample to all allowed; cache
3  SPLIT: hold out name_frac of symbols AND time_frac of the tail (+ embargo)
4  mine motifs on TRAIN slice only -> Tier-2 atoms
5  compute atom catalog on train; dedupe at |rho| > 0.95
6  seed population (random, or from archive elites in P7)
7  for each generation:
     a  evaluate every genome (see §8)
     b  update hall of fame with decorrelation gate
     c  update novelty archive
     d  select (NSGA-II frontier in P5, tournament before that)
     e  crossover + mutate (trees AND non-tree genes)
     f  migrate between islands every k generations
8  every epoch: mine elite subtrees -> promote to library vocabulary (INLINED
   into any genome that keeps them)
9  gauntlet the HOF survivors (§9)
10 write run artifacts; log trials to ledger; enqueue survivors to paper-forward
```

## 7.1 Two strategy shapes

- **Bool root** → a timing signal → `signal_backtest` (per-name).
- **Series root** → a ranking score → `rank_backtest` (cross-sectional
  dollar-neutral top/bottom slice). This is where documented equity anomalies
  actually live and per-symbol timing cannot express them.

Both harnesses already exist.

---

# 8. The judge

The judge is the most valuable component in the system. Removing priors
explodes the hypothesis space, which makes a prior-free searcher **the easiest
machine in the world to fool yourself with** — it will always find *something*
that fits history.

## 8.1 Execution realism
- Signal computed at `close[t]`, filled at `open[t+1]`. Never same-bar.
- Stops that gap through fill at the open, not the stop price.
- ATR-based risk unit; fixed fractional risk per trade.
- Limit entries model non-fills.

## 8.2 Fitness
Scored over **(name × time-fold) cells**, not names. Each name's train window is
cut into k contiguous folds with an embargo at each boundary; fitness is:

```
median(cells) − λ·std(cells) − γ·complexity − δ·turnover_penalty
```

A candidate must work across names **and** across sub-periods. One hot quarter
fails. Sample adequacy is checked (PSR), not assumed.

## 8.3 Cost stress
Every survivor is re-run at 1×, 2×, and 3× the cost tier. An edge that dies at
2× costs is a spread-capture illusion, not an edge.

## 8.4 Multiple-testing deflation — the load-bearing wall
The ledger records **every candidate ever evaluated** against a given manifest's
holdout, across all sessions — including timeframe variants, island
populations, ensemble combinations, archive-seeded reruns, and every HOF
holdout peek. The passing bar is the expected maximum Sharpe under the null
given that cumulative count (with a correlation haircut, since GP populations
are not independent trials).

**Everything in this document multiplies trials. The bar rising is the system
working, not a knob to tune away when nothing survives.**

## 8.5 The final gate
Survivors are validated **once** on held-out names during the held-out time
period. That is the only honest number in the system.

---

# 9. Adversarial and causal validation

Passive holdout is the weakest form of proof. Layer on:

**Adversarial population** (`adversary.py`). A second population whose fitness
is *breaking* survivors: find the cost assumption, regime, subperiod, or
universe slice where the edge dies. Surviving an attacker specifically hunting
your failure mode is far stronger evidence than passing a passive holdout.

**Synthetic stress** (`stress.py`). Re-test survivors on bootstrapped,
block-shuffled, and regime-permuted histories. A structure that only works on
the exact historical path is curve-fit; one that survives resampled paths found
something structural. *Cheap, brutal, high-yield — build this early.*

**Ablation** (`explain.py`). Remove each atom/subtree and measure the damage. No
damage means decoration. Turns opaque winners into understood ones.

**Perturbation.** Jitter inputs and observe what the strategy actually responds
to — a causal probe rather than a correlational story.

**Natural-language explanation.** An LLM reads a survivor's tree and describes
what it does economically ("buys illiquid names after volume dries up and vol
compresses"). Two payoffs: you can sanity-check real-vs-artifact, and you can
seed populations with *directions* ("explore microstructure mean reversion") —
the closest thing to a creative collaborator.

> **Explanations are commentary, never evidence.** Nothing passes because the
> story sounded good. A convincing economic narrative for a curve-fit artifact
> is the easiest thing in the world to generate, and treating it as
> confirmation would undo every other safeguard here.

---

# 10. Meta-evolution — improving the search itself

**Islands.** N populations evolving in parallel with occasional migration.
Single-population GP reliably collapses into one basin; islands don't.
Additionally, islands may run **different configurations** — one heavy on
volatility atoms, one on microstructure, one pure math — and productive
configurations get replicated. The machine discovers which corner of the
toolkit is fertile for a given asset class instead of you guessing.

**Adaptive mutation.** Mutation rate rises when best-fitness plateaus, falls
when it's climbing.

**Novelty search.** Reward candidates for *behaving differently* from everything
in an archive, independent of fitness. Behavior = the signal vector; novelty =
distance to k nearest neighbors. This explores unmapped regions instead of
polishing one local optimum, and is one of the few techniques that reliably
finds genuinely unusual structures.

**Multi-objective selection (NSGA-II).** Optimize Sharpe, drawdown, turnover,
and stability simultaneously rather than mashing them into one scalar. Output
is a **Pareto frontier** of survivors — you pick the trade-off, rather than
inheriting whatever weighting was hardcoded.

**Meta-genes.** Population size, tree depth cap, crossover style, and which atom
blocks are enabled all become evolvable.

---

# 11. Self-discovered building blocks

| Mechanism | What it does |
|---|---|
| Library mining *(exists)* | Best subtrees promoted to named vocabulary each epoch. Must be **inlined** into any genome that outlives the epoch. |
| Motif mining *(§6)* | Recurring shapes → similarity atoms. |
| Regime clustering | Unsupervised clustering of market states nobody labeled; cluster membership becomes a conditioning atom. |
| Learned features | Autoencoder-compressed recent-bar representations as atoms. Opaque — keep behind a flag, and never let an opaque atom be a survivor's *sole* dependency. An edge you cannot reason about is one you cannot debug when it stops working. |
| Residual search | Symbolic regression on the *residuals* of an existing survivor: what is it missing? |

---

# 12. Ensembles as first-class objects

The HOF yields decorrelated survivors. The **combination** is itself a
searchable object: evolve which survivors run, their weights, and how they vote
(unanimous / majority / weighted-sum threshold).

Uncorrelated mediocre edges beat one good edge at the portfolio level. The
ensemble is judged by the same referee on the same holdout, and its trials count
in the ledger like everything else.

---

# 13. Persistent memory across runs

**`archive.py`** stores every structure ever evolved, with its fitness, the
universe and timeframe it was found in, and the regime of discovery.

Enables: seeding new runs from historical elites instead of noise; "I have tried
this before" detection; and asking *why* something failed. This is the biggest
single upgrade toward the system feeling alive rather than restarting from zero
every session.

**Discipline:** the archive shares the ledger's manifest keying. Seeding from
archived elites still counts as trials — otherwise the archive becomes a
laundering mechanism, reusing structures that already learned from a holdout
without paying for them.

---

# 14. Universe as a gene

Instead of a fixed symbol list, give it a **pool**; which assets, how many, and
liquidity/volatility filters become genes. Edges are often universe-specific —
"this works on the 30 most illiquid names" is a discovery you would never make
by hand. Requires deterministic, point-in-time resolution to avoid survivorship
bias (§3).

---

# 15. Paper-forward queue — the only metric that finally matters

Every survivor is auto-deployed to live paper trading the moment it passes, and
its forward performance is recorded **against its backtest prediction**.

After a year this yields the one number no backtest can produce: how often this
search's survivors survive contact with the future. It feeds back as a prior on
how much to trust the whole pipeline — and if forward results systematically
undershoot backtests, the size of the gap tells you exactly how much the judge
is still being fooled.

---

# 16. Data streams — the biggest real upside

Seventy atoms over OHLCV is one information source refracted seventy ways.
**Genuinely new information is worth more than any transformation**, because an
edge *is* information the price hasn't fully absorbed.

| Category | Streams |
|---|---|
| Microstructure | Order-book depth and imbalance, trade prints (aggressor ratio, trade-size distribution), cross-venue divergence |
| Crypto native | Funding, basis, open interest, liquidations, long/short ratio; on-chain: exchange netflow, active addresses, stablecoin supply |
| Equity native | Options-implied vol and skew, short interest, borrow cost, ETF flows, earnings/event calendars |
| Macro / event | GDELT-style event counts and surprise measures |

All arrive through `features.add_external()`.

> **HARD RULE: every external series must be timestamped at PUBLICATION time,
> not event time.** On-chain data and fundamentals are notorious for this and
> will manufacture spectacular fake edges. The loader owns this burden; lag
> anything whose publication lags its event.

---

# 17. Permanently out of scope

Leverage, position sizing beyond the fixed risk unit, and capital allocation are
**not evolvable**. They are portfolio decisions that come after an edge is
confirmed. Letting the search touch them produces "edges" that are leverage
applied to noise — the fastest way to generate an impressive equity curve with
no information content.

---

# 18. Build phases

| Phase | Deliverable |
|---|---|
| **P0** | `universe.py` manifest + ledger keyed by manifest hash; `normalize.py`; `atoms.py` Tier-1 catalog + ρ>0.95 dedupe |
| **P1** | `motifs.py` (train-only mining + leak test); multi-timeframe resample and cache; timeframe / direction / exit / entry-style genes |
| **P2** | `pipeline.py` single entry point (split → mine → evolve → HOF → gate → ledger → persisted `runs/<ts>/`); regime tree; `tests/` with positive and negative controls |
| **P3** | `runtime.py` export + paper-deploy smoke check |
| **P4** | Data streams via `add_external` — crypto native first (funding, basis, OI) |
| **P5** | Meta-evolution: islands, adaptive mutation, novelty archive, NSGA-II |
| **P6** | Self-discovered blocks: regime clustering, residual search, learned features (flagged); ensemble evolution |
| **P7** | Sandbox layer: meta-genes, persistent archive, universe-as-gene |
| **P8** | Adversarial population, synthetic stress, ablation/perturbation, NL explanation |
| **P9** | Paper-forward queue + backtest-vs-forward tracking |

**Order rationale.** P0–P3 make the thing real and deployable. P4 is where the
actual money probably is — new information beats new transformations. P5–P7 are
search-quality multipliers that only pay once the pipeline is honest. P8–P9 are
what keep it honest *as the freedom grows*: the more creative the sandbox, the
more the validation layer carries.

**Exception to the ordering:** synthetic stress (P8) is cheap and high-yield.
Pull it forward into P2 if anything ever survives before then.

---

# 19. Failure modes to watch for

| Failure | Symptom | Guard |
|---|---|---|
| Drift to fastest timeframe | All survivors are 5m | Honest costs, turnover penalty, cost stress |
| Division pathology | Survivors depend on `div` by near-zero | Protected division |
| Motif/library index reuse | Validation silently differs from training | Inline and freeze at capture |
| Archive laundering | Bar stops rising across sessions | Archive seeding counts as trials |
| Survivorship bias | Universe resolved with today's data | Point-in-time resolution |
| Publication-lag leakage | Spectacular edges from alt data | Loader timestamps at publication |
| Cosmetic HOF diversity | Five "different" survivors trade identically | Atom dedupe + signal-correlation gate |
| Narrative capture | Passing something because the story was good | Explanations are commentary, never evidence |

---

# 20. Expectation setting

On liquid US large-caps this system should — and probably will — keep saying
**"no edge."** Now with more statistical authority than before. That is the
correct answer for that data, and a framework that produces it is worth more
than one that hands you a beautiful curve.

Its real chance of paying is where structural asymmetry still exists: broad
crypto cross-sections, funding/basis atoms, venue microstructure, event data.
**Point the pointer there first.**

The value proposition is not "this will find you an edge." It is: *if* an edge
in the expressible space exists in this data, this finds it; and if one doesn't,
this tells you so instead of letting you spend a year believing otherwise.
