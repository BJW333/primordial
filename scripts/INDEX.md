# scripts/ index

`scripts/` = what's still in use. `scripts/archive/` = closed experiments, still runnable from the
repo root (e.g. `python3.10 scripts/archive/gapstep.py`). Their logs are in `logs/archive/`.

## Passed research

| script | what it does |
|---|---|
| `flows_screen.py` | futures/commodity flow screen -- source of TSY_CAL, the one PASS |

## CHIMERA core (momentum, regime, exposure judge)

| script | what it does |
|---|---|
| `momentum_v2.py` | momentum on ALL liquid US stocks, fixed three ways after |
| `regime_sizer.py` | REGIME SIZER alone (TEST A) and with the DD2 dimmer on |
| `dd2_lab.py` | DD2: the drawdown dimmer, rebuilt. Pass test: judge.py. |
| `judge.py` | the shared, pre-stated pass test for exposure rules on CHIMERA |
| `crash_warning_search.py` | can an EARLY WARNING spot momentum crashes before |
| `blowup_scan.py` | three lesser-known "before it blows up" signals on the |
| `blowup_us.py` | the blow-up scan (blowup_scan.py, same signals, same |
| `turbo_portfolio.py` | momentum, turbocharged two ways, on the same |
| `factor_portfolio.py` | monthly stock-picking portfolio on the S&P 500, |

## Primordial tools (genomes, gauntlet, survivors)

| script | what it does |
|---|---|
| `make_genome.py` | no JSON by hand. |
| `test_genome.py` | Score a HAND-WRITTEN rule with primordial's judge. |
| `validate_genome.py` | Run ANY hand-built genome through the FULL gauntlet on a real manifest -- |
| `run_controls.py` | Full-pipeline controls. Run BOTH after any change to the judge, backtest, or |
| `list_survivors.py` | Scan every runs/*/run.json and build the survivor registry the project never |
| `compare_series.py` | are two rotation strategies the same edge twice? |
| `last5.py` | re-score every saved portfolio on the LAST FIVE YEARS only |

## Universe builders

| script | what it does |
|---|---|
| `build_universe.py` | Build a symbols file for an index's CURRENT constituents (S&P 600 / 400 / 500) |
| `build_us_universe.py` | every US-listed common stock Alpaca has ever |
| `build_foreign.py` | Build foreign mid-cap universes for the deep-state geography test. |
| `build_fresh_universes.py` | writes symbol lists for markets the dip work has |
| `rank_universe.py` | Rank a symbols file by TRADABILITY and keep the top N. |

## Data fetchers

| script | what it does |
|---|---|
| `fetch_13f.py` | hedge-fund "best ideas" from SEC Form 13F holdings, 2015+. |
| `fetch_attention_all.py` | extend Wikipedia page views from the S&P 500 to |
| `fetch_attention_short.py` | two non-price datasets for the S&P 500 stocks |
| `fetch_commod_rth.py` | builds REGULAR-HOURS 1h bars for the commodities_1h |
| `fetch_data.py` | Prefetch (warm the cache for) every symbol a manifest needs, BEFORE a search. |
| `fetch_diversifiers.py` | download daily prices (dividend- and split-adjusted) |
| `fetch_earnings.py` | every US company's quarterly net income, stamped with |
| `fetch_insider_buys.py` | build a clean list of insider OPEN-MARKET PURCHASES |
| `fetch_insider_sales.py` | insider OPEN-MARKET SALES (Form 4, code S, |
| `fetch_intraday.py` | Fetch Alpaca minute bars for a small pilot universe and cache to parquet. |
| `fetch_rebound_data.py` | data for rebound2.py (both free, resumable): |
| `fetch_sec_fundamentals.py` | annual quality metrics for every SEC filer, |
| `fetch_shares.py` | every company's shares outstanding as reported on the |
| `fetch_spinoffs.py` | list US spinoffs since 2006 from SEC EDGAR, free. |

# archive/

## Night-only momentum / GAPSTEP and structural screens (2026-11-15, shelved -- see docs/NIGHT_GAPSTEP.md)

| script | what it does |
|---|---|
| `archive/gapstep.py` | GAPSTEP: a stand-alone momentum strategy (its own picks, its own account; nothing to do |
| `archive/mine_seismic.py` | Blake's idea (2026-11-15): use seismograph data around coal and gravel mines to |
| `archive/night_gap.py` | "trade less often": skip the DAY only when it is most likely to lose. |
| `archive/night_screen.py` | the NIGHT idea on cheap, liquid funds: a STOCK version and a COMMODITY version. |
| `archive/structural_screen.py` | four STRUCTURAL ideas from the 2026-11-15 list that can be tested on data |

## FX (closed 2026-11-15, 0 of 14 pass)

| script | what it does |
|---|---|
| `archive/fx_active.py` | ACTIVE FX (5+ trades a week). PRE-REGISTERED 2026-11-15, before any real-data run. |
| `archive/fx_calendar.py` | the last FX screen: three CALENDAR-FLOW ideas (the TSY_CAL family). |
| `archive/fx_grid.py` | the TikTok grid bot, rebuilt on the AUD/NZD gap instead of gold. |
| `archive/fx_rotation.py` | Blake's "trade currency to currency, harvest small gains, almost every country". |
| `archive/fx_screen.py` | FX ideas on hourly data. PRE-REGISTERED 2026-11-10, before any real-data run. |
| `archive/fxmh_screen.py` | multi-horizon sign vote (mh_vote) on G10 FX, daily. |

## Futures / commodities / patterns (closed)

| script | what it does |
|---|---|
| `archive/commodities_test.py` | five commodity strategies across 20 futures markets, |
| `archive/commodity_adx.py` | ADX trend-strength and volume signals on commodity |
| `archive/commodity_attention.py` | does the FADE carry over to futures-type markets? |
| `archive/futures_mr_orb.py` | Primordial screen: futures MEAN REVERSION (daily spreads) |
| `archive/gold_choch.py` | Primordial screen of the "premium zone + CHoCH" setup on GOLD, |
| `archive/nq_mr_ga.py` | a genetic algorithm that EVOLVES the entry and exit rules of an |
| `archive/season_check.py` | is commodity SEASON (commodities_test.py, commit |
| `archive/shape_edge.py` | do chart SHAPES (alone or combined with an indicator) |
| `archive/shape_scan.py` | counts the most common chart SHAPES in the commodity ETFs. |
| `archive/spy_noise_band.py` | intraday MOMENTUM on the S&P 500 (the "noise band"), |

## Attention fade / rebound (closed)

| script | what it does |
|---|---|
| `archive/attention2_test.py` | three follow-ups to attention_short_test.py (where |
| `archive/attention_bottom.py` | Blake's bottom-catcher: after a good-news attention |
| `archive/attention_fade_book.py` | turn the confirmed attention FADE into a real |
| `archive/attention_rebound.py` | after a good-news attention spike, small/mid caps |
| `archive/attention_short_test.py` | two NON-PRICE signals on S&P 500 stocks |
| `archive/export_fade_spikes.py` | write the fade's hype-spike events to a small CSV |
| `archive/export_fade_trades.py` | the EXACT trades Primordial's validated fade book |
| `archive/fade_book2.py` | REALITY CHECK on the fade_predict.py PASS. The rule is |
| `archive/fade_predict.py` | after a good-news attention spike, can we tell WHICH |
| `archive/fade_stop.py` | fade_book2.py showed the liquid-MID WEAK_TREND short wins |
| `archive/fade_stop_check.py` | PROVE OR BREAK the fade's 50% stop-loss, which so far |
| `archive/portfolio_combo.py` | SIZING STUDY (no pass/fail, nothing new is being |
| `archive/rebound2.py` | second round of "will this faded stock come back?" signals, |
| `archive/rebound_signals.py` | the INVERSE of the fade. After a good-news attention |
| `archive/sector_tailwind.py` | rebound idea: faded stocks come back when their WHOLE |

## Momentum add-ons, crash warnings, drawdown rules (closed)

| script | what it does |
|---|---|
| `archive/buyback_lab.py` | creative BUY-BACK switches for a momentum stop, each built |
| `archive/combo_search.py` | every combination of the momentum gap rule and Blake's |
| `archive/crash_warning.py` | MOMENTUM CRASH WARNING from the fade work (long-only). |
| `archive/crowd_combo_search.py` | CROWD and WBETA combined with every warning sign |
| `archive/crowding_signal.py` | TWO research-based crash signals for a winners-only |
| `archive/dd_check.py` | two checks on DDCTRL (drawdown control) before trusting it. |
| `archive/exit_rules.py` | fix the sell rule. Same machine as momentum_v2.py (all |
| `archive/macro_combo_search.py` | can an early warning predict momentum's drops if we |
| `archive/momentum_avoid.py` | does the confirmed attention fade make the all-US |
| `archive/momentum_cap.py` | ONE pre-stated test: does capping how many of the top 20 |
| `archive/momentum_gap.py` | Huang (2022, Review of Financial Studies) "The Momentum |
| `archive/momentum_hype.py` | DIAGNOSTIC (post-hoc, no pass/fail). momentum_avoid.py |
| `archive/momentum_more.py` | a second batch of add-ons on steady momentum (all |
| `archive/momentum_overlay.py` | test CHIMERA's on/off rules on Primordial's own daily |
| `archive/momentum_plus.py` | add-ons stacked on steady momentum (momentum_v2 |
| `archive/momentum_rebound.py` | Blake's EXACT stop-and-buy-back idea on top of the |
| `archive/momentum_rebound10.py` | Blake's stop-and-buy-back idea, ADX swapped for a NEW 10-DAY HIGH on top of the |
| `archive/momentum_tranche.py` | fix the HOLDINGS, not the timing. |
| `archive/momentum_variants.py` | change WHICH stocks the book picks instead of timing |
| `archive/profit_guard.py` | Blake's "protect the year" idea, two parts tested separately. |
| `archive/quality_portfolio.py` | the survivorship-free S&P 500 top-20 portfolio |
| `archive/regime_fast.py` | can REGIME react faster than once a month? |
| `archive/residual_momentum.py` | momentum with the market exposure stripped out |
| `archive/stack3.py` | SIZING STUDY (no pass/fail): momentum + two-way futures trend + |
| `archive/trend_ls.py` | TWO-WAY trend following across stocks, bonds, commodities, |
| `archive/trend_sleeve.py` | a LONG-ONLY multi-asset trend-following sleeve to STACK on |
| `archive/turbo_oos.py` | out-of-sample test of the two momentum candidates on |
| `archive/value_momentum.py` | does a VALUE book (cheap stocks) cancel momentum's bad |
| `archive/vol_reduce.py` | LOWER THE VOLATILITY of the stack (momentum + trend + |

## Dip-buy family (closed)

| script | what it does |
|---|---|
| `archive/check_c2_cohorts.py` | the pre-registered gate between "passed both |
| `archive/dip5_screen.py` | quick SCREEN of Blake's idea: an S&P 500 name closes down |
| `archive/dip_combo_classic.py` | SECOND combo search (recorded as such): the same |
| `archive/dip_combo_search.py` | explicit combo search: which COMBINATIONS of up to 3 |
| `archive/dip_count_hazard.py` | test the DIP-COUNT hypothesis across timeframes. |
| `archive/dip_features.py` | can ANY of ~17 price/volume measurements tell a dip |
| `archive/dip_hazard.py` | measure the thing Blake described, directly: |
| `archive/dipband_screen.py` | Blake's dip-buy with a DYNAMIC drop band. |
| `archive/dipbase_screen.py` | is the SIZE of a one-day drop a real indicator, or |
| `archive/dipgap_screen.py` | does HOW a stock dropped decide whether it bounces? |
| `archive/dipz_screen.py` | vol-scaled dip-buy screen (successor to dip5_screen.py). |
| `archive/insider_dip_test.py` | does an INSIDER PURCHASE tell a dip that rebounds |
| `archive/market_trend_dips.py` | ONE indicator, fresh markets: do dips bought while |
| `archive/market_trend_fresh.py` | the out-of-sample test for "buy dips while the |
| `archive/validate_taxloss.py` | the two quick gates owed by the C2 tax-loss cell |

## Stock pickers: earnings, insiders, buybacks, spinoffs (closed)

| script | what it does |
|---|---|
| `archive/earnings_test.py` | earnings surprise: stocks that just reported profits |
| `archive/funds_buybacks.py` | two more non-momentum stock pickers, each tested on |
| `archive/insider_season.py` | two non-momentum stock pickers, each tested on its |
| `archive/pit_distress_sp500.py` | THE one test that |
| `archive/seed_ledger_distress.py` | Seed a manifest's ledger with the DISTRESS family's cross-universe debt. |
| `archive/spinoff_test.py` | the spinoff dip: forced selling of newly spun-off |

## Zero-trial explorers (screens)

| script | what it does |
|---|---|
| `archive/explore_crypto_funding.py` | ZERO-TRIAL screen: crypto FORCED-DELEVERAGING absorption, daily hold. |
| `archive/explore_edgar_secondaries.py` | ZERO-TRIAL event study: SECONDARY OFFERINGS from EDGAR (free, timestamped). |
| `archive/explore_gap.py` | ZERO-TRIAL screen: do overnight gaps revert intraday? TRAIN names x TRAIN |
| `archive/explore_index_deletion.py` | ZERO-TRIAL event study: index DELETION flow. |
| `archive/explore_intraday_dip.py` | ZERO-TRIAL screen: do intraday dips get bought back? Buckets bars by how far |
| `archive/explore_open_gap.py` | ZERO-TRIAL screen on MINUTE bars: does the opening gap reverse intraday, and |
| `archive/explore_premium_conditioning.py` | ZERO-TRIAL conditioning screen: WHEN is the dip-absorption premium fat? |
| `archive/explore_vol.py` | ZERO-TRIAL screen: does volatility CONTRACTION predict expansion + direction? |
| `archive/explore_vwap_mechanisms.py` | ZERO-TRIAL screen, three VWAP-mechanism hypotheses, ONE pass over the |
| `archive/explore_vwap_program.py` | ZERO-TRIAL screen on MINUTE bars: unfinished VWAP programs. |
| `archive/explore_xs.py` | ZERO-TRIAL explorer: does cross-sectional divergence mean-revert on this |

## Crypto (closed)

| script | what it does |
|---|---|
| `archive/crypto_attention.py` | the FADE idea on crypto, long-only. |
| `archive/crypto_sleeve.py` | ADD RETURN to momentum, long-only, with a second engine: |
| `archive/pump_study.py` | crypto pump-and-dump event study: after a coin suddenly |
| `archive/pump_study2.py` | the honest version of pump_study.py's two verdicts. |

## Indicators / breadth (closed)

| script | what it does |
|---|---|
| `archive/adline_obv.py` | Advance-Decline line (breadth) + ADX (trend strength) + |
| `archive/adx_fundamentals.py` | ADX (trend strength, from price) paired with |
| `archive/adx_obv_variants.py` | DIAGNOSTIC (no pass/fail): is the ADX+OBV filter |
| `archive/adx_psar.py` | Wilder's own pairing: ADX (is there a strong trend?) + |

## Deep-state (closed)

| script | what it does |
|---|---|
| `archive/control_deepstate.py` | Equity-side machinery control: a planted DEEP-STATE-shaped edge must survive |
| `archive/kill_test_options_expression.py` | can deep-state's moves pay for calls? |
| `archive/seed_ledger_deepstate.py` | Seed the ledger with the deep-state research session's manual trials. |
| `archive/validate_deepstate_real.py` | hand-built, not evolved -- through the FULL |

## QC algorithm replays

| script | what it does |
|---|---|
| `archive/test_crypto_rotation.py` | Test the CryptoTrifecta rotation as written. |
| `archive/test_etf_rotation.py` | Test the EtfRotationAlgorithm end to end. |
| `archive/test_lev_rotation.py` | Test TheOmniscientParadox: single-asset daily momentum rotation. |
| `archive/test_volume_rotation.py` | Industry rotation on BUYING VOLUME vs on PRICE momentum. |
| `archive/test_weekly_rotation.py` | Weekly (or monthly) cross-sectional rotation on S&P 400 / 500 / 600 names. |

## Other

| script | what it does |
|---|---|
| `archive/coverage_probe.py` | does Alpaca have the dead stocks Yahoo dropped? |
