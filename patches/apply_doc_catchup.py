#!/usr/bin/env python3
"""apply_doc_catchup.py -- bring MECHANISM_MAP.md and .gitignore up to date
with everything run since the mh_vote close (5684bc9). Docs only; no code.

    python3.10 patches/apply_doc_catchup.py   # from repo root
    # then the git commands it prints

Idempotent: each MECHANISM_MAP section is keyed by its heading and skipped
if already present; .gitignore lines are added only if missing.
"""
import os
import sys

MAP = "MECHANISM_MAP.md"
GI = ".gitignore"

SECTIONS = [
("## Breakout family, sp400 ground -- DEAD 2026-08-14 (4 trials)", """
Prereg BREAKOUT_PREREG.md (signed late but pre-result, 41aa9f4; violation
recorded there). sp400, --splits 5, all four 0/5:

| genome | run | 5-split median SR | holdout trades/split |
|---|---|---|---|
| breakout_donchian20 (control) | 20260814_133352 | -0.07 | ~2,650 |
| breakout_donchian55 (control) | 20260814_140304 | -0.14 | ~1,440 |
| breakout_compress | 20260814_143209 | -0.24 | ~1,140 |
| breakout_rvolconf | 20260814_163614 | -0.16 | ~1,050 |

Family CLOSED on sp400 per the decision rule; neither mechanism variant
beat the decayed baseline. rvolconf dead -> the analyst-price-target leg
gated on it is CLOSED with it (no data spend). sp600 half of the prereg
(4 candidates) was never run: recorded here as ABANDONED, not run-and-
hidden; running it now would be a fresh look owing a note in the prereg.
"""),
("## Seconds-scale dip timing (BTC 1s) -- DEAD BY MECHANISM 2026-09-18 (0 trials)", """
scripts/dip_hazard.py, 48h / 989k Coinbase BTC-USD ticks, memoryless and
rising-hazard planted controls both recovered. 1s hazard P(dip ends | it
lasted k) never rises: 0.68 -> ~0.50 (flat=break), ~0.25-0.27 (flat=hold).
No exhaustion clock. "Wait the train-window average dip, then buy":
-0.09 bps gross @5s (break), -0.08 bps (hold); no k with n > 30 clears
1 bps gross vs ~50 bps taker RT. Dead by mechanism, not just costs; the
Kalshi wrapper idea closes with it. Off-repo screen, ledger not charged.
"""),
("## Dip-count exhaustion (crypto minute-scale) -- REAL BUT UNTRADEABLE 2026-09-18 (0 trials)", """
scripts/dip_count_hazard.py (chain = consecutive lower-low dips; planted
controls recovered). Flat at <= 15s. At 1m-15m a weak, consistent
gradient: P(rebound | N) climbs ~4-5pp over N=1..5 on BTC (60s .486 ->
.535) and replicates on ETH/SOL/XRP/DOGE (5/5 assets). Best cells pay
3-16 bps vs ~70 bps taker RT (~15 maker): an order of magnitude short.
Hourly: present but low power. Daily: no read (sample-starved; raw tables
were pure bull-market drift). N=4 kink recurs across views -> covered by
runlen4 (see next entry). Off-repo screens, ledger not charged.
"""),
("## Runlen crypto daily family -- DEAD 2026-09-18 (4 trials) -- PROCESS VIOLATION", """
**Violation, recorded not backfilled:** prereg/RUNLEN_PREREG.md was never
signed and was not committed before the four runs below; genomes and
prereg sat untracked until this entry. Results stand as observed; the
family gets no benefit of the doubt.

coinbase_daily, --splits 5, all 0/5:

| genome | run | 5-split median SR | holdout trades/split |
|---|---|---|---|
| runlen2 (2 down closes) | 20260918_014452 | -0.52 | ~560 |
| runlen3 | 20260918_020535 | -0.29 | ~345 |
| runlen4 | 20260918_022627 | -0.18 | ~190 |
| runlen3_deep (+3-day move bottom 10%) | 20260918_032726 | -0.59 | ~75 |

Dose-response: SR improves with k (2 -> 3 -> 4) but never crosses zero;
runlen3_deep worse than runlen3, so the depth filter adds nothing. The
minute-scale exhaustion gradient does not survive to a tradeable daily
edge. Family CLOSED; the dip-count seam is fully closed with it.
"""),
("## Single-day dip-buying, US + intl equities -- SEAM CLOSED 2026-10-25 (off-ledger screens)", """
Idea: S&P name drops > X% in a day -> buy -> ride the rebound. Ten
screens, all off-ledger event studies with pre-stated rules in the script
headers; outcome measured vs SPY or vs the same-day index average.

| test | commit | result |
|---|---|---|
| dip5: fixed >5%, sp500 survivor vs PIT | (untracked) | survivor H20 t +6.3 -> PIT t -1.5, 2018+ t -2.9. VOID (41% dead-name coverage); direction: survivorship was the edge |
| dipz: drop >= 3x lagged ATR, sp400 kill-only | 165f4b4 | t -0.29, both halves < 0. CLOSED |
| dipgap: gap vs intraday drop split | dea8104 | spread WRONG sign on sp600/400/500. CLOSED. **Prereg unsigned at run (violation, recorded)** |
| dipband: 2%-X bands, close entry | eeec2fc | passes on sp400/600 only, sp500 PIT negative everywhere = survivorship fingerprint |
| dipbase: bands vs same-day index avg | 706e441 | dose-response rises every step on sp400/600, not on sp500 PIT; indistinguishable from asymmetric survivorship. Open only for Norgate PIT |
| dip_features: 17 features + GBM per index | ee79943 | NO SIGNAL (sp500 PIT model t +1.5). sp400/600 model t +6..+8 = log_dollarvol LOOK-AHEAD via current membership (small-then names that grew into the index) |
| market_trend_dips: WEAK (index < 200dma) vs STRONG, 8 intl | be599a3 | passed barely: 6/8, t +2.25 |
| market_trend_fresh: same spec, 6 fresh markets | 3a73f2b | DOES NOT REPLICATE: 3/6, -13 bps t -0.92. be599a3 = luck |
| dip_combo_search: 19,635 combos x H10/20 | dce4200 | none pass; discovery max t ~4-5 = null max |
| dip_combo_classic: +RSI/BB/MACD/stoch/MAs, 64,875 combos (look #2) | 7eeb903 | none pass |

Durable lessons:
- Current-membership lists manufacture dip-rebound edges. Any feature
  that proxies "will still be in the index" (small size, low liquidity,
  deep drawdown that later recovered) wins on survivor lists and becomes
  a severe falling knife on PIT data (big-drop & low-dollarvol combos:
  -350..-490 bps per 20d, t -7 on sp500 PIT 2018+).
- Same-day index baselines cancel uniform survivor lift, not the
  asymmetric part (big droppers delist most).
- Price/volume/classic-TA separators exhausted. Remaining levers: event
  data (earnings dates, news, downgrades), Norgate PIT sp400/sp600,
  deep_state forward paper (the multi-day dip edge already in the book).
"""),
]

GI_LINES = ["# fetched tick / candle data (dip_hazard, dip_count_hazard)",
            "*_1s.csv", "*_candles.csv", "*_trades.csv"]


def main():
    if not os.path.exists(MAP):
        sys.exit("run from the repo root (MECHANISM_MAP.md not found)")
    text = open(MAP).read()
    added = 0
    for head, body in SECTIONS:
        if head in text:
            print(f"  = {head[3:60]}... (already)")
            continue
        text = text.rstrip("\n") + "\n\n" + head + "\n" + body
        added += 1
        print(f"  + {head[3:60]}...")
    if added:
        open(MAP, "w").write(text if text.endswith("\n") else text + "\n")
    gi = open(GI).read() if os.path.exists(GI) else ""
    new = [ln for ln in GI_LINES if ln not in gi.splitlines()]
    if new:
        open(GI, "a").write(("\n" if gi and not gi.endswith("\n") else "")
                            + "\n".join(new) + "\n")
        print(f"  + .gitignore: {', '.join(new[1:]) or new[0]}")
    else:
        print("  = .gitignore (already)")
    print("\nNext (two commits, so the violation history stays readable):")
    print("  git add genomes/runlen*.json prereg/RUNLEN_PREREG.md "
          "patches/apply_runlen_idea.py runs/20260918_*")
    print('  git commit -m "runlen family: artifacts committed AFTER the runs '
          '(prereg never signed) -- violation recorded in MECHANISM_MAP"')
    print("  git add MECHANISM_MAP.md .gitignore scripts/dip5_screen.py "
          "scripts/dip_hazard.py scripts/dip_count_hazard.py "
          "patches/apply_doc_catchup.py")
    print('  git commit -m "docs catch-up: breakout sp400 dead, seconds/dip-'
          'count seams, runlen dead (violation), dip-buy seam closed"')


if __name__ == "__main__":
    main()
