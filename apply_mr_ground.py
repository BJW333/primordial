#!/usr/bin/env python3
"""apply_mr_ground.py -- MR search manifest + prereg. Idempotent."""
import os, sys

def put(path, content, label):
    if os.path.exists(path):
        if open(path).read() == content:
            print(f"  = {label}: already present"); return
        sys.exit(f"{path} exists with DIFFERENT content -- refusing to overwrite.")
    open(path, "w").write(content); print(f"  + {label}")

if not os.path.exists("manifests/us_sp400.yaml"):
    sys.exit("run from the primordial repo root.")
put("manifests/us_sp400_mr.yaml", "# MEAN-REVERSION SEARCH GROUND (2026-08-11)\n# Same names/costs/split as us_sp400.yaml, with three deliberate constraints\n# taken from things this project already learned the hard way:\n#\n#   allowed_timeframes: [1d] ONLY -- in the us_liquid_5m run evolution\n#     abandoned the intended timeframe within one generation and produced\n#     zero survivors. If the hypothesis is daily mean reversion, do not\n#     leave a 1w escape hatch open.\n#   engines: both -- absolute (timing) and relative (cross_sectional) MR are\n#     different claims; let the search tell you which one it finds, and the\n#     ledger charge the whole search either way.\n#   embargo raised 20 -> 40 -- MR holds are multi-week; a 20-bar embargo\n#     leaks a hold across the boundary.\n#\n# SURVIVORSHIP: today's membership. Any survivor owes a point-in-time\n# re-test before capital -- same queue as the sp400 rotation.\nname: us_sp400_mr\nasset_class: equity\nsource: yfinance\nsymbols_file: sp400_symbols.txt\nbase_timeframe: 1d\nallowed_timeframes: [1d]\nstart: \"2010-01-01\"\nend: \"2026-06-01\"\ncost_tier: equity_midcap\nholdout_name_frac: 0.4\nholdout_time_frac: 0.3\nembargo_bars: 40\nengines: [timing, cross_sectional]\nn_motifs: 6\nmotif_scales: [20, 40]\ncontext_symbols: [SPY, TLT, IWM]\n", "manifests/us_sp400_mr.yaml")
put("MR_PREREG.md", "# MR search \u2014 pre-registration (2026-08-11)\n\nWHAT: let primordial evolve on us_sp400_mr, no rule handed to it.\n\nWHY THIS GROUND: sp400 daily. New fingerprint, so the bar comes from THIS\nsearch's own trial count (pop x generations x islands), not from an empty\nledger \u2014 a search-run bar is honest in a way a single hand-genome run on\nfresh ground is not.\n\nTHE MR SEAM IS MOSTLY MINED. Already dead on honest bars, do not re-accept:\n  distress (dip below SMA200)          0/5 splits, sp600\n  rsi dip sp400                        dead\n  one-day crash sp600                  +0.08 vs 0.53\n  5-day pullback us_liquid_daily       +0.81, bootstrap 0\n  cross-sectional dip sp600            +0.23 vs 0.26\n  intraday gap reversal                0-for-3, 2025 inverted -2.6%\n  VIX panic buy                        dead on PSR\nOnly survivor of the family: deep-state \u2014 slow, ABOVE-trend absorption.\n\nREGISTERED: if a survivor's entry is a re-description of any dead shape\nabove (dip depth / oscillator extreme / below-trend), it is NOT a new\nfinding regardless of its SR \u2014 record it as a re-hit and close. A pass\ncounts only if the entry names something structurally different.\n\nPASS = whatever the gauntlet says. No threshold edits, no variant mining\nafter seeing results. Zero survivors is the expected outcome and is a\ncomplete result.\n\nSigned: __________\n", "MR_PREREG.md")
print("\nNext: commit, then")
print("  python3.10 scripts/fetch_data.py manifests/us_sp400_mr.yaml")
print("  python3.10 -m primordial run --manifest manifests/us_sp400_mr.yaml \\")
print("      --generations 20 --pop 40 --islands 3 --seed 42")
