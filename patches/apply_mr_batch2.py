#!/usr/bin/env python3
"""apply_mr_batch2.py -- cross-sectional-only MR/momentum ground. Idempotent."""
import os, sys

if not os.path.exists("MR_PREREG.md"):
    sys.exit("run from the primordial repo root.")
M = "# CROSS-SECTIONAL ONLY -- MR/momentum batch #2 on the sp400_mr fingerprint\n# (batch #1: 0 survivors of 6, bar +1.20, 71 effective trials; evolution\n# abandoned per-name signals for ctx_SPY/ctx_IWM market timing).\n#\n# Two constraints, both defining the hypothesis BEFORE the run:\n#   context_symbols: []       -- no SPY/IWM/TLT atoms exist, so the search\n#                                cannot flee to index timing again. It must\n#                                use stock-level and rank structure.\n#   engines: [cross_sectional] -- the actual MR/momentum question: rank names\n#                                against each other, buy losers or winners.\n#\n# Ledger: this is a SECOND search on this ground. The bar will be higher than\n# +1.20 and that is correct -- it is the price of the second look.\nname: us_sp400_mr\nasset_class: equity\nsource: yfinance\nsymbols_file: sp400_symbols.txt\nbase_timeframe: 1d\nallowed_timeframes: [1d]\nstart: \"2010-01-01\"\nend: \"2026-06-01\"\ncost_tier: equity_midcap\nholdout_name_frac: 0.4\nholdout_time_frac: 0.3\nembargo_bars: 40\nengines: [cross_sectional]\nn_motifs: 6\nmotif_scales: [20, 40]\ncontext_symbols: []\n"
p = "manifests/us_sp400_mr_xs.yaml"
if os.path.exists(p) and open(p).read() != M:
    sys.exit(f"{p} exists with different content -- refusing to overwrite.")
if not os.path.exists(p):
    open(p, "w").write(M); print(f"  + {p}")
else:
    print(f"  = {p}: already present")
A = "\n## Batch #2 (2026-08-11) \u2014 cross-sectional only\n\nBatch #1 result: 0 survivors of 6 candidates, holdout +0.33 best vs bar\n+1.20, 71 effective trials. Train fitness never exceeded +0.06 across 20\ngenerations; finalists were market-context conditions (ctx_SPY_trend,\nctx_IWM_roc, ctx_breadth), i.e. the search abandoned per-name reversion\nand reached for index timing.\n\nBatch #2 registered BEFORE running: manifests/us_sp400_mr_xs.yaml \u2014\ncontext_symbols emptied (no index-timing escape hatch) and\nengines=[cross_sectional] only (rank-relative MR/momentum, the actual\nclaim). Same names, costs, split, embargo. Same pass rules; bar will be\nhigher than +1.20 because this is the second search on this fingerprint.\n\nREGISTERED STOP: if batch #2 returns 0 survivors, the daily-MR/momentum\nseam on midcap US equities is CLOSED. No batch #3, no threshold edits, no\n\"one more universe.\" Next effort goes to forced-flow hypotheses\n(reconstitution, month-end) which are gated on point-in-time data.\n"
s = open("MR_PREREG.md").read()
if "Batch #2 (2026-08-11)" in s:
    print("  = MR_PREREG.md: batch #2 already recorded")
else:
    open("MR_PREREG.md", "a").write(A); print("  + MR_PREREG.md: batch #1 result + batch #2 registration")
print("\nCommit BEFORE running:")
print("  python3.10 -m primordial run --manifest manifests/us_sp400_mr_xs.yaml \\")
print("      --generations 20 --pop 40 --islands 3 --seed 42")
