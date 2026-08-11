#!/usr/bin/env python3
"""apply_opengap_holdout.py -- one-shot pre-registered holdout for the gap cell.

Run from the primordial repo root:  python3.10 apply_opengap_holdout.py
Anchored edit; aborts writing nothing on mismatch. Idempotent."""
import os, sys

def patch(path, old, new, label):
    s = open(path).read()
    if new in s:
        print(f"  = {label}: already applied"); return
    if old not in s or s.count(old) != 1:
        sys.exit(f"ANCHOR problem in {path} ({label}) -- aborting, nothing written.")
    open(path, "w").write(s.replace(old, new)); print(f"  + {label}")

if not os.path.exists("scripts/explore_open_gap.py"):
    sys.exit("run from the primordial repo root.")
patch("scripts/explore_open_gap.py", "    p.add_argument(\"--train-frac\", type=float, default=0.70)\n    a = p.parse_args()", "    p.add_argument(\"--train-frac\", type=float, default=0.70)\n    p.add_argument(\"--holdout-cell\", action=\"store_true\",\n                   help=\"THE one look: score the pre-registered cell \"\n                        \"(gap<=-2 ATR, 0m entry, 120m exit) on the held-out \"\n                        \"date tail, once. No grid is shown in this mode.\")\n    a = p.parse_args()", "--holdout-cell flag")
patch("scripts/explore_open_gap.py", "    tr_ = all_[all_[\"date\"] < cut]\n    print(f\"[open_gap] {all_['sym'].nunique()} names | {len(dates):,} \"\n          f\"sessions | TRAIN to {cut} ({tr_['date'].nunique():,} sessions)\")", "    tr_ = all_[all_[\"date\"] < cut]\n\n    if a.holdout_cell:\n        # ================= THE ONE LOOK (pre-registered) =================\n        # Cell fixed BEFORE this look, from the 2026-08-10 train grid:\n        #   gap <= -2 ATR | enter 09:30 auction (0m) | exit 120m | cost as\n        #   passed via --cost-bps (registered at 25).\n        # Scored ONLY on dates >= cut, which the train grid never printed.\n        # PASS (registered): net mean > 0 at 25bps AND t >= 2.0. Else DEAD.\n        # Re-running teaches nothing; examining any OTHER cell down here is\n        # a new, unregistered trial. Known caveats carried into this look:\n        #   - auction fills on 2-9 ATR gap-downs trade far wider than\n        #     25bps; a marginal pass is a cost-model artifact until proven\n        #     otherwise;\n        #   - 2026 SIP data is missing for most names (subscription), so\n        #     the tail ends earlier than the calendar suggests;\n        #   - the daily-bar gap family is already 0-for-2 vs real bars.\n        ho = all_[(all_[\"date\"] >= cut) & (all_[\"gap\"] <= -2.0)\n                  & (all_[\"entry\"] == 0) & (all_[\"exit\"] == 120)]\n        net = ho[\"ret\"].dropna() - a.cost_bps / 1e4\n        n = len(net)\n        print(f\"  ===== HOLDOUT, one look: gap<=-2 ATR, 0m -> 120m =====\")\n        print(f\"  dates {cut} .. {max(dates)} | \"\n              f\"{ho['date'].nunique():,} sessions | {n:,} trades | \"\n              f\"cost {a.cost_bps:.0f} bps\")\n        if n < 30:\n            print(\"  fewer than 30 trades -- unscorable. Record and stop.\")\n            return\n        t = net.mean() / (net.std(ddof=1) / n ** 0.5)\n        print(f\"  net {net.mean()*100:+.3f}%/trade | t = {t:+.2f} | \"\n              f\"win {(net > 0).mean()*100:.0f}% | \"\n              f\"(train reference was +0.191%)\")\n        yr = ho.assign(net=net).dropna(subset=[\"net\"])\n        yr[\"y\"] = [d.year for d in yr[\"date\"]]\n        for y, g in yr.groupby(\"y\"):\n            print(f\"    {y}: {g['net'].mean()*100:+.3f}% on {len(g):,} \"\n                  f\"trades\")\n        verdict = net.mean() > 0 and t >= 2.0\n        print(\"  VERDICT vs registered thresholds: \"\n              + (\"PASS -- next step is a REAL cost model (spread at the \"\n                 \"auction), not size\" if verdict else\n                 \"DEAD -- record it and close the gap family.\"))\n        return\n        # =================================================================\n\n    print(f\"[open_gap] {all_['sym'].nunique()} names | {len(dates):,} \"\n          f\"sessions | TRAIN to {cut} ({tr_['date'].nunique():,} sessions)\")", "one-look holdout branch (registered cell + thresholds)")
if os.path.exists("genomes/opengap_deep.json"):
    os.remove("genomes/opengap_deep.json")
    print("  - genomes/opengap_deep.json removed (market-entry daily genome does not test the auction/120m claim)")
print("\nOne look only:  python3.10 scripts/explore_open_gap.py --cost-bps 25 --holdout-cell")
