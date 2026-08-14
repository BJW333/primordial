#!/usr/bin/env python3
"""apply_series_compare.py -- export each rotation's return series and add
scripts/compare_series.py to answer: are the industry-ETF rotation and the
sp400 name rotation the same edge expressed twice?

Run from repo root:  python3.10 apply_series_compare.py   (idempotent)"""
import os, sys

def patch(path, old, new, label):
    s = open(path).read()
    if new in s:
        print(f"  = {label}: already applied"); return
    if s.count(old) != 1:
        sys.exit(f"ANCHOR problem in {path} ({label}) -- aborting, nothing written.")
    open(path, "w").write(s.replace(old, new)); print(f"  + {label}")

def append_helper(path, label):
    s = open(path).read()
    if "def _export_series(" in s:
        print(f"  = {label}: already applied"); return
    open(path, "a").write(HELPER); print(f"  + {label}")

for p in ("scripts/test_volume_rotation.py", "scripts/test_weekly_rotation.py"):
    if not os.path.exists(p):
        sys.exit("run from the primordial repo root.")

HELPER = "\n\ndef _export_series(r, dates, tag):\n    \"\"\"Write the rule's own return series so two rotations can be compared\n    directly. Rebalance i is the return REALIZED over the period ENDING at\n    dates[i+1], so the series is labelled with dates[1:len(r)+1] -- getting\n    this off by one would manufacture or destroy correlation.\n    Writes to out/series/<tag>.csv (date,ret). Never affects the run.\"\"\"\n    try:\n        import os as _os\n        d = _os.path.join(\"out\", \"series\")\n        _os.makedirs(d, exist_ok=True)\n        idx = [str(x) for x in dates[1:len(r) + 1]]\n        if len(idx) != len(r):\n            print(f\"  (series export skipped: {len(idx)} dates vs {len(r)} rets)\")\n            return\n        p = _os.path.join(d, f\"{tag}.csv\")\n        with open(p, \"w\") as f:\n            f.write(\"date,ret\\n\")\n            for dt, v in zip(idx, r):\n                f.write(f\"{dt},{float(v):.10f}\\n\")\n        print(f\"  series -> {p} ({len(r)} periods)\")\n    except Exception as e:\n        print(f\"  (series export failed: {e})\")\n"

append_helper("scripts/test_volume_rotation.py", "volume rotation: _export_series helper")
append_helper("scripts/test_weekly_rotation.py", "weekly rotation: _export_series helper")
patch("scripts/test_volume_rotation.py", "    r_real, npos, turn, swaps = run_strategy(close, open_, score, vol, dates,\n                                             real_picker)\n    print(\"FULL PERIOD\")", "    r_real, npos, turn, swaps = run_strategy(close, open_, score, vol, dates,\n                                             real_picker)\n    _export_series(r_real, dates, f\"industry_{SIGNAL}_{MODE}_{DIRECTION}\")\n    print(\"FULL PERIOD\")", "volume rotation: export call")
patch("scripts/test_weekly_rotation.py", "    r_real, npos, turn, swaps = run_strategy(close, open_, score, vol, dates,\n                                             real_picker, allowed=GATE())", "    r_real, npos, turn, swaps = run_strategy(close, open_, score, vol, dates,\n                                             real_picker, allowed=GATE())\n    _export_series(r_real, dates, f\"names_{UNIV}_{REBAL}_{MEMBERSHIP}\"\n                   if UNIV == \"sp500pit\" else f\"names_{UNIV}_{REBAL}\")", "weekly rotation: export call")

CMP = "\"\"\"\ncompare_series.py -- are two rotation strategies the same edge twice?\n\n    python3.10 scripts/compare_series.py out/series/A.csv out/series/B.csv\n\nReads two period-return series written by the rotation scripts, aligns them\non DATE (inner join -- different rebalance calendars are fine, only shared\ndates are compared), and reports:\n\n  overlap        how many periods actually line up. Under 24, stop; the\n                 correlation is not measurable.\n  corr           Pearson on the overlap. This is the headline.\n  corr, holdout  same on the last 30% of shared periods -- correlation in\n                 the window where both were validated matters more than\n                 correlation over a full sample that includes both train\n                 sets.\n  hit agreement  fraction of periods where both are up or both are down.\n                 Correlation can be dragged by a few shared crashes; this\n                 is the blunter check.\n  combined       equal-weight blend of the two, and its SR against each\n                 leg's SR. If the blend's SR is not meaningfully above the\n                 better leg, holding both buys nothing.\n\nREAD IT THIS WAY (registered before looking):\n  corr >= 0.80   the same edge expressed twice. Trade ONE -- whichever has\n                 the better cost profile and the cleaner data story. The\n                 second adds fees, not diversification.\n  0.50-0.80      overlapping but not identical. A blend may help; the\n                 honest test is whether combined SR beats the better leg by\n                 more than the extra cost, which this prints.\n  < 0.50         genuinely different exposures. Two streams beat one.\n\nNote both series are gross of the correlation of their ERRORS: two rules on\noverlapping universes in the same decade share regimes, so some correlation\nis structural, not evidence of a shared signal.\n\"\"\"\nimport sys\n\nimport numpy as np\nimport pandas as pd\n\n\ndef load(p):\n    df = pd.read_csv(p)\n    if \"date\" not in df.columns or \"ret\" not in df.columns:\n        sys.exit(f\"{p}: expected columns date,ret\")\n    df[\"date\"] = pd.to_datetime(df[\"date\"]).dt.normalize()\n    return df.set_index(\"date\")[\"ret\"].astype(float).sort_index()\n\n\ndef sr(x):\n    x = np.asarray(x, dtype=float)\n    if len(x) < 3 or x.std(ddof=1) == 0:\n        return float(\"nan\")\n    per_yr = 12.0\n    return float(x.mean() / x.std(ddof=1) * np.sqrt(per_yr))\n\n\ndef main():\n    if len(sys.argv) != 3:\n        sys.exit(__doc__)\n    a, b = load(sys.argv[1]), load(sys.argv[2])\n    j = pd.concat({\"a\": a, \"b\": b}, axis=1).dropna()\n    n = len(j)\n    print(f\"  {sys.argv[1]}: {len(a)} periods  {a.index.min().date()} .. \"\n          f\"{a.index.max().date()}\")\n    print(f\"  {sys.argv[2]}: {len(b)} periods  {b.index.min().date()} .. \"\n          f\"{b.index.max().date()}\")\n    print(f\"  overlap: {n} shared dates\")\n    if n < 24:\n        sys.exit(\"  fewer than 24 shared periods -- correlation not \"\n                 \"measurable. Align the rebalance calendars first.\")\n\n    c = float(j[\"a\"].corr(j[\"b\"]))\n    cut = int(n * 0.7)\n    c_ho = float(j[\"a\"].iloc[cut:].corr(j[\"b\"].iloc[cut:]))\n    agree = float(((j[\"a\"] > 0) == (j[\"b\"] > 0)).mean())\n    print(f\"\\n  corr (all {n})      {c:+.3f}\")\n    print(f\"  corr (holdout {n - cut})  {c_ho:+.3f}\")\n    print(f\"  hit agreement       {agree:.0%}\")\n\n    sa, sb = sr(j[\"a\"]), sr(j[\"b\"])\n    blend = 0.5 * j[\"a\"] + 0.5 * j[\"b\"]\n    sc = sr(blend)\n    better = max(sa, sb)\n    print(f\"\\n  SR a {sa:+.2f} | SR b {sb:+.2f} | 50/50 blend {sc:+.2f} \"\n          f\"({sc - better:+.2f} vs better leg)\")\n\n    print(\"\\n  ===== READING =====\")\n    if c >= 0.80:\n        print(f\"  SAME EDGE TWICE (corr {c:+.2f}). Trade one. The second \"\n              f\"leg adds costs, not diversification.\")\n    elif c >= 0.50:\n        print(f\"  OVERLAPPING (corr {c:+.2f}). Blend only if the +\"\n              f\"{sc - better:.2f} SR gain survives the extra turnover -- \"\n              f\"and it is not free, both legs pay full spread.\")\n    else:\n        print(f\"  DISTINCT (corr {c:+.2f}). Two streams; the blend gain \"\n              f\"({sc - better:+.2f}) is real diversification.\")\n    print(\"  Caveat: shared decade and overlapping universes create \"\n          \"structural correlation independent of signal overlap.\")\n\n\nif __name__ == \"__main__\":\n    main()\n"
cp = "scripts/compare_series.py"
if os.path.exists(cp) and open(cp).read() != CMP:
    sys.exit(f"{cp} exists with different content -- refusing to overwrite.")
if not os.path.exists(cp):
    open(cp, "w").write(CMP); print(f"  + {cp}")
else:
    print(f"  = {cp}: already present")

print("\nRe-run both winners to write their series, then compare:")
print("  SIGNAL=roc DIRECTION=best MODE=cluster python3.10 scripts/test_volume_rotation.py")
print("  UNIV=sp400 REBAL=monthly python3.10 scripts/test_weekly_rotation.py")
print("  python3.10 scripts/compare_series.py \\")
print("      out/series/industry_roc_cluster_best.csv out/series/names_sp400_monthly.csv")
