"""
Prefetch (warm the cache for) every symbol a manifest needs, BEFORE a search.

    python3.10 scripts/fetch_data.py                          # all manifests
    python3.10 scripts/fetch_data.py manifests/coinbase_spot.yaml
    python3.10 scripts/fetch_data.py manifests/*.yaml --source omnifeed

Why: the search should never wait on the network. Run this once (slow --
Coinbase pages 300 candles/request), then every `primordial run` reads
parquet instantly. Also the thing to run locally before shipping the cache
dir to a RunPod box.

--source omnifeed overrides every manifest's source to route through your
omnifeed package (better pagination/retry/rate-limit handling; needs
`pip install` from your omnifeed repo). Cache entries are keyed by source,
so pick one and stick with it per manifest.
"""
import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from primordial.data import fetch, cache_dir          # noqa: E402
from primordial.universe import Manifest              # noqa: E402


def prefetch(path: str, source_override=None):
    mf = Manifest.load(path)
    source = source_override or mf.source
    if source == "synthetic":
        print(f"[{mf.name}] synthetic -- nothing to download")
        return
    print(f"[{mf.name}] {len(mf.symbols)} symbols | {mf.base_timeframe} | "
          f"{mf.start} -> {mf.end} | source={source}")
    t0 = time.time()
    ok, bad = 0, []
    for i, sym in enumerate(mf.symbols, 1):
        try:
            got = fetch(source, [sym], mf.base_timeframe, mf.start, mf.end)
            if sym in got:
                ok += 1
                print(f"  [{i:>3}/{len(mf.symbols)}] {sym:<12} "
                      f"{len(got[sym]):>8,} bars", flush=True)
            else:
                bad.append(sym)
                print(f"  [{i:>3}/{len(mf.symbols)}] {sym:<12} "
                      f"-- too few bars / empty", flush=True)
        except Exception as e:
            bad.append(sym)
            print(f"  [{i:>3}/{len(mf.symbols)}] {sym:<12} FAILED: {e}",
                  flush=True)
    print(f"[{mf.name}] done: {ok} cached, {len(bad)} failed "
          f"({time.time()-t0:.0f}s) -> {cache_dir()}")
    if bad:
        print(f"  failed: {', '.join(bad)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifests", nargs="*",
                   help="manifest yaml paths (default: manifests/*.yaml)")
    p.add_argument("--source", default=None,
                   help="override every manifest's source, e.g. omnifeed")
    a = p.parse_args()
    paths = a.manifests or sorted(glob.glob(os.path.join(ROOT, "manifests",
                                                         "*.yaml")))
    for path in paths:
        prefetch(path, a.source)
