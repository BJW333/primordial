"""
fetch_attention_all.py -- extend Wikipedia page views from the S&P 500 to
every US stock in data/us_universe/ (small and mid caps included), using
the Wikidata ticker -> article mapping already saved by
fetch_attention_short.py. Resumable; already-fetched companies are skipped.

Run (repo root, ~30-40 min first time):
    python3.10 scripts/fetch_attention_all.py --email you@example.com
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_attention_short as FA  # noqa: E402

ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    a = ap.parse_args()
    close = pd.read_parquet(os.path.join(ROOT, "data", "us_universe",
                                         "close.parquet"), columns=None)
    tickers = sorted({c for c in close.columns if "~" not in c})
    print(f"  {len(tickers)} US tickers in the universe")
    FA.fetch_attention(FA.sess(a.email), tickers)


if __name__ == "__main__":
    main()
