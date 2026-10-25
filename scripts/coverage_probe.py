"""
coverage_probe.py -- does Alpaca have the dead stocks Yahoo dropped?

Survivorship killed or clouded every test on 2026-10-25 because Yahoo
deletes delisted tickers. Alpaca (already wired into primordial.data, with
your ALPACA_API_KEY / ALPACA_SECRET_KEY) serves history from 2016. This
probe checks, in ~2 minutes, whether it keeps delisted names:

  1. 13 well-known deaths/buyouts 2016-2025 (SIVB, FRC, BBBY, ATVI, ...)
  2. every spinoff ticker from data/spinoffs.csv that Yahoo FAILED on
     (no yfinance cache file for spinoff_test.py's window)

Anything Alpaca returns is cached, so spinoff_test.py --fallback alpaca
picks it up without downloading again.

    python3.10 scripts/coverage_probe.py
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial import data as D  # noqa: E402

SPIN = os.path.join(ROOT, "data", "spinoffs.csv")
Y_START, END = "2006-01-01", "2026-06-01"    # spinoff_test.py window
A_START = "2016-01-01"
DEAD = ["SIVB", "FRC", "SBNY", "BBBY", "ATVI", "TWTR", "CELG", "XLNX",
        "PXD", "HES", "WBA", "JNPR", "X"]


def alpaca(sym):
    """Fetch via primordial's cache; any API error = not available."""
    p = D._cache_path("alpaca", sym, "1d", A_START, END)
    if os.path.exists(p):
        return pd.read_parquet(p)
    try:
        df = D._fetch_one("alpaca", sym, "1d", A_START, END)
    except Exception:  # noqa: BLE001  (unknown symbol, auth, etc.)
        return None
    if df is not None and len(df):
        os.makedirs(D.cache_dir(), exist_ok=True)
        df.to_parquet(p)
        return df
    return None


def main():
    if not os.environ.get("ALPACA_API_KEY"):
        sys.exit("set ALPACA_API_KEY and ALPACA_SECRET_KEY first")

    print("  1) known deaths / buyouts:")
    hit = 0
    for s in DEAD:
        df = alpaca(s)
        ok = df is not None and len(df) > 20
        hit += ok
        span = f"{df.index[0].date()} -> {df.index[-1].date()}" if ok else "-"
        print(f"     {s:<6} {'YES' if ok else 'no ':<4} {span}")
    print(f"     Alpaca has {hit}/{len(DEAD)}")

    if not os.path.exists(SPIN):
        return
    sp = pd.read_csv(SPIN, parse_dates=["first_10_12b"]).dropna(
        subset=["ticker"])
    sp = sp[sp["first_10_12b"] >= pd.Timestamp("2015-06-01")]
    miss = [t for t in sorted(sp["ticker"].unique()) if not os.path.exists(
        D._cache_path("yfinance", t, "1d", Y_START, END))]
    print(f"\n  2) spinoffs registered since mid-2015 missing on Yahoo: "
          f"{len(miss)}")
    got = [t for t in miss if (lambda d: d is not None and len(d) > 140)(
        alpaca(t))]
    print(f"     recovered from Alpaca (>140 bars): {len(got)}/{len(miss)}")
    if got:
        print(f"     e.g. {', '.join(got[:12])}")
    print("\n  Next if recovery is meaningful:\n"
          "    python3.10 scripts/spinoff_test.py --fallback alpaca")


if __name__ == "__main__":
    main()
