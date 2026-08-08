"""
Build a symbols file for an index's CURRENT constituents (S&P 600 / 400 / 500)
from Wikipedia, for use via `symbols_file:` in a manifest.

    python3.10 scripts/build_universe.py sp600
    python3.10 scripts/build_universe.py sp400
    python3.10 scripts/build_universe.py sp500

Writes manifests/<name>_symbols.txt (one ticker per line, yfinance format).

*** SURVIVORSHIP WARNING -- read before believing any result ***
This is the index's membership TODAY. Names that got delisted, acquired, or
demoted over your backtest window are missing, and in small caps that's a big,
optimistic bias: the companies that died are exactly the ones a strategy
could have been long. Treat absolute Sharpes on this universe as inflated.
It's still legitimate for STRUCTURE discovery (the name x time holdout keeps
the search honest about generalization across names it trained on), but a
survivor here graduates to a point-in-time universe check before any capital.
"""
import sys
import os

PAGES = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def build(which: str) -> str:
    import io
    import urllib.request
    import pandas as pd
    url = PAGES[which]
    # Wikipedia 403s the default python/pandas user-agent; fetch as a browser
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    tables = pd.read_html(io.StringIO(html))
    tab = next(t for t in tables
               if "Symbol" in t.columns or "Ticker symbol" in t.columns)
    col = "Symbol" if "Symbol" in tab.columns else "Ticker symbol"
    syms = sorted({str(s).strip().replace(".", "-")     # BRK.B -> BRK-B
                   for s in tab[col].dropna()})
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "manifests", f"{which}_symbols.txt")
    with open(out, "w") as f:
        f.write(f"# {which.upper()} current constituents "
                f"({len(syms)} names) -- see survivorship warning in "
                f"scripts/build_universe.py\n")
        f.write("\n".join(syms) + "\n")
    print(f"{which}: {len(syms)} symbols -> {out}")
    return out


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "sp600").lower()
    if which not in PAGES:
        sys.exit(f"choose one of: {', '.join(PAGES)}")
    build(which)
