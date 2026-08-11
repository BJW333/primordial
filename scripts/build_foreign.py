"""
Build foreign mid-cap universes for the deep-state geography test.
Scrapes index constituents from Wikipedia and maps them to Yahoo tickers
with the right exchange suffix.

    python3.10 scripts/build_foreign.py ftse250
    python3.10 scripts/build_foreign.py tsx
    python3.10 scripts/build_foreign.py asx200
    python3.10 scripts/build_foreign.py dax_mid
"""
import io
import os
import sys

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

SPECS = {
    "ftse250": dict(
        url="https://en.wikipedia.org/wiki/FTSE_250_Index",
        cols=("Ticker", "EPIC", "Symbol"), suffix=".L"),
    "tsx": dict(
        url="https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index",
        cols=("Ticker", "Symbol"), suffix=".TO"),
    "asx200": dict(
        url="https://en.wikipedia.org/wiki/S%26P/ASX_200",
        cols=("Code", "Ticker", "ASX code", "Symbol"), suffix=".AX"),
    "dax_mid": dict(
        url="https://en.wikipedia.org/wiki/MDAX",
        cols=("Symbol", "Ticker"), suffix=".DE"),
    # Japan: Nikkei tickers are 4-digit codes; Yahoo wants <code>.T
    "nikkei": dict(
        url="https://en.wikipedia.org/wiki/Nikkei_225",
        cols=("Code", "Ticker", "Symbol"), suffix=".T"),
    "jpx400": dict(
        url="https://en.wikipedia.org/wiki/JPX-Nikkei_Index_400",
        cols=("Code", "Ticker", "Symbol"), suffix=".T"),
    # Korea: 6-digit codes, Yahoo wants <code>.KS
    "kospi": dict(
        url="https://en.wikipedia.org/wiki/KOSPI",
        cols=("Symbol", "Code", "Ticker"), suffix=".KS"),
}


def clean(t, suffix):
    t = str(t).strip().upper().replace(" ", "")
    if not t or t in ("NAN", "-"):
        return None
    t = t.split(":")[-1].replace(".", "-")
    return t + suffix


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in SPECS:
        sys.exit(f"usage: build_foreign.py [{'|'.join(SPECS)}]")
    key = sys.argv[1]
    spec = SPECS[key]
    html = requests.get(spec["url"], headers=UA, timeout=30).text
    syms = []
    for tb in pd.read_html(io.StringIO(html)):
        for c in spec["cols"]:
            if c in tb.columns:
                syms = [clean(x, spec["suffix"]) for x in tb[c]]
                syms = [s for s in syms if s]
                if len(syms) > 40:
                    break
        if len(syms) > 40:
            break
    if len(syms) < 40:
        sys.exit(f"only found {len(syms)} tickers -- page layout changed; "
                 f"inspect {spec['url']}")
    seen, out = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "manifests", f"{key}_symbols.txt")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"{key}: {len(out)} symbols -> {path}")
    print("  sample:", ", ".join(out[:8]))


if __name__ == "__main__":
    main()
