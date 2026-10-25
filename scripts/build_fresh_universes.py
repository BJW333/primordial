"""
build_fresh_universes.py -- writes symbol lists for markets the dip work has
never touched, scraped from Wikipedia constituent tables ON YOUR MACHINE.
Current membership (same survivorship caveat as every other list here).

    python3.10 scripts/build_fresh_universes.py
    -> manifests/fresh_<market>_symbols.txt  (+ prints counts and a sample)

A market whose page can't be parsed is skipped and says so; nothing is
guessed. Check the printed samples look like real tickers before running
market_trend_fresh.py.
"""
import io
import os
import re
import sys

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "manifests")
W = "https://en.wikipedia.org/wiki/"

# market: (pages to merge, yfinance suffix, normalizer kind)
MARKETS = {
    "france":      (["SBF_120"], ".PA", "plain"),
    "sweden":      (["OMX_Stockholm_30"], ".ST", "dash"),
    "switzerland": (["Swiss_Market_Index", "SMI_MID"], ".SW", "plain"),
    "hongkong":    (["Hang_Seng_Index"], ".HK", "hk"),
    "brazil":      (["Ibovespa"], ".SA", "plain"),
    "spain":       (["IBEX_35"], ".MC", "dash"),
    "italy":       (["FTSE_MIB"], ".MI", "plain"),
    "netherlands": (["AEX_index", "AMX_index"], ".AS", "plain"),
    "singapore":   (["Straits_Times_Index"], ".SI", "plain"),
    "mexico":      (["S%26P/BMV_IPC"], ".MX", "plain"),
}
TICKER_COLS = ["ticker", "symbol", "ticker symbol", "code", "stock code",
               "sehk", "trading symbol", "epic"]


def tables(page):
    r = requests.get(W + page, headers={"User-Agent": "Mozilla/5.0"},
                     timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def pick_column(df):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for want in TICKER_COLS:
        for k, c in cols.items():
            if k == want or k.startswith(want):
                return df[c]
    return None


def norm(raw, suffix, kind):
    s = re.sub(r"\[.*?\]", "", str(raw)).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.split(":")[-1].strip()          # "EPA: AI" -> "AI"
    s = s.split()[0] if kind == "hk" else s
    if s.upper().endswith(suffix.upper()):
        s = s[: -len(suffix)]
    if kind == "hk":
        digits = re.sub(r"\D", "", s)
        return f"{int(digits):04d}{suffix}" if digits else None
    if len(s.split()) > 2:                 # a company name, not a ticker
        return None
    s = s.replace("*", "").strip()
    if kind == "dash":
        s = s.replace(" ", "-").replace(".", "-")
    s = s.replace(" ", "")
    if not re.fullmatch(r"[A-Za-z0-9\-&]{1,12}", s):
        return None
    return s.upper() + suffix


def build(market, pages, suffix, kind):
    syms = []
    for p in pages:
        try:
            best = []
            for t in tables(p):
                col = pick_column(t)
                if col is None:
                    continue
                got = [x for x in (norm(v, suffix, kind) for v in col) if x]
                if len(got) > len(best):
                    best = got
            syms += best
        except Exception as e:   # noqa: BLE001
            print(f"  {market}: {p} failed ({e.__class__.__name__}: {e})")
    return sorted(set(syms))


def main():
    ok = 0
    for m, (pages, suffix, kind) in MARKETS.items():
        syms = build(m, pages, suffix, kind)
        if len(syms) < 20:
            print(f"  {m:<12} SKIPPED -- only {len(syms)} tickers parsed")
            continue
        path = os.path.join(OUT, f"fresh_{m}_symbols.txt")
        with open(path, "w") as f:
            f.write(f"# {m} current constituents from Wikipedia "
                    f"({', '.join(pages)}) -- survivorship caveat applies\n")
            f.write("\n".join(syms) + "\n")
        ok += 1
        print(f"  {m:<12} {len(syms):>4} tickers -> {os.path.basename(path)}"
              f"   e.g. {', '.join(syms[:5])}")
    print(f"\n  {ok} markets written. market_trend_fresh.py needs >= 6.")
    if ok < 6:
        sys.exit(1)


if __name__ == "__main__":
    main()
