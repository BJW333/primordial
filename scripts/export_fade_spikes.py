"""
export_fade_spikes.py -- write the fade's hype-spike events to a small CSV
for QuantConnect (the raw Wikipedia data is ~150 MB; the events are ~8k rows).

Each row: ticker, spike date, size group (SMALL/MID), liquid (top half of
its size group by dollar volume at the spike). Exactly the events
attention_rebound.py / fade_predict.py use. Nothing after the spike date is
used, so the file is point-in-time safe.

Run (repo root):  python3.10 scripts/export_fade_spikes.py
-> data/fade_spikes.csv   (upload to QuantConnect's Object Store as 'fade_spikes.csv')
-> data/wiki_mapping.csv  (upload as 'wiki_mapping.csv' -- the live feed needs it)
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attention_rebound as AR        # noqa: E402
import attention_short_test as AS     # noqa: E402
import factor_portfolio as FP         # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "fade_spikes.csv")


def main():
    views = AS.norm_index(pd.read_parquet(AS.ATT))
    sp_members, _ = FP.membership()
    px, vol, unis = AR.universes(sp_members)
    dv = (px * vol).rolling(63, min_periods=40).mean()
    rows = []
    for name, (mem, _) in unis.items():
        ev = AR.good_spikes(views, px, mem)
        for s, d in zip(ev["sym"], ev["d"]):
            live = [m for m in mem(d) if m in dv.columns]
            liq = bool(dv.at[d, s] >= dv.loc[d, live].median()) if live else False
            rows.append({"ticker": s, "date": d.strftime("%Y-%m-%d"),
                         "group": name, "liquid": int(liq)})
    df = pd.DataFrame(rows).sort_values(["date", "ticker"])
    df.to_csv(OUT, index=False)
    print(f"  {len(df)} spikes ({(df.liquid == 1).sum()} liquid) -> {OUT}")


def export_mapping():
    """ticker -> Wikipedia article candidates, for the live feed."""
    src = os.path.join(ROOT, "data", "attention", "mapping.csv")
    m = pd.read_csv(src)
    m["ticker"] = m["ticker"].astype(str).str.upper()
    out = os.path.join(ROOT, "data", "wiki_mapping.csv")
    m[["ticker", "article"]].drop_duplicates().to_csv(out, index=False)
    print(f"  {m['ticker'].nunique()} tickers mapped -> {out}")


if __name__ == "__main__":
    main()
    export_mapping()
