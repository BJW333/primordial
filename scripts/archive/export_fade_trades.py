"""
export_fade_trades.py -- the EXACT trades Primordial's validated fade book
takes (liquid half, WEAK_TREND, 50% stop covered the day after), for a
QuantConnect REPLAY.

DIAGNOSTIC ONLY: exit dates use later prices (the stop), so this list is not
a strategy. It answers one question: when QuantConnect trades the same
stocks on the same days, does it get the same result as Primordial?

Run (repo root):  python3.10 scripts/export_fade_trades.py
-> data/fade_trades.csv   (upload to the QC Object Store as 'fade_trades.csv')
"""
import os as _os, sys as _sys  # archived script: shared modules live one folder up
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attention_rebound as AR        # noqa: E402
import attention_short_test as AS     # noqa: E402
import factor_portfolio as FP         # noqa: E402
import fade_predict as P              # noqa: E402
import fade_stop_check as FSC         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "data", "fade_trades.csv")


def main():
    P.CLIP = False
    views = AS.norm_index(pd.read_parquet(AS.ATT))
    sp_members, _ = FP.membership()
    px, vol, unis = AR.universes(sp_members)
    dv = (px * vol).rolling(63, min_periods=40).mean()
    idx = px.index
    cv = px.ffill().values
    cols = {c: j for j, c in enumerate(px.columns)}
    rows = []
    for name in [u for u in ("SMALL", "MID") if u in unis]:
        mem, bench = unis[name]
        b = bench.reindex(idx).ffill().values
        df = P.build(name, AR.good_spikes(views, px, mem), px, bench,
                     {}, {}, mem, dv)
        tr = df[df["liq"] & df["W"]].sort_values("d")
        print(f"\n  {name}: {len(tr)} trades -- Primordial book, no borrow:")
        r, st = FSC.book(tr, px, bench, 0.0, 0.50, 1)
        FSC.line("paper result (compare with QC)", r, st)
        for s, d, e, x in zip(tr["sym"], tr["d"], tr["e"], tr["x"]):
            c = cv[:, cols[s]]
            xx, stopped = x, 0
            hit = np.flatnonzero(c[e + 1:x + 1] >= c[e] * 1.5)
            if len(hit):
                xx, stopped = min(e + 1 + hit[0] + 1, len(idx) - 1), 1
            rows.append({"ticker": s, "group": name,
                         "spike": d.strftime("%Y-%m-%d"),
                         "entry": idx[e].strftime("%Y-%m-%d"),
                         "exit": idx[xx].strftime("%Y-%m-%d"),
                         "stopped": stopped,
                         "stock_ret": round(c[xx] / c[e] - 1, 4),
                         "bench_ret": round(b[xx] / b[e] - 1, 4)})
    out = pd.DataFrame(rows).sort_values("entry")
    out = out[(out["entry"] >= "2017-01-01") & (out["entry"] < "2026-06-01")]
    out.to_csv(OUT, index=False)
    days = pd.bdate_range("2017-01-01", "2026-06-01")
    e, x = pd.to_datetime(out["entry"]).values, pd.to_datetime(out["exit"]).values
    n_open = np.array([((e <= d) & (x > d)).sum() for d in days.values])
    gap = out["bench_ret"] - out["stock_ret"]
    print(f"\n  {len(out)} trades -> {OUT}")
    print(f"  open at once: avg {n_open.mean():.0f}, max {n_open.max()}")
    print(f"  per trade: hedge {out['bench_ret'].mean()*100:+.1f}%, "
          f"stock {out['stock_ret'].mean()*100:+.1f}%, "
          f"short-minus-hedge profit {gap.mean()*100:+.1f}% "
          f"(median {gap.median()*100:+.1f}%), stopped {out['stopped'].mean()*100:.0f}%")


if __name__ == "__main__":
    main()
