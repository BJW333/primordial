"""
report.py -- one self-contained HTML per run. No external JS, no network;
curves are inline SVG. Open it in a browser and the whole run is legible in
thirty seconds: what was searched, who made the finals, what killed each one,
which vocabulary the population actually used, and how high the luck bar has
climbed on this manifest.

    python3.10 -m primordial report runs/20260729_190119
"""
from __future__ import annotations
import html
import json
import os

from .universe import Manifest

GREEN, RED, GREY = "#1a7f37", "#c22", "#888"


def _svg_curve(points, w=560, h=120, color="#245"):
    """points: [[ts, val], ...] -> inline SVG polyline with min/max labels."""
    if not points or len(points) < 3:
        return "<i>no curve</i>"
    vals = [p[1] for p in points]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    n = len(vals)
    pts = " ".join(f"{i * w / (n - 1):.1f},{h - (v - lo) / rng * (h - 12) - 6:.1f}"
                   for i, v in enumerate(vals))
    final = vals[-1]
    fcol = GREEN if final >= vals[0] else RED
    return (f'<svg width="{w}" height="{h}" style="background:#fafafa;'
            f'border:1px solid #ddd">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="1.5"/>'
            f'<text x="4" y="12" font-size="10" fill="{GREY}">{hi:.3f}</text>'
            f'<text x="4" y="{h-2}" font-size="10" fill="{GREY}">{lo:.3f}</text>'
            f'<text x="{w-60}" y="12" font-size="11" fill="{fcol}">'
            f'{(final/vals[0]-1)*100:+.1f}%</text></svg>')


def _gates(row):
    """Per-gate verdicts inferred from what the gauntlet recorded. A missing
    later field means the candidate died before reaching that gate."""
    bar = row.get("luck_bar", 0)
    med = row.get("holdout_median_sharpe", 0)
    out = [("holdout SR > luck bar",
            f"{med:+.2f} vs {bar:+.2f}", med > bar)]
    p = row.get("holdout_psr", 0)
    out.append(("PSR > 0.95", f"{p:.3f}", p > 0.95))
    cs = row.get("cost_stress_sharpes")
    if cs is None:
        out.append(("cost stress 2x/3x", "not reached", None))
    else:
        out.append(("cost stress 2x/3x",
                    " / ".join(f"{x:+.2f}" for x in cs), all(x > 0 for x in cs)))
    br = row.get("breadth")
    if br is not None:
        out.append(("breadth: frac>0.5 & p<0.05",
                    f"{br['frac']:.2f} on {br['names']} names, p={br['p']:.3f}",
                    br["frac"] > 0.5 and br["p"] < 0.05))
    pb = row.get("pbo")
    if pb is None:
        out.append(("PBO < 0.5", "not reached / n too small", None))
    else:
        out.append(("PBO < 0.5", f"{pb:.2f}", pb < 0.5))
    bs = row.get("bootstrap_sharpes")
    if bs is None:
        out.append(("bootstrap >= 50% of holdout", "not reached", None))
    else:
        import statistics
        m = statistics.median(bs)
        need = 0.5 * med
        floor = max(1.0, bar)
        out.append(("bootstrap: >=50% holdout OR > max(1, bar)",
                    f"med {m:+.2f} vs {need:+.2f} | floor {floor:+.2f}",
                    (m > need or m > floor) if med > 0 else False))
    b48 = row.get("bootstrap_sharpes_blk48")
    if b48 is not None:
        import statistics
        out.append(("bootstrap fixed-48 (informational, not a gate)",
                    f"med {statistics.median(b48):+.2f} "
                    f"({', '.join(f'{x:+.2f}' for x in b48)})", None))
    return out


def _atom_usage(rows):
    counts = {}
    for r in rows:
        gj = r.get("genome_json")
        if not gj:
            continue
        g = json.loads(gj)
        def walk(t):
            if not isinstance(t, dict):
                return
            if t.get("op") == "term":
                counts[t["name"]] = counts.get(t["name"], 0) + 1
            for c in t.get("ch", []):
                walk(c)
        walk(g.get("entry_tree"))
        if g.get("regime_tree"):
            walk(g["regime_tree"])
    return sorted(counts.items(), key=lambda x: -x[1])


def _ledger_history(fingerprint):
    from .pipeline import LEDGER_PATH
    if not os.path.exists(LEDGER_PATH):
        return []
    try:
        with open(LEDGER_PATH) as f:
            d = json.load(f)
    except Exception:
        return []
    return d.get(fingerprint, [])


def generate(run_dir: str) -> str:
    with open(os.path.join(run_dir, "run.json")) as f:
        art = json.load(f)
    rows = art.get("report", [])
    fp = art.get("fingerprint", "?")
    hist = _ledger_history(fp)

    H = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>primordial -- {html.escape(os.path.basename(run_dir))}</title>
<style>
 body{{font:14px -apple-system,Helvetica,sans-serif;margin:24px;color:#222;
      max-width:960px}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
 table{{border-collapse:collapse;margin:6px 0}}
 td,th{{border:1px solid #ddd;padding:4px 8px;font-size:13px;text-align:left}}
 .ok{{color:{GREEN};font-weight:600}} .bad{{color:{RED};font-weight:600}}
 .na{{color:{GREY}}}
 .cand{{border:1px solid #ccc;border-radius:6px;padding:12px;margin:14px 0;
        background:#fff}}
 .surv{{border:2px solid {GREEN}}}
 code{{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:12px}}
 .muted{{color:{GREY};font-size:12px}}
</style></head><body>
<h1>primordial run report</h1>
<p class=muted>run: {html.escape(run_dir)} &middot; manifest:
{html.escape(str(art.get('manifest')))} &middot; fingerprint <code>{fp}</code>
&middot; {art.get('n_evaluated','?')} genomes evaluated in
{art.get('runtime_sec','?')}s &middot; all-time effective trials on this
holdout: <b>{art.get('effective_trials_alltime','?')}</b></p>"""]

    n_surv = sum(1 for r in rows if r.get("survived"))
    H.append(f"<h2>{n_surv} survivor(s) of {len(rows)} finalists</h2>")
    if n_surv == 0:
        H.append("<p>Nothing survived. That is a real answer -- do not "
                 "loosen a threshold to change it.</p>")

    for i, r in enumerate(rows):
        cls = "cand surv" if r.get("survived") else "cand"
        verdict = ("<span class=ok>SURVIVED</span>" if r.get("survived")
                   else "<span class=bad>rejected</span>")
        H.append(f'<div class="{cls}"><b>#{i}</b> {verdict} &middot; '
                 f'<code>{html.escape(r.get("genome",""))}</code>'
                 f'<p class=muted>timeframe {r.get("timeframe")} &middot; '
                 f'train fit {r.get("train_fit",0):+.2f} &middot; holdout '
                 f'trades {r.get("holdout_trades",0)}</p>')
        H.append(_svg_curve(r.get("holdout_equity")))
        H.append("<table><tr><th>gate</th><th>value</th><th></th></tr>")
        for name, val, ok in _gates(r):
            mark = ('<span class=na>&mdash;</span>' if ok is None else
                    '<span class=ok>&#10003;</span>' if ok else
                    '<span class=bad>&#10007;</span>')
            H.append(f"<tr><td>{name}</td><td>{val}</td><td>{mark}</td></tr>")
        H.append("</table></div>")

    usage = _atom_usage(rows)
    if usage:
        H.append("<h2>vocabulary the finalists actually used</h2><table>"
                 "<tr><th>atom</th><th>uses</th></tr>")
        for name, c in usage[:25]:
            H.append(f"<tr><td><code>{html.escape(name)}</code></td>"
                     f"<td>{'&#9608;' * c} {c}</td></tr>")
        H.append("</table>")

    if hist:
        H.append("<h2>trial ledger on this manifest</h2><table>"
                 "<tr><th>when</th><th>engine</th><th>candidates</th>"
                 "<th>holdout peeks</th></tr>")
        for e in hist[-15:]:
            H.append(f"<tr><td>{e.get('ts','')}</td><td>{e.get('engine','')}"
                     f"</td><td>{e.get('n_candidates','')}</td>"
                     f"<td>{e.get('holdout_peeks','')}</td></tr>")
        H.append("</table><p class=muted>every entry raises the luck bar for "
                 "the next run on this exact universe -- that is the "
                 "anti-p-hacking mechanism, not a nuisance.</p>")

    H.append("</body></html>")
    out = os.path.join(run_dir, "report.html")
    with open(out, "w") as f:
        f.write("\n".join(H))
    return out
