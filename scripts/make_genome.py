"""
Write an idea in the SAME notation the logs already print -- no JSON by hand.

    python3.10 scripts/make_genome.py \
        "and(lt(rsi_m, 0.3), gt(close, roll_mean(close, n=200)))" \
        --tf 1d --dir long --stop 3 --max-hold 40 --anchor 26 > idea.json

Then:  python3.10 scripts/validate_genome.py --manifest <m> --genome idea.json

Entry expression language (exactly what describe() prints):
  atoms by name        rsi_m, volz, dist_sma_s, ema_spread_atr, roc_m, ...
  raw columns          close, open, high, low, volume
  numbers              0.3, -0.25, 15
  ops                  and(a,b) or(a,b) not(a) lt gt lte gte cross_above
                       add sub mul div neg abs clip lag diff
                       roll_mean roll_max roll_min roll_sum roll_rank (n=INT)
Common atoms: rsi_m, dist_sma_f/dist_sma_s, volz, roc_s/roc_m, bb_z_m,
  body_frac, rvol_m, atrp, ema_spread_atr, spread_runlen, obv_slope,
  xs_rank_roc_m, ctx_breadth, ctx_<BENCH>_trend.
"""
import argparse
import re
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from primordial import tree as T          # noqa: E402
from primordial.genome import Genome      # noqa: E402

TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|-?\d+\.?\d*|[(),=])")


def tokenize(s):
    out, i = [], 0
    while i < len(s):
        m = TOKEN.match(s, i)
        if not m:
            sys.exit(f"parse error near: {s[i:i+20]!r}")
        out.append(m.group(1))
        i = m.end()
    return out


class Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, want=None):
        tok = self.peek()
        if tok is None or (want and tok != want):
            sys.exit(f"expected {want!r}, got {tok!r} at position {self.i}")
        self.i += 1
        return tok

    def expr(self):
        tok = self.take()
        if re.fullmatch(r"-?\d+\.?\d*", tok):
            return T.const(float(tok))
        if self.peek() != "(":
            return T.term(tok)
        self.take("(")
        ch, n = [], None
        while self.peek() != ")":
            if self.peek() == "n":
                self.take("n"); self.take("=")
                n = int(float(self.take()))
            else:
                ch.append(self.expr())
            if self.peek() == ",":
                self.take(",")
        self.take(")")
        return T.node(tok, ch, n=n) if n is not None else T.node(tok, ch)


def _known_atoms():
    """Real atom vocabulary, computed once on a small synthetic frame."""
    import numpy as np
    import pandas as pd
    from primordial import atoms as A
    n = 400
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({
        "open": px, "high": px * 1.01, "low": px * 0.99, "close": px,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    cols = A.compute_atoms(df, list(A.BLOCKS.keys()), always_open=False)
    names = set(cols or [])
    names |= set(df.columns)
    names |= {"xs_rank_roc_m", "xs_rank_rvol_m"}
    return names


def _check_terminals(tree):
    """Fail loudly on atom names that do not exist -- a genome referencing an
    unknown terminal silently produces ZERO trades and still burns a ledger
    trial (2026-08-08: 'atr14' does not exist; the real atoms are atr_pct
    and ema_spread_atr)."""
    import sys
    known = _known_atoms()
    used, ops = set(), set()

    def walk(nd):
        ops.add(nd["op"])
        if nd["op"] == "term":
            used.add(nd["name"])
        for ch in nd.get("ch", []):
            walk(ch)
    walk(tree)
    from primordial.tree import OPS          # the real operator table
    bad_ops = sorted({o for o in ops if o not in OPS
                      and o not in ("term", "const")})
    if bad_ops:
        print(f"ERROR: unknown operator(s): {', '.join(bad_ops)}",
              file=sys.stderr)
        print(f"  valid ops: {', '.join(sorted(OPS))}", file=sys.stderr)
        sys.exit(2)
    # ctx_* (benchmark context), xs_* (cross-sectional) and match_m* (motifs)
    # are generated at RUN time from the manifest (context_symbols, motif
    # mining) and cannot exist on the synthetic probe frame. Accept them by
    # prefix; everything else must be a real computed atom.
    runtime_prefixes = ("ctx_", "xs_", "match_m")
    unknown = sorted(u for u in used
                     if u not in known
                     and not u.startswith(runtime_prefixes))
    if unknown:
        print(f"ERROR: unknown atom(s): {', '.join(unknown)}", file=sys.stderr)
        near = sorted(k for k in known
                      if any(u[:3] in k for u in unknown))[:12]
        if near:
            print(f"  did you mean: {', '.join(near)}", file=sys.stderr)
        sys.exit(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("entry", help="entry expression, describe() notation")
    p.add_argument("--tf", default="1d")
    p.add_argument("--dir", default="long", choices=["long", "short"])
    p.add_argument("--style", default="market",
                   choices=["market", "limit", "confirm"])
    p.add_argument("--stop", type=float, default=3.0,
                   help="ATR units; 99 = effectively no stop")
    p.add_argument("--target", type=float, default=0.0, help="R multiple")
    p.add_argument("--time-stop", type=int, default=0)
    p.add_argument("--trail", default="none",
                   choices=["none", "breakeven", "atr"])
    p.add_argument("--max-hold", type=int, default=40)
    p.add_argument("--anchor", type=int, default=0,
                   help="exit at close crossing this EMA (0 = off)")
    a = p.parse_args()
    entry = Parser(tokenize(a.entry)).expr()
    _check_terminals(entry)
    g = Genome(timeframe=a.tf, root_type="bool", entry_tree=entry,
               regime_tree=None, direction=a.dir, entry_style=a.style,
               entry_param=1, stop_atr=a.stop, target_r=a.target,
               time_stop=a.time_stop, trail_mode=a.trail,
               max_hold=a.max_hold, anchor_ema=a.anchor)
    print(g.to_json(), end="")
    print(f"\n# {g.describe()}", file=sys.stderr)


if __name__ == "__main__":
    main()
