"""
Typed GP trees over an injected terminal set.

Two node value types flow through evaluation:
  Series  -- float array, one value per bar
  Bool    -- 0/1 array

Ops are pure math (Tier 0). NO domain content lives here: no indicator logic,
no thresholds, no market opinion. Terminals are column names supplied by the
caller (atoms.py decides what exists for a manifest) -- there is no module
global to mutate, so parallel islands with different terminal sets cannot
corrupt each other.

Trees are plain dicts -> JSON-serializable -> archivable and deployable.
"""
from __future__ import annotations
import copy
import random
import numpy as np

# ---- op table: name -> (arg_types, return_type, arity) ------------------
S, B = "series", "bool"
OPS = {
    # logic
    "and": ([B, B], B), "or": ([B, B], B), "not": ([B], B), "xor": ([B, B], B),
    # comparison
    "gt": ([S, S], B), "lt": ([S, S], B), "gte": ([S, S], B), "lte": ([S, S], B),
    "cross_above": ([S, S], B), "cross_below": ([S, S], B),
    # arithmetic
    "add": ([S, S], S), "sub": ([S, S], S), "mul": ([S, S], S), "div": ([S, S], S),
    "absv": ([S], S), "neg": ([S], S), "minv": ([S, S], S), "maxv": ([S, S], S),
    # temporal (carry n)
    "lag": ([S], S), "diff": ([S], S), "ema": ([S], S),
    "roll_mean": ([S], S), "roll_std": ([S], S), "roll_min": ([S], S),
    "roll_max": ([S], S), "roll_rank": ([S], S), "roll_sum": ([S], S),
    # scaling (carry n)
    "zscore": ([S], S), "pctile_rank": ([S], S),
    "clip": ([S], S),
    # conditional
    "if_then_else": ([B, S, S], S),
}
N_OPS = {"lag", "diff", "ema", "roll_mean", "roll_std", "roll_min", "roll_max",
         "roll_rank", "roll_sum", "zscore", "pctile_rank"}
N_RANGE = (3, 200)
MAX_DEPTH = 5          # per tree; entry and regime trees budgeted separately


def node(op, ch=None, n=None, value=None, name=None):
    d = {"op": op, "ch": ch or []}
    if n is not None: d["n"] = int(n)
    if value is not None: d["value"] = float(value)
    if name is not None: d["name"] = name
    return d

def term(name):  return node("term", name=name)
def const(v):    return node("const", value=v)


def _roll(a, n, fn):
    import pandas as pd
    return fn(pd.Series(a).rolling(n, min_periods=max(2, n // 3))).to_numpy()


def evaluate(tree, df):
    """Evaluate a tree against a bar dataframe. Returns np.ndarray.
    Protected division (|denom| > 1e-9 -> else 0). NaNs propagate and are
    handled at the signal boundary."""
    import pandas as pd
    op = tree["op"]
    if op == "term":
        col = tree["name"]
        if col not in df.columns:
            raise KeyError(f"terminal {col!r} not in dataframe")
        return df[col].to_numpy(dtype=float)
    if op == "const":
        return np.full(len(df), tree["value"])
    ch = [evaluate(c, df) for c in tree["ch"]]
    n = tree.get("n", 14)
    if op == "and":  return ((ch[0] > 0) & (ch[1] > 0)).astype(float)
    if op == "or":   return ((ch[0] > 0) | (ch[1] > 0)).astype(float)
    if op == "xor":  return ((ch[0] > 0) ^ (ch[1] > 0)).astype(float)
    if op == "not":  return (~(ch[0] > 0)).astype(float)
    if op == "gt":   return (ch[0] > ch[1]).astype(float)
    if op == "lt":   return (ch[0] < ch[1]).astype(float)
    if op == "gte":  return (ch[0] >= ch[1]).astype(float)
    if op == "lte":  return (ch[0] <= ch[1]).astype(float)
    if op == "cross_above":
        a, b = ch
        pa, pb = np.roll(a, 1), np.roll(b, 1); pa[0], pb[0] = a[0], b[0]
        return ((a > b) & (pa <= pb)).astype(float)
    if op == "cross_below":
        a, b = ch
        pa, pb = np.roll(a, 1), np.roll(b, 1); pa[0], pb[0] = a[0], b[0]
        return ((a < b) & (pa >= pb)).astype(float)
    if op == "add":  return ch[0] + ch[1]
    if op == "sub":  return ch[0] - ch[1]
    if op == "mul":  return ch[0] * ch[1]
    if op == "div":
        return np.divide(ch[0], ch[1], out=np.zeros_like(ch[0]),
                         where=np.abs(ch[1]) > 1e-9)
    if op == "absv": return np.abs(ch[0])
    if op == "neg":  return -ch[0]
    if op == "minv": return np.minimum(ch[0], ch[1])
    if op == "maxv": return np.maximum(ch[0], ch[1])
    if op == "lag":  return pd.Series(ch[0]).shift(n).to_numpy()
    if op == "diff": return pd.Series(ch[0]).diff(n).to_numpy()
    if op == "ema":  return pd.Series(ch[0]).ewm(span=n, min_periods=2).mean().to_numpy()
    if op == "roll_mean": return _roll(ch[0], n, lambda r: r.mean())
    if op == "roll_std":  return _roll(ch[0], n, lambda r: r.std())
    if op == "roll_min":  return _roll(ch[0], n, lambda r: r.min())
    if op == "roll_max":  return _roll(ch[0], n, lambda r: r.max())
    if op == "roll_sum":  return _roll(ch[0], n, lambda r: r.sum())
    if op == "roll_rank":
        s = pd.Series(ch[0])
        return s.rolling(n, min_periods=max(2, n // 3)).rank(pct=True).to_numpy()
    if op == "zscore":
        s = pd.Series(ch[0]); r = s.rolling(n, min_periods=max(2, n // 3))
        sd = r.std()
        return ((s - r.mean()) / sd.replace(0, np.nan)).to_numpy()
    if op == "pctile_rank":
        s = pd.Series(ch[0])
        return s.rolling(n, min_periods=max(2, n // 3)).rank(pct=True).to_numpy()
    if op == "clip":
        return np.clip(ch[0], -3.0, 3.0)
    if op == "if_then_else":
        return np.where(ch[0] > 0, ch[1], ch[2])
    raise ValueError(f"unknown op {op}")


def bool_signal(tree, df):
    """Bool-rooted tree -> {0,1} entry signal with NaNs forced to 0."""
    v = evaluate(tree, df)
    return np.nan_to_num(v, nan=0.0).astype(float)


def series_score(tree, df):
    """Series-rooted tree -> cross-sectional score."""
    import pandas as pd
    v = evaluate(tree, df)
    return pd.Series(v, index=df.index)


# ---- random construction / mutation / crossover --------------------------
def random_tree(rng: random.Random, terminals, want=B, depth=0):
    if want == S:
        if depth >= MAX_DEPTH or rng.random() < 0.35:
            return term(rng.choice(terminals)) if rng.random() < 0.85 else \
                   const(round(rng.uniform(-2, 2), 2))
        ops = [o for o, (a, r) in OPS.items() if r == S]
    else:
        if depth >= MAX_DEPTH:
            return node("gt", [random_tree(rng, terminals, S, MAX_DEPTH),
                               random_tree(rng, terminals, S, MAX_DEPTH)])
        ops = [o for o, (a, r) in OPS.items() if r == B]
    op = rng.choice(ops)
    args, _ = OPS[op]
    ch = [random_tree(rng, terminals, a, depth + 1) for a in args]
    nd = node(op, ch)
    if op in N_OPS:
        nd["n"] = rng.randint(*N_RANGE)
    return nd


def _all_nodes(tree, want=None, acc=None):
    acc = acc if acc is not None else []
    rt = return_type(tree)
    if want is None or rt == want:
        acc.append(tree)
    for c in tree["ch"]:
        _all_nodes(c, want, acc)
    return acc


def return_type(tree):
    op = tree["op"]
    if op in ("term", "const"):
        return S
    return OPS[op][1]


def mutate(tree, rng, terminals):
    t = copy.deepcopy(tree)
    nodes = _all_nodes(t)
    tgt = rng.choice(nodes)
    r = rng.random()
    if r < 0.30 and "n" in tgt:                       # nudge a period
        tgt["n"] = int(np.clip(tgt["n"] + rng.randint(-20, 20), *N_RANGE))
    elif r < 0.55 and tgt["op"] == "const":           # nudge a constant
        tgt["value"] = round(tgt["value"] + rng.uniform(-0.5, 0.5), 3)
    elif r < 0.80:                                    # replace subtree
        sub = random_tree(rng, terminals, return_type(tgt),
                          depth=MAX_DEPTH - 1)
        tgt.clear(); tgt.update(sub)
    else:                                             # swap a terminal
        terms = [nd for nd in _all_nodes(t) if nd["op"] == "term"]
        if terms:
            rng.choice(terms)["name"] = rng.choice(terminals)
    return t


def crossover(a, b, rng):
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    for want in (B, S):
        na = _all_nodes(a2, want)
        nb = _all_nodes(b2, want)
        na = [x for x in na if x is not a2]
        nb = [x for x in nb if x is not b2]
        if na and nb:
            x, y = rng.choice(na), rng.choice(nb)
            xc = copy.deepcopy(x)
            x.clear(); x.update(copy.deepcopy(y))
            y.clear(); y.update(xc)
            return a2, b2
    return a2, b2


def size(tree):
    return 1 + sum(size(c) for c in tree["ch"])


def describe(tree):
    op = tree["op"]
    if op == "term":  return tree["name"]
    if op == "const": return f"{tree['value']:g}"
    inner = ", ".join(describe(c) for c in tree["ch"])
    nn = f", n={tree['n']}" if "n" in tree else ""
    return f"{op}({inner}{nn})"


def terminal_names(tree) -> list:
    """Every atom column this tree reads. Used by the fitness guard to
    evaluate a signal only where its inputs are actually defined."""
    out = []
    for nd in _all_nodes(tree):
        if nd["op"] == "term":
            out.append(nd["name"])
    return sorted(set(out))


def has_terminal(tree) -> bool:
    """True if any leaf reads market data. A tree of constants is an
    always-true / always-false switch, not a strategy."""
    if tree["op"] == "term":
        return True
    return any(has_terminal(c) for c in tree["ch"])
