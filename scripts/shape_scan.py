"""
shape_scan.py -- counts the most common chart SHAPES in the commodity ETFs.

What it does
  1 cuts every chart into windows of W hourly bars (default 14 ~ 2 days and
    35 ~ 1 week), stepping W bars each time so windows don't overlap
  2 rescales each window so only its SHAPE is left (same height, same level)
  3 groups similar windows together (k-means, K shapes per window length)
  4 prints each shape's count and share, and draws them in a PNG grid,
    most common first

Data: data/commod_rth_1h/<SYM>.csv from scripts/fetch_commod_rth.py.
Run (repo root):
    python3.10 scripts/shape_scan.py
    python3.10 scripts/shape_scan.py --windows 7 14 35 --k 16 --symbols GLD SLV
Output: printed table + logs/shapes_<W>.png
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "commod_rth_1h")
OUT = os.path.join(ROOT, "logs")
CONTEXT = {"UUP", "TLT", "SPY"}


def load(symbols):
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        s = os.path.basename(p)[:-4]
        if (symbols and s not in symbols) or (not symbols and s in CONTEXT):
            continue
        out[s] = pd.read_csv(p, index_col=0, parse_dates=True)["close"].dropna().values
    return out


def windows(closes, w, stride):
    X, who = [], []
    for s, c in closes.items():
        for i in range(0, len(c) - w + 1, stride):
            x = np.log(c[i:i + w])
            x = x - x.mean()
            sd = x.std()
            if sd > 0:
                X.append(x / sd)
                who.append(s)
    return np.array(X), np.array(who)


def kmeans(X, k, seed=0, iters=60, n_init=3):
    best = None
    for r in range(n_init):
        rng = np.random.default_rng(seed + r)
        C = X[rng.integers(len(X))][None]
        for _ in range(1, k):                       # k-means++ start
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1).min(1)
            C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]


def describe(c):
    """plain-English label: what each third of the window did (merged)."""
    w = len(c)
    pts = [c[0], c[w // 3], c[2 * w // 3], c[-1]]
    rng = c.max() - c.min()
    moves = []
    for a, b in zip(pts, pts[1:]):
        d = b - a
        mv = "rise" if d > 0.25 * rng else ("fall" if d < -0.25 * rng else "flat")
        if not moves or moves[-1] != mv:
            moves.append(mv)
    if moves == ["flat"]:
        return "sideways"
    return ", then ".join(moves)


def plot(C, counts, w, total, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed -> no PNG)")
        return
    k = len(C)
    cols = 4
    rows = int(np.ceil(k / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.3))
    for i, a in enumerate(np.ravel(ax)):
        if i >= k:
            a.axis("off")
            continue
        a.plot(C[i], lw=2.2, color="#2a6f97")
        a.set_title(f"#{i + 1}  {counts[i]:,}  ({counts[i] / total:.1%})\n{describe(C[i])}",
                    fontsize=8.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Most common {w}-hour shapes ({total:,} windows)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  picture: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, nargs="+", default=[14, 35])
    ap.add_argument("--k", type=int, default=12, help="number of shapes per window length")
    ap.add_argument("--stride", type=int, default=0, help="step between windows (0 = W)")
    ap.add_argument("--symbols", nargs="*", default=None)
    a = ap.parse_args()
    closes = load(set(a.symbols) if a.symbols else None)
    if not closes:
        raise SystemExit(f"  no CSVs in {DATA} -- run scripts/fetch_commod_rth.py first")
    print(f"  symbols: {', '.join(closes)}")
    os.makedirs(OUT, exist_ok=True)
    for w in a.windows:
        X, who = windows(closes, w, a.stride or w)
        C, lab = kmeans(X, a.k)
        counts = np.bincount(lab, minlength=a.k)
        order = np.argsort(-counts)
        C, counts = C[order], counts[order]
        remap = {old: new for new, old in enumerate(order)}
        lab = np.array([remap[x] for x in lab])
        print(f"\n  ===== {w}-hour windows: {len(X):,} windows -> {a.k} shapes")
        print(f"  {'#':>3}  {'count':>7}  {'share':>6}  shape                         most common in")
        for i in range(a.k):
            top = pd.Series(who[lab == i]).value_counts().head(3)
            tops = ", ".join(f"{s} {n}" for s, n in top.items())
            print(f"  {i + 1:>3}  {counts[i]:>7,}  {counts[i] / len(X):>6.1%}  "
                  f"{describe(C[i]):<29} {tops}")
        plot(C, counts, w, len(X), os.path.join(OUT, f"shapes_{w}.png"))


if __name__ == "__main__":
    main()
