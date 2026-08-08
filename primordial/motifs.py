"""
motifs.py -- pattern_scanner's discovery mode, refactored into an atom
generator. Shapes become VOCABULARY, not answers.

Mine recurring z-normalized window shapes from TRAIN data ONLY (the holdout
has never been clustered -- pipeline enforces and tests this). Each motif
becomes a similarity SERIES the GP can build conditions on:

    match_m3_s40[t] = correlation between the trailing 40-bar z-normed window
                      and motif #3

Motif centroids are FROZEN into any survivor that references them (stored in
the genome artifact) -- same bug class as edgesearch's library-key reuse:
an index that means something different at validation time silently
invalidates the result.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _znorm_windows(closes: np.ndarray, scale: int, step: int):
    wins = []
    for i in range(0, len(closes) - scale, step):
        w = closes[i:i + scale]
        sd = w.std()
        if sd > 0:
            wins.append((w - w.mean()) / sd)
    return np.array(wins) if wins else np.empty((0, scale))


def _kmeans(X: np.ndarray, k: int, seed=0, iters=25):
    """Small numpy k-means -- no sklearn dependency in the deploy path."""
    rng = np.random.default_rng(seed)
    if len(X) < k:
        return X.copy()
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def mine(train_data: dict, syms: list, scales, k_per_scale: int, seed=0) -> dict:
    """Returns {"m{i}_s{scale}": centroid_list} -- JSON-serializable so the
    motif set travels inside run artifacts and survivor exports."""
    motifs = {}
    for scale in scales:
        allw = []
        for s in syms:
            c = train_data[s]["close"].to_numpy(dtype=float)
            allw.append(_znorm_windows(c, scale, step=max(1, scale // 4)))
        X = np.vstack([w for w in allw if len(w)]) if allw else np.empty((0, scale))
        if len(X) < k_per_scale * 3:
            continue
        C = _kmeans(X, k_per_scale, seed=seed)
        for i, cent in enumerate(C):
            motifs[f"m{i}_s{scale}"] = cent.tolist()
    return motifs


def match_series(close: pd.Series, centroid: np.ndarray) -> pd.Series:
    """Rolling correlation of the trailing z-normed window vs the centroid.
    Causal: window ends at t."""
    scale = len(centroid)
    c = close.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    cent = np.asarray(centroid)
    cent = (cent - cent.mean()) / (cent.std() or 1.0)
    for t in range(scale, len(c)):
        w = c[t - scale:t]
        sd = w.std()
        if sd > 0:
            wz = (w - w.mean()) / sd
            out[t] = float(np.dot(wz, cent) / scale)
    return pd.Series(out, index=close.index)


def add_motif_atoms(data: dict, motifs: dict) -> list:
    """Add match_* columns to every symbol frame. Returns column names."""
    cols = []
    for name, cent in motifs.items():
        col = f"match_{name}"
        cent = np.asarray(cent)
        for s in data:
            data[s][col] = match_series(data[s]["close"], cent)
        cols.append(col)
    return cols
