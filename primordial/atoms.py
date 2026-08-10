"""
Tier-1 measurement atoms.

Every atom MEASURES; none INSTRUCTS (no thresholds, no buy/sell content).
Every atom is a SERIES exposed scale-free -- z-scored, percentile-ranked, or in
sigma/ATR units -- so one tree means the same thing on BTC at 60k and a $4
small-cap.

compute_atoms(df, blocks) adds columns in place and returns the column names.
Which blocks are valid for a manifest is the AssetClassAdapter's decision;
unavailable atoms are DROPPED from the terminal set, never zero-filled (a
constant-zero atom is a free constant the GP will abuse).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FAST, MED, SLOW = 10, 30, 100        # coarse spine; tree ops add continuous n


def _z(s, n=100):
    r = s.rolling(n, min_periods=max(5, n // 4))
    sd = r.std()
    return (s - r.mean()) / sd.where(sd > 0)


def _pct(s, n=200):
    return s.rolling(n, min_periods=max(5, n // 4)).rank(pct=True)


# ---- block implementations ------------------------------------------------
def _trend(df, out):
    c = df["close"]
    for n, tag in ((FAST, "f"), (MED, "m"), (SLOW, "s")):
        out[f"dist_sma_{tag}"] = c / c.rolling(n, min_periods=5).mean() - 1
        out[f"roc_{tag}"] = c.pct_change(n)
        e = c.ewm(span=n, min_periods=5).mean()
        out[f"ema_slope_{tag}"] = e.pct_change(max(2, n // 5))
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    out["macd_line"] = _z(macd)
    out["macd_hist"] = _z(macd - macd.ewm(span=9).mean())
    # linreg slope, med window, in sigma units
    n = MED
    x = np.arange(n)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    slope = c.rolling(n).apply(lambda w: float(((w - w.mean()) * xc).sum() / denom),
                               raw=True)
    out["linreg_slope"] = _z(slope)
    out["price_pctile"] = _pct(c)
    # deep-state primitives (validated Jul-Aug 2026, two-universe gates):
    # ema_spread_atr -- the 12/26 EMA spread in ATR units, SIGN-PRESERVING
    # (macd_line above is z-scored, which destroys the below-zero semantics).
    # depth of a sag = -ema_spread_atr. spread_runlen -- bars the current
    # sag has lasted (0 whenever spread >= 0); fresh sags revert, stale ones
    # are falling knives (fresh<=5d +1.10%/tr vs stale>15d +0.10%/tr).
    h_, l_ = df["high"], df["low"]
    pc_ = c.shift()
    tr_ = pd.concat([h_ - l_, (h_ - pc_).abs(), (l_ - pc_).abs()],
                    axis=1).max(axis=1)
    atr_ = tr_.rolling(14, min_periods=5).mean()
    spread = c.ewm(span=12, min_periods=5).mean() - c.ewm(span=26,
                                                          min_periods=5).mean()
    out["ema_spread_atr"] = spread / atr_.where(atr_ > 0)
    neg = spread < 0
    run = (neg.groupby((~neg).cumsum()).cumcount()
           + neg.astype(int) - 1).clip(lower=0)
    out["spread_runlen"] = run.astype(float)


def _oscillators(df, out):
    c, h, l = df["close"], df["high"], df["low"]
    for n, tag in ((FAST, "f"), (MED, "m")):
        delta = c.diff()
        up = delta.clip(lower=0).rolling(n, min_periods=3).mean()
        dn = (-delta.clip(upper=0)).rolling(n, min_periods=3).mean()
        out[f"rsi_{tag}"] = (100 - 100 / (1 + up / dn.where(dn > 0))) / 100.0
        hh = h.rolling(n, min_periods=3).max()
        ll = l.rolling(n, min_periods=3).min()
        out[f"stoch_k_{tag}"] = (c - ll) / (hh - ll).where(hh > ll)
    out["stoch_d"] = out["stoch_k_m"].rolling(3).mean()
    tp = (h + l + c) / 3
    md = tp.rolling(MED).apply(lambda w: float(np.abs(w - w.mean()).mean()), raw=True)
    out["cci"] = _z((tp - tp.rolling(MED).mean()) / (0.015 * md.where(md > 0)))
    out["bb_z_m"] = _z(c, MED)
    out["bb_z_s"] = _z(c, SLOW)


def _vol_structure(df, out):
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    r = c.pct_change()
    for n, tag in ((FAST, "f"), (MED, "m"), (SLOW, "s")):
        out[f"rvol_{tag}"] = r.rolling(n, min_periods=5).std()
    out["vol_ratio"] = out["rvol_f"] / out["rvol_s"].where(out["rvol_s"] > 0)
    hl = np.log(h / l.where(l > 0))
    out["parkinson"] = np.sqrt((hl ** 2).rolling(MED, min_periods=5).mean()
                               / (4 * np.log(2)))
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * np.log(c / o.where(o > 0)) ** 2
    out["garman_klass"] = np.sqrt(gk.rolling(MED, min_periods=5).mean().clip(lower=0))
    out["vol_of_vol"] = out["rvol_m"].rolling(MED, min_periods=5).std()
    out["vol_pctile"] = _pct(out["rvol_m"])
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    out["atr_pct"] = atr / c
    bw = out["rvol_m"] * 4
    out["bb_width_pctile"] = _pct(bw)


def _distribution(df, out):
    r = df["close"].pct_change()
    out["skew"] = r.rolling(SLOW, min_periods=20).skew()
    out["kurt"] = r.rolling(SLOW, min_periods=20).kurt()
    out["up_bar_frac"] = (r > 0).rolling(MED, min_periods=5).mean()
    up = r.clip(lower=0).rolling(MED, min_periods=5).mean()
    dn = (-r.clip(upper=0)).rolling(MED, min_periods=5).mean()
    out["up_down_ratio"] = _z(up / dn.where(dn > 0))
    eq = (1 + r.fillna(0)).cumprod()
    roll_max = eq.rolling(SLOW, min_periods=10).max()
    out["max_dd_window"] = eq / roll_max - 1
    out["ret_pctile"] = _pct(r)


def _persistence(df, out):
    r = df["close"].pct_change()
    for lagn in (1, 5):
        out[f"autocorr_{lagn}"] = r.rolling(SLOW, min_periods=30).apply(
            lambda w: float(pd.Series(w).autocorr(lagn)) if len(w) > lagn + 2 else np.nan,
            raw=False)
    # variance ratio: var of k-period returns vs k * var of 1-period
    k = 5
    rk = df["close"].pct_change(k)
    out["variance_ratio"] = (rk.rolling(SLOW, min_periods=30).var()
                             / (k * r.rolling(SLOW, min_periods=30).var()).where(
                                 lambda s: s > 0))
    net = df["close"].diff(MED).abs()
    path = df["close"].diff().abs().rolling(MED, min_periods=5).sum()
    out["efficiency_ratio"] = net / path.where(path > 0)


def _microstructure(df, out):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l)
    out["close_loc"] = (c - l) / rng.where(rng > 0)
    out["body_frac"] = (c - o).abs() / rng.where(rng > 0)
    out["upper_wick_frac"] = (h - np.maximum(o, c)) / rng.where(rng > 0)
    out["lower_wick_frac"] = (np.minimum(o, c) - l) / rng.where(rng > 0)
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    out["gap_atr"] = (o - pc) / atr.where(atr > 0)
    # OPEN-SAFE variant: today's gap scaled by YESTERDAY's ATR. gap_atr above
    # divides by atr[t], which contains today's high/low -- fine for a signal
    # read at the close, LOOK-AHEAD for anything filled at today's open.
    # gap_atr_open uses only o[t], c[t-1] and bars <= t-1, so it is knowable
    # at 09:30. Required for entry_style="session".
    _atr_prev = atr.shift(1)
    out["gap_atr_open"] = (o - pc) / _atr_prev.where(_atr_prev > 0)
    out["true_range_z"] = _z(tr)


def _volume(df, out):
    c, v = df["close"], df["volume"]
    r = c.pct_change()
    obv = (np.sign(r.fillna(0)) * v).cumsum()
    out["obv_slope"] = _z(obv.diff(MED))
    out["volz"] = _z(v)
    out["turnover_ratio"] = v / v.rolling(SLOW, min_periods=10).mean().where(lambda s: s > 0)
    dv = (c * v)
    out["amihud"] = _z((r.abs() / dv.where(dv > 0)).rolling(MED, min_periods=5).mean())
    out["vol_price_corr"] = r.rolling(SLOW, min_periods=20).corr(v.pct_change())


def _session(df, out, always_open: bool):
    """Calendar + overnight split. Overnight atoms only where a session close
    exists -- on 24/7 venues they are meaningless and are NOT emitted."""
    idx = df.index
    out["hod"] = idx.hour.astype(float)
    out["dow"] = idx.dayofweek.astype(float)
    out["dom"] = idx.day.astype(float)
    out["dist_to_weekend"] = (5 - idx.dayofweek).clip(0).astype(float)
    if not always_open:
        pc = df["close"].shift()
        out["overnight_ret"] = df["open"] / pc.where(pc > 0) - 1
        out["intraday_ret"] = df["close"] / df["open"].where(df["open"] > 0) - 1


BLOCKS = {
    "trend": _trend,
    "oscillators": _oscillators,
    "vol_structure": _vol_structure,
    "distribution": _distribution,
    "persistence": _persistence,
    "microstructure": _microstructure,
    "volume": _volume,
    # "session" handled specially (needs adapter flag)
}


def compute_atoms(df: pd.DataFrame, blocks: list, always_open: bool) -> list:
    """Compute the requested blocks in place; return new column names."""
    out = {}
    for b in blocks:
        if b == "session":
            _session(df, out, always_open)
        elif b in BLOCKS:
            BLOCKS[b](df, out)
    cols = []
    for k, s in out.items():
        df[k] = pd.Series(s, index=df.index).replace([np.inf, -np.inf], np.nan)
        cols.append(k)
    return cols


def add_cross_sectional(data: dict, syms: list, base_cols=("roc_m", "rvol_m")) -> list:
    """Cross-sectional ranks computed WITHIN the given symbol set only --
    call separately on train and holdout name sets so train features never
    embed holdout names (§4.2 leakage note)."""
    added = []
    for col in base_cols:
        panel = pd.DataFrame({s: data[s][col] for s in syms if col in data[s]})
        ranks = panel.rank(axis=1, pct=True)
        name = f"xs_rank_{col}"
        for s in panel.columns:
            data[s][name] = ranks[s].reindex(data[s].index)
        added.append(name)
    # relative return vs universe median
    rets = pd.DataFrame({s: data[s]["close"].pct_change(MED) for s in syms})
    med = rets.median(axis=1)
    for s in syms:
        data[s]["rel_ret"] = (rets[s] - med).reindex(data[s].index)
    added.append("rel_ret")
    return added


def dedupe(data, cols: list, thresh: float = 0.95, max_syms: int = 5):
    """Collapse atom clusters with |rho| > thresh to one representative
    (first by column order = cheapest/most interpretable first).
    Accepts one DataFrame OR a {sym: df} dict -- with a dict, correlation is
    measured on data CONCATENATED across up to max_syms names, so "duplicate"
    is decided universe-wide, not on one representative ticker.
    Constant columns (zero variance) are dropped outright: a constant atom is
    a free constant for the GP (e.g. hour-of-day on daily bars).
    Returns (kept, merges) where merges maps dropped -> representative."""
    if isinstance(data, dict):
        frames = [d[[c for c in cols if c in d.columns]].dropna()
                  for d in list(data.values())[:max_syms]]
        sub = pd.concat(frames, axis=0, ignore_index=True)
    else:
        sub = data[cols].dropna()
    if len(sub) < 50:
        return cols, {}
    variances = sub.var()
    dead = [c for c in cols if not np.isfinite(variances.get(c, np.nan))
            or variances[c] < 1e-12]
    cols = [c for c in cols if c not in dead]
    sub = sub[cols]
    corr = sub.corr().abs()
    kept, merges = [], {c: "<constant>" for c in dead}
    for c in cols:
        rep = None
        for k in kept:
            if corr.loc[c, k] > thresh:
                rep = k
                break
        if rep is None:
            kept.append(c)
        else:
            merges[c] = rep
    return kept, merges


def add_context(data: dict, syms: list, benchmarks: dict | None = None) -> list:
    """Market-context columns broadcast to every name -- the vocabulary for
    CONDITIONAL behavior ("short EWY strength only when SPY vol is elevated").

    Universe context (computed WITHIN the given name-set, like
    add_cross_sectional, so train context never embeds holdout names):
      ctx_breadth      frac of names above their own 100-bar SMA
      ctx_breadth_d20  20-bar change in breadth
      ctx_disp         cross-name std of 20-bar returns (dispersion regime)
      ctx_vol          z-scored median of names' rolling vol (vol regime)

    Benchmark context (external reference series, e.g. SPY/TLT/GLD -- their
    own frames passed in; they are data streams, not universe members):
      ctx_<B>_trend    benchmark distance from its 100-bar SMA, z-scored
      ctx_<B>_volz     z-scored benchmark rolling vol
      ctx_<B>_roc      benchmark 30-bar return
    """
    added = []
    frames = {s: data[s] for s in syms if s in data}
    if len(frames) >= 5:
        above = pd.DataFrame({
            s: (df["close"] > df["close"].rolling(100, min_periods=30).mean())
            .astype(float) for s, df in frames.items()})
        breadth = above.mean(axis=1)
        rets = pd.DataFrame({s: df["close"].pct_change(20)
                             for s, df in frames.items()})
        disp = rets.std(axis=1)
        vols = pd.DataFrame({s: df["close"].pct_change()
                             .rolling(MED, min_periods=10).std()
                             for s, df in frames.items()})
        vmed = _z(vols.median(axis=1))
        ctx = {"ctx_breadth": breadth,
               "ctx_breadth_d20": breadth.diff(20),
               "ctx_disp": _z(disp),
               "ctx_vol": vmed}
        for k, s_ in ctx.items():
            for s in frames:
                data[s][k] = s_.reindex(data[s].index)
            added.append(k)
    for bname, bdf in (benchmarks or {}).items():
        c = bdf["close"]
        trend = _z(c / c.rolling(100, min_periods=30).mean() - 1)
        volz = _z(c.pct_change().rolling(MED, min_periods=10).std())
        roc = c.pct_change(MED)
        for s in frames:
            idx = data[s].index
            data[s][f"ctx_{bname}_trend"] = trend.reindex(idx)
            data[s][f"ctx_{bname}_volz"] = volz.reindex(idx)
            data[s][f"ctx_{bname}_roc"] = roc.reindex(idx)
        added += [f"ctx_{bname}_trend", f"ctx_{bname}_volz",
                  f"ctx_{bname}_roc"]
    return added
