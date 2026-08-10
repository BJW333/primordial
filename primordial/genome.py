"""
genome.py -- what a candidate IS, and the backtest that judges one.

A Genome is a full strategy specification: timeframe, entry tree, regime tree,
direction, entry style, exit parameters, holding horizon. Everywhere a human
default used to sit is a gene.

The backtest here supersedes edgesearch's:
  - signal at close[t], FILL at open[t+1] (never same-bar)
  - stops that gap through fill at the open, not the stop price
  - equity is MARKED TO MARKET every bar (edgesearch marked only at exits,
    understating max drawdown -- fixed here because drawdown is a first-class
    objective)
  - carry (funding/borrow) charged per bar held, from the CostModel
  - entry styles: market at next open; limit at k*ATR (models non-fills);
    confirm(n) -- signal must persist n bars
"""
from __future__ import annotations
import copy
import dataclasses
import json
import random
import numpy as np
import pandas as pd

from . import tree as T

TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d", "1w"]
TRAIL_MODES = ["none", "breakeven_1r", "trail_1r"]
ENTRY_STYLES = ["market", "limit", "confirm"]
# "session" is deliberately NOT in ENTRY_STYLES: evolution must never pick it,
# because it is only sound when the entry tree uses open-knowable atoms.
# Reachable solely via make_genome --style session, which enforces the
# whitelist below. 2026-08-10.
OPEN_SAFE_TERMS = {"gap_atr_open", "open"}


@dataclasses.dataclass
class Genome:
    timeframe: str
    root_type: str                  # "bool" | "series"
    entry_tree: dict
    regime_tree: dict | None
    direction: str                  # "long" | "short" | "both"
    entry_style: str                # market | limit | confirm
    entry_param: float              # limit: k*ATR offset ; confirm: n bars
    stop_atr: float
    target_r: float                 # 0 = no target
    time_stop: int                  # bars; 0 = none
    trail_mode: str
    max_hold: int
    # exit-at-reference (deep-state family): 0 = off; else EMA span. Exit at
    # the CLOSE of the first bar (after the entry bar) where close crosses the
    # reference EMA in the profit direction. Stops/target/time still apply --
    # the search decides whether they help (on mean reversion they should not:
    # adding a stop back onto the anchor exit cost Sharpe 1.27 -> 0.92).
    anchor_ema: int = 0

    # ---- serialization (runtime.py depends on this being complete) -------
    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        return cls(**json.loads(s))

    def complexity(self) -> int:
        c = T.size(self.entry_tree)
        if self.regime_tree is not None:
            c += T.size(self.regime_tree)
        return c

    def describe(self) -> str:
        d = f"[{self.timeframe} {self.direction} {self.entry_style}] {T.describe(self.entry_tree)}"
        if self.regime_tree is not None:
            d += f" WHEN {T.describe(self.regime_tree)}"
        d += (f" | stop {self.stop_atr:.1f}A tgt {self.target_r:.1f}R "
              f"ts {self.time_stop} {self.trail_mode}")
        if self.anchor_ema:
            d += f" anchor{self.anchor_ema}"
        return d

    # ---- signals ---------------------------------------------------------
    def signal(self, df: pd.DataFrame) -> np.ndarray:
        s = T.bool_signal(self.entry_tree, df)
        if self.regime_tree is not None:
            s = s * T.bool_signal(self.regime_tree, df)
        if self.entry_style == "confirm":
            n = max(1, int(self.entry_param))
            s = (pd.Series(s).rolling(n, min_periods=n).min()).fillna(0).to_numpy()
        return s

    def score(self, df: pd.DataFrame) -> pd.Series:
        v = T.series_score(self.entry_tree, df)
        if self.regime_tree is not None:
            g = T.bool_signal(self.regime_tree, df)
            v = v.where(pd.Series(g, index=df.index) > 0)
        return v


# ---- random construction / variation -------------------------------------
def random_genome(rng: random.Random, terminals, allowed_timeframes,
                  root_type="bool", allow_short=True) -> Genome:
    direction = rng.choice(["long"] + (["short"] if allow_short else []))
    return Genome(
        timeframe=rng.choice(allowed_timeframes),
        root_type=root_type,
        entry_tree=T.random_tree(rng, terminals,
                                 want=T.B if root_type == "bool" else T.S),
        regime_tree=(T.random_tree(rng, terminals, want=T.B)
                     if rng.random() < 0.4 else None),
        direction=direction,
        entry_style=rng.choice(ENTRY_STYLES),
        entry_param=rng.choice([0.25, 0.5, 1.0]) if rng.random() < 0.5
                    else rng.choice([1, 2, 3]),
        stop_atr=round(rng.uniform(0.5, 4.0), 2),
        target_r=round(rng.choice([0.0, 1.0, 1.5, 2.0, 3.0]), 2),
        time_stop=rng.choice([0, 12, 24, 48, 96]),
        trail_mode=rng.choice(TRAIL_MODES),
        max_hold=rng.choice([48, 96, 192, 500]),
        anchor_ema=rng.choice([0, 0, 0, 0, 13, 26, 52]),
    )


def mutate_genome(g: Genome, rng: random.Random, terminals,
                  allowed_timeframes) -> Genome:
    g = copy.deepcopy(g)
    r = rng.random()
    if r < 0.45:
        g.entry_tree = T.mutate(g.entry_tree, rng, terminals)
    elif r < 0.60:
        if g.regime_tree is None:
            g.regime_tree = T.random_tree(rng, terminals, want=T.B,
                                          depth=T.MAX_DEPTH - 2)
        elif rng.random() < 0.3:
            g.regime_tree = None
        else:
            g.regime_tree = T.mutate(g.regime_tree, rng, terminals)
    elif r < 0.72:
        # timeframe shifts to an ADJACENT allowed one only: a structure that
        # half-works at 30m often works at 1h; 5m->4h is a new random candidate
        tfs = [t for t in TIMEFRAMES if t in allowed_timeframes]
        i = tfs.index(g.timeframe)
        g.timeframe = tfs[max(0, min(len(tfs) - 1, i + rng.choice([-1, 1])))]
    elif r < 0.90:
        which = rng.choice(["stop", "target", "ts", "trail", "hold", "style",
                            "anchor"])
        if which == "stop":   g.stop_atr = float(np.clip(g.stop_atr + rng.uniform(-0.5, 0.5), 0.3, 6.0))
        elif which == "target": g.target_r = rng.choice([0.0, 1.0, 1.5, 2.0, 3.0])
        elif which == "ts":   g.time_stop = rng.choice([0, 12, 24, 48, 96])
        elif which == "trail": g.trail_mode = rng.choice(TRAIL_MODES)
        elif which == "hold": g.max_hold = rng.choice([48, 96, 192, 500])
        elif which == "anchor": g.anchor_ema = rng.choice([0, 13, 26, 39, 52])
        else:
            g.entry_style = rng.choice(ENTRY_STYLES)
            g.entry_param = rng.choice([0.25, 0.5, 1.0, 1, 2, 3])
    else:
        g.direction = "short" if g.direction == "long" else "long"
    return g


def crossover_genomes(a: Genome, b: Genome, rng: random.Random):
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    a2.entry_tree, b2.entry_tree = T.crossover(a2.entry_tree, b2.entry_tree, rng)
    if rng.random() < 0.5:
        a2.regime_tree, b2.regime_tree = b2.regime_tree, a2.regime_tree
    if rng.random() < 0.5:
        for f in ("stop_atr", "target_r", "time_stop", "trail_mode", "max_hold",
                  "anchor_ema"):
            setattr(a2, f, getattr(b, f))
            setattr(b2, f, getattr(a, f))
    return a2, b2


# ---- the backtest ---------------------------------------------------------
@dataclasses.dataclass
class BTResult:
    trades: list                    # (entry_t, exit_t, side, r_mult)
    equity: pd.Series               # MARK-TO-MARKET, every bar
    n_trades: int
    turnover: float                 # entries per 100 bars


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=5).mean().to_numpy()


def _assert_open_safe(g: "Genome"):
    """session mode fills at TODAY's open; every terminal must be knowable
    at 09:30. Enforced HERE (not only in make_genome) so hand-edited JSON,
    future tooling, or any other path cannot smuggle look-ahead in."""
    bad = set()

    def walk(nd):
        if nd["op"] == "term" and nd["name"] not in OPEN_SAFE_TERMS:
            bad.add(nd["name"])
        for ch in nd.get("ch", []):
            walk(ch)
    walk(g.entry_tree)
    if g.regime_tree is not None:
        walk(g.regime_tree)
    if bad:
        raise ValueError(
            f"entry_style='session' with non-open-safe atoms {sorted(bad)}; "
            f"allowed: {sorted(OPEN_SAFE_TERMS)}")


def run_backtest(g: Genome, df: pd.DataFrame, cost, risk_per_trade=0.01,
                 bar_seconds=3600.0) -> BTResult:
    if g.entry_style == "session":
        _assert_open_safe(g)
    sig = g.signal(df)
    atr = _atr(df)
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    idx = df.index
    n = len(df)
    adverse = cost.adverse_frac()
    carry = cost.carry_frac_per_bar(bar_seconds)

    anch = (df["close"].ewm(span=g.anchor_ema, min_periods=5).mean().to_numpy()
            if g.anchor_ema else None)
    equity = 1.0
    eq = np.full(n, np.nan)
    trades = []
    pos = None      # dict: side, entry, stop, r, entered_i, target
    order = None    # working limit: {side, lim, r, placed}
    LIMIT_TTL = 3   # bars a limit order stays working before it expires
    warm = 20

    for t in range(warm, n):
        # ------------- SAME-SESSION mode: enter at open[t], exit at close[t]
        # Legal only because the signal is read at bar t from OPEN-SAFE atoms
        # (OPEN_SAFE_TERMS): everything in the tree is knowable at 09:30.
        # Both prices are real and separated in time, so this does NOT hand
        # the strategy the bar's own range the way a same-bar limit+target
        # would -- that remains forbidden.
        if g.entry_style == "session":
            if sig[t] != 0 and o[t] > 0 and np.isfinite(c[t]):
                side = g.direction
                fill = o[t] * (1 + adverse) if side == "long" \
                    else o[t] * (1 - adverse)
                ex = c[t] * (1 - adverse) if side == "long" \
                    else c[t] * (1 + adverse)
                sgn = 1.0 if side == "long" else -1.0
                ret = sgn * (ex - fill) / fill
                equity *= (1.0 + ret)
                denom = (atr[t - 1] / fill) if (np.isfinite(atr[t - 1])
                                                and fill > 0) else np.nan
                trades.append({"i": t, "bars": 1, "ret": ret,
                               "r": (ret / denom) if denom and denom > 0
                               else 0.0})
            eq[t] = equity
            continue

        # ---------------- entry: decided at close[t-1], filled at bar t ----
        if pos is None:
            side = g.direction        # bool signal is 0/1; the DIRECTION gene
            fill = None               # decides which way a fire trades
            fired = (sig[t - 1] != 0 and np.isfinite(atr[t - 1])
                     and atr[t - 1] > 0)
            if g.entry_style == "limit":
                if fired:             # fresh signal places/replaces the order
                    k = float(g.entry_param)
                    order = {"side": side,
                             "lim": (c[t - 1] - k * atr[t - 1] if side == "long"
                                     else c[t - 1] + k * atr[t - 1]),
                             "r": g.stop_atr * atr[t - 1], "placed": t}
                if order is not None:
                    if t - order["placed"] >= LIMIT_TTL:
                        order = None          # expired unfilled
                    # honest non-fill: the bar must trade THROUGH the limit
                    elif order["side"] == "long" and l[t] <= order["lim"]:
                        fill = min(order["lim"], o[t])
                    elif order["side"] == "short" and h[t] >= order["lim"]:
                        fill = max(order["lim"], o[t])
            elif fired:
                fill = o[t]
            if fill is not None and fill > 0:
                if g.entry_style == "limit":
                    side = order["side"]
                    r_price = order["r"]
                    order = None
                else:
                    r_price = g.stop_atr * atr[t - 1]
                fill = fill * (1 + adverse) if side == "long" else fill * (1 - adverse)
                stop = fill - r_price if side == "long" else fill + r_price
                pos = {"side": side, "entry": fill, "stop": stop, "r": r_price,
                       "i": t, "t0": idx[t],
                       "target": (fill + g.target_r * r_price if side == "long"
                                  else fill - g.target_r * r_price)
                                 if g.target_r > 0 else None}

        # ---------------- manage ------------------------------------------
        # PROFIT NEVER ON THE ENTRY BAR. A limit fill lands near bar t's low
        # (long) / high (short); checking the SAME bar's range for the target
        # hands the strategy the bar's own span as free profit -- an
        # always-true entry scored Sharpe +6.5 long AND short on real data,
        # and +4.6/+3.5 on a pure random walk, purely from this. Intrabar
        # ordering is unknowable in OHLC, so the honest rules are:
        #   entry bar: only the STOP can trigger (losses count immediately),
        #   later bars: stop is checked BEFORE target, so a bar touching both
        #   resolves as a loss.
        if pos is not None and pos["i"] == t:
            sgn = 1 if pos["side"] == "long" else -1
            if (pos["side"] == "long" and l[t] <= pos["stop"]) or \
               (pos["side"] == "short" and h[t] >= pos["stop"]):
                exit_px = pos["stop"]
                exit_px = exit_px * (1 - adverse) if pos["side"] == "long" \
                          else exit_px * (1 + adverse)
                r_mult = sgn * (exit_px - pos["entry"]) / pos["r"]
                equity += equity * risk_per_trade * r_mult
                trades.append((pos["t0"], idx[t], pos["side"], float(r_mult)))
                pos = None
        elif pos is not None:
            side = pos["side"]; sgn = 1 if side == "long" else -1
            exit_px = None
            # stop (gap-through: fill at open if it opened beyond)
            if side == "long" and l[t] <= pos["stop"]:
                exit_px = min(pos["stop"], o[t])
            elif side == "short" and h[t] >= pos["stop"]:
                exit_px = max(pos["stop"], o[t])
            # anchor exit (deep-state): thesis complete when close is back
            # at the reference EMA. Fills at the CLOSE -- no intrabar claim --
            # and never on the entry bar (entry-bar block above allows stop
            # only, per the range-harvesting lesson).
            elif anch is not None and np.isfinite(anch[t]) and (
                    (side == "long" and c[t] >= anch[t]) or
                    (side == "short" and c[t] <= anch[t])):
                exit_px = c[t]
            # target (limit: never fills worse)
            elif pos["target"] is not None and (
                    (side == "long" and h[t] >= pos["target"]) or
                    (side == "short" and l[t] <= pos["target"])):
                exit_px = pos["target"]
            # time stop / max hold
            elif (g.time_stop and t - pos["i"] >= g.time_stop) or \
                 (t - pos["i"] >= g.max_hold):
                exit_px = c[t]
            # trailing
            if exit_px is None and g.trail_mode != "none":
                prof_r = sgn * (c[t] - pos["entry"]) / pos["r"]
                if prof_r >= 1.0:
                    if g.trail_mode == "breakeven_1r":
                        pos["stop"] = (max(pos["stop"], pos["entry"]) if side == "long"
                                       else min(pos["stop"], pos["entry"]))
                    else:  # trail_1r
                        new = c[t] - sgn * pos["r"]
                        pos["stop"] = (max(pos["stop"], new) if side == "long"
                                       else min(pos["stop"], new))
            # (carry is charged at exit, held-bars * carry, in R units)
            if exit_px is not None:
                exit_px = exit_px * (1 - adverse) if side == "long" else exit_px * (1 + adverse)
                r_mult = sgn * (exit_px - pos["entry"]) / pos["r"]
                held = t - pos["i"]
                r_mult -= carry * held * (pos["entry"] / pos["r"])
                equity += equity * risk_per_trade * r_mult
                trades.append((pos["t0"], idx[t], side, float(r_mult)))
                pos = None
        # ---------------- mark to market ----------------------------------
        if pos is not None:
            sgn = 1 if pos["side"] == "long" else -1
            open_r = sgn * (c[t] - pos["entry"]) / pos["r"]
            eq[t] = equity * (1 + risk_per_trade * open_r)
        else:
            eq[t] = equity

    eqs = pd.Series(eq, index=idx).dropna()
    return BTResult(trades=trades, equity=eqs, n_trades=len(trades),
                    turnover=100.0 * len(trades) / max(n - warm, 1))
