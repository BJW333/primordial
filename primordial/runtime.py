"""
runtime.py -- load a survivor artifact and emit signals live.

Deliberately imports ONLY tree/genome/atoms/motifs (pure evaluation) -- no
engine, no pipeline, no judge. This is the export path into TRIAD-style
deployment: feed bars, get {-1, 0, +1} plus exit parameters.

    strat = Strategy.load("survivor_0.json")
    sig = strat.signal(df_ohlcv)          # df at the survivor's timeframe
    print(strat.exit_spec())              # stop/target/trail for the executor
"""
from __future__ import annotations
import json
import numpy as np

from .genome import Genome
from . import atoms as A
from . import motifs as M


class Strategy:
    def __init__(self, art: dict):
        self.genome = Genome(**art["genome"])
        self.motifs = art.get("motifs", {})
        self.atom_blocks = art["atom_blocks"]
        self.always_open = art["always_open"]
        self.validation = art.get("validation", {})
        self.timeframe = self.genome.timeframe

    @classmethod
    def load(cls, path: str) -> "Strategy":
        with open(path) as f:
            art = json.load(f)
        blob = json.dumps(art.get("genome", {}))
        if any(k in blob for k in ("xs_rank_", "rel_ret", "ctx_")):
            raise ValueError(
                "This survivor conditions on CROSS-SECTIONAL atoms "
                "(xs_rank_*/rel_ret/ctx_*), which need universe or benchmark data every "
                "bar. Single-name runtime cannot compute them -- deploy it "
                "through a panel runner that calls "
                "atoms.add_cross_sectional() across all names first.")
        return cls(art)

    def prepare(self, df):
        """Compute exactly the atoms this strategy's trees may reference.
        Motif centroids come FROZEN from the artifact -- never re-mined."""
        A.compute_atoms(df, self.atom_blocks, self.always_open)
        if self.motifs:
            M.add_motif_atoms({"_": df}, self.motifs)
        return df

    def signal(self, df) -> np.ndarray:
        """0/1 fire per bar; trade in self.genome.direction. Caller fills at
        NEXT bar open -- same convention the strategy was validated under."""
        return self.genome.signal(self.prepare(df.copy()))

    def exit_spec(self) -> dict:
        g = self.genome
        return {"direction": g.direction, "stop_atr": g.stop_atr,
                "target_r": g.target_r, "time_stop": g.time_stop,
                "trail_mode": g.trail_mode, "max_hold": g.max_hold,
                "entry_style": g.entry_style, "entry_param": g.entry_param,
                # 0 = off; else EMA span. Executor must exit at the CLOSE of
                # the first bar (after entry) where close crosses this EMA in
                # the profit direction -- it is the PRIMARY exit for
                # deep-state-family survivors; omitting it changes the
                # strategy being traded.
                "anchor_ema": getattr(g, "anchor_ema", 0),
                # 0 = off; else exit at entry*(1 +/- pct), level fill.
                "target_pct": getattr(g, "target_pct", 0.0)}
