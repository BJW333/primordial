"""
universe.py -- the pointer, the adapter layer, and the cost model.

A run is fully specified by a Manifest (YAML). The manifest's hash keys the
trial ledger and the archive, so trials on one universe never deflate another.

AssetClassAdapter carries every per-class concern from spec §3.4: session
shape, which atom blocks are valid, cost model, adjustment, engines allowed.
CostModel is an OBJECT passed down -- no module globals, so parallel islands
with different tiers cannot corrupt each other.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import yaml


# ---- cost model (object, not global) -------------------------------------
@dataclasses.dataclass(frozen=True)
class CostModel:
    name: str
    spread_bps: float          # half-spread paid per side
    commission_bps: float      # per side
    slippage_bps: float        # per side, adverse
    carry_bps_per_day: float = 0.0   # funding / borrow, charged on hold time
    stress_mult: float = 1.0

    def adverse_frac(self) -> float:
        per_side = (self.spread_bps + self.commission_bps + self.slippage_bps) / 1e4
        return per_side * self.stress_mult

    def carry_frac_per_bar(self, bar_seconds: float) -> float:
        return (self.carry_bps_per_day / 1e4) * (bar_seconds / 86400.0) * self.stress_mult

    def stressed(self, mult: float) -> "CostModel":
        return dataclasses.replace(self, stress_mult=mult)


COST_TIERS = {
    "equity_liquid":    CostModel("equity_liquid",    1.0, 0.0, 1.0),
    "equity_midcap":    CostModel("equity_midcap",    4.0, 0.0, 3.0),
    "equity_smallcap":  CostModel("equity_smallcap",  10.0, 0.0, 8.0),
    "index_etf":        CostModel("index_etf",        0.5, 0.0, 0.5),
    "crypto_spot_taker": CostModel("crypto_spot_taker", 5.0, 25.0, 5.0),
    "crypto_spot_maker": CostModel("crypto_spot_maker", 5.0, 15.0, 3.0),
    # Kalshi perps: fee schedule + FUNDING AS ONGOING CARRY. Omit the carry
    # and every perp backtest is optimistic in proportion to holding time --
    # evolution would discover "hold forever" as an edge.
    "kalshi_perp":      CostModel("kalshi_perp",      8.0, 10.0, 5.0,
                                  carry_bps_per_day=3.0),
}


# ---- asset-class adapters -------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AssetClassAdapter:
    name: str
    always_open: bool              # 24/7 venue -> no overnight atoms
    atom_blocks: tuple             # which Tier-1 blocks are meaningful
    engines: tuple                 # ("timing", "cross_sectional")
    needs_adjustment: bool         # splits/dividends
    shortable: bool
    external_streams: tuple = ()

    def valid_blocks(self) -> list:
        return list(self.atom_blocks) + ["session"]


ADAPTERS = {
    "equity": AssetClassAdapter(
        "equity", always_open=False,
        atom_blocks=("trend", "oscillators", "vol_structure", "distribution",
                     "persistence", "microstructure", "volume"),
        engines=("timing", "cross_sectional"),
        needs_adjustment=True, shortable=True),
    "index": AssetClassAdapter(
        # index composites/ETFs: global indexes universe. Shorting an index
        # composite directly isn't a thing -- run long/flat timing, or use
        # the ETF proxy tier if you want short.
        "index", always_open=False,
        atom_blocks=("trend", "oscillators", "vol_structure", "distribution",
                     "persistence", "microstructure"),
        engines=("timing", "cross_sectional"),
        needs_adjustment=False, shortable=False),
    "crypto": AssetClassAdapter(
        "crypto", always_open=True,
        atom_blocks=("trend", "oscillators", "vol_structure", "distribution",
                     "persistence", "microstructure", "volume"),
        engines=("timing", "cross_sectional"),
        needs_adjustment=False, shortable=False),
    "crypto_perp": AssetClassAdapter(
        "crypto_perp", always_open=True,
        atom_blocks=("trend", "oscillators", "vol_structure", "distribution",
                     "persistence", "microstructure", "volume"),
        engines=("timing",),            # too few contracts to rank
        needs_adjustment=False, shortable=True,
        external_streams=("funding_rate", "open_interest")),
}


# ---- symbol presets (resolvers) ------------------------------------------
# Point-in-time caveat: explicit lists are what they are. The liquid-universe
# resolvers for equities MUST include delisted names when you build them
# against a survivorship-free source; yfinance presets below are index
# composites, which sidestep single-name survivorship (an index is already
# point-in-time by construction).
PRESETS = {
    # global stock market indexes: USA, Europe, Asia (Korea, India, Japan, HK)
    "global_indexes": [
        "^GSPC", "^NDX", "^DJI", "^RUT",                    # USA
        "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^IBEX",   # Europe
        "^AEX", "^SSMI", "^OMX",                            # Europe
        "^N225", "^KS11", "^KQ11",                          # Japan, Korea
        "^BSESN", "^NSEI",                                  # India
        "^HSI", "^STI", "^AXJO", "^GSPTSE", "^BVSP",        # HK/SG/AU/CA/BR
    ],
    # TRADABLE proxies for the index result: country + regional + sector ETFs.
    # The ^ index tickers are not instruments -- any index-level finding must
    # be re-tested on these before it means anything implementable (timezone
    # mismatch vs foreign closes, FX, and tracking error all live here).
    "global_etfs_wide": [
        "SPY", "QQQ", "DIA", "IWM", "MDY",                   # US broad
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU",     # US sectors
        "XLB", "XLY", "XLC", "XLRE",
        "EWJ", "EWG", "EWU", "EWA", "EWC", "EWZ", "EWY",     # countries
        "EWT", "EWH", "EWS", "EWW", "EWL", "EWD", "EWN",
        "EWP", "EWI", "EWQ", "INDA", "FXI", "EZA", "EWM",
        "EFA", "EEM", "VGK", "EWX", "ILF", "AAXJ",           # regions
    ],
    # TRADABLE proxies for the index result: country + regional + sector ETFs.
    # The ^ index tickers are not instruments -- any index-level finding must
    # be re-tested on these before it means anything implementable (timezone
    # mismatch vs foreign closes, FX, and tracking error all live here).
    "global_etfs_wide": [
        "SPY", "QQQ", "DIA", "IWM", "MDY",                   # US broad
        "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLU",     # US sectors
        "XLB", "XLY", "XLC", "XLRE",
        "EWJ", "EWG", "EWU", "EWA", "EWC", "EWZ", "EWY",     # countries
        "EWT", "EWH", "EWS", "EWW", "EWL", "EWD", "EWN",
        "EWP", "EWI", "EWQ", "INDA", "FXI", "EZA", "EWM",
        "EFA", "EEM", "VGK", "EWX", "ILF", "AAXJ",           # regions
    ],
    "us_megacap": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                   "JPM", "XOM", "WMT", "KO", "DIS", "NFLX", "CRM", "ORCL",
                   "INTC", "CSCO", "PEP", "MCD", "AMD"],
    "coinbase_liquid": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD",
        "AVAX-USD", "LINK-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
        "UNI-USD", "ATOM-USD", "XLM-USD", "ETC-USD", "FIL-USD", "APT-USD",
        "ARB-USD", "OP-USD", "NEAR-USD", "INJ-USD", "AAVE-USD", "MKR-USD",
        "ALGO-USD", "SAND-USD", "MANA-USD", "GRT-USD", "IMX-USD", "RNDR-USD",
    ],
}


@dataclasses.dataclass
class Manifest:
    name: str
    asset_class: str               # key into ADAPTERS
    source: str                    # "yfinance" | "coinbase" | "alpaca" | "csv" | "synthetic"
    symbols: list
    base_timeframe: str
    allowed_timeframes: list
    start: str
    end: str
    cost_tier: str
    holdout_name_frac: float = 0.4
    holdout_time_frac: float = 0.3
    embargo_bars: int = 50
    engines: list = dataclasses.field(default_factory=lambda: ["timing"])
    n_motifs: int = 6
    motif_scales: list = dataclasses.field(default_factory=lambda: [20, 40])
    context_symbols: list = dataclasses.field(default_factory=list)
    seed: int = 42

    @property
    def adapter(self) -> AssetClassAdapter:
        return ADAPTERS[self.asset_class]

    @property
    def cost(self) -> CostModel:
        return COST_TIERS[self.cost_tier]

    def fingerprint(self) -> str:
        """Keys the ledger + archive. Symbols + dates + holdout geometry --
        the things that define WHICH holdout is being burned."""
        raw = json.dumps({"symbols": sorted(self.symbols), "start": self.start,
                          "end": self.end, "hn": self.holdout_name_frac,
                          "ht": self.holdout_time_frac}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def load(cls, path: str) -> "Manifest":
        import os
        with open(path) as f:
            d = yaml.safe_load(f)
        if "symbols_file" in d:                      # one ticker per line
            p = d.pop("symbols_file")
            if not os.path.isabs(p):
                p = os.path.join(os.path.dirname(os.path.abspath(path)), p)
            with open(p) as sf:
                d["symbols"] = [ln.strip() for ln in sf
                                if ln.strip() and not ln.startswith("#")]
        if isinstance(d.get("symbols"), str):        # preset name
            d["symbols"] = PRESETS[d["symbols"]]
        allowed = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in allowed})
