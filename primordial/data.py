"""
data.py -- sources, multi-timeframe resampling, parquet cache.

Fetch ONCE at the manifest's base timeframe, resample up in memory, cache
everything to parquet keyed by (source, symbol, timeframe, start, end).
Build the cache locally and ship it to RunPod -- never re-download per pod.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

def cache_dir():
    """Resolved at CALL time so tests/pods can redirect via env."""
    return os.environ.get("PRIMORDIAL_CACHE",
                          os.path.expanduser("~/.primordial_cache"))

_FREQ = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h",
         "4h": "4h", "1d": "1D", "1w": "1W"}
BAR_SECONDS = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
               "4h": 14400, "1d": 86400, "1w": 604800}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe not in _FREQ:
        raise ValueError(timeframe)
    o = df.resample(_FREQ[timeframe]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum"}).dropna(subset=["open", "close"])
    return o


def _cache_path(source, sym, timeframe, start, end):
    safe = sym.replace("/", "_").replace("^", "IDX_")
    return os.path.join(cache_dir(),
                        f"{source}_{safe}_{timeframe}_{start}_{end}.parquet")


def fetch(source: str, symbols, timeframe, start, end, min_bars=300) -> dict:
    """Returns {sym: df}. Sources: synthetic | yfinance | coinbase | alpaca | csv:<dir>.
    Every result is cached to parquet; cache hits never touch the network."""
    out = {}
    os.makedirs(cache_dir(), exist_ok=True)
    for sym in symbols:
        p = _cache_path(source, sym, timeframe, start, end)
        if os.path.exists(p):
            df = pd.read_parquet(p)
        else:
            df = _fetch_one(source, sym, timeframe, start, end)
            if df is not None and len(df):
                df.to_parquet(p)
        if df is not None and len(df) >= min_bars:
            out[sym] = df
    return out


def _fetch_one(source, sym, timeframe, start, end):
    if source == "synthetic":
        return _synthetic(sym, timeframe, start, end)
    if source == "yfinance":
        return _yfinance(sym, timeframe, start, end)
    if source == "coinbase":
        return _coinbase(sym, timeframe, start, end)
    if source == "alpaca":
        return _alpaca(sym, timeframe, start, end)
    if source == "omnifeed":
        return _omnifeed(sym, timeframe, start, end)
    if source.startswith("csv:"):
        path = os.path.join(source[4:], f"{sym}.csv")
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.columns = [c.lower() for c in df.columns]
        return df.loc[start:end]
    raise ValueError(f"unknown source {source}")


def _synthetic(sym, timeframe, start, end):
    """Random walk with vol clustering. Valid OHLC (open inside [low, high]).
    NO edge exists here by construction -- the negative control."""
    import hashlib as _hl
    rng = np.random.default_rng(int(_hl.sha256(sym.encode()).hexdigest()[:8], 16))
    idx = pd.date_range(start, end, freq=_FREQ[timeframe])
    n = len(idx)
    vol = 0.004 * np.exp(0.35 * rng.standard_normal(n).cumsum() * 0.05)
    vol = np.clip(vol, 0.0005, 0.05)
    r = rng.standard_normal(n) * vol
    close = 100 * np.exp(np.cumsum(r))
    o = np.roll(close, 1) * (1 + rng.standard_normal(n) * vol * 0.3)
    o[0] = close[0]
    hi = np.maximum(o, close) * (1 + np.abs(rng.standard_normal(n)) * vol * 0.5)
    lo = np.minimum(o, close) * (1 - np.abs(rng.standard_normal(n)) * vol * 0.5)
    v = np.abs(rng.standard_normal(n) + 3) * 1e6 * (1 + 10 * vol)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": close,
                         "volume": v}, index=idx)


def _omnifeed(sym, timeframe, start, end):
    """Blake's omnifeed package: one call, any asset class -- it routes
    (crypto -> Coinbase, index/equity daily -> yfinance, intraday equity ->
    Alpaca), and owns pagination/retries/rate limits + its own incremental
    cache. Primordial's parquet cache wraps it, so a warmed cache never even
    imports omnifeed again. pip install it from your omnifeed repo first."""
    from omnifeed import get_bars
    df = get_bars(sym, timeframe, start=start, end=end)
    if df is None or df.empty:
        return None
    df = df.rename(columns=str.lower)
    need = ["open", "high", "low", "close"]
    if any(c not in df.columns for c in need):
        return None
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].sort_index()


def _yfinance(sym, timeframe, start, end):
    """Global equities and INDEX composites (^GSPC, ^FTSE, ^KS11, ^NSEI...).
    auto_adjust=True -> splits and dividends handled. yfinance limits
    intraday history; for indexes use 1d, which is also where the deep
    multi-regime history is."""
    import yfinance as yf
    iv = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
          "1d": "1d", "1w": "1wk"}[timeframe]
    df = yf.download(sym, start=start, end=end, interval=iv,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _coinbase(sym, timeframe, start, end):
    """Public candles endpoint, no key, paginated backwards (~3 req/s,
    300 candles per request)."""
    import requests, time as _t
    gran = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
    if timeframe not in gran:
        raise ValueError(f"coinbase does not serve {timeframe}; fetch a finer "
                         "granularity and resample")
    g = gran[timeframe]
    t0 = pd.Timestamp(start); t1 = pd.Timestamp(end)
    rows = []
    cur = t1
    while cur > t0:
        lo = max(t0, cur - pd.Timedelta(seconds=300 * g))
        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{sym}/candles",
            params={"granularity": g, "start": lo.isoformat(),
                    "end": cur.isoformat()}, timeout=20)
        if r.status_code != 200:
            break
        rows += r.json()
        cur = lo
        _t.sleep(0.34)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close",
                                     "volume"]).drop_duplicates("time")
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return (df.set_index("time").sort_index()
              [["open", "high", "low", "close", "volume"]])


def _alpaca(sym, timeframe, start, end):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    try:
        from alpaca.data.enums import Adjustment
        adj = Adjustment.ALL          # RAW default turns splits into -90% gaps
    except ImportError:
        adj = "all"
    tf = {"5m": TimeFrame(5, TimeFrameUnit.Minute),
          "15m": TimeFrame(15, TimeFrameUnit.Minute),
          "30m": TimeFrame(30, TimeFrameUnit.Minute),
          "1h": TimeFrame.Hour, "1d": TimeFrame.Day}[timeframe]
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"],
                                       os.environ["ALPACA_SECRET_KEY"])
    bars = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=sym, timeframe=tf, start=start, end=end,
        adjustment=adj)).df
    if bars.empty:
        return None
    df = bars.xs(sym) if isinstance(bars.index, pd.MultiIndex) else bars
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def multi_timeframe(data_base: dict, base_tf: str, timeframes: list) -> dict:
    """{tf: {sym: df}} resampled up from one base fetch."""
    out = {}
    for tf in timeframes:
        if BAR_SECONDS[tf] < BAR_SECONDS[base_tf]:
            continue                     # cannot resample downward
        if tf == base_tf:
            out[tf] = {s: df.copy() for s, df in data_base.items()}
        else:
            out[tf] = {s: resample_ohlcv(df, tf) for s, df in data_base.items()}
    return out
