"""
Data layer: S&P 500 universe, price history, and fundamental data.

Sources:
  - Universe  : Wikipedia S&P 500 constituent list (current)
  - Prices    : yfinance bulk OHLCV download
  - Fundamntls: yfinance per-ticker info + quarterly financials

Survivorship bias note:
  We use the *current* S&P 500 constituent list. This introduces mild
  survivorship bias (dead/delisted stocks are excluded). A production system
  would use a point-in-time constituent database (e.g., Compustat CRSP).
  For research purposes, results here will overstate returns slightly;
  the bias is documented and the methodology is otherwise rigorous.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).resolve().parents[3] / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"


# ------------------------------------------------------------------
# Universe
# ------------------------------------------------------------------

def get_sp500_universe(cache: bool = True) -> list[str]:
    """
    Fetch current S&P 500 constituents from Wikipedia.

    Returns cleaned ticker symbols (BRK.B → BRK-B for yfinance compatibility).
    Falls back to a hardcoded 150-ticker liquid subset if Wikipedia is unavailable.
    """
    cache_path = RAW_DIR / "sp500_tickers.txt"
    if cache and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime).days < 7:
            return cache_path.read_text().strip().split("\n")

    try:
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0
        )[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
    except Exception as e:
        log.warning(f"Wikipedia fetch failed ({e}), using fallback universe")
        tickers = _FALLBACK_UNIVERSE

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(tickers))
    return tickers


# ------------------------------------------------------------------
# Price data
# ------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str | None = None,
    cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download adjusted OHLCV data for all tickers via yfinance bulk API.

    Returns
    -------
    dict mapping ticker → DataFrame[open, high, low, close, volume, returns, log_returns]

    Bulk download is ~50× faster than looping; yfinance handles rate limiting internally.
    """
    end = end or date.today().isoformat()
    cache_path = RAW_DIR / f"prices_{start[:4]}_to_{end[:4]}.parquet"

    if cache and cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if (datetime.now() - mtime).days < 1:
            log.info(f"Loading cached prices from {cache_path}")
            panel = pd.read_parquet(cache_path)
            return _panel_to_dict(panel)

    log.info(f"Downloading prices for {len(tickers)} tickers ({start} → {end})")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Flatten MultiIndex columns → (field, ticker) → pivot to (date, ticker) per field
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw["Close"].copy()
        volume = raw["Volume"].copy()
        high   = raw["High"].copy()
        low    = raw["Low"].copy()
    else:
        close  = raw[["Close"]]
        volume = raw[["Volume"]]
        high   = raw[["High"]]
        low    = raw[["Low"]]

    close.index  = pd.to_datetime(close.index).tz_localize(None)
    volume.index = close.index
    high.index   = close.index
    low.index    = close.index

    # Drop tickers with too many missing values (> 20%)
    coverage = close.notna().mean()
    good_tickers = coverage[coverage >= 0.80].index.tolist()
    log.info(f"Kept {len(good_tickers)}/{len(tickers)} tickers with ≥80% coverage")

    close  = close[good_tickers]
    volume = volume[good_tickers]
    high   = high[good_tickers]
    low    = low[good_tickers]

    # Build panel for caching
    panel = pd.concat(
        [close.stack(), volume.stack(), high.stack(), low.stack()],
        axis=1,
        keys=["close", "volume", "high", "low"],
    )
    panel.index.names = ["date", "ticker"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path)
    log.info(f"Saved price panel ({panel.shape}) to {cache_path}")

    return _panel_to_dict(panel)


def load_close_panel(
    tickers: list[str] | None = None,
    start: str = "2019-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Load closing prices as a (date × ticker) DataFrame — the primary input
    for all factor computations.
    """
    price_dict = download_prices(tickers or get_sp500_universe(), start=start, end=end)
    close = pd.DataFrame({t: df["close"] for t, df in price_dict.items()})
    close = close.sort_index()
    # Forward-fill up to 5 days (handles holidays/trading halts)
    close = close.ffill(limit=5)
    return close


def load_volume_panel(
    tickers: list[str] | None = None,
    start: str = "2019-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    price_dict = download_prices(tickers or get_sp500_universe(), start=start, end=end)
    return pd.DataFrame({t: df["volume"] for t, df in price_dict.items()}).sort_index()


# ------------------------------------------------------------------
# Return computation
# ------------------------------------------------------------------

def compute_returns(
    close: pd.DataFrame,
    periods: list[int] = [1, 5, 21],
) -> dict[int, pd.DataFrame]:
    """
    Compute forward returns at multiple horizons.

    Parameters
    ----------
    close   : (date × ticker) close price panel
    periods : list of horizons in trading days

    Returns
    -------
    dict mapping horizon → DataFrame of forward returns (same shape as close)

    Note: forward return at t for horizon h = close[t+h] / close[t] - 1.
    All returns are *forward-shifted* so factor[t] predicts return[t].
    """
    fwd = {}
    for h in periods:
        # shift(-h): at time t, this gives the return from t to t+h
        fwd[h] = close.pct_change(h).shift(-h)
    return fwd


# ------------------------------------------------------------------
# Fundamental data
# ------------------------------------------------------------------

def download_fundamentals(
    tickers: list[str],
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch point-in-time fundamental snapshot for each ticker.

    Fields fetched: trailingPE, priceToBook, returnOnEquity, grossMargins,
                    marketCap, trailingEps, totalRevenue, totalAssets

    IMPORTANT: yfinance returns *current* (not historical) fundamentals.
    These are used as a static cross-sectional snapshot. A production
    system would use Compustat/FactSet for proper point-in-time data.

    Returns
    -------
    DataFrame (ticker × fundamental field)
    """
    cache_path = RAW_DIR / f"fundamentals_{date.today().isoformat()}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    FIELDS = [
        "trailingPE", "priceToBook", "returnOnEquity",
        "grossMargins", "marketCap", "trailingEps",
        "totalRevenue", "totalAssets", "debtToEquity",
        "operatingMargins", "earningsGrowth",
    ]

    rows = []
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).fast_info
            info_full = yf.Ticker(ticker).info
            row = {"ticker": ticker}
            for f in FIELDS:
                row[f] = info_full.get(f, np.nan)
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": ticker})
        if i % 50 == 0:
            log.info(f"  Fundamentals: {i}/{len(tickers)}")
            time.sleep(0.5)  # gentle rate limiting

    df = pd.DataFrame(rows).set_index("ticker")
    df = df.apply(pd.to_numeric, errors="coerce")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    log.info(f"Saved fundamentals ({df.shape}) to {cache_path}")
    return df


# ------------------------------------------------------------------
# Market returns (SPY benchmark)
# ------------------------------------------------------------------

def load_market_returns(start: str = "2019-01-01") -> pd.Series:
    """
    Load daily SPY returns as the market factor for beta calculation
    and market-neutral return attribution.
    """
    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    return spy["Close"].pct_change().rename("market")


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _panel_to_dict(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for ticker, group in panel.groupby(level="ticker"):
        df = group.droplevel("ticker")
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        result[ticker] = df
    return result


# ------------------------------------------------------------------
# Fallback universe (top 150 S&P 500 names by market cap, 2024)
# ------------------------------------------------------------------

_FALLBACK_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","UNH","LLY",
    "JPM","XOM","V","AVGO","JNJ","PG","MA","COST","HD","MRK",
    "ABBV","CVX","CRM","BAC","PEP","KO","WMT","ORCL","MCD","TMO",
    "CSCO","ABT","ACN","DHR","IBM","NEE","TXN","PM","LIN","QCOM",
    "GE","UPS","CAT","AMGN","RTX","HON","SPGI","MS","GS","BLK",
    "LOW","ISRG","ELV","MDT","SYK","VRTX","AXP","AMAT","DE","ADI",
    "GILD","CB","MMM","REGN","ETN","PLD","MDLZ","NOW","ZTS","BMY",
    "SCHW","MO","DUK","SO","ICE","CI","USB","TJX","CL","PNC",
    "AON","EOG","SLB","CME","TGT","ITW","APD","CSX","ADP","GM",
    "F","FDX","NSC","EW","ATVI","KLAC","PANW","LRCX","MCHP","SNPS",
    "CDNS","NXPI","FTNT","ADSK","WM","GIS","K","HSY","SJM","CPB",
    "HRL","CAG","MKC","KHC","TSN","TAP","STZ","KMB","CHD","CLX",
    "EL","PH","ROP","IDXX","IQV","PKG","AME","FMC","PPG","SHW",
    "ECL","NEM","FCX","ALB","CF","MOS","NUE","STLD","RS","ATI",
]
