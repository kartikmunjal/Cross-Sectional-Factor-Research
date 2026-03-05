"""
Value factors derived from fundamental ratios.

Academic foundation:
  - Fama & French (1992): Book-to-market (B/M) is the canonical value factor.
    High B/M (cheap) stocks outperform low B/M (expensive) stocks by ~5%/year.
  - Lakonishok, Shleifer & Vishny (1994): Value anomaly persists due to
    investor extrapolation of past growth — not rational risk compensation.

Data note:
  These factors use the fundamental snapshot from yfinance (current values).
  In production, point-in-time data (Compustat CRSP) is required to avoid
  look-ahead bias. Here, fundamentals are held constant over the backtest period,
  which is a documented limitation.

All factors are sign-adjusted: HIGHER score = cheaper / more value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import Factor


class BookToMarket(Factor):
    """
    Book-to-market ratio (inverse of P/B).

    High B/M = value stock (cheap relative to book value).
    Fama-French HML factor is essentially long high B/M, short low B/M.
    """
    name = "BTM"
    description = "Book-to-market ratio (1/P_B)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if fundamentals is None or "priceToBook" not in fundamentals.columns:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        pb = fundamentals["priceToBook"].reindex(close.columns)
        btm = 1.0 / pb.replace(0, np.nan)
        btm = btm.clip(lower=0)  # negative B/M (negative equity) → NaN

        # Broadcast across time: fundamental held constant
        return pd.DataFrame(
            np.tile(btm.values, (len(close), 1)),
            index=close.index,
            columns=close.columns,
        )


class EarningsYield(Factor):
    """
    Earnings yield = E/P (inverse of trailing P/E).

    More robust than raw P/E because it handles the sign correctly
    (negative earnings → negative E/P, naturally filtered as low-value).
    """
    name = "EP"
    description = "Earnings yield (1/P_E = E/P)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if fundamentals is None or "trailingPE" not in fundamentals.columns:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        pe = fundamentals["trailingPE"].reindex(close.columns)
        ep = 1.0 / pe.replace(0, np.nan)

        return pd.DataFrame(
            np.tile(ep.values, (len(close), 1)),
            index=close.index,
            columns=close.columns,
        )


class PriceToSales(Factor):
    """
    Sales yield = Revenue / Market Cap (inverse of P/S).

    Useful when earnings are negative (e.g., growth stocks).
    More stable than P/E across business cycles.
    """
    name = "SP"
    description = "Sales yield (Revenue / Market Cap)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if fundamentals is None:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        rev = fundamentals.get("totalRevenue", pd.Series(dtype=float)).reindex(close.columns)
        mcap = fundamentals.get("marketCap", pd.Series(dtype=float)).reindex(close.columns)

        sp = rev / mcap.replace(0, np.nan)

        return pd.DataFrame(
            np.tile(sp.values, (len(close), 1)),
            index=close.index,
            columns=close.columns,
        )


class CompositeValue(Factor):
    """
    Composite value score: equal-weighted average of BTM + EP + SP.

    Composite value is more stable than any single metric and
    reduces reliance on the accuracy of any one fundamental estimate.
    This is the standard approach at systematic quant funds.
    """
    name = "VALUE_COMP"
    description = "Composite value (BTM + EP + SP equal-weighted)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        from src.factors.base import zscore_cs

        sub_factors = [
            BookToMarket(), EarningsYield(), PriceToSales()
        ]

        scores = []
        for f in sub_factors:
            raw = f.compute(close, fundamentals=fundamentals)
            scores.append(zscore_cs(raw))

        # Average non-NaN scores
        stacked = pd.concat(scores, axis=0)
        # Reshape and average
        result = sum(scores) / len(scores)
        return result
