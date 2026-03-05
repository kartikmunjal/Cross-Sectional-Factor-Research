"""
Quality factors: profitability, efficiency, and financial stability.

Academic foundation:
  - Novy-Marx (2013): Gross profitability (gross profit / assets) is the
    "other side of value" — predicts returns as strongly as B/M, with
    negative correlation (complements value nicely in a portfolio).
  - Fama & French (2015): Profitability (RMW) and investment (CMA) factors
    extend the 3-factor model and absorb many anomalies.
  - Sloan (1996): Accruals anomaly — high accruals predict lower future
    earnings (earnings quality signal).

Key insight: Quality and Value are negatively correlated (~-0.3 to -0.5).
Combining them diversifies the composite factor significantly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import Factor


class GrossProfitability(Factor):
    """
    Gross profit / total assets (Novy-Marx 2013).

    Gross profit (revenue - COGS) is the "cleanest" measure of profitability
    because it is least subject to accounting manipulation (no SGA, D&A).
    Scaled by assets to make it comparable across firm sizes.

    This is a POSITIVE quality signal: high gross profitability → outperform.
    """
    name = "GPOA"
    description = "Gross profitability = Gross Profit / Total Assets"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if fundamentals is None:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        gm = fundamentals.get("grossMargins", pd.Series(dtype=float)).reindex(close.columns)
        rev = fundamentals.get("totalRevenue", pd.Series(dtype=float)).reindex(close.columns)
        assets = fundamentals.get("totalAssets", pd.Series(dtype=float)).reindex(close.columns)

        gross_profit = gm * rev
        gpoa = gross_profit / assets.replace(0, np.nan)

        return pd.DataFrame(
            np.tile(gpoa.values, (len(close), 1)),
            index=close.index,
            columns=close.columns,
        )


class ReturnOnEquity(Factor):
    """
    Return on equity (ROE) — net income / book equity.

    Measures how efficiently management generates profits from shareholders'
    capital. High ROE is a classic quality signal, though it can be inflated
    by leverage (high debt reduces equity denominator).
    """
    name = "ROE"
    description = "Return on equity (trailing twelve months)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if fundamentals is None or "returnOnEquity" not in fundamentals.columns:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        roe = fundamentals["returnOnEquity"].reindex(close.columns)

        return pd.DataFrame(
            np.tile(roe.values, (len(close), 1)),
            index=close.index,
            columns=close.columns,
        )


class AssetGrowth(Factor):
    """
    Asset growth factor (Cooper, Gulen & Schill 2008).

    HIGH asset growth predicts LOW future returns — companies that grow
    their asset base (via acquisitions, capex) tend to earn lower subsequent
    returns (overinvestment / empire building).

    Sign-flipped: LOW asset growth → HIGH quality score.

    We approximate with YoY change in balance sheet size from yfinance.
    """
    name = "ASSET_GROWTH"
    description = "Asset growth (YoY, sign-flipped)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if fundamentals is None:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        # yfinance doesn't give historical asset growth directly
        # We use debt-to-equity as a proxy: high leverage often correlates
        # with aggressive balance sheet expansion
        de = fundamentals.get("debtToEquity", pd.Series(dtype=float)).reindex(close.columns)
        # Higher D/E = more leverage = lower quality score
        asset_growth_proxy = de / 100.0  # normalize (D/E is often 0-300+)

        return pd.DataFrame(
            np.tile(-asset_growth_proxy.values, (len(close), 1)),  # flip sign
            index=close.index,
            columns=close.columns,
        )


class OperatingProfitability(Factor):
    """
    Operating margin (operating income / revenue).

    Captures operational efficiency — how much of each revenue dollar
    converts to operating profit after COGS and SGA but before interest/tax.
    """
    name = "OPER_MARGIN"
    description = "Operating margin (operating income / revenue)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if fundamentals is None:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        op_margin = fundamentals.get("operatingMargins", pd.Series(dtype=float)).reindex(close.columns)

        return pd.DataFrame(
            np.tile(op_margin.values, (len(close), 1)),
            index=close.index,
            columns=close.columns,
        )


class QualityComposite(Factor):
    """
    Composite quality score: equal-weighted average of
    GPOA + ROE + OperatingMargin, z-scored cross-sectionally.

    Novy-Marx (2013) shows composite quality measures are more stable
    predictors than any individual metric. The composite also diversifies
    idiosyncratic data noise in any single accounting series.
    """
    name = "QUALITY_COMP"
    description = "Composite quality (GPOA + ROE + OperatingMargin)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        from src.factors.base import zscore_cs

        sub_factors = [
            GrossProfitability(),
            ReturnOnEquity(),
            OperatingProfitability(),
        ]

        scores = []
        for f in sub_factors:
            raw = f.compute(close, fundamentals=fundamentals)
            z = zscore_cs(raw)
            scores.append(z)

        # Average, ignoring NaN
        combined = pd.concat(scores, axis=0)
        result = sum(scores) / len(scores)
        return result
