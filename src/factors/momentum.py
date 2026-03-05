"""
Momentum and reversal factors.

Academic foundation:
  - Jegadeesh & Titman (1993): 12-month momentum (skip last month) generates
    ~1%/month in the US. One of the most replicated factors in finance.
  - Jegadeesh (1990): Short-term reversal — last month's losers outperform
    next month. Microstructure-driven, fades quickly.

Implementation convention:
  - Skip the most recent month (t-21 to t-1) when computing long-run momentum
    to avoid contamination by short-term reversal. This is the standard.
  - All factors are sign-adjusted so HIGHER = expected outperformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import Factor


class Momentum12_1(Factor):
    """
    12-month momentum, skipping last month.
    Return from t-252 to t-21 (trading days).

    This is the canonical momentum factor (Jegadeesh & Titman 1993).
    Winner stocks continue to outperform for ~6-12 months before reverting.
    """
    name = "MOM_12_1"
    description = "12-month price momentum (skip 1 month)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        # Return from 12 months ago to 1 month ago
        ret = close.shift(21) / close.shift(252) - 1
        return ret


class Momentum6_1(Factor):
    """
    6-month momentum, skipping last month.
    Return from t-126 to t-21.
    """
    name = "MOM_6_1"
    description = "6-month price momentum (skip 1 month)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        return close.shift(21) / close.shift(126) - 1


class Momentum3_1(Factor):
    """
    3-month momentum, skipping last month.
    Return from t-63 to t-21.
    """
    name = "MOM_3_1"
    description = "3-month price momentum (skip 1 month)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        return close.shift(21) / close.shift(63) - 1


class ShortTermReversal(Factor):
    """
    Short-term reversal: last month's return (t-21 to t).

    Negative sign: past losers outperform in the near term.
    Driven by microstructure (liquidity provision, bid-ask bounce).
    Decays rapidly — strong at 1-week horizon, weak at 1-month.

    Note: sign-flipped so higher score = better expected return.
    """
    name = "STR"
    description = "Short-term reversal (1-month return, sign-flipped)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        ret_1m = close / close.shift(21) - 1
        return -ret_1m  # flip sign: past losers expected to outperform


class MomentumQuality(Factor):
    """
    Residual momentum (Blitz et al. 2011): momentum orthogonal to market beta.

    Compute cumulative 12-1 return, then remove the component explained
    by market exposure. Residual momentum has higher IC and lower crash risk
    than raw momentum.

    Uses rolling 252-day market beta to compute market-adjusted return.
    """
    name = "MOM_RESID"
    description = "Residual momentum (market-adjusted 12-1 return)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if market_returns is None:
            # Fall back to raw momentum
            return close.shift(21) / close.shift(252) - 1

        raw_mom = close.shift(21) / close.shift(252) - 1

        # Compute rolling 252-day beta for each ticker
        mkt = market_returns.reindex(close.index).fillna(0)
        betas = {}
        for col in close.columns:
            stock_ret = close[col].pct_change()
            beta_series = (
                stock_ret.rolling(252).cov(mkt)
                / mkt.rolling(252).var()
            )
            betas[col] = beta_series

        beta_df = pd.DataFrame(betas)

        # 12-1 market return
        mkt_cum = (1 + mkt).rolling(252).apply(np.prod, raw=True) - 1
        mkt_cum_shifted = mkt_cum.shift(21)

        # Residual = total - beta * market
        mkt_contribution = beta_df.multiply(mkt_cum_shifted, axis=0)
        return raw_mom - mkt_contribution


class MomentumVolatilityRatio(Factor):
    """
    Momentum / Volatility ratio (TMOM / BVOL — Barroso & Santa-Clara 2015).

    Risk-adjusts momentum by trailing volatility. High momentum + low vol
    is the strongest signal — risk-adjusted momentum has smaller crashes.

    Score = MOM_12_1 / trailing_vol_63d
    """
    name = "MOM_VOL_ADJ"
    description = "Volatility-adjusted 12-1 momentum"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        mom = close.shift(21) / close.shift(252) - 1
        vol = close.pct_change().rolling(63).std() * np.sqrt(252)
        return mom / vol.replace(0, np.nan)
