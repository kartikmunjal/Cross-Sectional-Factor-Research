"""
Volatility, beta, and liquidity factors.

Academic foundation:
  - Ang et al. (2006): Stocks with high idiosyncratic volatility earn LOW
    future returns — the "idiosyncratic volatility puzzle." Counterintuitive
    given CAPM but extremely robust empirically.
  - Frazzini & Pedersen (2014): "Betting Against Beta" — low-beta stocks
    outperform after leverage-adjusting. Driven by leverage-constrained
    investors overpaying for high-beta (lottery-like) stocks.
  - Amihud (2002): Illiquidity ratio predicts returns positively — investors
    demand a liquidity premium.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import Factor


class IdiosyncraticVol(Factor):
    """
    Idiosyncratic volatility: std of residuals from market model.

    Residuals from daily regression: r_it = α + β * r_mt + ε_it
    IVOL = std(ε) over trailing 21 or 63 days.

    Sign-flipped: low IVOL → high expected return (Ang et al. 2006).
    """
    name = "IVOL"
    description = "Idiosyncratic volatility (21d, sign-flipped)"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        returns = close.pct_change()

        if market_returns is None:
            # Fall back to total volatility if no market data
            vol = returns.rolling(self.window).std() * np.sqrt(252)
            return -vol

        mkt = market_returns.reindex(returns.index).fillna(0)
        ivol = {}

        for col in returns.columns:
            stock = returns[col].dropna()
            resid_vol = pd.Series(np.nan, index=returns.index)

            for i in range(self.window, len(returns)):
                window_stock = returns[col].iloc[i - self.window:i]
                window_mkt = mkt.iloc[i - self.window:i]

                valid = window_stock.notna() & window_mkt.notna()
                if valid.sum() < self.window * 0.8:
                    continue

                s = window_stock[valid].values
                m = window_mkt[valid].values

                # OLS beta
                cov = np.cov(s, m)
                if cov[1, 1] > 1e-10:
                    beta = cov[0, 1] / cov[1, 1]
                else:
                    beta = 0.0

                resid = s - beta * m
                resid_vol.iloc[i] = resid.std() * np.sqrt(252)

            ivol[col] = resid_vol

        ivol_df = pd.DataFrame(ivol, index=returns.index)
        return -ivol_df  # flip sign: low IVOL → positive score


class Beta(Factor):
    """
    Rolling market beta, sign-flipped (Betting Against Beta).

    Low-beta stocks outperform high-beta on a risk-adjusted basis.
    Frazzini & Pedersen (2014): constrained investors bid up high-beta,
    creating a security market line that is too flat.

    Beta estimated via 252-day rolling OLS.
    Score = -beta (so low-beta stocks score high).
    """
    name = "BAB"
    description = "Betting-Against-Beta (1-year rolling beta, sign-flipped)"

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        returns = close.pct_change()

        if market_returns is None:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        mkt = market_returns.reindex(returns.index).fillna(0)
        mkt_var = mkt.rolling(252).var()

        betas = returns.apply(
            lambda col: col.rolling(252).cov(mkt) / mkt_var
        )
        return -betas  # sign-flip: low beta → positive score


class RealizedVol(Factor):
    """
    Total realized volatility (annualized), sign-flipped.

    Simpler than IVOL — no market model needed. Less precise but
    highly correlated with IVOL for most large-cap stocks.
    """
    name = "RVOL"
    description = "Realized volatility (63d, sign-flipped)"

    def __init__(self, window: int = 63):
        self.window = window

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        vol = close.pct_change().rolling(self.window).std() * np.sqrt(252)
        return -vol  # low vol → positive score


class AmihudIlliquidity(Factor):
    """
    Amihud (2002) illiquidity ratio.

    ILLIQ_t = mean(|r_d| / DollarVolume_d)  over trailing 21 days

    where DollarVolume = close × volume (in $ millions).

    Higher illiquidity → higher expected return (liquidity premium).
    Sign is kept positive: illiquid stocks are expected to outperform.

    Note: this is the *illiquidity* factor — stocks that are hard to
    trade demand a premium. The sign convention here is that small/illiquid
    stocks are expected to outperform (consistent with the small-cap premium).
    """
    name = "ILLIQ"
    description = "Amihud illiquidity ratio (21d mean, log-scaled)"

    def __init__(self, window: int = 21):
        self.window = window

    def compute(self, close, volume=None, market_returns=None, fundamentals=None):
        if volume is None:
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)

        abs_ret = close.pct_change().abs()
        dollar_vol = close * volume / 1e6  # in $ millions

        # Avoid division by zero
        illiq = (abs_ret / dollar_vol.replace(0, np.nan)).rolling(self.window).mean()

        # Log-scale: distribution is highly right-skewed
        return np.log1p(illiq)
