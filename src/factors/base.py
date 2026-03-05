"""
Factor base class and cross-sectional preprocessing utilities.

Every factor inherits from Factor and exposes a single method:
    compute(close, volume, market_returns, fundamentals) -> pd.DataFrame

The output is always a (date × ticker) DataFrame of raw scores,
which are then passed through the standard preprocessing pipeline:
    1. Winsorize at 1st/99th percentile  → remove outliers
    2. Cross-sectional z-score           → mean 0, std 1 each day
    3. Rank-normalize (optional)         → uniform distribution

Cross-sectional processing is done *within each date*, never across time,
to avoid look-ahead bias.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Factor(ABC):
    """Abstract base for all factors."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def compute(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        market_returns: pd.Series | None = None,
        fundamentals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute raw factor scores.

        Parameters
        ----------
        close       : (date × ticker) adjusted close prices
        volume      : (date × ticker) daily share volume
        market_returns : (date,) SPY daily returns (for beta calc)
        fundamentals   : (ticker × field) fundamental snapshot

        Returns
        -------
        pd.DataFrame : (date × ticker) raw factor scores
                       Higher = better (all factors already sign-adjusted)
        """
        ...

    def preprocess(
        self,
        raw: pd.DataFrame,
        winsor_pct: float = 0.01,
        method: str = "zscore",
    ) -> pd.DataFrame:
        """
        Standard cross-sectional preprocessing pipeline.

        Parameters
        ----------
        raw       : raw factor panel (date × ticker)
        winsor_pct: one-sided winsorization percentile (default 1%)
        method    : 'zscore' | 'rank' | 'none'
        """
        processed = winsorize_cs(raw, pct=winsor_pct)
        if method == "zscore":
            processed = zscore_cs(processed)
        elif method == "rank":
            processed = rank_cs(processed)
        return processed

    def compute_and_preprocess(self, **kwargs) -> pd.DataFrame:
        raw = self.compute(**kwargs)
        return self.preprocess(raw)


# ------------------------------------------------------------------
# Cross-sectional preprocessing utilities
# ------------------------------------------------------------------

def winsorize_cs(df: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    """
    Winsorize each row (cross-section) at pct and 1-pct percentiles.

    Capping extreme outliers is critical in factor research — a single
    stock with a P/E of -1000 or a one-day return of +300% will
    dominate the cross-section without this step.
    """
    def _winsor_row(row: pd.Series) -> pd.Series:
        lo = row.quantile(pct)
        hi = row.quantile(1 - pct)
        return row.clip(lo, hi)

    return df.apply(_winsor_row, axis=1)


def zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score: subtract cross-sectional mean, divide by std.

    After z-scoring, each day's factor values have mean ≈ 0, std ≈ 1.
    This makes factor magnitudes comparable across time and across factors
    — essential for combining multiple factors into a composite score.
    """
    mu  = df.mean(axis=1)
    sig = df.std(axis=1)
    return df.sub(mu, axis=0).div(sig, axis=0)


def rank_cs(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """
    Cross-sectional rank normalization: convert raw scores to percentile ranks.

    Rank normalization makes factor distributions uniform and removes
    sensitivity to outliers beyond winsorization. Spearman IC (rank IC)
    is the industry standard for this reason.

    Output: scores in [0, 1] where 1 = highest-ranked stock.
    """
    ranked = df.rank(axis=1, pct=True, ascending=ascending, na_option="keep")
    return ranked


def neutralize_sector(
    factor: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    Remove sector effects from factor scores via cross-sectional demeaning.

    For each date, subtract the sector mean from each stock's score.
    This ensures factor bets are within-sector, not sector-tilts.
    E.g., a high-momentum score should reflect outperformance *within*
    technology, not just that tech stocks have high momentum overall.

    Parameters
    ----------
    factor     : (date × ticker) z-scored factor
    sector_map : (ticker,) mapping ticker → sector string

    Returns
    -------
    Sector-neutralized factor DataFrame
    """
    result = factor.copy()
    for date, row in factor.iterrows():
        row = row.dropna()
        for sector, group in row.groupby(sector_map.reindex(row.index)):
            result.loc[date, group.index] = group - group.mean()
    return result


def factor_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute the average pairwise cross-sectional correlation between factors.

    High factor correlations (>0.7) indicate redundancy — combining them
    adds little diversification. This guides factor selection and weighting.

    Method: for each date, compute rank correlation matrix; average over time.
    """
    names = list(factors.keys())
    n = len(names)
    corr_sum = pd.DataFrame(0.0, index=names, columns=names)
    count = 0

    # Sample monthly to speed up computation
    all_dates = list(factors[names[0]].index)
    monthly_dates = pd.DatetimeIndex(all_dates).to_period("M").drop_duplicates()
    sample_dates = [
        factors[names[0]].index[factors[names[0]].index.to_period("M") == m][-1]
        for m in monthly_dates
    ]

    for dt in sample_dates:
        slices = {}
        for name in names:
            try:
                s = factors[name].loc[dt].dropna()
                slices[name] = s
            except KeyError:
                continue

        if len(slices) < 2:
            continue

        combined = pd.DataFrame(slices).dropna()
        if len(combined) < 20:
            continue

        corr_sum += combined.corr(method="spearman")
        count += 1

    return (corr_sum / max(count, 1)).round(3)
