"""
Multi-factor ranking model and quintile portfolio construction.

Pipeline:
  1. Z-score each factor cross-sectionally
  2. Compute IC-weighted composite score (or equal-weight)
  3. Rank stocks into quintiles (Q1=best, Q5=worst)
  4. Long Q1, Short Q5 — dollar-neutral long-short portfolio
  5. Apply turnover control: limit position changes per rebalance

Portfolio construction philosophy:
  - Equal-weight within quintiles (simpler, more robust than optimization)
  - Market-neutral: match long and short gross exposure
  - Turnover control prevents excessive transaction costs from rank instability
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MultiFactorRanking:
    """
    Combines multiple factor scores into a single composite ranking.

    Parameters
    ----------
    factor_weights : dict mapping factor_name → weight
                     If None, equal-weights all factors.
    n_quintiles    : number of portfolio buckets (default 5)
    """

    def __init__(
        self,
        factor_weights: dict[str, float] | None = None,
        n_quintiles: int = 5,
    ):
        self.factor_weights = factor_weights
        self.n_quintiles = n_quintiles

    def composite_score(
        self,
        factors: dict[str, pd.DataFrame],
        date: pd.Timestamp | str,
    ) -> pd.Series:
        """
        Compute composite score for all stocks on a given date.

        Parameters
        ----------
        factors : dict of factor_name → (date × ticker) z-scored scores
        date    : rebalancing date

        Returns
        -------
        pd.Series of composite scores (higher = stronger buy signal)
        """
        slices = {}
        for fname, fdata in factors.items():
            if date in fdata.index:
                slices[fname] = fdata.loc[date]

        if not slices:
            return pd.Series(dtype=float)

        combined = pd.DataFrame(slices)

        if self.factor_weights is None:
            # Equal weight
            score = combined.mean(axis=1, skipna=True)
        else:
            # IC-weighted or custom weights
            weights = pd.Series(self.factor_weights)
            weights = weights.reindex(combined.columns).fillna(0)
            weights /= weights.abs().sum()  # normalize to sum to 1
            score = combined.dot(weights)

        return score.dropna()

    def quintile_assignments(
        self,
        composite: pd.Series,
    ) -> pd.Series:
        """
        Assign stocks to quintiles based on composite score.

        Q1 = top 20% (highest composite = strongest buy signal)
        Q5 = bottom 20% (lowest composite = strongest sell signal)

        Returns
        -------
        pd.Series of quintile labels (1–5) indexed by ticker
        """
        return pd.qcut(
            composite,
            q=self.n_quintiles,
            labels=range(1, self.n_quintiles + 1),
            duplicates="drop",
        )

    def build_portfolio(
        self,
        composite: pd.Series,
        long_quintile: int = 1,
        short_quintile: int = 5,
    ) -> pd.Series:
        """
        Build dollar-neutral long-short portfolio weights.

        Long top quintile (highest composite), short bottom quintile.
        Within each leg, equal-weight.

        Returns
        -------
        pd.Series of portfolio weights (positive = long, negative = short)
        Sum of long weights = 1.0, sum of short weights = -1.0
        """
        quintiles = self.quintile_assignments(composite)

        long_stocks  = quintiles[quintiles == long_quintile].index
        short_stocks = quintiles[quintiles == short_quintile].index

        weights = pd.Series(0.0, index=composite.index)

        if len(long_stocks) > 0:
            weights[long_stocks] = 1.0 / len(long_stocks)

        if len(short_stocks) > 0:
            weights[short_stocks] = -1.0 / len(short_stocks)

        return weights


def quintile_returns(
    factors: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    rebal_dates: list[pd.Timestamp],
    ranker: MultiFactorRanking | None = None,
    n_quintiles: int = 5,
    horizon: int = 21,
) -> pd.DataFrame:
    """
    Compute equal-weighted return for each quintile at each rebalancing date.

    This is the fundamental output for evaluating a factor's monotonicity:
    if Q1 > Q2 > Q3 > Q4 > Q5 consistently, the factor has strong signal.

    Parameters
    ----------
    factors     : dict of computed factor panels
    returns     : (date × ticker) forward 1-month returns
    rebal_dates : monthly (or weekly) rebalancing dates
    ranker      : MultiFactorRanking instance (uses equal-weight if None)
    n_quintiles : number of buckets
    horizon     : forward return horizon (trading days)

    Returns
    -------
    DataFrame of shape (rebal_dates × quintile) with average returns per bucket
    """
    if ranker is None:
        ranker = MultiFactorRanking(n_quintiles=n_quintiles)

    quintile_ret_rows = []

    for date in rebal_dates:
        composite = ranker.composite_score(factors, date)
        if composite.empty or len(composite) < n_quintiles * 5:
            continue

        try:
            quintiles = ranker.quintile_assignments(composite)
        except Exception:
            continue

        # Forward returns starting from this date
        if date not in returns.index:
            continue

        fwd_ret = returns.loc[date].dropna()

        row = {"date": date}
        for q in range(1, n_quintiles + 1):
            stocks_in_q = quintiles[quintiles == q].index
            common = stocks_in_q.intersection(fwd_ret.index)
            row[f"Q{q}"] = float(fwd_ret[common].mean()) if len(common) > 0 else np.nan

        # Long-short spread
        row["Q1-Q5"] = row.get("Q1", np.nan) - row.get(f"Q{n_quintiles}", np.nan)
        quintile_ret_rows.append(row)

    result = pd.DataFrame(quintile_ret_rows).set_index("date")
    return result


def apply_turnover_control(
    new_weights: pd.Series,
    prev_weights: pd.Series,
    max_turnover: float = 0.30,
) -> pd.Series:
    """
    Limit portfolio turnover by blending new and previous weights.

    Without turnover control, a pure ranking model rebalances 40-60% of
    the portfolio monthly, incurring excessive transaction costs.

    Implementation: if the total two-way turnover of the proposed new portfolio
    exceeds max_turnover, we pull weights back toward the existing portfolio
    until the turnover constraint is satisfied.

    Parameters
    ----------
    new_weights  : proposed new portfolio weights (post-signal)
    prev_weights : existing portfolio weights (before rebalancing)
    max_turnover : maximum allowed one-way turnover (e.g., 0.30 = 30%)

    Returns
    -------
    pd.Series of turnover-controlled weights
    """
    all_tickers = new_weights.index.union(prev_weights.index)
    new  = new_weights.reindex(all_tickers).fillna(0.0)
    prev = prev_weights.reindex(all_tickers).fillna(0.0)

    # One-way turnover = half the sum of absolute weight changes
    turnover = (new - prev).abs().sum() / 2

    if turnover <= max_turnover:
        return new

    # Linearly blend toward prev until turnover is satisfied
    # new_blend = α * new + (1-α) * prev
    # turnover_blend = α * turnover_raw → α = max_turnover / turnover
    alpha = max_turnover / turnover
    blended = alpha * new + (1 - alpha) * prev

    # Re-normalize long and short legs
    long_mask  = blended > 0
    short_mask = blended < 0

    if long_mask.sum() > 0:
        blended[long_mask] /= blended[long_mask].sum()
    if short_mask.sum() > 0:
        blended[short_mask] /= blended[short_mask].abs().sum()
        blended[short_mask] *= -1

    return blended
