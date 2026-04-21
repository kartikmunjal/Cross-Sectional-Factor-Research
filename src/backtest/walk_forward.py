"""
Walk-forward backtest engine.

Methodology:
  - Monthly rebalancing on the last trading day of each month
  - Long top quintile, short bottom quintile (dollar-neutral)
  - Transaction costs: 5 bps one-way (conservative for large-cap equity)
  - Turnover control: max 30% one-way turnover per period
  - No lookahead: factors at t predict returns from t to t+horizon

Walk-forward vs. expanding window:
  We use a pure walk-forward approach (fixed lookback) rather than expanding,
  because expanding windows weight early (potentially unrepresentative) periods
  heavily as time progresses. Fixed lookback keeps the factor computation
  stationary across the test period.

Performance attribution:
  - Total return decomposed into: factor selection (alpha) vs. market (beta)
  - Gross vs. net returns (before and after transaction costs)
  - Per-quintile return contribution
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.portfolio.ranking import MultiFactorRanking, apply_turnover_control

log = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtest for a multi-factor long-short equity strategy.

    Parameters
    ----------
    transaction_cost : one-way cost per unit of turnover (default 5 bps)
    max_turnover     : maximum one-way turnover per period (default 30%)
    n_quintiles      : number of portfolio buckets (default 5)
    rebal_freq       : rebalancing frequency ('M' = monthly, 'W' = weekly)
    """

    def __init__(
        self,
        transaction_cost: float = 0.0005,
        max_turnover: float = 0.30,
        n_quintiles: int = 5,
        rebal_freq: str = "M",
    ):
        self.tc = transaction_cost
        self.max_turnover = max_turnover
        self.n_quintiles = n_quintiles
        self.rebal_freq = rebal_freq

    def run(
        self,
        factors: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        factor_weights: dict[str, float] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        factors        : dict of factor_name → (date × ticker) z-scored scores
        returns        : (date × ticker) 1-month forward returns
        factor_weights : weight per factor (None = equal weight)
        start, end     : optional date range filter

        Returns
        -------
        DataFrame with columns:
          gross_ret, net_ret, turnover, n_long, n_short,
          cumulative_gross, cumulative_net, drawdown
        """
        ranker = MultiFactorRanking(factor_weights, n_quintiles=self.n_quintiles)

        # Determine rebalancing dates
        all_dates = returns.index
        if start:
            all_dates = all_dates[all_dates >= start]
        if end:
            all_dates = all_dates[all_dates <= end]

        freq = self.rebal_freq.replace("M", "ME").replace("W", "W-FRI")
        rebal_dates = pd.DatetimeIndex(
            pd.Series(all_dates, index=all_dates).resample(freq).last().dropna()
        )

        records = []
        prev_weights = pd.Series(dtype=float)

        for date in rebal_dates:
            # Build composite score
            composite = ranker.composite_score(factors, date)
            if composite.empty or len(composite) < self.n_quintiles:
                continue

            # Target weights from signal
            target_weights = ranker.build_portfolio(composite)

            # Apply turnover control
            controlled_weights = apply_turnover_control(
                target_weights, prev_weights, self.max_turnover
            )

            # Compute actual turnover
            all_t = controlled_weights.index.union(prev_weights.index)
            new_w  = controlled_weights.reindex(all_t).fillna(0)
            prev_w = prev_weights.reindex(all_t).fillna(0)
            turnover = (new_w - prev_w).abs().sum() / 2

            # Get forward returns for this period
            if date not in returns.index:
                continue

            fwd_ret = returns.loc[date]
            common = controlled_weights.index.intersection(fwd_ret.index)
            w = controlled_weights[common]
            r = fwd_ret[common]

            gross_ret = float((w * r).sum())
            tc_drag   = float(turnover * self.tc * 2)  # two-way
            net_ret   = gross_ret - tc_drag

            n_long  = int((w > 0).sum())
            n_short = int((w < 0).sum())

            records.append({
                "date":      date,
                "gross_ret": gross_ret,
                "net_ret":   net_ret,
                "turnover":  turnover,
                "tc_drag":   tc_drag,
                "n_long":    n_long,
                "n_short":   n_short,
                "n_stocks":  n_long + n_short,
            })

            prev_weights = controlled_weights

        if not records:
            raise ValueError("No valid rebalancing periods found.")

        df = pd.DataFrame(records).set_index("date")
        df["cumulative_gross"] = (1 + df["gross_ret"]).cumprod()
        df["cumulative_net"]   = (1 + df["net_ret"]).cumprod()

        # Drawdown
        roll_max = df["cumulative_net"].cummax()
        df["drawdown"] = df["cumulative_net"] / roll_max - 1

        return df

    @staticmethod
    def summarize(results: pd.DataFrame) -> dict:
        """
        Compute strategy performance metrics.

        Returns
        -------
        dict with annualized return, Sharpe, max drawdown, turnover, hit rate
        """
        n = len(results)
        periods_per_year = 12  # monthly

        gross = results["gross_ret"]
        net   = results["net_ret"]

        def _sharpe(ret):
            mu  = ret.mean() * periods_per_year
            sig = ret.std() * np.sqrt(periods_per_year)
            return mu / sig if sig > 0 else 0.0

        return {
            "ann_gross_ret":     float(gross.mean() * periods_per_year),
            "ann_net_ret":       float(net.mean() * periods_per_year),
            "gross_sharpe":      _sharpe(gross),
            "net_sharpe":        _sharpe(net),
            "max_drawdown":      float(results["drawdown"].min()),
            "avg_turnover":      float(results["turnover"].mean()),
            "avg_tc_drag_ann":   float(results["tc_drag"].mean() * periods_per_year),
            "hit_rate":          float((net > 0).mean()),
            "n_periods":         n,
            "avg_n_stocks":      float(results["n_stocks"].mean()),
            "total_gross":       float((1 + gross).prod() - 1),
            "total_net":         float((1 + net).prod() - 1),
        }

    @staticmethod
    def factor_attribution(
        factors: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
        rebal_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Attribute portfolio return to individual factor contributions.

        For each factor, run the backtest independently (single-factor portfolio)
        and report annualized return and Sharpe. Shows which factors drive
        composite performance.

        Returns
        -------
        DataFrame indexed by factor with columns: ann_net_ret, sharpe, hit_rate
        """
        rows = []
        bt = WalkForwardBacktest()

        for fname, fdata in factors.items():
            try:
                res = bt.run({fname: fdata}, returns)
                stats = WalkForwardBacktest.summarize(res)
                rows.append({
                    "factor":       fname,
                    "ann_net_ret":  stats["ann_net_ret"],
                    "net_sharpe":   stats["net_sharpe"],
                    "max_dd":       stats["max_drawdown"],
                    "hit_rate":     stats["hit_rate"],
                    "avg_turnover": stats["avg_turnover"],
                })
            except Exception as e:
                log.warning(f"Attribution for {fname} failed: {e}")

        return pd.DataFrame(rows).set_index("factor").sort_values("net_sharpe", ascending=False)
