"""
Information Coefficient (IC) analysis framework.

The IC is the Spearman rank correlation between factor scores and
subsequent returns. It is the primary metric used at systematic quant funds
to evaluate whether a signal has predictive power.

Key metrics:
  IC       : single-period rank correlation ∈ [-1, 1]
  Mean IC  : average IC over time — the "expected predictive power"
  IC Std   : volatility of IC — stability matters as much as the mean
  ICIR     : IC Information Ratio = Mean IC / Std IC
             The Sharpe ratio of the signal. ICIR > 0.3 is considered good.
  IC t-stat: ICIR × √T — the statistical significance
  IC Decay : how fast IC falls as the forecast horizon lengthens

Benchmarks (from academic and practitioner experience):
  Mean IC > 0.03  : weak but potentially useful signal
  Mean IC > 0.05  : solid signal
  Mean IC > 0.10  : very strong signal (rare in large-cap equity)
  ICIR > 0.3      : good signal quality
  ICIR > 0.5      : excellent signal quality
"""

from __future__ import annotations

from scipy import stats
import numpy as np
import pandas as pd


class ICAnalysis:
    """
    IC analysis for a single factor.

    Usage:
        ica = ICAnalysis(factor, returns)
        ica.fit()
        print(ica.summary())
        ica.plot_ic_decay(horizons=[1, 5, 10, 21, 63])
    """

    def __init__(
        self,
        factor: pd.DataFrame,
        returns_dict: dict[int, pd.DataFrame],
        factor_name: str = "factor",
        min_stocks: int = 20,
    ):
        """
        Parameters
        ----------
        factor       : (date × ticker) factor scores (z-scored)
        returns_dict : dict mapping horizon (days) → (date × ticker) fwd returns
        factor_name  : label for plots and summary
        """
        self.factor = factor
        self.returns_dict = returns_dict
        self.factor_name = factor_name
        self.min_stocks = min_stocks
        self._ic_series: dict[int, pd.Series] = {}

    def fit(self) -> "ICAnalysis":
        """Compute IC time series for all horizons."""
        for horizon, ret_df in self.returns_dict.items():
            self._ic_series[horizon] = _compute_ic_series(
                self.factor,
                ret_df,
                min_stocks=self.min_stocks,
            )
        return self

    def summary(self, horizon: int = 21) -> dict:
        """
        Return key IC statistics for a given horizon.

        Parameters
        ----------
        horizon : forecast horizon in trading days (default 21 = 1 month)
        """
        if horizon not in self._ic_series:
            raise ValueError(f"Horizon {horizon} not computed. Call fit() first.")

        ic = self._ic_series[horizon].dropna()
        T  = len(ic)

        mean_ic = float(ic.mean())
        std_ic  = float(ic.std(ddof=1))
        icir    = mean_ic / std_ic if std_ic > 0 else 0.0
        t_stat  = icir * np.sqrt(T)
        p_val   = 2 * (1 - stats.t.cdf(abs(t_stat), df=T - 1))
        pct_pos = float((ic > 0).mean())

        return {
            "factor":       self.factor_name,
            "horizon_days": horizon,
            "mean_ic":      mean_ic,
            "std_ic":       std_ic,
            "icir":         icir,
            "t_stat":       t_stat,
            "p_value":      p_val,
            "pct_positive": pct_pos,
            "n_periods":    T,
            "min_ic":       float(ic.min()),
            "max_ic":       float(ic.max()),
        }

    def ic_decay(self, horizons: list[int] | None = None) -> pd.DataFrame:
        """
        Return IC statistics across multiple horizons — the IC decay curve.

        A fast-decaying IC means the signal is short-lived (high turnover needed).
        A slow-decaying IC allows lower-frequency rebalancing (lower costs).

        Returns
        -------
        DataFrame indexed by horizon with columns: mean_ic, icir, t_stat
        """
        horizons = horizons or sorted(self._ic_series.keys())
        rows = []
        for h in horizons:
            if h in self._ic_series:
                rows.append(self.summary(h))
        return pd.DataFrame(rows).set_index("horizon_days")

    def ic_series(self, horizon: int = 21) -> pd.Series:
        """Return the raw IC time series for a given horizon."""
        return self._ic_series.get(horizon, pd.Series(dtype=float))

    def cumulative_ic(self, horizon: int = 21) -> pd.Series:
        """Cumulative sum of IC — useful for spotting regime changes."""
        return self._ic_series[horizon].dropna().cumsum()


def compute_ic_table(
    factors: dict[str, pd.DataFrame],
    returns_dict: dict[int, pd.DataFrame],
    primary_horizon: int = 21,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """
    Compute IC summary for all factors at the primary horizon.

    Returns a ranked DataFrame — the go-to table for factor selection.

    Parameters
    ----------
    factors         : dict of factor_name → (date × ticker) factor scores
    returns_dict    : dict of horizon → (date × ticker) forward returns
    primary_horizon : main evaluation horizon in trading days

    Returns
    -------
    DataFrame sorted by |ICIR|, one row per factor
    """
    rows = []
    for fname, fdata in factors.items():
        ica = ICAnalysis(fdata, returns_dict, factor_name=fname, min_stocks=min_stocks)
        ica.fit()
        try:
            row = ica.summary(primary_horizon)
            rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows).set_index("factor")
    df = df.sort_values("icir", key=abs, ascending=False)
    return df


def compute_ic_decay_table(
    factors: dict[str, pd.DataFrame],
    returns_dict: dict[int, pd.DataFrame],
    horizons: list[int] = [1, 5, 10, 21, 63, 126],
    min_stocks: int = 20,
) -> pd.DataFrame:
    """
    Full IC decay table: (factor × horizon) → mean IC.

    Shows how signal strength evolves with forecast horizon.
    Useful for determining optimal rebalancing frequency.
    """
    rows = {}
    for fname, fdata in factors.items():
        ica = ICAnalysis(fdata, returns_dict, factor_name=fname, min_stocks=min_stocks)
        ica.fit()
        row = {}
        for h in horizons:
            if h in returns_dict:
                try:
                    row[h] = ica.summary(h)["mean_ic"]
                except Exception:
                    row[h] = np.nan
        rows[fname] = row

    return pd.DataFrame(rows).T.rename(columns={h: f"{h}d" for h in horizons})


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _compute_ic_series(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    method: str = "spearman",
    min_stocks: int = 20,
) -> pd.Series:
    """
    Compute period-by-period IC (rank correlation) between factor and returns.

    Rank IC (Spearman) is preferred over Pearson because:
    1. More robust to outliers — returns and factors have fat tails
    2. Tests whether the *ordering* is correct, not the magnitude
    3. Standard in the industry (most IC reports use Spearman)

    Parameters
    ----------
    factor  : (date × ticker) factor scores
    returns : (date × ticker) forward returns (aligned: factor[t] predicts returns[t])
    method  : 'spearman' or 'pearson'

    Returns
    -------
    pd.Series of IC values indexed by date
    """
    common_dates = factor.index.intersection(returns.index)
    ic_values = {}

    for date in common_dates:
        f_t = factor.loc[date].dropna()
        r_t = returns.loc[date].dropna()

        common_tickers = f_t.index.intersection(r_t.index)
        if len(common_tickers) < min_stocks:
            continue

        f_aligned = f_t[common_tickers]
        r_aligned = r_t[common_tickers]

        if method == "spearman":
            ic, _ = stats.spearmanr(f_aligned, r_aligned)
        else:
            ic = np.corrcoef(f_aligned, r_aligned)[0, 1]

        ic_values[date] = ic

    return pd.Series(ic_values, name="ic")
