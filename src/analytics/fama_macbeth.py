"""
Fama-MacBeth (1973) cross-sectional regression framework.

Methodology:
  Step 1 — For each time period t, run a cross-sectional OLS regression:
              R_{i,t} = α_t + λ_{1,t}·f_{i,t-1} + ... + λ_{k,t}·f_{k,i,t-1} + ε_{i,t}

            This gives one set of slope coefficients (λ_t) per period.

  Step 2 — Test whether the time-series mean of λ_t is significantly different
            from zero:
              t-stat = mean(λ_t) / (se(λ_t) / √T)

            Standard errors use Newey-West (HAC) adjustment to account for
            serial correlation in λ_t — critical because factor returns
            are autocorrelated (momentum persists, value reverts slowly).

Why Fama-MacBeth:
  - Standard panel OLS would understate standard errors due to cross-sectional
    error correlation (all stocks exposed to common macro shocks on same day)
  - Fama-MacBeth sidesteps this by running one regression *per period*, then
    averaging — the t-test is on the time series of cross-section estimates

References:
  Fama, E.F. & MacBeth, J.D. (1973). Risk, Return, and Equilibrium:
  Empirical Tests. Journal of Political Economy.

  Newey, W.K. & West, K.D. (1987). A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  Econometrica.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


class FamaMacBeth:
    """
    Fama-MacBeth cross-sectional regression.

    Parameters
    ----------
    nw_lags : int
        Number of lags for Newey-West HAC standard error correction.
        Rule of thumb: floor(4 * (T/100)^(2/9)) — typically 4-8 lags
        for monthly data over 5 years.
    min_stocks : int
        Minimum number of stocks required in each cross-section.
        Periods with fewer stocks are dropped (data quality filter).
    """

    def __init__(self, nw_lags: int = 6, min_stocks: int = 50):
        self.nw_lags = nw_lags
        self.min_stocks = min_stocks
        self._lambdas: pd.DataFrame | None = None
        self._r2s: pd.Series | None = None
        self._n_stocks: pd.Series | None = None

    def fit(
        self,
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        controls: dict[str, pd.DataFrame] | None = None,
    ) -> "FamaMacBeth":
        """
        Run Fama-MacBeth regressions for a single factor.

        Parameters
        ----------
        factor   : (date × ticker) factor scores (pre-standardized)
        returns  : (date × ticker) forward returns
        controls : optional additional factors to include as controls
                   (e.g., size, beta) to test for incremental explanatory power

        Returns
        -------
        self (for chaining)
        """
        # Align dates
        common_dates = factor.index.intersection(returns.index)
        factor  = factor.loc[common_dates]
        returns = returns.loc[common_dates]

        lambdas, r2s, n_stocks = [], [], []

        for date in common_dates:
            f_t = factor.loc[date].dropna()
            r_t = returns.loc[date].dropna()

            # Align to same tickers
            common = f_t.index.intersection(r_t.index)
            if len(common) < self.min_stocks:
                continue

            f_t = f_t[common].values
            r_t = r_t[common].values

            # Build design matrix X = [1, f] or [1, f, controls...]
            if controls:
                ctrl_cols = []
                for ctrl_name, ctrl_df in controls.items():
                    if date in ctrl_df.index:
                        c_t = ctrl_df.loc[date].reindex(common).fillna(0).values
                        ctrl_cols.append(c_t)
                X = np.column_stack([np.ones(len(common)), f_t] + ctrl_cols)
            else:
                X = np.column_stack([np.ones(len(common)), f_t])

            # OLS: λ = (X'X)^{-1} X'R
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coef, res, rank, sv = np.linalg.lstsq(X, r_t, rcond=None)
            except np.linalg.LinAlgError:
                continue

            lambdas.append(coef)

            # R² for this cross-section
            y_hat = X @ coef
            ss_res = np.sum((r_t - y_hat) ** 2)
            ss_tot = np.sum((r_t - r_t.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2s.append(r2)
            n_stocks.append(len(common))

        if not lambdas:
            raise ValueError("No valid cross-sections found — check data alignment")

        col_names = ["alpha", "lambda"] + ([f"ctrl_{k}" for k in (controls or {}).keys()])
        self._lambdas = pd.DataFrame(lambdas, index=common_dates[:len(lambdas)], columns=col_names)
        self._r2s = pd.Series(r2s, name="r2")
        self._n_stocks = pd.Series(n_stocks, name="n_stocks")
        return self

    def summary(self) -> pd.DataFrame:
        """
        Return the Fama-MacBeth summary table.

        Columns:
          mean_lambda : time-series mean of cross-sectional slope coefficients
          std_lambda  : standard deviation of λ_t
          t_stat_ols  : naive t-stat (ignores autocorrelation)
          t_stat_nw   : Newey-West adjusted t-stat (use this one)
          p_value     : two-tailed p-value of NW t-stat
          annualized  : annualized factor return (mean_lambda × 12 for monthly)
          mean_r2     : average cross-sectional R²
          n_periods   : number of cross-sections used
        """
        if self._lambdas is None:
            raise RuntimeError("Must call .fit() before .summary()")

        rows = []
        for col in self._lambdas.columns:
            lam = self._lambdas[col].dropna().values
            T = len(lam)

            mean_lam = lam.mean()
            std_lam  = lam.std(ddof=1)
            se_ols   = std_lam / np.sqrt(T)
            t_ols    = mean_lam / se_ols if se_ols > 0 else np.nan

            # Newey-West standard error
            se_nw  = _newey_west_se(lam, self.nw_lags)
            t_nw   = mean_lam / se_nw if se_nw > 0 else np.nan
            p_val  = 2 * (1 - stats.t.cdf(abs(t_nw), df=T - 1)) if not np.isnan(t_nw) else np.nan

            rows.append({
                "coefficient":   col,
                "mean_lambda":   mean_lam,
                "std_lambda":    std_lam,
                "t_stat_ols":    t_ols,
                "t_stat_nw":     t_nw,
                "p_value_nw":    p_val,
                "significant":   abs(t_nw) > 2.0 if not np.isnan(t_nw) else False,
                "annualized":    mean_lam * 12,   # assumes monthly rebalancing
                "mean_r2":       float(self._r2s.mean()),
                "n_periods":     T,
            })

        return pd.DataFrame(rows).set_index("coefficient")

    @property
    def lambdas(self) -> pd.DataFrame:
        """Time series of cross-sectional slope coefficients λ_t."""
        return self._lambdas

    @property
    def mean_r2(self) -> float:
        return float(self._r2s.mean()) if self._r2s is not None else np.nan


def run_fama_macbeth_panel(
    factors: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    frequency: str = "M",
    nw_lags: int = 6,
) -> pd.DataFrame:
    """
    Run Fama-MacBeth for every factor in the registry.

    Resamples data to monthly frequency (standard for cross-sectional work —
    daily noise drowns the signal, and monthly aligns with rebalancing cycles).

    Parameters
    ----------
    factors   : dict of factor_name → (date × ticker) DataFrame
    returns   : (date × ticker) forward 1-month returns
    frequency : pandas resample frequency ('M' = monthly, 'W' = weekly)
    nw_lags   : Newey-West lags for t-stat correction

    Returns
    -------
    DataFrame with one row per factor, columns = summary statistics
    """
    # Resample to monthly: use end-of-month values
    def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
        return df.resample(frequency.replace("M", "ME")).last()

    ret_m = to_monthly(returns)
    results = []

    for fname, fdata in factors.items():
        f_m = to_monthly(fdata)  # noqa
        try:
            fm = FamaMacBeth(nw_lags=nw_lags)
            fm.fit(f_m, ret_m)
            row = fm.summary().loc["lambda"].to_dict()
            row["factor"] = fname
            results.append(row)
        except Exception:
            results.append({
                "factor": fname,
                "mean_lambda": np.nan,
                "std_lambda": np.nan,
                "t_stat_ols": np.nan,
                "t_stat_nw": np.nan,
                "p_value_nw": np.nan,
                "significant": False,
                "annualized": np.nan,
                "mean_r2": np.nan,
                "n_periods": 0,
            })

    return pd.DataFrame(results).set_index("factor").sort_values("t_stat_nw", ascending=False)


# ------------------------------------------------------------------
# Newey-West HAC standard error
# ------------------------------------------------------------------

def _newey_west_se(x: np.ndarray, lags: int) -> float:
    """
    Compute Newey-West heteroskedasticity and autocorrelation consistent (HAC)
    standard error for a time series x.

    Formula:
        Var_NW = γ_0 + 2 * Σ_{l=1}^{L} w_l * γ_l

    where γ_l = autocovariance at lag l, and the Bartlett kernel weights are:
        w_l = 1 - l / (L + 1)   (linearly declining)

    This downweights higher-lag autocorrelations, ensuring the variance
    estimate is positive semi-definite.
    """
    T = len(x)
    x_demean = x - x.mean()

    # Lag-0 variance
    var = np.sum(x_demean ** 2) / T

    # Add autocorrelation corrections
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)  # Bartlett kernel
        gamma_l = np.sum(x_demean[lag:] * x_demean[:-lag]) / T
        var += 2 * weight * gamma_l

    var = max(var, 1e-12)  # numerical safety
    return np.sqrt(var / T)
