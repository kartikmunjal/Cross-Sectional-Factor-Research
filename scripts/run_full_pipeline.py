#!/usr/bin/env python3
"""
Full factor research pipeline: data → factors → IC → Fama-MacBeth → backtest.

This is the single script that runs the entire research workflow end-to-end.
Results are printed to stdout and saved to results/.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --n-tickers 150 --save
    python scripts/run_full_pipeline.py --factors MOM_12_1 MOM_6_1 IVOL --save
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive for script mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch import (
    get_sp500_universe, load_close_panel, load_volume_panel,
    load_market_returns, download_fundamentals, compute_returns,
)
from src.factors.registry import build_all_factors, factor_summary_table, FACTOR_REGISTRY
from src.factors.altdata import load_alt_factor_panels, merge_factor_panels
from src.analytics.ic_analysis import compute_ic_table, compute_ic_decay_table
from src.analytics.fama_macbeth import run_fama_macbeth_panel
from src.portfolio.ranking import MultiFactorRanking, quintile_returns
from src.backtest.walk_forward import WalkForwardBacktest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    parser = argparse.ArgumentParser(description="Cross-sectional factor research pipeline")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--n-tickers", type=int, default=None)
    parser.add_argument("--factors", nargs="+", default=None,
                        help="Subset of factors to run (default: all)")
    parser.add_argument("--alt-factor-dir", default=None,
                        help="Directory of exported alt-data parquet panels, e.g. ../alt-data-equity-signals/results/run/factor_panels")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Data
    # ----------------------------------------------------------------
    log.info("=== STEP 1: Loading data ===")
    tickers = get_sp500_universe()
    if args.n_tickers:
        tickers = tickers[:args.n_tickers]

    close        = load_close_panel(tickers, start=args.start)
    volume       = load_volume_panel(tickers, start=args.start)
    mkt_returns  = load_market_returns(start=args.start)
    fundamentals = download_fundamentals(tickers)
    returns_dict = compute_returns(close, periods=[1, 5, 21, 63])

    log.info(f"Panel: {close.shape[0]} days × {close.shape[1]} tickers")

    # ----------------------------------------------------------------
    # 2. Factors
    # ----------------------------------------------------------------
    log.info("=== STEP 2: Computing factors ===")
    factor_meta = factor_summary_table()
    print("\nFACTOR LIBRARY:")
    print(factor_meta.to_string(index=False))

    factors = build_all_factors(
        close=close,
        volume=volume,
        market_returns=mkt_returns,
        fundamentals=fundamentals,
        factor_names=args.factors,
        preprocess=True,
    )
    if args.alt_factor_dir:
        alt_factors = load_alt_factor_panels(
            args.alt_factor_dir,
            tickers=tickers,
            start=args.start,
        )
        factors = merge_factor_panels(factors, alt_factors)
        log.info(f"Loaded {len(alt_factors)} alt-data factors from {args.alt_factor_dir}")
    log.info(f"Computed {len(factors)} factors")

    # ----------------------------------------------------------------
    # 3. IC Analysis
    # ----------------------------------------------------------------
    log.info("=== STEP 3: IC Analysis (primary horizon: 21d) ===")
    ic_table = compute_ic_table(factors, returns_dict, primary_horizon=21)

    print("\n" + "=" * 70)
    print("  IC SUMMARY TABLE — 21-day horizon")
    print("=" * 70)
    display_cols = ["mean_ic", "std_ic", "icir", "t_stat", "p_value", "pct_positive", "n_periods"]
    print(ic_table[display_cols].round(4).to_string())
    print("=" * 70)
    print("  Benchmark: Mean IC > 0.03 = useful | ICIR > 0.3 = strong signal")

    if args.save:
        ic_table.to_csv(RESULTS_DIR / "ic_summary.csv")

    # IC decay
    ic_decay = compute_ic_decay_table(factors, returns_dict)
    print("\n  IC DECAY TABLE (Mean IC by horizon):")
    print(ic_decay.round(4).to_string())

    if args.save:
        ic_decay.to_csv(RESULTS_DIR / "ic_decay.csv")

    # ----------------------------------------------------------------
    # 4. Fama-MacBeth
    # ----------------------------------------------------------------
    log.info("=== STEP 4: Fama-MacBeth regressions ===")
    fm_table = run_fama_macbeth_panel(factors, returns_dict[21])

    print("\n" + "=" * 70)
    print("  FAMA-MACBETH RESULTS (Newey-West t-stats, monthly)")
    print("=" * 70)
    fm_cols = ["mean_lambda", "t_stat_nw", "p_value_nw", "significant", "annualized", "mean_r2"]
    print(fm_table[fm_cols].round(4).to_string())
    print("=" * 70)
    print("  Interpretation: |t| > 2.0 = significant at 5% level")

    if args.save:
        fm_table.to_csv(RESULTS_DIR / "fama_macbeth.csv")

    # ----------------------------------------------------------------
    # 5. Quintile Portfolios
    # ----------------------------------------------------------------
    log.info("=== STEP 5: Quintile portfolio analysis ===")
    # Use top-3 factors by ICIR for composite
    top_factors = ic_table["icir"].abs().nlargest(3).index.tolist()
    log.info(f"Composite factors: {top_factors}")

    ranker = MultiFactorRanking()
    all_dates = returns_dict[21].index
    rebal_dates = pd.DatetimeIndex(
        pd.Series(all_dates, index=all_dates).resample("ME").last().dropna()
    )

    quintile_ret = quintile_returns(
        {k: factors[k] for k in top_factors if k in factors},
        returns_dict[21],
        rebal_dates=rebal_dates,
        ranker=ranker,
    )

    print("\n  QUINTILE RETURNS (1=Best, 5=Worst, Q1-Q5=L/S spread):")
    print(quintile_ret.mean().round(4).to_string())
    print(f"  Monotonicity: {_check_monotonicity(quintile_ret)}")

    if args.save:
        quintile_ret.to_csv(RESULTS_DIR / "quintile_returns.csv")

    # ----------------------------------------------------------------
    # 6. Walk-Forward Backtest
    # ----------------------------------------------------------------
    log.info("=== STEP 6: Walk-forward backtest ===")
    bt = WalkForwardBacktest(transaction_cost=0.0005, max_turnover=0.30)

    composite_factors = {k: factors[k] for k in top_factors if k in factors}
    bt_results = bt.run(composite_factors, returns_dict[21])
    stats = WalkForwardBacktest.summarize(bt_results)

    print("\n" + "=" * 70)
    print(f"  WALK-FORWARD BACKTEST — {', '.join(top_factors)}")
    print("=" * 70)
    print(f"  Ann. Gross Return:   {stats['ann_gross_ret']:>8.1%}")
    print(f"  Ann. Net Return:     {stats['ann_net_ret']:>8.1%}   (after 5bp TC)")
    print(f"  Net Sharpe Ratio:    {stats['net_sharpe']:>8.2f}")
    print(f"  Max Drawdown:        {stats['max_drawdown']:>8.1%}")
    print(f"  Hit Rate (monthly):  {stats['hit_rate']:>8.1%}")
    print(f"  Avg Turnover:        {stats['avg_turnover']:>8.1%}/period")
    print(f"  TC Drag (annual):    {stats['avg_tc_drag_ann']:>8.2%}")
    print(f"  Total Net Return:    {stats['total_net']:>8.1%}")
    print("=" * 70)

    # Factor attribution
    attribution = WalkForwardBacktest.factor_attribution(
        composite_factors, returns_dict[21], rebal_dates
    )
    print("\n  SINGLE-FACTOR ATTRIBUTION:")
    print(attribution.round(4).to_string())

    if args.save:
        bt_results.to_csv(RESULTS_DIR / "backtest_results.csv")
        attribution.to_csv(RESULTS_DIR / "factor_attribution.csv")

    # ----------------------------------------------------------------
    # 7. Plots
    # ----------------------------------------------------------------
    log.info("=== STEP 7: Generating plots ===")
    _plot_all(ic_table, fm_table, quintile_ret, bt_results, attribution, args.save)
    log.info("Pipeline complete.")


def _check_monotonicity(quintile_ret: pd.DataFrame) -> str:
    means = [quintile_ret[f"Q{q}"].mean() for q in range(1, 6) if f"Q{q}" in quintile_ret.columns]
    is_mono = all(means[i] >= means[i+1] for i in range(len(means)-1))
    return "YES ✓" if is_mono else "PARTIAL"


def _plot_all(ic_table, fm_table, quintile_ret, bt_results, attribution, save):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. IC bar chart
    ax = fig.add_subplot(gs[0, :2])
    ic_sorted = ic_table["mean_ic"].sort_values(ascending=True)
    colors = ["seagreen" if v > 0 else "crimson" for v in ic_sorted]
    ic_sorted.plot.barh(ax=ax, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.03, color="green", lw=1, ls="--", alpha=0.5, label="IC=0.03 threshold")
    ax.set_title("Factor Mean IC (21-day horizon)", fontsize=12)
    ax.set_xlabel("Mean IC (Spearman rank correlation)")
    ax.legend(fontsize=8)

    # 2. ICIR bar chart
    ax = fig.add_subplot(gs[0, 2])
    icir_sorted = ic_table["icir"].sort_values(ascending=True)
    colors = ["steelblue" if v > 0 else "crimson" for v in icir_sorted]
    icir_sorted.plot.barh(ax=ax, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(0.3, color="green", lw=1, ls="--", alpha=0.5, label="ICIR=0.3")
    ax.set_title("Factor ICIR", fontsize=12)
    ax.legend(fontsize=8)

    # 3. FM t-stats
    ax = fig.add_subplot(gs[1, :2])
    fm_valid = fm_table["t_stat_nw"].dropna().sort_values(ascending=True)
    colors = ["seagreen" if v > 2 else ("crimson" if v < -2 else "steelblue") for v in fm_valid]
    fm_valid.plot.barh(ax=ax, color=colors, alpha=0.8)
    ax.axvline(2.0, color="green", lw=1, ls="--", alpha=0.6, label="|t|=2.0")
    ax.axvline(-2.0, color="green", lw=1, ls="--", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Fama-MacBeth t-statistics (Newey-West)", fontsize=12)
    ax.set_xlabel("t-statistic (|t| > 2 = significant at 5%)")
    ax.legend(fontsize=8)

    # 4. Quintile bar chart
    ax = fig.add_subplot(gs[1, 2])
    q_means = quintile_ret[[c for c in quintile_ret.columns if c.startswith("Q")]].mean()
    q_means = q_means[[c for c in q_means.index if c != "Q1-Q5"]]
    q_colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(q_means)))
    q_means.plot.bar(ax=ax, color=q_colors, alpha=0.9, rot=0)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Avg Monthly Return by Quintile\n(Q1=Best, Q5=Worst)", fontsize=10)
    ax.set_ylabel("Mean Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

    # 5. Backtest equity curve
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(bt_results.index, bt_results["cumulative_gross"],
            color="steelblue", lw=1.5, label="Gross")
    ax.plot(bt_results.index, bt_results["cumulative_net"],
            color="seagreen", lw=2, label="Net (after 5bp TC)")
    ax.fill_between(bt_results.index, bt_results["cumulative_net"], 1,
                    where=bt_results["cumulative_net"] >= 1, alpha=0.1, color="green")
    ax.fill_between(bt_results.index, bt_results["cumulative_net"], 1,
                    where=bt_results["cumulative_net"] < 1, alpha=0.2, color="red")
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_title("Cumulative Return — Multi-Factor L/S Portfolio", fontsize=12)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)

    # 6. Drawdown
    ax = fig.add_subplot(gs[2, 2])
    ax.fill_between(bt_results.index, bt_results["drawdown"] * 100, 0,
                    color="crimson", alpha=0.7)
    ax.set_title("Drawdown (%)", fontsize=12)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.suptitle("Cross-Sectional Factor Research — S&P 500 Universe", fontsize=14, y=1.01)

    if save:
        path = FIGURES_DIR / "factor_research_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved summary figure to {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
