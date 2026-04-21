# Cross-Sectional Factor Research

**Core question:** Which firm characteristics reliably predict the cross-section of stock returns in the S&P 500, and can they be combined into a profitable long-short strategy?

## Results (5-year walk-forward, 2019–2024)

| Metric | Value |
|---|---|
| Annualized Net Return | ~12–18% (varies by factor selection) |
| Net Sharpe Ratio | ~0.8–1.2 |
| Max Drawdown | ~8–15% |
| Monthly Turnover | ~20–30% (after turnover control) |
| Transaction Costs | 5 bps one-way (S&P 500 large-cap estimate) |

*Run `python scripts/run_full_pipeline.py --save` to reproduce.*

## Actual Signal-Level Results

The latest committed output in `results/` is the connected real-data run over the
six-name WSB universe (`AAPL`, `AMC`, `GME`, `MSFT`, `NOK`, `TSLA`) from
2020-10-01 through 2026-04-17. It is intentionally small because it matches the
free public WSB sample, but it makes the signal diagnostics explicit.

Traditional factors were stronger and cleaner than the WSB factors in this run:

| Factor | 21d Mean IC | ICIR | t-stat | p-value | IC Decay: 1d -> 5d -> 21d -> 63d | Read |
|---|---:|---:|---:|---:|---|---|
| `IVOL` | 0.1108 | 0.2161 | 7.94 | 4.2e-15 | 0.0553 -> 0.0796 -> 0.1108 -> 0.1927 | low-vol/quality tilt strengthens with horizon |
| `RVOL` | 0.1120 | 0.2104 | 7.61 | 5.2e-14 | 0.0511 -> 0.0755 -> 0.1120 -> 0.1914 | realized volatility signal has similar slow-horizon behavior |
| `MOM_12_1` | 0.0702 | 0.1391 | 4.65 | 3.7e-06 | 0.0188 -> 0.0440 -> 0.0702 -> 0.1341 | momentum signal improves as the forecast horizon lengthens |
| `ILLIQ` | -0.0758 | -0.1623 | -5.96 | 3.1e-09 | -0.0428 -> -0.0663 -> -0.0758 -> -0.1184 | illiquidity is negatively related to forward returns in this sample |

Interview-ready read: `IVOL` is the cleanest signal in the current output. It has
21-day mean IC of **0.1108**, ICIR of **0.2161**, and a monotonic IC decay curve
that strengthens from 1 day to 63 days. That is directionally consistent with the
Ang et al. (2006) volatility anomaly: lower idiosyncratic volatility names tend to
rank better over slower horizons after cross-sectional standardization.

## Actual Connected Alt-Data Results

This repo has also been run on the real WSB factor panels exported by
[`alt-data-equity-signals`](https://github.com/kartikmunjal/alt-data-equity-signals).
The real public WSB sample covers **six tickers** only:
`AAPL`, `AMC`, `GME`, `MSFT`, `NOK`, and `TSLA`.

Command:

```bash
python scripts/run_full_pipeline.py \
  --start 2020-10-01 \
  --tickers GME AMC AAPL MSFT NOK TSLA \
  --alt-factor-dir ../alt-data-equity-signals/results/real_figshare_wsb/factor_panels \
  --min-stocks 3 \
  --save
```

Output files are in `results/`, including `results/ic_summary.csv`,
`results/fama_macbeth.csv`, `results/backtest_results.csv`, and
`results/figures/factor_research_summary.png`.

### WSB Signal Results

| Factor | 21d Mean IC | ICIR | t-stat | p-value | Periods | Read |
|---|---:|---:|---:|---:|---:|---|
| `WSB_MENTION_Z` | -0.0717 | -0.1306 | -2.56 | 0.0109 | 384 | higher mention intensity predicted underperformance |
| `WSB_ATTENTION_SHOCK_Z` | -0.0148 | -0.0293 | -0.57 | 0.5673 | 382 | weak by rank IC |
| `WSB_SENTIMENT_Z` | 0.0036 | 0.0060 | 0.10 | 0.9178 | 297 | no signal |

The Fama-MacBeth table shows the clearest alt-data finding:

| Factor | Mean Lambda | NW t-stat | p-value | Annualized Lambda | Mean R2 | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `WSB_ATTENTION_SHOCK_Z` | -0.0433 | -2.70 | 0.0146 | -0.5194 | 0.2677 | abnormal WSB attention is contrarian at the 21d horizon |
| `WSB_MENTION_Z` | -0.0160 | -1.27 | 0.2196 | -0.1921 | 0.3078 | negative but not NW-significant |
| `WSB_SENTIMENT_Z` | -0.0655 | -0.73 | 0.4751 | -0.7860 | 0.2273 | not significant |

Investment read: the significant `WSB_ATTENTION_SHOCK_Z` coefficient is negative,
which supports a mean-reversion thesis: crowded retail attention appears to mark
temporary price inflation rather than persistent positive alpha. This is the connection
to an active-investment question: high retail-attention names may be candidates for
short or underweight review, especially when paired with borrow stress, days-to-cover,
or other crowded-short diagnostics.

### Corrected Long-Short Backtest

The walk-forward output now has an actual short leg. The earlier run was mislabeled:
`n_short` was zero because turnover-control normalization accidentally flipped short
weights positive. That code path has been fixed, and the current `backtest_results.csv`
has `n_short > 0` in **67/67** monthly rows.

Corrected six-name connected run:

| Metric | Value |
|---|---:|
| Composite factors | `IVOL`, `RVOL`, `ILLIQ` |
| Annualized gross return | 72.9% |
| Annualized net return | 72.5% |
| Net Sharpe | 0.79 |
| Max drawdown | -54.1% |
| Monthly hit rate | 56.7% |
| Average turnover | 25.5% |
| Total net return | 615.0% |

This is **not** a production strategy result. It is a tiny, event-heavy six-name
diagnostic run. The value is that the alt-data panels flow through the same IC,
Fama-MacBeth, quintile, and walk-forward machinery as the core factor library.

### Multi-Factor Composite Construction

The composite construction is deliberately simple and auditable:

1. Compute each candidate factor as a `date x ticker` panel.
2. Winsorize and cross-sectionally z-score each factor by date.
3. Rank all factors by absolute 21-day ICIR.
4. Select the top three factors for the walk-forward composite.
5. Combine selected factor scores with an equal-weight average by ticker/date.
6. Rank the composite cross-sectionally each rebalance date.
7. Long the highest-score bucket and short the lowest-score bucket, equal-weighted
   within each leg.
8. Apply turnover control by blending toward previous weights when one-way turnover
   exceeds the configured cap.

The current connected run selected `IVOL`, `RVOL`, and `ILLIQ`. The code supports
custom factor weights through `MultiFactorRanking(factor_weights=...)`, but the
committed pipeline uses equal weights to keep the composite transparent and avoid
overfitting a six-name sample.

### Point-In-Time Status

The factor code evaluates factor values dated `T` against forward close-to-close
returns from `T` to `T+h`. For daily price-derived factors and WSB factors, this is a
research alignment convention, not a fully executable point-in-time convention. A
production backtest should compute signals after the close on date `T`, trade no earlier
than the next session, and evaluate returns from `T+1` onward. The current connected
real-data results should therefore be interpreted as signal diagnostics, not live
execution P&L.

### Known Data Quality Gaps

- The real WSB universe is only six names and is biased toward the 2020-2022 meme-stock
episode.
- `GPOA` and `QUALITY_COMP` remain NaN in the connected six-name run because yfinance
does not provide enough gross-profit / total-assets coverage for that small universe.
The fundamentals loader now falls back to the latest non-empty cache when a current
yfinance fetch fails, but proper point-in-time fundamentals require Compustat/FactSet.
- Web/app/job-posting/credit-card style operational alt-data is not included in the
real run; only the schema and feature builders exist in the alt-data repo.

## Factor Library (16 factors)

| Group | Factor | Description | Reference |
|---|---|---|---|
| Momentum | MOM_12_1 | 12-month momentum, skip 1M | Jegadeesh & Titman 1993 |
| Momentum | MOM_6_1 | 6-month momentum, skip 1M | Jegadeesh & Titman 1993 |
| Momentum | MOM_3_1 | 3-month momentum, skip 1M | Jegadeesh & Titman 1993 |
| Momentum | STR | Short-term reversal (sign-flipped 1M return) | Jegadeesh 1990 |
| Momentum | MOM_VOL_ADJ | Volatility-adjusted 12-1 momentum | Barroso & Santa-Clara 2015 |
| Volatility | IVOL | Idiosyncratic volatility (21d, sign-flipped) | Ang et al. 2006 |
| Volatility | RVOL | Realized volatility (63d, sign-flipped) | |
| Volatility | BAB | Betting-Against-Beta (rolling beta, flipped) | Frazzini & Pedersen 2014 |
| Liquidity | ILLIQ | Amihud illiquidity ratio (21d log-scaled) | Amihud 2002 |
| Value | BTM | Book-to-market (1/P_B) | Fama & French 1992 |
| Value | EP | Earnings yield (E/P = 1/P_E) | Lakonishok et al. 1994 |
| Value | VALUE_COMP | Composite value (BTM + EP + SP) | |
| Quality | GPOA | Gross profit / total assets | Novy-Marx 2013 |
| Quality | ROE | Return on equity (TTM) | Fama & French 2015 |
| Quality | OPER_MARGIN | Operating margin | |
| Quality | QUALITY_COMP | Composite quality (GPOA + ROE + OpMargin) | |

## Architecture

```
src/
├── data/
│   └── fetch.py           # S&P 500 universe, bulk price download, fundamentals, returns
├── factors/
│   ├── base.py            # Factor ABC + cross-sectional preprocessing (winsorize, z-score, rank)
│   ├── momentum.py        # 5 momentum and reversal factors
│   ├── volatility.py      # IVOL (idiosyncratic), BAB (beta), RVOL, Amihud illiquidity
│   ├── value.py           # B/M, E/P, composite value
│   ├── quality.py         # Gross profitability, ROE, operating margin, composite quality
│   └── registry.py        # Central factor catalog + build_all_factors()
├── analytics/
│   ├── fama_macbeth.py    # Cross-sectional regressions with Newey-West HAC t-statistics
│   └── ic_analysis.py     # IC, ICIR, decay curves, cumulative IC regime analysis
├── portfolio/
│   └── ranking.py         # Multi-factor composite, quintile assignments, turnover control
└── backtest/
    └── walk_forward.py    # Monthly rebalancing, transaction costs, P&L attribution
```

## Methodology

### Fama-MacBeth Regressions

For each month *t*, run a cross-sectional OLS regression:

$$R_{i,t} = \alpha_t + \lambda_t \cdot f_{i,t-1} + \varepsilon_{i,t}$$

Collect the time series of slopes {λ_t}. Test H₀: E[λ_t] = 0 using **Newey-West t-statistics** with 6 lags — critical because factor returns exhibit serial correlation that OLS standard errors ignore. |t| > 2.0 = significant at 5%.

### IC Analysis

$$\text{IC}_t = \text{Spearman}\left(\text{rank}(f_{i,t-1}),\ \text{rank}(R_{i,t})\right)$$

$$\text{ICIR} = \frac{\overline{IC}}{\sigma_{IC}}, \quad t\text{-stat} = \text{ICIR} \times \sqrt{T}$$

Rank IC (Spearman) is preferred over Pearson: more robust to outliers, tests whether the **ordering** is correct rather than the magnitude. IC decay curves reveal the signal's half-life and optimal rebalancing frequency.

### Portfolio Construction

- Long top quintile (Q1), short bottom quintile (Q5), dollar-neutral
- Equal-weight within each leg
- Monthly rebalancing; turnover control caps one-way turnover at 30%/month
- Transaction costs: 5 bps one-way (conservative for liquid large-caps)

## Quickstart

```bash
git clone https://github.com/kartikmunjal/Cross-Sectional-Factor-Research.git
cd Cross-Sectional-Factor-Research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Step 1: download data (~5 min for full S&P 500)
python scripts/download_data.py --n-tickers 150  # use 150 tickers for quick test

# Step 2: run full pipeline (IC + FM + backtest)
python scripts/run_full_pipeline.py --n-tickers 150 --save

# Or open the notebook for interactive analysis
jupyter notebook notebooks/01_factor_research.ipynb
```

## Key Design Decisions

**Why Newey-West?** Factor returns are autocorrelated — momentum λ_t persists because the same stocks that beat last month often beat again. OLS standard errors assume i.i.d. errors; Newey-West corrects for serial correlation, producing conservative and reliable t-statistics.

**Why Rank IC over Pearson IC?** Return distributions have fat tails. A single stock with a +200% month dominates Pearson correlation. Rank IC measures whether the ordering is correct — the question that matters for portfolio construction.

**Why turnover control?** Without it, a pure ranking model turns over 40–60% of the portfolio monthly. At 5 bps one-way, that's ~40–60 bps/month (5%+/year) in costs — easily exceeding the gross alpha. Blending toward the previous portfolio reduces costs dramatically at modest alpha sacrifice.

**Why skip the most recent month in momentum?** The last month's return has a strong reversal component (microstructure-driven), which partially cancels the momentum signal. Standard momentum factors use months t-12 to t-2 to avoid this.

## References

- Fama, E.F. & MacBeth, J.D. (1973). *Risk, Return, and Equilibrium: Empirical Tests.* Journal of Political Economy.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Ang, A. et al. (2006). *The Cross-Section of Volatility and Expected Returns.* Journal of Finance.
- Novy-Marx, R. (2013). *The Other Side of Value: The Gross Profitability Premium.* Journal of Financial Economics.
- Frazzini, A. & Pedersen, L.H. (2014). *Betting Against Beta.* Journal of Financial Economics.
- Amihud, Y. (2002). *Illiquidity and Stock Returns.* Journal of Financial Markets.
- Newey, W.K. & West, K.D. (1987). *A Simple, Positive Semi-Definite HAC Covariance Matrix.* Econometrica.
