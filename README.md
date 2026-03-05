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
