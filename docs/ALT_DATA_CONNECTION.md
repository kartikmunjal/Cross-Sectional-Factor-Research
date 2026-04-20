# Alt-Data Equity Signals Connection

This repo can now consume exported factor panels from
[`alt-data-equity-signals`](https://github.com/kartikmunjal/alt-data-equity-signals).

## Data Contract

The alt-data repo writes parquet files under `results/<run>/factor_panels/`:

```text
WSB_MENTION_Z.parquet
WSB_SENTIMENT_Z.parquet
WSB_ATTENTION_SHOCK_Z.parquet
WSB_WEB_TRAFFIC_LEVEL_Z.parquet
WSB_WEB_TRAFFIC_GROWTH_Z.parquet
WSB_WEB_TRAFFIC_SHOCK_Z.parquet
```

Each file is a point-in-time factor panel:

```text
index = date
columns = ticker
values = cross-sectional signal score
```

That is the same `date x ticker` interface used by native factors such as
`MOM_12_1`, `IVOL`, `BTM`, and `QUALITY_COMP`.

## Workflow

First build alt-data panels:

```bash
cd ../alt-data-equity-signals
python scripts/run_pipeline.py \
  --posts data/raw/wsb_posts.csv \
  --prices data/raw/close_panel.parquet \
  --out results/wsb_retail_attention
```

Then include those panels in this repo's factor pipeline:

```bash
cd ../Cross-Sectional-Factor-Research
python scripts/run_full_pipeline.py \
  --n-tickers 150 \
  --alt-factor-dir ../alt-data-equity-signals/results/wsb_retail_attention/factor_panels \
  --save
```

## Why This Connection Matters

`alt-data-equity-signals` owns messy-data ingestion and signal construction.
This repo owns broader cross-sectional comparison against traditional factors.

The combined workflow answers the actual research question:

> Do retail attention and operational web-traffic signals add incremental
> stock-level predictive power relative to price, volatility, value, quality,
> and liquidity factors?

## Implementation Notes

- `src/factors/altdata.py` loads exported parquet panels and protects against
  factor-name collisions.
- `scripts/run_full_pipeline.py --alt-factor-dir ...` merges alt-data panels
  after native factor construction.
- Alt-data factors are listed in `factor_summary_table()` metadata, but they are
  not computed by this repo. The source of truth remains `alt-data-equity-signals`.
