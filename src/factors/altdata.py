"""Adapters for externally generated alternative-data factor panels.

The alt-data source repo (`alt-data-equity-signals`) exports parquet files with
the same shape used throughout this project: index=date, columns=ticker,
values=point-in-time factor score. This module keeps that connection explicit
without making the factor research repo depend on the alt-data package.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ALT_FACTOR_META = {
    "WSB_MENTION_Z": {
        "group": "Alternative Data",
        "horizon": "Daily",
        "paper": "Retail attention / social media sentiment",
        "description": "Cross-sectional z-score of log WallStreetBets ticker mentions.",
    },
    "WSB_SENTIMENT_Z": {
        "group": "Alternative Data",
        "horizon": "Daily",
        "paper": "Retail attention / social media sentiment",
        "description": "Cross-sectional z-score of bullish-minus-bearish WSB text sentiment.",
    },
    "WSB_ATTENTION_SHOCK_Z": {
        "group": "Alternative Data",
        "horizon": "Daily",
        "paper": "Retail attention / social media sentiment",
        "description": "Cross-sectional z-score of abnormal WSB ticker attention vs trailing baseline.",
    },
    "WSB_WEB_TRAFFIC_LEVEL_Z": {
        "group": "Alternative Data",
        "horizon": "Monthly",
        "paper": "Operational web traffic / consumer demand",
        "description": "Cross-sectional z-score of log monthly web visits.",
    },
    "WSB_WEB_TRAFFIC_GROWTH_Z": {
        "group": "Alternative Data",
        "horizon": "Monthly",
        "paper": "Operational web traffic / consumer demand",
        "description": "Cross-sectional z-score of monthly web traffic growth.",
    },
    "WSB_WEB_TRAFFIC_SHOCK_Z": {
        "group": "Alternative Data",
        "horizon": "Monthly",
        "paper": "Operational web traffic / consumer demand",
        "description": "Cross-sectional z-score of abnormal web traffic versus trailing baseline.",
    },
}


def load_alt_factor_panels(
    factor_dir: str | Path,
    *,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    pattern: str = "*.parquet",
) -> dict[str, pd.DataFrame]:
    """Load exported alt-data factors from a directory of parquet panels."""
    factor_dir = Path(factor_dir)
    if not factor_dir.exists():
        raise FileNotFoundError(factor_dir)

    panels: dict[str, pd.DataFrame] = {}
    for path in sorted(factor_dir.glob(pattern)):
        panel = pd.read_parquet(path)
        panel.index = pd.to_datetime(panel.index).tz_localize(None)
        panel.columns = [str(col).upper() for col in panel.columns]
        panel = panel.sort_index()

        if start is not None:
            panel = panel.loc[panel.index >= pd.Timestamp(start)]
        if end is not None:
            panel = panel.loc[panel.index <= pd.Timestamp(end)]
        if tickers is not None:
            panel = panel.reindex(columns=[ticker.upper() for ticker in tickers])

        if not panel.dropna(how="all").empty:
            panels[path.stem.upper()] = panel

    return panels


def merge_factor_panels(
    base_factors: dict[str, pd.DataFrame],
    alt_factors: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Combine native and alt-data factors with collision protection."""
    overlap = set(base_factors).intersection(alt_factors)
    if overlap:
        raise ValueError(f"duplicate factor names: {sorted(overlap)}")
    return {**base_factors, **alt_factors}
