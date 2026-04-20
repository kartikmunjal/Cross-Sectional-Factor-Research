"""
Factor registry — single source of truth for all available factors.

Usage:
    from src.factors.registry import FACTOR_REGISTRY, build_all_factors

    factors = build_all_factors(close, volume, market_returns, fundamentals)
    # Returns dict: factor_name → (date × ticker) DataFrame
"""

from __future__ import annotations

import logging

import pandas as pd

from src.factors.momentum  import Momentum12_1, Momentum6_1, Momentum3_1, ShortTermReversal, MomentumVolatilityRatio
from src.factors.volatility import IdiosyncraticVol, Beta, RealizedVol, AmihudIlliquidity
from src.factors.value      import BookToMarket, EarningsYield, CompositeValue
from src.factors.quality    import GrossProfitability, ReturnOnEquity, OperatingProfitability, QualityComposite
from src.factors.altdata import ALT_FACTOR_META

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Registry: all factors used in the research
# ------------------------------------------------------------------

FACTOR_REGISTRY = {
    # --- Momentum ---
    "MOM_12_1":     Momentum12_1(),
    "MOM_6_1":      Momentum6_1(),
    "MOM_3_1":      Momentum3_1(),
    "STR":          ShortTermReversal(),
    "MOM_VOL_ADJ":  MomentumVolatilityRatio(),

    # --- Volatility / Risk ---
    "IVOL":         IdiosyncraticVol(window=21),
    "RVOL":         RealizedVol(window=63),
    "BAB":          Beta(),
    "ILLIQ":        AmihudIlliquidity(window=21),

    # --- Value ---
    "BTM":          BookToMarket(),
    "EP":           EarningsYield(),
    "VALUE_COMP":   CompositeValue(),

    # --- Quality ---
    "GPOA":         GrossProfitability(),
    "ROE":          ReturnOnEquity(),
    "OPER_MARGIN":  OperatingProfitability(),
    "QUALITY_COMP": QualityComposite(),
}

# Factor metadata for display and grouping
FACTOR_META = {
    "MOM_12_1":    {"group": "Momentum",   "horizon": "12M", "paper": "Jegadeesh & Titman 1993"},
    "MOM_6_1":     {"group": "Momentum",   "horizon": "6M",  "paper": "Jegadeesh & Titman 1993"},
    "MOM_3_1":     {"group": "Momentum",   "horizon": "3M",  "paper": "Jegadeesh & Titman 1993"},
    "STR":         {"group": "Momentum",   "horizon": "1M",  "paper": "Jegadeesh 1990"},
    "MOM_VOL_ADJ": {"group": "Momentum",   "horizon": "12M", "paper": "Barroso & Santa-Clara 2015"},
    "IVOL":        {"group": "Volatility", "horizon": "21d", "paper": "Ang et al. 2006"},
    "RVOL":        {"group": "Volatility", "horizon": "63d", "paper": ""},
    "BAB":         {"group": "Volatility", "horizon": "252d","paper": "Frazzini & Pedersen 2014"},
    "ILLIQ":       {"group": "Liquidity",  "horizon": "21d", "paper": "Amihud 2002"},
    "BTM":         {"group": "Value",      "horizon": "TTM", "paper": "Fama & French 1992"},
    "EP":          {"group": "Value",      "horizon": "TTM", "paper": "Lakonishok et al. 1994"},
    "VALUE_COMP":  {"group": "Value",      "horizon": "TTM", "paper": "Composite"},
    "GPOA":        {"group": "Quality",    "horizon": "TTM", "paper": "Novy-Marx 2013"},
    "ROE":         {"group": "Quality",    "horizon": "TTM", "paper": "Fama & French 2015"},
    "OPER_MARGIN": {"group": "Quality",    "horizon": "TTM", "paper": ""},
    "QUALITY_COMP":{"group": "Quality",    "horizon": "TTM", "paper": "Composite"},
    **ALT_FACTOR_META,
}


def build_all_factors(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    market_returns: pd.Series | None = None,
    fundamentals: pd.DataFrame | None = None,
    factor_names: list[str] | None = None,
    preprocess: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Compute and optionally preprocess all factors in the registry.

    Parameters
    ----------
    close          : (date × ticker) close price panel
    volume         : (date × ticker) share volume panel
    market_returns : (date,) SPY daily returns
    fundamentals   : (ticker × field) snapshot
    factor_names   : subset of FACTOR_REGISTRY keys to compute
    preprocess     : if True, winsorize + z-score each factor

    Returns
    -------
    dict: factor_name → (date × ticker) DataFrame
    """
    names = factor_names or list(FACTOR_REGISTRY.keys())
    results = {}

    for name in names:
        if name not in FACTOR_REGISTRY:
            log.warning(f"Factor '{name}' not in registry, skipping")
            continue

        factor = FACTOR_REGISTRY[name]
        log.info(f"Computing {name}...")

        try:
            raw = factor.compute(
                close=close,
                volume=volume,
                market_returns=market_returns,
                fundamentals=fundamentals,
            )
            if preprocess:
                raw = factor.preprocess(raw)
            results[name] = raw

        except Exception as e:
            log.error(f"Factor {name} failed: {e}")
            continue

    log.info(f"Built {len(results)}/{len(names)} factors successfully")
    return results


def factor_summary_table() -> pd.DataFrame:
    """Return a formatted table of all factors with metadata."""
    rows = []
    for name, meta in FACTOR_META.items():
        if name in ALT_FACTOR_META:
            rows.append({
                "Factor": name,
                "Group": meta["group"],
                "Horizon": meta["horizon"],
                "Academic Reference": meta["paper"],
                "Description": meta["description"],
            })
            continue
        rows.append({
            "Factor": name,
            "Group": meta["group"],
            "Horizon": meta["horizon"],
            "Academic Reference": meta["paper"],
            "Description": FACTOR_REGISTRY[name].description,
        })
    return pd.DataFrame(rows)
