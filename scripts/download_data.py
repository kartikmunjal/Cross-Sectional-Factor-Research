#!/usr/bin/env python3
"""
Step 1: Download and cache all data.

Run this once before any analysis. Downloads ~5 years of daily prices
for the full S&P 500 universe and saves fundamentals snapshot.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --start 2018-01-01 --n-tickers 200
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch import (
    get_sp500_universe, download_prices,
    load_market_returns, download_fundamentals,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--n-tickers", type=int, default=None,
                        help="Limit to first N tickers (for testing)")
    parser.add_argument("--no-fundamentals", action="store_true")
    args = parser.parse_args()

    tickers = get_sp500_universe()
    if args.n_tickers:
        tickers = tickers[:args.n_tickers]
    log.info(f"Universe: {len(tickers)} tickers")

    log.info("Downloading price data...")
    price_dict = download_prices(tickers, start=args.start)
    log.info(f"Downloaded prices for {len(price_dict)} tickers")

    log.info("Downloading market (SPY) returns...")
    mkt = load_market_returns(start=args.start)
    log.info(f"Market returns: {len(mkt)} days")

    if not args.no_fundamentals:
        log.info("Downloading fundamentals (this may take a few minutes)...")
        fundamentals = download_fundamentals(tickers)
        log.info(f"Fundamentals: {fundamentals.notna().mean().mean():.0%} coverage")

    log.info("All data downloaded and cached. Ready for factor computation.")


if __name__ == "__main__":
    main()
