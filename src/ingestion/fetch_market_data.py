"""
src/ingestion/fetch_market_data.py
Downloads daily OHLCV data from yfinance for all configured assets
and saves raw CSVs to data/raw/.
Author: Fidel Mehra
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from src.utils.io import ensure_dir, load_config, save_dataframe


def fetch_asset(ticker: str, period_years: int = 5, interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data for a single asset from yfinance.

    Parameters
    ----------
    ticker       : Yahoo Finance ticker symbol, e.g. 'BTC-USD'
    period_years : How many years of history to download
    interval     : Bar interval, default daily '1d'

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume]
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * period_years)

    logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
    df = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_date),
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    logger.info(f"{ticker}: {len(df)} rows downloaded")
    return df


def ingest_all(config_path: str = "config/config.yaml") -> None:
    """
    Run full ingestion for all assets defined in config.
    Saves one CSV per asset to data/raw/<TICKER>.csv.
    """
    cfg = load_config(config_path)
    raw_dir = cfg["data"]["raw_dir"]
    period_years = cfg["data"]["period_years"]
    interval = cfg["data"]["interval"]
    assets = cfg["data"]["assets"]

    ensure_dir(raw_dir)

    summary = []
    for asset in assets:
        ticker = asset["ticker"]
        try:
            df = fetch_asset(ticker, period_years=period_years, interval=interval)
            out_path = Path(raw_dir) / f"{ticker.replace('=', '_')}.csv"
            save_dataframe(df, out_path, index=True)
            summary.append({"ticker": ticker, "rows": len(df), "status": "OK"})
        except Exception as exc:
            logger.error(f"Failed to fetch {ticker}: {exc}")
            summary.append({"ticker": ticker, "rows": 0, "status": str(exc)})

    # Print ingestion summary
    logger.info("Ingestion complete:")
    for row in summary:
        logger.info(f"  {row['ticker']:10s} | rows={row['rows']:5d} | {row['status']}")


if __name__ == "__main__":
    ingest_all()
