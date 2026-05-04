"""Pytest configuration and shared fixtures for the test suite.

Author: Fidel Mehra
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """Session-scoped OHLCV price dataframe used across multiple test modules."""
    np.random.seed(2024)
    n = 504  # ~2 trading years
    dates = pd.date_range("2022-01-03", periods=n, freq="B")

    # GBM price simulation
    mu, sigma = 0.0003, 0.015
    log_returns = np.random.normal(mu, sigma, n)
    close = 150.0 * np.exp(np.cumsum(log_returns))
    high = close * np.exp(np.abs(np.random.normal(0, 0.004, n)))
    low = close * np.exp(-np.abs(np.random.normal(0, 0.004, n)))
    open_ = np.roll(close, 1)
    open_[0] = 150.0
    volume = np.random.randint(500_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture(scope="session")
def log_returns(synthetic_prices) -> pd.Series:
    """Log returns computed from synthetic prices."""
    return np.log(synthetic_prices["close"]).diff().dropna()


@pytest.fixture(scope="session")
def volatility_features(log_returns) -> pd.DataFrame:
    """Volatility feature matrix derived from log returns."""
    df = log_returns.to_frame(name="log_return")
    for w in [5, 10, 20, 60]:
        df[f"vol_{w}"] = df["log_return"].rolling(w).std() * np.sqrt(252)
    df["abs_ret"] = df["log_return"].abs()
    df["ret_sq"] = df["log_return"] ** 2
    return df.dropna()
