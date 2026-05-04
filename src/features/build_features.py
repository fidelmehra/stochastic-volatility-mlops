"""
src/features/build_features.py
Feature engineering pipeline for stochastic volatility forecasting.
Builds lag, rolling RV, EWMA, jump, volume, drawdown, and calendar features
from raw OHLCV data.
Author: Fidel Mehra
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.io import load_config, load_dataframe, save_dataframe


# ---------------------------------------------------------------------------
# Core feature builders
# ---------------------------------------------------------------------------

def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from Close prices."""
    return np.log(df["Close"] / df["Close"].shift(1)).rename("log_return")


def compute_realised_vol(returns: pd.Series, windows: list[int]) -> pd.DataFrame:
    """
    Compute rolling realised volatility (annualised) over multiple windows.
    RV_w = sqrt(sum of squared returns over window w)
    """
    frames = {}
    for w in windows:
        frames[f"rv_{w}"] = returns.rolling(w).apply(
            lambda x: np.sqrt(np.sum(x ** 2)), raw=True
        )
    return pd.DataFrame(frames)


def compute_ewma_variance(returns: pd.Series, spans: list[int]) -> pd.DataFrame:
    """Exponentially weighted moving variance (RiskMetrics style)."""
    frames = {}
    for s in spans:
        frames[f"ewma_var_{s}"] = returns.ewm(span=s, adjust=False).var()
    return pd.DataFrame(frames)


def compute_lag_features(returns: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Lagged return features to capture autocorrelation structure."""
    frames = {}
    for lag in lags:
        frames[f"ret_lag_{lag}"] = returns.shift(lag)
    return pd.DataFrame(frames)


def compute_rolling_stats(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """Rolling skewness and excess kurtosis over a fixed window."""
    return pd.DataFrame({
        f"roll_skew_{window}": returns.rolling(window).skew(),
        f"roll_kurt_{window}": returns.rolling(window).kurt(),
        f"roll_mean_{window}": returns.rolling(window).mean(),
        f"roll_std_{window}": returns.rolling(window).std(),
    })


def compute_jump_features(returns: pd.Series, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Jump proxy features.
    A jump is flagged when |return| exceeds z_threshold standard deviations
    from the rolling 20-day mean.
    """
    roll_mean = returns.rolling(20).mean()
    roll_std = returns.rolling(20).std()
    z_score = (returns - roll_mean) / (roll_std + 1e-8)
    abs_ret_z = z_score.abs().rename("abs_ret_z")
    jump_flag = (abs_ret_z > z_threshold).astype(int).rename("jump_flag")
    return pd.concat([abs_ret_z, jump_flag], axis=1)


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based market activity features."""
    log_vol = np.log1p(df["Volume"]).rename("log_volume")
    vol_change = df["Volume"].pct_change().rename("vol_change")
    return pd.concat([log_vol, vol_change], axis=1)


def compute_drawdown(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling maximum drawdown over a lookback window."""
    rolling_max = df["Close"].rolling(window).max()
    drawdown = (df["Close"] - rolling_max) / (rolling_max + 1e-8)
    return drawdown.rename(f"max_drawdown_{window}")


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode calendar seasonality features."""
    idx = df.index
    return pd.DataFrame({
        "day_of_week": idx.dayofweek,
        "month": idx.month,
        "is_month_end": idx.is_month_end.astype(int),
        "is_quarter_end": idx.is_quarter_end.astype(int),
    }, index=df.index)


def compute_target(returns: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Compute the forward-looking realised volatility target.
    RV_target_t = sqrt( sum_{k=1}^{horizon} r_{t+k}^2 )
    """
    def forward_rv(x):
        return np.sqrt(np.sum(x ** 2))

    target = (
        returns
        .shift(-horizon)  # align future returns
        .rolling(horizon)
        .apply(forward_rv, raw=True)
        .rename(f"rv_target_{horizon}d")
    )
    # Shift back so target aligns with current row
    return target.shift(horizon - horizon)  # already aligned


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(
    ticker: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for a single asset.

    Loads raw OHLCV, computes all features, attaches target variable,
    drops NaN rows, and saves to data/processed/<TICKER>_features.csv.

    Returns the feature DataFrame.
    """
    cfg = load_config(config_path)
    feat_cfg = cfg["features"]
    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]

    raw_path = f"{raw_dir}/{ticker.replace('=', '_')}.csv"
    df = load_dataframe(raw_path)

    logger.info(f"Building features for {ticker} | raw rows={len(df)}")

    returns = compute_log_returns(df)

    feature_parts = [
        returns.rename("log_return"),
        compute_realised_vol(returns, feat_cfg["rv_windows"]),
        compute_ewma_variance(returns, feat_cfg["ewma_spans"]),
        compute_lag_features(returns, feat_cfg["lag_windows"]),
        compute_rolling_stats(returns, feat_cfg["rolling_stats_window"]),
        compute_jump_features(returns, feat_cfg["jump_z_threshold"]),
        compute_volume_features(df),
        compute_drawdown(df),
        compute_calendar_features(df),
    ]

    features = pd.concat(feature_parts, axis=1)

    # Attach target: forward RV
    horizon = feat_cfg["target_horizon"]
    target = compute_target(returns, horizon=horizon)
    features[f"rv_target_{horizon}d"] = target

    # Drop rows with NaN (burn-in period)
    n_before = len(features)
    features = features.dropna()
    logger.info(f"{ticker}: dropped {n_before - len(features)} NaN rows | final={len(features)}")

    # Save processed features
    out_path = f"{processed_dir}/{ticker.replace('=', '_')}_features.csv"
    save_dataframe(features, out_path)

    return features


def build_all_features(config_path: str = "config/config.yaml") -> None:
    """Build features for all assets defined in config."""
    cfg = load_config(config_path)
    for asset in cfg["data"]["assets"]:
        ticker = asset["ticker"]
        try:
            build_features(ticker, config_path=config_path)
        except Exception as exc:
            logger.error(f"Feature engineering failed for {ticker}: {exc}")


if __name__ == "__main__":
    build_all_features()
