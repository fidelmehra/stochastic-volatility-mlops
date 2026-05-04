"""Unit tests for feature engineering module."""
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV dataframe for testing."""
    np.random.seed(42)
    n = 252  # one trading year
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests for log-return computation
# ---------------------------------------------------------------------------

class TestLogReturns:
    def test_shape(self, sample_ohlcv):
        returns = np.log(sample_ohlcv["close"]).diff().dropna()
        assert len(returns) == len(sample_ohlcv) - 1

    def test_no_nan_after_dropna(self, sample_ohlcv):
        returns = np.log(sample_ohlcv["close"]).diff().dropna()
        assert returns.isna().sum() == 0

    def test_dtype_float(self, sample_ohlcv):
        returns = np.log(sample_ohlcv["close"]).diff().dropna()
        assert returns.dtype == float


# ---------------------------------------------------------------------------
# Tests for rolling volatility
# ---------------------------------------------------------------------------

class TestRollingVolatility:
    def test_window_10(self, sample_ohlcv):
        returns = np.log(sample_ohlcv["close"]).diff()
        vol = returns.rolling(10).std() * np.sqrt(252)
        # First 9 values should be NaN
        assert vol.iloc[:9].isna().all()
        assert not vol.iloc[10:].isna().all()

    def test_annualised_positive(self, sample_ohlcv):
        returns = np.log(sample_ohlcv["close"]).diff()
        vol = returns.rolling(20).std() * np.sqrt(252)
        assert (vol.dropna() > 0).all()

    def test_different_windows_differ(self, sample_ohlcv):
        returns = np.log(sample_ohlcv["close"]).diff()
        vol5 = returns.rolling(5).std().dropna()
        vol20 = returns.rolling(20).std().dropna()
        # Align and compare – they should generally differ
        common = vol5.index.intersection(vol20.index)
        assert not (vol5[common] == vol20[common]).all()


# ---------------------------------------------------------------------------
# Tests for Parkinson range-based volatility estimator
# ---------------------------------------------------------------------------

class TestParkinsonVolatility:
    """Parkinson formula: sigma = sqrt(1/(4*n*ln2) * sum(ln(H/L)^2))"""

    def _parkinson(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        log_hl = np.log(df["high"] / df["low"])
        return np.sqrt((log_hl ** 2).rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)

    def test_positive_values(self, sample_ohlcv):
        vol = self._parkinson(sample_ohlcv)
        assert (vol.dropna() > 0).all()

    def test_no_inf(self, sample_ohlcv):
        vol = self._parkinson(sample_ohlcv)
        assert not np.isinf(vol.dropna()).any()

    def test_window_respected(self, sample_ohlcv):
        vol = self._parkinson(sample_ohlcv, window=20)
        assert vol.iloc[:19].isna().all()


# ---------------------------------------------------------------------------
# Tests for RSI calculation
# ---------------------------------------------------------------------------

class TestRSI:
    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def test_bounds(self, sample_ohlcv):
        rsi = self._rsi(sample_ohlcv["close"]).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_nan_count(self, sample_ohlcv):
        rsi = self._rsi(sample_ohlcv["close"], period=14)
        # First 14 values should be NaN (period) + 1 for diff
        assert rsi.iloc[:14].isna().all()


# ---------------------------------------------------------------------------
# Tests for ATR
# ---------------------------------------------------------------------------

class TestATR:
    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"] - df["close"].shift()).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    def _atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        tr = self._true_range(df)
        return tr.rolling(window).mean()

    def test_atr_positive(self, sample_ohlcv):
        atr = self._atr(sample_ohlcv).dropna()
        assert (atr > 0).all()

    def test_atr_shape(self, sample_ohlcv):
        atr = self._atr(sample_ohlcv)
        assert len(atr) == len(sample_ohlcv)
