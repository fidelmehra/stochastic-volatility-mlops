"""
app/schemas.py
Pydantic request and response models for the FastAPI inference service.
Author: Fidel Mehra
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & model info
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    ticker: str
    trained_on: str
    rmse: Optional[float] = None
    mae: Optional[float] = None
    qlike: Optional[float] = None
    r2: Optional[float] = None
    feature_count: int


# ---------------------------------------------------------------------------
# Volatility prediction
# ---------------------------------------------------------------------------

class VolatilityRequest(BaseModel):
    ticker: str = Field(..., example="BTC-USD")
    features: Dict[str, float] = Field(
        ...,
        example={
            "rv_5": 0.042,
            "rv_10": 0.038,
            "rv_20": 0.035,
            "ret_lag_1": -0.012,
            "ret_lag_2": 0.008,
            "ewma_var_10": 0.0017,
            "jump_flag": 0,
            "hmm_state_0": 0.1,
            "hmm_state_1": 0.7,
            "hmm_state_2": 0.2,
        },
    )


class VolatilityResponse(BaseModel):
    ticker: str
    predicted_rv: float
    confidence_interval: List[float]  # [lower, upper]
    horizon_days: int = 5


# ---------------------------------------------------------------------------
# Regime prediction
# ---------------------------------------------------------------------------

class RegimeRequest(BaseModel):
    ticker: str = Field(..., example="BTC-USD")
    features: Dict[str, float] = Field(
        ...,
        example={"rv_5": 0.042, "ret_lag_1": -0.012},
    )


class RegimeResponse(BaseModel):
    ticker: str
    predicted_regime: str
    regime_probabilities: Dict[str, float]  # {"Low-Vol": 0.1, "Normal": 0.2, "Stress": 0.7}


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

class BatchVolatilityRequest(BaseModel):
    requests: List[VolatilityRequest]


class BatchVolatilityResponse(BaseModel):
    results: List[VolatilityResponse]
