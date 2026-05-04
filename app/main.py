"""
app/main.py
FastAPI application for stochastic volatility forecasting and regime detection.
Exposes /health, /model-info, /predict-volatility, /predict-regime endpoints.
Author: Fidel Mehra
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.schemas import (
    BatchVolatilityRequest,
    BatchVolatilityResponse,
    HealthResponse,
    ModelInfoResponse,
    RegimeRequest,
    RegimeResponse,
    VolatilityRequest,
    VolatilityResponse,
)
from app.service import PredictionService


# ---------------------------------------------------------------------------
# Application lifespan: load model once at startup
# ---------------------------------------------------------------------------

service: PredictionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prediction service on startup, release on shutdown."""
    global service
    logger.info("Loading prediction service...")
    service = PredictionService()
    logger.info("Service ready.")
    yield
    logger.info("Shutting down service.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stochastic Volatility MLOps API",
    description=(
        "End-to-end inference service for volatility forecasting and "
        "latent risk regime detection. Author: Fidel Mehra."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns service health status and model version."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return HealthResponse(status="ok", model_version=service.model_version)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Returns metadata about the currently loaded model."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    return service.get_model_info()


@app.post("/predict-volatility", response_model=VolatilityResponse, tags=["Predictions"])
async def predict_volatility(request: VolatilityRequest):
    """
    Predict next-period realised volatility for the given asset and features.

    Returns predicted RV and a 95% confidence interval.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    try:
        result = service.predict_volatility(request)
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/predict-regime", response_model=RegimeResponse, tags=["Predictions"])
async def predict_regime(request: RegimeRequest):
    """
    Detect the current latent market regime (Low-Vol / Normal / Stress).

    Returns the predicted regime label and class probabilities.
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    try:
        result = service.predict_regime(request)
    except Exception as exc:
        logger.error(f"Regime prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    return result


@app.post("/predict-volatility/batch", response_model=BatchVolatilityResponse, tags=["Predictions"])
async def predict_volatility_batch(request: BatchVolatilityRequest):
    """Batch endpoint for multiple volatility predictions."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialised")
    results = [service.predict_volatility(r) for r in request.requests]
    return BatchVolatilityResponse(results=results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
