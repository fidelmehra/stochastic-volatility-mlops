"""
app/service.py
Prediction service layer: loads model artifacts and runs inference.
Author: Fidel Mehra
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from app.schemas import (
    ModelInfoResponse,
    RegimeRequest,
    RegimeResponse,
    VolatilityRequest,
    VolatilityResponse,
)
from src.utils.io import load_config, load_metadata, load_model

REGIME_LABELS = {0: "Low-Vol", 1: "Normal", 2: "Stress"}
CONFIDENCE_MULTIPLIER = 1.96  # approx 95% CI using bootstrapped std


class PredictionService:
    """
    Loads the best registered model and regime classifier on initialisation.
    Provides predict_volatility() and predict_regime() methods used by
    the FastAPI routes.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        cfg = load_config(config_path)
        api_cfg = cfg["api"]

        self.model_path = api_cfg["model_path"]
        self.metadata_path = api_cfg["metadata_path"]
        self.model_registry_dir = cfg["training"]["model_registry_dir"]

        # Load regression model
        self.vol_model = load_model(self.model_path)
        logger.info(f"Volatility model loaded from {self.model_path}")

        # Load metadata
        try:
            self.metadata = load_metadata(self.metadata_path)
        except FileNotFoundError:
            self.metadata = {}
            logger.warning("Model metadata not found; using empty dict.")

        self.model_version = self.metadata.get("model_name", "unknown")
        self.feature_names: list[str] = self.metadata.get("feature_names", [])

        # Load regime classifier (optional)
        ticker = self.metadata.get("ticker", "BTC-USD")
        cls_path = f"{self.model_registry_dir}/{ticker.replace('=','_')}_classifier.pkl"
        try:
            self.regime_model = load_model(cls_path)
            logger.info(f"Regime classifier loaded from {cls_path}")
        except FileNotFoundError:
            self.regime_model = None
            logger.warning("Regime classifier not found; /predict-regime will use heuristics.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _features_to_array(self, features: dict[str, float]) -> np.ndarray:
        """
        Align incoming feature dict to the model's expected feature order.
        Missing features default to 0; extra features are ignored.
        """
        if self.feature_names:
            arr = np.array(
                [features.get(f, 0.0) for f in self.feature_names], dtype=np.float64
            )
        else:
            arr = np.array(list(features.values()), dtype=np.float64)
        return arr.reshape(1, -1)

    # ------------------------------------------------------------------
    # Volatility prediction
    # ------------------------------------------------------------------

    def predict_volatility(self, request: VolatilityRequest) -> VolatilityResponse:
        """Run volatility regression model on incoming features."""
        X = self._features_to_array(request.features)
        pred = float(self.vol_model.predict(X)[0])
        pred = max(pred, 0.0)  # volatility cannot be negative

        # Simple symmetric CI: pred +/- CONFIDENCE_MULTIPLIER * rolling_std
        # In production this would use bootstrapped or quantile regression estimates
        residual_std = self.metadata.get("rmse", pred * 0.15)
        margin = CONFIDENCE_MULTIPLIER * residual_std
        ci = [round(max(pred - margin, 0.0), 6), round(pred + margin, 6)]

        return VolatilityResponse(
            ticker=request.ticker,
            predicted_rv=round(pred, 6),
            confidence_interval=ci,
            horizon_days=5,
        )

    # ------------------------------------------------------------------
    # Regime prediction
    # ------------------------------------------------------------------

    def predict_regime(self, request: RegimeRequest) -> RegimeResponse:
        """Run regime classifier on incoming features."""
        if self.regime_model is not None:
            X = self._features_to_array(request.features)
            y_pred = int(self.regime_model.predict(X)[0])
            y_prob = self.regime_model.predict_proba(X)[0]
            probs = {
                REGIME_LABELS.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(y_prob)
            }
        else:
            # Heuristic fallback using RV features
            rv = request.features.get("rv_5", 0.03)
            if rv < 0.015:
                y_pred, probs = 0, {"Low-Vol": 0.80, "Normal": 0.15, "Stress": 0.05}
            elif rv < 0.04:
                y_pred, probs = 1, {"Low-Vol": 0.15, "Normal": 0.70, "Stress": 0.15}
            else:
                y_pred, probs = 2, {"Low-Vol": 0.05, "Normal": 0.20, "Stress": 0.75}

        return RegimeResponse(
            ticker=request.ticker,
            predicted_regime=REGIME_LABELS.get(y_pred, str(y_pred)),
            regime_probabilities=probs,
        )

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def get_model_info(self) -> ModelInfoResponse:
        """Return metadata about the loaded model."""
        return ModelInfoResponse(
            model_name=self.metadata.get("model_name", "unknown"),
            ticker=self.metadata.get("ticker", "unknown"),
            trained_on=self.metadata.get("trained_on", "unknown"),
            rmse=self.metadata.get("rmse"),
            mae=self.metadata.get("mae"),
            qlike=self.metadata.get("qlike"),
            r2=self.metadata.get("r2"),
            feature_count=self.metadata.get("feature_count", 0),
        )
