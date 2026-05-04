"""
src/evaluation/metrics.py
Evaluation metrics for volatility forecasting and regime classification.
Includes RMSE, MAE, QLIKE, R2 for regression and F1/ROC-AUC for classification.
Author: Fidel Mehra
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R-squared."""
    return float(r2_score(y_true, y_pred))


def qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    QLIKE volatility loss function.
    Used in volatility forecast evaluation because it penalises
    underestimation of variance more than MSE.

    QLIKE = mean[ log(sigma_hat^2) + RV / sigma_hat^2 ]

    where sigma_hat = predicted volatility and RV = realised volatility.
    """
    sigma_sq = np.maximum(y_pred ** 2, eps)
    rv = np.maximum(y_true ** 2, eps)
    return float(np.mean(np.log(sigma_sq) + rv / sigma_sq))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute full regression evaluation report.

    Returns
    -------
    dict with keys: rmse, mae, qlike, r2
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "qlike": qlike(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Classification metrics (regime detection)
# ---------------------------------------------------------------------------

def regime_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_classes: int = 3,
) -> dict:
    """
    Classification report for regime detection.

    Parameters
    ----------
    y_true   : True regime labels (integer)
    y_pred   : Predicted regime labels (integer)
    y_prob   : Predicted probabilities (n_samples, n_classes), optional
    n_classes: Number of regime states

    Returns
    -------
    dict with f1_weighted and optionally roc_auc_macro
    """
    report = {
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_prob is not None:
        try:
            report["roc_auc_macro"] = float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            report["roc_auc_macro"] = float("nan")

    return report


# ---------------------------------------------------------------------------
# Utility: pretty-print evaluation results
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, model_name: str = "") -> None:
    """Print a formatted metrics table to stdout."""
    header = f"  {'Metric':<20} {'Value':>12}"
    separator = "  " + "-" * 34
    title = f"\n  === {model_name} Results ==="
    print(title)
    print(separator)
    print(header)
    print(separator)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:>12.6f}")
        else:
            print(f"  {k:<20} {str(v):>12}")
    print(separator)
