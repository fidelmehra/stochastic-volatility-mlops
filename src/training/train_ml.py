"""
src/training/train_ml.py
ML model training: XGBoost regressor, LightGBM regressor, LSTM regressor,
XGBoost classifier for regimes, and HMM+XGBoost hybrid (flagship model).
Author: Fidel Mehra
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from src.evaluation.metrics import print_metrics, regression_report, regime_classification_report
from src.utils.io import load_config, load_dataframe, save_metadata, save_model


# ---------------------------------------------------------------------------
# Feature/target split helpers
# ---------------------------------------------------------------------------

HMM_COLS = ["hmm_state_0", "hmm_state_1", "hmm_state_2"]
EXCLUDE_COLS = ["hmm_hard_label", "hmm_regime"]


def prepare_regression_data(
    df: pd.DataFrame,
    target_col: str,
    include_hmm: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare X, y arrays for volatility regression."""
    drop_cols = [target_col] + EXCLUDE_COLS
    if not include_hmm:
        drop_cols += HMM_COLS
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df[target_col].values
    feature_names = X.columns.tolist()
    return X.values, y, feature_names


def prepare_classification_data(
    df: pd.DataFrame,
    label_col: str = "hmm_hard_label",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Prepare X, y arrays for regime classification."""
    drop_cols = [label_col, "hmm_regime"] + [c for c in HMM_COLS if c in df.columns]
    target_names = [c for c in ["rv_target_5d"] if c in df.columns]
    drop_cols += target_names

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=[np.number])
    y = df[label_col].values.astype(int)
    return X.values, y, X.columns.tolist()


# ---------------------------------------------------------------------------
# XGBoost Regressor
# ---------------------------------------------------------------------------

def train_xgboost_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    model_name: str = "xgboost_regressor",
) -> XGBRegressor:
    """Train XGBoost regressor with early stopping on validation set."""
    model = XGBRegressor(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = model.predict(X_val)
    metrics = regression_report(y_val, preds)
    print_metrics(metrics, model_name=model_name)
    return model


# ---------------------------------------------------------------------------
# LightGBM Regressor
# ---------------------------------------------------------------------------

def train_lightgbm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    model_name: str = "lightgbm_regressor",
) -> lgb.LGBMRegressor:
    """Train LightGBM regressor."""
    model = lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 63),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    preds = model.predict(X_val)
    metrics = regression_report(y_val, preds)
    print_metrics(metrics, model_name=model_name)
    return model


# ---------------------------------------------------------------------------
# LSTM Regressor (PyTorch)
# ---------------------------------------------------------------------------

class VolatilityLSTM(nn.Module):
    """LSTM model for volatility sequence forecasting."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        out = out[:, -1, :]   # take last timestep
        return self.fc(out).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 30):
    """Convert flat feature matrix to overlapping sequences for LSTM."""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    model_name: str = "lstm_regressor",
) -> VolatilityLSTM:
    """Train LSTM regressor on time-series sequences."""
    seq_len = params.get("sequence_length", 30)
    hidden_size = params.get("hidden_size", 64)
    num_layers = params.get("num_layers", 2)
    dropout = params.get("dropout", 0.2)
    lr = params.get("learning_rate", 0.001)
    epochs = params.get("epochs", 100)
    batch_size = params.get("batch_size", 32)
    patience = params.get("patience", 15)

    # Scale features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - X_mean) / X_std
    X_val_s = (X_val - X_mean) / X_std

    X_tr_seq, y_tr_seq = make_sequences(X_train_s, y_train, seq_len)
    X_v_seq, y_v_seq = make_sequences(X_val_s, y_val, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VolatilityLSTM(X_train.shape[1], hidden_size, num_layers, dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr_seq).to(device)
    y_tr_t = torch.tensor(y_tr_seq).to(device)
    X_v_t = torch.tensor(X_v_seq).to(device)
    y_v_t = torch.tensor(y_v_seq).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v_t)
            val_loss = criterion(val_pred, y_v_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"LSTM early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_preds = model(X_v_t).cpu().numpy()
    metrics = regression_report(y_v_seq, val_preds)
    print_metrics(metrics, model_name=model_name)
    return model


# ---------------------------------------------------------------------------
# XGBoost Classifier (regime detection)
# ---------------------------------------------------------------------------

def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    model_name: str = "xgboost_classifier",
) -> XGBClassifier:
    """Train XGBoost classifier for regime detection."""
    n_classes = len(np.unique(y_train))
    model = XGBClassifier(
        n_estimators=params.get("n_estimators", 500),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
        eval_metric="mlogloss",
        num_class=n_classes,
        objective="multi:softprob",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    metrics = regime_classification_report(y_val, y_pred, y_prob)
    print_metrics(metrics, model_name=model_name)
    return model


# ---------------------------------------------------------------------------
# HMM + XGBoost Hybrid (flagship model)
# ---------------------------------------------------------------------------

def train_hmm_xgboost_hybrid(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    target_col: str,
    params: dict,
    model_name: str = "hmm_xgboost_hybrid",
) -> XGBRegressor:
    """
    Flagship hybrid model: HMM state posteriors are included as features
    alongside all other engineered features in an XGBoost regressor.

    This explicitly combines stochastic latent-state modelling with
    discriminative ML in a single pipeline.
    """
    X_train, y_train, feature_names = prepare_regression_data(
        df_train, target_col, include_hmm=True
    )
    X_val, y_val, _ = prepare_regression_data(df_val, target_col, include_hmm=True)

    logger.info(
        f"Training {model_name} | "
        f"train={X_train.shape}, val={X_val.shape}, features={len(feature_names)}"
    )
    model = train_xgboost_regressor(X_train, y_train, X_val, y_val, params, model_name)
    return model


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_all_models(ticker: str, config_path: str = "config/config.yaml") -> None:
    """
    Full model training pipeline for a single asset.
    Trains XGBoost, LightGBM, LSTM, XGBoost classifier, and HMM+XGBoost hybrid.
    Saves all models and metadata to the registry.
    """
    cfg = load_config(config_path)
    processed_dir = cfg["data"]["processed_dir"]
    model_dir = cfg["training"]["model_registry_dir"]
    xgb_params = cfg["training"]["xgboost"]
    lgb_params = cfg["training"]["lightgbm"]
    lstm_params = cfg["training"]["lstm"]
    test_size = cfg["training"]["test_size"]
    target_col = f"rv_target_{cfg['features']['target_horizon']}d"

    feat_path = f"{processed_dir}/{ticker.replace('=', '_')}_features.csv"
    df = load_dataframe(feat_path)

    # Time-ordered train/val split
    split_idx = int(len(df) * (1 - test_size))
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    X_train, y_train, feat_names = prepare_regression_data(df_train, target_col, include_hmm=False)
    X_val, y_val, _ = prepare_regression_data(df_val, target_col, include_hmm=False)

    # 1. XGBoost regressor (no HMM features)
    logger.info("Training XGBoost regressor...")
    xgb_model = train_xgboost_regressor(X_train, y_train, X_val, y_val, xgb_params, "xgboost")
    save_model(xgb_model, f"{model_dir}/{ticker.replace('=','_')}_xgboost.pkl")

    # 2. LightGBM regressor
    logger.info("Training LightGBM regressor...")
    lgb_model = train_lightgbm_regressor(X_train, y_train, X_val, y_val, lgb_params, "lightgbm")
    save_model(lgb_model, f"{model_dir}/{ticker.replace('=','_')}_lightgbm.pkl")

    # 3. LSTM regressor
    logger.info("Training LSTM regressor...")
    lstm_model = train_lstm(X_train, y_train, X_val, y_val, lstm_params, "lstm")
    save_model(lstm_model, f"{model_dir}/{ticker.replace('=','_')}_lstm.pkl")

    # 4. HMM + XGBoost hybrid (flagship)
    if all(c in df.columns for c in HMM_COLS):
        logger.info("Training HMM+XGBoost hybrid...")
        hybrid_model = train_hmm_xgboost_hybrid(df_train, df_val, target_col, xgb_params)
        save_model(hybrid_model, f"{model_dir}/{ticker.replace('=','_')}_hybrid.pkl")

        # Register hybrid as best model
        X_val_hmm, y_val_hmm, _ = prepare_regression_data(df_val, target_col, include_hmm=True)
        best_preds = hybrid_model.predict(X_val_hmm)
        best_metrics = regression_report(y_val_hmm, best_preds)

        metadata = {
            "model_name": "hmm_xgboost_hybrid",
            "ticker": ticker,
            "trained_on": str(datetime.date.today()),
            "feature_count": X_val_hmm.shape[1],
            "feature_names": feat_names,
            **best_metrics,
        }
        save_model(hybrid_model, f"{model_dir}/best_model.pkl")
        save_metadata(metadata, f"{model_dir}/best_model_metadata.json")
        logger.info(f"Best model saved | RMSE={best_metrics['rmse']:.6f}")

    # 5. Regime classifier
    if "hmm_hard_label" in df.columns:
        logger.info("Training XGBoost regime classifier...")
        X_cls_train, y_cls_train, _ = prepare_classification_data(df_train)
        X_cls_val, y_cls_val, _ = prepare_classification_data(df_val)
        cls_model = train_xgboost_classifier(
            X_cls_train, y_cls_train, X_cls_val, y_cls_val, xgb_params, "xgb_classifier"
        )
        save_model(cls_model, f"{model_dir}/{ticker.replace('=','_')}_classifier.pkl")


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    train_all_models(ticker)
