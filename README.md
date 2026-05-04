# Stochastic Volatility Forecasting & Risk Regime Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CI](https://github.com/fidelmehra/stochastic-volatility-mlops/actions/workflows/ci.yml/badge.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2.svg)

> **Author:** Fidel Mehra  
> **Stack:** Python · XGBoost · LSTM · HMM · FastAPI · MLflow · Docker · GitHub Actions  
> **Domain:** Quantitative Finance · Time Series · Stochastic Processes · MLOps

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why Stochastic Volatility](#why-stochastic-volatility)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Repository Structure](#repository-structure)
5. [Dataset](#dataset)
6. [Feature Engineering](#feature-engineering)
7. [Regime Labeling](#regime-labeling)
8. [Models](#models)
9. [Validation Strategy](#validation-strategy)
10. [Experiment Tracking](#experiment-tracking)
11. [API Reference](#api-reference)
12. [Docker](#docker)
13. [CI/CD](#cicd)
14. [Results](#results)
15. [Installation & Quickstart](#installation--quickstart)
16. [Future Work](#future-work)
17. [Author](#author)

---

## Project Overview

This repository implements a **production-grade, end-to-end MLOps pipeline** for:

- **Forecasting next-period realised volatility** of crypto and equity assets using time-series feature engineering and machine learning.
- **Detecting latent market regimes** (Low-Vol, Normal, Stress) using a Hidden Markov Model, whose inferred state probabilities are fed back into the ML regressors as meta-features.
- **Serving live predictions** through a FastAPI microservice containerised with Docker.
- **Tracking all experiments** reproducibly with MLflow.
- **Enforcing code quality** through a GitHub Actions CI pipeline.

The project is intentionally structured as a real engineering system rather than a notebook-only experiment. Every stage — from raw data ingestion to a live REST endpoint — is modular, tested, and reproducible.

---

## Why Stochastic Volatility

Financial return series violate the assumptions of constant variance. Empirically they exhibit:

- **Volatility clustering**: large moves tend to follow large moves (Mandelbrot, 1963).
- **Heavy tails**: return distributions have excess kurtosis well beyond the Gaussian.
- **Regime changes**: the market alternates between latent states with distinct volatility levels.
- **Mean-reversion of volatility**: high volatility periods do not persist indefinitely.

Stochastic volatility models (Heston, 1993; Hull-White, 1987) capture these properties by treating volatility itself as a latent diffusion process:

```
dS_t = mu * S_t dt + sqrt(V_t) * S_t dW_t^S
dV_t = kappa*(theta - V_t) dt + xi * sqrt(V_t) dW_t^V
corr(dW^S, dW^V) = rho
```

This project uses those stochastic intuitions to motivate a data-driven ML system: rather than estimating the SDE parameters directly, we engineer features that reflect the latent volatility process and train supervised models to predict its next realisation.

---

## Pipeline Architecture

```
+------------------+     +-------------------+     +----------------------+
|  Data Ingestion  | --> | Validation &      | --> | Feature Engineering  |
|  (yfinance API)  |     | Schema Checks     |     | (lags, RV, EWMA,     |
|                  |     |                   |     |  jumps, HMM states)  |
+------------------+     +-------------------+     +----------+-----------+
                                                              |
                                                              v
                                              +---------------+-------------+
                                              |   Model Training Layer      |
                                              |  Naive | GARCH | XGBoost   |
                                              |  LSTM  | HMM+XGB Hybrid    |
                                              +---------------+-------------+
                                                              |
                                              +---------------+-------------+
                                              |  Rolling Backtest &         |
                                              |  Evaluation (RMSE, QLIKE,  |
                                              |  F1, ROC-AUC)              |
                                              +---------------+-------------+
                                                              |
                                              +---------------+-------------+
                                              |  MLflow Model Registry      |
                                              |  (best model artifact)      |
                                              +---------------+-------------+
                                                              |
                                              +---------------+-------------+
                                              |  FastAPI Inference Service  |
                                              |  /predict-volatility        |
                                              |  /predict-regime            |
                                              |  /health  /model-info       |
                                              +---------------+-------------+
                                                              |
                                              +---------------+-------------+
                                              |  Docker + GitHub Actions CI |
                                              +-----------------------------+
```

---

## Repository Structure

```
stochastic-volatility-mlops/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                  # Lint, test, Docker build on every push
├── config/
│   └── config.yaml                 # All hyperparams, paths, asset config
├── data/
│   ├── raw/                        # Raw OHLCV CSVs from yfinance
│   └── processed/                  # Feature tables ready for training
├── models/
│   └── registry/                   # Serialised model artifacts + metadata
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   └── 02_model_diagnostics.ipynb  # Backtest plots & regime visualisation
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── fetch_market_data.py    # Download OHLCV from yfinance
│   ├── validation/
│   │   ├── __init__.py
│   │   └── validate_data.py        # Schema + quality checks
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py       # Lag, RV, EWMA, jump, calendar features
│   ├── labeling/
│   │   ├── __init__.py
│   │   └── build_regimes.py        # HMM-based regime labeling
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_baselines.py      # Naive & GARCH baselines
│   │   ├── train_ml.py             # XGBoost, LSTM, hybrid models
│   │   └── tune_models.py          # Optuna hyperparameter tuning
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── backtest.py             # Rolling-window backtester
│   │   └── metrics.py              # RMSE, MAE, QLIKE, F1, ROC-AUC
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── log_experiments.py      # MLflow logging helpers
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py            # Load model & run inference
│   └── utils/
│       ├── __init__.py
│       └── io.py                   # File I/O helpers
├── app/
│   ├── main.py                     # FastAPI app entrypoint
│   ├── schemas.py                  # Pydantic request/response models
│   └── service.py                  # Prediction service layer
├── tests/
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_api.py
├── Dockerfile
└── docker-compose.yml
```

---

## Dataset

The pipeline ingests **daily OHLCV data** for a configurable set of assets via `yfinance`. Default assets:

| Asset | Ticker | Description |
|---|---|---|
| Bitcoin | BTC-USD | Crypto flagship, high volatility |
| Ethereum | ETH-USD | Crypto alt-coin |
| S&P 500 ETF | SPY | Equity market benchmark |
| Gold | GC=F | Safe-haven commodity |

The ingestion script downloads data for a configurable historical window (default: 5 years) and saves raw CSVs to `data/raw/`. Log returns and realised volatility are computed in preprocessing and saved to `data/processed/`.

**Target variable:** 5-day ahead realised volatility, defined as:

```
RV_t = sqrt( sum_{i=0}^{4} r_{t+i}^2 )
```

where `r_t = log(P_t / P_{t-1})` is the log return.

---

## Feature Engineering

All features are engineered in `src/features/build_features.py`. The feature set is motivated by stochastic volatility theory and market microstructure.

| Feature Group | Features | Motivation |
|---|---|---|
| Lagged returns | `ret_lag_1` ... `ret_lag_20` | Autocorrelation in returns |
| Realised volatility | `rv_5`, `rv_10`, `rv_20`, `rv_30` | Volatility clustering |
| EWMA variance | `ewma_var_10`, `ewma_var_20` | Exponential weighting of recent vol |
| Rolling statistics | `roll_skew_20`, `roll_kurt_20` | Distribution shape dynamics |
| Jump proxy | `jump_flag`, `abs_ret_z` | Large-move detection |
| Volume features | `vol_change`, `log_volume` | Market activity signal |
| Drawdown | `max_drawdown_20` | Tail-risk proxy |
| Calendar | `day_of_week`, `month`, `is_month_end` | Seasonality effects |
| HMM state probs | `hmm_state_0`, `hmm_state_1`, `hmm_state_2` | Latent regime meta-features |

---

## Regime Labeling

A **3-state Gaussian Hidden Markov Model** is fitted to the return series in `src/labeling/build_regimes.py` using `hmmlearn`. The three inferred states correspond empirically to:

| State | Label | Characteristics |
|---|---|---|
| 0 | Low-Vol | Calm trending markets, low realised vol |
| 1 | Normal | Moderate volatility, mixed directionality |
| 2 | Stress | High volatility, sharp drawdowns, fat tails |

The HMM is fitted on training data only and applied to test periods in a walk-forward fashion to prevent lookahead bias. State posterior probabilities (not hard labels) are used as features to preserve uncertainty information.

---

## Models

### Baseline Models

| Model | Description |
|---|---|
| Naive | Previous period RV as forecast |
| Rolling Mean | 20-day rolling mean of RV |
| EWMA | Exponentially weighted variance forecast |

### Machine Learning Models

| Model | File | Target |
|---|---|---|
| XGBoost Regressor | `train_ml.py` | Next-period RV (regression) |
| LightGBM Regressor | `train_ml.py` | Next-period RV (regression) |
| LSTM (PyTorch) | `train_ml.py` | Next-period RV (sequence model) |
| XGBoost Classifier | `train_ml.py` | Regime label (3-class) |
| HMM + XGBoost Hybrid | `train_ml.py` | RV with HMM state probs as features |

The **HMM + XGBoost Hybrid** is the flagship model: it first infers latent volatility states via HMM, then passes state posterior probabilities as additional features into an XGBoost regressor, combining probabilistic stochastic modelling with discriminative ML.

---

## Validation Strategy

All models are evaluated using a **rolling-window backtest** in `src/evaluation/backtest.py`:

- **No random train-test splits** — all splits are time-ordered.
- **Expanding or fixed window** — configurable via `config.yaml`.
- **Multiple test folds** — default 5 folds over the final 20% of data.
- **No lookahead** — HMM states are estimated on training windows only.

### Regression Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| RMSE | `sqrt(mean((y_hat - y)^2))` | Scale-dependent forecast error |
| MAE | `mean(abs(y_hat - y))` | Robust to outliers |
| QLIKE | `log(sigma^2) + (y/sigma^2)` | Volatility-specific loss function |
| R² | `1 - SS_res/SS_tot` | Variance explained |

### Classification Metrics (Regime)

- Weighted F1-score
- Macro precision and recall
- ROC-AUC (one-vs-rest)
- Confusion matrix

---

## Experiment Tracking

All experiments are logged to **MLflow** (`src/tracking/log_experiments.py`):

- Hyperparameters per run
- All evaluation metrics
- Model artifact (serialised model + metadata JSON)
- Dataset version tag
- Feature schema

Start the MLflow UI locally:

```bash
mlflow ui --port 5000
```

Then open `http://localhost:5000` to compare runs, register the best model, and inspect artifacts.

---

## API Reference

The FastAPI service in `app/main.py` exposes the following endpoints:

### `GET /health`

Returns service status.

```json
{ "status": "ok", "model_version": "1.0.0" }
```

### `GET /model-info`

Returns loaded model metadata.

```json
{
  "model_name": "hmm_xgboost_hybrid",
  "trained_on": "2025-01-15",
  "asset": "BTC-USD",
  "rmse": 0.0031,
  "feature_count": 42
}
```

### `POST /predict-volatility`

Request:
```json
{
  "ticker": "BTC-USD",
  "features": {
    "rv_5": 0.042,
    "rv_10": 0.038,
    "rv_20": 0.035,
    "ret_lag_1": -0.012,
    "ret_lag_2": 0.008,
    "ewma_var_10": 0.0017,
    "jump_flag": 0,
    "hmm_state_0": 0.1,
    "hmm_state_1": 0.7,
    "hmm_state_2": 0.2
  }
}
```

Response:
```json
{
  "ticker": "BTC-USD",
  "predicted_rv": 0.0389,
  "confidence_interval": [0.031, 0.047],
  "horizon_days": 5
}
```

### `POST /predict-regime`

Request:
```json
{
  "ticker": "BTC-USD",
  "features": { "rv_5": 0.042, "ret_lag_1": -0.012 }
}
```

Response:
```json
{
  "ticker": "BTC-USD",
  "predicted_regime": "Stress",
  "regime_probabilities": { "Low-Vol": 0.05, "Normal": 0.21, "Stress": 0.74 }
}
```

---

## Docker

### Build and run locally

```bash
# Build the image
docker build -t stochastic-vol-api .

# Run the container
docker run -p 8000:8000 stochastic-vol-api

# API will be live at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Using docker-compose (API + MLflow)

```bash
docker-compose up --build
```

Services:
- `api` — FastAPI on port 8000
- `mlflow` — MLflow tracking server on port 5000

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request:

| Step | Action |
|---|---|
| Lint | `flake8` + `black --check` |
| Tests | `pytest tests/` with coverage |
| Docker build | Verify image builds without errors |

---

## Results

Rolling backtest results on BTC-USD (2022–2024):

| Model | RMSE | MAE | QLIKE | R² |
|---|---|---|---|---|
| Naive Persistence | 0.0071 | 0.0054 | -3.21 | 0.12 |
| Rolling Mean (20d) | 0.0063 | 0.0048 | -3.35 | 0.21 |
| EWMA | 0.0058 | 0.0044 | -3.41 | 0.27 |
| XGBoost | 0.0041 | 0.0031 | -3.68 | 0.51 |
| LSTM | 0.0039 | 0.0029 | -3.72 | 0.54 |
| **HMM + XGBoost Hybrid** | **0.0034** | **0.0026** | **-3.89** | **0.63** |

Regime classification (BTC-USD, 3-state HMM labels):

| Model | Weighted F1 | Macro ROC-AUC |
|---|---|---|
| Logistic Regression | 0.61 | 0.74 |
| XGBoost Classifier | 0.78 | 0.87 |

---

## Installation & Quickstart

### Prerequisites

- Python 3.10+
- Docker (optional, for containerised deployment)

### Setup

```bash
# Clone the repository
git clone https://github.com/fidelmehra/stochastic-volatility-mlops.git
cd stochastic-volatility-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# 1. Ingest raw market data
python -m src.ingestion.fetch_market_data

# 2. Validate data quality
python -m src.validation.validate_data

# 3. Build features
python -m src.features.build_features

# 4. Label regimes (HMM)
python -m src.labeling.build_regimes

# 5. Train all models
python -m src.training.train_baselines
python -m src.training.train_ml

# 6. Run rolling backtest
python -m src.evaluation.backtest

# 7. Launch MLflow UI
mlflow ui --port 5000

# 8. Start FastAPI server
uvicorn app.main:app --reload --port 8000
```

### Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Future Work

- [ ] Add Bayesian volatility model (PyMC) as an additional baseline
- [ ] Implement online learning / incremental retraining trigger
- [ ] Add Grafana dashboard for live monitoring of forecasts
- [ ] Deploy API to Railway or Render with auto-deploy on push
- [ ] Extend to multi-asset joint volatility forecasting
- [ ] Add GARCH-LSTM hybrid for richer temporal modelling
- [ ] Incorporate options-implied volatility (VIX) as exogenous feature

---

## Author

**Fidel Mehra**  
MSc Data Science — Newcastle University  
Research interests: stochastic processes, time series forecasting, quantitative finance, MLOps

---

*MIT License — see [LICENSE](LICENSE) for details.*
