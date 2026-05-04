"""
src/labeling/build_regimes.py
Hidden Markov Model (HMM) based regime labeling.
Fits a Gaussian HMM on log returns and assigns latent state posteriors.
Used as meta-features in the hybrid model and for classification targets.
Author: Fidel Mehra
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from src.utils.io import load_config, load_dataframe, save_dataframe, save_model

# Empirical label map (assigned after sorting states by mean return variance)
REGIME_LABELS = {0: "Low-Vol", 1: "Normal", 2: "Stress"}


def fit_hmm(
    returns: np.ndarray,
    n_components: int = 3,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
) -> GaussianHMM:
    """
    Fit a Gaussian HMM on the return series.

    Parameters
    ----------
    returns         : 1-D numpy array of log returns
    n_components    : Number of hidden states
    covariance_type : HMM covariance structure ('full', 'diag', 'tied')
    n_iter          : EM iterations
    random_state    : Reproducibility seed

    Returns
    -------
    Fitted GaussianHMM model
    """
    X = returns.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)
    logger.info(
        f"HMM fitted | n_states={n_components} "
        f"| log-likelihood={model.score(X):.2f}"
    )
    return model


def get_state_posteriors(
    model: GaussianHMM, returns: np.ndarray
) -> np.ndarray:
    """
    Compute posterior state probabilities (soft assignments).

    Returns
    -------
    np.ndarray of shape (n_samples, n_states) - posterior probabilities
    """
    X = returns.reshape(-1, 1)
    posteriors = model.predict_proba(X)  # shape: (T, n_states)
    return posteriors


def reorder_states_by_volatility(
    model: GaussianHMM, returns: np.ndarray
) -> np.ndarray:
    """
    Reorder HMM states so that state 0 = lowest vol, state 2 = highest vol.
    This makes the label mapping consistent across assets.
    """
    hard_states = model.predict(returns.reshape(-1, 1))
    state_vols = [
        np.std(returns[hard_states == s]) if np.sum(hard_states == s) > 0 else 0
        for s in range(model.n_components)
    ]
    order = np.argsort(state_vols)  # ascending: low-vol -> high-vol
    return order


def build_regimes(
    ticker: str,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Full regime labeling pipeline for a single asset.

    Loads processed features, fits HMM on training portion only,
    predicts states walk-forward, adds state posteriors as features,
    and saves augmented feature table.

    Returns updated DataFrame with HMM columns appended.
    """
    cfg = load_config(config_path)
    hmm_cfg = cfg["training"]["hmm"]
    processed_dir = cfg["data"]["processed_dir"]
    model_dir = cfg["training"]["model_registry_dir"]
    test_size = cfg["training"]["test_size"]

    feat_path = f"{processed_dir}/{ticker.replace('=', '_')}_features.csv"
    df = load_dataframe(feat_path)

    returns = df["log_return"].values

    # Split: fit HMM on training portion only to prevent lookahead
    split_idx = int(len(returns) * (1 - test_size))
    train_returns = returns[:split_idx]

    logger.info(f"Fitting HMM on {ticker} | train_n={len(train_returns)}")
    model = fit_hmm(
        train_returns,
        n_components=hmm_cfg["n_components"],
        covariance_type=hmm_cfg["covariance_type"],
        n_iter=hmm_cfg["n_iter"],
        random_state=hmm_cfg["random_state"],
    )

    # Get state posteriors for full series
    posteriors = get_state_posteriors(model, returns)  # (T, n_states)

    # Reorder states by volatility for consistent labeling
    order = reorder_states_by_volatility(model, train_returns)
    posteriors_reordered = posteriors[:, order]

    # Append posterior columns to feature table
    n_states = model.n_components
    for i in range(n_states):
        df[f"hmm_state_{i}"] = posteriors_reordered[:, i]

    # Hard label (argmax of posterior)
    hard_label = np.argmax(posteriors_reordered, axis=1)
    df["hmm_hard_label"] = hard_label
    df["hmm_regime"] = [REGIME_LABELS.get(s, str(s)) for s in hard_label]

    # Save augmented features
    out_path = f"{processed_dir}/{ticker.replace('=', '_')}_features.csv"
    save_dataframe(df, out_path)
    logger.info(f"{ticker}: HMM states appended and saved")

    # Save HMM model
    hmm_path = f"{model_dir}/{ticker.replace('=', '_')}_hmm.pkl"
    save_model(model, hmm_path)

    return df


def build_all_regimes(config_path: str = "config/config.yaml") -> None:
    """Build regime labels for all configured assets."""
    cfg = load_config(config_path)
    for asset in cfg["data"]["assets"]:
        ticker = asset["ticker"]
        try:
            build_regimes(ticker, config_path=config_path)
        except Exception as exc:
            logger.error(f"Regime labeling failed for {ticker}: {exc}")


if __name__ == "__main__":
    build_all_regimes()
