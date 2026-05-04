"""
src/utils/io.py
File I/O helpers for loading configs, reading/writing dataframes, and
serialising model artifacts.
Author: Fidel Mehra
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML config file and return as a nested dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return cfg


def ensure_dir(directory: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, filepath: str | Path, index: bool = True) -> None:
    """Save a pandas DataFrame as a CSV file."""
    path = Path(filepath)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    logger.info(f"DataFrame saved to {path} | shape={df.shape}")


def load_dataframe(filepath: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(path, index_col=0, parse_dates=parse_dates)
    logger.info(f"DataFrame loaded from {path} | shape={df.shape}")
    return df


def save_model(model: Any, filepath: str | Path) -> None:
    """Serialise a model artifact with pickle."""
    path = Path(filepath)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(filepath: str | Path) -> Any:
    """Deserialise a pickled model artifact."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model


def save_metadata(metadata: dict, filepath: str | Path) -> None:
    """Save model metadata dict as a JSON file."""
    path = Path(filepath)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {path}")


def load_metadata(filepath: str | Path) -> dict:
    """Load a JSON metadata file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {filepath}")
    with open(path, "r") as f:
        metadata = json.load(f)
    return metadata


def list_raw_files(raw_dir: str = "data/raw") -> list[Path]:
    """List all CSV files in the raw data directory."""
    return sorted(Path(raw_dir).glob("*.csv"))
