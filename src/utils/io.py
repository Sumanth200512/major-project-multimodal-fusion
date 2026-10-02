"""
I/O, Logging, and Artifact Validation Utilities.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np

# Configure standard logging
LOG_FORMAT = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    return logging.getLogger(name)

logger = get_logger(__name__)

def save_artifact(obj: Any, path: Path) -> Path:
    """Save a Python object/model/scaler using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved artifact to {path}")
    return path

def load_artifact(path: Path) -> Any:
    """Load a Python object/model/scaler using joblib."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    obj = joblib.load(path)
    logger.info(f"Loaded artifact from {path}")
    return obj

def save_json(data: Dict[str, Any], path: Path) -> Path:
    """Save dictionary to formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON to {path}")
    return path

def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_numpy(arr: np.ndarray, path: Path) -> Path:
    """Save numpy array to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.info(f"Saved numpy array ({arr.shape}) to {path}")
    return path

def load_numpy(path: Path) -> np.ndarray:
    """Load numpy array from file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Numpy file not found at {path}")
    arr = np.load(path)
    logger.info(f"Loaded numpy array ({arr.shape}) from {path}")
    return arr
