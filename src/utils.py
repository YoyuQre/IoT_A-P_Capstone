"""
Utility functions for deterministic random generation, I/O helpers, and common operations.
"""

import numpy as np
from pathlib import Path
from typing import Any, Optional
import json
import pandas as pd

from .config import (
    MASTER_SEED,
    NUMPY_SEED,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_RESULTS_DIR,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_REPORTS_DIR,
    EPSILON,
)


# ============================================================
# DETERMINISTIC RANDOM NUMBER GENERATION
# ============================================================
class DeterministicRNG:
    """
    Wrapper around numpy's Generator to ensure reproducible random numbers.
    
    Usage:
        rng = DeterministicRNG(seed=42)
        value = rng.uniform(0, 1)
        arr = rng.normal(0, 1, size=100)
    """
    
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed if seed is not None else NUMPY_SEED
        self._rng = np.random.default_rng(self._seed)
    
    @property
    def seed(self) -> int:
        return self._seed
    
    @property
    def rng(self) -> np.random.Generator:
        return self._rng
    
    # Delegate common methods
    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any = None) -> Any:
        return self._rng.uniform(low, high, size)
    
    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Any = None) -> Any:
        return self._rng.normal(loc, scale, size)
    
    def choice(self, a: Any, size: Any = None, replace: bool = True, p: Any = None) -> Any:
        return self._rng.choice(a, size=size, replace=replace, p=p)
    
    def integers(self, low: int, high: int, size: Any = None) -> Any:
        return self._rng.integers(low, high, size)
    
    def shuffle(self, x: Any) -> None:
        self._rng.shuffle(x)
    
    def permutation(self, x: Any) -> Any:
        return self._rng.permutation(x)
    
    def random(self, size: Any = None) -> Any:
        return self._rng.random(size)
    
    def exponential(self, scale: float = 1.0, size: Any = None) -> Any:
        return self._rng.exponential(scale, size)
    
    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size: Any = None) -> Any:
        return self._rng.lognormal(mean, sigma, size)
    
    def beta(self, a: float, b: float, size: Any = None) -> Any:
        return self._rng.beta(a, b, size)


# Global default RNG instance
_default_rng: Optional[DeterministicRNG] = None


def get_rng(seed: Optional[int] = None) -> DeterministicRNG:
    """Get or create the global deterministic RNG."""
    global _default_rng
    if _default_rng is None or seed is not None:
        _default_rng = DeterministicRNG(seed)
    return _default_rng


def set_global_seed(seed: int) -> None:
    """Set the global seed for reproducibility."""
    global _default_rng
    _default_rng = DeterministicRNG(seed)


# ============================================================
# SAFE STATISTICAL OPERATIONS
# ============================================================
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is near zero."""
    if abs(denominator) < EPSILON:
        return default
    return numerator / denominator


def safe_z_score(value: float, mean: float, std: float) -> float:
    """Calculate z-score with safe handling of zero/near-zero standard deviation."""
    return safe_divide(value - mean, std, default=0.0)


def safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    """Calculate mean and std with safe handling of edge cases."""
    if len(values) == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if len(values) == 1:
        std = EPSILON
    else:
        std = float(np.std(values, ddof=1))  # Sample standard deviation
        if std < EPSILON:
            std = EPSILON
    return mean, std


# ============================================================
# FILE I/O HELPERS
# ============================================================
def save_csv(df: pd.DataFrame, directory: Path, filename: str, index: bool = False) -> Path:
    """Save DataFrame to CSV in the specified directory."""
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    df.to_csv(filepath, index=index)
    return filepath


def load_csv(directory: Path, filename: str, **kwargs) -> pd.DataFrame:
    """Load DataFrame from CSV in the specified directory."""
    filepath = directory / filename
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")
    return pd.read_csv(filepath, **kwargs)


def save_json(data: Any, directory: Path, filename: str, indent: int = 2) -> Path:
    """Save data as JSON in the specified directory."""
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    return filepath


def load_json(directory: Path, filename: str) -> Any:
    """Load data from JSON in the specified directory."""
    filepath = directory / filename
    if not filepath.exists():
        raise FileNotFoundError(f"JSON not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================
# TIMESTAMP HELPERS
# ============================================================
def generate_timestamps(
    start: str,
    count: int,
    interval_seconds: int,
    rng: Optional[DeterministicRNG] = None,
    jitter_seconds: int = 5
) -> list[str]:
    """
    Generate a list of ISO 8601 timestamps with optional jitter.
    
    Args:
        start: Start timestamp in ISO format
        count: Number of timestamps to generate
        interval_seconds: Base interval between timestamps
        rng: Random number generator for jitter
        jitter_seconds: Maximum random jitter to add (±seconds)
    
    Returns:
        List of ISO 8601 timestamp strings
    """
    if rng is None:
        rng = get_rng()
    
    base = pd.Timestamp(start)
    timestamps = []
    
    for i in range(count):
        # Base time + small random jitter
        jitter = rng.integers(-jitter_seconds, jitter_seconds + 1)
        ts = base + pd.Timedelta(seconds=i * interval_seconds + jitter)
        timestamps.append(ts.isoformat())
    
    return timestamps


# ============================================================
# DEVICE ID GENERATION
# ============================================================
def generate_device_ids(
    categories: list[str],
    devices_per_category: int,
    prefix_map: dict[str, str]
) -> list[dict[str, str]]:
    """
    Generate device IDs with category assignments.
    
    Returns:
        List of dicts with keys: device_id, device_type
    """
    devices = []
    for category in categories:
        prefix = prefix_map[category]
        for i in range(1, devices_per_category + 1):
            device_id = f"{prefix}_{i:02d}"
            devices.append({
                "device_id": device_id,
                "device_type": category,
            })
    return devices


# ============================================================
# PROTOCOL SELECTION
# ============================================================
def select_protocol(
    protocols: tuple[str, ...],
    rng: Optional[DeterministicRNG] = None,
    weights: Optional[list[float]] = None
) -> str:
    """Select a protocol from the allowed list, optionally weighted."""
    if rng is None:
        rng = get_rng()
    return rng.choice(protocols, p=weights)


# ============================================================
# VALIDATION HELPERS
# ============================================================
def validate_dataframe_schema(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """Check if DataFrame has all required columns."""
    return all(col in df.columns for col in required_columns)


def validate_no_nan_inf(df: pd.DataFrame, numeric_columns: list[str]) -> bool:
    """Check that numeric columns have no NaN or Inf values."""
    for col in numeric_columns:
        if col in df.columns:
            if df[col].isna().any():
                return False
            if np.isinf(df[col]).any():
                return False
    return True