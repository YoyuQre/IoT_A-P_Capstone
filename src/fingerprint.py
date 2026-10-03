"""
Phase 3: Behavioural Fingerprinting

Computes statistical behavioural fingerprints for each device from normal traffic.
For each device and fingerprint feature, calculates mean and standard deviation.

Output: data/processed/device_fingerprints.csv
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from .config import (
    MASTER_SEED,
    FINGERPRINT_FEATURES,
    DATA_PROCESSED_DIR,
    DEVICE_FINGERPRINTS_CSV,
    EPSILON,
)
from .schemas import DeviceFingerprint, validate_fingerprint
from .utils import (
    DeterministicRNG,
    get_rng,
    safe_mean_std,
    load_csv,
    save_csv,
    validate_no_nan_inf,
)


# ============================================================
# FINGERPRINT CALCULATOR
# ============================================================
class FingerprintCalculator:
    """
    Calculates device-specific behavioural fingerprints from normal traffic.
    
    Only uses normal (non-anomalous) observations to establish baselines.
    """
    
    def __init__(
        self,
        fingerprint_features: Optional[list[str]] = None,
        epsilon: float = EPSILON,
    ):
        self.fingerprint_features = fingerprint_features or FINGERPRINT_FEATURES
        self.epsilon = epsilon
    
    def calculate_fingerprints(self, normal_traffic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fingerprints for all devices in the normal traffic dataset.
        
        Args:
            normal_traffic_df: DataFrame with normal observations (must have is_anomaly=False)
        
        Returns:
            DataFrame with columns: device_id, feature, mean, std
        """
        # Verify input has required columns
        required_cols = ["device_id"] + self.fingerprint_features
        for col in required_cols:
            if col not in normal_traffic_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure we only use normal traffic
        if "is_anomaly" in normal_traffic_df.columns:
            # Handle both boolean and numeric is_anomaly columns
            anomaly_col = normal_traffic_df["is_anomaly"]
            if anomaly_col.dtype == bool:
                normal_mask = ~anomaly_col
            else:
                normal_mask = anomaly_col == 0
            normal_df = normal_traffic_df[normal_mask].copy()
            if len(normal_df) == 0:
                raise ValueError("No normal observations found in dataset")
        else:
            normal_df = normal_traffic_df.copy()
        
        fingerprint_rows = []
        
        for device_id in normal_df["device_id"].unique():
            device_df = normal_df[normal_df["device_id"] == device_id]
            
            for feature in self.fingerprint_features:
                values = device_df[feature].values
                
                # Calculate mean and std with safe handling
                mean, std = safe_mean_std(values)
                
                # Ensure std is not below epsilon
                if std < self.epsilon:
                    std = self.epsilon
                
                fp: DeviceFingerprint = {
                    "device_id": device_id,
                    "feature": feature,
                    "mean": round(mean, 6),
                    "std": round(std, 10),  # Higher precision for small std values
                }
                
                # Validate
                if not validate_fingerprint(fp):
                    raise ValueError(f"Invalid fingerprint generated: {fp}")
                
                fingerprint_rows.append(fp)
        
        return pd.DataFrame(fingerprint_rows)
    
    def save_fingerprints(self, fingerprints_df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
        """Save fingerprints to CSV."""
        if output_path is None:
            output_path = DATA_PROCESSED_DIR / DEVICE_FINGERPRINTS_CSV
        return save_csv(fingerprints_df, output_path.parent, output_path.name)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================
def compute_device_fingerprints(
    normal_traffic_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    fingerprint_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to load normal traffic, compute fingerprints, and save.
    
    Args:
        normal_traffic_path: Path to normal_traffic.csv (default: data/raw/normal_traffic.csv)
        output_path: Output path for fingerprints (default: data/processed/device_fingerprints.csv)
        fingerprint_features: Features to fingerprint (default: FINGERPRINT_FEATURES)
    
    Returns:
        DataFrame with device fingerprints
    """
    if normal_traffic_path is None:
        normal_traffic_path = DATA_PROCESSED_DIR.parent / "raw" / "normal_traffic.csv"
    
    # Load normal traffic
    df = load_csv(normal_traffic_path.parent, normal_traffic_path.name)
    
    # Calculate fingerprints
    calculator = FingerprintCalculator(fingerprint_features=fingerprint_features)
    fingerprints = calculator.calculate_fingerprints(df)
    
    # Save
    calculator.save_fingerprints(fingerprints, output_path)
    
    return fingerprints


def load_device_fingerprints(
    fingerprints_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load device fingerprints from CSV."""
    if fingerprints_path is None:
        fingerprints_path = DATA_PROCESSED_DIR / DEVICE_FINGERPRINTS_CSV
    
    return load_csv(fingerprints_path.parent, fingerprints_path.name)


def get_device_fingerprint_dict(
    fingerprints_df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Convert fingerprints DataFrame to nested dict for easy lookup.
    
    Returns:
        {device_id: {feature: {"mean": float, "std": float}}}
    """
    result = {}
    
    for device_id in fingerprints_df["device_id"].unique():
        device_fps = fingerprints_df[fingerprints_df["device_id"] == device_id]
        result[device_id] = {}
        
        for _, row in device_fps.iterrows():
            result[device_id][row["feature"]] = {
                "mean": row["mean"],
                "std": row["std"],
            }
    
    return result


# ============================================================
# VALIDATION
# ============================================================
def validate_fingerprints(fingerprints_df: pd.DataFrame) -> dict:
    """
    Validate computed fingerprints.
    
    Returns:
        Dict with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # Check required columns
    required_cols = ["device_id", "feature", "mean", "std"]
    missing = [c for c in required_cols if c not in fingerprints_df.columns]
    if missing:
        results["valid"] = False
        results["errors"].append(f"Missing columns: {missing}")
        return results
    
    # Check each fingerprint entry
    for _, row in fingerprints_df.iterrows():
        fp = row.to_dict()
        if not validate_fingerprint(fp):
            results["valid"] = False
            results["errors"].append(f"Invalid fingerprint entry: {fp}")
    
    # Check for NaN/inf in numeric columns
    if not validate_no_nan_inf(fingerprints_df, ["mean", "std"]):
        results["valid"] = False
        results["errors"].append("NaN or Inf values in mean/std")
    
    # Check std >= epsilon
    if (fingerprints_df["std"] < EPSILON / 2).any():
        results["warnings"].append("Some std values are very small (< epsilon/2)")
    
    # Stats
    results["stats"] = {
        "total_entries": len(fingerprints_df),
        "unique_devices": fingerprints_df["device_id"].nunique(),
        "features": fingerprints_df["feature"].unique().tolist(),
        "mean_range": {
            "min": fingerprints_df["mean"].min(),
            "max": fingerprints_df["mean"].max(),
        },
        "std_range": {
            "min": fingerprints_df["std"].min(),
            "max": fingerprints_df["std"].max(),
        },
    }
    
    return results


if __name__ == "__main__":
    # Direct execution for testing
    print("Computing device fingerprints...")
    fps = compute_device_fingerprints()
    print(f"Computed {len(fps)} fingerprint entries for {fps['device_id'].nunique()} devices")
    
    validation = validate_fingerprints(fps)
    print(f"Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation["errors"]:
        print(f"Errors: {validation['errors']}")
    if validation["warnings"]:
        print(f"Warnings: {validation['warnings']}")
    print(f"Stats: {validation['stats']}")