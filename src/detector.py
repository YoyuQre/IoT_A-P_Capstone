"""
Phase 5: Statistical Anomaly Detection

Calculates feature-level standardized deviations and composite anomaly score.
Compares observations against device-specific normal fingerprints.

Detection formula:
  Z_feature = (x - μ) / σ
  A = (|Z_packet_rate| + |Z_destination_count| + |Z_packet_size|) / 3

Classification:
  A >= threshold → anomalous
  A < threshold → normal

Thresholds: 2.0, 3.0, 4.0
"""

from pathlib import Path
from typing import Optional, Literal
import pandas as pd
import numpy as np

from .config import (
    MASTER_SEED,
    FINGERPRINT_FEATURES,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
    EPSILON,
    DATA_PROCESSED_DIR,
    DATA_RESULTS_DIR,
)
from .schemas import DetectionResult, validate_observation
from .utils import (
    DeterministicRNG,
    get_rng,
    safe_z_score,
    load_csv,
    save_csv,
)
from .fingerprint import get_device_fingerprint_dict
from .fingerprint import load_device_fingerprints


# ============================================================
# ANOMALY DETECTOR
# ============================================================
class AnomalyDetector:
    """
    Statistical anomaly detector using device-specific fingerprints.
    
    For each observation, computes z-scores against the device's normal
    behavioural fingerprint, then calculates a composite anomaly score.
    """
    
    def __init__(
        self,
        fingerprints_df: pd.DataFrame,
        thresholds: Optional[list[float]] = None,
        fingerprint_features: Optional[list[str]] = None,
    ):
        """
        Initialize detector with fingerprints.
        
        Args:
            fingerprints_df: DataFrame with columns device_id, feature, mean, std
            thresholds: List of detection thresholds (default: [2.0, 3.0, 4.0])
            fingerprint_features: Features to use for detection (default: FINGERPRINT_FEATURES)
        """
        self.fingerprints_df = fingerprints_df.copy()
        self.thresholds = thresholds or DETECTION_THRESHOLDS
        self.fingerprint_features = fingerprint_features or FINGERPRINT_FEATURES
        
        # Convert fingerprints to lookup dict for fast access
        self.fingerprint_dict = get_device_fingerprint_dict(fingerprints_df)
        
        # Verify all required features are present
        for device_id, features in self.fingerprint_dict.items():
            for feature in self.fingerprint_features:
                if feature not in features:
                    raise ValueError(f"Missing fingerprint for {device_id}/{feature}")
    
    def calculate_z_scores(self, observations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate z-scores for each fingerprint feature.
        
        Args:
            observations_df: DataFrame with observations (must have device_id and fingerprint features)
        
        Returns:
            DataFrame with added z-score columns
        """
        df = observations_df.copy()
        
        # Initialize z-score columns
        for feature in self.fingerprint_features:
            z_col = f"z_{feature}"
            df[z_col] = 0.0
        
        # Calculate z-scores per device per feature
        for device_id in df["device_id"].unique():
            if device_id not in self.fingerprint_dict:
                # No fingerprint for this device - skip (z-scores remain 0)
                continue
            
            device_mask = df["device_id"] == device_id
            device_fps = self.fingerprint_dict[device_id]
            
            for feature in self.fingerprint_features:
                if feature not in df.columns:
                    continue
                
                fp = device_fps[feature]
                mean = fp["mean"]
                std = fp["std"]
                
                # Calculate z-score: (x - mean) / std
                values = df.loc[device_mask, feature].values
                z_scores = np.array([safe_z_score(v, mean, std) for v in values])
                
                df.loc[device_mask, f"z_{feature}"] = z_scores
        
        return df
    
    def calculate_anomaly_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite anomaly score A.
        
        A = (|Z_packet_rate| + |Z_destination_count| + |Z_packet_size|) / 3
        """
        df = df.copy()
        
        # Sum absolute z-scores
        z_cols = [f"z_{f}" for f in self.fingerprint_features]
        df["anomaly_score"] = df[z_cols].abs().sum(axis=1) / len(z_cols)
        
        return df
    
    def classify(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Classify observations as anomalous/normal based on threshold.
        
        Args:
            df: DataFrame with anomaly_score column
            threshold: Detection threshold
        
        Returns:
            DataFrame with added predicted_label column
        """
        df = df.copy()
        df["predicted_label"] = df["anomaly_score"] >= threshold
        df["threshold"] = threshold
        return df
    
    def detect(
        self,
        observations_df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> list[DetectionResult]:
        """
        Run complete detection pipeline on observations.
        
        Args:
            observations_df: DataFrame with observations (must have ground truth columns)
            threshold: Single threshold to use (default: DEFAULT_THRESHOLD)
        
        Returns:
            List of DetectionResult dicts
        """
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        
        # Calculate z-scores
        df = self.calculate_z_scores(observations_df)
        
        # Calculate anomaly score
        df = self.calculate_anomaly_score(df)
        
        # Classify
        df = self.classify(df, threshold)
        
        # Build detection results
        results = []
        for _, row in df.iterrows():
            result: DetectionResult = {
                "device_id": row["device_id"],
                "timestamp": row["timestamp"],
                "anomaly_type": row.get("anomaly_type", "normal"),
                "ground_truth": bool(row.get("is_anomaly", False)),
                "z_packet_rate": float(row.get("z_packet_rate", 0.0)),
                "z_destination_count": float(row.get("z_destination_count", 0.0)),
                "z_packet_size": float(row.get("z_packet_size", 0.0)),
                "anomaly_score": float(row["anomaly_score"]),
                "predicted_label": bool(row["predicted_label"]),
                "threshold": float(row["threshold"]),
            }
            results.append(result)
        
        return results
    
    def detect_all_thresholds(
        self,
        observations_df: pd.DataFrame,
    ) -> dict[float, list[DetectionResult]]:
        """
        Run detection with all configured thresholds.
        
        Returns:
            Dict mapping threshold -> list of DetectionResult
        """
        results = {}
        for threshold in self.thresholds:
            results[threshold] = self.detect(observations_df, threshold)
        return results


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================
def run_detection(
    fingerprints_path: Optional[Path] = None,
    observations_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    threshold: Optional[float] = None,
    thresholds: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Convenience function to load fingerprints + observations, run detection, save results.
    
    Args:
        fingerprints_path: Path to device_fingerprints.csv
        observations_path: Path to anomalous_traffic.csv (with ground truth)
        output_path: Output path for detection results
        threshold: Single threshold (default: DEFAULT_THRESHOLD)
        thresholds: Multiple thresholds (overrides threshold if provided)
    
    Returns:
        DataFrame with detection results
    """
    if fingerprints_path is None:
        fingerprints_path = DATA_PROCESSED_DIR / "device_fingerprints.csv"
    
    if observations_path is None:
        observations_path = DATA_PROCESSED_DIR / "anomalous_traffic_combined.csv"
    
    # Load fingerprints
    fingerprints = load_device_fingerprints(fingerprints_path)
    
    # Load observations
    observations = load_csv(observations_path.parent, observations_path.name)
    
    # Run detection
    detector = AnomalyDetector(fingerprints, thresholds=thresholds)
    
    if thresholds:
        all_results = detector.detect_all_thresholds(observations)
        # Combine all threshold results
        combined_results = []
        for threshold, results in all_results.items():
            combined_results.extend(results)
        results_df = pd.DataFrame(combined_results)
    else:
        threshold = threshold or DEFAULT_THRESHOLD
        results = detector.detect(observations, threshold)
        results_df = pd.DataFrame(results)
    
    # Save
    if output_path is None:
        output_path = DATA_RESULTS_DIR / "detection_results.csv"
    save_csv(results_df, output_path.parent, output_path.name)
    
    return results_df


def run_detection_on_all_scenarios(
    fingerprints_path: Optional[Path] = None,
    observations_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Run detection on all anomaly scenario files.
    
    Returns:
        Dict mapping anomaly_type -> detection results DataFrame
    """
    if fingerprints_path is None:
        fingerprints_path = DATA_PROCESSED_DIR / "device_fingerprints.csv"
    
    if observations_dir is None:
        observations_dir = DATA_PROCESSED_DIR
    
    if output_dir is None:
        output_dir = DATA_RESULTS_DIR
    
    fingerprints = load_device_fingerprints(fingerprints_path)
    detector = AnomalyDetector(fingerprints)
    
    anomaly_types = ["traffic_spike", "destination_expansion", "combined"]
    results = {}
    
    for anomaly_type in anomaly_types:
        obs_path = observations_dir / f"anomalous_traffic_{anomaly_type}.csv"
        if not obs_path.exists():
            continue
        
        observations = load_csv(obs_path.parent, obs_path.name)
        detection_results = detector.detect_all_thresholds(observations)
        
        # Combine all thresholds
        combined = []
        for threshold, res in detection_results.items():
            combined.extend(res)
        
        results_df = pd.DataFrame(combined)
        
        output_path = output_dir / f"detection_results_{anomaly_type}.csv"
        save_csv(results_df, output_path.parent, output_path.name)
        
        results[anomaly_type] = results_df
    
    return results


# ============================================================
# VALIDATION
# ============================================================
def validate_detection_results(results_df: pd.DataFrame) -> dict:
    """
    Validate detection results.
    
    Returns:
        Validation results dict
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # Check required columns
    required_cols = [
        "device_id", "timestamp", "anomaly_type", "ground_truth",
        "z_packet_rate", "z_destination_count", "z_packet_size",
        "anomaly_score", "predicted_label", "threshold",
    ]
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        validation["valid"] = False
        validation["errors"].append(f"Missing columns: {missing}")
        return validation
    
    # Check data types
    if not pd.api.types.is_bool_dtype(results_df["ground_truth"]):
        validation["warnings"].append("ground_truth not boolean")
    if not pd.api.types.is_bool_dtype(results_df["predicted_label"]):
        validation["warnings"].append("predicted_label not boolean")
    
    # Check anomaly_score is non-negative
    if (results_df["anomaly_score"] < 0).any():
        validation["valid"] = False
        validation["errors"].append("Negative anomaly_score values")
    
    # Check z-scores are finite
    z_cols = ["z_packet_rate", "z_destination_count", "z_packet_size"]
    for col in z_cols:
        if not np.isfinite(results_df[col]).all():
            validation["valid"] = False
            validation["errors"].append(f"Non-finite values in {col}")
    
    # Check thresholds are valid
    if not results_df["threshold"].isin(DETECTION_THRESHOLDS).all():
        validation["warnings"].append("Unexpected threshold values")
    
    # Summary stats
    validation["stats"] = {
        "total_results": len(results_df),
        "unique_devices": results_df["device_id"].nunique(),
        "thresholds": sorted(results_df["threshold"].unique().tolist()),
        "anomaly_score_range": {
            "min": results_df["anomaly_score"].min(),
            "max": results_df["anomaly_score"].max(),
            "mean": results_df["anomaly_score"].mean(),
        },
        "ground_truth_anomalies": int(results_df["ground_truth"].sum()),
        "predicted_anomalies": int(results_df["predicted_label"].sum()),
    }
    
    return validation


if __name__ == "__main__":
    # Direct execution for testing
    print("Running anomaly detection...")
    results = run_detection()
    print(f"Generated {len(results)} detection results")
    
    validation = validate_detection_results(results)
    print(f"Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation["errors"]:
        print(f"Errors: {validation['errors']}")
    if validation["warnings"]:
        print(f"Warnings: {validation['warnings']}")
    print(f"Stats: {validation['stats']}")