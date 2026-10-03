"""
Phase 6: Experimental Framework

Runs predefined simulation experiments systematically:
- Experiment A: Normal traffic only (baseline false positive rate)
- Experiment B: Traffic spike anomalies
- Experiment C: Destination expansion anomalies  
- Experiment D: Combined anomalies
- Experiment E: Threshold study (compare thresholds 2, 3, 4)

Output: data/results/experiment_results.csv
"""

from pathlib import Path
from typing import Optional, Literal
import pandas as pd
import numpy as np
import json
from datetime import datetime

from .config import (
    MASTER_SEED,
    EXPERIMENT_NAMES,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
    DATA_PROCESSED_DIR,
    DATA_RESULTS_DIR,
    EXPERIMENT_RESULTS_CSV,
)
from .schemas import ExperimentConfig, ExperimentResult, ExperimentResult as ExperimentResultType
from .simulator import generate_normal_traffic
from .fingerprint import compute_device_fingerprints
from .anomalies import AnomalyInjector
from .detector import AnomalyDetector, validate_detection_results
from .utils import load_csv, save_csv


# ============================================================
# EXPERIMENT RUNNER
# ============================================================
class ExperimentRunner:
    """
    Runs the complete experimental pipeline for a given experiment configuration.
    """
    
    def __init__(self, seed: int = MASTER_SEED):
        self.seed = seed
        self.normal_df = None
        self.fingerprints = None
        self.detector = None
        self._setup_base_data()
    
    def _setup_base_data(self):
        """Generate normal traffic and fingerprints once."""
        print("  Generating normal traffic...")
        self.normal_df = generate_normal_traffic(seed=self.seed)
        
        print("  Computing fingerprints...")
        self.fingerprints = compute_device_fingerprints()
        
        print("  Initializing detector...")
        self.detector = AnomalyDetector(self.fingerprints)
    
    def run_experiment_a_normal(self) -> ExperimentResult:
        """
        Experiment A: Normal traffic only.
        
        Evaluates false positive rate on purely normal traffic.
        """
        print("  Running Experiment A: Normal traffic (baseline)...")
        
        # Use pure normal traffic (no anomalies injected)
        observations = self.normal_df.copy()
        observations["is_anomaly"] = False
        observations["anomaly_type"] = "normal"
        
        # Run detection at all thresholds
        all_results = self.detector.detect_all_thresholds(observations)
        
        # Combine results
        combined = []
        for threshold, results in all_results.items():
            combined.extend(results)
        
        results_df = pd.DataFrame(combined)
        
        # Compute metrics
        metrics = self._compute_metrics(results_df)
        
        config = ExperimentConfig(
            name="experiment_a_normal",
            anomaly_type="normal",
            threshold=DEFAULT_THRESHOLD,  # Reference threshold
            anomaly_device_fraction=0.0,
            anomaly_observation_fraction=0.0,
            seed=self.seed,
        )
        
        return ExperimentResult(
            experiment="experiment_a_normal",
            config=config,
            detection_results=combined,
            metrics=metrics,
            anomaly_type_metrics=[],  # No anomalies in this experiment
        )
    
    def run_experiment_b_traffic_spike(self) -> ExperimentResult:
        """
        Experiment B: Traffic spike anomalies.
        
        Injects traffic spike anomalies and evaluates detection.
        """
        print("  Running Experiment B: Traffic spike...")
        
        injector = AnomalyInjector(seed=self.seed)
        observations = injector.inject_anomalies(self.normal_df, "traffic_spike")
        
        # Run detection at all thresholds
        all_results = self.detector.detect_all_thresholds(observations)
        
        # Combine results
        combined = []
        for threshold, results in all_results.items():
            combined.extend(results)
        
        results_df = pd.DataFrame(combined)
        
        # Compute metrics
        metrics = self._compute_metrics(results_df)
        anomaly_type_metrics = self._compute_anomaly_type_metrics(results_df)
        
        config = ExperimentConfig(
            name="experiment_b_traffic_spike",
            anomaly_type="traffic_spike",
            threshold=DEFAULT_THRESHOLD,
            anomaly_device_fraction=0.4,
            anomaly_observation_fraction=0.15,
            seed=self.seed,
        )
        
        return ExperimentResult(
            experiment="experiment_b_traffic_spike",
            config=config,
            detection_results=combined,
            metrics=metrics,
            anomaly_type_metrics=anomaly_type_metrics,
        )
    
    def run_experiment_c_destination_expansion(self) -> ExperimentResult:
        """
        Experiment C: Destination expansion anomalies.
        """
        print("  Running Experiment C: Destination expansion...")
        
        injector = AnomalyInjector(seed=self.seed)
        observations = injector.inject_anomalies(self.normal_df, "destination_expansion")
        
        all_results = self.detector.detect_all_thresholds(observations)
        
        combined = []
        for threshold, results in all_results.items():
            combined.extend(results)
        
        results_df = pd.DataFrame(combined)
        
        metrics = self._compute_metrics(results_df)
        anomaly_type_metrics = self._compute_anomaly_type_metrics(results_df)
        
        config = ExperimentConfig(
            name="experiment_c_destination_expansion",
            anomaly_type="destination_expansion",
            threshold=DEFAULT_THRESHOLD,
            anomaly_device_fraction=0.4,
            anomaly_observation_fraction=0.15,
            seed=self.seed,
        )
        
        return ExperimentResult(
            experiment="experiment_c_destination_expansion",
            config=config,
            detection_results=combined,
            metrics=metrics,
            anomaly_type_metrics=anomaly_type_metrics,
        )
    
    def run_experiment_d_combined(self) -> ExperimentResult:
        """
        Experiment D: Combined anomalies (traffic spike + destination expansion).
        """
        print("  Running Experiment D: Combined anomalies...")
        
        injector = AnomalyInjector(seed=self.seed)
        observations = injector.inject_anomalies(self.normal_df, "combined")
        
        all_results = self.detector.detect_all_thresholds(observations)
        
        combined = []
        for threshold, results in all_results.items():
            combined.extend(results)
        
        results_df = pd.DataFrame(combined)
        
        metrics = self._compute_metrics(results_df)
        anomaly_type_metrics = self._compute_anomaly_type_metrics(results_df)
        
        config = ExperimentConfig(
            name="experiment_d_combined",
            anomaly_type="combined",
            threshold=DEFAULT_THRESHOLD,
            anomaly_device_fraction=0.4,
            anomaly_observation_fraction=0.15,
            seed=self.seed,
        )
        
        return ExperimentResult(
            experiment="experiment_d_combined",
            config=config,
            detection_results=combined,
            metrics=metrics,
            anomaly_type_metrics=anomaly_type_metrics,
        )
    
    def run_experiment_e_threshold_study(self) -> ExperimentResult:
        """
        Experiment E: Threshold study.
        
        Compares detection performance across thresholds 2, 3, 4
        using all anomaly types combined.
        """
        print("  Running Experiment E: Threshold study...")
        
        # Use combined anomalies for comprehensive threshold comparison
        injector = AnomalyInjector(seed=self.seed)
        observations = injector.inject_anomalies(self.normal_df, "combined")
        
        all_results = self.detector.detect_all_thresholds(observations)
        
        combined = []
        for threshold, results in all_results.items():
            combined.extend(results)
        
        results_df = pd.DataFrame(combined)
        
        metrics = self._compute_metrics(results_df)
        anomaly_type_metrics = self._compute_anomaly_type_metrics(results_df)
        
        config = ExperimentConfig(
            name="experiment_e_threshold_study",
            anomaly_type="combined",
            threshold=DEFAULT_THRESHOLD,
            anomaly_device_fraction=0.4,
            anomaly_observation_fraction=0.15,
            seed=self.seed,
        )
        
        return ExperimentResult(
            experiment="experiment_e_threshold_study",
            config=config,
            detection_results=combined,
            metrics=metrics,
            anomaly_type_metrics=anomaly_type_metrics,
        )
    
    def _compute_metrics(self, results_df: pd.DataFrame) -> dict:
        """Compute overall detection metrics per threshold."""
        metrics_list = []
        
        for threshold in DETECTION_THRESHOLDS:
            threshold_df = results_df[results_df["threshold"] == threshold]
            
            if len(threshold_df) == 0:
                continue
            
            y_true = threshold_df["ground_truth"].astype(bool)
            y_pred = threshold_df["predicted_label"].astype(bool)
            
            tp = int((y_true & y_pred).sum())
            tn = int((~y_true & ~y_pred).sum())
            fp = int((~y_true & y_pred).sum())
            fn = int((y_true & ~y_pred).sum())
            
            total = len(threshold_df)
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            metrics_list.append({
                "threshold": threshold,
                "total_observations": total,
                "anomaly_count": int(y_true.sum()),
                "normal_count": int((~y_true).sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": round(accuracy, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1, 6),
                "false_positive_rate": round(fpr, 6),
            })
        
        return metrics_list
    
    def _compute_anomaly_type_metrics(self, results_df: pd.DataFrame) -> list[dict]:
        """Compute metrics broken down by anomaly type."""
        type_metrics = []
        
        anomaly_types = ["traffic_spike", "destination_expansion", "combined"]
        
        for anomaly_type in anomaly_types:
            # Get observations of this anomaly type + normal observations for same devices
            anomaly_df = results_df[results_df["anomaly_type"] == anomaly_type]
            if len(anomaly_df) == 0:
                continue
            
            # Get normal observations for the same devices
            anomaly_devices = anomaly_df["device_id"].unique()
            normal_df = results_df[
                (results_df["device_id"].isin(anomaly_devices)) & 
                (results_df["anomaly_type"] == "normal")
            ]
            
            combined_df = pd.concat([anomaly_df, normal_df])
            
            for threshold in DETECTION_THRESHOLDS:
                threshold_df = combined_df[combined_df["threshold"] == threshold]
                
                if len(threshold_df) == 0:
                    continue
                
                y_true = threshold_df["ground_truth"].astype(bool)
                y_pred = threshold_df["predicted_label"].astype(bool)
                
                tp = int((y_true & y_pred).sum())
                tn = int((~y_true & ~y_pred).sum())
                fp = int((~y_true & y_pred).sum())
                fn = int((y_true & ~y_pred).sum())
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                type_metrics.append({
                    "anomaly_type": anomaly_type,
                    "threshold": threshold,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1_score": round(f1, 6),
                })
        
        return type_metrics
    
    def run_all_experiments(self) -> dict[str, ExperimentResult]:
        """Run all experiments and return results."""
        experiments = {
            "experiment_a_normal": self.run_experiment_a_normal,
            "experiment_b_traffic_spike": self.run_experiment_b_traffic_spike,
            "experiment_c_destination_expansion": self.run_experiment_c_destination_expansion,
            "experiment_d_combined": self.run_experiment_d_combined,
            "experiment_e_threshold_study": self.run_experiment_e_threshold_study,
        }
        
        results = {}
        for name, func in experiments.items():
            results[name] = func()
        
        return results


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================
def run_experiments(
    output_path: Optional[Path] = None,
    seed: int = MASTER_SEED,
) -> dict[str, ExperimentResult]:
    """
    Run all experiments and save consolidated results.
    
    Args:
        output_path: Path to save consolidated results CSV
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping experiment name -> ExperimentResult
    """
    runner = ExperimentRunner(seed=seed)
    results = runner.run_all_experiments()
    
    # Create consolidated DataFrame for saving
    consolidated_rows = []
    
    for exp_name, exp_result in results.items():
        for det_result in exp_result["detection_results"]:
            row = det_result.copy()
            row["experiment"] = exp_name
            consolidated_rows.append(row)
    
    consolidated_df = pd.DataFrame(consolidated_rows)
    
    if output_path is None:
        output_path = DATA_RESULTS_DIR / EXPERIMENT_RESULTS_CSV
    
    save_csv(consolidated_df, output_path.parent, output_path.name)
    
    # Also save detailed experiment results as JSON
    json_path = output_path.with_suffix(".json")
    json_data = {}
    for exp_name, exp_result in results.items():
        json_data[exp_name] = {
            "experiment": exp_result["experiment"],
            "config": exp_result["config"],
            "metrics": exp_result["metrics"],
            "anomaly_type_metrics": exp_result["anomaly_type_metrics"],
        }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    return results


def load_experiment_results(
    results_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load consolidated experiment results."""
    if results_path is None:
        results_path = DATA_RESULTS_DIR / EXPERIMENT_RESULTS_CSV
    
    return load_csv(results_path.parent, results_path.name)


# ============================================================
# VALIDATION
# ============================================================
def validate_experiment_results(results_df: pd.DataFrame) -> dict:
    """Validate experiment results."""
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    required_cols = [
        "device_id", "timestamp", "anomaly_type", "ground_truth",
        "z_packet_rate", "z_destination_count", "z_packet_size",
        "anomaly_score", "predicted_label", "threshold", "experiment",
    ]
    missing = [c for c in required_cols if c not in results_df.columns]
    if missing:
        validation["valid"] = False
        validation["errors"].append(f"Missing columns: {missing}")
        return validation
    
    # Check all experiments present
    expected_experiments = set(EXPERIMENT_NAMES)
    actual_experiments = set(results_df["experiment"].unique())
    missing_exps = expected_experiments - actual_experiments
    if missing_exps:
        validation["warnings"].append(f"Missing experiments: {missing_exps}")
    
    # Check thresholds
    for exp in EXPERIMENT_NAMES:
        exp_df = results_df[results_df["experiment"] == exp]
        if len(exp_df) == 0:
            continue
        thresholds = set(exp_df["threshold"].unique())
        if thresholds != set(DETECTION_THRESHOLDS):
            validation["warnings"].append(f"Experiment {exp} missing thresholds: {set(DETECTION_THRESHOLDS) - thresholds}")
    
    validation["stats"] = {
        "total_rows": len(results_df),
        "experiments": results_df["experiment"].nunique(),
        "devices": results_df["device_id"].nunique(),
        "thresholds": sorted(results_df["threshold"].unique().tolist()),
    }
    
    return validation


if __name__ == "__main__":
    print("Running all experiments...")
    results = run_experiments()
    
    print("\nExperiment Summary:")
    for name, exp_result in results.items():
        metrics = exp_result["metrics"]
        print(f"\n{name}:")
        for m in metrics:
            print(f"  Threshold {m['threshold']}: TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, "
                  f"Acc={m['accuracy']:.3f}, Prec={m['precision']:.3f}, Rec={m['recall']:.3f}, F1={m['f1_score']:.3f}")