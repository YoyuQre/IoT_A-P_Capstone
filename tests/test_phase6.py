"""
Phase 6 Tests: Experimental Framework
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.config import (
    MASTER_SEED,
    EXPERIMENT_NAMES,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
)
from src.schemas import ExperimentConfig, ExperimentResult
from src.experiments import (
    ExperimentRunner,
    run_experiments,
    load_experiment_results,
    validate_experiment_results,
)


# ============================================================
# EXPERIMENT RUNNER TESTS
# ============================================================
class TestExperimentRunner:
    """Test ExperimentRunner class."""
    
    def test_runner_creation(self):
        """Test runner instantiation."""
        runner = ExperimentRunner(seed=42)
        assert runner.seed == 42
        assert runner.normal_df is not None
        assert runner.fingerprints is not None
        assert runner.detector is not None
    
    def test_experiment_a_normal(self):
        """Test Experiment A: Normal traffic only."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_a_normal()
        
        assert result["experiment"] == "experiment_a_normal"
        assert "config" in result
        assert "detection_results" in result
        assert "metrics" in result
        assert "anomaly_type_metrics" in result
        
        # Should have no ground truth anomalies
        detection_df = pd.DataFrame(result["detection_results"])
        assert detection_df["ground_truth"].sum() == 0
        
        # Should have metrics for all thresholds
        assert len(result["metrics"]) == len(DETECTION_THRESHOLDS)
    
    def test_experiment_b_traffic_spike(self):
        """Test Experiment B: Traffic spike."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_b_traffic_spike()
        
        assert result["experiment"] == "experiment_b_traffic_spike"
        assert result["config"]["anomaly_type"] == "traffic_spike"
        
        detection_df = pd.DataFrame(result["detection_results"])
        assert detection_df["ground_truth"].sum() > 0  # Has anomalies
    
    def test_experiment_c_destination_expansion(self):
        """Test Experiment C: Destination expansion."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_c_destination_expansion()
        
        assert result["experiment"] == "experiment_c_destination_expansion"
        assert result["config"]["anomaly_type"] == "destination_expansion"
    
    def test_experiment_d_combined(self):
        """Test Experiment D: Combined anomalies."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_d_combined()
        
        assert result["experiment"] == "experiment_d_combined"
        assert result["config"]["anomaly_type"] == "combined"
    
    def test_experiment_e_threshold_study(self):
        """Test Experiment E: Threshold study."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_e_threshold_study()
        
        assert result["experiment"] == "experiment_e_threshold_study"
        assert result["config"]["anomaly_type"] == "combined"
    
    def test_run_all_experiments(self):
        """Test running all experiments at once."""
        runner = ExperimentRunner(seed=42)
        results = runner.run_all_experiments()
        
        assert isinstance(results, dict)
        assert set(results.keys()) == set(EXPERIMENT_NAMES)
        
        for exp_name in EXPERIMENT_NAMES:
            assert exp_name in results
            assert results[exp_name]["experiment"] == exp_name
    
    def test_deterministic_experiments(self):
        """Test that experiments are deterministic."""
        runner1 = ExperimentRunner(seed=123)
        results1 = runner1.run_all_experiments()
        
        runner2 = ExperimentRunner(seed=123)
        results2 = runner2.run_all_experiments()
        
        # Compare detection results for each experiment
        for exp_name in EXPERIMENT_NAMES:
            res1 = results1[exp_name]["detection_results"]
            res2 = results2[exp_name]["detection_results"]
            
            scores1 = [r["anomaly_score"] for r in res1]
            scores2 = [r["anomaly_score"] for r in res2]
            assert scores1 == scores2
    
    def test_metrics_structure(self):
        """Test that metrics have correct structure."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_b_traffic_spike()
        
        metrics = result["metrics"]
        assert isinstance(metrics, list)
        assert len(metrics) == len(DETECTION_THRESHOLDS)
        
        for m in metrics:
            assert "threshold" in m
            assert "tp" in m
            assert "tn" in m
            assert "fp" in m
            assert "fn" in m
            assert "accuracy" in m
            assert "precision" in m
            assert "recall" in m
            assert "f1_score" in m
            assert "false_positive_rate" in m
    
    def test_anomaly_type_metrics_structure(self):
        """Test that anomaly type metrics have correct structure."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_b_traffic_spike()
        
        type_metrics = result["anomaly_type_metrics"]
        assert isinstance(type_metrics, list)
        
        for m in type_metrics:
            assert "anomaly_type" in m
            assert "threshold" in m
            assert "tp" in m
            assert "tn" in m
            assert "fp" in m
            assert "fn" in m
            assert "precision" in m
            assert "recall" in m
            assert "f1_score" in m


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_run_experiments(self, tmp_path):
        """Test run_experiments convenience function."""
        output_path = tmp_path / "experiment_results.csv"
        
        results = run_experiments(output_path=output_path, seed=42)
        
        assert isinstance(results, dict)
        assert set(results.keys()) == set(EXPERIMENT_NAMES)
        assert output_path.exists()
        
        # Verify JSON also created
        json_path = output_path.with_suffix(".json")
        assert json_path.exists()
        
        # Load and verify
        loaded = load_experiment_results(output_path)
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) > 0
        assert "experiment" in loaded.columns
    
    def test_load_experiment_results(self, tmp_path):
        """Test loading experiment results."""
        output_path = tmp_path / "experiment_results.csv"
        run_experiments(output_path=output_path, seed=42)
        
        loaded = load_experiment_results(output_path)
        assert isinstance(loaded, pd.DataFrame)
    
    def test_validate_experiment_results_valid(self):
        """Test validation passes for valid results."""
        runner = ExperimentRunner(seed=42)
        results = runner.run_all_experiments()
        
        # Create consolidated DataFrame
        consolidated_rows = []
        for exp_name, exp_result in results.items():
            for det_result in exp_result["detection_results"]:
                row = det_result.copy()
                row["experiment"] = exp_name
                consolidated_rows.append(row)
        
        consolidated_df = pd.DataFrame(consolidated_rows)
        
        validation = validate_experiment_results(consolidated_df)
        
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_validate_experiment_results_invalid(self):
        """Test validation fails for invalid results."""
        invalid_df = pd.DataFrame({
            "device_id": ["test_01"],
            "timestamp": ["2024-01-01T00:00:00"],
            "anomaly_type": ["normal"],
            "ground_truth": [False],
            "z_packet_rate": [0.0],
            "z_destination_count": [0.0],
            "z_packet_size": [0.0],
            "anomaly_score": [0.0],
            "predicted_label": [False],
            "threshold": [3.0],
            "experiment": ["invalid_experiment"],  # Invalid experiment name
        })
        
        validation = validate_experiment_results(invalid_df)
        
        # Should warn about missing experiments
        assert any("Missing experiments" in w for w in validation["warnings"])


# ============================================================
# INTEGRATION TESTS
# ============================================================
class TestExperimentIntegration:
    """Integration tests for experimental framework."""
    
    def test_experiment_a_false_positive_rate(self):
        """Test Experiment A measures false positive rate."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_a_normal()
        
        # Since all observations are normal, any detection is a false positive
        for m in result["metrics"]:
            fp = m["fp"]
            tn = m["tn"]
            fpr = m["false_positive_rate"]
            
            # FPR should match fp / (fp + tn)
            expected_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            assert abs(fpr - expected_fpr) < 1e-6
    
    def test_experiment_b_detection_performance(self):
        """Test Experiment B has meaningful detection performance."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_b_traffic_spike()
        
        # Should detect some anomalies
        detection_df = pd.DataFrame(result["detection_results"])
        
        # At default threshold (3.0), should have some TP
        threshold_3 = detection_df[detection_df["threshold"] == 3.0]
        tp = int((threshold_3["ground_truth"] & threshold_3["predicted_label"]).sum())
        
        # Should detect at least some traffic spikes
        assert tp > 0
    
    def test_experiment_e_threshold_comparison(self):
        """Test Experiment E provides threshold comparison."""
        runner = ExperimentRunner(seed=42)
        result = runner.run_experiment_e_threshold_study()
        
        # Should have metrics for all thresholds
        assert len(result["metrics"]) == len(DETECTION_THRESHOLDS)
        
        # Lower threshold should have higher recall (more detections)
        recall_by_threshold = {m["threshold"]: m["recall"] for m in result["metrics"]}
        
        # Threshold 2 should have >= recall of threshold 3 >= threshold 4
        assert recall_by_threshold[2.0] >= recall_by_threshold[3.0]
        assert recall_by_threshold[3.0] >= recall_by_threshold[4.0]
        
        # Lower threshold should have higher FPR
        fpr_by_threshold = {m["threshold"]: m["false_positive_rate"] for m in result["metrics"]}
        assert fpr_by_threshold[2.0] >= fpr_by_threshold[3.0]
        assert fpr_by_threshold[3.0] >= fpr_by_threshold[4.0]
    
    def test_all_experiments_have_results(self):
        """Test all 5 experiments produce results."""
        runner = ExperimentRunner(seed=42)
        results = runner.run_all_experiments()
        
        for exp_name in EXPERIMENT_NAMES:
            assert exp_name in results
            assert len(results[exp_name]["detection_results"]) > 0
            assert len(results[exp_name]["metrics"]) > 0
    
    def test_consolidated_results_format(self):
        """Test consolidated results have correct format."""
        runner = ExperimentRunner(seed=42)
        results = runner.run_all_experiments()
        
        # Build consolidated
        consolidated_rows = []
        for exp_name, exp_result in results.items():
            for det_result in exp_result["detection_results"]:
                row = det_result.copy()
                row["experiment"] = exp_name
                consolidated_rows.append(row)
        
        df = pd.DataFrame(consolidated_rows)
        
        assert "experiment" in df.columns
        assert df["experiment"].nunique() == 5
        
        # Each experiment should have 3 thresholds
        for exp in EXPERIMENT_NAMES:
            exp_df = df[df["experiment"] == exp]
            assert set(exp_df["threshold"].unique()) == set(DETECTION_THRESHOLDS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])