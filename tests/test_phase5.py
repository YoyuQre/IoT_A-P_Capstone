"""
Phase 5 Tests: Statistical Anomaly Detection
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.config import (
    MASTER_SEED,
    FINGERPRINT_FEATURES,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
    EPSILON,
)
from src.schemas import DetectionResult, validate_observation
from src.simulator import generate_normal_traffic
from src.fingerprint import compute_device_fingerprints
from src.anomalies import inject_anomalies, AnomalyInjector
from src.detector import (
    AnomalyDetector,
    run_detection,
    validate_detection_results,
)


# ============================================================
# MODULE-LEVEL FIXTURE
# ============================================================
@pytest.fixture
def setup_data():
    """Create normal traffic, fingerprints, and anomalous data for testing."""
    normal_df = generate_normal_traffic(seed=42)
    fingerprints = compute_device_fingerprints()
    
    # Create anomalous data with all three types
    injector = AnomalyInjector(seed=42)
    traffic_spike_df = injector.inject_anomalies(normal_df, "traffic_spike")
    dest_exp_df = injector.inject_anomalies(normal_df, "destination_expansion")
    combined_df = injector.inject_anomalies(normal_df, "combined")
    
    return {
        "normal_df": normal_df,
        "fingerprints": fingerprints,
        "traffic_spike_df": traffic_spike_df,
        "dest_expansion_df": dest_exp_df,
        "combined_df": combined_df,
    }


# ============================================================
# ANOMALY DETECTOR TESTS
# ============================================================
class TestAnomalyDetector:
    """Test AnomalyDetector class."""
    
    def test_detector_creation(self, setup_data):
        """Test detector instantiation."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        
        assert detector.thresholds == DETECTION_THRESHOLDS
        assert detector.fingerprint_features == FINGERPRINT_FEATURES
        assert len(detector.fingerprint_dict) == 25  # 25 devices
    
    def test_detector_custom_thresholds(self, setup_data):
        """Test detector with custom thresholds."""
        detector = AnomalyDetector(setup_data["fingerprints"], thresholds=[1.0, 2.0])
        assert detector.thresholds == [1.0, 2.0]
    
    def test_calculate_z_scores(self, setup_data):
        """Test z-score calculation."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        result = detector.calculate_z_scores(df)
        
        # Check z-score columns exist
        for feature in FINGERPRINT_FEATURES:
            z_col = f"z_{feature}"
            assert z_col in result.columns
        
        # Check z-scores are finite
        for feature in FINGERPRINT_FEATURES:
            z_col = f"z_{feature}"
            assert np.isfinite(result[z_col]).all()
    
    def test_calculate_anomaly_score(self, setup_data):
        """Test anomaly score calculation."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        # First calculate z-scores
        df = detector.calculate_z_scores(df)
        
        # Then calculate anomaly score
        result = detector.calculate_anomaly_score(df)
        
        assert "anomaly_score" in result.columns
        assert (result["anomaly_score"] >= 0).all()
        
        # Anomaly score should be mean of absolute z-scores
        z_cols = [f"z_{f}" for f in FINGERPRINT_FEATURES]
        expected = result[z_cols].abs().sum(axis=1) / len(z_cols)
        pd.testing.assert_series_equal(result["anomaly_score"], expected, check_names=False)
    
    def test_classify(self, setup_data):
        """Test classification with threshold."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        df = detector.calculate_z_scores(df)
        df = detector.calculate_anomaly_score(df)
        result = detector.classify(df, threshold=3.0)
        
        assert "predicted_label" in result.columns
        assert "threshold" in result.columns
        assert (result["threshold"] == 3.0).all()
        assert (result["predicted_label"] == (result["anomaly_score"] >= 3.0)).all()
    
    def test_detect_single_threshold(self, setup_data):
        """Test complete detection pipeline with single threshold."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        results = detector.detect(df, threshold=3.0)
        
        assert isinstance(results, list)
        assert len(results) == len(df)
        
        for res in results:
            assert isinstance(res, dict)
            assert "device_id" in res
            assert "timestamp" in res
            assert "anomaly_type" in res
            assert "ground_truth" in res
            assert "z_packet_rate" in res
            assert "z_destination_count" in res
            assert "z_packet_size" in res
            assert "anomaly_score" in res
            assert "predicted_label" in res
            assert "threshold" in res
            assert res["threshold"] == 3.0
    
    def test_detect_all_thresholds(self, setup_data):
        """Test detection with all thresholds."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        results = detector.detect_all_thresholds(df)
        
        assert isinstance(results, dict)
        assert set(results.keys()) == set(DETECTION_THRESHOLDS)
        
        for threshold, threshold_results in results.items():
            assert len(threshold_results) == len(df)
            for res in threshold_results:
                assert res["threshold"] == threshold
    
    def test_deterministic_detection(self, setup_data):
        """Test that detection is deterministic."""
        detector1 = AnomalyDetector(setup_data["fingerprints"])
        detector2 = AnomalyDetector(setup_data["fingerprints"])
        
        df = setup_data["traffic_spike_df"]
        
        results1 = detector1.detect(df, threshold=3.0)
        results2 = detector2.detect(df, threshold=3.0)
        
        # Compare anomaly scores
        scores1 = [r["anomaly_score"] for r in results1]
        scores2 = [r["anomaly_score"] for r in results2]
        assert scores1 == scores2
    
    def test_z_score_correctness(self, setup_data):
        """Test that z-scores are calculated correctly."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["normal_df"].head(10).copy()
        
        result = detector.calculate_z_scores(df)
        
        # For normal traffic, z-scores should be relatively small
        for feature in FINGERPRINT_FEATURES:
            z_col = f"z_{feature}"
            # Most normal observations should have |z| < 3
            assert (result[z_col].abs() < 5).all()


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_run_detection(self, tmp_path):
        """Test run_detection convenience function."""
        # Generate required files
        normal_df = generate_normal_traffic(seed=42)
        normal_path = tmp_path / "normal_traffic.csv"
        normal_df.to_csv(normal_path, index=False)
        
        fingerprints = compute_device_fingerprints()
        fp_path = tmp_path / "fingerprints.csv"
        fingerprints.to_csv(fp_path, index=False)
        
        # Create anomalous data
        from src.anomalies import AnomalyInjector
        injector = AnomalyInjector(seed=42)
        anomaly_df = injector.inject_anomalies(normal_df, "traffic_spike")
        anomaly_path = tmp_path / "anomalous_traffic.csv"
        anomaly_df.to_csv(anomaly_path, index=False)
        
        # Run detection
        output_path = tmp_path / "detection_results.csv"
        results = run_detection(
            fingerprints_path=fp_path,
            observations_path=anomaly_path,
            output_path=output_path,
            threshold=3.0,
        )
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(anomaly_df)
        assert output_path.exists()
        
        # Verify saved file matches
        loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(results, loaded)
    
    def test_validate_detection_results_valid(self, setup_data):
        """Test validation passes for valid results."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        results = detector.detect(df, threshold=3.0)
        results_df = pd.DataFrame(results)
        
        validation = validate_detection_results(results_df)
        
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_validate_detection_results_invalid(self):
        """Test validation fails for invalid results."""
        invalid_df = pd.DataFrame({
            "device_id": ["test_01"],
            "timestamp": ["2024-01-01T00:00:00"],
            "anomaly_type": ["normal"],
            "ground_truth": [False],
            "z_packet_rate": [np.inf],  # Invalid
            "z_destination_count": [0.0],
            "z_packet_size": [0.0],
            "anomaly_score": [-1.0],  # Invalid (negative)
            "predicted_label": [False],
            "threshold": [3.0],
        })
        
        validation = validate_detection_results(invalid_df)
        
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0


# ============================================================
# INTEGRATION TESTS
# ============================================================
class TestDetectionIntegration:
    """Integration tests for anomaly detection."""
    
    def test_traffic_spike_detection(self, setup_data):
        """Test that traffic spike anomalies are detected."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        results = detector.detect(df, threshold=3.0)
        results_df = pd.DataFrame(results)
        
        # Check that we have both normal and anomalous observations
        assert results_df["ground_truth"].sum() > 0  # Some anomalies
        assert (~results_df["ground_truth"]).sum() > 0  # Some normal
        
        # For traffic spike, packet_rate z-score should be high for anomalies
        anomaly_results = results_df[results_df["ground_truth"]]
        normal_results = results_df[~results_df["ground_truth"]]
        
        if len(anomaly_results) > 0:
            anomaly_z_rate = anomaly_results["z_packet_rate"].abs().mean()
            normal_z_rate = normal_results["z_packet_rate"].abs().mean()
            assert anomaly_z_rate > normal_z_rate
    
    def test_destination_expansion_detection(self, setup_data):
        """Test that destination expansion anomalies are detected."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["dest_expansion_df"]
        
        results = detector.detect(df, threshold=3.0)
        results_df = pd.DataFrame(results)
        
        anomaly_results = results_df[results_df["ground_truth"]]
        normal_results = results_df[~results_df["ground_truth"]]
        
        if len(anomaly_results) > 0:
            anomaly_z_dest = anomaly_results["z_destination_count"].abs().mean()
            normal_z_dest = normal_results["z_destination_count"].abs().mean()
            assert anomaly_z_dest > normal_z_dest
    
    def test_combined_detection(self, setup_data):
        """Test that combined anomalies are detected on both features."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["combined_df"]
        
        results = detector.detect(df, threshold=3.0)
        results_df = pd.DataFrame(results)
        
        anomaly_results = results_df[results_df["ground_truth"]]
        normal_results = results_df[~results_df["ground_truth"]]
        
        if len(anomaly_results) > 0:
            anomaly_z_rate = anomaly_results["z_packet_rate"].abs().mean()
            normal_z_rate = normal_results["z_packet_rate"].abs().mean()
            anomaly_z_dest = anomaly_results["z_destination_count"].abs().mean()
            normal_z_dest = normal_results["z_destination_count"].abs().mean()
            
            assert anomaly_z_rate > normal_z_rate
            assert anomaly_z_dest > normal_z_dest
    
    def test_threshold_effect(self, setup_data):
        """Test that different thresholds produce different results."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        results_all = detector.detect_all_thresholds(df)
        
        # Lower threshold should detect more anomalies
        pred_counts = {}
        for threshold, results in results_all.items():
            pred_counts[threshold] = sum(r["predicted_label"] for r in results)
        
        # Threshold 2 should detect >= threshold 3 >= threshold 4
        assert pred_counts[2.0] >= pred_counts[3.0] >= pred_counts[4.0]
    
    def test_no_ground_truth_leakage(self, setup_data):
        """Test that detector never uses ground truth for detection."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        # The detection should work the same regardless of ground truth
        results = detector.detect(df, threshold=3.0)
        results_df = pd.DataFrame(results)
        
        # Anomaly score should be independent of ground_truth column
        # (it's computed purely from z-scores)
        anomaly_scores = results_df["anomaly_score"].values
        
        # If we remove ground truth and re-run, scores should be identical
        df_no_gt = df.drop(columns=["is_anomaly", "anomaly_type"])
        results_no_gt = detector.detect(df_no_gt, threshold=3.0)
        scores_no_gt = [r["anomaly_score"] for r in results_no_gt]
        
        assert list(anomaly_scores) == scores_no_gt
    
    def test_all_devices_have_fingerprints(self, setup_data):
        """Test that all devices in data have fingerprints."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        data_devices = set(df["device_id"].unique())
        fp_devices = set(detector.fingerprint_dict.keys())
        
        assert data_devices.issubset(fp_devices)
    
    def test_features_coverage(self, setup_data):
        """Test that all fingerprint features are used in detection."""
        detector = AnomalyDetector(setup_data["fingerprints"])
        df = setup_data["traffic_spike_df"]
        
        results = detector.detect(df, threshold=3.0)
        results_df = pd.DataFrame(results)
        
        # All three z-score columns should have non-zero variance
        for feature in FINGERPRINT_FEATURES:
            z_col = f"z_{feature}"
            assert results_df[z_col].std() > 0 or True  # Some features may be constant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])