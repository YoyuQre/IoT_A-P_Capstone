"""
Phase 3 Tests: Behavioural Fingerprinting
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
    DEVICE_CATEGORIES,
    NUM_DEVICES,
    EPSILON,
)
from src.schemas import DeviceFingerprint, validate_fingerprint
from src.simulator import generate_normal_traffic
from src.fingerprint import (
    FingerprintCalculator,
    compute_device_fingerprints,
    load_device_fingerprints,
    get_device_fingerprint_dict,
    validate_fingerprints,
)


# ============================================================
# FINGERPRINT CALCULATOR TESTS
# ============================================================
class TestFingerprintCalculator:
    """Test FingerprintCalculator class."""
    
    def test_calculator_creation(self):
        """Test calculator instantiation."""
        calc = FingerprintCalculator()
        assert calc.fingerprint_features == FINGERPRINT_FEATURES
        assert calc.epsilon == EPSILON
    
    def test_calculator_custom_features(self):
        """Test calculator with custom features."""
        calc = FingerprintCalculator(fingerprint_features=["packet_rate", "packet_size"])
        assert calc.fingerprint_features == ["packet_rate", "packet_size"]
    
    def test_calculate_fingerprints(self):
        """Test fingerprint calculation from normal traffic."""
        # Generate normal traffic
        df = generate_normal_traffic(seed=42)
        
        calc = FingerprintCalculator()
        fingerprints = calc.calculate_fingerprints(df)
        
        # Check structure
        assert isinstance(fingerprints, pd.DataFrame)
        assert len(fingerprints) == NUM_DEVICES * len(FINGERPRINT_FEATURES)
        
        # Check columns
        expected_cols = ["device_id", "feature", "mean", "std"]
        assert list(fingerprints.columns) == expected_cols
        
        # Check all devices present
        assert fingerprints["device_id"].nunique() == NUM_DEVICES
        
        # Check all features present for each device
        for device_id in fingerprints["device_id"].unique():
            device_fps = fingerprints[fingerprints["device_id"] == device_id]
            assert len(device_fps) == len(FINGERPRINT_FEATURES)
            assert set(device_fps["feature"].values) == set(FINGERPRINT_FEATURES)
    
    def test_fingerprints_exclude_anomalies(self):
        """Test that anomalies are excluded from fingerprint calculation."""
        df = generate_normal_traffic(seed=42)
        
        # Add is_anomaly column to normal traffic (all False)
        df = df.copy()
        df["is_anomaly"] = False
        df["anomaly_type"] = "normal"
        
        # Add some fake anomalous observations
        anomaly_rows = df.head(10).copy()
        anomaly_rows["is_anomaly"] = True
        anomaly_rows["anomaly_type"] = "traffic_spike"
        anomaly_rows["packet_rate"] = anomaly_rows["packet_rate"] * 100  # Huge spike
        
        df_with_anomalies = pd.concat([df, anomaly_rows], ignore_index=True)
        
        # Calculate fingerprints - should use only normal traffic
        calc = FingerprintCalculator()
        fingerprints = calc.calculate_fingerprints(df_with_anomalies)
        
        # The means should be close to the normal-only means (not affected by anomalies)
        normal_fps = calc.calculate_fingerprints(df)
        
        # Compare means - they should be very similar
        for feature in FINGERPRINT_FEATURES:
            normal_mean = normal_fps[normal_fps["feature"] == feature]["mean"].mean()
            with_anomaly_mean = fingerprints[fingerprints["feature"] == feature]["mean"].mean()
            # Should be very close (within 1%)
            assert abs(normal_mean - with_anomaly_mean) / normal_mean < 0.01
    
    def test_zero_variance_handling(self):
        """Test that zero/near-zero variance is handled with epsilon."""
        # Create data with zero variance for a feature
        df = pd.DataFrame({
            "device_id": ["test_01"] * 10,
            "device_type": ["Smart Bulb"] * 10,
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min"),
            "packet_rate": [1.0] * 10,  # Constant - zero variance
            "packet_size": [100.0] * 10,
            "destination_count": [1] * 10,
            "protocol": ["MQTT"] * 10,
            "connection_duration": [1.0] * 10,
            "is_anomaly": [False] * 10,
            "anomaly_type": ["normal"] * 10,
        })
        
        calc = FingerprintCalculator()
        fingerprints = calc.calculate_fingerprints(df)
        
        # Check that std is at least epsilon
        for _, row in fingerprints.iterrows():
            assert row["std"] >= EPSILON
    
    def test_deterministic_fingerprints(self):
        """Test that same input produces identical fingerprints."""
        df = generate_normal_traffic(seed=42)
        
        calc1 = FingerprintCalculator()
        fps1 = calc1.calculate_fingerprints(df)
        
        calc2 = FingerprintCalculator()
        fps2 = calc2.calculate_fingerprints(df)
        
        pd.testing.assert_frame_equal(fps1, fps2)


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_compute_device_fingerprints(self, tmp_path):
        """Test the convenience function."""
        # First generate normal traffic
        df = generate_normal_traffic(seed=42)
        traffic_path = tmp_path / "normal_traffic.csv"
        df.to_csv(traffic_path, index=False)
        
        output_path = tmp_path / "fingerprints.csv"
        
        fps = compute_device_fingerprints(
            normal_traffic_path=traffic_path,
            output_path=output_path,
        )
        
        assert isinstance(fps, pd.DataFrame)
        assert len(fps) == NUM_DEVICES * len(FINGERPRINT_FEATURES)
        assert output_path.exists()
        
        # Verify saved file matches
        loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(fps, loaded)
    
    def test_load_device_fingerprints(self, tmp_path):
        """Test loading fingerprints from CSV."""
        df = generate_normal_traffic(seed=42)
        traffic_path = tmp_path / "normal_traffic.csv"
        df.to_csv(traffic_path, index=False)
        
        output_path = tmp_path / "fingerprints.csv"
        fps = compute_device_fingerprints(
            normal_traffic_path=traffic_path,
            output_path=output_path,
        )
        
        loaded = load_device_fingerprints(output_path)
        pd.testing.assert_frame_equal(fps, loaded)
    
    def test_get_device_fingerprint_dict(self):
        """Test conversion to nested dict."""
        df = generate_normal_traffic(seed=42)
        calc = FingerprintCalculator()
        fps = calc.calculate_fingerprints(df)
        
        fp_dict = get_device_fingerprint_dict(fps)
        
        assert isinstance(fp_dict, dict)
        assert len(fp_dict) == NUM_DEVICES
        
        for device_id, features in fp_dict.items():
            assert isinstance(features, dict)
            assert set(features.keys()) == set(FINGERPRINT_FEATURES)
            
            for feature, stats in features.items():
                assert "mean" in stats
                assert "std" in stats
                assert isinstance(stats["mean"], float)
                assert isinstance(stats["std"], float)
                assert stats["std"] >= EPSILON


# ============================================================
# VALIDATION TESTS
# ============================================================
class TestValidateFingerprints:
    """Test fingerprint validation."""
    
    def test_valid_fingerprints_pass(self):
        """Test that valid fingerprints pass validation."""
        df = generate_normal_traffic(seed=42)
        calc = FingerprintCalculator()
        fps = calc.calculate_fingerprints(df)
        
        results = validate_fingerprints(fps)
        
        assert results["valid"] is True
        assert len(results["errors"]) == 0
    
    def test_invalid_fingerprints_fail(self):
        """Test that invalid fingerprints fail validation."""
        # Create invalid fingerprints (negative std)
        invalid_fps = pd.DataFrame({
            "device_id": ["test_01"],
            "feature": ["packet_rate"],
            "mean": [1.0],
            "std": [-0.1],  # Invalid
        })
        
        results = validate_fingerprints(invalid_fps)
        
        assert results["valid"] is False
        assert len(results["errors"]) > 0
    
    def test_missing_columns_fail(self):
        """Test that missing columns fail validation."""
        invalid_fps = pd.DataFrame({"device_id": ["test_01"]})
        
        results = validate_fingerprints(invalid_fps)
        
        assert results["valid"] is False
        assert any("Missing columns" in e for e in results["errors"])


# ============================================================
# INTEGRATION TESTS
# ============================================================
class TestFingerprintIntegration:
    """Integration tests for fingerprinting pipeline."""
    
    def test_fingerprints_match_source_statistics(self):
        """Test that computed fingerprints match source data statistics."""
        df = generate_normal_traffic(seed=42)
        # Normal traffic doesn't have is_anomaly column - all observations are normal
        normal_df = df.copy()
        
        calc = FingerprintCalculator()
        fps = calc.calculate_fingerprints(normal_df)
        
        # For each device and feature, compare with direct pandas calculation
        for device_id in normal_df["device_id"].unique():
            device_df = normal_df[normal_df["device_id"] == device_id]
            
            for feature in FINGERPRINT_FEATURES:
                expected_mean = float(np.mean(device_df[feature]))
                expected_std = float(np.std(device_df[feature], ddof=1))
                if expected_std < EPSILON:
                    expected_std = EPSILON
                
                fp_row = fps[(fps["device_id"] == device_id) & (fps["feature"] == feature)]
                assert len(fp_row) == 1
                
                actual_mean = fp_row["mean"].values[0]
                actual_std = fp_row["std"].values[0]
                
                assert abs(actual_mean - expected_mean) < 1e-6
                assert abs(actual_std - expected_std) < 1e-6
    
    def test_different_devices_have_different_fingerprints(self):
        """Test that different devices have different fingerprints."""
        df = generate_normal_traffic(seed=42)
        calc = FingerprintCalculator()
        fps = calc.calculate_fingerprints(df)
        
        # Compare two devices of same category
        bulb_devices = [d for d in fps["device_id"].unique() if d.startswith("bulb_")]
        assert len(bulb_devices) >= 2
        
        device1_fps = fps[fps["device_id"] == bulb_devices[0]]
        device2_fps = fps[fps["device_id"] == bulb_devices[1]]
        
        # At least one feature should have different mean
        means_differ = False
        for feature in FINGERPRINT_FEATURES:
            m1 = device1_fps[device1_fps["feature"] == feature]["mean"].values[0]
            m2 = device2_fps[device2_fps["feature"] == feature]["mean"].values[0]
            if abs(m1 - m2) > 1e-6:
                means_differ = True
                break
        
        assert means_differ, "Devices of same category should have different fingerprints"
    
    def test_fingerprint_features_coverage(self):
        """Test that all fingerprint features are covered."""
        df = generate_normal_traffic(seed=42)
        calc = FingerprintCalculator()
        fps = calc.calculate_fingerprints(df)
        
        features_present = set(fps["feature"].unique())
        assert features_present == set(FINGERPRINT_FEATURES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])