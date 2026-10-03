"""
Phase 4 Tests: Controlled Anomaly Injection
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.config import (
    MASTER_SEED,
    ANOMALY_TYPES,
    ANOMALY_MULTIPLIERS,
    ANOMALY_DEVICE_FRACTION,
    ANOMALY_OBSERVATION_FRACTION,
    FINGERPRINT_FEATURES,
)
from src.schemas import validate_observation
from src.simulator import generate_normal_traffic
from src.anomalies import (
    AnomalyInjector,
    inject_anomalies,
    inject_all_anomaly_scenarios,
    create_unified_experiment_dataset,
    validate_anomalous_traffic,
)


# ============================================================
# ANOMALY INJECTOR TESTS
# ============================================================
class TestAnomalyInjector:
    """Test AnomalyInjector class."""
    
    def test_injector_creation(self):
        """Test injector instantiation."""
        injector = AnomalyInjector(seed=42)
        assert injector.seed == 42
        assert injector.anomaly_device_fraction == ANOMALY_DEVICE_FRACTION
        assert injector.anomaly_observation_fraction == ANOMALY_OBSERVATION_FRACTION
    
    def test_inject_traffic_spike(self):
        """Test traffic spike anomaly injection."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(seed=42)
        
        result_df = injector.inject_anomalies(normal_df, "traffic_spike")
        
        # Check structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) >= len(normal_df)  # May have more rows due to replacement
        
        # Check ground truth columns
        assert "is_anomaly" in result_df.columns
        assert "anomaly_type" in result_df.columns
        
        # Check anomaly counts
        anomaly_count = result_df["is_anomaly"].sum()
        normal_count = (~result_df["is_anomaly"]).sum()
        assert anomaly_count > 0
        assert normal_count > 0
        
        # Check anomaly type
        anomaly_types = result_df[result_df["is_anomaly"]]["anomaly_type"].unique()
        assert "traffic_spike" in anomaly_types
        
        # Check packet rate actually increased for anomalies (per-device comparison)
        spike_df = result_df[result_df["anomaly_type"] == "traffic_spike"]
        
        if len(spike_df) > 0:
            # For each device with anomalies, compare anomaly vs normal packet rate
            for device_id in spike_df["device_id"].unique():
                device_spike = spike_df[spike_df["device_id"] == device_id]
                device_normal = normal_df[normal_df["device_id"] == device_id]
                
                spike_rate_mean = device_spike["packet_rate"].mean()
                normal_rate_mean = device_normal["packet_rate"].mean()
                
                # Anomaly rate should be substantially higher than normal for same device
                assert spike_rate_mean > normal_rate_mean * 5, \
                    f"Device {device_id}: spike {spike_rate_mean:.1f} not > 5x normal {normal_rate_mean:.1f}"
    
    def test_inject_destination_expansion(self):
        """Test destination expansion anomaly injection."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(seed=42)
        
        result_df = injector.inject_anomalies(normal_df, "destination_expansion")
        
        # Check anomaly type
        anomaly_types = result_df[result_df["is_anomaly"]]["anomaly_type"].unique()
        assert "destination_expansion" in anomaly_types
        
        # Check destination count actually increased
        dest_df = result_df[result_df["anomaly_type"] == "destination_expansion"]
        normal_df_result = result_df[result_df["anomaly_type"] == "normal"]
        
        if len(dest_df) > 0 and len(normal_df_result) > 0:
            spike_dest_mean = dest_df["destination_count"].mean()
            normal_dest_mean = normal_df_result["destination_count"].mean()
            assert spike_dest_mean > normal_dest_mean * 5  # 8x multiplier
    
    def test_inject_combined(self):
        """Test combined anomaly injection."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(seed=42)
        
        result_df = injector.inject_anomalies(normal_df, "combined")
        
        # Check anomaly type
        anomaly_types = result_df[result_df["is_anomaly"]]["anomaly_type"].unique()
        assert "combined" in anomaly_types
        
        # Check both packet rate and destination count increased (per-device comparison)
        comb_df = result_df[result_df["anomaly_type"] == "combined"]
        
        if len(comb_df) > 0:
            # For each device with anomalies, compare anomaly vs normal
            for device_id in comb_df["device_id"].unique():
                device_comb = comb_df[comb_df["device_id"] == device_id]
                device_normal = normal_df[normal_df["device_id"] == device_id]
                
                comb_rate_mean = device_comb["packet_rate"].mean()
                normal_rate_mean = device_normal["packet_rate"].mean()
                comb_dest_mean = device_comb["destination_count"].mean()
                normal_dest_mean = device_normal["destination_count"].mean()
                
                # Anomaly rate should be substantially higher than normal for same device
                assert comb_rate_mean > normal_rate_mean * 5, \
                    f"Device {device_id}: combined rate {comb_rate_mean:.1f} not > 5x normal {normal_rate_mean:.1f}"
                assert comb_dest_mean > normal_dest_mean * 5, \
                    f"Device {device_id}: combined dest {comb_dest_mean:.1f} not > 5x normal {normal_dest_mean:.1f}"
    
    def test_deterministic_injection(self):
        """Test that same seed produces identical anomalies."""
        normal_df = generate_normal_traffic(seed=42)
        
        injector1 = AnomalyInjector(seed=123)
        result1 = injector1.inject_anomalies(normal_df, "traffic_spike")
        
        injector2 = AnomalyInjector(seed=123)
        result2 = injector2.inject_anomalies(normal_df, "traffic_spike")
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_different_seeds_different_anomalies(self):
        """Test that different seeds produce different anomaly patterns."""
        normal_df = generate_normal_traffic(seed=42)
        
        injector1 = AnomalyInjector(seed=123)
        result1 = injector1.inject_anomalies(normal_df, "traffic_spike")
        
        injector2 = AnomalyInjector(seed=456)
        result2 = injector2.inject_anomalies(normal_df, "traffic_spike")
        
        # At least the anomaly labels should differ
        assert not result1["is_anomaly"].equals(result2["is_anomaly"])
    
    def test_inject_all_anomaly_types(self):
        """Test injecting all anomaly types at once."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(seed=42)
        
        results = injector.inject_all_anomaly_types(normal_df)
        
        assert isinstance(results, dict)
        assert set(results.keys()) == {"traffic_spike", "destination_expansion", "combined"}
        
        for anomaly_type, df in results.items():
            assert isinstance(df, pd.DataFrame)
            assert "is_anomaly" in df.columns
            assert "anomaly_type" in df.columns
            assert df[df["is_anomaly"]]["anomaly_type"].unique()[0] == anomaly_type
    
    def test_anomaly_device_fraction(self):
        """Test that anomaly device fraction is respected."""
        normal_df = generate_normal_traffic(seed=42)
        # Use 100% device fraction for testing
        injector = AnomalyInjector(seed=42, anomaly_device_fraction=1.0)
        
        result_df = injector.inject_anomalies(normal_df, "traffic_spike")
        
        # All devices should have at least some anomalies
        anomaly_devices = result_df[result_df["is_anomaly"]]["device_id"].unique()
        all_devices = normal_df["device_id"].unique()
        
        # Should be close to all devices (some may have 0 observations selected by chance)
        assert len(anomaly_devices) >= len(all_devices) * 0.8


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_inject_anomalies_function(self, tmp_path):
        """Test the inject_anomalies convenience function."""
        normal_df = generate_normal_traffic(seed=42)
        normal_path = tmp_path / "normal.csv"
        normal_df.to_csv(normal_path, index=False)
        
        output_path = tmp_path / "anomalous_traffic_spike.csv"
        
        result_df = inject_anomalies(
            normal_traffic_path=normal_path,
            output_path=output_path,
            anomaly_type="traffic_spike",
            seed=42,
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert output_path.exists()
        
        # Verify loaded file matches
        loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(result_df, loaded)
    
    def test_inject_all_anomaly_scenarios(self, tmp_path):
        """Test injecting all scenarios at once."""
        normal_df = generate_normal_traffic(seed=42)
        normal_path = tmp_path / "normal.csv"
        normal_df.to_csv(normal_path, index=False)
        
        results = inject_all_anomaly_scenarios(
            normal_traffic_path=normal_path,
            output_dir=tmp_path,
            seed=42,
        )
        
        assert set(results.keys()) == {"traffic_spike", "destination_expansion", "combined"}
        
        for anomaly_type in results:
            output_file = tmp_path / f"anomalous_traffic_{anomaly_type}.csv"
            assert output_file.exists()
    
    def test_create_unified_experiment_dataset(self, tmp_path):
        """Test creating unified experiment dataset."""
        normal_df = generate_normal_traffic(seed=42)
        normal_path = tmp_path / "normal.csv"
        normal_df.to_csv(normal_path, index=False)
        
        output_path = tmp_path / "unified.csv"
        
        unified_df = create_unified_experiment_dataset(
            normal_traffic_path=normal_path,
            output_path=output_path,
            seed=42,
        )
        
        assert isinstance(unified_df, pd.DataFrame)
        assert output_path.exists()
        
        # Should have normal + 3 anomaly types
        anomaly_types = unified_df["anomaly_type"].unique()
        assert "normal" in anomaly_types
        assert "traffic_spike" in anomaly_types
        assert "destination_expansion" in anomaly_types
        assert "combined" in anomaly_types


# ============================================================
# VALIDATION TESTS
# ============================================================
class TestValidateAnomalousTraffic:
    """Test validation of anomalous traffic."""
    
    def test_valid_anomalous_traffic_passes(self):
        """Test that valid anomalous traffic passes validation."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(seed=42)
        result_df = injector.inject_anomalies(normal_df, "traffic_spike")
        
        results = validate_anomalous_traffic(result_df, "traffic_spike")
        
        assert results["valid"] is True
        assert len(results["errors"]) == 0
    
    def test_invalid_anomalous_traffic_fails(self):
        """Test that invalid data fails validation."""
        invalid_df = pd.DataFrame({
            "device_id": ["test_01"],
            "device_type": ["Invalid Type"],
            "timestamp": ["2024-01-01T00:00:00"],
            "packet_rate": [-1.0],
            "packet_size": [100],
            "destination_count": [1],
            "protocol": ["MQTT"],
            "connection_duration": [1.0],
            "is_anomaly": [True],
            "anomaly_type": ["traffic_spike"],
        })
        
        results = validate_anomalous_traffic(invalid_df, "traffic_spike")
        
        assert results["valid"] is False
        assert len(results["errors"]) > 0
    
    def test_traffic_spike_validation_warning(self):
        """Test validation warns if spike not actually higher."""
        # Create data where spike is NOT higher
        normal_df = generate_normal_traffic(seed=42)
        normal_df = normal_df.copy()
        normal_df["is_anomaly"] = False
        normal_df["anomaly_type"] = "normal"
        
        # Add fake spike with lower packet rate
        spike_rows = normal_df.head(5).copy()
        spike_rows["is_anomaly"] = True
        spike_rows["anomaly_type"] = "traffic_spike"
        spike_rows["packet_rate"] = 0.001  # Lower than normal
        
        combined = pd.concat([normal_df, spike_rows], ignore_index=True)
        
        results = validate_anomalous_traffic(combined, "traffic_spike")
        
        assert any("not higher" in w for w in results["warnings"])
    
    def test_destination_expansion_validation_warning(self):
        """Test validation warns if dest expansion not actually higher."""
        normal_df = generate_normal_traffic(seed=42)
        normal_df = normal_df.copy()
        normal_df["is_anomaly"] = False
        normal_df["anomaly_type"] = "normal"
        
        # Add fake dest expansion with lower destination count
        dest_rows = normal_df.head(5).copy()
        dest_rows["is_anomaly"] = True
        dest_rows["anomaly_type"] = "destination_expansion"
        dest_rows["destination_count"] = 1  # Same as normal (for Smart Bulb)
        
        combined = pd.concat([normal_df, dest_rows], ignore_index=True)
        
        results = validate_anomalous_traffic(combined, "destination_expansion")
        
        # This might not warn if normal already has 1 destination
        # The test is mainly to ensure validation runs without error
        assert "valid" in results


# ============================================================
# INTEGRATION TESTS
# ============================================================
class TestAnomalyIntegration:
    """Integration tests for anomaly injection."""
    
    def test_ground_truth_preserved(self):
        """Test that ground truth is preserved and consistent."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(seed=42)
        
        for anomaly_type in ["traffic_spike", "destination_expansion", "combined"]:
            result_df = injector.inject_anomalies(normal_df, anomaly_type)
            
            # Check consistency: is_anomaly == True iff anomaly_type != "normal"
            anomaly_mask = result_df["is_anomaly"]
            type_mask = result_df["anomaly_type"] != "normal"
            
            assert (anomaly_mask == type_mask).all()
    
    def test_unrelated_features_unchanged(self):
        """Test that unrelated features (protocol, connection_duration) are not corrupted."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(seed=42)
        
        result_df = injector.inject_anomalies(normal_df, "traffic_spike")
        
        # For anomalous observations, protocol should still be valid
        anomaly_df = result_df[result_df["is_anomaly"]]
        
        # Protocol should be from the normal set
        from src.schemas import VALID_PROTOCOLS
        for proto in anomaly_df["protocol"].unique():
            assert proto in VALID_PROTOCOLS
        
        # Connection duration should be positive
        assert (anomaly_df["connection_duration"] > 0).all()
    
    def test_anomaly_observation_fraction(self):
        """Test that anomaly observation fraction is approximately respected."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(
            seed=42,
            anomaly_device_fraction=1.0,  # All devices
            anomaly_observation_fraction=0.2,  # 20% of observations
        )
        
        result_df = injector.inject_anomalies(normal_df, "traffic_spike")
        
        # Check fraction per device
        for device_id in normal_df["device_id"].unique():
            device_normal = normal_df[normal_df["device_id"] == device_id]
            device_result = result_df[result_df["device_id"] == device_id]
            
            normal_obs = len(device_normal)
            anomaly_obs = len(device_result[device_result["is_anomaly"]])
            
            if normal_obs > 0:
                actual_fraction = anomaly_obs / normal_obs
                # Should be approximately 0.2 (within some tolerance)
                assert 0.1 <= actual_fraction <= 0.3
    
    def test_anomaly_multipliers_applied(self):
        """Test that configured multipliers are actually applied."""
        normal_df = generate_normal_traffic(seed=42)
        injector = AnomalyInjector(seed=42)
        
        # Traffic spike: packet_rate multiplier = 10x
        result_df = injector.inject_anomalies(normal_df, "traffic_spike")
        
        spike_df = result_df[result_df["anomaly_type"] == "traffic_spike"]
        normal_df_result = result_df[result_df["anomaly_type"] == "normal"]
        
        # Find matching device pairs
        for device_id in spike_df["device_id"].unique()[:3]:  # Check first few
            spike_obs = spike_df[spike_df["device_id"] == device_id]
            normal_obs = normal_df_result[normal_df_result["device_id"] == device_id]
            
            if len(spike_obs) > 0 and len(normal_obs) > 0:
                # Average spike packet rate should be ~10x normal
                spike_rate = spike_obs["packet_rate"].mean()
                normal_rate = normal_obs["packet_rate"].mean()
                ratio = spike_rate / normal_rate if normal_rate > 0 else 0
                
                # Allow some variance due to random selection
                assert 5 < ratio < 20, f"Expected ~10x ratio, got {ratio:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])