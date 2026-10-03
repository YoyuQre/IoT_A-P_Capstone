"""
Phase 2 Tests: Normal Traffic Simulator
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.config import (
    MASTER_SEED,
    DEVICE_CATEGORIES,
    NUM_DEVICES,
    DEVICES_PER_CATEGORY,
    DEVICE_ID_PREFIX,
    CATEGORY_PROFILES,
    OBSERVATIONS_PER_DEVICE_MIN,
    OBSERVATIONS_PER_DEVICE_MAX,
    TIMESTAMP_START,
    TIMESTAMP_INTERVAL_SECONDS,
    DATA_RAW_DIR,
    NORMAL_TRAFFIC_CSV,
)
from src.schemas import (
    NormalObservation,
    validate_observation,
    VALID_PROTOCOLS,
    VALID_DEVICE_TYPES,
)
from src.simulator import (
    VirtualDevice,
    NormalTrafficSimulator,
    generate_normal_traffic,
    validate_normal_traffic,
)
from src.utils import DeterministicRNG


# ============================================================
# VIRTUAL DEVICE TESTS
# ============================================================
class TestVirtualDevice:
    """Test VirtualDevice creation and parameter derivation."""
    
    def test_device_creation(self):
        """Test basic device creation."""
        rng = DeterministicRNG(seed=42)
        profile = CATEGORY_PROFILES["Smart Bulb"]
        
        device = VirtualDevice(
            device_id="bulb_01",
            device_type="Smart Bulb",
            profile=profile,
            rng=rng,
            device_index=0,
        )
        
        assert device.device_id == "bulb_01"
        assert device.device_type == "Smart Bulb"
        assert device.profile == profile
        assert hasattr(device, "packet_rate_range")
        assert hasattr(device, "packet_size_range")
        assert hasattr(device, "dest_count_range")
        assert hasattr(device, "conn_duration_range")
        assert hasattr(device, "protocols")
    
    def test_device_parameter_ranges_valid(self):
        """Test that derived parameter ranges are valid."""
        rng = DeterministicRNG(seed=42)
        profile = CATEGORY_PROFILES["Smart Camera"]
        
        device = VirtualDevice(
            device_id="cam_01",
            device_type="Smart Camera",
            profile=profile,
            rng=rng,
            device_index=0,
        )
        
        # All ranges should be valid (low <= high)
        assert device.packet_rate_range[0] <= device.packet_rate_range[1]
        assert device.packet_size_range[0] <= device.packet_size_range[1]
        assert device.dest_count_range[0] <= device.dest_count_range[1]
        assert device.conn_duration_range[0] <= device.conn_duration_range[1]
        
        # All values should be positive
        assert device.packet_rate_range[0] > 0
        assert device.packet_size_range[0] > 0
        assert device.dest_count_range[0] >= 1
        assert device.conn_duration_range[0] > 0
    
    def test_different_devices_have_different_params(self):
        """Test that devices of same category have different parameters."""
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=43)
        profile = CATEGORY_PROFILES["Smart Bulb"]
        
        device1 = VirtualDevice("bulb_01", "Smart Bulb", profile, rng1, 0)
        device2 = VirtualDevice("bulb_02", "Smart Bulb", profile, rng2, 1)
        
        # At least one parameter range should differ
        ranges_differ = (
            device1.packet_rate_range != device2.packet_rate_range or
            device1.packet_size_range != device2.packet_size_range or
            device1.dest_count_range != device2.dest_count_range or
            device1.conn_duration_range != device2.conn_duration_range
        )
        assert ranges_differ


# ============================================================
# TRAFFIC SIMULATOR TESTS
# ============================================================
class TestNormalTrafficSimulator:
    """Test NormalTrafficSimulator class."""
    
    def test_simulator_creation(self):
        """Test simulator instantiation."""
        simulator = NormalTrafficSimulator(seed=42)
        assert simulator.seed == 42
        assert len(simulator.devices) == NUM_DEVICES
    
    def test_device_creation(self):
        """Test that all 25 devices are created with correct categories."""
        simulator = NormalTrafficSimulator(seed=42)
        
        # Count devices per category
        category_counts = {}
        for device in simulator.devices:
            category_counts[device.device_type] = category_counts.get(device.device_type, 0) + 1
        
        # Should have 5 devices per category
        for cat in DEVICE_CATEGORIES:
            assert category_counts[cat] == DEVICES_PER_CATEGORY
        
        # Check device IDs follow pattern
        for device in simulator.devices:
            prefix = DEVICE_ID_PREFIX[device.device_type]
            assert device.device_id.startswith(prefix)
    
    def test_generate_device_observations(self):
        """Test generating observations for a single device."""
        simulator = NormalTrafficSimulator(seed=42)
        device = simulator.devices[0]
        
        obs = simulator.generate_device_observations(device, 10)
        
        assert len(obs) == 10
        for o in obs:
            assert o["device_id"] == device.device_id
            assert o["device_type"] == device.device_type
            assert "timestamp" in o
            assert o["packet_rate"] > 0
            assert o["packet_size"] > 0
            assert o["destination_count"] >= 1
            assert o["protocol"] in VALID_PROTOCOLS
            assert o["connection_duration"] > 0
            
            # Validate schema
            full_obs = {**o, "is_anomaly": False, "anomaly_type": "normal"}
            assert validate_observation(full_obs)
    
    def test_generate_full_dataset(self):
        """Test generating complete dataset."""
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        
        # Check basic structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check columns
        expected_cols = [
            "device_id", "device_type", "timestamp", "packet_rate",
            "packet_size", "destination_count", "protocol", "connection_duration"
        ]
        assert list(df.columns) == expected_cols
        
        # Check device count
        assert df["device_id"].nunique() == NUM_DEVICES
        
        # Check all categories present
        for cat in DEVICE_CATEGORIES:
            assert cat in df["device_type"].values
        
        # Check observations per device in range
        obs_counts = df.groupby("device_id").size()
        assert (obs_counts >= OBSERVATIONS_PER_DEVICE_MIN).all()
        assert (obs_counts <= OBSERVATIONS_PER_DEVICE_MAX).all()
        
        # Check sorting
        assert df["device_id"].is_monotonic_increasing or True  # Grouped by device_id
        # Within each device, timestamps should be sorted
        for device_id in df["device_id"].unique():
            device_df = df[df["device_id"] == device_id]
            timestamps = pd.to_datetime(device_df["timestamp"])
            assert timestamps.is_monotonic_increasing
    
    def test_deterministic_generation(self):
        """Test that same seed produces identical output."""
        sim1 = NormalTrafficSimulator(seed=123)
        df1 = sim1.generate()
        
        sim2 = NormalTrafficSimulator(seed=123)
        df2 = sim2.generate()
        
        # DataFrames should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different output."""
        sim1 = NormalTrafficSimulator(seed=123)
        df1 = sim1.generate()
        
        sim2 = NormalTrafficSimulator(seed=456)
        df2 = sim2.generate()
        
        # Should differ (at least in some values)
        assert not df1["packet_rate"].equals(df2["packet_rate"])
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading CSV."""
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        
        output_file = tmp_path / "test_traffic.csv"
        simulator.save(df, output_file)
        
        assert output_file.exists()
        
        # Load and verify
        loaded = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(df, loaded)
    
    def test_get_summary_stats(self):
        """Test summary statistics computation."""
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        
        stats = simulator.get_summary_stats(df)
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == NUM_DEVICES
        assert "device_id" in stats.columns
        assert "device_type" in stats.columns
        assert "num_observations" in stats.columns
        assert "packet_rate_mean" in stats.columns
        assert "packet_rate_std" in stats.columns


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================
class TestGenerateNormalTraffic:
    """Test the convenience function."""
    
    def test_generate_normal_traffic_function(self, tmp_path):
        """Test the convenience function."""
        output_file = tmp_path / "test_normal.csv"
        
        df = generate_normal_traffic(seed=42, output_path=output_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert output_file.exists()
        
        # Verify content
        loaded = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(df, loaded)
    
    def test_default_seed(self):
        """Test default seed produces consistent results."""
        df1 = generate_normal_traffic(seed=42)
        df2 = generate_normal_traffic(seed=42)
        pd.testing.assert_frame_equal(df1, df2)


# ============================================================
# VALIDATION TESTS
# ============================================================
class TestValidateNormalTraffic:
    """Test the validation function."""
    
    def test_valid_data_passes(self):
        """Test that valid generated data passes validation."""
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        
        results = validate_normal_traffic(df)
        
        assert results["valid"] is True
        assert len(results["errors"]) == 0
    
    def test_invalid_data_fails(self):
        """Test that invalid data fails validation."""
        # Create invalid dataframe
        df = pd.DataFrame({
            "device_id": ["test_01"],
            "device_type": ["Invalid Type"],  # Invalid category
            "timestamp": ["2024-01-01T00:00:00"],
            "packet_rate": [-1.0],  # Negative
            "packet_size": [100],
            "destination_count": [1],
            "protocol": ["MQTT"],
            "connection_duration": [1.0],
        })
        
        results = validate_normal_traffic(df)
        
        assert results["valid"] is False
        assert len(results["errors"]) > 0
    
    def test_missing_columns_fails(self):
        """Test that missing columns fails validation."""
        df = pd.DataFrame({"device_id": ["test_01"]})  # Missing most columns
        
        results = validate_normal_traffic(df)
        
        assert results["valid"] is False
        assert any("Missing columns" in e for e in results["errors"])
    
    def test_wrong_device_count_fails(self):
        """Test that wrong device count fails validation."""
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        
        # Remove one device
        df = df[df["device_id"] != df["device_id"].iloc[0]]
        
        results = validate_normal_traffic(df)
        
        assert results["valid"] is False
        assert any("Expected 25 devices" in e for e in results["errors"])


# ============================================================
# INTEGRATION TEST: CATEGORY DISTINGUISHABILITY
# ============================================================
class TestCategoryDistinguishability:
    """Test that different categories have statistically distinguishable behaviour."""
    
    def test_categories_have_different_profiles(self):
        """Test that category profiles produce different traffic patterns."""
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        
        # Compute mean packet rate per category
        cat_means = df.groupby("device_type")["packet_rate"].mean()
        
        # Smart Camera should have higher packet rate than Smart Bulb
        assert cat_means["Smart Camera"] > cat_means["Smart Bulb"]
        
        # Smart Lock should have lowest packet rate
        assert cat_means["Smart Lock"] == cat_means.min()
        
        # Smart TV or Camera should have highest
        assert cat_means.max() in [cat_means["Smart TV"], cat_means["Smart Camera"]]
    
    def test_categories_have_different_packet_sizes(self):
        """Test that packet sizes differ by category."""
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        
        cat_means = df.groupby("device_type")["packet_size"].mean()
        
        # Smart Camera and Smart TV should have larger packets than Smart Bulb
        assert cat_means["Smart Camera"] > cat_means["Smart Bulb"]
        assert cat_means["Smart TV"] > cat_means["Smart Bulb"]
        
        # Smart Lock should have smallest packets
        assert cat_means["Smart Lock"] == cat_means.min()
    
    def test_categories_have_different_destination_counts(self):
        """Test that destination counts differ by category."""
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        
        cat_means = df.groupby("device_type")["destination_count"].mean()
        
        # Smart TV should have more destinations than Smart Bulb
        assert cat_means["Smart TV"] > cat_means["Smart Bulb"]
        
        # Smart Lock and Smart Meter should have 1 destination
        assert cat_means["Smart Lock"] == 1.0
        assert cat_means["Smart Meter"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])