"""
Phase 1 Tests: Configuration, Schemas, Deterministic RNG, and Project Structure
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.config import (
    PROJECT_ROOT,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_RESULTS_DIR,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_REPORTS_DIR,
    MASTER_SEED,
    NUMPY_SEED,
    DEVICE_CATEGORIES,
    NUM_DEVICES,
    DEVICES_PER_CATEGORY,
    CATEGORY_PROFILES,
    FINGERPRINT_FEATURES,
    ANOMALY_TYPES,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
    EPSILON,
)
from src.schemas import (
    Observation,
    NormalObservation,
    DeviceFingerprint,
    validate_observation,
    validate_fingerprint,
    REQUIRED_OBSERVATION_FIELDS,
    VALID_PROTOCOLS,
    VALID_ANOMALY_TYPES,
    VALID_DEVICE_TYPES,
)
from src.utils import (
    DeterministicRNG,
    get_rng,
    set_global_seed,
    safe_divide,
    safe_z_score,
    safe_mean_std,
    generate_device_ids,
    select_protocol,
    validate_dataframe_schema,
    validate_no_nan_inf,
)
from src import (
    MASTER_SEED as EXPORTED_MASTER_SEED,
    DEVICE_CATEGORIES as EXPORTED_CATEGORIES,
)


# ============================================================
# CONFIGURATION TESTS
# ============================================================
class TestConfiguration:
    """Test centralized configuration loading and values."""
    
    def test_master_seed_defined(self):
        assert MASTER_SEED == 42
        assert NUMPY_SEED == 42
        assert EXPORTED_MASTER_SEED == 42
    
    def test_device_categories(self):
        assert len(DEVICE_CATEGORIES) == 5
        assert DEVICE_CATEGORIES == [
            "Smart Bulb",
            "Smart Camera", 
            "Smart Lock",
            "Smart Meter",
            "Smart TV",
        ]
        assert EXPORTED_CATEGORIES == DEVICE_CATEGORIES
    
    def test_device_counts(self):
        assert NUM_DEVICES == 25
        assert DEVICES_PER_CATEGORY == 5
    
    def test_category_profiles_exist(self):
        assert len(CATEGORY_PROFILES) == 5
        for cat in DEVICE_CATEGORIES:
            assert cat in CATEGORY_PROFILES
            profile = CATEGORY_PROFILES[cat]
            assert "packet_rate" in profile
            assert "packet_size" in profile
            assert "destination_count" in profile
            assert "connection_duration" in profile
            assert "protocol" in profile
    
    def test_fingerprint_features(self):
        assert FINGERPRINT_FEATURES == [
            "packet_rate",
            "packet_size", 
            "destination_count",
        ]
    
    def test_anomaly_types(self):
        assert ANOMALY_TYPES == [
            "normal",
            "traffic_spike",
            "destination_expansion", 
            "combined",
        ]
    
    def test_detection_thresholds(self):
        assert DETECTION_THRESHOLDS == [2.0, 3.0, 4.0]
        assert DEFAULT_THRESHOLD == 3.0
    
    def test_epsilon_positive(self):
        assert EPSILON > 0
        assert EPSILON < 1e-6
    
    def test_directories_exist(self):
        for dir_path in [
            DATA_RAW_DIR,
            DATA_PROCESSED_DIR,
            DATA_RESULTS_DIR,
            OUTPUTS_FIGURES_DIR,
            OUTPUTS_REPORTS_DIR,
        ]:
            assert dir_path.exists(), f"Directory missing: {dir_path}"
            assert dir_path.is_dir()


# ============================================================
# SCHEMA TESTS
# ============================================================
class TestSchemas:
    """Test data schema definitions and validation."""
    
    def test_required_observation_fields(self):
        assert len(REQUIRED_OBSERVATION_FIELDS) == 10
        expected = [
            "device_id", "device_type", "timestamp", "packet_rate",
            "packet_size", "destination_count", "protocol",
            "connection_duration", "is_anomaly", "anomaly_type"
        ]
        assert list(REQUIRED_OBSERVATION_FIELDS) == expected
    
    def test_valid_protocols(self):
        assert "MQTT" in VALID_PROTOCOLS
        assert "HTTPS" in VALID_PROTOCOLS
        assert len(VALID_PROTOCOLS) >= 5
    
    def test_valid_anomaly_types(self):
        assert VALID_ANOMALY_TYPES == ("normal", "traffic_spike", "destination_expansion", "combined")
    
    def test_valid_device_types(self):
        assert list(VALID_DEVICE_TYPES) == DEVICE_CATEGORIES
    
    def test_validate_observation_normal(self):
        obs = {
            "device_id": "bulb_01",
            "device_type": "Smart Bulb",
            "timestamp": "2024-01-01T00:00:00",
            "packet_rate": 1.0,
            "packet_size": 100,
            "destination_count": 1,
            "protocol": "MQTT",
            "connection_duration": 0.5,
            "is_anomaly": False,
            "anomaly_type": "normal",
        }
        assert validate_observation(obs) is True
    
    def test_validate_observation_anomaly(self):
        obs = {
            "device_id": "cam_01",
            "device_type": "Smart Camera",
            "timestamp": "2024-01-01T00:00:00",
            "packet_rate": 50.0,
            "packet_size": 1000,
            "destination_count": 5,
            "protocol": "RTSP",
            "connection_duration": 10.0,
            "is_anomaly": True,
            "anomaly_type": "traffic_spike",
        }
        assert validate_observation(obs) is True
    
    def test_validate_observation_invalid_device_type(self):
        obs = {
            "device_id": "test_01",
            "device_type": "Invalid Device",
            "timestamp": "2024-01-01T00:00:00",
            "packet_rate": 1.0,
            "packet_size": 100,
            "destination_count": 1,
            "protocol": "MQTT",
            "connection_duration": 0.5,
            "is_anomaly": False,
            "anomaly_type": "normal",
        }
        assert validate_observation(obs) is False
    
    def test_validate_observation_invalid_protocol(self):
        obs = {
            "device_id": "bulb_01",
            "device_type": "Smart Bulb",
            "timestamp": "2024-01-01T00:00:00",
            "packet_rate": 1.0,
            "packet_size": 100,
            "destination_count": 1,
            "protocol": "INVALID",
            "connection_duration": 0.5,
            "is_anomaly": False,
            "anomaly_type": "normal",
        }
        assert validate_observation(obs) is False
    
    def test_validate_observation_negative_packet_rate(self):
        obs = {
            "device_id": "bulb_01",
            "device_type": "Smart Bulb",
            "timestamp": "2024-01-01T00:00:00",
            "packet_rate": -1.0,
            "packet_size": 100,
            "destination_count": 1,
            "protocol": "MQTT",
            "connection_duration": 0.5,
            "is_anomaly": False,
            "anomaly_type": "normal",
        }
        assert validate_observation(obs) is False
    
    def test_validate_observation_inconsistent_anomaly(self):
        obs = {
            "device_id": "bulb_01",
            "device_type": "Smart Bulb",
            "timestamp": "2024-01-01T00:00:00",
            "packet_rate": 1.0,
            "packet_size": 100,
            "destination_count": 1,
            "protocol": "MQTT",
            "connection_duration": 0.5,
            "is_anomaly": True,  # True but anomaly_type is normal
            "anomaly_type": "normal",
        }
        assert validate_observation(obs) is False
    
    def test_validate_fingerprint_valid(self):
        fp = {
            "device_id": "bulb_01",
            "feature": "packet_rate",
            "mean": 1.5,
            "std": 0.3,
        }
        assert validate_fingerprint(fp) is True
    
    def test_validate_fingerprint_invalid_feature(self):
        fp = {
            "device_id": "bulb_01",
            "feature": "invalid_feature",
            "mean": 1.5,
            "std": 0.3,
        }
        assert validate_fingerprint(fp) is False
    
    def test_validate_fingerprint_negative_std(self):
        fp = {
            "device_id": "bulb_01",
            "feature": "packet_rate",
            "mean": 1.5,
            "std": -0.1,
        }
        assert validate_fingerprint(fp) is False


# ============================================================
# DETERMINISTIC RNG TESTS
# ============================================================
class TestDeterministicRNG:
    """Test reproducible random number generation."""
    
    def test_same_seed_produces_same_sequence(self):
        rng1 = DeterministicRNG(seed=123)
        rng2 = DeterministicRNG(seed=123)
        
        seq1 = [rng1.uniform(0, 1) for _ in range(100)]
        seq2 = [rng2.uniform(0, 1) for _ in range(100)]
        
        assert seq1 == seq2
    
    def test_different_seeds_produce_different_sequences(self):
        rng1 = DeterministicRNG(seed=123)
        rng2 = DeterministicRNG(seed=456)
        
        seq1 = [rng1.uniform(0, 1) for _ in range(100)]
        seq2 = [rng2.uniform(0, 1) for _ in range(100)]
        
        assert seq1 != seq2
    
    def test_global_rng_consistency(self):
        set_global_seed(999)
        rng1 = get_rng()
        val1 = rng1.uniform(0, 1)
        
        set_global_seed(999)
        rng2 = get_rng()
        val2 = rng2.uniform(0, 1)
        
        assert val1 == val2
    
    def test_rng_methods_work(self):
        rng = DeterministicRNG(seed=42)
        
        # Test various methods don't crash
        assert 0 <= rng.uniform(0, 1) <= 1
        assert isinstance(rng.normal(0, 1), float)
        assert rng.choice([1, 2, 3]) in [1, 2, 3]
        assert 1 <= rng.integers(1, 10) < 10
        assert 0 <= rng.random() <= 1
        assert rng.exponential(1.0) >= 0
        assert rng.lognormal(0, 1) > 0
        assert 0 <= rng.beta(2, 2) <= 1
    
    def test_rng_shuffle_deterministic(self):
        arr1 = list(range(10))
        arr2 = list(range(10))
        rng1 = DeterministicRNG(seed=42)
        rng2 = DeterministicRNG(seed=42)
        rng1.shuffle(arr1)
        rng2.shuffle(arr2)
        assert arr1 == arr2


# ============================================================
# SAFE STATISTICAL OPERATIONS TESTS
# ============================================================
class TestSafeStatistics:
    """Test safe division and z-score calculations."""
    
    def test_safe_divide_normal(self):
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 3) == pytest.approx(3.333333)
    
    def test_safe_divide_by_zero(self):
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=-1) == -1
        assert safe_divide(10, 1e-10) == 0.0  # Near zero
    
    def test_safe_z_score_normal(self):
        assert safe_z_score(10, 5, 2) == 2.5
        assert safe_z_score(5, 5, 2) == 0.0
    
    def test_safe_z_score_zero_std(self):
        assert safe_z_score(10, 5, 0) == 0.0
        assert safe_z_score(10, 5, 1e-10) == 0.0
    
    def test_safe_mean_std_normal(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std = safe_mean_std(vals)
        assert mean == 3.0
        assert std > 0
    
    def test_safe_mean_std_single_value(self):
        vals = np.array([5.0])
        mean, std = safe_mean_std(vals)
        assert mean == 5.0
        assert std == EPSILON  # Should be clamped to epsilon
    
    def test_safe_mean_std_empty(self):
        vals = np.array([])
        mean, std = safe_mean_std(vals)
        assert mean == 0.0
        assert std == 0.0


# ============================================================
# UTILITY FUNCTION TESTS
# ============================================================
class TestUtilities:
    """Test utility functions."""
    
    def test_generate_device_ids(self):
        from src.config import DEVICE_CATEGORIES, DEVICE_ID_PREFIX
        devices = generate_device_ids(DEVICE_CATEGORIES, 5, DEVICE_ID_PREFIX)
        
        assert len(devices) == 25
        # Check each category has 5 devices
        for cat in DEVICE_CATEGORIES:
            cat_devices = [d for d in devices if d["device_type"] == cat]
            assert len(cat_devices) == 5
            # Check IDs are unique and follow pattern
            for i, d in enumerate(cat_devices, 1):
                assert d["device_id"] == f"{DEVICE_ID_PREFIX[cat]}_{i:02d}"
    
    def test_select_protocol(self):
        rng = DeterministicRNG(seed=42)
        protocols = ("MQTT", "CoAP", "HTTPS")
        selected = select_protocol(protocols, rng=rng)
        assert selected in protocols
    
    def test_select_protocol_weighted(self):
        rng = DeterministicRNG(seed=42)
        protocols = ("MQTT", "CoAP")
        # Weight heavily towards MQTT
        selected = select_protocol(protocols, rng=rng, weights=[0.9, 0.1])
        # With fixed seed, should be deterministic
        assert selected in protocols
    
    def test_validate_dataframe_schema(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        assert validate_dataframe_schema(df, ["a", "b"]) is True
        assert validate_dataframe_schema(df, ["a", "d"]) is False
    
    def test_validate_no_nan_inf(self):
        df_clean = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        assert validate_no_nan_inf(df_clean, ["a", "b"]) is True
        
        df_nan = pd.DataFrame({"a": [1.0, np.nan], "b": [3.0, 4.0]})
        assert validate_no_nan_inf(df_nan, ["a", "b"]) is False
        
        df_inf = pd.DataFrame({"a": [1.0, np.inf], "b": [3.0, 4.0]})
        assert validate_no_nan_inf(df_inf, ["a", "b"]) is False


# ============================================================
# INTEGRATION TEST: IMPORTS WORK
# ============================================================
class TestImports:
    """Verify all modules can be imported without errors."""
    
    def test_import_config(self):
        import src.config
        assert hasattr(src.config, 'MASTER_SEED')
    
    def test_import_schemas(self):
        import src.schemas
        assert hasattr(src.schemas, 'Observation')
    
    def test_import_utils(self):
        import src.utils
        assert hasattr(src.utils, 'DeterministicRNG')
    
    def test_import_src_package(self):
        import src
        assert hasattr(src, 'MASTER_SEED')
        assert hasattr(src, 'Observation')
        assert hasattr(src, 'DeterministicRNG')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])