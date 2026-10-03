"""
Centralized configuration for IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection.

All simulation parameters, device categories, thresholds, and constants are defined here.
"""

from pathlib import Path
from typing import Final, List, Dict

# ============================================================
# PROJECT ROOT & DIRECTORIES
# ============================================================
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DATA_RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "data" / "results"
OUTPUTS_FIGURES_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "figures"
OUTPUTS_REPORTS_DIR: Final[Path] = PROJECT_ROOT / "outputs" / "reports"
SIMULATION_WOKWI_DIR: Final[Path] = PROJECT_ROOT / "simulation" / "wokwi"

# Ensure directories exist
for _dir in (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_RESULTS_DIR,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_REPORTS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================
MASTER_SEED: Final[int] = 42
NUMPY_SEED: Final[int] = MASTER_SEED

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
DEVICE_CATEGORIES: Final[List[str]] = [
    "Smart Bulb",
    "Smart Camera",
    "Smart Lock",
    "Smart Meter",
    "Smart TV",
]

NUM_DEVICES: Final[int] = 25
DEVICES_PER_CATEGORY: Final[int] = NUM_DEVICES // len(DEVICE_CATEGORIES)  # 5

# Device ID prefix per category
DEVICE_ID_PREFIX: Final[Dict[str, str]] = {
    "Smart Bulb": "bulb",
    "Smart Camera": "cam",
    "Smart Lock": "lock",
    "Smart Meter": "meter",
    "Smart TV": "tv",
}

# ============================================================
# SYNTHETIC TRAFFIC PARAMETERS (per device category)
# These are SIMULATION PARAMETERS, not measurements of real devices.
# ============================================================
# All values are synthetic ranges chosen for statistical distinguishability.
CATEGORY_PROFILES: Final[Dict[str, Dict[str, tuple]]] = {
    "Smart Bulb": {
        # Low packet rate, small packets, few destinations
        "packet_rate": (0.5, 2.0),           # packets/second
        "packet_size": (50, 150),            # bytes
        "destination_count": (1, 2),          # unique destinations
        "connection_duration": (0.1, 1.0),    # seconds
        "protocol": ("MQTT", "CoAP"),
    },
    "Smart Camera": {
        # High packet rate, large packets, few destinations
        "packet_rate": (10.0, 50.0),
        "packet_size": (800, 1500),
        "destination_count": (1, 3),
        "connection_duration": (5.0, 30.0),
        "protocol": ("RTSP", "HTTPS"),
    },
    "Smart Lock": {
        # Very low packet rate, small packets, very few destinations
        "packet_rate": (0.01, 0.1),
        "packet_size": (30, 100),
        "destination_count": (1, 1),
        "connection_duration": (0.5, 2.0),
        "protocol": ("MQTT", "CoAP"),
    },
    "Smart Meter": {
        # Periodic moderate rate, medium packets, single destination
        "packet_rate": (0.1, 1.0),
        "packet_size": (100, 300),
        "destination_count": (1, 1),
        "connection_duration": (1.0, 5.0),
        "protocol": ("MQTT", "HTTP"),
    },
    "Smart TV": {
        # Bursty high rate, large packets, multiple destinations
        "packet_rate": (5.0, 100.0),
        "packet_size": (500, 1500),
        "destination_count": (2, 6),
        "connection_duration": (10.0, 300.0),
        "protocol": ("HTTPS", "QUIC", "RTMP"),
    },
}

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
OBSERVATIONS_PER_DEVICE_MIN: Final[int] = 100
OBSERVATIONS_PER_DEVICE_MAX: Final[int] = 200
TIMESTAMP_START: Final[str] = "2024-01-01 00:00:00"
TIMESTAMP_INTERVAL_SECONDS: Final[int] = 60  # base interval between observations

# ============================================================
# FEATURE NAMES (canonical observation schema)
# ============================================================
FEATURE_NAMES: Final[List[str]] = [
    "packet_rate",
    "packet_size",
    "destination_count",
    "connection_duration",
]

NUMERICAL_FEATURES: Final[List[str]] = [
    "packet_rate",
    "packet_size",
    "destination_count",
]

CATEGORICAL_FEATURES: Final[List[str]] = [
    "protocol",
]

ALL_FEATURES: Final[List[str]] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Features used for fingerprinting (must be numerical)
FINGERPRINT_FEATURES: Final[List[str]] = [
    "packet_rate",
    "packet_size",
    "destination_count",
]

# ============================================================
# ANOMALY CONFIGURATION
# ============================================================
ANOMALY_TYPES: Final[List[str]] = [
    "normal",
    "traffic_spike",
    "destination_expansion",
    "combined",
]

# Multipliers for anomaly injection (applied to normal baseline)
ANOMALY_MULTIPLIERS: Final[Dict[str, Dict[str, float]]] = {
    "traffic_spike": {
        "packet_rate": 10.0,           # 10x increase
        "destination_count": 1.0,      # unchanged
        "packet_size": 1.0,            # unchanged
    },
    "destination_expansion": {
        "packet_rate": 1.0,            # unchanged
        "destination_count": 8.0,      # 8x increase
        "packet_size": 1.0,            # unchanged
    },
    "combined": {
        "packet_rate": 8.0,            # 8x increase
        "destination_count": 6.0,      # 6x increase
        "packet_size": 1.2,            # slight increase
    },
}

# Fraction of devices to inject anomalies into (per experiment)
ANOMALY_DEVICE_FRACTION: Final[float] = 0.4  # 40% of devices = 10 devices
# Fraction of observations to make anomalous (for selected devices)
ANOMALY_OBSERVATION_FRACTION: Final[float] = 0.15  # 15% of observations

# ============================================================
# DETECTION THRESHOLDS
# ============================================================
DETECTION_THRESHOLDS: Final[List[float]] = [2.0, 3.0, 4.0]
DEFAULT_THRESHOLD: Final[float] = 3.0

# Small epsilon to avoid division by zero in z-score
EPSILON: Final[float] = 1e-8

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
EXPERIMENT_NAMES: Final[List[str]] = [
    "experiment_a_normal",
    "experiment_b_traffic_spike",
    "experiment_c_destination_expansion",
    "experiment_d_combined",
    "experiment_e_threshold_study",
]

# ============================================================
# OUTPUT FILENAMES
# ============================================================
NORMAL_TRAFFIC_CSV: Final[str] = "normal_traffic.csv"
DEVICE_FINGERPRINTS_CSV: Final[str] = "device_fingerprints.csv"
ANOMALOUS_TRAFFIC_CSV: Final[str] = "anomalous_traffic.csv"
EXPERIMENT_RESULTS_CSV: Final[str] = "experiment_results.csv"
EVALUATION_METRICS_CSV: Final[str] = "evaluation_metrics.csv"