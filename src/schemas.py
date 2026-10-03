"""
Canonical data schemas for IoT Device Behaviour Fingerprinting.

Defines the observation model, fingerprint model, and experiment result structures.
Uses TypedDict for structural typing without runtime overhead.
"""

from typing import TypedDict, Literal, NotRequired
from datetime import datetime


# ============================================================
# CANONICAL OBSERVATION SCHEMA
# ============================================================
class Observation(TypedDict):
    """
    Single synthetic network observation for a device.
    
    This is the core data model used throughout the pipeline.
    """
    device_id: str
    device_type: str
    timestamp: str  # ISO 8601 format
    packet_rate: float
    packet_size: float
    destination_count: int
    protocol: str
    connection_duration: float
    is_anomaly: bool
    anomaly_type: Literal["normal", "traffic_spike", "destination_expansion", "combined"]


class NormalObservation(TypedDict):
    """Observation without anomaly fields (for normal traffic generation)."""
    device_id: str
    device_type: str
    timestamp: str
    packet_rate: float
    packet_size: float
    destination_count: int
    protocol: str
    connection_duration: float


# ============================================================
# FINGERPRINT SCHEMA
# ============================================================
class DeviceFingerprint(TypedDict):
    """Statistical fingerprint for a single device-feature pair."""
    device_id: str
    feature: str
    mean: float
    std: float


class DeviceFingerprints(TypedDict):
    """Complete fingerprint set for one device across all fingerprint features."""
    device_id: str
    fingerprints: dict[str, DeviceFingerprint]  # key = feature name


# ============================================================
# DETECTION RESULT SCHEMA
# ============================================================
class DetectionResult(TypedDict):
    """Result of anomaly detection for a single observation."""
    device_id: str
    timestamp: str
    anomaly_type: Literal["normal", "traffic_spike", "destination_expansion", "combined"]
    ground_truth: bool  # True if actually anomalous
    z_packet_rate: float
    z_destination_count: float
    z_packet_size: float
    anomaly_score: float
    predicted_label: bool  # True if detected as anomalous
    threshold: float


# ============================================================
# EVALUATION METRICS SCHEMA
# ============================================================
class ConfusionMatrix(TypedDict):
    """Confusion matrix components."""
    tp: int
    tn: int
    fp: int
    fn: int


class DetectionMetrics(TypedDict):
    """Full detection metrics for an experiment/threshold combination."""
    experiment: str
    threshold: float
    total_observations: int
    anomaly_count: int
    normal_count: int
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float


# Per-anomaly-type metrics
class AnomalyTypeMetrics(TypedDict):
    """Metrics broken down by anomaly type."""
    anomaly_type: Literal["traffic_spike", "destination_expansion", "combined"]
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1_score: float


# ============================================================
# EXPERIMENT CONFIGURATION SCHEMA
# ============================================================
class ExperimentConfig(TypedDict):
    """Configuration for a single experiment run."""
    name: str
    anomaly_type: Literal["normal", "traffic_spike", "destination_expansion", "combined"]
    threshold: float
    anomaly_device_fraction: float
    anomaly_observation_fraction: float
    seed: int


# ============================================================
# EXPERIMENT RESULT SCHEMA
# ============================================================
class ExperimentResult(TypedDict):
    """Complete result of an experiment run."""
    experiment: str
    config: ExperimentConfig
    detection_results: list[DetectionResult]
    metrics: DetectionMetrics
    anomaly_type_metrics: list[AnomalyTypeMetrics]


# ============================================================
# VALIDATION HELPERS
# ============================================================
REQUIRED_OBSERVATION_FIELDS: tuple[str, ...] = (
    "device_id",
    "device_type",
    "timestamp",
    "packet_rate",
    "packet_size",
    "destination_count",
    "protocol",
    "connection_duration",
    "is_anomaly",
    "anomaly_type",
)

VALID_PROTOCOLS: tuple[str, ...] = (
    "MQTT", "CoAP", "RTSP", "HTTPS", "HTTP", "QUIC", "RTMP"
)

VALID_ANOMALY_TYPES: tuple[str, ...] = (
    "normal", "traffic_spike", "destination_expansion", "combined"
)

VALID_DEVICE_TYPES: tuple[str, ...] = (
    "Smart Bulb", "Smart Camera", "Smart Lock", "Smart Meter", "Smart TV"
)


def validate_observation(obs: dict) -> bool:
    """Validate that an observation dict has all required fields with valid values."""
    # Check required fields
    for field in REQUIRED_OBSERVATION_FIELDS:
        if field not in obs:
            return False
    
    # Validate device_type
    if obs["device_type"] not in VALID_DEVICE_TYPES:
        return False
    
    # Validate protocol
    if obs["protocol"] not in VALID_PROTOCOLS:
        return False
    
    # Validate anomaly_type
    if obs["anomaly_type"] not in VALID_ANOMALY_TYPES:
        return False
    
    # Validate numeric ranges
    if obs["packet_rate"] < 0:
        return False
    if obs["packet_size"] <= 0:
        return False
    if obs["destination_count"] < 1:
        return False
    if obs["connection_duration"] < 0:
        return False
    
    # Validate boolean
    if not isinstance(obs["is_anomaly"], bool):
        return False
    
    # Consistency check: is_anomaly should match anomaly_type
    if obs["is_anomaly"] != (obs["anomaly_type"] != "normal"):
        return False
    
    return True


def validate_fingerprint(fp: dict) -> bool:
    """Validate a device fingerprint entry."""
    required = ("device_id", "feature", "mean", "std")
    for field in required:
        if field not in fp:
            return False
    if fp["feature"] not in ("packet_rate", "packet_size", "destination_count"):
        return False
    if fp["std"] < 0:
        return False
    return True