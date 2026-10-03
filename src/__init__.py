"""
IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection

A reproducible simulation pipeline for IoT device behavioural fingerprinting
and statistical anomaly detection.
"""

__version__ = "1.0.0"
__author__ = "IoT A&P Capstone Team"

# Core configuration and schemas
from .config import (
    PROJECT_ROOT,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_RESULTS_DIR,
    OUTPUTS_FIGURES_DIR,
    OUTPUTS_REPORTS_DIR,
    SIMULATION_WOKWI_DIR,
    MASTER_SEED,
    DEVICE_CATEGORIES,
    NUM_DEVICES,
    DEVICES_PER_CATEGORY,
    CATEGORY_PROFILES,
    OBSERVATIONS_PER_DEVICE_MIN,
    OBSERVATIONS_PER_DEVICE_MAX,
    FEATURE_NAMES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FINGERPRINT_FEATURES,
    ANOMALY_TYPES,
    ANOMALY_MULTIPLIERS,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
    EPSILON,
    EXPERIMENT_NAMES,
    NORMAL_TRAFFIC_CSV,
    DEVICE_FINGERPRINTS_CSV,
    ANOMALOUS_TRAFFIC_CSV,
)

from .schemas import (
    Observation,
    NormalObservation,
    DeviceFingerprint,
    DeviceFingerprints,
    DetectionResult,
    ConfusionMatrix,
    DetectionMetrics,
    AnomalyTypeMetrics,
    ExperimentConfig,
    ExperimentResult,
    validate_observation,
    validate_fingerprint,
    REQUIRED_OBSERVATION_FIELDS,
    VALID_PROTOCOLS,
    VALID_ANOMALY_TYPES,
    VALID_DEVICE_TYPES,
)

from .utils import (
    DeterministicRNG,
    get_rng,
    set_global_seed,
    safe_divide,
    safe_z_score,
    safe_mean_std,
    save_csv,
    load_csv,
    save_json,
    load_json,
    generate_timestamps,
    generate_device_ids,
    select_protocol,
    validate_dataframe_schema,
    validate_no_nan_inf,
)

from .io_bridge import (
    WokwiLogIngester,
    WokwiExportAdapter,
    create_sample_wokwi_log,
)

from .simulator import (
    VirtualDevice,
    NormalTrafficSimulator,
    generate_normal_traffic,
    validate_normal_traffic,
)

from .fingerprint import (
    FingerprintCalculator,
    compute_device_fingerprints,
    load_device_fingerprints,
    get_device_fingerprint_dict,
    validate_fingerprints,
)

from .anomalies import (
    AnomalyInjector,
    inject_anomalies,
    inject_all_anomaly_scenarios,
    create_unified_experiment_dataset,
    validate_anomalous_traffic,
)

from .detector import (
    AnomalyDetector,
    run_detection,
    run_detection_on_all_scenarios,
    validate_detection_results,
)

from .experiments import (
    ExperimentRunner,
    run_experiments,
    load_experiment_results,
    validate_experiment_results,
)

from .evaluation import (
    EvaluationEngine,
    run_evaluation,
    load_evaluation_summary,
    validate_evaluation_outputs,
)

from .visualization import (
    plot_packet_rate_vs_time,
    plot_destination_count_vs_time,
    plot_anomaly_score_vs_time,
    plot_device_wise_anomaly_count,
    plot_threshold_comparison,
    plot_detection_performance,
    generate_all_visualizations,
    validate_visualizations,
)

__all__ = [
    # Config
    "PROJECT_ROOT",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "DATA_RESULTS_DIR",
    "OUTPUTS_FIGURES_DIR",
    "OUTPUTS_REPORTS_DIR",
    "SIMULATION_WOKWI_DIR",
    "MASTER_SEED",
    "DEVICE_CATEGORIES",
    "NUM_DEVICES",
    "DEVICES_PER_CATEGORY",
    "CATEGORY_PROFILES",
    "OBSERVATIONS_PER_DEVICE_MIN",
    "OBSERVATIONS_PER_DEVICE_MAX",
    "FEATURE_NAMES",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "FINGERPRINT_FEATURES",
    "ANOMALY_TYPES",
    "ANOMALY_MULTIPLIERS",
    "DETECTION_THRESHOLDS",
    "DEFAULT_THRESHOLD",
    "EPSILON",
    "EXPERIMENT_NAMES",
    "NORMAL_TRAFFIC_CSV",
    "DEVICE_FINGERPRINTS_CSV",
    "ANOMALOUS_TRAFFIC_CSV",
    # Schemas
    "Observation",
    "NormalObservation",
    "DeviceFingerprint",
    "DeviceFingerprints",
    "DetectionResult",
    "ConfusionMatrix",
    "DetectionMetrics",
    "AnomalyTypeMetrics",
    "ExperimentConfig",
    "ExperimentResult",
    "validate_observation",
    "validate_fingerprint",
    "REQUIRED_OBSERVATION_FIELDS",
    "VALID_PROTOCOLS",
    "VALID_ANOMALY_TYPES",
    "VALID_DEVICE_TYPES",
    # Utils
    "DeterministicRNG",
    "get_rng",
    "set_global_seed",
    "safe_divide",
    "safe_z_score",
    "safe_mean_std",
    "save_csv",
    "load_csv",
    "save_json",
    "load_json",
    "generate_timestamps",
    "generate_device_ids",
    "select_protocol",
    "validate_dataframe_schema",
    "validate_no_nan_inf",
    # IO Bridge
    "WokwiLogIngester",
    "WokwiExportAdapter",
    "create_sample_wokwi_log",
    # Simulator
    "VirtualDevice",
    "NormalTrafficSimulator",
    "generate_normal_traffic",
    "validate_normal_traffic",
    # Fingerprint
    "FingerprintCalculator",
    "compute_device_fingerprints",
    "load_device_fingerprints",
    "get_device_fingerprint_dict",
    "validate_fingerprints",
    # Anomalies
    "AnomalyInjector",
    "inject_anomalies",
    "inject_all_anomaly_scenarios",
    "create_unified_experiment_dataset",
    "validate_anomalous_traffic",
    # Detector
    "AnomalyDetector",
    "run_detection",
    "run_detection_on_all_scenarios",
    "validate_detection_results",
    # Experiments
    "ExperimentRunner",
    "run_experiments",
    "load_experiment_results",
    "validate_experiment_results",
    # Evaluation
    "EvaluationEngine",
    "run_evaluation",
    "load_evaluation_summary",
    "validate_evaluation_outputs",
    # Visualization
    "plot_packet_rate_vs_time",
    "plot_destination_count_vs_time",
    "plot_anomaly_score_vs_time",
    "plot_device_wise_anomaly_count",
    "plot_threshold_comparison",
    "plot_detection_performance",
    "generate_all_visualizations",
    "validate_visualizations",
]