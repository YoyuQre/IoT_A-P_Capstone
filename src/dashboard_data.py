"""
Dashboard Data Access Layer

Provides clean data loading functions for the Streamlit dashboard.
All data is loaded from pre-generated project outputs.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
import json
import streamlit as st

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUTS_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


# ============================================================
# CACHED DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_raw_traffic() -> pd.DataFrame:
    """Load normal (baseline) traffic data."""
    path = DATA_RAW_DIR / "normal_traffic.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_anomalous_traffic(anomaly_type: str) -> pd.DataFrame:
    """Load anomalous traffic data for a specific anomaly type."""
    path = DATA_PROCESSED_DIR / f"anomalous_traffic_{anomaly_type}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_fingerprints() -> pd.DataFrame:
    """Load device behavioural fingerprints."""
    path = DATA_PROCESSED_DIR / "device_fingerprints.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_detection_results(anomaly_type: str) -> pd.DataFrame:
    """Load detection results for a specific anomaly type."""
    path = DATA_RESULTS_DIR / f"detection_results_{anomaly_type}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_experiment_results() -> pd.DataFrame:
    """Load consolidated experiment results."""
    path = DATA_RESULTS_DIR / "experiment_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_evaluation_overall_metrics() -> pd.DataFrame:
    """Load overall evaluation metrics."""
    path = DATA_RESULTS_DIR / "evaluation_overall_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_evaluation_threshold_comparison() -> pd.DataFrame:
    """Load threshold comparison metrics."""
    path = DATA_RESULTS_DIR / "evaluation_threshold_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_evaluation_anomaly_type_metrics() -> pd.DataFrame:
    """Load per-anomaly-type evaluation metrics."""
    path = DATA_RESULTS_DIR / "evaluation_anomaly_type_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_evaluation_summary() -> Dict[str, Any]:
    """Load evaluation summary."""
    path = DATA_RESULTS_DIR / "evaluation_summary.json"
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


# ============================================================
# DERIVED DATA HELPERS
# ============================================================

def get_device_list() -> List[str]:
    """Get sorted list of all device IDs."""
    df = load_raw_traffic()
    if df.empty:
        return []
    return sorted(df["device_id"].unique())


def get_device_categories() -> List[str]:
    """Get sorted list of device categories."""
    df = load_raw_traffic()
    if df.empty:
        return []
    return sorted(df["device_type"].unique())


def get_devices_by_category(category: str) -> List[str]:
    """Get devices belonging to a specific category."""
    df = load_raw_traffic()
    if df.empty:
        return []
    return sorted(df[df["device_type"] == category]["device_id"].unique())


def get_anomaly_types() -> List[str]:
    """Get list of available anomaly types."""
    return ["traffic_spike", "destination_expansion", "combined"]


def get_thresholds() -> List[float]:
    """Get list of available thresholds."""
    return [2.0, 3.0, 4.0]


def get_experiments() -> List[str]:
    """Get list of available experiments."""
    return [
        "experiment_a_normal",
        "experiment_b_traffic_spike",
        "experiment_c_destination_expansion",
        "experiment_d_combined",
        "experiment_e_threshold_study",
    ]


def get_experiment_display_name(exp_name: str) -> str:
    """Convert experiment name to display format."""
    return exp_name.replace("experiment_", "").replace("_", " ").title()


def get_device_category(device_id: str) -> str:
    """Get device category for a device ID."""
    df = load_raw_traffic()
    if df.empty:
        return "Unknown"
    match = df[df["device_id"] == device_id]
    if match.empty:
        return "Unknown"
    return match.iloc[0]["device_type"]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data_availability() -> Dict[str, bool]:
    """Check which required data files exist."""
    checks = {
        "raw_traffic": (DATA_RAW_DIR / "normal_traffic.csv").exists(),
        "fingerprints": (DATA_PROCESSED_DIR / "device_fingerprints.csv").exists(),
        "traffic_spike": (DATA_PROCESSED_DIR / "anomalous_traffic_traffic_spike.csv").exists(),
        "destination_expansion": (DATA_PROCESSED_DIR / "anomalous_traffic_destination_expansion.csv").exists(),
        "combined": (DATA_PROCESSED_DIR / "anomalous_traffic_combined.csv").exists(),
        "detection_traffic_spike": (DATA_RESULTS_DIR / "detection_results_traffic_spike.csv").exists(),
        "detection_destination_expansion": (DATA_RESULTS_DIR / "detection_results_destination_expansion.csv").exists(),
        "detection_combined": (DATA_RESULTS_DIR / "detection_results_combined.csv").exists(),
        "experiment_results": (DATA_RESULTS_DIR / "experiment_results.csv").exists(),
        "evaluation_overall": (DATA_RESULTS_DIR / "evaluation_overall_metrics.csv").exists(),
        "evaluation_threshold": (DATA_RESULTS_DIR / "evaluation_threshold_comparison.csv").exists(),
        "evaluation_type_metrics": (DATA_RESULTS_DIR / "evaluation_anomaly_type_metrics.csv").exists(),
    }
    return checks


def get_data_status_message() -> str:
    """Get a summary message about data availability."""
    checks = validate_data_availability()
    available = sum(checks.values())
    total = len(checks)
    if available == total:
        return f"All {total} data files available [OK]"
    else:
        missing = [k for k, v in checks.items() if not v]
        return f"{available}/{total} data files available. Missing: {', '.join(missing)}"


# ============================================================
# KPI CALCULATIONS
# ============================================================

def compute_overview_kpis() -> Dict[str, Any]:
    """Compute KPIs for the Overview page."""
    raw_df = load_raw_traffic()
    if raw_df.empty:
        return {}
    
    # Load all anomalous traffic
    all_anomalies = 0
    for atype in get_anomaly_types():
        df = load_anomalous_traffic(atype)
        all_anomalies += df["is_anomaly"].sum()
    
    # Get detection results at default threshold (3.0)
    # Use a single anomaly type (traffic_spike) as representative
    det_df = load_detection_results("traffic_spike")
    if not det_df.empty:
        threshold_df = det_df[det_df["threshold"] == 3.0]
        total_detected = int(threshold_df["predicted_label"].sum())
        total_ground_truth = int(threshold_df["ground_truth"].sum())
    else:
        total_detected = 0
        total_ground_truth = 0
    
    kpis = {
        "total_devices": raw_df["device_id"].nunique(),
        "device_categories": raw_df["device_type"].nunique(),
        "total_observations": len(raw_df),
        "injected_anomalies": int(all_anomalies / 3),  # Each anomaly type has same count
        "detected_anomalies": total_detected,
        "ground_truth_anomalies": total_ground_truth,
    }
    
    # Detection rate
    if kpis["ground_truth_anomalies"] > 0:
        kpis["detection_rate"] = kpis["detected_anomalies"] / kpis["ground_truth_anomalies"]
    else:
        kpis["detection_rate"] = 0.0
    
    # False positive rate (from experiment A)
    eval_overall = load_evaluation_overall_metrics()
    if not eval_overall.empty:
        exp_a = eval_overall[eval_overall["experiment"] == "experiment_a_normal"]
        if not exp_a.empty:
            kpis["false_positive_rate"] = float(exp_a[exp_a["threshold"] == 3.0]["false_positive_rate"].values[0])
        else:
            kpis["false_positive_rate"] = 0.0
    else:
        kpis["false_positive_rate"] = 0.0
    
    return kpis


def compute_device_kpis(device_id: str) -> Dict[str, Any]:
    """Compute KPIs for a specific device."""
    raw_df = load_raw_traffic()
    device_df = raw_df[raw_df["device_id"] == device_id]
    
    if device_df.empty:
        return {}
    
    # Get anomaly data for this device
    anomalous_traffic = load_anomalous_traffic("traffic_spike")
    device_anomalous = anomalous_traffic[anomalous_traffic["device_id"] == device_id]
    
    kpis = {
        "device_id": device_id,
        "device_type": device_df.iloc[0]["device_type"],
        "num_observations": len(device_df),
        "normal_observations": len(device_df),
        "anomalous_observations": int(device_anomalous["is_anomaly"].sum()),
        "anomaly_rate": 0.0,
        "mean_packet_rate": float(device_df["packet_rate"].mean()),
        "mean_packet_size": float(device_df["packet_size"].mean()),
        "mean_destination_count": float(device_df["destination_count"].mean()),
        "mean_connection_duration": float(device_df["connection_duration"].mean()),
    }
    
    if kpis["num_observations"] > 0:
        kpis["anomaly_rate"] = kpis["anomalous_observations"] / kpis["num_observations"]
    
    return kpis


def compute_device_fingerprint(device_id: str) -> pd.DataFrame:
    """Get fingerprint data for a specific device."""
    fps = load_fingerprints()
    if fps.empty:
        return pd.DataFrame()
    return fps[fps["device_id"] == device_id].copy()


# ============================================================
# EXPERIMENT METRICS HELPERS
# ============================================================

def get_experiment_metrics(experiment: str, threshold: float) -> Dict[str, Any]:
    """Get metrics for a specific experiment at a threshold."""
    eval_overall = load_evaluation_overall_metrics()
    if eval_overall.empty:
        return {}
    
    mask = (eval_overall["experiment"] == experiment) & (eval_overall["threshold"] == threshold)
    result = eval_overall[mask]
    if result.empty:
        return {}
    
    return result.iloc[0].to_dict()


def get_experiment_comparison_table(threshold: float = 3.0) -> pd.DataFrame:
    """Get comparison table for all experiments at a threshold."""
    eval_overall = load_evaluation_overall_metrics()
    if eval_overall.empty:
        return pd.DataFrame()
    
    result = eval_overall[eval_overall["threshold"] == threshold].copy()
    result["experiment_display"] = result["experiment"].apply(get_experiment_display_name)
    return result[["experiment_display", "tp", "tn", "fp", "fn", 
                   "accuracy", "precision", "recall", "f1_score", "false_positive_rate"]]


def get_threshold_comparison_data() -> pd.DataFrame:
    """Get threshold comparison for Experiment E."""
    return load_evaluation_threshold_comparison()


def get_anomaly_type_metrics() -> pd.DataFrame:
    """Get anomaly type specific metrics."""
    return load_evaluation_anomaly_type_metrics()


# ============================================================
# CATEGORY ANALYSIS HELPERS
# ============================================================

def compute_category_stats() -> pd.DataFrame:
    """Compute statistics per device category."""
    raw_df = load_raw_traffic()
    if raw_df.empty:
        return pd.DataFrame()
    
    # Load anomalous traffic for anomaly counts
    all_anomalous = []
    for atype in get_anomaly_types():
        df = load_anomalous_traffic(atype)
        all_anomalous.append(df)
    
    stats = []
    for category in get_device_categories():
        cat_devices = raw_df[raw_df["device_type"] == category]["device_id"].unique()
        cat_raw = raw_df[raw_df["device_type"] == category]
        
        # Count anomalies across all types
        anomaly_count = 0
        for atype_df in all_anomalous:
            cat_anomalous = atype_df[atype_df["device_id"].isin(cat_devices)]
            anomaly_count += cat_anomalous["is_anomaly"].sum()
        
        stats.append({
            "category": category,
            "num_devices": len(cat_devices),
            "total_observations": len(cat_raw),
            "avg_packet_rate": float(cat_raw["packet_rate"].mean()),
            "avg_packet_size": float(cat_raw["packet_size"].mean()),
            "avg_destination_count": float(cat_raw["destination_count"].mean()),
            "avg_connection_duration": float(cat_raw["connection_duration"].mean()),
            "anomaly_count": int(anomaly_count / 3),  # Divide by 3 anomaly types
            "anomaly_rate": 0.0,
        })
        
        if stats[-1]["total_observations"] > 0:
            stats[-1]["anomaly_rate"] = stats[-1]["anomaly_count"] / stats[-1]["total_observations"]
    
    return pd.DataFrame(stats)


# ============================================================
# WOKWI STATUS
# ============================================================

def get_wokwi_status() -> Dict[str, Any]:
    """Get Wokwi simulation layer status."""
    wokwi_dir = PROJECT_ROOT / "simulation" / "wokwi"
    diagram_path = wokwi_dir / "diagram.json"
    firmware_dir = wokwi_dir / "firmware"
    
    status = {
        "diagram_available": diagram_path.exists(),
        "firmware_available": firmware_dir.exists(),
        "firmware_files": [],
    }
    
    if firmware_dir.exists():
        status["firmware_files"] = [f.name for f in firmware_dir.iterdir() if f.is_file()]
    
    return status