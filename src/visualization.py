"""
Phase 8: Visualization & Analysis

Creates publication-ready visualizations from experiment and evaluation outputs.

Required visualizations:
1. Packet Rate vs Time (normal + traffic spike)
2. Destination Count vs Time (normal + destination expansion)
3. Anomaly Score vs Time (with threshold line)
4. Device-Wise Anomaly Count (bar chart)
5. Threshold Comparison (metrics across thresholds 2, 3, 4)
6. Detection Performance (confusion matrix / TP-FP-FN)
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .config import (
    MASTER_SEED,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
    OUTPUTS_FIGURES_DIR,
    DATA_RESULTS_DIR,
    DATA_PROCESSED_DIR,
    FINGERPRINT_FEATURES,
    EXPERIMENT_NAMES,
)
from .utils import load_csv


# ============================================================
# PLOTTING CONFIGURATION
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
COLORS = {
    'normal': '#2E86AB',
    'traffic_spike': '#E74C3C',
    'destination_expansion': '#F39C12',
    'combined': '#8E44AD',
    'anomaly_score': '#C0392B',
    'threshold_line': '#2C3E50',
    'device_bars': '#3498DB',
}

# Figure settings
FIG_SIZE = (10, 6)
DPI = 300
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 11


def save_figure(fig: plt.Figure, filename: str, output_dir: Optional[Path] = None) -> Path:
    """Save figure to output directory."""
    if output_dir is None:
        output_dir = OUTPUTS_FIGURES_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return filepath


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def plot_packet_rate_vs_time(
    normal_traffic_path: Optional[Path] = None,
    anomalous_traffic_path: Optional[Path] = None,
    device_id: str = "bulb_01",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Visualization 1: Packet Rate vs Time.
    
    Shows normal behaviour followed by injected traffic spike for a selected device.
    """
    if normal_traffic_path is None:
        normal_traffic_path = DATA_PROCESSED_DIR.parent / "raw" / "normal_traffic.csv"
    if anomalous_traffic_path is None:
        anomalous_traffic_path = DATA_PROCESSED_DIR / "anomalous_traffic_traffic_spike.csv"
    
    # Load data
    normal_df = load_csv(normal_traffic_path.parent, normal_traffic_path.name)
    anomaly_df = load_csv(anomalous_traffic_path.parent, anomalous_traffic_path.name)
    
    # Filter for selected device
    normal_device = normal_df[normal_df["device_id"] == device_id].copy()
    anomaly_device = anomaly_df[anomaly_df["device_id"] == device_id].copy()
    
    normal_device["timestamp"] = pd.to_datetime(normal_device["timestamp"])
    anomaly_device["timestamp"] = pd.to_datetime(anomaly_device["timestamp"])
    
    # Sort by timestamp
    normal_device = normal_device.sort_values("timestamp")
    anomaly_device = anomaly_device.sort_values("timestamp")
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Plot normal traffic
    ax.plot(
        normal_device["timestamp"],
        normal_device["packet_rate"],
        label="Normal",
        color=COLORS['normal'],
        alpha=0.7,
        linewidth=1.5,
        marker='o',
        markersize=3,
    )
    
    # Highlight anomalous observations
    anomaly_obs = anomaly_device[anomaly_device["is_anomaly"] == True]
    if len(anomaly_obs) > 0:
        ax.scatter(
            anomaly_obs["timestamp"],
            anomaly_obs["packet_rate"],
            label="Traffic Spike (Anomaly)",
            color=COLORS['traffic_spike'],
            s=50,
            zorder=5,
            marker='^',
            edgecolors='black',
            linewidth=0.5,
        )
    
    ax.set_xlabel("Time", fontsize=LABEL_SIZE)
    ax.set_ylabel("Packet Rate (packets/sec)", fontsize=LABEL_SIZE)
    ax.set_title(f"Packet Rate vs Time — {device_id} (Smart Bulb)", fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, pos: pd.Timestamp(x).strftime('%H:%M')
    ))
    fig.autofmt_xdate(rotation=45)
    
    fig.tight_layout()
    return save_figure(fig, "01_packet_rate_vs_time.png", output_dir)


def plot_destination_count_vs_time(
    normal_traffic_path: Optional[Path] = None,
    anomalous_traffic_path: Optional[Path] = None,
    device_id: str = "tv_01",
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Visualization 2: Destination Count vs Time.
    
    Shows normal behaviour followed by destination expansion for a selected device.
    """
    if normal_traffic_path is None:
        normal_traffic_path = DATA_PROCESSED_DIR.parent / "raw" / "normal_traffic.csv"
    if anomalous_traffic_path is None:
        anomalous_traffic_path = DATA_PROCESSED_DIR / "anomalous_traffic_destination_expansion.csv"
    
    normal_df = load_csv(normal_traffic_path.parent, normal_traffic_path.name)
    anomaly_df = load_csv(anomalous_traffic_path.parent, anomalous_traffic_path.name)
    
    normal_device = normal_df[normal_df["device_id"] == device_id].copy()
    anomaly_device = anomaly_df[anomaly_df["device_id"] == device_id].copy()
    
    normal_device["timestamp"] = pd.to_datetime(normal_device["timestamp"])
    anomaly_device["timestamp"] = pd.to_datetime(anomaly_device["timestamp"])
    
    normal_device = normal_device.sort_values("timestamp")
    anomaly_device = anomaly_device.sort_values("timestamp")
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Plot normal traffic
    ax.plot(
        normal_device["timestamp"],
        normal_device["destination_count"],
        label="Normal",
        color=COLORS['normal'],
        alpha=0.7,
        linewidth=1.5,
        marker='o',
        markersize=3,
    )
    
    # Highlight anomalous observations
    anomaly_obs = anomaly_device[anomaly_device["is_anomaly"] == True]
    if len(anomaly_obs) > 0:
        ax.scatter(
            anomaly_obs["timestamp"],
            anomaly_obs["destination_count"],
            label="Destination Expansion (Anomaly)",
            color=COLORS['destination_expansion'],
            s=50,
            zorder=5,
            marker='^',
            edgecolors='black',
            linewidth=0.5,
        )
    
    ax.set_xlabel("Time", fontsize=LABEL_SIZE)
    ax.set_ylabel("Destination Count", fontsize=LABEL_SIZE)
    ax.set_title(f"Destination Count vs Time — {device_id} (Smart TV)", fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, pos: pd.Timestamp(x).strftime('%H:%M')
    ))
    fig.autofmt_xdate(rotation=45)
    
    fig.tight_layout()
    return save_figure(fig, "02_destination_count_vs_time.png", output_dir)


def plot_anomaly_score_vs_time(
    detection_results_path: Optional[Path] = None,
    device_id: str = "bulb_01",
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Visualization 3: Anomaly Score vs Time.
    
    Shows anomaly score and the selected threshold for a device.
    """
    if detection_results_path is None:
        detection_results_path = DATA_RESULTS_DIR / "detection_results_traffic_spike.csv"
    
    results_df = load_csv(detection_results_path.parent, detection_results_path.name)
    
    # Filter for device and threshold
    device_results = results_df[
        (results_df["device_id"] == device_id) & 
        (results_df["threshold"] == threshold)
    ].copy()
    
    device_results["timestamp"] = pd.to_datetime(device_results["timestamp"])
    device_results = device_results.sort_values("timestamp")
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Plot anomaly score
    ax.plot(
        device_results["timestamp"],
        device_results["anomaly_score"],
        label="Anomaly Score",
        color=COLORS['anomaly_score'],
        alpha=0.8,
        linewidth=1.5,
        marker='o',
        markersize=3,
    )
    
    # Plot threshold line
    ax.axhline(
        y=threshold,
        color=COLORS['threshold_line'],
        linestyle='--',
        linewidth=2,
        label=f"Threshold ({threshold})",
    )
    
    # Highlight anomalies
    anomaly_points = device_results[device_results["ground_truth"] == True]
    if len(anomaly_points) > 0:
        ax.scatter(
            anomaly_points["timestamp"],
            anomaly_points["anomaly_score"],
            label="Ground Truth Anomaly",
            color='black',
            s=60,
            zorder=5,
            marker='x',
            linewidth=2,
        )
    
    # Shade region above threshold
    ax.fill_between(
        device_results["timestamp"],
        threshold,
        device_results["anomaly_score"].max() * 1.1,
        alpha=0.1,
        color=COLORS['traffic_spike'],
        label="Anomalous Region",
    )
    
    ax.set_xlabel("Time", fontsize=LABEL_SIZE)
    ax.set_ylabel("Anomaly Score (A)", fontsize=LABEL_SIZE)
    ax.set_title(f"Anomaly Score vs Time — {device_id} (Threshold={threshold})", fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, pos: pd.Timestamp(x).strftime('%H:%M')
    ))
    fig.autofmt_xdate(rotation=45)
    
    fig.tight_layout()
    return save_figure(fig, "03_anomaly_score_vs_time.png", output_dir)


def plot_device_wise_anomaly_count(
    experiment_results_path: Optional[Path] = None,
    normal_traffic_path: Optional[Path] = None,
    experiment_name: str = "experiment_b_traffic_spike",
    threshold: float = DEFAULT_THRESHOLD,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Visualization 4: Device-Wise Anomaly Count.
    
    Compares flagged anomalies across devices for a given experiment and threshold.
    """
    if experiment_results_path is None:
        experiment_results_path = DATA_RESULTS_DIR / "experiment_results.csv"
    if normal_traffic_path is None:
        normal_traffic_path = DATA_PROCESSED_DIR.parent / "raw" / "normal_traffic.csv"
    
    results_df = load_csv(experiment_results_path.parent, experiment_results_path.name)
    normal_df = load_csv(normal_traffic_path.parent, normal_traffic_path.name)
    
    # Filter for experiment and threshold
    exp_results = results_df[
        (results_df["experiment"] == experiment_name) & 
        (results_df["threshold"] == threshold)
    ].copy()
    
    # Count anomalies per device
    anomaly_counts = exp_results[exp_results["predicted_label"] == True].groupby("device_id").size()
    all_devices = exp_results["device_id"].unique()
    
    # Ensure all devices appear (even with 0 anomalies)
    counts_dict = {d: anomaly_counts.get(d, 0) for d in all_devices}
    
    # Sort by device_id
    sorted_devices = sorted(counts_dict.keys())
    counts = [counts_dict[d] for d in sorted_devices]
    
    # Get device types from normal traffic data
    device_type_map = normal_df.drop_duplicates("device_id").set_index("device_id")["device_type"].to_dict()
    
    # Color by device type
    bar_colors = []
    for d in sorted_devices:
        dtype = device_type_map.get(d, "Unknown")
        if "Bulb" in dtype:
            bar_colors.append('#3498DB')
        elif "Camera" in dtype:
            bar_colors.append('#E74C3C')
        elif "Lock" in dtype:
            bar_colors.append('#2ECC71')
        elif "Meter" in dtype:
            bar_colors.append('#F39C12')
        elif "TV" in dtype:
            bar_colors.append('#9B59B6')
        else:
            bar_colors.append('#95A5A6')
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    bars = ax.bar(
        range(len(sorted_devices)),
        counts,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
    )
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha='center',
                va='bottom',
                fontsize=FONT_SIZE - 2,
                fontweight='bold',
            )
    
    ax.set_xlabel("Device ID", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of Detected Anomalies", fontsize=LABEL_SIZE)
    ax.set_title(f"Device-Wise Anomaly Count — {experiment_name.replace('_', ' ').title()} (Threshold={threshold})", 
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(range(len(sorted_devices)))
    ax.set_xticklabels(sorted_devices, rotation=45, ha='right', fontsize=FONT_SIZE - 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for device types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498DB', label='Smart Bulb'),
        Patch(facecolor='#E74C3C', label='Smart Camera'),
        Patch(facecolor='#2ECC71', label='Smart Lock'),
        Patch(facecolor='#F39C12', label='Smart Meter'),
        Patch(facecolor='#9B59B6', label='Smart TV'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZE - 1)
    
    fig.tight_layout()
    return save_figure(fig, "04_device_wise_anomaly_count.png", output_dir)


def plot_threshold_comparison(
    evaluation_results_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Visualization 5: Threshold Comparison.
    
    Compares detection metrics across thresholds 2, 3, 4 for all experiments.
    """
    if evaluation_results_path is None:
        evaluation_results_path = DATA_RESULTS_DIR / "evaluation_overall_metrics.csv"
    
    eval_df = load_csv(evaluation_results_path.parent, evaluation_results_path.name)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall / Detection Rate"),
        ("f1_score", "F1-Score"),
    ]
    
    # Color by experiment
    exp_colors = {
        "experiment_a_normal": '#95A5A6',
        "experiment_b_traffic_spike": '#E74C3C',
        "experiment_c_destination_expansion": '#F39C12',
        "experiment_d_combined": '#8E44AD',
        "experiment_e_threshold_study": '#2E86AB',
    }
    
    exp_labels = {
        "experiment_a_normal": "A: Normal",
        "experiment_b_traffic_spike": "B: Traffic Spike",
        "experiment_c_destination_expansion": "C: Dest. Expansion",
        "experiment_d_combined": "D: Combined",
        "experiment_e_threshold_study": "E: Threshold Study",
    }
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for exp in EXPERIMENT_NAMES:
            exp_data = eval_df[eval_df["experiment"] == exp].sort_values("threshold")
            if len(exp_data) == 0:
                continue
            
            ax.plot(
                exp_data["threshold"],
                exp_data[metric],
                label=exp_labels[exp],
                color=exp_colors[exp],
                marker='o',
                markersize=8,
                linewidth=2,
                alpha=0.8,
            )
        
        ax.set_xlabel("Threshold", fontsize=LABEL_SIZE)
        ax.set_ylabel(title, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_xticks(DETECTION_THRESHOLDS)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FONT_SIZE - 1)
    
    fig.suptitle("Threshold Comparison Across Experiments", fontsize=TITLE_SIZE + 2, fontweight='bold')
    fig.tight_layout()
    return save_figure(fig, "05_threshold_comparison.png", output_dir)


def plot_detection_performance(
    evaluation_results_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Visualization 6: Detection Performance.
    
    Shows TP, FP, FN counts (grouped bar chart) for each experiment at default threshold.
    """
    if evaluation_results_path is None:
        evaluation_results_path = DATA_RESULTS_DIR / "evaluation_overall_metrics.csv"
    
    eval_df = load_csv(evaluation_results_path.parent, evaluation_results_path.name)
    
    # Filter for default threshold
    threshold_df = eval_df[eval_df["threshold"] == DEFAULT_THRESHOLD].copy()
    
    experiments = threshold_df["experiment"].tolist()
    tp = threshold_df["tp"].tolist()
    fp = threshold_df["fp"].tolist()
    fn = threshold_df["fn"].tolist()
    
    # Short experiment names
    exp_short = [e.replace("experiment_", "").replace("_", " ").title() for e in experiments]
    
    x = np.arange(len(experiments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    bars1 = ax.bar(
        x - width,
        tp,
        width,
        label='True Positives (TP)',
        color='#2ECC71',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x,
        fp,
        width,
        label='False Positives (FP)',
        color='#E74C3C',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
    )
    bars3 = ax.bar(
        x + width,
        fn,
        width,
        label='False Negatives (FN)',
        color='#F39C12',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
    )
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 5,
                    str(int(height)),
                    ha='center',
                    va='bottom',
                    fontsize=FONT_SIZE - 1,
                    fontweight='bold',
                )
    
    ax.set_xlabel("Experiment", fontsize=LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax.set_title(f"Detection Performance at Threshold={DEFAULT_THRESHOLD}", fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_short, rotation=15, ha='right', fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE - 1, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add TN as text annotation (since TN is very large)
    for i, exp in enumerate(experiments):
        tn = threshold_df[threshold_df["experiment"] == exp]["tn"].values[0]
        if tn > 0:
            ax.annotate(
                f'TN={tn}',
                xy=(x[i], 0),
                xytext=(0, -30),
                textcoords='offset points',
                ha='center',
                fontsize=FONT_SIZE - 2,
                color='gray',
            )
    
    fig.tight_layout()
    return save_figure(fig, "06_detection_performance.png", output_dir)


# ============================================================
# MASTER VISUALIZATION FUNCTION
# ============================================================
def generate_all_visualizations(
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Generate all 6 required visualizations.
    
    Returns:
        List of paths to generated figures
    """
    if output_dir is None:
        output_dir = OUTPUTS_FIGURES_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    
    print("Generating visualizations...")
    
    # 1. Packet Rate vs Time
    print("  1/6: Packet Rate vs Time...")
    try:
        path = plot_packet_rate_vs_time(output_dir=output_dir)
        generated.append(path)
        print(f"     Saved: {path.name}")
    except Exception as e:
        print(f"     FAILED: {e}")
    
    # 2. Destination Count vs Time
    print("  2/6: Destination Count vs Time...")
    try:
        path = plot_destination_count_vs_time(output_dir=output_dir)
        generated.append(path)
        print(f"     Saved: {path.name}")
    except Exception as e:
        print(f"     FAILED: {e}")
    
    # 3. Anomaly Score vs Time
    print("  3/6: Anomaly Score vs Time...")
    try:
        path = plot_anomaly_score_vs_time(output_dir=output_dir)
        generated.append(path)
        print(f"     Saved: {path.name}")
    except Exception as e:
        print(f"     FAILED: {e}")
    
    # 4. Device-Wise Anomaly Count
    print("  4/6: Device-Wise Anomaly Count...")
    try:
        path = plot_device_wise_anomaly_count(output_dir=output_dir)
        generated.append(path)
        print(f"     Saved: {path.name}")
    except Exception as e:
        print(f"     FAILED: {e}")
    
    # 5. Threshold Comparison
    print("  5/6: Threshold Comparison...")
    try:
        path = plot_threshold_comparison(output_dir=output_dir)
        generated.append(path)
        print(f"     Saved: {path.name}")
    except Exception as e:
        print(f"     FAILED: {e}")
    
    # 6. Detection Performance
    print("  6/6: Detection Performance...")
    try:
        path = plot_detection_performance(output_dir=output_dir)
        generated.append(path)
        print(f"     Saved: {path.name}")
    except Exception as e:
        print(f"     FAILED: {e}")
    
    print(f"\nGenerated {len(generated)}/6 visualizations")
    return generated


# ============================================================
# VALIDATION
# ============================================================
def validate_visualizations(output_dir: Optional[Path] = None) -> dict:
    """Validate that all expected visualizations exist."""
    if output_dir is None:
        output_dir = OUTPUTS_FIGURES_DIR
    
    expected = [
        "01_packet_rate_vs_time.png",
        "02_destination_count_vs_time.png",
        "03_anomaly_score_vs_time.png",
        "04_device_wise_anomaly_count.png",
        "05_threshold_comparison.png",
        "06_detection_performance.png",
    ]
    
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "files_found": [],
        "files_missing": [],
    }
    
    for fname in expected:
        fpath = output_dir / fname
        if fpath.exists():
            validation["files_found"].append(fname)
        else:
            validation["valid"] = False
            validation["errors"].append(f"Missing visualization: {fname}")
            validation["files_missing"].append(fname)
    
    return validation


if __name__ == "__main__":
    print("Generating all visualizations...")
    paths = generate_all_visualizations()
    
    print("\nValidation:")
    validation = validate_visualizations()
    for f in validation["files_found"]:
        print(f"  [OK] {f}")
    for f in validation["files_missing"]:
        print(f"  [MISSING] {f}")