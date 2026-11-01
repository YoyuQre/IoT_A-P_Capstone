#!/usr/bin/env python3
"""
IoT Device Behaviour Fingerprinting & Anomaly Detection — Streamlit Dashboard

Interactive exploration of the simulation outputs:
    - Overview KPIs
    - Device Explorer
    - Behavioural Fingerprints
    - Anomaly Scenarios
    - Anomaly Detection
    - Experiments
    - Evaluation & Metrics
    - Threshold Analysis
    - Figures & Simulation

Usage:
    streamlit run app.py

All data is loaded from pre-generated pipeline outputs (run `python main.py --phase 9`
first if the data/ directory is empty).
"""

import sys
from pathlib import Path

# Ensure the project root is importable (so `src.*` resolves when launched from anywhere)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    ANOMALY_MULTIPLIERS,
    DETECTION_THRESHOLDS,
    DEVICE_CATEGORIES,
    MASTER_SEED,
    NUM_DEVICES,
    OUTPUTS_FIGURES_DIR,
)
from src.dashboard_data import (
    compute_category_stats,
    compute_device_fingerprint,
    compute_device_kpis,
    compute_overview_kpis,
    get_anomaly_type_metrics,
    get_anomaly_types,
    get_data_status_message,
    get_device_categories,
    get_device_category,
    get_device_list,
    get_devices_by_category,
    get_experiment_comparison_table,
    get_experiment_display_name,
    get_experiments,
    get_thresholds,
    get_threshold_comparison_data,
    get_wokwi_status,
    load_anomalous_traffic,
    load_detection_results,
    load_evaluation_anomaly_type_metrics,
    load_evaluation_overall_metrics,
    load_evaluation_summary,
    load_evaluation_threshold_comparison,
    load_experiment_results,
    load_fingerprints,
    load_raw_traffic,
    validate_data_availability,
)
from src.dashboard_utils import (
    apply_filters,
    create_anomaly_score_chart,
    create_bar_chart,
    create_confusion_matrix_chart,
    create_distribution_chart,
    create_download_button,
    create_fingerprint_chart,
    create_line_chart,
    create_pie_chart,
    create_scatter_chart,
    create_time_series_chart,
    get_anomaly_color,
    get_category_color,
    render_sidebar_status,
    show_data_status,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IoT Behaviour Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Overview",
    "Device Explorer",
    "Behavioural Fingerprints",
    "Anomaly Scenarios",
    "Anomaly Detection",
    "Experiments",
    "Evaluation & Metrics",
    "Threshold Analysis",
    "Figures & Simulation",
]

FEATURE_LABELS = {
    "packet_rate": "Packet Rate (pkt/s)",
    "packet_size": "Packet Size (bytes)",
    "destination_count": "Destination Count",
    "connection_duration": "Connection Duration (s)",
}

# ============================================================
# SMALL HELPERS
# ============================================================

def fmt_pct(value: float) -> str:
    """Format a 0-1 ratio as a percentage string."""
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def metric_row(items) -> None:
    """Render a row of st.metric widgets. `items` is a list of (label, value, delta)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        col.metric(label, value, delta)


def no_data_warning(what: str = "data") -> None:
    """Standard message when required data is missing."""
    st.warning(f"No {what} available.")
    st.info("Run the pipeline first to generate outputs:\n\n```bash\npython main.py --phase 9\n```")


# ============================================================
# PAGE: OVERVIEW
# ============================================================

def page_overview():
    st.title("Overview")
    st.caption("IoT Device Behaviour Fingerprinting & Simulation-Based Anomaly Detection")

    kpis = compute_overview_kpis()
    if not kpis:
        no_data_warning("overview data")
        return

    metric_row([
        ("Total Devices", f"{kpis.get('total_devices', 0)}"),
        ("Device Categories", f"{kpis.get('device_categories', 0)}"),
        ("Normal Observations", f"{kpis.get('total_observations', 0):,}"),
        ("Injected Anomalies", f"{kpis.get('injected_anomalies', 0):,}"),
        ("Detection Rate", fmt_pct(kpis.get("detection_rate", 0.0))),
        ("False Positive Rate", fmt_pct(kpis.get("false_positive_rate", 0.0))),
    ])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Devices per Category")
        cat_stats = compute_category_stats()
        if not cat_stats.empty:
            fig = create_pie_chart(cat_stats, names="category", values="num_devices",
                                   title="Device Distribution")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Category statistics unavailable.")

    with col2:
        st.subheader("Observations per Category")
        if not cat_stats.empty:
            fig = create_bar_chart(
                cat_stats, x="category", y="total_observations", color="category",
                title="Normal Observations by Category",
                x_label="Category", y_label="Observations",
                color_map={c: get_category_color(c) for c in DEVICE_CATEGORIES},
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Category statistics unavailable.")

    st.markdown("---")
    st.subheader("Anomalies by Scenario")

    counts = []
    for atype in get_anomaly_types():
        df = load_anomalous_traffic(atype)
        n = int(df["is_anomaly"].sum()) if not df.empty else 0
        counts.append({"anomaly_type": atype, "anomalies": n})

    if counts and any(c["anomalies"] for c in counts):
        counts_df = pd.DataFrame(counts)
        fig = create_bar_chart(
            counts_df, x="anomaly_type", y="anomalies", color="anomaly_type",
            title="Injected Anomalies per Scenario",
            x_label="Scenario", y_label="Anomaly Count",
            color_map={a: get_anomaly_color(a) for a in get_anomaly_types()},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No anomalous traffic files found.")

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.code(
        "Configuration -> Device Creation -> Normal Traffic\n"
        "      -> Fingerprinting -> Anomaly Injection -> Detection\n"
        "      -> Experiments -> Evaluation -> Visualization -> Dashboard",
        language="text",
    )
    st.caption(
        "Synthetic simulation study — no real hardware, no real packet capture. "
        f"Reproducible with master seed = {MASTER_SEED}."
    )


# ============================================================
# PAGE: DEVICE EXPLORER
# ============================================================

def page_device_explorer():
    st.title("Device Explorer")
    st.caption("Inspect per-device behaviour and anomalies.")

    devices = get_device_list()
    if not devices:
        no_data_warning("device/traffic data")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        category = st.selectbox("Device Category", ["All"] + get_device_categories())
    device_options = get_devices_by_category(category) if category != "All" else devices
    with c2:
        device_id = st.selectbox("Device", device_options)
    with c3:
        atype = st.selectbox("Anomaly Scenario", get_anomaly_types())

    st.markdown("---")

    # Device KPIs
    dkpis = compute_device_kpis(device_id)
    if dkpis:
        metric_row([
            ("Device Type", dkpis.get("device_type", "—")),
            ("Normal Observations", f"{dkpis.get('normal_observations', 0):,}"),
            ("Anomalous Observations", f"{dkpis.get('anomalous_observations', 0):,}"),
            ("Mean Packet Rate", f"{dkpis.get('mean_packet_rate', 0):.2f}"),
            ("Mean Packet Size", f"{dkpis.get('mean_packet_size', 0):.1f}"),
            ("Mean Destinations", f"{dkpis.get('mean_destination_count', 0):.2f}"),
        ])

    st.markdown("---")
    st.subheader(f"Behavioural Time Series — {device_id} ({atype})")

    device_df = load_anomalous_traffic(atype)
    if device_df.empty:
        no_data_warning(f"anomalous traffic for '{atype}'")
        return
    device_df = device_df[device_df["device_id"] == device_id].copy()
    if device_df.empty:
        st.info(f"No observations for {device_id} in the '{atype}' scenario.")
        return

    feature_tabs = st.tabs([FEATURE_LABELS[f] for f in ["packet_rate", "packet_size", "destination_count"]])
    for tab, feature in zip(feature_tabs, ["packet_rate", "packet_size", "destination_count"]):
        with tab:
            fig = create_time_series_chart(
                device_df, y_col=feature, title=f"{FEATURE_LABELS[feature]} — {device_id}",
                device_id=device_id, y_label=FEATURE_LABELS[feature],
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Observation Table")
    st.dataframe(device_df.sort_values("timestamp"), width="stretch", height=320)
    create_download_button(
        device_df.sort_values("timestamp"),
        filename=f"observations_{device_id}_{atype}.csv",
        label="Download observations (CSV)",
        key=f"dl_device_{device_id}_{atype}",
    )


# ============================================================
# PAGE: FINGERPRINTS
# ============================================================

def page_fingerprints():
    st.title("Behavioural Fingerprints")
    st.caption("Per-device mean / standard-deviation profiles used as the detection baseline.")

    fingerprints = load_fingerprints()
    if fingerprints.empty:
        no_data_warning("fingerprint data")
        return

    devices = sorted(fingerprints["device_id"].unique())
    c1, c2 = st.columns([1, 1])
    with c1:
        device_id = st.selectbox("Device", devices)
    with c2:
        feature = st.selectbox(
            "Feature", ["packet_rate", "packet_size", "destination_count"],
            format_func=lambda f: FEATURE_LABELS.get(f, f),
        )

    st.markdown("---")
    st.subheader(f"Fingerprint — {device_id}")
    dev_fp = compute_device_fingerprint(device_id)
    if dev_fp.empty:
        st.info("No fingerprint for this device.")
    else:
        display_fp = dev_fp.copy()
        display_fp["feature"] = display_fp["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
        display_fp["mean"] = display_fp["mean"].round(4)
        display_fp["std"] = display_fp["std"].round(6)
        st.dataframe(display_fp, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Fingerprint with ±1σ Bounds")
    obs = load_anomalous_traffic("combined")
    if obs.empty:
        obs = load_raw_traffic()
    if obs.empty:
        st.info("No observation data to overlay.")
    else:
        # Raw normal traffic has no is_anomaly column; create_fingerprint_chart expects it.
        if "is_anomaly" not in obs.columns:
            obs = obs.copy()
            obs["is_anomaly"] = False
        device_obs = obs[obs["device_id"] == device_id]
        fig = create_fingerprint_chart(dev_fp, feature, device_obs)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Mean Packet Rate — All Devices")
    overall = fingerprints[fingerprints["feature"] == "packet_rate"].copy()
    if not overall.empty:
        overall["category"] = overall["device_id"].apply(get_device_category)
        overall = overall.sort_values("mean", ascending=False)
        fig = create_bar_chart(
            overall, x="device_id", y="mean", color="category",
            title="Mean Packet Rate per Device",
            x_label="Device", y_label="Mean Packet Rate",
            color_map={c: get_category_color(c) for c in DEVICE_CATEGORIES},
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE: ANOMALY SCENARIOS
# ============================================================

def page_anomaly_scenarios():
    st.title("Anomaly Scenarios")
    st.caption("Controlled synthetic anomaly injections and their effect on device features.")

    st.subheader("Injection Configuration")
    rows = []
    for atype, mult in ANOMALY_MULTIPLIERS.items():
        rows.append({
            "Scenario": atype,
            "packet_rate ×": mult.get("packet_rate", 1.0),
            "destination_count ×": mult.get("destination_count", 1.0),
            "packet_size ×": mult.get("packet_size", 1.0),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("---")
    target = st.selectbox("Scenario to inspect", get_anomaly_types())
    feature = st.selectbox(
        "Feature", ["packet_rate", "packet_size", "destination_count"],
        format_func=lambda f: FEATURE_LABELS.get(f, f),
        key="scenario_feature",
    )

    anom_df = load_anomalous_traffic(target)
    normal_df = load_raw_traffic()
    if anom_df.empty:
        no_data_warning(f"traffic for '{target}'")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Normal (baseline)")
        fig = create_distribution_chart(
            normal_df, x=feature, title=f"Normal — {FEATURE_LABELS[feature]}",
            x_label=FEATURE_LABELS[feature], nbins=40,
        )
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader(f"Anomalous ({target})")
        fig = create_distribution_chart(
            anom_df, x=feature, color="is_anomaly",
            title=f"{target} — {FEATURE_LABELS[feature]}",
            x_label=FEATURE_LABELS[feature], nbins=40,
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Anomaly Count by Device")
    anom_only = anom_df[anom_df["is_anomaly"] == True]
    if not anom_only.empty:
        per_device = anom_only.groupby("device_id").size().reset_index(name="count")
        per_device["category"] = per_device["device_id"].apply(get_device_category)
        per_device = per_device.sort_values("count", ascending=False)
        fig = create_bar_chart(
            per_device, x="device_id", y="count", color="category",
            title=f"Anomalies per Device — {target}",
            x_label="Device", y_label="Anomalies",
            color_map={c: get_category_color(c) for c in DEVICE_CATEGORIES},
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No anomalies in this scenario.")


# ============================================================
# PAGE: ANOMALY DETECTION
# ============================================================

def page_anomaly_detection():
    st.title("Anomaly Detection")
    st.caption("Composite z-score detection: A = (|Z_rate| + |Z_dest| + |Z_size|) / 3")

    c1, c2, c3 = st.columns(3)
    with c1:
        atype = st.selectbox("Scenario", get_anomaly_types())
    with c2:
        threshold = st.selectbox("Threshold", get_thresholds(), index=1)
    det_all = load_detection_results(atype)
    if det_all.empty:
        no_data_warning(f"detection results for '{atype}'")
        return
    with c3:
        device_id = st.selectbox("Device", sorted(det_all["device_id"].unique()))

    det = det_all[det_all["threshold"] == threshold].copy()
    det_device = det[det["device_id"] == device_id].copy()

    st.markdown("---")

    # Classification breakdown
    tp = int(((det_device["ground_truth"] == True) & (det_device["predicted_label"] == True)).sum())
    tn = int(((det_device["ground_truth"] == False) & (det_device["predicted_label"] == False)).sum())
    fp = int(((det_device["ground_truth"] == False) & (det_device["predicted_label"] == True)).sum())
    fn = int(((det_device["ground_truth"] == True) & (det_device["predicted_label"] == False)).sum())
    metric_row([
        ("True Positives", f"{tp}"),
        ("True Negatives", f"{tn}"),
        ("False Positives", f"{fp}"),
        ("False Negatives", f"{fn}"),
    ])

    st.markdown("---")
    st.subheader(f"Anomaly Score vs Time — {device_id}")
    fig = create_anomaly_score_chart(det_device, device_id=device_id, threshold=threshold)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Z-Score Distributions (all devices in scenario)")
    z_features = ["z_packet_rate", "z_destination_count", "z_packet_size"]
    z_tabs = st.tabs([FEATURE_LABELS[f] for f in ["packet_rate", "destination_count", "packet_size"]])
    for tab, zcol in zip(z_tabs, z_features):
        with tab:
            fig = create_distribution_chart(
                det, x=zcol, color="ground_truth",
                title=f"Z-scores: {zcol}", x_label="Z-score", nbins=50,
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Detection Results Table")
    st.dataframe(det_device.sort_values("timestamp"), width="stretch", height=320)
    create_download_button(
        det_device.sort_values("timestamp"),
        filename=f"detection_{device_id}_{atype}_t{threshold}.csv",
        label="Download detection results (CSV)",
        key=f"dl_det_{device_id}_{atype}_{threshold}",
    )


# ============================================================
# PAGE: EXPERIMENTS
# ============================================================

def page_experiments():
    st.title("Experiments")
    st.caption("Five controlled experiments (A–E) evaluating the detector.")

    overall = load_evaluation_overall_metrics()
    if overall.empty:
        no_data_warning("experiment/evaluation data")
        return

    threshold = st.selectbox("Threshold", get_thresholds(), index=1, key="exp_threshold")

    st.subheader(f"Experiment Comparison (Threshold = {threshold})")
    comparison = get_experiment_comparison_table(threshold)
    if not comparison.empty:
        st.dataframe(comparison, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Confusion Matrices")
    rows = overall[overall["threshold"] == threshold]
    exp_names = rows["experiment"].tolist()
    cols = st.columns(2)
    for i, exp_name in enumerate(exp_names):
        row = rows[rows["experiment"] == exp_name].iloc[0]
        with cols[i % 2]:
            fig = create_confusion_matrix_chart(
                tp=int(row["tp"]), tn=int(row["tn"]), fp=int(row["fp"]), fn=int(row["fn"]),
                title=get_experiment_display_name(exp_name),
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Detection Performance (TP / FP / FN)")
    perf = rows[["experiment", "tp", "fp", "fn"]].copy()
    perf["experiment"] = perf["experiment"].apply(get_experiment_display_name)
    melted = perf.melt(id_vars="experiment", var_name="outcome", value_name="count")
    fig = px.bar(
        melted, x="experiment", y="count", color="outcome", barmode="group",
        labels={"experiment": "Experiment", "count": "Count", "outcome": "Outcome"},
        title=f"Detection Outcomes by Experiment (Threshold = {threshold})",
        color_discrete_map={"tp": "#2ECC71", "fp": "#E74C3C", "fn": "#F39C12"},
    )
    fig.update_layout(height=420, margin=dict(l=50, r=50, t=80, b=50))
    st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE: EVALUATION & METRICS
# ============================================================

def page_evaluation():
    st.title("Evaluation & Metrics")
    st.caption("Confusion-matrix derived metrics across experiments and anomaly types.")

    overall = load_evaluation_overall_metrics()
    if overall.empty:
        no_data_warning("evaluation data")
        return

    summary = load_evaluation_summary()
    if summary:
        metric_row([
            ("Experiments", f"{summary.get('total_experiments', 0)}"),
            ("Thresholds Tested", f"{len(summary.get('thresholds_tested', []))}"),
            ("Perfect Detection", f"{len(summary.get('perfect_detection_experiments', []))}"),
            ("Experiments w/ FP", f"{len(summary.get('experiments_with_false_positives', []))}"),
        ])
        st.markdown("---")

    st.subheader("Overall Metrics")
    display = overall.copy()
    display["experiment"] = display["experiment"].apply(get_experiment_display_name)
    for col in ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]:
        display[col] = display[col].round(3)
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Per-Anomaly-Type Metrics")
    type_metrics = load_evaluation_anomaly_type_metrics()
    if not type_metrics.empty:
        display_tm = type_metrics.copy()
        display_tm["experiment"] = display_tm["experiment"].apply(get_experiment_display_name)
        for col in ["precision", "recall", "f1_score"]:
            display_tm[col] = display_tm[col].round(3)
        st.dataframe(display_tm, width="stretch", hide_index=True)
    else:
        st.info("Per-anomaly-type metrics unavailable.")

    st.markdown("---")
    st.subheader("Metric Comparison by Experiment (Threshold = 3.0)")
    t3 = overall[overall["threshold"] == 3.0].copy()
    t3["experiment"] = t3["experiment"].apply(get_experiment_display_name)
    metric_choice = st.selectbox(
        "Metric", ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"],
    )
    fig = create_bar_chart(
        t3, x="experiment", y=metric_choice, color="experiment",
        title=f"{metric_choice.replace('_', ' ').title()} by Experiment",
        x_label="Experiment", y_label=metric_choice.replace("_", " ").title(),
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE: THRESHOLD ANALYSIS
# ============================================================

def page_threshold_analysis():
    st.title("Threshold Analysis")
    st.caption("Effect of the detection threshold on accuracy, precision, recall, F1 and FPR.")

    comp = load_evaluation_threshold_comparison()
    if comp.empty:
        no_data_warning("threshold comparison data")
        return

    base_cols = ["threshold", "accuracy", "precision", "recall", "f1_score", "false_positive_rate"]
    st.subheader("Threshold Comparison (Experiment E)")
    display = comp[base_cols].copy()
    for col in base_cols[1:]:
        display[col] = display[col].round(3)
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown("---")
    st.subheader("Metrics vs Threshold")
    melt_cols = ["accuracy", "precision", "recall", "f1_score"]
    melted = comp[["threshold"] + melt_cols].melt(
        id_vars="threshold", var_name="metric", value_name="value"
    )
    fig = px.line(
        melted, x="threshold", y="value", color="metric", markers=True,
        labels={"threshold": "Threshold", "value": "Score", "metric": "Metric"},
        title="Detection Metrics vs Threshold",
    )
    fig.update_layout(height=420, yaxis=dict(range=[-0.05, 1.05]), margin=dict(l=50, r=50, t=80, b=50))
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("False Positive Rate vs Threshold")
    fig = px.line(
        comp, x="threshold", y="false_positive_rate", markers=True,
        labels={"threshold": "Threshold", "false_positive_rate": "False Positive Rate"},
        title="False Positive Rate vs Threshold",
    )
    fig.update_layout(height=350, margin=dict(l=50, r=50, t=80, b=50))
    st.plotly_chart(fig, width="stretch")

    st.info(
        "All thresholds (2.0, 3.0, 4.0) yield perfect separation for the controlled synthetic "
        "anomalies, so metrics remain flat. Larger effect sizes were chosen for statistical "
        "distinguishability — this does not imply real-world security performance."
    )


# ============================================================
# PAGE: FIGURES & SIMULATION
# ============================================================

def page_figures():
    st.title("Figures & Simulation")
    st.caption("Static publication figures and the Wokwi hardware-simulation layer status.")

    st.subheader("Publication Figures")
    if OUTPUTS_FIGURES_DIR.exists():
        figures = sorted(OUTPUTS_FIGURES_DIR.glob("*.png"))
        if figures:
            cols = st.columns(2)
            for i, fig_path in enumerate(figures):
                with cols[i % 2]:
                    st.image(str(fig_path), caption=fig_path.stem.replace("_", " ").title(),
                             width="stretch")
        else:
            st.info("No figures found. Run `python main.py --phase 8` to generate them.")
    else:
        st.info("Figures directory not found.")

    st.markdown("---")
    st.subheader("Wokwi Simulation Layer")
    status = get_wokwi_status()
    st.write(f"- Diagram available: **{'Yes' if status['diagram_available'] else 'No'}**")
    st.write(f"- Firmware available: **{'Yes' if status['firmware_available'] else 'No'}**")
    if status["firmware_files"]:
        st.write(f"- Firmware files: {', '.join(status['firmware_files'])}")
    st.caption(
        "The Wokwi ESP32 layer is a design/visualization artifact. The reproducible "
        "experimental engine is the Python simulator (src/simulator.py)."
    )

    st.markdown("---")
    st.subheader("About & Limitations")
    st.markdown(
        """
        - **Synthetic traffic only** — no real network packets captured.
        - **Controlled anomalies** — parametric transformations, not real attacks.
        - **No physical hardware** — no live ESP32 or sensors.
        - **Statistical detector only** — no ML/DL, no adaptive learning.
        - **No cloud connectivity** — all data is stored and processed locally.
        - Results are deterministic and reproducible with a fixed master seed.
        """
    )


# ============================================================
# MAIN
# ============================================================

PAGE_FUNCTIONS = {
    "Overview": page_overview,
    "Device Explorer": page_device_explorer,
    "Behavioural Fingerprints": page_fingerprints,
    "Anomaly Scenarios": page_anomaly_scenarios,
    "Anomaly Detection": page_anomaly_detection,
    "Experiments": page_experiments,
    "Evaluation & Metrics": page_evaluation,
    "Threshold Analysis": page_threshold_analysis,
    "Figures & Simulation": page_figures,
}


def main():
    st.sidebar.title("IoT Anomaly Dashboard")
    st.sidebar.caption("Behavioural Fingerprinting & Detection")

    page = st.sidebar.radio("Navigation", PAGES, index=0)

    st.sidebar.markdown("---")
    render_sidebar_status()

    # Global data availability guard
    checks = validate_data_availability()
    available = sum(checks.values())
    total = len(checks)
    if available == 0:
        st.error("No pipeline data found.")
        st.info("Generate the data first:\n\n```bash\npython main.py --phase 9\n```")
        return

    st.sidebar.markdown("---")
    st.sidebar.caption(get_data_status_message())

    PAGE_FUNCTIONS[page]()


if __name__ == "__main__":
    main()
