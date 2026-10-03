"""
Dashboard Utilities

Helper functions for the Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PLOTLY THEME & STYLING
# ============================================================

# Color palette for device categories
CATEGORY_COLORS = {
    "Smart Bulb": "#3498DB",
    "Smart Camera": "#E74C3C",
    "Smart Lock": "#2ECC71",
    "Smart Meter": "#F39C12",
    "Smart TV": "#9B59B6",
}

# Anomaly colors
ANOMALY_COLORS = {
    "normal": "#2E86AB",
    "traffic_spike": "#E74C3C",
    "destination_expansion": "#F39C12",
    "combined": "#8E44AD",
}

# Threshold colors
THRESHOLD_COLORS = {
    2.0: "#3498DB",
    3.0: "#E74C3C",
    4.0: "#2ECC71",
}


def get_category_color(category: str) -> str:
    """Get color for device category."""
    return CATEGORY_COLORS.get(category, "#95A5A6")


def get_anomaly_color(anomaly_type: str) -> str:
    """Get color for anomaly type."""
    return ANOMALY_COLORS.get(anomaly_type, "#95A5A6")


def get_threshold_color(threshold: float) -> str:
    """Get color for threshold."""
    return THRESHOLD_COLORS.get(threshold, "#95A5A6")


# ============================================================
# CHART HELPERS
# ============================================================

def create_time_series_chart(
    df: pd.DataFrame,
    x_col: str = "timestamp",
    y_col: str = "packet_rate",
    color_col: Optional[str] = "is_anomaly",
    title: str = "",
    device_id: str = "",
    y_label: str = "",
    threshold_line: Optional[float] = None,
    threshold_label: str = "Threshold",
) -> go.Figure:
    """Create a time-series chart with optional anomaly highlighting."""
    fig = go.Figure()
    
    if df.empty:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(height=400, title=title)
        return fig
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
        df = df.copy()
        df[x_col] = pd.to_datetime(df[x_col])
    
    # Normal observations
    normal_df = df[df[color_col] == False] if color_col in df.columns else df
    fig.add_trace(go.Scatter(
        x=normal_df[x_col],
        y=normal_df[y_col],
        mode="lines+markers",
        name="Normal",
        line=dict(color="#2E86AB", width=1.5),
        marker=dict(size=4, color="#2E86AB"),
        hovertemplate=f"Time: %{{x}}<br>{y_label}: %{{y:.2f}}<extra></extra>",
    ))
    
    # Anomalous observations
    if color_col in df.columns:
        anomaly_df = df[df[color_col] == True]
        if not anomaly_df.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_df[x_col],
                y=anomaly_df[y_col],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    color="#E74C3C",
                    size=10,
                    symbol="triangle-up",
                    line=dict(color="black", width=1),
                ),
                hovertemplate=f"Time: %{{x}}<br>{y_label}: %{{y:.2f}}<br>ANOMALY<extra></extra>",
            ))
    
    # Threshold line
    if threshold_line is not None:
        fig.add_hline(
            y=threshold_line,
            line=dict(color="#2C3E50", width=2, dash="dash"),
            annotation_text=threshold_label,
            annotation_position="top right",
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_anomaly_score_chart(
    df: pd.DataFrame,
    device_id: str = "",
    threshold: float = 3.0,
) -> go.Figure:
    """Create anomaly score vs time chart with threshold."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No detection results available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = go.Figure()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Normal observations
    normal_df = df[df["ground_truth"] == False]
    fig.add_trace(go.Scatter(
        x=normal_df["timestamp"],
        y=normal_df["anomaly_score"],
        mode="markers",
        name="Normal",
        marker=dict(color="#2E86AB", size=5, opacity=0.6),
        hovertemplate="Time: %{x}<br>Score: %{y:.3f}<br>Normal<extra></extra>",
    ))
    
    # True Positives
    tp_df = df[(df["ground_truth"] == True) & (df["predicted_label"] == True)]
    if not tp_df.empty:
        fig.add_trace(go.Scatter(
            x=tp_df["timestamp"],
            y=tp_df["anomaly_score"],
            mode="markers",
            name="Detected Anomaly (TP)",
            marker=dict(color="#2ECC71", size=10, symbol="circle", line=dict(color="black", width=1)),
            hovertemplate="Time: %{x}<br>Score: %{y:.3f}<br>Detected Anomaly<extra></extra>",
        ))
    
    # False Negatives
    fn_df = df[(df["ground_truth"] == True) & (df["predicted_label"] == False)]
    if not fn_df.empty:
        fig.add_trace(go.Scatter(
            x=fn_df["timestamp"],
            y=fn_df["anomaly_score"],
            mode="markers",
            name="Missed Anomaly (FN)",
            marker=dict(color="#E74C3C", size=10, symbol="x", line=dict(color="black", width=2)),
            hovertemplate="Time: %{x}<br>Score: %{y:.3f}<br>Missed Anomaly<extra></extra>",
        ))
    
    # False Positives
    fp_df = df[(df["ground_truth"] == False) & (df["predicted_label"] == True)]
    if not fp_df.empty:
        fig.add_trace(go.Scatter(
            x=fp_df["timestamp"],
            y=fp_df["anomaly_score"],
            mode="markers",
            name="False Alarm (FP)",
            marker=dict(color="#F39C12", size=10, symbol="diamond", line=dict(color="black", width=1)),
            hovertemplate="Time: %{x}<br>Score: %{y:.3f}<br>False Alarm<extra></extra>",
        ))
    
    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color="#2C3E50", width=2, dash="dash"),
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top right",
    )
    
    fig.update_layout(
        title=f"Anomaly Score vs Time — {device_id} (Threshold={threshold})",
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_kpi_card(label: str, value: str, delta: Optional[str] = None) -> None:
    """Render a KPI card using Streamlit metrics."""
    st.metric(label=label, value=value, delta=delta)


def create_confusion_matrix_chart(
    tp: int, tn: int, fp: int, fn: int,
    title: str = "Confusion Matrix"
) -> go.Figure:
    """Create a confusion matrix heatmap."""
    z = [[tn, fp], [fn, tp]]
    x = ["Predicted Normal", "Predicted Anomaly"]
    y = ["Actual Normal", "Actual Anomaly"]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale="Blues",
        text=[[str(tn), str(fp)], [str(fn), str(tp)]],
        texttemplate="%{text}",
        textfont={"size": 20},
        hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>",
    ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(side="top"),
        height=350,
        margin=dict(l=80, r=50, t=80, b=50),
    )
    
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    color_map: Optional[Dict] = None,
) -> go.Figure:
    """Create a bar chart."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.bar(
        df, x=x, y=y, color=color,
        title=title,
        labels={x: x_label, y: y_label},
        color_discrete_map=color_map,
    )
    
    fig.update_layout(
        height=400,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    markers: bool = True,
) -> go.Figure:
    """Create a line chart."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.line(
        df, x=x, y=y, color=color,
        title=title,
        labels={x: x_label, y: y_label},
        markers=markers,
    )
    
    fig.update_layout(
        height=400,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_distribution_chart(
    df: pd.DataFrame,
    x: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "Count",
    nbins: int = 30,
) -> go.Figure:
    """Create a histogram/distribution chart."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.histogram(
        df, x=x, color=color,
        title=title,
        labels={x: x_label, "count": y_label},
        nbins=nbins,
        opacity=0.7,
    )
    
    fig.update_layout(
        height=400,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=50, r=50, t=80, b=50),
        bargap=0.1,
    )
    
    return fig


def create_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    """Create a scatter plot."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.scatter(
        df, x=x, y=y, color=color,
        title=title,
        labels={x: x_label, y: y_label},
        opacity=0.7,
    )
    
    fig.update_layout(
        height=400,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str = "",
) -> go.Figure:
    """Create a pie chart."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = px.pie(
        df, names=names, values=values,
        title=title,
        hole=0.4,
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


def create_fingerprint_chart(
    fingerprint_df: pd.DataFrame,
    feature: str,
    device_observations: pd.DataFrame,
) -> go.Figure:
    """Create a fingerprint visualization chart with mean ± std bounds."""
    fig = go.Figure()
    
    # Get fingerprint stats
    fp_row = fingerprint_df[fingerprint_df["feature"] == feature]
    if fp_row.empty:
        fig.add_annotation(text="No fingerprint data for this feature", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    mean_val = fp_row["mean"].values[0]
    std_val = fp_row["std"].values[0]
    
    # Ensure timestamp is datetime
    obs_df = device_observations.copy()
    if not pd.api.types.is_datetime64_any_dtype(obs_df["timestamp"]):
        obs_df["timestamp"] = pd.to_datetime(obs_df["timestamp"])
    
    obs_df = obs_df.sort_values("timestamp")
    
    # Normal observations
    normal_df = obs_df[obs_df["is_anomaly"] == False]
    fig.add_trace(go.Scatter(
        x=normal_df["timestamp"],
        y=normal_df[feature],
        mode="lines+markers",
        name="Normal",
        line=dict(color="#2E86AB", width=1),
        marker=dict(size=3, color="#2E86AB"),
    ))
    
    # Anomalous observations
    anomaly_df = obs_df[obs_df["is_anomaly"] == True]
    if not anomaly_df.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_df["timestamp"],
            y=anomaly_df[feature],
            mode="markers",
            name="Anomaly",
            marker=dict(color="#E74C3C", size=10, symbol="triangle-up", line=dict(color="black", width=1)),
        ))
    
    # Mean line
    fig.add_hline(
        y=mean_val,
        line=dict(color="#2C3E50", width=2),
        annotation_text=f"Mean ({mean_val:.2f})",
        annotation_position="right",
    )
    
    # Mean ± std
    fig.add_hline(
        y=mean_val + std_val,
        line=dict(color="#E74C3C", width=1, dash="dot"),
        annotation_text=f"Mean + 1σ ({mean_val + std_val:.2f})",
        annotation_position="right",
    )
    fig.add_hline(
        y=mean_val - std_val,
        line=dict(color="#E74C3C", width=1, dash="dot"),
        annotation_text=f"Mean - 1σ ({mean_val - std_val:.2f})",
        annotation_position="right",
    )
    
    fig.update_layout(
        title=f"Behavioural Fingerprint: {feature}",
        xaxis_title="Time",
        yaxis_title=feature.replace("_", " ").title(),
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    return fig


# ============================================================
# DOWNLOAD HELPERS
# ============================================================

def create_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download CSV",
    key: Optional[str] = None,
) -> None:
    """Create a download button for a DataFrame."""
    if df.empty:
        st.info("No data to download")
        return
    
    csv = df.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# ============================================================
# SIDEBAR HELPERS
# ============================================================

def render_sidebar_status():
    """Render data status indicators in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Status")
    
    from src.dashboard_data import validate_data_availability
    checks = validate_data_availability()
    
    for name, available in checks.items():
        status = "✓" if available else "✗"
        color = "green" if available else "red"
        st.sidebar.markdown(f"<span style='color:{color}'>{status}</span> {name.replace('_', ' ').title()}", unsafe_allow_html=True)


def render_global_filters() -> Dict[str, Any]:
    """Render global filter widgets and return selected values."""
    from src.dashboard_data import (
        get_device_list, get_device_categories, get_anomaly_types,
        get_thresholds, get_experiments, get_devices_by_category
    )
    
    filters = {}
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Global Filters")
    
    # Device Category
    categories = ["All"] + get_device_categories()
    filters["category"] = st.sidebar.selectbox("Device Category", categories, index=0)
    
    # Device ID (dependent on category)
    if filters["category"] != "All":
        devices = ["All"] + get_devices_by_category(filters["category"])
    else:
        devices = ["All"] + get_device_list()
    filters["device"] = st.sidebar.selectbox("Device ID", devices, index=0)
    
    # Anomaly Type
    anomaly_types = ["All"] + get_anomaly_types()
    filters["anomaly_type"] = st.sidebar.selectbox("Anomaly Type", anomaly_types, index=0)
    
    # Threshold
    thresholds = ["All"] + get_thresholds()
    filters["threshold"] = st.sidebar.selectbox("Threshold", thresholds, index=0)
    
    # Experiment
    experiments = ["All"] + get_experiments()
    filters["experiment"] = st.sidebar.selectbox("Experiment", experiments, index=0)
    
    return filters


def apply_filters(
    df: pd.DataFrame,
    filters: Dict[str, Any],
    device_col: str = "device_id",
    category_col: str = "device_type",
    anomaly_col: str = "anomaly_type",
    threshold_col: str = "threshold",
    experiment_col: str = "experiment",
) -> pd.DataFrame:
    """Apply global filters to a DataFrame."""
    result = df.copy()
    
    from src.dashboard_data import get_device_categories, get_devices_by_category
    
    if filters.get("category") and filters["category"] != "All":
        if category_col in result.columns:
            result = result[result[category_col] == filters["category"]]
    
    if filters.get("device") and filters["device"] != "All":
        if device_col in result.columns:
            result = result[result[device_col] == filters["device"]]
    
    if filters.get("anomaly_type") and filters["anomaly_type"] != "All":
        if anomaly_col in result.columns:
            result = result[result[anomaly_col] == filters["anomaly_type"]]
    
    if filters.get("threshold") and filters["threshold"] != "All":
        if threshold_col in result.columns:
            result = result[result[threshold_col] == filters["threshold"]]
    
    if filters.get("experiment") and filters["experiment"] != "All":
        if experiment_col in result.columns:
            result = result[result[experiment_col] == filters["experiment"]]
    
    return result


# ============================================================
# DATA VALIDATION HELPERS
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame",
) -> bool:
    """Validate that a DataFrame has required columns."""
    if df.empty:
        st.warning(f"{name} is empty")
        return False
    
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        st.error(f"{name} missing required columns: {missing}")
        return False
    
    return True


def show_data_status(available: bool, message: str = "") -> None:
    """Show data availability status."""
    if available:
        st.success(message or "Data loaded successfully")
    else:
        st.error(message or "Required data not available")
        st.info("Please run the simulation pipeline first: `python main.py --phase 9`")