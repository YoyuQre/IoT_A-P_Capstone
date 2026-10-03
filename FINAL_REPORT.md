# IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection

## Final Mini Project Report

**Anjuman-i-Islam's**  
**M. H. Saboo Siddik College of Engineering**  
8, Saboo Siddik Polytechnic Road, Byculla, Mumbai-8  

**Department of Computer Science and Engineering (IOTCSBCT)**  
**Academic Year 2026-2027**

---

## Title Page

**A Mini Project Report on**  
**IoT DEVICE BEHAVIOUR FINGERPRINTING AND SIMULATION-BASED ANOMALY DETECTION**

Submitted in partial fulfilment of the Term Work for  
**IoT Mini Project (Course Code: IOTCSBCT)**

Submitted by  
**Mohammed Yahya (Roll No. _______)**  
**______________ (Roll No. _______)**    **______________ (Roll No. _______)**  

Batch: ____    Semester: ____  

Under the guidance of  
**<Subject In-charge Name>**  

Department of Computer Science and Engineering (IOTCSBCT)  
M. H. Saboo Siddik College of Engineering, Byculla, Mumbai-8  
Academic Year 2026-2027

---

## Acknowledgement

We express our sincere gratitude to our project guide, <Subject In-charge Name>, for the continuous support, guidance, and encouragement throughout the development of this project. Their valuable suggestions and technical insights were instrumental in shaping this work.

We are thankful to the Department of Computer Science and Engineering (IOTCSBCT), M. H. Saboo Siddik College of Engineering, for providing the necessary infrastructure and academic environment to carry out this project.

We also acknowledge the open-source community for providing the tools and libraries (Python, NumPy, Pandas, Matplotlib, Streamlit, Wokwi) that made this simulation-based project possible.

Finally, we thank our parents and friends for their constant motivation and support.

---

## Abstract

The proliferation of Internet of Things (IoT) devices has introduced significant security challenges due to their limited computational resources, lack of regular updates, and diverse communication protocols. Traditional signature-based intrusion detection systems fail against novel or zero-day attacks. This project presents a simulation-based approach for IoT device behavioural fingerprinting and statistical anomaly detection.

The system simulates 25 virtual IoT devices across five categories (Smart Bulb, Smart Camera, Smart Lock, Smart Meter, Smart TV) using a hybrid architecture: a Wokwi-based ESP32 virtual device layer for demonstration, and a Python statistical engine for reproducible experimentation. Each device generates synthetic network traffic with category-specific statistical profiles (packet rate, packet size, destination count, connection duration, protocol).

Behavioural fingerprints are established per device using mean and standard deviation of numerical features computed from normal traffic only. Three controlled anomaly scenarios are injected: Traffic Spike (10× packet rate), Destination Expansion (8× destination count), and Combined (multi-vector). A statistical detector computes per-feature z-scores and a composite anomaly score A = (|Z₁| + |Z₂| + |Z₃|)/3, classifying observations using thresholds 2.0, 3.0, and 4.0.

Experiments demonstrate perfect detection (Precision=1.000, Recall=1.000, F1=1.000, FPR=0.000) across all scenarios and thresholds, validating the approach for clearly separated synthetic anomalies. An interactive Streamlit dashboard enables exploration of device behaviour, fingerprints, anomaly scores, threshold effects, and detection performance. All results are fully reproducible with a fixed seed (42).

**Keywords:** IoT Security, Behavioural Fingerprinting, Statistical Anomaly Detection, Synthetic Simulation, Z-Score, Streamlit Dashboard

---

## Table of Contents

| Section | Title | Page |
|---------|-------|------|
| | Title Page | i |
| | Acknowledgement | ii |
| | Abstract | iii |
| | Table of Contents | iv |
| | List of Figures | v |
| | List of Tables | vi |
| 1 | Introduction | 1 |
| 2 | Literature Survey / Existing Systems | 3 |
| 3 | System Design | 5 |
| 4 | Implementation | 10 |
| 5 | Results | 15 |
| 6 | Conclusion & Future Scope | 21 |
| | References | 23 |
| | Appendix | 25 |

---

## List of Figures

| Figure | Title | Page |
|--------|-------|------|
| Fig. 3.1 | Hybrid Simulation Architecture | 5 |
| Fig. 3.2 | Wokwi ESP32 Circuit Diagram | 6 |
| Fig. 3.3 | System Flowchart | 7 |
| Fig. 3.4 | Statistical Detection Pipeline | 8 |
| Fig. 4.1 | Device Category Traffic Profiles | 10 |
| Fig. 4.2 | Behavioural Fingerprint Example | 11 |
| Fig. 4.3 | Anomaly Injection Scenarios | 12 |
| Fig. 5.1 | Packet Rate vs Time (Smart Bulb) | 15 |
| Fig. 5.2 | Destination Count vs Time (Smart TV) | 16 |
| Fig. 5.3 | Anomaly Score vs Time | 16 |
| Fig. 5.4 | Device-Wise Anomaly Count | 17 |
| Fig. 5.5 | Threshold Comparison | 17 |
| Fig. 5.6 | Detection Performance | 18 |
| Fig. 5.7 | Streamlit Dashboard - Overview Page | 19 |
| Fig. 5.8 | Streamlit Dashboard - Device Explorer | 19 |
| Fig. 5.9 | Streamlit Dashboard - Anomaly Detection | 20 |

---

## List of Tables

| Table | Title | Page |
|-------|-------|------|
| Table 3.1 | Components Table | 5 |
| Table 3.2 | Device Category Profiles | 6 |
| Table 3.3 | Anomaly Injection Parameters | 8 |
| Table 4.1 | Sensor Simulation Parameters | 10 |
| Table 5.1 | Overall Metrics (Threshold = 3.0) | 15 |
| Table 5.2 | Per-Anomaly-Type Metrics (Threshold = 3.0) | 15 |
| Table 5.3 | Threshold Study Results (Experiment E) | 15 |
| Table 5.4 | Confusion Matrix - Experiment B | 15 |
| Table 5.5 | Confusion Matrix - Experiment C | 16 |
| Table 5.6 | Confusion Matrix - Experiment D | 16 |
| Table 5.7 | Confusion Matrix - Experiment A | 16 |
| Table 5.8 | Key Performance Indicators | 18 |

---

## 1. Introduction

### 1.1 Background

The Internet of Things (IoT) has transformed everyday objects into intelligent, connected devices. From smart homes to industrial automation, IoT devices now number in the billions worldwide. However, this proliferation brings significant security challenges. Unlike traditional computing systems, IoT devices often operate with limited computational resources, lack regular security updates, and communicate over diverse protocols. These constraints make them attractive targets for attackers seeking to compromise networks, exfiltrate data, or launch distributed denial-of-service (DDoS) attacks.

### 1.2 Problem Statement

Traditional signature-based intrusion detection systems (IDS) rely on known attack patterns and fail against novel or zero-day attacks. IoT environments require a detection approach that can learn normal device behaviour and flag deviations without prior knowledge of specific attack signatures. The challenge is to develop a lightweight, interpretable anomaly detection method suitable for resource-constrained IoT devices that does not require labelled attack data for training.

### 1.3 Objectives

The primary objectives of this project are:

1. **Create a reproducible synthetic IoT environment** with 25 virtual devices across 5 categories, each with statistically distinguishable communication profiles.
2. **Generate normal traffic** with temporal ordering, device-specific variation, and category-specific statistical profiles.
3. **Establish behavioural fingerprints** using only normal traffic (mean and standard deviation per device per feature).
4. **Inject controlled anomalies** representing three realistic attack scenarios with explicit ground truth.
5. **Implement a statistical detector** using per-feature z-scores and a composite anomaly score.
6. **Evaluate detection performance** across multiple thresholds (2, 3, 4) using confusion matrices and standard metrics.
7. **Produce publication-ready visualizations** and an interactive dashboard for exploration.
8. **Maintain strict academic integrity** by clearly distinguishing synthetic parameters from real-world measurements.

### 1.4 Scope

This project focuses on statistical anomaly detection in a controlled synthetic environment. It does not involve physical hardware deployment, real network traffic capture, or evaluation against real-world malware datasets. The Wokwi simulation layer serves as a demonstration of the device concept; the Python statistical engine is the reproducible experimental platform.

---

## 2. Literature Survey / Existing Systems

### 2.1 Existing Approaches

| Approach | Description | Limitation |
|----------|-------------|------------|
| **Signature-based IDS (Snort, Suricata)** | Pattern matching against known attack signatures | Fails against zero-day/novel attacks; requires constant signature updates |
| **Rule-based Firewalls** | Predefined rules for allowed/blocked traffic | Manual maintenance; cannot detect behavioural anomalies |
| **ML/DL Classifiers (Random Forest, LSTM, Autoencoders)** | Learn normal behaviour from labelled data | Require large labelled datasets; retraining for concept drift; black-box opacity |
| **Statistical Profiling (Mean/Std, PCA, Mahalanobis Distance)** | Model normal behaviour statistically | Often assume Gaussian distributions; sensitive to outliers |

### 2.2 Related Work

1. **Miettinen et al. (2017)** - "IoT SENTINEL: Automated Device-Type Identification for Security Enforcement in IoT" - Uses network traffic features for device fingerprinting but focuses on device identification rather than anomaly detection.

2. **Doshi et al. (2018)** - "Machine Learning DDoS Detection for IoT" - Applies ML classifiers to detect DDoS attacks but requires labelled attack data and significant computational resources.

3. **Bezawada et al. (2018)** - "Behavioural Fingerprinting of IoT Devices" - Proposes behavioural profiling using statistical features but evaluates on limited real-world datasets with small device counts.

### 2.3 How This Project Improves

- **No Training Data Required:** Uses only normal traffic to establish baselines; no labelled anomalies needed.
- **Interpretable:** Per-feature z-scores provide clear explanation of why an observation is flagged.
- **Lightweight:** Only mean/std storage per device per feature; O(1) computation per observation.
- **Reproducible:** Deterministic simulation with fixed seed enables exact reproduction.
- **Comprehensive Evaluation:** Includes confusion matrices, per-threshold metrics, per-anomaly-type breakdown, and threshold sensitivity analysis.
- **Interactive Dashboard:** Streamlit-based exploration of devices, fingerprints, anomalies, and detection performance.

---

## 3. System Design

### 3.1 Components Table

| Sr. No. | Component | Category | Model / Specification | Purpose in Project |
|---------|-----------|----------|----------------------|-------------------|
| 1 | DHT22 Temperature & Humidity Sensor | Sensor | DHT22 (Virtual) | Simulates environmental telemetry for IoT device |
| 2 | Virtual Network Traffic Generator | Sensor | Python Simulator (Custom) | Generates synthetic packet rate, packet size, destination count |
| 3 | Anomaly Injection Module | Actuator | Python Module (Custom) | Injects controlled Traffic Spike, Destination Expansion, Combined anomalies |
| 4 | ESP32 Microcontroller | Controller | ESP32 DevKit V1 (Virtual/Wokwi) | Reads virtual sensors, drives anomaly actuator, sends telemetry |
| 5 | Streamlit Dashboard | Cloud/Platform | Streamlit + Python | Live data visualization, device exploration, anomaly analysis |

### 3.2 Device Category Profiles

| Category | Packet Rate (pkt/s) | Packet Size (bytes) | Destinations | Protocols |
|----------|---------------------|---------------------|--------------|-----------|
| Smart Bulb | 0.5 – 2.0 | 50 – 150 | 1 – 2 | MQTT, CoAP |
| Smart Camera | 10 – 50 | 800 – 1500 | 1 – 3 | RTSP, HTTPS |
| Smart Lock | 0.01 – 0.1 | 30 – 100 | 1 | MQTT, CoAP |
| Smart Meter | 0.1 – 1.0 | 100 – 300 | 1 | MQTT, HTTP |
| Smart TV | 5 – 100 | 500 – 1500 | 2 – 6 | HTTPS, QUIC, RTMP |

*All ranges are synthetic simulation parameters chosen for statistical distinguishability, not measurements of real commercial devices.*

### 3.3 Hybrid Simulation Architecture

![Hybrid Simulation Architecture](Fig. 3.1)

The system employs a hybrid simulation approach:

```
Layer A: IoT Device Simulation (Wokwi)
    ├── ESP32 DevKit V1
    ├── DHT22 Sensor (Virtual)
    ├── Green LED (Status Heartbeat)
    ├── Red LED (Anomaly Indicator)
    └── Button (Anomaly Trigger)
         ↓ Serial/JSON Telemetry
Layer B: Python Statistical Engine
    ├── Normal Traffic Generation
    ├── Behavioural Fingerprinting
    ├── Controlled Anomaly Injection
    ├── Statistical Detection
    ├── Experiment Framework
    ├── Evaluation & Metrics
    ├── Visualization (Matplotlib)
    └── Interactive Dashboard (Streamlit)
         ↓ File-based Bridge (JSONL/CSV)
Layer C: Data Contract
    └── Reproducible Experiment Outputs
```

### 3.4 Wokwi ESP32 Circuit Diagram

![Wokwi ESP32 Circuit Diagram](Fig. 3.2)

The Wokwi simulation demonstrates the physical device concept:

| Connection | ESP32 Pin | Component Pin |
|------------|-----------|---------------|
| DHT22 VDD | GPIO 4 | VDD |
| DHT22 GND | GPIO 5 | GND |
| DHT22 DATA | GPIO 15 | DATA |
| Status LED (Green) | GPIO 21 | Anode |
| Status LED GND | GND | Cathode |
| Anomaly LED (Red) | GPIO 22 | Anode |
| Anomaly LED GND | GND | Cathode |
| Anomaly Button | GPIO 23 | Pin 1 |
| Button GND | GND | Pin 2 |

**Anomaly Modes (Button Press Cycles):**
1. **Normal** → Green LED heartbeat, Red LED off
2. **Traffic Spike** → 10× packet rate increase
3. **Destination Expansion** → 8× destination count increase
4. **Combined** → Both rate and destination increases
4. **Normal** (Auto-reset after 30 seconds)

### 3.5 System Flowchart

![System Flowchart](Fig. 3.3)

```
START
  │
  ▼
Initialize Configuration (Seed=42)
  │
  ▼
Create 25 Virtual Devices (5 categories × 5)
  │
  ▼
Generate Normal Traffic (100-200 obs/device)
  │
  ▼
Compute Behavioural Fingerprints (μ, σ per device/feature)
  │
  ▼
Inject Anomalies (3 Scenarios × 234 anomalies each)
  │
  ▼
Statistical Detection (Z-scores → Composite Score A)
  │
  ▼
Threshold Classification (2.0, 3.0, 4.0)
  │
  ▼
Run 5 Experiments (A-E)
  │
  ▼
Evaluation (TP/TN/FP/FN, Precision, Recall, F1)
  │
  ▼
Visualization (6 Figures) + Dashboard
  │
  ▼
Reproducibility Test (Seed=42)
  │
  ▼
END
```

### 3.6 Statistical Detection Pipeline

![Statistical Detection Pipeline](Fig. 3.4)

```
Normal Traffic (is_anomaly=False)
        │
        ▼
Behavioural Fingerprint
  ┌─────────────────────────────┐
  │ For each device, feature:   │
  │   μ = mean(normal_values)   │
  │   σ = std(normal_values)    │
  │   σ = max(σ, ε) where ε=1e-8│
  └─────────────────────────────┘
        │
        ▼
Anomaly Injection (Ground Truth Preserved)
        │
        ▼
Z-Score Calculation
  Z_feature = (x - μ) / σ
        │
        ▼
Composite Anomaly Score
  A = (|Z_packet_rate| + |Z_dest_count| + |Z_packet_size|) / 3
        │
        ▼
Threshold Classification
  A ≥ threshold → Anomalous
  A < threshold → Normal
        │
        ▼
Evaluation (TP, TN, FP, FN, Precision, Recall, F1, FPR)
```

---

## 4. Implementation

### 4.1 Hardware Setup

The project uses a **hybrid simulation approach**:

**Layer A - Wokwi Virtual Device (Demonstration):**
- ESP32 DevKit V1 virtual microcontroller
- DHT22 temperature/humidity sensor (virtual)
- Two LEDs (Green: status heartbeat, Red: anomaly indicator)
- Push button (anomaly mode trigger)
- Serial JSON telemetry output

**Layer B - Python Statistical Engine (Reproducible Experimentation):**
- 25 virtual devices with deterministic behaviour
- 100-200 observations per device (total: 3,853)
- 60-second base interval with ±10s jitter

### 4.2 Software Tools & Libraries

| Tool/Library | Version | Purpose |
|--------------|---------|---------|
| Python | 3.13.9 | Core programming language |
| NumPy | 2.3.5 | Numerical computation, random generation |
| Pandas | 2.3.3 | Data manipulation, CSV I/O |
| Matplotlib | 3.10.6 | Static visualization (6 publication figures) |
| Streamlit | 1.38+ | Interactive dashboard |
| Pytest | 8.3.5 | Unit testing (132 tests) |
| Wokwi | Web/CLI | ESP32 virtual simulation |

### 4.3 Cloud Platform Setup

**Streamlit Dashboard (Local Cloud Simulation):**
- **Platform:** Localhost Streamlit server (`streamlit run app.py`)
- **Data Source:** Pre-generated CSV files from Python engine
- **Dashboard Pages:** 9 interactive pages
  1. Overview - KPIs, distributions, pipeline diagram
  2. Device Explorer - Per-device time series
  3. Behavioural Fingerprint - μ ± σ bounds vs observations
  4. Anomaly Detection - Z-scores, anomaly score vs time, TP/FP/FN
  5. Experiment Analysis - 5 experiments comparison
  6. Threshold Analysis - Threshold vs metrics
  7. Device Categories - Category-wise statistics
  8. Evaluation - Confusion matrix, KPIs, per-type metrics
  9. Raw Data Explorer - Filterable tables with CSV download

### 4.4 Key Code Snippets

#### 4.4.1 Deterministic Random Number Generation

```python
# src/utils.py
class DeterministicRNG:
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        return self._rng.uniform(low, high, size)
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._rng.normal(loc, scale, size)
```

#### 4.4.2 Normal Traffic Generation

```python
# src/simulator.py
def generate_normal_traffic(seed=42) -> pd.DataFrame:
    simulator = NormalTrafficSimulator(seed=seed)
    df = simulator.generate()
    return df
```

#### 4.4.3 Behavioural Fingerprinting

```python
# src/fingerprint.py
def compute_device_fingerprints(normal_df) -> pd.DataFrame:
    for device_id in normal_df["device_id"].unique():
        for feature in FINGERPRINT_FEATURES:
            values = device_df[feature].values
            mean, std = safe_mean_std(values)
            if std < EPSILON: std = EPSILON
            fingerprints.append({"device_id": device_id, "feature": feature, 
                                "mean": round(mean, 6), "std": round(std, 6)})
    return pd.DataFrame(fingerprints)
```

#### 4.4.4 Anomaly Injection

```python
# src/anomalies.py
ANOMALY_MULTIPLIERS = {
    "traffic_spike": {"packet_rate": 10.0, "destination_count": 1.0, "packet_size": 1.0},
    "destination_expansion": {"packet_rate": 1.0, "destination_count": 8.0, "packet_size": 1.0},
    "combined": {"packet_rate": 8.0, "destination_count": 6.0, "packet_size": 1.2},
}
```

#### 4.4.5 Statistical Detection

```python
# src/detector.py
def detect(observations, fingerprints, threshold=3.0):
    # Z-scores
    z_packet = (obs["packet_rate"] - fp["packet_rate"]["mean"]) / fp["packet_rate"]["std"]
    z_dest = (obs["destination_count"] - fp["dest_count"]["mean"]) / fp["dest_count"]["std"]
    z_size = (obs["packet_size"] - fp["packet_size"]["mean"]) / fp["packet_size"]["std"]
    
    # Composite score
    A = (abs(z_packet) + abs(z_dest) + abs(z_size)) / 3
    
    # Classification
    predicted = A >= threshold
    return predicted, A
```

#### 4.4.6 Streamlit Dashboard Entry Point

```python
# app.py
import streamlit as st
from src.dashboard_data import *

def main():
    st.set_page_config(page_title="IoT Behaviour Anomaly Detection", layout="wide")
    # ... sidebar, navigation, page rendering
```

---

## 5. Results

### 5.1 Dataset Summary

| Dataset | Observations | Devices | Anomalies |
|---------|--------------|---------|-----------|
| Normal Traffic | 3,853 | 25 | 0 |
| Traffic Spike | 3,853 | 25 | 234 |
| Destination Expansion | 3,853 | 25 | 234 |
| Combined | 3,853 | 25 | 234 |
| Fingerprints | 75 entries | 25 devices × 3 features | N/A |

### 5.2 Overall Metrics (Threshold = 3.0)

| Experiment | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | FPR |
|------------|----|----|----|----|----------|-----------|--------|-----|-----|
| **A: Normal** | 0 | 3853 | 0 | 0 | 1.000 | – | – | – | **0.000** |
| **B: Traffic Spike** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **C: Destination Expansion** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **D: Combined** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **E: Threshold Study** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |

*Table 5.1: Overall Metrics (Threshold = 3.0)*

### 5.3 Per-Anomaly-Type Metrics (Threshold = 3.0)

| Experiment | Anomaly Type | TP | TN | FP | FN | Precision | Recall | F1 |
|------------|--------------|----|----|----|----|-----------|--------|-----|
| B | traffic_spike | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| C | destination_expansion | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| D | combined | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| E | combined | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 |

*Table 5.2: Per-Anomaly-Type Metrics (Threshold = 3.0)*

### 5.4 Threshold Study Results (Experiment E)

| Threshold | Accuracy | Precision | Recall | F1 | FPR |
|-----------|----------|-----------|--------|-----|-----|
| 2.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 3.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 4.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

*Table 5.3: Threshold Study Results (Experiment E)*

### 5.5 Confusion Matrices

**Experiment A - Normal (Baseline)**

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| Actual Normal | 3853 | 0 |
| Actual Anomaly | 0 | 0 |

*Table 5.4: Confusion Matrix - Experiment A*

**Experiment B - Traffic Spike**

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| Actual Normal | 3619 | 0 |
| Actual Anomaly | 0 | 234 |

*Table 5.5: Confusion Matrix - Experiment B*

**Experiment C - Destination Expansion**

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| Actual Normal | 3619 | 0 |
| Actual Anomaly | 0 | 234 |

*Table 5.6: Confusion Matrix - Experiment C*

**Experiment D - Combined**

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| Actual Normal | 3619 | 0 |
| Actual Anomaly | 0 | 234 |

*Table 5.7: Confusion Matrix - Experiment D*

### 5.6 Key Performance Indicators

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Detection Rate (Recall)** | 1.000 | All injected anomalies detected |
| **Precision** | 1.000 | No false alarms on anomaly observations |
| **F1-Score** | 1.000 | Perfect harmonic mean |
| **False Positive Rate** | 0.000 | Zero false alarms on normal traffic |
| **Accuracy** | 1.000 | Perfect overall classification |

*Table 5.8: Key Performance Indicators*

### 5.7 Visualizations

The project generates 6 publication-ready figures saved to `outputs/figures/`:

**Figure 5.1 - Packet Rate vs Time**  
Shows normal behaviour followed by injected traffic spike for a Smart Bulb device (bulb_01). Normal observations in blue, anomalous spikes marked with red triangles.

**Figure 5.2 - Destination Count vs Time**  
Shows normal behaviour followed by destination expansion for a Smart TV device (tv_01). Normal observations in blue, anomalous expansions marked with orange triangles.

**Figure 5.3 - Anomaly Score vs Time**  
Shows composite anomaly score A with threshold line (3.0) and shaded anomalous region. Ground truth anomalies marked with black X (missed) and green circles (detected).

**Figure 5.4 - Device-Wise Anomaly Count**  
Bar chart showing detected anomaly counts per device, color-coded by device category (Smart Bulb=blue, Camera=red, Lock=green, Meter=orange, TV=purple).

**Figure 5.5 - Threshold Comparison**  
2×2 subplot grid comparing Accuracy, Precision, Recall, and F1-Score across thresholds 2.0, 3.0, 4.0 for all 5 experiments.

**Figure 5.6 - Detection Performance**  
Grouped bar chart showing TP (green), FP (red), FN (orange) counts per experiment at threshold 3.0. TN values annotated below bars.

### 5.8 Streamlit Dashboard

**Figure 5.7 - Dashboard Overview Page**  
Displays KPI cards (Total Devices=25, Categories=5, Observations=3853, Injected Anomalies=234, Detection Rate=100%, FPR=0%), distribution charts, and pipeline diagram.

**Figure 5.8 - Device Explorer Page**  
Interactive device selector with per-device time series charts for packet rate, packet size, destination count, and anomaly score.

**Figure 5.9 - Anomaly Detection Page**  
Shows per-feature Z-scores, composite anomaly score vs time with threshold overlay, and TP/FP/FN/FN classification breakdown.

---

## 6. Conclusion & Future Scope

### 6.1 Conclusion

This project successfully demonstrates a complete, reproducible simulation pipeline for IoT behavioural fingerprinting and statistical anomaly detection. The key achievements are:

1. **Reproducible Synthetic Environment** — 25 virtual IoT devices across 5 categories, each with statistically distinguishable communication profiles, fully deterministic with seed=42.

2. **Deterministic Pipeline** — End-to-end pipeline from traffic generation through evaluation, fully reproducible with a single seed command: `python main.py --phase 9`.

3. **Perfect Detection in Controlled Conditions** — Statistical fingerprinting with composite z-score achieves perfect detection (Precision=1.000, Recall=1.000, F1=1.000, FPR=0.000) across all anomaly scenarios and thresholds, validating the approach for clearly separated anomalies.

4. **Comprehensive Evaluation Framework** — Confusion matrices, per-threshold metrics, per-anomaly-type breakdown, and threshold sensitivity analysis provide thorough performance characterization.

4. **Interactive Exploration** — Streamlit dashboard with 9 pages enables intuitive exploration of device behaviour, fingerprints, anomalies, detection performance, and threshold effects.

5. **Academic Integrity** — All parameters explicitly labeled synthetic; no real-world performance claims; controlled anomalies clearly distinguished from real attacks.

### 6.2 Future Scope

1. **Realistic Noise & Concept Drift** — Add network jitter, packet loss, and gradual behaviour evolution to evaluate robustness.

2. **Adaptive Adversaries** — Implement adversarial anomaly generation that mimics normal traffic to test detector robustness.

3. **Real Malware Dataset Evaluation** — Test against real IoT malware traffic captures (Mirai, BASHLITE, Hajime).

4. **Edge Deployment & Resource Profiling** — Profile memory, CPU, and latency for deployment on actual ESP32/ESP8266 hardware.

5. **Multi-Site Federated Fingerprinting** — Extend to distributed environments with privacy-preserving aggregation.

6. **Concept Drift Adaptation** — Implement sliding-window or exponential moving average fingerprints that adapt to legitimate behaviour changes.

7. **Protocol-Level Deep Inspection** — Extend beyond volumetric features to payload analysis (e.g., MQTT topic anomalies, CoAP method anomalies).

---

## References

[1] Miettinen, M., Marchal, S., Hafeez, I., Asokan, N., Sadeghi, A. R., & Seifert, J. P. (2017). IoT SENTINEL: Automated Device-Type Identification for Security Enforcement in IoT. *IEEE 37th International Conference on Distributed Computing Systems (ICDCS)*, 2171–2176.

[2] Doshi, R., Apthorpe, N., & Feamster, N. (2018). Machine Learning DDoS Detection for IoT. *Proceedings of the 2018 Workshop on IoT Security and Privacy*, 29–35.

[3] Bezawada, B., Bachani, M., Peterson, J., Shirazi, H., Ray, I., & Ray, I. (2018). Behavioral Fingerprinting of IoT Devices. *Proceedings of the 34th Annual Computer Security Applications Conference (ACSAC)*, 41–51.

[4] Sivanathan, A., Gharakheili, H. H., Loi, F., Radford, A., Wijenayake, C., Vishwanath, A., & Sivaraman, V. (2019). Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics. *IEEE Transactions on Mobile Computing*, 18(8), 1745–1759.

[5] Garcia, S., Grill, M., Stiborek, J., & Zunino, A. (2014). An Empirical Comparison of Botnet Detection Methods. *Computers & Security*, 45, 100–123.

[6] Antonakakis, M., April, T., Bailey, M., Bernhard, M., Bursztein, E., Cochran, J., ... & Weaver, N. (2017). Understanding the Mirai Botnet. *26th USENIX Security Symposium*, 1093–1110.

---

## Appendix

### A. Complete Source Code

The complete source code is organized in the `src/` directory:

```
src/
├── __init__.py              # Package exports
├── config.py                # Centralized configuration
├── schemas.py               # Data models & validation
├── utils.py                 # RNG, I/O, statistical helpers
├── io_bridge.py             # Wokwi↔Python integration design
├── simulator.py             # Normal traffic generation
├── fingerprint.py           # Behavioural fingerprinting
├── anomalies.py             # Anomaly injection
├── detector.py              # Statistical detection
├── experiments.py           # Experiment runner
├── evaluation.py            # Metrics calculation
├── visualization.py         # Matplotlib figures
├── dashboard_data.py        # Dashboard data access layer
├── dashboard_utils.py       # Dashboard chart helpers
```

#### Key Files:

**config.py** — Centralized configuration (device profiles, thresholds, paths)

**simulator.py** — Normal traffic generation with deterministic per-device variation

**fingerprint.py** — Behavioural fingerprint computation (μ, σ per device/feature)

**anomalies.py** — Controlled anomaly injection with explicit ground truth

**detector.py** — Z-score calculation, composite score, threshold classification

**experiments.py** — 5 experiments (A-E) with full metrics

**evaluation.py** — Metrics aggregation, LaTeX table generation

**visualization.py** — 6 publication-ready Matplotlib figures

**dashboard_data.py** — Cached data loading for Streamlit dashboard

**dashboard_utils.py** — Plotly chart helpers, filter widgets

**app.py** — Streamlit dashboard entry point (9 pages)

**main.py** — Pipeline orchestrator (phases 1-9)

**requirements.txt** — Dependencies (numpy, pandas, matplotlib, streamlit, pytest)

### B. Wokwi Simulation Files

**simulation/wokwi/diagram.json** — ESP32 + DHT22 + 2 LEDs + Button circuit

**simulation/wokwi/firmware/firmware.ino** — ESP32 sketch with anomaly modes

**simulation/wokwi/wokwi.toml** — Project metadata

### C. Data Files Generated

```
data/raw/normal_traffic.csv                    (3,853 obs, 25 devices)
data/processed/device_fingerprints.csv         (75 entries)
data/processed/anomalous_traffic_*.csv         (3 files × 3,853 rows)
data/results/detection_results_*.csv           (3 files × 11,559 rows)
data/results/experiment_results.csv            (57,795 rows)
data/results/evaluation_*.csv                  (5 metrics files)
outputs/figures/01-06_*.png                    (6 PNG figures)
```

### D. Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (Phases 1-9)
python main.py --phase 9

# Run individual phases
python main.py --phase 1   # Foundation
python main.py --phase 2   # Traffic generation
python main.py --phase 3   # Fingerprinting
python main.py --phase 4   # Anomaly injection
python main.py --phase 5   # Detection
python main.py --phase 6   # Experiments
python main.py --phase 7   # Evaluation
python main.py --phase 8   # Visualization
python main.py --phase 9   # Full pipeline + reproducibility test

# Launch interactive dashboard
streamlit run app.py
```

### E. Test Suite

```bash
# Run all tests (132 tests, all passing)
pytest tests/ -v

# Run specific phase tests
pytest tests/test_phase1.py -v
pytest tests/test_phase2.py -v
# ... etc.
```

---

**End of Report**