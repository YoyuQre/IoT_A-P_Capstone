# IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection

## Project Overview

This project implements a complete simulation pipeline for IoT device behavioural fingerprinting and statistical anomaly detection. It generates synthetic IoT device traffic, establishes behavioural baselines, injects controlled anomalies, and evaluates a statistical detector — all in a fully reproducible manner.

**Key characteristics:**
- 25 virtual IoT devices across 5 categories
- Synthetic network communication records (not real traffic)
- Device-specific behavioural fingerprints using z-score statistics
- Three controlled anomaly scenarios
- Statistical anomaly detection with configurable thresholds
- Comprehensive evaluation with confusion matrices and metrics
- Publication-ready visualizations
- Interactive Streamlit dashboard for exploration

> **Academic Integrity Notice:** This is a controlled simulation study. All traffic is synthetic. Anomalies are simulated. There is no physical IoT hardware, no real packet capture, and no claim of real-world commercial IoT security performance.

## Problem Statement

IoT devices exhibit predictable communication patterns under normal operation. Deviations from these patterns may indicate compromise, malfunction, or misconfiguration. This project investigates whether simple statistical fingerprinting (mean/std per device per feature) can reliably detect anomalous behaviour in a simulated environment.

## Research Question

*Can device-specific statistical behavioural fingerprints, combined with a composite z-score anomaly metric, effectively discriminate between normal and anomalous IoT communication behaviour in a controlled synthetic environment?*

## Objectives

1. Create a reproducible synthetic IoT environment with 25 devices across 5 categories
2. Generate statistically distinguishable normal communication behaviour per device category
3. Compute per-device behavioural fingerprints from normal traffic only
4. Inject three controlled anomaly types with explicit ground truth
5. Implement a statistical anomaly detector using standardized deviations
6. Evaluate detection performance across multiple thresholds
7. Produce publication-ready visualizations and a complete academic report

## System Architecture

```
┌─────────────────────┐
│   Configuration     │  (config.py)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Device Creation    │  (25 devices, 5 categories)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Normal Traffic Gen  │  (simulator.py) → data/raw/normal_traffic.csv
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Fingerprinting      │  (fingerprint.py) → data/processed/device_fingerprints.csv
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Anomaly Injection  │  (anomalies.py) → data/processed/anomalous_traffic_*.csv
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Statistical Det.   │  (detector.py) → anomaly scores + classifications
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Experiments       │  (experiments.py) → data/results/experiment_results.csv
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Evaluation        │  (evaluation.py) → metrics (TP/TN/FP/FN, precision, recall, F1)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Visualization      │  (visualization.py) → outputs/figures/
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Dashboard          │  (app.py) → Interactive exploration
└─────────────────────┘
```

## Simulation Environment

### Device Categories (5 categories × 5 devices = 25 total)

| Category | Packet Rate (pkt/s) | Packet Size (bytes) | Destinations | Protocols |
|----------|---------------------|---------------------|--------------|-----------|
| Smart Bulb | 0.5 – 2.0 | 50 – 150 | 1 – 2 | MQTT, CoAP |
| Smart Camera | 10 – 50 | 800 – 1500 | 1 – 3 | RTSP, HTTPS |
| Smart Lock | 0.01 – 0.1 | 30 – 100 | 1 | MQTT, CoAP |
| Smart Meter | 0.1 – 1.0 | 100 – 300 | 1 | MQTT, HTTP |
| Smart TV | 5 – 100 | 500 – 1500 | 2 – 6 | HTTPS, QUIC, RTMP |

*All ranges are synthetic simulation parameters chosen for statistical distinguishability, not measurements of real devices.*

### Observations per Device
- 100–200 observations per device
- Temporally ordered with 60-second base interval + jitter
- Total: ~3,853 normal observations

## Data Model

### Canonical Observation Schema

| Field | Type | Description |
|-------|------|-------------|
| device_id | string | Unique identifier (e.g., `bulb_01`) |
| device_type | string | One of 5 categories |
| timestamp | string | ISO 8601 format |
| packet_rate | float | Packets per second |
| packet_size | float | Bytes per packet |
| destination_count | int | Unique destinations |
| protocol | string | Communication protocol |
| connection_duration | float | Seconds |
| is_anomaly | bool | Ground truth label |
| anomaly_type | enum | `normal`, `traffic_spike`, `destination_expansion`, `combined` |

## Behavioural Fingerprinting

For each device and each numerical feature (packet_rate, packet_size, destination_count):

```
mean = average of normal observations
std = sample standard deviation of normal observations (clamped to ε if near-zero)
```

Only normal (`is_anomaly=false`) observations are used.

## Anomaly Scenarios

| Scenario | Description | Affected Features |
|----------|-------------|-------------------|
| **Traffic Spike** | 10× increase in packet rate | packet_rate |
| **Destination Expansion** | 8× increase in destination count | destination_count |
| **Combined** | 8× packet rate + 6× destinations + 1.2× packet size | packet_rate, destination_count, packet_size |

Ground truth (`is_anomaly`, `anomaly_type`) is **never** used by the detector.

## Detection Method

For each observation and each fingerprint feature:

```
Z_feature = (x - μ) / σ
```

Composite anomaly score:

```
A = (|Z_packet_rate| + |Z_destination_count| + |Z_packet_size|) / 3
```

Classification thresholds: 2.0, 3.0, 4.0

```
A ≥ threshold → anomalous
A < threshold → normal
```

## Experimental Design

| Experiment | Description |
|------------|-------------|
| **A** | Normal traffic only (baseline false positive rate) |
| **B** | Traffic spike anomalies |
| **C** | Destination expansion anomalies |
| **D** | Combined anomalies |
| **E** | Threshold study (2, 3, 4) across all anomaly types |

## Evaluation Metrics

- **Confusion Matrix:** TP, TN, FP, FN
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP)
- **Recall / Detection Rate** = TP / (TP + FN)
- **F1-Score** = 2 × Precision × Recall / (Precision + Recall)
- **False Positive Rate** = FP / (FP + TN)

Reported overall, per anomaly type, and per threshold.

## Results

### Overall Metrics (Threshold = 3.0)

| Experiment | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | FPR |
|------------|----|----|----|----|----------|-----------|--------|-----|-----|
| A: Normal | 0 | 3853 | 0 | 0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B: Traffic Spike | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| C: Destination Expansion | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| D: Combined | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| E: Threshold Study | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

### Threshold Comparison (Experiment E)

| Threshold | Accuracy | Precision | Recall | F1 | FPR |
|-----------|----------|-----------|--------|-----|-----|
| 2.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 3.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 4.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

*Perfect detection achieved across all thresholds due to controlled synthetic anomalies with large effect sizes.*

## Visualizations

The project generates 6 publication-ready figures:

1. **Packet Rate vs Time** — Normal + traffic spike for Smart Bulb
2. **Destination Count vs Time** — Normal + destination expansion for Smart TV
3. **Anomaly Score vs Time** — Composite score with threshold line
4. **Device-Wise Anomaly Count** — Bar chart by device (color-coded by type)
5. **Threshold Comparison** — 4 subplots: Accuracy, Precision, Recall, F1 across thresholds
6. **Detection Performance** — Grouped TP/FP/FN bars per experiment

Saved to `outputs/figures/`

## How to Run

### Prerequisites
- Python 3.11+
- pip

### Installation
```bash
git clone <repository>
cd IoT_A&P_Capstone
pip install -r requirements.txt
```

### Run Complete Pipeline (Phases 1–9)
```bash
python main.py --phase 9
```

### Run Individual Phases
```bash
python main.py --phase 1   # Foundation verification
python main.py --phase 2   # Normal traffic generation
python main.py --phase 3   # Fingerprinting
python main.py --phase 4   # Anomaly injection
python main.py --phase 5   # Detection
python main.py --phase 6   # Experiments
python main.py --phase 7   # Evaluation
python main.py --phase 8   # Visualization
python main.py --phase 9   # Full pipeline + reproducibility test
```

### Run Dashboard
```bash
streamlit run app.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Reproducibility
All results are deterministic given the same seed (default: 42). Re-run with:
```bash
python main.py --phase 9 --seed 42
```

## Project Structure

```
IoT_A&P_Capstone/
├── src/                    # Core Python package
│   ├── __init__.py         # Package exports
│   ├── config.py           # Centralized configuration
│   ├── schemas.py          # Data models & validation
│   ├── utils.py            # RNG, I/O, statistical helpers
│   ├── io_bridge.py        # Wokwi↔Python integration design
│   ├── simulator.py        # Phase 2: Normal traffic generation
│   ├── fingerprint.py      # Phase 3: Behavioural fingerprinting
│   ├── anomalies.py        # Phase 4: Anomaly injection
│   ├── detector.py         # Phase 5: Statistical detection
│   ├── experiments.py      # Phase 6: Experiment runner
│   ├── evaluation.py       # Phase 7: Metrics calculation
│   ├── visualization.py    # Phase 8: Matplotlib figures
│   ├── dashboard_data.py   # Dashboard data access layer
│   ├── dashboard_utils.py  # Dashboard chart helpers
│   └── __pycache__/
├── simulation/
│   └── wokwi/              # Wokwi design artifacts
│       ├── diagram.json
│       ├── wokwi.toml
│       └── firmware/
├── data/
│   ├── raw/                # normal_traffic.csv
│   ├── processed/          # fingerprints.csv, anomalous_traffic_*.csv
│   └── results/            # experiment_results.csv, detection_results_*.csv, evaluation_*.csv
├── outputs/
│   ├── figures/            # 6 visualizations (PNG)
│   └── reports/
├── tests/
│   ├── test_phase1.py      # Phase 1 validation tests
│   ├── test_phase2.py      # Phase 2 validation tests
│   ├── test_phase3.py      # Phase 3 validation tests
│   ├── test_phase4.py      # Phase 4 validation tests
│   ├── test_phase5.py      # Phase 5 validation tests
│   ├── test_phase6.py      # Phase 6 validation tests
│   └── test_phase7.py      # Phase 7 validation tests
├── app.py                  # Streamlit dashboard entry point
├── main.py                 # Pipeline entry point
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Limitations

- **Synthetic traffic only** — No real network packets captured
- **Controlled anomalies** — Injected anomalies are parametric transformations, not real attacks
- **No physical hardware** — No ESP32, no actual sensors
- **Statistical detector only** — No ML/DL, no adaptive learning
- **Single simulation study** — Results do not establish real-world IoT security performance
- **Wokwi integration** — Wokwi diagram/firmware provided as design artifacts; CLI not available in this environment; Python simulator is the reproducible engine

## Conclusion

This project demonstrates a complete, reproducible simulation pipeline for IoT behavioural fingerprinting and statistical anomaly detection. It provides a foundation for evaluating simple statistical approaches before investing in more complex ML-based detectors. The interactive dashboard enables intuitive exploration of devices, fingerprints, anomalies, and detection performance.

---

## References

1. Project proposal: `docs/IoT_Device_Behaviour_Fingerprinting_Professional_Project_Plan.pdf`
2. Phase specifications: `docs/PHASE *.md`