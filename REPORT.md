# IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection

## Final Report

**Project:** IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection  
**Type:** Academic Capstone Project  
**Date:** 2026

---

## 1. Introduction

The proliferation of Internet of Things (IoT) devices has introduced new security challenges. Unlike traditional computing systems, IoT devices often operate with limited computational resources, lack regular security updates, and communicate over diverse protocols. This makes them attractive targets for attackers seeking to compromise networks, exfiltrate data, or launch distributed denial-of-service attacks.

A promising approach to IoT security is behavioural fingerprinting — establishing statistical baselines of normal device communication and flagging deviations as potential anomalies. This project investigates whether simple statistical fingerprinting (mean and standard deviation per device per feature) combined with a composite z-score anomaly metric can effectively detect anomalous IoT communication behaviour in a controlled synthetic environment.

## 2. Problem Statement

IoT devices exhibit predictable communication patterns under normal operation: periodic sensor readings, scheduled firmware checks, and regular cloud synchronisation. However, when compromised — whether by malware, misconfiguration, or physical tampering — these patterns deviate. Examples include:

- **Traffic spikes** from botnet participation or data exfiltration
- **Destination expansion** from command-and-control communication or lateral movement
- **Protocol anomalies** from exploitation attempts

Traditional signature-based detection fails against novel attacks. Statistical anomaly detection offers a complementary approach by learning normal behaviour and flagging deviations without prior knowledge of specific attack signatures.

## 3. Objectives

1. **Create a reproducible synthetic IoT environment** with 25 devices across 5 categories, each with statistically distinguishable communication profiles
2. **Generate normal traffic** with temporal ordering, device-specific variation, and category-specific statistical profiles
3. **Establish behavioural fingerprints** using only normal traffic (mean and standard deviation per device per feature)
4. **Inject controlled anomalies** representing three realistic attack scenarios with explicit ground truth
5. **Implement a statistical detector** using per-feature z-scores and a composite anomaly score
6. **Evaluate detection performance** across multiple thresholds (2, 3, 4) using confusion matrices and standard metrics
7. **Produce publication-ready visualizations** and an interactive dashboard for exploration
8. **Maintain strict academic integrity** by clearly distinguishing synthetic parameters from real-world measurements

## 4. Proposed Methodology

### 4.1 Hybrid Simulation Architecture

The project employs a hybrid simulation approach:

- **Layer A (IoT Device Simulation):** Wokwi/ESP32 virtual device demonstrating device identity, periodic telemetry, and anomaly simulation via button press
- **Layer B (Python Statistical Engine):** Main simulation engine generating 25 devices × 100–200 observations with full experimental control
- **Layer C (Data Bridge):** File-based contract (JSONL/CSV) between layers for reproducibility

### 4.2 Statistical Detection Pipeline

```
Normal Traffic → Fingerprinting (μ, σ) → Anomaly Injection → Z-Scores → Composite Score → Classification
```

For each observation and fingerprint feature:
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

## 5. Simulation Environment

### Device Categories (5 × 5 = 25 devices)

| Category | Packet Rate (pkt/s) | Packet Size (bytes) | Destinations | Protocols |
|----------|---------------------|---------------------|--------------|-----------|
| Smart Bulb | 0.5 – 2.0 | 50 – 150 | 1 – 2 | MQTT, CoAP |
| Smart Camera | 10 – 50 | 800 – 1500 | 1 – 3 | RTSP, HTTPS |
| Smart Lock | 0.01 – 0.1 | 30 – 100 | 1 | MQTT, CoAP |
| Smart Meter | 0.1 – 1.0 | 100 – 300 | 1 | MQTT, HTTP |
| Smart TV | 5 – 100 | 500 – 1500 | 2 – 6 | HTTPS, QUIC, RTMP |

*All ranges are synthetic simulation parameters chosen for statistical distinguishability, not measurements of real devices.*

### Observations
- 100–200 observations per device (mean: 154)
- 60-second base interval + ±10 second jitter
- Total normal observations: 3,853

### Wokwi Device Simulation (Layer A)

The Wokwi simulation demonstrates:
- ESP32 virtual device with DHT22 sensor
- Device identity and category configuration
- Periodic JSON telemetry via serial
- Anomaly modes triggered by button press:
  1. Traffic Spike (10× packet rate)
  2. Destination Expansion (8× destinations)
  3. Combined (both)
  4. Normal (auto-reset after 30s)
- Visual indicators: Green LED (heartbeat), Red LED (anomaly active)

## 6. Behavioural Fingerprinting

For each device and each numerical feature (packet_rate, packet_size, destination_count):

```
mean = Σ(x_i) / n
std = √(Σ(x_i - mean)² / (n-1))  [clamped to ε = 1e-8 if near-zero]
```

Only normal (`is_anomaly=false`) observations are used. Fingerprints are stored per device per feature:

```
device_fingerprints.csv: 75 entries (25 devices × 3 features)
```

Fingerprints are deterministic given fixed seed.

## 7. Anomaly Injection Scenarios

Three controlled scenarios with explicit ground truth:

| Scenario | Multiplier | Target Features | Description |
|----------|------------|-----------------|-------------|
| **Traffic Spike** | 10× packet_rate | packet_rate | Botnet/data exfiltration simulation |
| **Destination Expansion** | 8× destination_count | destination_count | C2/lateral movement simulation |
| **Combined** | 8× rate, 6× dest, 1.2× size | packet_rate, destination_count, packet_size | Multi-vector attack |

Injection parameters:
- 40% of devices (10/25) receive anomalies
- 15% of observations per selected device become anomalous
- All anomaly observations retain explicit ground truth (`is_anomaly`, `anomaly_type`)
- Normal observations preserved unchanged

Output: `anomalous_traffic_*.csv` (3,853 observations each, 234 anomalies each)

## 8. Detection Method

### Z-Score Calculation
For each observation and fingerprint feature:
```
Z_feature = (x - μ) / σ
```
Safe division with ε-clamping prevents division by zero.

### Composite Anomaly Score
```
A = (|Z_packet_rate| + |Z_destination_count| + |Z_packet_size|) / 3
```

### Thresholds
| Threshold | Sensitivity | Use Case |
|-----------|-------------|----------|
| 2.0 | High | Maximum detection, more false positives |
| 3.0 | Balanced | Recommended default |
| 4.0 | Conservative | Minimum false alarms |

Classification:
```
A ≥ threshold → anomalous
A < threshold → normal
```

### Ground Truth Isolation
The detector **never** uses `is_anomaly` or `anomaly_type` for scoring or classification. Ground truth is used only for evaluation.

## 9. Experimental Results

### Experiment Design

| Experiment | Description | Anomaly Type |
|------------|-------------|--------------|
| A | Normal traffic baseline | None |
| B | Traffic spike | Traffic Spike |
| C | Destination expansion | Destination Expansion |
| D | Combined | Combined |
| E | Threshold study | Combined |

Each experiment runs detection at all three thresholds (2.0, 3.0, 4.0).

### Overall Metrics (Threshold = 3.0)

| Experiment | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | FPR |
|------------|----|----|----|----|----------|-----------|--------|-----|-----|
| **A: Normal** | 0 | 3853 | 0 | 0 | 1.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| **B: Traffic Spike** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **C: Destination Expansion** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **D: Combined** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **E: Threshold Study** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |

### Per-Anomaly-Type Metrics (Threshold = 3.0)

| Experiment | Anomaly Type | TP | TN | FP | FN | Precision | Recall | F1 |
|------------|--------------|----|----|----|----|-----------|--------|-----|
| B | traffic_spike | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| C | destination_expansion | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| D | combined | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| E | combined | 234 | 3619 | 0 | 0 | 1.000 | 1.000 | 1.000 |

### Threshold Study (Experiment E)

| Threshold | Accuracy | Precision | Recall | F1 | FPR |
|-----------|----------|-----------|--------|-----|-----|
| 2.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 3.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 4.0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

**Key Finding:** Perfect detection (Precision=1.0, Recall=1.0, F1=1.0) achieved across all experiments and thresholds. This is expected given the controlled synthetic anomalies with large effect sizes.

### Experiment A — False Positive Baseline

Experiment A evaluates the false positive rate on purely normal traffic:
- FP = 0 at all thresholds
- FPR = 0.000
- Perfect specificity on normal traffic

## 10. Performance Evaluation

### Confusion Matrices (Threshold = 3.0)

**Experiment B — Traffic Spike:**
```
                 Predicted
              Normal  Anomaly
Actual Normal  3619      0
Actual Anomaly   0     234
```

**Experiment C — Destination Expansion:**
```
                 Predicted
              Normal  Anomaly
Actual Normal  3619      0
Actual Anomaly   0     234
```

**Experiment D — Combined:**
```
                 Predicted
              Normal  Anomaly
Actual Normal  3619      0
Actual Anomaly   0     234
```

**Experiment A — Normal (Baseline):**
```
                 Predicted
              Normal  Anomaly
Actual Normal  3853      0
Actual Anomaly   0       0
```

### Key Performance Indicators

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Detection Rate (Recall)** | 1.000 | All injected anomalies detected |
| **Precision** | 1.000 | No false alarms on anomaly observations |
| **F1-Score** | 1.000 | Perfect harmonic mean |
| **False Positive Rate** | 0.000 | Zero false alarms on normal traffic |
| **Accuracy** | 1.000 | Perfect overall classification |

### Threshold Sensitivity Analysis

Since perfect detection is achieved at all thresholds, the threshold choice does not affect results in this controlled study. In real-world deployments with noisier data and smaller effect sizes, threshold selection would involve a precision-recall trade-off.

## 11. Discussion

### Why Perfect Detection?

The controlled synthetic environment produces perfect detection because:

1. **Large effect sizes:** Anomaly multipliers (6-10×) create clear separation from normal behaviour
2. **Clean separation:** No overlapping distributions between normal and anomalous observations
3. **No noise:** Synthetic data lacks the noise, missing values, and measurement errors of real deployments
4. **Optimal features:** The three fingerprint features (packet_rate, packet_size, destination_count) perfectly capture the injected anomalies

### Real-World Applicability

While perfect detection is achieved here, real-world deployments would face:

- **Smaller effect sizes:** Real attacks may only marginally exceed normal baselines
- **Noisy environments:** Network congestion, retransmissions, and protocol overhead
- **Concept drift:** Normal behaviour evolves over time (firmware updates, configuration changes)
- **Adaptive attackers:** Adversaries may mimic normal traffic patterns
- **Resource constraints:** Edge deployment requires lightweight computation

### Fingerprinting vs. ML Approaches

This project's statistical approach offers:
- **Interpretability:** Clear per-feature contribution to anomaly score
- **Low resource requirements:** Only mean/std storage per device
- **Fast adaptation:** New devices fingerprinted in seconds
- **No training data:** Works from first normal observations

Compared to ML approaches which require:
- Large labelled datasets
- Significant computational resources
- Retraining for concept drift
- Complex deployment pipelines

## 12. Limitations

### Explicitly Stated Limitations

1. **Synthetic traffic only** — No real network packets captured
2. **Controlled anomalies** — Injected anomalies are parametric transformations, not real attacks
3. **No physical hardware** — No ESP32, no actual sensors
4. **Statistical detector only** — No ML/DL, no adaptive learning
5. **Single simulation study** — Results do not establish real-world commercial IoT security performance
6. **Wokwi integration** — Wokwi diagram/firmware provided as design artifacts; CLI not available in this environment; Python simulator is the reproducible engine

### Scope Constraints

- No claim of real-world IoT attack detection capability
- No evaluation against real malware datasets (e.g., Mirai, BASHLITE)
- No adversarial robustness testing
- No concept drift or longitudinal analysis
- No multi-site or federated deployment considerations

## 13. Conclusion

This project successfully demonstrates a complete, reproducible simulation pipeline for IoT behavioural fingerprinting and statistical anomaly detection. The key achievements are:

1. **Reproducible synthetic environment** with 25 devices across 5 categories, each with statistically distinguishable profiles
2. **Deterministic pipeline** from traffic generation through evaluation, fully reproducible with a single seed
3. **Perfect detection** in controlled conditions, validating the statistical approach for clearly separated anomalies
3. **Comprehensive evaluation framework** with confusion matrices, per-threshold metrics, and per-anomaly-type breakdown
4. **Publication-ready visualizations** (6 figures) and an interactive Streamlit dashboard for exploration
5. **Strict academic integrity** — all parameters clearly labeled synthetic, no real-world claims

The statistical fingerprinting approach proves effective for controlled anomalies with large effect sizes. Future work should evaluate performance under realistic noise, concept drift, and adaptive adversaries.

---

## 14. References

1. Project Proposal: `docs/IoT_Device_Behaviour_Fingerprinting_Professional_Project_Plan.pdf`
2. Phase Specifications: `docs/PHASE 1-10.md`
3. Wokwi Simulation: `simulation/wokwi/`
4. Implementation: `src/`
5. Test Suite: `tests/`