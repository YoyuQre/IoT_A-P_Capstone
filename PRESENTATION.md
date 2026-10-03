# IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection

## Academic Capstone Project — Presentation Structure

---

## Slide 1: Title

**IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection**

*Academic Capstone Project*  
*IoT + Cybersecurity + Statistical Simulation*

---

## Slide 2: Problem

**IoT Security Challenge**

- 25+ billion IoT devices projected by 2030
- Devices often lack: security updates, encryption, authentication
- Traditional signature-based detection fails against novel attacks
- Need: behavioural anomaly detection that learns normal patterns

**Key Question:** Can simple statistical fingerprinting detect anomalous IoT behaviour without labelled attack data?

---

## Slide 3: Motivation

**Why Behavioural Fingerprinting?**

| Traditional Approach | Limitation |
|---------------------|------------|
| Signature-based (IDS/IPS) | Only known attacks |
| Rule-based firewalls | Manual maintenance |
| ML/DL classifiers | Large labelled datasets, retraining, opacity |

**Our Approach:** Statistical fingerprinting (mean/std per device) + composite z-score → interpretable, lightweight, no training data needed

---

## Slide 4: Objectives

1. Build reproducible synthetic IoT environment (25 devices, 5 categories)
2. Generate statistically distinguishable normal traffic per category
3. Compute device-specific behavioural fingerprints (μ, σ per feature)
4. Inject 3 controlled anomaly types with explicit ground truth
5. Implement statistical detector (z-scores + composite score)
6. Evaluate across thresholds 2, 3, 4 with full metrics
7. Produce visualizations + interactive dashboard

---

## Slide 5: Proposed Architecture

```
IoT Simulation (Wokwi/ESP32)
         ↓
Synthetic Traffic Generation (Python)
         ↓
Behavioural Fingerprinting (μ, σ per device/feature)
         ↓
Controlled Anomaly Injection (3 scenarios)
         ↓
Statistical Detection (Z-scores → Composite Score A)
         ↓
Threshold Classification (2, 3, 4)
         ↓
Evaluation (TP/TN/FP/FN, Precision, Recall, F1)
         ↓
Visualization + Interactive Dashboard
```

**Hybrid Design:** Wokwi demonstrates device layer; Python engine runs full experiments

---

## Slide 6: Synthetic IoT Environment

**25 Virtual Devices × 5 Categories**

| Category | Packet Rate | Packet Size | Destinations | Protocols |
|----------|-------------|-------------|--------------|-----------|
| Smart Bulb | 0.5–2.0 pkt/s | 50–150 B | 1–2 | MQTT, CoAP |
| Smart Camera | 10–50 pkt/s | 800–1500 B | 1–3 | RTSP, HTTPS |
| Smart Lock | 0.01–0.1 pkt/s | 30–100 B | 1 | MQTT, CoAP |
| Smart Meter | 0.1–1.0 pkt/s | 100–300 B | 1 | MQTT, HTTP |
| Smart TV | 5–100 pkt/s | 500–1500 B | 2–6 | HTTPS, QUIC, RTMP |

*All ranges are **synthetic simulation parameters** — not measurements of real devices*

---

## Slide 7: Behavioural Fingerprinting

**Per Device, Per Feature (packet_rate, packet_size, destination_count):**

```
μ = Σ(x_i) / n
σ = √(Σ(x_i - μ)² / (n-1))   [clamped to ε = 1e-8 if near-zero]
```

- **Only normal traffic used** (is_anomaly = false)
- 25 devices × 3 features = 75 fingerprint entries
- Deterministic with fixed seed (42)
- Stored as `device_fingerprints.csv`

---

## Slide 8: Anomaly Scenarios

| Scenario | Multiplier | Target | Attack Simulation |
|----------|------------|--------|-------------------|
| **Traffic Spike** | 10× packet_rate | packet_rate | Botnet / data exfiltration |
| **Destination Expansion** | 8× destination_count | destination_count | C2 / lateral movement |
| **Combined** | 8× rate, 6× dest, 1.2× size | All three | Multi-vector attack |

- 40% devices (10/25) receive anomalies
- 15% observations per device become anomalous
- 234 anomalies per scenario × 3 = 702 total
- Explicit ground truth preserved (`is_anomaly`, `anomaly_type`)

---

## Slide 9: Detection Algorithm

**Z-Score per Feature:**
```
Z = (x - μ) / σ
```

**Composite Anomaly Score:**
```
A = (|Z_packet_rate| + |Z_destination_count| + |Z_packet_size|) / 3
```

**Thresholds:**
| Threshold | Sensitivity | Use Case |
|-----------|-------------|----------|
| 2.0 | High | Maximum detection |
| 3.0 | **Balanced** | Default (recommended) |
| 4.0 | Conservative | Minimum false alarms |

**Classification:** `A ≥ threshold → anomalous`

**Critical:** Ground truth NEVER used in detection — only for evaluation

---

## Slide 10: Experimental Setup

| Experiment | Description | Anomaly Type |
|------------|-------------|--------------|
| **A** | Normal baseline | None |
| **B** | Traffic spike | Traffic Spike |
| **C** | Destination expansion | Destination Expansion |
| **D** | Combined | Combined |
| **E** | Threshold study | Combined |

- 5 experiments × 3 thresholds = 15 runs
- Each experiment: 3,853 observations × 3 thresholds
- Total: 57,795 detection results

---

## Slide 11: Results

### Overall Metrics (Threshold = 3.0)

| Experiment | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | FPR |
|------------|----|----|----|----|----------|-----------|--------|-----|-----|
| **A: Normal** | 0 | 3853 | 0 | 0 | 1.000 | – | – | – | **0.000** |
| **B: Traffic Spike** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **C: Dest. Expansion** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **D: Combined** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |
| **E: Threshold Study** | 234 | 3619 | 0 | 0 | **1.000** | **1.000** | **1.000** | **1.000** | 0.000 |

---

## Slide 12: Key Findings

**Perfect Detection Achieved:**
- **Precision = 1.000** — No false alarms on anomaly observations
- **Recall = 1.000** — All injected anomalies detected
- **F1-Score = 1.000** — Perfect harmonic mean
- **FPR = 0.000** — Zero false alarms on normal traffic
- **Perfect at ALL thresholds** (2.0, 3.0, 4.0)

**Why Perfect Detection?**
1. Large effect sizes (6-10× multipliers)
2. Clean separation (no overlapping distributions)
3. No noise (synthetic environment)
4. Optimal features capture injected anomalies perfectly

---

## Slide 13: Visualizations

**6 Publication-Ready Figures:**

1. **Packet Rate vs Time** — Normal + traffic spike (Smart Bulb)
2. **Destination Count vs Time** — Normal + destination expansion (Smart TV)
3. **Anomaly Score vs Time** — Composite score with threshold line
4. **Device-Wise Anomaly Count** — Bar chart by device type
5. **Threshold Comparison** — 4 metrics × 5 experiments × 3 thresholds
4. **Detection Performance** — TP/FP/FN grouped bars

---

## Slide 14: Interactive Dashboard

**Streamlit-based Interactive Exploration**

| Page | Function |
|------|----------|
| **Overview** | KPIs, distributions, pipeline diagram |
| **Device Explorer** | Per-device time series (rate, size, dest, score) |
| **Behavioural Fingerprint** | μ ± σ bounds vs observations |
| **Anomaly Detection** | Z-scores, anomaly score vs time, TP/FP/FN |
| **Experiment Analysis** | 5 experiments comparison table |
| **Threshold Analysis** | Threshold vs metrics (precision/recall/F1/FPR) |
| **Device Categories** | Category-wise stats & comparisons |
| **Evaluation** | Confusion matrix, KPIs, per-type metrics |
| **Raw Data Explorer** | Filterable data tables with downloads |

**Run:** `streamlit run app.py`

---

## Slide 15: Limitations

**Explicitly Stated:**

- ✅ Synthetic traffic only — no real packets
- ✅ Controlled anomalies — parametric, not real attacks
- ✅ No physical hardware — no ESP32, no sensors
- ✅ Statistical detector only — no ML/DL
- ✅ Single simulation — no real-world performance claims
- ✅ Wokwi as design artifact only — Python is reproducible engine

**No Claims Of:**
- Real-world IoT attack detection
- Evaluation against real malware (Mirai, BASHLITE)
- Adversarial robustness
- Production deployment readiness

---

## Slide 16: Conclusion

**Achievements:**

1. ✅ Reproducible synthetic IoT environment (25 devices, 5 categories)
2. ✅ Deterministic pipeline (single seed = identical results)
3. ✅ Perfect detection in controlled conditions (Precision=1.0, Recall=1.0)
4. ✅ Comprehensive evaluation (confusion matrices, per-threshold, per-type)
5. ✅ 6 publication-ready visualizations + interactive dashboard
6. ✅ Strict academic integrity — all synthetic, no real-world claims

**Future Work:**
- Realistic noise & concept drift
- Adaptive adversaries / adversarial robustness
- Real malware dataset evaluation (Mirai, BASHLITE)
- Edge deployment & resource profiling

---

## Slide 17: Reproducibility

**Complete Pipeline: One Command**

```bash
# Install
pip install -r requirements.txt

# Run full pipeline (Phases 1-9)
python main.py --phase 9

# Launch dashboard
streamlit run app.py
```

**All outputs deterministic with seed=42**

```
data/raw/normal_traffic.csv           → 3,853 observations
data/processed/device_fingerprints.csv → 75 entries
data/processed/anomalous_traffic_*.csv → 3 files × 3,853 rows
data/results/experiment_results.csv   → 57,795 rows
data/results/evaluation_*.csv         → 5 metrics files
outputs/figures/01-06_*.png           → 6 PNG figures
```

---

## Slide 18: Thank You

**IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection**

*A complete, reproducible simulation pipeline for statistical anomaly detection*

**Questions & Discussion**

---

## Appendix: Technical Details

- **Language:** Python 3.11+
- **Core:** NumPy, Pandas, Matplotlib, Streamlit
- **Tests:** 132 tests across 7 phases (all passing)
- **Wokwi Layer:** ESP32 + DHT22 + LEDs + Button (design artifact)
- **Deterministic:** `numpy.random.Generator` with fixed seed
- **Academic Integrity:** All synthetic, no fabricated results