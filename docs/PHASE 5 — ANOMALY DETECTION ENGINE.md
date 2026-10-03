> # **PHASE 5 — ANOMALY DETECTION ENGINE**

> ## **Prompt**

> Implement **Phase 5 — Statistical Anomaly Detection**.

> The detector must compare observations against the previously generated device-specific normal fingerprints.

> ### **Objective**

> Calculate feature-level standardized deviations and a composite anomaly score.

> For each observation:

> `z = (x - μ) / σ`

> Calculate standardized deviations for:

> * packet rate  
> * destination count  
> * packet size

> Then calculate:

> `A = (|Z_packet| + |Z_destination| + |Z_size|) / 3`

> ### **Classification**

> Support configurable thresholds:

> * 2  
> * 3  
> * 4

> Classification rule:

> anomaly\_score \>= threshold → anomalous

> anomaly\_score \< threshold  → normal

> ### **Requirements**

> * Use only the device fingerprint as the baseline.  
> * Never use `is_anomaly` or `anomaly_type` to calculate the score.  
> * Handle zero standard deviation safely.  
> * Avoid NaN and infinite scores.  
> * Preserve the ground-truth columns separately.  
> * Make threshold configurable.

> ### **Output**

> Add detection results containing:

> * device\_id  
> * timestamp  
> * anomaly\_type  
> * ground\_truth  
> * z\_packet\_rate  
> * z\_destination\_count  
> * z\_packet\_size  
> * anomaly\_score  
> * predicted\_label  
> * threshold

> ### **Tests**

> Test:

> * z-score correctness  
> * composite score correctness  
> * threshold behaviour  
> * zero-standard-deviation handling  
> * detector does not depend on ground-truth labels  
> * deterministic results

> ### **Completion criteria**

> The detector must successfully process both normal and anomalous observations and produce reproducible anomaly scores and classifications.

> 

