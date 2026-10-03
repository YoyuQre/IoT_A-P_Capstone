> # **PHASE 4 — CONTROLLED ANOMALY INJECTION**

> ## **Prompt**

> Implement **Phase 4 — Controlled Anomaly Injection**.

> Reuse the existing normal traffic and device fingerprints. Do not regenerate the baseline unnecessarily.

> ### **Objective**

> Create controlled synthetic anomalous observations while maintaining explicit ground truth.

> Implement exactly three anomaly scenarios:

> ### **1\. Traffic Spike**

> Increase packet rate substantially for selected devices.

> The anomaly should be clearly outside the normal behaviour of the selected device while remaining numerically plausible.

> ### **2\. Destination Expansion**

> Increase destination count substantially beyond the device's normal range.

> ### **3\. Combined Anomaly**

> Simultaneously modify:

> * packet rate  
> * destination count

> ### **Ground truth**

> Every generated observation must retain:

> * `is_anomaly`  
> * `anomaly_type`

> Use:

> * `normal`  
> * `traffic_spike`  
> * `destination_expansion`  
> * `combined`

> Do not rely on the detector to determine the ground truth.

> ### **Requirements**

> * Preserve the original normal dataset.  
> * Create anomalous observations from controlled copies/transforms.  
> * Ensure anomalies can be traced back to their device and timestamp.  
> * Use deterministic randomness.  
> * Do not leak the ground-truth label into the detector's feature calculations.

> ### **Output**

> Produce a dataset such as:

> data/processed/anomalous\_traffic.csv

> or a unified experiment dataset if the architecture already supports this cleanly.

> ### **Validation**

> Verify:

> * every anomaly has a valid anomaly type  
> * normal observations remain labelled normal  
> * anomaly counts are known  
> * traffic spike observations actually have increased packet rate  
> * destination anomalies have increased destination count  
> * combined anomalies modify both features  
> * no accidental corruption of unrelated features

> ### **Completion criteria**

> All three anomaly scenarios must be reproducible and independently identifiable.

> Do not implement the final anomaly detector yet.

> 

