> # **PHASE 3 — BEHAVIOURAL FINGERPRINTING**

> ## **Prompt**

> Implement **Phase 3 — Device Behavioural Fingerprinting**.

> First inspect and validate the Phase 2 dataset. Do not modify the simulator unless a genuine Phase 2 defect prevents fingerprinting.

> ### **Objective**

> Establish a statistical behavioural fingerprint for each simulated IoT device using its normal traffic.

> For each device and selected numerical feature, calculate:

> * mean  
> * standard deviation

> At minimum fingerprint:

> * packet rate  
> * packet size  
> * destination count

> ### **Method**

> For feature `x`:

> `z = (x - mean) / std`

> Handle zero or near-zero standard deviation safely. Do not allow division-by-zero or NaN/inf values to silently enter the detector.

> ### **Requirements**

> * Fingerprints must be device-specific.  
> * Fingerprints must be generated only from normal baseline observations.  
> * Do not include injected anomalies in the baseline.  
> * Preserve device identity.  
> * Make the fingerprint reusable by the detection module.

> ### **Output**

> Produce a structured fingerprint artifact, preferably:

> data/processed/device\_fingerprints.csv

> with fields such as:

> device\_id

> feature

> mean

> std

> ### **Testing**

> Verify:

> * every device has a fingerprint  
> * every required feature has mean/std  
> * means match the source dataset  
> * standard deviations are valid  
> * zero-variance features are handled safely  
> * fingerprints are deterministic

> ### **Completion criteria**

> The phase is complete only when the fingerprint artifact can be loaded independently by the future detector and all tests pass.

> Do not implement anomaly injection yet.

> 

