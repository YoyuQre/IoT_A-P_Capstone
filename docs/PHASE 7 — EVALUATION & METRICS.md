> # **PHASE 7 — EVALUATION & METRICS**

> ## **Prompt**

> Implement **Phase 7 — Detection Evaluation and Metrics**.

> Use the known simulation ground truth to quantitatively evaluate the detector.

> ### **Required confusion-matrix components**

> Calculate:

> * TP  
> * TN  
> * FP  
> * FN

> Definitions:

> * TP \= anomaly correctly detected  
> * TN \= normal observation correctly classified  
> * FP \= normal observation incorrectly flagged  
> * FN \= anomaly not detected

> ### **Required metrics**

> At minimum calculate:

> Accuracy \= (TP \+ TN) / (TP \+ TN \+ FP \+ FN)

> Also calculate useful detection metrics such as:

> * precision  
> * recall/detection rate  
> * F1-score  
> * false-positive rate

> Handle zero denominators explicitly.

> ### **Evaluation dimensions**

> Produce metrics:

> * overall  
> * per anomaly type  
> * per threshold  
> * optionally per device category where meaningful

> ### **Requirements**

> Do not modify the detector to improve metrics.

> The evaluation module must consume detector outputs and ground truth independently.

> ### **Output**

> Produce structured result tables suitable for:

> * CSV  
> * report  
> * visualization

> ### **Validation**

> Include tests with known toy confusion matrices where expected metric values can be calculated manually.

> ### **Completion criteria**

> All required experiments from Phase 6 can be evaluated automatically and produce consistent metrics.

> 

