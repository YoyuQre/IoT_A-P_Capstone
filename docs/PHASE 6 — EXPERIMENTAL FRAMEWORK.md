> # **PHASE 6 — EXPERIMENTAL FRAMEWORK**

> ## **Prompt**

> Implement **Phase 6 — Experimental Framework**.

> The purpose of this phase is to run the predefined simulation experiments systematically rather than manually.

> ### **Required experiments**

> **Experiment A — Normal**

> * 25 devices  
> * no injected anomaly  
> * establish baseline false-positive behaviour

> **Experiment B — Traffic Spike**

> * selected devices  
> * increased packet rate  
> * evaluate detection

> **Experiment C — Destination Expansion**

> * selected devices  
> * increased destination count  
> * evaluate detection

> **Experiment D — Combined**

> * selected devices  
> * increased packet rate and destination count  
> * evaluate multi-feature detection

> **Experiment E — Threshold Study**

> Compare:

> * threshold \= 2  
> * threshold \= 3  
> * threshold \= 4

> ### **Requirements**

> Create a reusable experiment runner.

> Every experiment must record:

> * experiment name  
> * configuration  
> * threshold  
> * number of observations  
> * anomaly count  
> * detection results

> Do not manually edit datasets between experiments.

> ### **Reproducibility**

> Every experiment must be reproducible using the project seed/configuration.

> ### **Outputs**

> Store machine-readable experiment results under:

> data/results/

> ### **Completion criteria**

> All experiments execute successfully from a single command/function and produce structured results ready for Phase 7\.

> 

