> # **PHASE 2 — IoT DEVICE & TRAFFIC SIMULATION**

> ## **Prompt**

> Phase 1 has been completed and verified. Now implement **Phase 2 — IoT Device & Traffic Simulation**.

> First inspect the existing Phase 1 implementation and reuse its configuration, schemas, utilities, and project structure. Do not rewrite working Phase 1 components unnecessarily.

> ### **Objective**

> Build a deterministic synthetic IoT environment containing:

> * 25 virtual devices  
> * 5 device categories  
> * approximately 5 devices per category  
> * 100–200 observations per device

> Generate normal synthetic communication behaviour for every device.

> ### **Device categories**

> Use:

> * Smart Bulb  
> * Smart Camera  
> * Smart Lock  
> * Smart Meter  
> * Smart TV

> ### **Synthetic features**

> Generate:

> * timestamp  
> * packet rate  
> * packet size  
> * destination count  
> * protocol  
> * connection duration

> Every device must have device-specific behaviour while remaining within its configured synthetic category characteristics.

> ### **Important**

> These are synthetic simulation ranges. Do not describe them as measured or representative of real commercial IoT devices.

> ### **Traffic generation requirements**

> 1. Generate a unique device ID.  
> 2. Assign each device exactly one category.  
> 3. Generate normal observations.  
> 4. Maintain temporal ordering.  
> 5. Ensure numerical features remain valid.  
> 6. Ensure categorical protocol values are valid.  
> 7. Make different device categories statistically distinguishable.  
> 8. Allow individual devices of the same category to have small variations.  
> 9. Use the centralized random seed.  
> 10. Ensure running the simulator twice with the same seed produces identical output.

> ### **Output**

> Produce at minimum:

> data/raw/normal\_traffic.csv

> Also expose the simulator through a reusable Python function/module rather than putting all logic in `main.py`.

> ### **Validation**

> Add tests verifying:

> * exactly 25 devices  
> * all five categories are represented  
> * approximately five devices per category  
> * expected number of observations  
> * required columns exist  
> * no invalid numeric values  
> * valid protocols  
> * timestamps are valid  
> * deterministic generation

> Run the simulator and inspect the resulting dataset statistically.

> ### **Completion criteria**

> Phase 2 is complete when:

> * normal synthetic traffic is generated successfully  
> * CSV output exists  
> * all 25 devices are represented  
> * all five categories are represented  
> * data passes validation  
> * deterministic reproduction works  
> * tests pass  
> * basic summary statistics are generated

> Do not implement fingerprinting or anomaly detection yet.

> 

