# **PHASE 1 — PROJECT FOUNDATION**

## **Prompt**

> You are working as a senior Python engineer and cybersecurity/IoT simulation engineer.

> We are beginning **Phase 1** of the project:

> **IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection**

> The project is a simulation-based IoT cybersecurity study. It must remain moderate in scope and must be completely executable without physical IoT hardware.

> ### **Project objective**

> Build a reproducible simulation pipeline that:

> `Synthetic IoT Devices → Normal Traffic → Behavioural Fingerprint → Injected Anomaly → Statistical Detection → Evaluation → Visualization`

> The project will eventually contain:

> * 25 simulated IoT devices  
> * 5 device categories  
> * Synthetic network observations  
> * Device-specific behavioural fingerprints  
> * 3 anomaly scenarios  
> * Statistical anomaly scoring  
> * Threshold experiments  
> * Quantitative evaluation  
> * Visualizations

> ### **Phase 1 objectives**

> Establish the complete project foundation before implementing the actual simulator.

> ### **Requirements**

> 1. Inspect the current project directory before creating or modifying files.  
> 2. Do not assume files, folders, dependencies, or functionality that do not exist.  
> 3. Create a clean modular Python project structure.  
> 4. Define centralized configuration/constants for:  
>    * device categories  
>    * number of devices  
>    * simulation parameters  
>    * random seed  
>    * feature names  
>    * anomaly types  
>    * detection thresholds  
> 5. Establish deterministic random-number generation so experiments can be reproduced.  
> 6. Define the canonical synthetic observation schema.  
> 7. Define the expected output/data directories.  
> 8. Add appropriate package initialization files where required.  
> 9. Add a requirements/dependency definition only if appropriate for the existing project structure.  
> 10. Use Python with NumPy, Pandas, and Matplotlib as the primary stack.  
> 11. Do not introduce deep learning, reinforcement learning, cloud services, databases, real packet capture, or physical IoT hardware.  
> 12. Keep the architecture modular enough that three people can independently work on:  
>     * simulation  
>     * detection  
>     * analysis/visualization

> ### **Suggested architecture**

project/  
├── src/  
│   ├── \_\_init\_\_.py  
│   ├── config.py  
│   ├── schemas.py  
│   ├── utils.py  
│   ├── simulator.py  
│   ├── fingerprint.py  
│   ├── anomalies.py  
│   ├── detector.py  
│   ├── evaluation.py  
│   ├── experiments.py  
│   └── visualization.py  
│  
├── data/  
│   ├── raw/  
│   ├── processed/  
│   └── results/  
│  
├── outputs/  
│   ├── figures/  
│   └── reports/  
│  
├── tests/  
├── main.py  
├── requirements.txt  
> └── README.md

> You may modify this structure if a better clean architecture is justified, but do not add unnecessary complexity.

> ### **Canonical observation fields**

> The system should eventually support:

> * `device_id`  
> * `device_type`  
> * `timestamp`  
> * `packet_rate`  
> * `packet_size`  
> * `destination_count`  
> * `protocol`  
> * `connection_duration`  
> * `is_anomaly`  
> * `anomaly_type`

> Keep the schema extensible without overengineering it.

> ### **Device categories**

> Define the initial five categories:

> * Smart Bulb  
> * Smart Camera  
> * Smart Lock  
> * Smart Meter  
> * Smart TV

> These are **synthetic simulation categories**, not measured characteristics of real commercial devices.

> ### **Important constraints**

> * Do not generate actual network traffic.  
> * Do not use real device telemetry.  
> * Do not claim that the chosen ranges represent real IoT devices.  
> * Do not implement the detector yet.  
> * Do not implement anomaly injection yet.  
> * Do not create a dashboard yet.

> ### **Testing**

> Add basic tests for:

> * configuration loading  
> * schema validity  
> * deterministic random seed  
> * device configuration availability  
> * required directories/configuration

> ### **Completion criteria**

> Phase 1 is complete only when:

> * project structure exists  
> * configuration is centralized  
> * schemas are defined  
> * deterministic randomness is established  
> * required directories exist  
> * imports work  
> * basic tests pass  
> * README explains the architecture and Phase 1 foundation  
> * no Phase 2+ functionality has been prematurely implemented

> At the end, provide:

> 1. Files created/modified  
> 2. Architecture summary  
> 3. Tests executed  
> 4. Test results  
> 5. Any assumptions made  
> 6. Anything that must be addressed in Phase 2

> Do not proceed to Phase 2 unless Phase 1 completion criteria are satisfied.

