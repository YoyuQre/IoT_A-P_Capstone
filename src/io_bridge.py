"""
Wokwi ↔ Python Data Bridge (Design Specification)

This module defines the integration contract between the Wokwi IoT simulation layer
and the Python statistical simulation engine.

IMPORTANT: Since Wokwi CLI is not available in this environment, direct live
communication is not implemented. This module documents the DESIGN for a
file/event-based bridge that would enable integration when Wokwi is available.

Architecture:
    Wokwi (ESP32 + sensors)
        ↓ Serial/JSON output
    Exported log file (JSON Lines or CSV)
        ↓ Python ingestion adapter
    Normalized observations → Common schema (src/schemas.Observation)
        ↓ Analysis pipeline
    Fingerprinting → Anomaly Detection → Evaluation
"""

from pathlib import Path
from typing import Iterator, Optional
import json
import pandas as pd

from .schemas import Observation, NormalObservation, validate_observation
from .config import SIMULATION_WOKWI_DIR, DATA_RAW_DIR
from .utils import DeterministicRNG, get_rng


# ============================================================
# WOKWI OUTPUT FORMAT SPECIFICATION
# ============================================================
"""
Expected Wokwi serial output format (JSON Lines, one per telemetry event):

{
  "device_id": "bulb_01",
  "device_type": "Smart Bulb",
  "timestamp": 1704067200000,           # milliseconds since boot
  "temperature": 22.5,
  "humidity": 51.2,
  "packet_rate": 3,
  "packet_size": 100,
  "destination_count": 1,
  "protocol": "MQTT",
  "anomaly_mode": 0                      # 0=normal, 1=traffic_spike, 2=dest_expansion, 3=combined
}

Note: This is a SIMULATED telemetry format. The actual statistical pipeline
uses synthetic parameters defined in config.CATEGORY_PROFILES, not real
sensor readings.
"""


# ============================================================
# INGESTION ADAPTER
# ============================================================
class WokwiLogIngester:
    """
    Ingests Wokwi simulation log files and converts to canonical Observation schema.
    
    Since direct Wokwi-Python communication is not available, this expects
    an exported log file (JSON Lines or CSV) from the Wokwi simulation.
    """
    
    ANOMALY_MODE_MAP = {
        0: "normal",
        1: "traffic_spike", 
        2: "destination_expansion",
        3: "combined",
    }
    
    def __init__(self, log_path: Path, rng: Optional[DeterministicRNG] = None):
        self.log_path = Path(log_path)
        self.rng = rng or get_rng()
    
    def ingest_jsonl(self) -> list[Observation]:
        """Parse JSON Lines log file into canonical observations."""
        observations = []
        
        if not self.log_path.exists():
            raise FileNotFoundError(f"Wokwi log not found: {self.log_path}")
        
        with open(self.log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    raw = json.loads(line)
                    obs = self._convert_raw_to_observation(raw, line_num)
                    if obs:
                        observations.append(obs)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {e}")
        
        return observations
    
    def ingest_csv(self) -> list[Observation]:
        """Parse CSV log file into canonical observations."""
        df = pd.read_csv(self.log_path)
        observations = []
        
        for idx, row in df.iterrows():
            raw = row.to_dict()
            obs = self._convert_raw_to_observation(raw, idx)
            if obs:
                observations.append(obs)
        
        return observations
    
    def _convert_raw_to_observation(self, raw: dict, source_id: int) -> Optional[Observation]:
        """Convert raw Wokwi telemetry to canonical Observation."""
        # Map anomaly_mode to anomaly_type
        anomaly_mode = raw.get("anomaly_mode", 0)
        anomaly_type = self.ANOMALY_MODE_MAP.get(anomaly_mode, "normal")
        is_anomaly = anomaly_type != "normal"
        
        # Convert timestamp (ms since boot → ISO format with base date)
        # Using a fixed base date for reproducibility
        base_timestamp = pd.Timestamp("2024-01-01 00:00:00")
        ts_ms = raw.get("timestamp", 0)
        timestamp = (base_timestamp + pd.Timedelta(milliseconds=ts_ms)).isoformat()
        
        # Build canonical observation
        obs: Observation = {
            "device_id": raw.get("device_id", f"unknown_{source_id}"),
            "device_type": raw.get("device_type", "Unknown"),
            "timestamp": timestamp,
            "packet_rate": float(raw.get("packet_rate", 0)),
            "packet_size": float(raw.get("packet_size", 0)),
            "destination_count": int(raw.get("destination_count", 1)),
            "protocol": raw.get("protocol", "MQTT"),
            "connection_duration": 1.0,  # Not in Wokwi telemetry, use default
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
        }
        
        # Validate
        if validate_observation(obs):
            return obs
        else:
            print(f"Warning: Invalid observation from source {source_id}: {obs}")
            return None


# ============================================================
# EXPORT ADAPTER (Python → Wokwi format)
# ============================================================
class WokwiExportAdapter:
    """
    Exports Python synthetic observations to Wokwi-compatible format.
    
    Useful for:
    - Feeding Python-generated scenarios into Wokwi for visualization
    - Creating test log files for ingestion testing
    """
    
    @staticmethod
    def to_jsonl(observations: list[Observation], output_path: Path) -> Path:
        """Export observations as JSON Lines file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mode_map = {v: k for k, v in WokwiLogIngester.ANOMALY_MODE_MAP.items()}
        
        with open(output_path, 'w') as f:
            for obs in observations:
                raw = {
                    "device_id": obs["device_id"],
                    "device_type": obs["device_type"],
                    "timestamp": 0,  # Would need proper conversion
                    "temperature": 22.0,
                    "humidity": 50.0,
                    "packet_rate": obs["packet_rate"],
                    "packet_size": obs["packet_size"],
                    "destination_count": obs["destination_count"],
                    "protocol": obs["protocol"],
                    "anomaly_mode": mode_map.get(obs["anomaly_type"], 0),
                }
                f.write(json.dumps(raw) + "\n")
        
        return output_path
    
    @staticmethod
    def to_csv(observations: list[Observation], output_path: Path) -> Path:
        """Export observations as CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for obs in observations:
            rows.append({
                "device_id": obs["device_id"],
                "device_type": obs["device_type"],
                "timestamp": obs["timestamp"],
                "packet_rate": obs["packet_rate"],
                "packet_size": obs["packet_size"],
                "destination_count": obs["destination_count"],
                "protocol": obs["protocol"],
                "connection_duration": obs["connection_duration"],
                "is_anomaly": obs["is_anomaly"],
                "anomaly_type": obs["anomaly_type"],
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return output_path


# ============================================================
# DEMONSTRATION / TESTING
# ============================================================
def create_sample_wokwi_log(output_path: Path, num_entries: int = 100) -> Path:
    """
    Create a sample Wokwi log file for testing the ingestion pipeline.
    
    This simulates what the ESP32 firmware would output over serial.
    """
    rng = get_rng()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    devices = [
        ("bulb_01", "Smart Bulb"),
        ("cam_01", "Smart Camera"),
        ("lock_01", "Smart Lock"),
    ]
    
    with open(output_path, 'w') as f:
        for i in range(num_entries):
            device_id, device_type = rng.choice(devices)
            
            # Simulate anomaly occasionally
            anomaly_mode = 0
            if rng.random() < 0.1:
                anomaly_mode = rng.integers(1, 4)
            
            entry = {
                "device_id": device_id,
                "device_type": device_type,
                "timestamp": i * 5000,  # 5 second intervals
                "temperature": round(22.0 + rng.normal(0, 2), 1),
                "humidity": round(50.0 + rng.normal(0, 5), 1),
                "packet_rate": rng.integers(1, 10),
                "packet_size": 100,
                "destination_count": 1,
                "protocol": "MQTT",
                "anomaly_mode": int(anomaly_mode),
            }
            f.write(json.dumps(entry) + "\n")
    
    return output_path


# ============================================================
# LIMITATIONS & DOCUMENTATION
# ============================================================
"""
INTEGRATION LIMITATIONS (Current Environment):

1. NO WOKWI CLI: The Wokwi CLI tool is not installed/available.
   → Cannot run headless simulations or automate log export.

2. NO SERIAL BRIDGE: No direct serial/USB connection to Wokwi web simulator.
   → Manual export from Wokwi UI required (copy serial monitor → file).

3. PYTHON PIPELINE IS PRIMARY: The statistical experiment engine
   (src/simulator.py, etc.) is the REPRODUCIBLE, AUTOMATED pipeline.
   Wokwi serves only as a VISUALIZATION/DEMONSTRATION layer.

4. FILE-BASED CONTRACT: The integration point is a FILE (JSONL/CSV).
   This is the only reliable bridge in the current environment.

FUTURE INTEGRATION PATH (if Wokwi CLI becomes available):
- Use `wokwi-cli` to run simulation headlessly
- Capture stdout/stderr as JSONL stream
- Pipe directly into Python ingestion (subprocess + streaming parser)
- Eliminate manual file export step

For now, the Python simulator (Phase 2+) is the authoritative source
for all experimental data. Wokwi artifacts in simulation/wokwi/ are
design documentation only.
"""