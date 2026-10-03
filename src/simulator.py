"""
Phase 2: IoT Device & Normal Traffic Simulation

Generates deterministic synthetic normal communication behaviour for 25 IoT devices
across 5 categories. Each device has category-specific statistical profiles with
individual variation.

Output: data/raw/normal_traffic.csv
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from .config import (
    MASTER_SEED,
    DEVICE_CATEGORIES,
    NUM_DEVICES,
    DEVICES_PER_CATEGORY,
    DEVICE_ID_PREFIX,
    CATEGORY_PROFILES,
    OBSERVATIONS_PER_DEVICE_MIN,
    OBSERVATIONS_PER_DEVICE_MAX,
    TIMESTAMP_START,
    TIMESTAMP_INTERVAL_SECONDS,
    DATA_RAW_DIR,
    NORMAL_TRAFFIC_CSV,
)
from .schemas import (
    NormalObservation,
    validate_observation,
    VALID_PROTOCOLS,
    VALID_DEVICE_TYPES,
)
from .utils import (
    DeterministicRNG,
    get_rng,
    generate_device_ids,
    generate_timestamps,
    select_protocol,
    save_csv,
    safe_mean_std,
)


# ============================================================
# DEVICE MODEL
# ============================================================
class VirtualDevice:
    """Represents a single virtual IoT device with its configuration."""
    
    def __init__(
        self,
        device_id: str,
        device_type: str,
        profile: dict,
        rng: DeterministicRNG,
        device_index: int,
    ):
        self.device_id = device_id
        self.device_type = device_type
        self.profile = profile
        self.rng = rng
        self.device_index = device_index  # 0-based index within category
        
        # Derive device-specific parameters from category profile + individual variation
        self._derive_device_parameters()
    
    def _derive_device_parameters(self) -> None:
        """Calculate device-specific parameter ranges from category profile."""
        # Base ranges from category profile
        pr_low, pr_high = self.profile["packet_rate"]
        ps_low, ps_high = self.profile["packet_size"]
        dc_low, dc_high = self.profile["destination_count"]
        cd_low, cd_high = self.profile["connection_duration"]
        protocols = self.profile["protocol"]
        
        # Individual variation: ±20% around category mean for numerical params
        # Use device_index to create deterministic but distinct variation
        variation_seed = hash((self.device_id, "variation")) % 1000
        var_rng = DeterministicRNG(seed=variation_seed)
        
        # Variation factor: 0.8 to 1.2 (±20%)
        var_factor = var_rng.uniform(0.8, 1.2)
        
        # Apply variation to category means
        pr_mean = (pr_low + pr_high) / 2
        ps_mean = (ps_low + ps_high) / 2
        dc_mean = (dc_low + dc_high) / 2
        cd_mean = (cd_low + cd_high) / 2
        
        # Device-specific ranges (narrower than category, centered on varied mean)
        spread = 0.15  # ±15% around device mean
        
        self.packet_rate_range = (
            max(0.01, pr_mean * var_factor * (1 - spread)),
            pr_mean * var_factor * (1 + spread),
        )
        self.packet_size_range = (
            max(10, ps_mean * var_factor * (1 - spread)),
            ps_mean * var_factor * (1 + spread),
        )
        self.dest_count_range = (
            max(1, int(dc_mean * var_factor * (1 - spread))),
            max(1, int(dc_mean * var_factor * (1 + spread))),
        )
        self.conn_duration_range = (
            max(0.01, cd_mean * var_factor * (1 - spread)),
            cd_mean * var_factor * (1 + spread),
        )
        self.protocols = protocols
        
        # Protocol selection weights (prefer first protocol in list)
        self.protocol_weights = [0.7] + [0.3 / (len(protocols) - 1)] * (len(protocols) - 1) \
            if len(protocols) > 1 else [1.0]


# ============================================================
# TRAFFIC GENERATOR
# ============================================================
class NormalTrafficSimulator:
    """
    Generates deterministic normal traffic for all virtual devices.
    
    Usage:
        simulator = NormalTrafficSimulator(seed=42)
        df = simulator.generate()
        simulator.save(df)
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        observations_per_device: Optional[tuple[int, int]] = None,
    ):
        self.seed = seed if seed is not None else MASTER_SEED
        self.rng = DeterministicRNG(seed=self.seed)
        
        # Override observation range if provided
        self.obs_min, self.obs_max = observations_per_device or (
            OBSERVATIONS_PER_DEVICE_MIN,
            OBSERVATIONS_PER_DEVICE_MAX,
        )
        
        # Create devices
        self.devices = self._create_devices()
    
    def _create_devices(self) -> list[VirtualDevice]:
        """Create all 25 virtual devices with deterministic IDs."""
        device_configs = generate_device_ids(
            DEVICE_CATEGORIES,
            DEVICES_PER_CATEGORY,
            DEVICE_ID_PREFIX,
        )
        
        devices = []
        for idx, config in enumerate(device_configs):
            device_type = config["device_type"]
            profile = CATEGORY_PROFILES[device_type]
            
            # Create device-specific RNG (derived from master seed + device index)
            device_seed = hash((self.seed, config["device_id"])) % (2**32)
            device_rng = DeterministicRNG(seed=device_seed)
            
            device = VirtualDevice(
                device_id=config["device_id"],
                device_type=device_type,
                profile=profile,
                rng=device_rng,
                device_index=idx % DEVICES_PER_CATEGORY,
            )
            devices.append(device)
        
        return devices
    
    def generate_device_observations(
        self,
        device: VirtualDevice,
        num_observations: int,
    ) -> list[NormalObservation]:
        """Generate normal observations for a single device."""
        observations = []
        
        # Generate timestamps
        timestamps = generate_timestamps(
            start=TIMESTAMP_START,
            count=num_observations,
            interval_seconds=TIMESTAMP_INTERVAL_SECONDS,
            rng=device.rng,
            jitter_seconds=10,  # ±10 seconds jitter
        )
        
        for i, ts in enumerate(timestamps):
            # Sample from device-specific ranges
            packet_rate = device.rng.uniform(*device.packet_rate_range)
            packet_size = device.rng.uniform(*device.packet_size_range)
            dest_count = device.rng.integers(
                device.dest_count_range[0],
                device.dest_count_range[1] + 1,
            )
            conn_duration = device.rng.uniform(*device.conn_duration_range)
            protocol = select_protocol(
                device.protocols,
                rng=device.rng,
                weights=device.protocol_weights,
            )
            
            obs: NormalObservation = {
                "device_id": device.device_id,
                "device_type": device.device_type,
                "timestamp": ts,
                "packet_rate": round(packet_rate, 4),
                "packet_size": round(packet_size, 2),
                "destination_count": int(dest_count),
                "protocol": protocol,
                "connection_duration": round(conn_duration, 4),
            }
            
            # Validate (should always pass for generated data)
            if not validate_observation({**obs, "is_anomaly": False, "anomaly_type": "normal"}):
                raise ValueError(f"Generated invalid observation: {obs}")
            
            observations.append(obs)
        
        return observations
    
    def generate(self) -> pd.DataFrame:
        """
        Generate normal traffic for all devices.
        
        Returns:
            DataFrame with columns matching NormalObservation schema
        """
        all_observations = []
        
        for device in self.devices:
            # Determine number of observations for this device
            num_obs = self.rng.integers(self.obs_min, self.obs_max + 1)
            
            device_obs = self.generate_device_observations(device, num_obs)
            all_observations.extend(device_obs)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_observations)
        
        # Ensure column order matches schema
        column_order = [
            "device_id",
            "device_type",
            "timestamp",
            "packet_rate",
            "packet_size",
            "destination_count",
            "protocol",
            "connection_duration",
        ]
        df = df[column_order]
        
        # Sort by device_id then timestamp for consistent ordering
        df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
        
        return df
    
    def save(self, df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
        """Save generated traffic to CSV."""
        if output_path is None:
            output_path = DATA_RAW_DIR / NORMAL_TRAFFIC_CSV
        return save_csv(df, output_path.parent, output_path.name)
    
    def get_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics per device."""
        stats = df.groupby(["device_id", "device_type"]).agg(
            num_observations=("packet_rate", "count"),
            packet_rate_mean=("packet_rate", "mean"),
            packet_rate_std=("packet_rate", "std"),
            packet_size_mean=("packet_size", "mean"),
            packet_size_std=("packet_size", "std"),
            dest_count_mean=("destination_count", "mean"),
            dest_count_std=("destination_count", "std"),
            conn_duration_mean=("connection_duration", "mean"),
            conn_duration_std=("connection_duration", "std"),
        ).reset_index()
        
        return stats.round(4)


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================
def generate_normal_traffic(
    seed: Optional[int] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Convenience function to generate and save normal traffic.
    
    Args:
        seed: Random seed for reproducibility (default: MASTER_SEED)
        output_path: Custom output path (default: data/raw/normal_traffic.csv)
    
    Returns:
        DataFrame with normal traffic observations
    """
    simulator = NormalTrafficSimulator(seed=seed)
    df = simulator.generate()
    simulator.save(df, output_path)
    return df


# ============================================================
# VALIDATION
# ============================================================
def validate_normal_traffic(df: pd.DataFrame) -> dict:
    """
    Validate generated normal traffic dataset.
    
    Returns:
        Dict with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # Check required columns
    required_cols = [
        "device_id", "device_type", "timestamp", "packet_rate",
        "packet_size", "destination_count", "protocol", "connection_duration"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        results["valid"] = False
        results["errors"].append(f"Missing columns: {missing}")
        # Return early if required columns are missing
        return results
    
    # Check device count
    unique_devices = df["device_id"].nunique()
    if unique_devices != NUM_DEVICES:
        results["valid"] = False
        results["errors"].append(f"Expected {NUM_DEVICES} devices, got {unique_devices}")
    
    # Check categories
    unique_categories = df["device_type"].nunique()
    if unique_categories != len(DEVICE_CATEGORIES):
        results["warnings"].append(f"Expected {len(DEVICE_CATEGORIES)} categories, got {unique_categories}")
    
    for cat in DEVICE_CATEGORIES:
        if cat not in df["device_type"].values:
            results["warnings"].append(f"Missing category: {cat}")
    
    # Check observations per device
    obs_per_device = df.groupby("device_id").size()
    if (obs_per_device < OBSERVATIONS_PER_DEVICE_MIN).any():
        results["warnings"].append("Some devices have fewer than minimum observations")
    if (obs_per_device > OBSERVATIONS_PER_DEVICE_MAX).any():
        results["warnings"].append("Some devices have more than maximum observations")
    
    # Check numeric validity
    numeric_cols = ["packet_rate", "packet_size", "destination_count", "connection_duration"]
    for col in numeric_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                results["valid"] = False
                results["errors"].append(f"Negative values in {col}")
            if df[col].isna().any():
                results["valid"] = False
                results["errors"].append(f"NaN values in {col}")
    
    # Check protocols
    invalid_protocols = set(df["protocol"].unique()) - set(VALID_PROTOCOLS)
    if invalid_protocols:
        results["valid"] = False
        results["errors"].append(f"Invalid protocols: {invalid_protocols}")
    
    # Check device types
    invalid_types = set(df["device_type"].unique()) - set(VALID_DEVICE_TYPES)
    if invalid_types:
        results["valid"] = False
        results["errors"].append(f"Invalid device types: {invalid_types}")
    
    # Check timestamps
    try:
        pd.to_datetime(df["timestamp"])
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Invalid timestamps: {e}")
    
    # Summary stats
    results["stats"] = {
        "total_observations": len(df),
        "unique_devices": unique_devices,
        "unique_categories": unique_categories,
        "observations_per_device": obs_per_device.describe().to_dict(),
        "categories": df["device_type"].value_counts().to_dict(),
    }
    
    return results


if __name__ == "__main__":
    # Direct execution for testing
    print("Generating normal traffic...")
    df = generate_normal_traffic()
    print(f"Generated {len(df)} observations for {df['device_id'].nunique()} devices")
    
    validation = validate_normal_traffic(df)
    print(f"Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation["errors"]:
        print(f"Errors: {validation['errors']}")
    if validation["warnings"]:
        print(f"Warnings: {validation['warnings']}")
    print(f"Stats: {validation['stats']}")