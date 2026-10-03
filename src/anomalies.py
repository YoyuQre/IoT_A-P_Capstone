"""
Phase 4: Controlled Anomaly Injection

Creates controlled synthetic anomalous observations while maintaining explicit ground truth.
Implements three anomaly scenarios:
1. Traffic Spike - increase packet rate
2. Destination Expansion - increase destination count
3. Combined - increase both packet rate and destination count

Output: data/processed/anomalous_traffic.csv
"""

from pathlib import Path
from typing import Optional, Literal
import pandas as pd
import numpy as np

from .config import (
    MASTER_SEED,
    ANOMALY_TYPES,
    ANOMALY_MULTIPLIERS,
    ANOMALY_DEVICE_FRACTION,
    ANOMALY_OBSERVATION_FRACTION,
    DATA_PROCESSED_DIR,
    ANOMALOUS_TRAFFIC_CSV,
    FINGERPRINT_FEATURES,
)
from .schemas import Observation, validate_observation
from .utils import (
    DeterministicRNG,
    get_rng,
    load_csv,
    save_csv,
    generate_device_ids,
)
from .simulator import generate_normal_traffic


# ============================================================
# ANOMALY INJECTOR
# ============================================================
class AnomalyInjector:
    """
    Injects controlled anomalies into normal traffic data.
    
    Maintains explicit ground truth labels for all observations.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        anomaly_device_fraction: float = ANOMALY_DEVICE_FRACTION,
        anomaly_observation_fraction: float = ANOMALY_OBSERVATION_FRACTION,
        multipliers: Optional[dict] = None,
    ):
        self.seed = seed if seed is not None else MASTER_SEED
        self.rng = DeterministicRNG(seed=self.seed)
        self.anomaly_device_fraction = anomaly_device_fraction
        self.anomaly_observation_fraction = anomaly_observation_fraction
        self.multipliers = multipliers or ANOMALY_MULTIPLIERS
    
    def inject_anomalies(
        self,
        normal_df: pd.DataFrame,
        anomaly_type: Literal["traffic_spike", "destination_expansion", "combined"],
    ) -> pd.DataFrame:
        """
        Inject anomalies of specified type into normal traffic.
        
        Args:
            normal_df: DataFrame with normal observations (no is_anomaly column needed)
            anomaly_type: Type of anomaly to inject
        
        Returns:
            DataFrame with mixed normal + anomalous observations with ground truth
        """
        if anomaly_type not in self.multipliers:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        # Ensure normal_df has is_anomaly and anomaly_type columns
        df = normal_df.copy()
        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = False
        if "anomaly_type" not in df.columns:
            df["anomaly_type"] = "normal"
        
        # Select devices to inject anomalies into
        unique_devices = df["device_id"].unique()
        num_anomaly_devices = max(1, int(len(unique_devices) * self.anomaly_device_fraction))
        
        # Deterministic selection of anomaly devices
        anomaly_device_ids = self.rng.choice(
            unique_devices, size=num_anomaly_devices, replace=False
        )
        
        # Get multiplier for this anomaly type
        multiplier = self.multipliers[anomaly_type]
        
        # Create anomalous observations
        anomalous_rows = []
        
        for device_id in anomaly_device_ids:
            device_df = df[df["device_id"] == device_id].copy()
            num_obs = len(device_df)
            num_anomaly_obs = max(1, int(num_obs * self.anomaly_observation_fraction))
            
            # Select random observations to make anomalous
            anomaly_indices = self.rng.choice(
                num_obs, size=num_anomaly_obs, replace=False
            )
            
            for idx in anomaly_indices:
                row = device_df.iloc[idx].copy()
                
                # Apply multipliers to selected features
                if "packet_rate" in multiplier:
                    row["packet_rate"] = row["packet_rate"] * multiplier["packet_rate"]
                if "destination_count" in multiplier:
                    row["destination_count"] = max(1, int(row["destination_count"] * multiplier["destination_count"]))
                if "packet_size" in multiplier:
                    row["packet_size"] = row["packet_size"] * multiplier["packet_size"]
                
                # Set ground truth labels
                row["is_anomaly"] = True
                row["anomaly_type"] = anomaly_type
                
                anomalous_rows.append(row)
        
        # Combine normal and anomalous
        if anomalous_rows:
            anomaly_df = pd.DataFrame(anomalous_rows)
            # Remove the anomalous rows from normal df
            normal_df_clean = df.drop(index=[r.name for r in anomalous_rows])
            # Combine
            result_df = pd.concat([normal_df_clean, anomaly_df], ignore_index=True)
        else:
            result_df = df
        
        # Sort by device_id then timestamp
        result_df = result_df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
        
        return result_df
    
    def inject_all_anomaly_types(
        self,
        normal_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """
        Inject all anomaly types separately for comparison experiments.
        
        Returns:
            Dict mapping anomaly_type -> DataFrame with that anomaly type injected
        """
        results = {}
        for anomaly_type in ["traffic_spike", "destination_expansion", "combined"]:
            results[anomaly_type] = self.inject_anomalies(normal_df, anomaly_type)
        return results


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================
def inject_anomalies(
    normal_traffic_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    anomaly_type: Literal["traffic_spike", "destination_expansion", "combined"] = "traffic_spike",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience function to load normal traffic, inject anomalies, and save.
    
    Args:
        normal_traffic_path: Path to normal_traffic.csv
        output_path: Output path for anomalous traffic
        anomaly_type: Type of anomaly to inject
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with mixed normal + anomalous observations
    """
    if normal_traffic_path is None:
        normal_traffic_path = DATA_PROCESSED_DIR.parent / "raw" / "normal_traffic.csv"
    
    # Load normal traffic
    df = load_csv(normal_traffic_path.parent, normal_traffic_path.name)
    
    # Inject anomalies
    injector = AnomalyInjector(seed=seed)
    result_df = injector.inject_anomalies(df, anomaly_type)
    
    # Save
    if output_path is None:
        output_path = DATA_PROCESSED_DIR / f"anomalous_traffic_{anomaly_type}.csv"
    save_csv(result_df, output_path.parent, output_path.name)
    
    return result_df


def inject_all_anomaly_scenarios(
    normal_traffic_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """
    Inject all three anomaly scenarios and save each to separate file.
    
    Returns:
        Dict mapping anomaly_type -> DataFrame
    """
    if normal_traffic_path is None:
        normal_traffic_path = DATA_PROCESSED_DIR.parent / "raw" / "normal_traffic.csv"
    
    if output_dir is None:
        output_dir = DATA_PROCESSED_DIR
    
    df = load_csv(normal_traffic_path.parent, normal_traffic_path.name)
    injector = AnomalyInjector(seed=seed)
    
    results = injector.inject_all_anomaly_types(df)
    
    for anomaly_type, anomaly_df in results.items():
        output_path = output_dir / f"anomalous_traffic_{anomaly_type}.csv"
        save_csv(anomaly_df, output_path.parent, output_path.name)
    
    return results


def create_unified_experiment_dataset(
    normal_traffic_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a unified dataset containing normal + all anomaly types.
    Each observation has ground truth labels.
    
    This is the main dataset used for experiments.
    """
    if normal_traffic_path is None:
        normal_traffic_path = DATA_PROCESSED_DIR.parent / "raw" / "normal_traffic.csv"
    
    df = load_csv(normal_traffic_path.parent, normal_traffic_path.name)
    injector = AnomalyInjector(seed=seed)
    
    # Inject all anomaly types into separate copies
    all_dfs = [df.copy()]  # Start with pure normal
    
    for anomaly_type in ["traffic_spike", "destination_expansion", "combined"]:
        anomaly_df = injector.inject_anomalies(df, anomaly_type)
        all_dfs.append(anomaly_df)
    
    # Combine all
    unified_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort
    unified_df = unified_df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
    
    if output_path is None:
        output_path = DATA_PROCESSED_DIR / "unified_experiment_dataset.csv"
    save_csv(unified_df, output_path.parent, output_path.name)
    
    return unified_df


# ============================================================
# VALIDATION
# ============================================================
def validate_anomalous_traffic(df: pd.DataFrame, anomaly_type: str = "any") -> dict:
    """
    Validate anomalous traffic dataset.
    
    Args:
        df: DataFrame with observations including ground truth
        anomaly_type: Expected anomaly type ("any" for unified dataset)
    
    Returns:
        Validation results dict
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
        "packet_size", "destination_count", "protocol", "connection_duration",
        "is_anomaly", "anomaly_type",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        results["valid"] = False
        results["errors"].append(f"Missing columns: {missing}")
        return results
    
    # Validate each observation
    for idx, row in df.iterrows():
        obs = row.to_dict()
        if not validate_observation(obs):
            results["valid"] = False
            results["errors"].append(f"Invalid observation at index {idx}: {obs}")
    
    # Check anomaly counts
    anomaly_counts = df["anomaly_type"].value_counts().to_dict()
    results["stats"]["anomaly_counts"] = anomaly_counts
    
    if anomaly_type != "any":
        # Single anomaly type dataset
        expected_anomalies = anomaly_type
        actual_anomalies = df[df["is_anomaly"]]["anomaly_type"].unique()
        if len(actual_anomalies) > 0 and expected_anomalies not in actual_anomalies:
            results["warnings"].append(f"Expected anomaly type {expected_anomalies}, found {actual_anomalies}")
    
    # Verify traffic spike actually has increased packet rate
    if "traffic_spike" in df["anomaly_type"].values:
        spike_df = df[df["anomaly_type"] == "traffic_spike"]
        normal_df = df[df["anomaly_type"] == "normal"]
        if len(spike_df) > 0 and len(normal_df) > 0:
            spike_rate_mean = spike_df["packet_rate"].mean()
            normal_rate_mean = normal_df["packet_rate"].mean()
            if spike_rate_mean <= normal_rate_mean:
                results["warnings"].append("Traffic spike packet_rate not higher than normal")
    
    # Verify destination expansion has increased destination_count
    if "destination_expansion" in df["anomaly_type"].values:
        dest_df = df[df["anomaly_type"] == "destination_expansion"]
        normal_df = df[df["anomaly_type"] == "normal"]
        if len(dest_df) > 0 and len(normal_df) > 0:
            spike_dest_mean = dest_df["destination_count"].mean()
            normal_dest_mean = normal_df["destination_count"].mean()
            if spike_dest_mean <= normal_dest_mean:
                results["warnings"].append("Destination expansion destination_count not higher than normal")
    
    # Verify combined has both increased
    if "combined" in df["anomaly_type"].values:
        comb_df = df[df["anomaly_type"] == "combined"]
        normal_df = df[df["anomaly_type"] == "normal"]
        if len(comb_df) > 0 and len(normal_df) > 0:
            comb_rate_mean = comb_df["packet_rate"].mean()
            normal_rate_mean = normal_df["packet_rate"].mean()
            comb_dest_mean = comb_df["destination_count"].mean()
            normal_dest_mean = normal_df["destination_count"].mean()
            if comb_rate_mean <= normal_rate_mean:
                results["warnings"].append("Combined anomaly packet_rate not higher than normal")
            if comb_dest_mean <= normal_dest_mean:
                results["warnings"].append("Combined anomaly destination_count not higher than normal")
    
    # Check no corruption of unrelated features (protocol should be unchanged)
    for atype in ["traffic_spike", "destination_expansion", "combined"]:
        if atype in df["anomaly_type"].values:
            anomaly_df = df[df["anomaly_type"] == atype]
            normal_df = df[(df["device_id"].isin(anomaly_df["device_id"])) & (df["anomaly_type"] == "normal")]
            if len(anomaly_df) > 0 and len(normal_df) > 0:
                # Check protocol distribution is similar
                anomaly_protocols = set(anomaly_df["protocol"].unique())
                normal_protocols = set(normal_df["protocol"].unique())
                if not anomaly_protocols.issubset(normal_protocols):
                    results["warnings"].append(f"Anomaly type {atype} has unexpected protocols")
    
    # Summary stats
    results["stats"].update({
        "total_observations": len(df),
        "normal_count": int((~df["is_anomaly"]).sum()),
        "anomaly_count": int(df["is_anomaly"].sum()),
        "anomaly_fraction": df["is_anomaly"].mean(),
        "unique_devices": df["device_id"].nunique(),
    })
    
    return results


if __name__ == "__main__":
    # Direct execution for testing
    print("Injecting anomalies...")
    df = inject_anomalies(anomaly_type="traffic_spike")
    print(f"Generated {len(df)} observations")
    
    validation = validate_anomalous_traffic(df, "traffic_spike")
    print(f"Validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation["errors"]:
        print(f"Errors: {validation['errors']}")
    if validation["warnings"]:
        print(f"Warnings: {validation['warnings']}")
    print(f"Stats: {validation['stats']}")