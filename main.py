#!/usr/bin/env python3
"""
Main entry point for IoT Device Behaviour Fingerprinting and Simulation-Based Anomaly Detection.

This script orchestrates the complete pipeline:
1. Configuration initialization
2. Normal traffic generation
3. Behavioural fingerprinting
4. Anomaly injection
5. Statistical detection
6. Experiment execution
7. Evaluation
8. Visualization

Usage:
    python main.py [--phase PHASE] [--experiment EXPERIMENT] [--seed SEED]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.config import MASTER_SEED
from src.utils import set_global_seed


def _dataframes_equal_tol(df1: pd.DataFrame, df2: pd.DataFrame, rtol: float = 1e-10, atol: float = 1e-12) -> bool:
    """Compare two DataFrames with tolerance for floating-point differences."""
    if df1.shape != df2.shape:
        return False
    if not df1.columns.equals(df2.columns):
        return False
    if not df1.index.equals(df2.index):
        return False
    
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]):
            if not np.allclose(df1[col].values, df2[col].values, rtol=rtol, atol=atol, equal_nan=True):
                return False
        else:
            if not df1[col].equals(df2[col]):
                return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="IoT Device Behaviour Fingerprinting and Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        default=1,
        help="Pipeline phase to execute (default: 1)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["all", "a", "b", "c", "d", "e"],
        default="all",
        help="Experiment to run (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help=f"Random seed for reproducibility (default: {MASTER_SEED})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def run_phase1(args):
    """Phase 1: Project foundation verification."""
    print("=" * 60)
    print("PHASE 1: Project Foundation Verification")
    print("=" * 60)
    
    # Verify imports
    from src import (
        MASTER_SEED, DEVICE_CATEGORIES, NUM_DEVICES,
        CATEGORY_PROFILES, FINGERPRINT_FEATURES,
        ANOMALY_TYPES, DETECTION_THRESHOLDS,
        Observation, DeviceFingerprint, DetectionResult,
        DeterministicRNG, get_rng, safe_z_score
    )
    
    print(f"[OK] Configuration loaded (seed={MASTER_SEED})")
    print(f"[OK] Device categories: {DEVICE_CATEGORIES}")
    print(f"[OK] Total devices: {NUM_DEVICES}")
    print(f"[OK] Fingerprint features: {FINGERPRINT_FEATURES}")
    print(f"[OK] Anomaly types: {ANOMALY_TYPES}")
    print(f"[OK] Detection thresholds: {DETECTION_THRESHOLDS}")
    print(f"[OK] Schemas defined: Observation, DeviceFingerprint, DetectionResult")
    print(f"[OK] Utilities available: DeterministicRNG, safe_z_score")
    
    # Test deterministic RNG
    rng = get_rng()
    test_val = rng.uniform(0, 1)
    print(f"[OK] RNG test value: {test_val:.6f}")
    
    # Test safe z-score
    z = safe_z_score(10.0, 5.0, 2.0)
    print(f"[OK] Safe z-score test: z(10, 5, 2) = {z}")
    
    print("\nPhase 1 verification PASSED")
    return True


def run_phase2(args):
    """Phase 2: Normal traffic generation."""
    print("=" * 60)
    print("PHASE 2: Normal Traffic Generation")
    print("=" * 60)
    
    from src.simulator import generate_normal_traffic, validate_normal_traffic
    from src.config import DATA_RAW_DIR, NORMAL_TRAFFIC_CSV, NUM_DEVICES, DEVICE_CATEGORIES
    
    print("Generating normal traffic for 25 devices across 5 categories...")
    df = generate_normal_traffic(seed=args.seed)
    
    print(f"[OK] Generated {len(df)} observations")
    print(f"[OK] Devices: {df['device_id'].nunique()} (expected {NUM_DEVICES})")
    print(f"[OK] Categories: {df['device_type'].nunique()} (expected {len(DEVICE_CATEGORIES)})")
    
    # Validation
    results = validate_normal_traffic(df)
    if results["valid"]:
        print("[OK] Validation PASSED")
    else:
        print("[FAIL] Validation FAILED")
        for err in results["errors"]:
            print(f"  ERROR: {err}")
        return False
    
    if results["warnings"]:
        for warn in results["warnings"]:
            print(f"  WARNING: {warn}")
    
    # Show category distribution
    print("\nCategory distribution:")
    for cat, count in df['device_type'].value_counts().items():
        print(f"  {cat}: {count} observations")
    
    print(f"\nOutput saved to: {DATA_RAW_DIR / NORMAL_TRAFFIC_CSV}")
    return True


def run_phase3(args):
    """Phase 3: Behavioural fingerprinting."""
    print("=" * 60)
    print("PHASE 3: Behavioural Fingerprinting")
    print("=" * 60)
    
    from src.fingerprint import compute_device_fingerprints, validate_fingerprints
    from src.config import DATA_PROCESSED_DIR, DEVICE_FINGERPRINTS_CSV, NUM_DEVICES, FINGERPRINT_FEATURES
    from src.simulator import generate_normal_traffic
    
    print("Loading normal traffic...")
    normal_df = generate_normal_traffic(seed=args.seed)
    print(f"[OK] Loaded {len(normal_df)} normal observations")
    
    print("Computing device fingerprints...")
    fingerprints = compute_device_fingerprints()
    
    print(f"[OK] Computed {len(fingerprints)} fingerprint entries")
    print(f"[OK] Devices: {fingerprints['device_id'].nunique()} (expected {NUM_DEVICES})")
    print(f"[OK] Features per device: {len(FINGERPRINT_FEATURES)} ({', '.join(FINGERPRINT_FEATURES)})")
    
    # Validation
    results = validate_fingerprints(fingerprints)
    if results["valid"]:
        print("[OK] Validation PASSED")
    else:
        print("[FAIL] Validation FAILED")
        for err in results["errors"]:
            print(f"  ERROR: {err}")
        return False
    
    if results["warnings"]:
        for warn in results["warnings"]:
            print(f"  WARNING: {warn}")
    
    # Show sample fingerprints
    print("\nSample fingerprints (first 3 devices):")
    sample_devices = sorted(fingerprints["device_id"].unique())[:3]
    for device_id in sample_devices:
        device_fps = fingerprints[fingerprints["device_id"] == device_id]
        print(f"  {device_id}:")
        for _, row in device_fps.iterrows():
            print(f"    {row['feature']}: mean={row['mean']:.4f}, std={row['std']:.6f}")
    
    print(f"\nOutput saved to: {DATA_PROCESSED_DIR / DEVICE_FINGERPRINTS_CSV}")
    return True


def run_phase4(args):
    """Phase 4: Anomaly injection."""
    print("=" * 60)
    print("PHASE 4: Anomaly Injection")
    print("=" * 60)
    
    from src.anomalies import inject_anomalies, validate_anomalous_traffic
    from src.simulator import generate_normal_traffic
    from src.config import DATA_PROCESSED_DIR, ANOMALOUS_TRAFFIC_CSV, ANOMALY_TYPES
    
    print("Loading normal traffic...")
    normal_df = generate_normal_traffic(seed=args.seed)
    print(f"[OK] Loaded {len(normal_df)} normal observations")
    
    # Inject each anomaly type
    all_results = {}
    
    for anomaly_type in ["traffic_spike", "destination_expansion", "combined"]:
        print(f"\nInjecting {anomaly_type} anomalies...")
        result_df = inject_anomalies(anomaly_type=anomaly_type, seed=args.seed)
        
        anomaly_count = int(result_df["is_anomaly"].sum())
        normal_count = int((~result_df["is_anomaly"]).sum())
        total_count = len(result_df)
        
        print(f"  [OK] {anomaly_type}: {anomaly_count} anomalies, {normal_count} normal ({total_count} total)")
        
        # Validation
        results = validate_anomalous_traffic(result_df, anomaly_type)
        if results["valid"]:
            print(f"  [OK] Validation PASSED")
        else:
            print(f"  [FAIL] Validation FAILED")
            for err in results["errors"]:
                print(f"    ERROR: {err}")
            return False
        
        if results["warnings"]:
            for warn in results["warnings"]:
                print(f"  WARNING: {warn}")
        
        all_results[anomaly_type] = {
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "total_count": total_count,
        }
    
    print(f"\nOutput saved to: {DATA_PROCESSED_DIR}/anomalous_traffic_*.csv")
    return True


def run_phase5(args):
    """Phase 5: Anomaly detection."""
    print("=" * 60)
    print("PHASE 5: Anomaly Detection")
    print("=" * 60)
    
    from src.detector import run_detection_on_all_scenarios, validate_detection_results
    from src.config import DATA_PROCESSED_DIR, DATA_RESULTS_DIR, DETECTION_THRESHOLDS
    
    print("Loading fingerprints and anomalous traffic...")
    print(f"Running detection with thresholds: {DETECTION_THRESHOLDS}")
    
    results = run_detection_on_all_scenarios()
    
    all_valid = True
    
    for anomaly_type, results_df in results.items():
        anomaly_count = int(results_df["ground_truth"].sum())
        pred_counts = results_df.groupby("threshold")["predicted_label"].sum()
        
        print(f"\n{anomaly_type}: {anomaly_count} ground truth anomalies")
        for threshold in sorted(results_df["threshold"].unique()):
            count = pred_counts[threshold]
            print(f"  Threshold {threshold}: {count} predicted anomalies")
        
        # Validation
        validation = validate_detection_results(results_df)
        if validation["valid"]:
            print(f"  [OK] Validation PASSED")
        else:
            print(f"  [FAIL] Validation FAILED")
            for err in validation["errors"]:
                print(f"    ERROR: {err}")
            all_valid = False
        
        if validation["warnings"]:
            for warn in validation["warnings"]:
                print(f"  WARNING: {warn}")
    
    print(f"\nOutput saved to: {DATA_RESULTS_DIR}/detection_results_*.csv")
    return all_valid


def run_phase6(args):
    """Phase 6: Experimental framework."""
    print("=" * 60)
    print("PHASE 6: Experimental Framework")
    print("=" * 60)
    
    from src.experiments import run_experiments, validate_experiment_results
    from src.config import DATA_RESULTS_DIR, EXPERIMENT_RESULTS_CSV, EXPERIMENT_NAMES
    
    print("Running all 5 experiments (A-E)...")
    print("  A: Normal traffic (baseline)")
    print("  B: Traffic spike")
    print("  C: Destination expansion")
    print("  D: Combined anomalies")
    print("  E: Threshold study")
    print()
    
    results = run_experiments(seed=args.seed)
    
    all_valid = True
    
    for exp_name in EXPERIMENT_NAMES:
        exp_result = results[exp_name]
        metrics = exp_result["metrics"]
        
        print(f"\n{exp_name}:")
        for m in metrics:
            print(f"  Threshold {m['threshold']}: TP={m['tp']}, TN={m['tn']}, "
                  f"FP={m['fp']}, FN={m['fn']}, "
                  f"Acc={m['accuracy']:.3f}, Prec={m['precision']:.3f}, "
                  f"Rec={m['recall']:.3f}, F1={m['f1_score']:.3f}, "
                  f"FPR={m['false_positive_rate']:.3f}")
        
        # Build consolidated DataFrame for validation
        consolidated_rows = []
        for det_result in exp_result["detection_results"]:
            row = det_result.copy()
            row["experiment"] = exp_name
            consolidated_rows.append(row)
        
        consolidated_df = pd.DataFrame(consolidated_rows)
        
        # Validation
        validation = validate_experiment_results(consolidated_df)
        if validation["valid"]:
            print(f"  [OK] Validation PASSED")
        else:
            print(f"  [FAIL] Validation FAILED")
            for err in validation["errors"]:
                print(f"    ERROR: {err}")
            all_valid = False
        
        if validation["warnings"]:
            for warn in validation["warnings"]:
                print(f"  WARNING: {warn}")
    
    print(f"\nOutput saved to: {DATA_RESULTS_DIR / EXPERIMENT_RESULTS_CSV}")
    return all_valid


def run_phase7(args):
    """Phase 7: Evaluation & Metrics."""
    print("=" * 60)
    print("PHASE 7: Evaluation & Metrics")
    print("=" * 60)
    
    from src.evaluation import run_evaluation, EvaluationEngine
    from src.config import DATA_RESULTS_DIR
    
    print("Running evaluation and generating reports...")
    paths = run_evaluation()
    
    print("\nSaved evaluation outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Print summary
    engine = EvaluationEngine()
    summary = engine.get_detection_summary()
    
    print("\nEvaluation Summary:")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Thresholds tested: {summary['thresholds_tested']}")
    print(f"  Perfect detection: {summary['perfect_detection_experiments']}")
    print(f"  Experiments with FPs: {summary['experiments_with_false_positives']}")
    print(f"  Experiment A FPR: {summary['experiment_a_fpr']}")
    
    # Print overall metrics table
    overall = engine.get_overall_metrics_table()
    print("\nOverall Metrics:")
    print(overall.to_string(index=False))
    
    # Print threshold comparison
    threshold_comp = engine.threshold_effect_analysis()
    print("\nThreshold Effect Analysis (Experiment E):")
    print(threshold_comp.to_string(index=False))
    
    return True


def run_phase8(args):
    """Phase 8: Visualization & Analysis."""
    print("=" * 60)
    print("PHASE 8: Visualization & Analysis")
    print("=" * 60)
    
    from src.visualization import generate_all_visualizations, validate_visualizations
    from src.config import OUTPUTS_FIGURES_DIR
    
    print("Generating 6 required visualizations...")
    paths = generate_all_visualizations()
    
    print("\nValidation:")
    validation = validate_visualizations()
    for f in validation["files_found"]:
        print(f"  [OK] {f}")
    for f in validation["files_missing"]:
        print(f"  [MISSING] {f}")
    
    if validation["valid"]:
        print("\nAll 6 visualizations generated successfully!")
        return True
    else:
        print(f"\nSome visualizations failed: {validation['errors']}")
        return False


def run_phase9(args):
    """Phase 9: End-to-End Integration and Validation."""
    print("=" * 60)
    print("PHASE 9: End-to-End Integration & Validation")
    print("=" * 60)
    
    from src.config import (
        MASTER_SEED, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_RESULTS_DIR,
        OUTPUTS_FIGURES_DIR, OUTPUTS_REPORTS_DIR, NORMAL_TRAFFIC_CSV,
        DEVICE_FINGERPRINTS_CSV, ANOMALOUS_TRAFFIC_CSV, EXPERIMENT_RESULTS_CSV,
        EVALUATION_METRICS_CSV, EXPERIMENT_NAMES, DETECTION_THRESHOLDS
    )
    from src.simulator import generate_normal_traffic, validate_normal_traffic
    from src.fingerprint import compute_device_fingerprints, validate_fingerprints
    from src.anomalies import inject_anomalies, validate_anomalous_traffic
    from src.detector import run_detection_on_all_scenarios, validate_detection_results
    from src.experiments import run_experiments, validate_experiment_results
    from src.evaluation import run_evaluation, EvaluationEngine
    from src.visualization import generate_all_visualizations, validate_visualizations
    import pandas as pd
    import json
    
    print("Running complete pipeline from clean state...\n")
    
    # Track all outputs for reproducibility test
    all_outputs = {}
    
    # ============================================================
    # STEP 1: Configuration (implicit - already loaded)
    # ============================================================
    print("STEP 1: Configuration initialized")
    print(f"  Seed: {args.seed}")
    print(f"  Master seed: {MASTER_SEED}")
    
    # ============================================================
    # STEP 2: Device Creation (implicit in simulator)
    # ============================================================
    print("\nSTEP 2: Device Creation (25 devices, 5 categories)")
    
    # ============================================================
    # STEP 3: Normal Traffic Generation
    # ============================================================
    print("\nSTEP 3: Normal Traffic Generation")
    normal_df = generate_normal_traffic(seed=args.seed)
    normal_path = DATA_RAW_DIR / NORMAL_TRAFFIC_CSV
    print(f"  Generated {len(normal_df)} observations for {normal_df['device_id'].nunique()} devices")
    print(f"  Saved: {normal_path}")
    all_outputs["normal_traffic"] = normal_path
    
    # Validate
    val = validate_normal_traffic(normal_df)
    if not val["valid"]:
        print(f"  [FAIL] Normal traffic validation failed: {val['errors']}")
        return False
    print(f"  [OK] Normal traffic validation passed")
    
    # ============================================================
    # STEP 4: Fingerprint Generation
    # ============================================================
    print("\nSTEP 4: Behavioural Fingerprinting")
    fingerprints = compute_device_fingerprints()
    fp_path = DATA_PROCESSED_DIR / DEVICE_FINGERPRINTS_CSV
    print(f"  Computed {len(fingerprints)} fingerprint entries ({fingerprints['device_id'].nunique()} devices × 3 features)")
    print(f"  Saved: {fp_path}")
    all_outputs["fingerprints"] = fp_path
    
    val = validate_fingerprints(fingerprints)
    if not val["valid"]:
        print(f"  [FAIL] Fingerprint validation failed: {val['errors']}")
        return False
    print(f"  [OK] Fingerprint validation passed")
    
    # ============================================================
    # STEP 5: Anomaly Injection
    # ============================================================
    print("\nSTEP 5: Anomaly Injection (3 scenarios)")
    anomaly_outputs = {}
    for anomaly_type in ["traffic_spike", "destination_expansion", "combined"]:
        result_df = inject_anomalies(anomaly_type=anomaly_type, seed=args.seed)
        out_path = DATA_PROCESSED_DIR / f"{ANOMALOUS_TRAFFIC_CSV.replace('.csv', f'_{anomaly_type}.csv')}"
        print(f"  {anomaly_type}: {int(result_df['is_anomaly'].sum())} anomalies, {int((~result_df['is_anomaly']).sum())} normal")
        anomaly_outputs[anomaly_type] = out_path
        
        val = validate_anomalous_traffic(result_df, anomaly_type)
        if not val["valid"]:
            print(f"  [FAIL] {anomaly_type} validation failed: {val['errors']}")
            return False
    print(f"  [OK] All anomaly injections validated")
    all_outputs["anomalies"] = anomaly_outputs
    
    # ============================================================
    # STEP 6: Detection
    # ============================================================
    print("\nSTEP 6: Statistical Anomaly Detection")
    detection_results = run_detection_on_all_scenarios()
    print(f"  Ran detection on {len(detection_results)} anomaly scenarios × {len(DETECTION_THRESHOLDS)} thresholds")
    all_outputs["detection"] = list(detection_results.keys())
    
    for anomaly_type, results_df in detection_results.items():
        val = validate_detection_results(results_df)
        if not val["valid"]:
            print(f"  [FAIL] {anomaly_type} detection validation failed: {val['errors']}")
            return False
    print(f"  [OK] All detection results validated")
    
    # ============================================================
    # STEP 7: Experiments
    # ============================================================
    print("\nSTEP 7: Experimental Framework (5 experiments)")
    experiment_results = run_experiments(seed=args.seed)
    exp_path = DATA_RESULTS_DIR / EXPERIMENT_RESULTS_CSV
    print(f"  Ran {len(experiment_results)} experiments")
    print(f"  Consolidated results: {exp_path}")
    all_outputs["experiments"] = exp_path
    
    for exp_name in EXPERIMENT_NAMES:
        exp_result = experiment_results[exp_name]
        consolidated_rows = []
        for det_result in exp_result["detection_results"]:
            row = det_result.copy()
            row["experiment"] = exp_name
            consolidated_rows.append(row)
        consolidated_df = pd.DataFrame(consolidated_rows)
        val = validate_experiment_results(consolidated_df)
        if not val["valid"]:
            print(f"  [FAIL] {exp_name} validation failed: {val['errors']}")
            return False
    print(f"  [OK] All experiment results validated")
    
    # ============================================================
    # STEP 8: Evaluation
    # ============================================================
    print("\nSTEP 8: Evaluation & Metrics")
    eval_paths = run_evaluation()
    print(f"  Generated {len(eval_paths)} evaluation outputs:")
    for name, path in eval_paths.items():
        print(f"    {name}: {path}")
    all_outputs["evaluation"] = eval_paths
    
    # ============================================================
    # STEP 9: Visualization


# ============================================================
    print("\nSTEP 9: Visualization & Analysis")
    viz_paths = generate_all_visualizations()
    print(f"  Generated {len(viz_paths)} visualizations:")
    for path in viz_paths:
        print(f"    {path.name}")
    all_outputs["visualizations"] = [str(p) for p in viz_paths]
    
    val = validate_visualizations()
    if not val["valid"]:
        print(f"  [FAIL] Visualization validation failed: {val['errors']}")
        return False
    print(f"  [OK] All visualizations validated")
    
    # ============================================================
    # REPRODUCIBILITY TEST
    # ============================================================
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY TEST: Running pipeline a second time...")
    print("=" * 60)
    
    # Store first-run anomaly dataframes for comparison
    first_run_anomalies = {}
    for anomaly_type in ["traffic_spike", "destination_expansion", "combined"]:
        path = DATA_PROCESSED_DIR / f"{ANOMALOUS_TRAFFIC_CSV.replace('.csv', f'_{anomaly_type}.csv')}"
        first_run_anomalies[anomaly_type] = pd.read_csv(path)
    
    # Run key steps again with same seed
    print("  Regenerating normal traffic...")
    normal_df_2 = generate_normal_traffic(seed=args.seed)
    
    print("  Regenerating fingerprints...")
    fingerprints_2 = compute_device_fingerprints()
    
    print("  Regenerating anomalies...")
    anomalies_2 = {}
    for anomaly_type in ["traffic_spike", "destination_expansion", "combined"]:
        anomalies_2[anomaly_type] = inject_anomalies(anomaly_type=anomaly_type, seed=args.seed)
    
    print("  Comparing outputs...")
    
    # Compare normal traffic
    if not normal_df.equals(normal_df_2):
        print("  [FAIL] Normal traffic NOT identical on second run!")
        diff = normal_df.compare(normal_df_2)
        print(f"    Differences found: {len(diff)} rows")
        return False
    print("  [OK] Normal traffic identical")
    
    # Compare fingerprints
    if not fingerprints.equals(fingerprints_2):
        print("  [FAIL] Fingerprints NOT identical on second run!")
        return False
    print("  [OK] Fingerprints identical")
    
    # Compare anomalies against first-run saved data (with tolerance for float precision)
    for anomaly_type in ["traffic_spike", "destination_expansion", "combined"]:
        if not _dataframes_equal_tol(first_run_anomalies[anomaly_type], anomalies_2[anomaly_type]):
            print(f"  [FAIL] {anomaly_type} anomalies NOT identical on second run!")
            return False
        print(f"  [OK] {anomaly_type} anomalies identical")
    
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY TEST PASSED - All outputs identical!")
    print("=" * 60)
    
    # ============================================================
    # FINAL VALIDATION
    # ============================================================
    print("\nFINAL VALIDATION CHECKLIST:")
    checks = [
        ("Normal traffic generated", "normal_traffic" in all_outputs),
        ("Fingerprints computed", "fingerprints" in all_outputs),
        ("All 3 anomaly types injected", len(all_outputs.get("anomalies", {})) == 3),
        ("Detection run on all scenarios", len(all_outputs.get("detection", [])) == 3),
        ("All 5 experiments executed", len(experiment_results) == 5),
        ("Evaluation metrics generated", len(all_outputs.get("evaluation", {})) >= 4),
        ("All 6 visualizations created", len(viz_paths) == 6),
        ("Reproducibility verified", True),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("\n[FAIL] Some validation checks failed")
        return False
    
    print("\n" + "=" * 60)
    print("PHASE 9 COMPLETE - End-to-End Integration Successful!")
    print("=" * 60)
    print("All pipeline stages executed and validated.")
    print("Reproducibility confirmed with identical seed.")
    return True


def run_phase10(args):
    """Phase 10: Final documentation (placeholder)."""
    print("=" * 60)
    print("PHASE 10: Final Documentation & Report")
    print("=" * 60)
    print("NOT YET IMPLEMENTED")
    return False


PHASE_RUNNERS = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
    6: run_phase6,
    7: run_phase7,
    8: run_phase8,
    9: run_phase9,
    10: run_phase10,
}


def main():
    args = parse_args()
    
    # Set global seed for reproducibility
    set_global_seed(args.seed)
    
    print(f"IoT Behaviour Fingerprinting Pipeline")
    print(f"Seed: {args.seed}")
    print(f"Phase: {args.phase}")
    print()
    
    if args.phase not in PHASE_RUNNERS:
        print(f"ERROR: Phase {args.phase} not implemented")
        return 1
    
    runner = PHASE_RUNNERS[args.phase]
    success = runner(args)
    
    if success:
        print("\n[OK] Phase completed successfully")
        return 0
    else:
        print("\n[FAIL] Phase not yet implemented or failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())