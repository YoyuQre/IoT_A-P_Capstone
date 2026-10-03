"""
Phase 7: Evaluation & Metrics

Consolidates metrics across experiments, generates final comparison tables,
and provides statistical analysis of detection performance.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import json
from scipy import stats

from .config import (
    MASTER_SEED,
    EXPERIMENT_NAMES,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
    DATA_RESULTS_DIR,
    EVALUATION_METRICS_CSV,
)
from .schemas import DetectionMetrics, AnomalyTypeMetrics
from .utils import load_csv, save_csv


# ============================================================
# EVALUATION ENGINE
# ============================================================
class EvaluationEngine:
    """
    Evaluates and compares detection performance across experiments and thresholds.
    """
    
    def __init__(
        self,
        experiment_results_path: Optional[Path] = None,
        experiment_json_path: Optional[Path] = None,
    ):
        """
        Initialize with experiment results.
        
        Args:
            experiment_results_path: Path to consolidated experiment_results.csv
            experiment_json_path: Path to experiment_results.json with detailed metrics
        """
        if experiment_results_path is None:
            experiment_results_path = DATA_RESULTS_DIR / "experiment_results.csv"
        if experiment_json_path is None:
            experiment_json_path = DATA_RESULTS_DIR / "experiment_results.json"
        
        self.results_df = load_csv(experiment_results_path.parent, experiment_results_path.name)
        
        with open(experiment_json_path, 'r') as f:
            self.json_data = json.load(f)
    
    def get_overall_metrics_table(self) -> pd.DataFrame:
        """
        Generate overall metrics comparison table across experiments and thresholds.
        
        Returns:
            DataFrame with columns: experiment, threshold, tp, tn, fp, fn, 
            accuracy, precision, recall, f1_score, false_positive_rate
        """
        rows = []
        
        for exp_name, exp_data in self.json_data.items():
            for m in exp_data["metrics"]:
                row = {"experiment": exp_name}
                row.update(m)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns
        col_order = [
            "experiment", "threshold", "tp", "tn", "fp", "fn",
            "accuracy", "precision", "recall", "f1_score", "false_positive_rate",
        ]
        df = df[col_order]
        
        return df
    
    def get_anomaly_type_metrics_table(self) -> pd.DataFrame:
        """
        Generate metrics broken down by anomaly type.
        
        Returns:
            DataFrame with per-anomaly-type metrics
        """
        rows = []
        
        for exp_name, exp_data in self.json_data.items():
            for m in exp_data["anomaly_type_metrics"]:
                row = {"experiment": exp_name}
                row.update(m)
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if len(df) > 0:
            col_order = [
                "experiment", "anomaly_type", "threshold",
                "tp", "tn", "fp", "fn",
                "precision", "recall", "f1_score",
            ]
            df = df[col_order]
        
        return df
    
    def get_threshold_comparison(self) -> pd.DataFrame:
        """
        Generate threshold comparison for Experiment E (threshold study).
        
        Returns:
            DataFrame comparing metrics across thresholds
        """
        # Experiment E uses combined anomalies
        exp_data = self.json_data.get("experiment_e_threshold_study", {})
        metrics = exp_data.get("metrics", [])
        
        df = pd.DataFrame(metrics)
        return df
    
    def get_detection_summary(self) -> dict:
        """
        Generate a summary of detection performance.
        
        Returns:
            Dict with key summary statistics
        """
        overall = self.get_overall_metrics_table()
        
        summary = {
            "total_experiments": len(self.json_data),
            "thresholds_tested": sorted(self.results_df["threshold"].unique().tolist()),
            "default_threshold": DEFAULT_THRESHOLD,
            "perfect_detection_experiments": [],
            "experiments_with_false_positives": [],
        }
        
        # Check which experiments have perfect detection
        for exp_name in EXPERIMENT_NAMES:
            exp_df = overall[overall["experiment"] == exp_name]
            if len(exp_df) > 0:
                # Check if any threshold has perfect detection (recall=1, precision=1, fpr=0)
                perfect = ((exp_df["recall"] == 1.0) & 
                          (exp_df["precision"] == 1.0) & 
                          (exp_df["false_positive_rate"] == 0.0)).any()
                if perfect:
                    summary["perfect_detection_experiments"].append(exp_name)
                
                # Check for false positives at any threshold
                has_fp = (exp_df["fp"] > 0).any()
                if has_fp:
                    summary["experiments_with_false_positives"].append(exp_name)
        
        # Experiment A specific: false positive rate
        exp_a = overall[overall["experiment"] == "experiment_a_normal"]
        if len(exp_a) > 0:
            summary["experiment_a_fpr"] = exp_a["false_positive_rate"].tolist()
        
        return summary
    
    def statistical_significance_test(
        self,
        exp1: str,
        exp2: str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> dict:
        """
        Perform McNemar's test to compare two experiments at a given threshold.
        
        Note: This requires paired predictions on the same observations.
        Since each experiment has different injected anomalies, this is 
        only meaningful for comparing thresholds within the same experiment.
        
        Returns:
            Dict with test statistic and p-value
        """
        # Get detection results for both experiments at threshold
        df1 = self.results_df[
            (self.results_df["experiment"] == exp1) & 
            (self.results_df["threshold"] == threshold)
        ]
        df2 = self.results_df[
            (self.results_df["experiment"] == exp2) & 
            (self.results_df["threshold"] == threshold)
        ]
        
        # For threshold comparison within same experiment, use paired test
        if exp1 == exp2:
            return {"error": "Cannot compare experiment with itself"}
        
        # Since experiments have different anomaly injections, we can't do 
        # a proper paired test. Instead, we report that comparison is not
        # statistically valid across different anomaly injections.
        return {
            "test": "McNemar's test",
            "valid": False,
            "reason": "Experiments have different anomaly injections; paired comparison not valid",
            "note": "Use threshold comparison within same experiment for statistical tests",
        }
    
    def threshold_effect_analysis(self) -> pd.DataFrame:
        """
        Analyze the effect of threshold on detection metrics.
        
        Returns:
            DataFrame with threshold effect metrics
        """
        overall = self.get_overall_metrics_table()
        
        # Focus on Experiment E (threshold study) or aggregate across experiments
        exp_e = overall[overall["experiment"] == "experiment_e_threshold_study"]
        
        if len(exp_e) == 0:
            return pd.DataFrame()
        
        # Compute changes between thresholds
        rows = []
        thresholds = sorted(exp_e["threshold"].unique())
        
        for i, t in enumerate(thresholds):
            row = {"threshold": t}
            m = exp_e[exp_e["threshold"] == t].iloc[0]
            
            for metric in ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]:
                row[metric] = m[metric]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values("threshold")
        
        # Add delta columns
        for metric in ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]:
            df[f"{metric}_delta"] = df[metric].diff()
        
        return df
    
    def save_evaluation_report(
        self,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        Save comprehensive evaluation report.
        
        Returns:
            Dict with paths to saved files
        """
        if output_dir is None:
            output_dir = DATA_RESULTS_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # 1. Overall metrics table
        overall = self.get_overall_metrics_table()
        overall_path = output_dir / "evaluation_overall_metrics.csv"
        save_csv(overall, overall_path.parent, overall_path.name)
        paths["overall_metrics"] = overall_path
        
        # 2. Anomaly type metrics
        type_metrics = self.get_anomaly_type_metrics_table()
        if len(type_metrics) > 0:
            type_path = output_dir / "evaluation_anomaly_type_metrics.csv"
            save_csv(type_metrics, type_path.parent, type_path.name)
            paths["anomaly_type_metrics"] = type_path
        
        # 3. Threshold comparison
        threshold_comp = self.threshold_effect_analysis()
        if len(threshold_comp) > 0:
            threshold_path = output_dir / "evaluation_threshold_comparison.csv"
            save_csv(threshold_comp, threshold_path.parent, threshold_path.name)
            paths["threshold_comparison"] = threshold_path
        
        # 4. Summary
        summary = self.get_detection_summary()
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        paths["summary"] = summary_path
        
        # 5. LaTeX table for paper
        latex_path = output_dir / "evaluation_latex_table.txt"
        latex_table = self._generate_latex_table(overall)
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        paths["latex_table"] = latex_path
        
        return paths
    
    def _generate_latex_table(self, overall_df: pd.DataFrame) -> str:
        """Generate LaTeX table for the paper."""
        # Create a nice formatted table for the main results
        lines = []
        lines.append("% Detection Performance Comparison Table")
        lines.append("% Generated automatically from experiment results")
        lines.append("")
        lines.append("\\begin{table}[htbp]")
        lines.append("  \\centering")
        lines.append("  \\caption{Detection performance across experiments and thresholds.}")
        lines.append("  \\label{tab:detection_performance}")
        lines.append("  \\begin{tabular}{lcccccc}")
        lines.append("    \\toprule")
        lines.append("    Experiment & Threshold & Accuracy & Precision & Recall & F1-Score & FPR \\\\")
        lines.append("    \\midrule")
        
        for _, row in overall_df.iterrows():
            exp_short = row["experiment"].replace("experiment_", "").replace("_", " ").title()
            lines.append(
                f"    {exp_short} & {row['threshold']:.1f} & "
                f"{row['accuracy']:.3f} & {row['precision']:.3f} & "
                f"{row['recall']:.3f} & {row['f1_score']:.3f} & "
                f"{row['false_positive_rate']:.3f} \\\\"
            )
        
        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================
def run_evaluation(
    experiment_results_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run complete evaluation and save report.
    
    Returns:
        Dict with saved file paths
    """
    engine = EvaluationEngine(experiment_results_path)
    paths = engine.save_evaluation_report(output_dir)
    return paths


def load_evaluation_summary(
    summary_path: Optional[Path] = None,
) -> dict:
    """Load evaluation summary from JSON."""
    if summary_path is None:
        summary_path = DATA_RESULTS_DIR / "evaluation_summary.json"
    
    with open(summary_path, 'r') as f:
        return json.load(f)


# ============================================================
# VALIDATION
# ============================================================
def validate_evaluation_outputs(output_dir: Path) -> dict:
    """Validate that all evaluation outputs exist and are well-formed."""
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "files_found": [],
    }
    
    expected_files = [
        "evaluation_overall_metrics.csv",
        "evaluation_anomaly_type_metrics.csv",
        "evaluation_threshold_comparison.csv",
        "evaluation_summary.json",
        "evaluation_latex_table.txt",
    ]
    
    for fname in expected_files:
        fpath = output_dir / fname
        if fpath.exists():
            validation["files_found"].append(fname)
        else:
            validation["warnings"].append(f"Missing expected file: {fname}")
    
    # Check JSON is valid
    summary_path = output_dir / "evaluation_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                json.load(f)
        except json.JSONDecodeError:
            validation["valid"] = False
            validation["errors"].append("Invalid JSON in evaluation_summary.json")
    
    return validation


if __name__ == "__main__":
    print("Running evaluation...")
    paths = run_evaluation()
    
    print("\nSaved files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Print summary
    engine = EvaluationEngine()
    summary = engine.get_detection_summary()
    print("\nEvaluation Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Print overall metrics
    overall = engine.get_overall_metrics_table()
    print("\nOverall Metrics:")
    print(overall.to_string(index=False))