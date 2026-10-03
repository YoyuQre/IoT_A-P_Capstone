"""
Phase 7 Tests: Evaluation & Metrics
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.config import (
    MASTER_SEED,
    EXPERIMENT_NAMES,
    DETECTION_THRESHOLDS,
    DEFAULT_THRESHOLD,
)
from src.evaluation import (
    EvaluationEngine,
    run_evaluation,
    validate_evaluation_outputs,
)


# ============================================================
# FIXTURE
# ============================================================
@pytest.fixture(scope="module")
def evaluation_engine():
    """Create evaluation engine from existing experiment results."""
    # Run experiments first if not already done
    from src.experiments import run_experiments
    run_experiments(seed=MASTER_SEED)
    
    return EvaluationEngine()


# ============================================================
# EVALUATION ENGINE TESTS
# ============================================================
class TestEvaluationEngine:
    """Test EvaluationEngine class."""
    
    def test_engine_creation(self, evaluation_engine):
        """Test engine instantiation."""
        assert evaluation_engine.results_df is not None
        assert len(evaluation_engine.results_df) > 0
        assert evaluation_engine.json_data is not None
        assert set(evaluation_engine.json_data.keys()) == set(EXPERIMENT_NAMES)
    
    def test_get_overall_metrics_table(self, evaluation_engine):
        """Test overall metrics table generation."""
        df = evaluation_engine.get_overall_metrics_table()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(EXPERIMENT_NAMES) * len(DETECTION_THRESHOLDS)
        
        expected_cols = [
            "experiment", "threshold", "tp", "tn", "fp", "fn",
            "accuracy", "precision", "recall", "f1_score", "false_positive_rate",
        ]
        assert list(df.columns) == expected_cols
        
        # Check all experiments present
        assert set(df["experiment"].unique()) == set(EXPERIMENT_NAMES)
        
        # Check all thresholds present
        assert set(df["threshold"].unique()) == set(DETECTION_THRESHOLDS)
    
    def test_get_anomaly_type_metrics_table(self, evaluation_engine):
        """Test anomaly type metrics table."""
        df = evaluation_engine.get_anomaly_type_metrics_table()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        expected_cols = [
            "experiment", "anomaly_type", "threshold",
            "tp", "tn", "fp", "fn",
            "precision", "recall", "f1_score",
        ]
        assert list(df.columns) == expected_cols
        
        # Should have metrics for traffic_spike, destination_expansion, combined
        anomaly_types = set(df["anomaly_type"].unique())
        assert "traffic_spike" in anomaly_types
        assert "destination_expansion" in anomaly_types
        assert "combined" in anomaly_types
    
    def test_get_threshold_comparison(self, evaluation_engine):
        """Test threshold comparison for Experiment E."""
        df = evaluation_engine.get_threshold_comparison()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(DETECTION_THRESHOLDS)
        
        expected_cols = [
            "threshold", "total_observations", "anomaly_count", "normal_count",
            "tp", "tn", "fp", "fn",
            "accuracy", "precision", "recall", "f1_score", "false_positive_rate",
        ]
        assert list(df.columns) == expected_cols
    
    def test_get_detection_summary(self, evaluation_engine):
        """Test detection summary generation."""
        summary = evaluation_engine.get_detection_summary()
        
        assert isinstance(summary, dict)
        assert "total_experiments" in summary
        assert "thresholds_tested" in summary
        assert "perfect_detection_experiments" in summary
        assert "experiments_with_false_positives" in summary
        assert "experiment_a_fpr" in summary
        
        assert summary["total_experiments"] == 5
        assert set(summary["thresholds_tested"]) == set(DETECTION_THRESHOLDS)
        
        # Experiment A should have FPR = 0
        assert summary["experiment_a_fpr"] == [0.0, 0.0, 0.0]
    
    def test_threshold_effect_analysis(self, evaluation_engine):
        """Test threshold effect analysis."""
        df = evaluation_engine.threshold_effect_analysis()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(DETECTION_THRESHOLDS)
        
        # Should have delta columns
        for metric in ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]:
            assert f"{metric}_delta" in df.columns
    
    def test_save_evaluation_report(self, evaluation_engine, tmp_path):
        """Test saving evaluation report."""
        paths = evaluation_engine.save_evaluation_report(tmp_path)
        
        assert isinstance(paths, dict)
        expected_keys = ["overall_metrics", "anomaly_type_metrics", "threshold_comparison", "summary", "latex_table"]
        for key in expected_keys:
            assert key in paths
            assert paths[key].exists()
        
        # Verify JSON is valid
        import json
        with open(paths["summary"]) as f:
            summary = json.load(f)
        assert "total_experiments" in summary
    
    def test_generate_latex_table(self, evaluation_engine):
        """Test LaTeX table generation."""
        overall = evaluation_engine.get_overall_metrics_table()
        latex = evaluation_engine._generate_latex_table(overall)
        
        assert isinstance(latex, str)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "tabular" in latex
        assert "Experiment" in latex


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_run_evaluation(self, tmp_path):
        """Test run_evaluation function."""
        # First run experiments
        from src.experiments import run_experiments
        run_experiments(seed=MASTER_SEED)
        
        paths = run_evaluation(output_dir=tmp_path)
        
        assert isinstance(paths, dict)
        for key, path in paths.items():
            assert path.exists()
    
    def test_load_evaluation_summary(self, tmp_path):
        """Test loading evaluation summary."""
        from src.experiments import run_experiments
        from src.evaluation import load_evaluation_summary
        run_experiments(seed=MASTER_SEED)
        
        run_evaluation(output_dir=tmp_path)
        
        summary = load_evaluation_summary(tmp_path / "evaluation_summary.json")
        
        assert isinstance(summary, dict)
        assert "total_experiments" in summary
    
    def test_validate_evaluation_outputs(self, tmp_path):
        """Test validation of evaluation outputs."""
        from src.experiments import run_experiments
        run_experiments(seed=MASTER_SEED)
        
        run_evaluation(output_dir=tmp_path)
        
        validation = validate_evaluation_outputs(tmp_path)
        
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
        assert len(validation["files_found"]) >= 4


# ============================================================
# INTEGRATION TESTS
# ============================================================
class TestEvaluationIntegration:
    """Integration tests for evaluation."""
    
    def test_experiment_a_perfect_specificity(self, evaluation_engine):
        """Test Experiment A has perfect specificity (FPR=0)."""
        overall = evaluation_engine.get_overall_metrics_table()
        exp_a = overall[overall["experiment"] == "experiment_a_normal"]
        
        for _, row in exp_a.iterrows():
            assert row["false_positive_rate"] == 0.0
            assert row["fp"] == 0
            assert row["tn"] > 0
    
    def test_anomaly_experiments_perfect_detection(self, evaluation_engine):
        """Test anomaly experiments have perfect detection at threshold 3."""
        overall = evaluation_engine.get_overall_metrics_table()
        
        anomaly_experiments = ["experiment_b_traffic_spike", "experiment_c_destination_expansion", "experiment_d_combined"]
        
        for exp in anomaly_experiments:
            exp_df = overall[(overall["experiment"] == exp) & (overall["threshold"] == 3.0)]
            assert len(exp_df) == 1
            
            row = exp_df.iloc[0]
            # Should have perfect recall (all anomalies detected)
            assert row["recall"] == 1.0
            # Should have perfect precision (no false alarms on anomaly obs)
            assert row["precision"] == 1.0
            # Should have F1 = 1.0
            assert row["f1_score"] == 1.0
    
    def test_threshold_monotonicity(self, evaluation_engine):
        """Test that recall decreases and FPR decreases with higher threshold."""
        overall = evaluation_engine.get_overall_metrics_table()
        exp_e = overall[overall["experiment"] == "experiment_e_threshold_study"]
        
        thresholds = sorted(exp_e["threshold"].unique())
        
        recalls = [exp_e[exp_e["threshold"] == t]["recall"].values[0] for t in thresholds]
        fprs = [exp_e[exp_e["threshold"] == t]["false_positive_rate"].values[0] for t in thresholds]
        
        # Recall should be non-increasing with threshold
        for i in range(1, len(recalls)):
            assert recalls[i] <= recalls[i-1] + 1e-6
        
        # FPR should be non-increasing with threshold
        for i in range(1, len(fprs)):
            assert fprs[i] <= fprs[i-1] + 1e-6
    
    def test_all_experiments_have_metrics(self, evaluation_engine):
        """Test all 5 experiments have metrics for all 3 thresholds."""
        overall = evaluation_engine.get_overall_metrics_table()
        
        for exp in EXPERIMENT_NAMES:
            exp_df = overall[overall["experiment"] == exp]
            assert len(exp_df) == 3  # 3 thresholds
    
    def test_no_nan_in_metrics(self, evaluation_engine):
        """Test no NaN values in metrics."""
        overall = evaluation_engine.get_overall_metrics_table()
        
        numeric_cols = ["accuracy", "precision", "recall", "f1_score", "false_positive_rate"]
        for col in numeric_cols:
            assert not overall[col].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])