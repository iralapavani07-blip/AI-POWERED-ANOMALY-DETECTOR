"""
Unit Tests — AI-Powered Production Anomaly Detector
----------------------------------------------------
Run: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np


# ── Data generation tests ────────────────────────────────────────────────────
class TestDataGeneration:

    def test_data_file_exists(self):
        assert os.path.exists("data/raw/server_metrics.csv"), \
            "Raw data file missing — run src/data/generate_data.py first"

    def test_data_shape(self):
        df = pd.read_csv("data/raw/server_metrics.csv")
        assert df.shape[0] == 10000, f"Expected 10000 rows, got {df.shape[0]}"
        assert df.shape[1] == 7, f"Expected 7 columns, got {df.shape[1]}"

    def test_required_columns(self):
        df = pd.read_csv("data/raw/server_metrics.csv")
        required = ["timestamp", "cpu_percent", "memory_percent",
                    "latency_ms", "error_rate", "requests_per_sec", "is_anomaly"]
        for col in required:
            assert col in df.columns, f"Column '{col}' missing from raw data"

    def test_anomaly_rate(self):
        df = pd.read_csv("data/raw/server_metrics.csv")
        rate = df["is_anomaly"].mean()
        assert 0.01 <= rate <= 0.05, \
            f"Anomaly rate {rate:.3f} outside expected range 1%-5%"

    def test_no_nulls_in_raw(self):
        df = pd.read_csv("data/raw/server_metrics.csv")
        assert df.isnull().sum().sum() == 0, "Raw data has null values"

    def test_cpu_range(self):
        df = pd.read_csv("data/raw/server_metrics.csv")
        # Anomaly spikes can exceed 100 in synthetic data — just check base is reasonable
        assert df["cpu_percent"].min() > 0, "CPU has non-positive values"


# ── Feature engineering tests ────────────────────────────────────────────────
class TestFeatureEngineering:

    def test_features_file_exists(self):
        assert os.path.exists("data/processed/server_metrics_features.csv"), \
            "Features file missing — run src/features/feature_engineering.py first"

    def test_new_features_present(self):
        df = pd.read_csv("data/processed/server_metrics_features.csv")
        new_features = ["cpu_rolling_mean", "cpu_rolling_std", "cpu_change", "latency_change"]
        for feat in new_features:
            assert feat in df.columns, f"Feature '{feat}' not found"

    def test_no_nulls_after_engineering(self):
        df = pd.read_csv("data/processed/server_metrics_features.csv")
        assert df.isnull().sum().sum() == 0, \
            "Feature engineering left null values — check fillna(0)"


# ── Isolation Forest tests ────────────────────────────────────────────────────
class TestIsolationForest:

    def test_iforest_results_exist(self):
        assert os.path.exists("data/processed/isolation_forest_results.csv"), \
            "Isolation Forest results missing — run src/models/isolation_forest.py"

    def test_iforest_prediction_column(self):
        df = pd.read_csv("data/processed/isolation_forest_results.csv")
        assert "iforest_prediction" in df.columns, \
            "'iforest_prediction' column missing"

    def test_iforest_prediction_values(self):
        df = pd.read_csv("data/processed/isolation_forest_results.csv")
        unique_vals = set(df["iforest_prediction"].unique())
        assert unique_vals.issubset({1, -1}), \
            f"IForest predictions should be 1 or -1, got {unique_vals}"

    def test_iforest_anomaly_rate(self):
        df = pd.read_csv("data/processed/isolation_forest_results.csv")
        anomaly_rate = (df["iforest_prediction"] == -1).mean()
        # contamination=0.02, so expect around 2% anomalies
        assert 0.01 <= anomaly_rate <= 0.06, \
            f"IForest anomaly rate {anomaly_rate:.3f} outside expected range"


# ── LSTM tests ───────────────────────────────────────────────────────────────
class TestLSTM:

    def test_lstm_model_saved(self):
        assert os.path.exists("models/lstm_autoencoder.keras"), \
            "LSTM model not saved — run src/models/lstm_autoencoder.py"

    def test_lstm_results_exist(self):
        assert os.path.exists("data/processed/lstm_results.csv"), \
            "LSTM results missing — run src/models/lstm_anomaly_scoring.py"

    def test_lstm_anomaly_column(self):
        df = pd.read_csv("data/processed/lstm_results.csv")
        assert "lstm_anomaly" in df.columns, "'lstm_anomaly' column missing"
        assert "lstm_error" in df.columns, "'lstm_error' column missing"

    def test_lstm_error_non_negative(self):
        df = pd.read_csv("data/processed/lstm_results.csv")
        assert (df["lstm_error"] >= 0).all(), "LSTM reconstruction errors should be non-negative"


# ── Ensemble tests ────────────────────────────────────────────────────────────
class TestEnsemble:

    def test_ensemble_results_exist(self):
        assert os.path.exists("data/processed/ensemble_results.csv"), \
            "Ensemble results missing — run src/models/ensemble_detector.py"

    def test_ensemble_column_present(self):
        df = pd.read_csv("data/processed/ensemble_results.csv")
        assert "ensemble_anomaly" in df.columns, "'ensemble_anomaly' column missing"

    def test_ensemble_binary_values(self):
        df = pd.read_csv("data/processed/ensemble_results.csv")
        unique = set(df["ensemble_anomaly"].unique())
        assert unique.issubset({0, 1}), f"ensemble_anomaly should be 0 or 1, got {unique}"


# ── Root cause tests ──────────────────────────────────────────────────────────
class TestRootCause:

    def test_root_cause_results_exist(self):
        assert os.path.exists("data/processed/root_cause_results.csv"), \
            "Root cause results missing — run src/models/root_cause_analysis.py"

    def test_root_cause_columns(self):
        df = pd.read_csv("data/processed/root_cause_results.csv")
        assert "root_cause" in df.columns, "'root_cause' column missing"
        assert "severity" in df.columns, "'severity' column missing"

    def test_severity_valid_values(self):
        df = pd.read_csv("data/processed/root_cause_results.csv")
        valid = {"Critical", "High", "Medium", "Low"}
        actual = set(df["severity"].unique())
        assert actual.issubset(valid), \
            f"Invalid severity values found: {actual - valid}"

    def test_no_empty_root_causes(self):
        df = pd.read_csv("data/processed/root_cause_results.csv")
        assert df["root_cause"].isnull().sum() == 0, "Some root causes are null"
        assert (df["root_cause"] == "").sum() == 0, "Some root causes are empty string"


# ── Data loader utility tests ─────────────────────────────────────────────────
class TestDataLoader:

    def test_load_root_cause_results(self):
        from src.utils.data_loader import load_root_cause_results
        df = load_root_cause_results()
        assert len(df) > 0, "Root cause results loaded empty"

    def test_get_summary_stats(self):
        from src.utils.data_loader import load_root_cause_results, get_summary_stats
        df = load_root_cause_results()
        stats = get_summary_stats(df)
        assert "total_records" in stats
        assert "total_anomalies" in stats
        assert stats["total_records"] > 0
        assert 0 <= stats["anomaly_rate"] <= 100
