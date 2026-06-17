"""
Data Loader Utilities
---------------------
Centralised functions to load all pipeline CSV outputs.
"""
import pandas as pd
import os


def load_raw_metrics(path: str = "data/raw/server_metrics.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def load_features(path: str = "data/processed/server_metrics_features.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def load_isolation_forest_results(path: str = "data/processed/isolation_forest_results.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_lstm_results(path: str = "data/processed/lstm_results.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_ensemble_results(path: str = "data/processed/ensemble_results.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_root_cause_results(path: str = "data/processed/root_cause_results.csv") -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def get_anomalies_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only anomaly rows."""
    return df[df["ensemble_anomaly"] == 1].reset_index(drop=True)


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return key summary statistics for dashboard KPI cards."""
    anomalies = df[df["ensemble_anomaly"] == 1]
    return {
        "total_records": len(df),
        "total_anomalies": int(df["ensemble_anomaly"].sum()),
        "anomaly_rate": round(df["ensemble_anomaly"].mean() * 100, 2),
        "critical_count": int((df["severity"] == "Critical").sum()) if "severity" in df.columns else 0,
        "high_count": int((df["severity"] == "High").sum()) if "severity" in df.columns else 0,
        "medium_count": int((df["severity"] == "Medium").sum()) if "severity" in df.columns else 0,
        "low_count": int((df["severity"] == "Low").sum()) if "severity" in df.columns else 0,
        "avg_cpu": round(anomalies["cpu_percent"].mean(), 2) if len(anomalies) else 0,
        "avg_latency": round(anomalies["latency_ms"].mean(), 2) if len(anomalies) else 0,
        "top_root_cause":  df[df["ensemble_anomaly"]==1]["root_cause"].mode()[0] if "root_cause" in df.columns and len(df) else "N/A",
    }
