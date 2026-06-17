import pandas as pd

df = pd.read_csv(
    "data/processed/root_cause_results.csv"
)

anomalies = df[
    df["ensemble_anomaly"] == 1
]

print(
    anomalies[
        [
            "cpu_percent",
            "memory_percent",
            "latency_ms",
            "error_rate",
            "root_cause",
            "severity"
        ]
    ].head(20)
)