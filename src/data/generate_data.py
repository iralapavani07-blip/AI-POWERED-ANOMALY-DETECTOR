import pandas as pd
import numpy as np

np.random.seed(42)

rows = 10000

timestamps = pd.date_range(
    start="2025-01-01",
    periods=rows,
    freq="min"
)

cpu = np.random.normal(50, 10, rows)
memory = np.random.normal(60, 8, rows)
latency = np.random.normal(100, 20, rows)
error_rate = np.random.normal(1, 0.5, rows)
requests = np.random.normal(500, 100, rows)

anomaly_idx = np.random.choice(rows, 200)

cpu[anomaly_idx] += np.random.uniform(30, 50, 200)
latency[anomaly_idx] += np.random.uniform(200, 500, 200)
error_rate[anomaly_idx] += np.random.uniform(5, 15, 200)

is_anomaly = np.zeros(rows)
is_anomaly[anomaly_idx] = 1

df = pd.DataFrame({
    "timestamp": timestamps,
    "cpu_percent": cpu,
    "memory_percent": memory,
    "latency_ms": latency,
    "error_rate": error_rate,
    "requests_per_sec": requests,
    "is_anomaly": is_anomaly
})

df.to_csv(
    "data/raw/server_metrics.csv",
    index=False
)

print("Dataset Created Successfully")
print(df.shape)