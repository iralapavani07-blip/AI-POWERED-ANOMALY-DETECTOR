import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/server_metrics.csv")

# Rolling Mean
df["cpu_rolling_mean"] = df["cpu_percent"].rolling(window=10).mean()

# Rolling Standard Deviation
df["cpu_rolling_std"] = df["cpu_percent"].rolling(window=10).std()

# CPU Change Rate
df["cpu_change"] = df["cpu_percent"].diff()

# Latency Change Rate
df["latency_change"] = df["latency_ms"].diff()

# Fill missing values created by rolling/diff
df = df.fillna(0)

# Save processed dataset
df.to_csv(
    "data/processed/server_metrics_features.csv",
    index=False
)

print("Feature Engineering Completed")
print(df.shape)

print("\nNew Features Added:")
print([
    "cpu_rolling_mean",
    "cpu_rolling_std",
    "cpu_change",
    "latency_change"
])