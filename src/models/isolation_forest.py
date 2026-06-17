import pandas as pd
from sklearn.ensemble import IsolationForest

# Load processed dataset
df = pd.read_csv("data/processed/server_metrics_features.csv")

# Features for model
features = [
    "cpu_percent",
    "memory_percent",
    "latency_ms",
    "error_rate",
    "requests_per_sec",
    "cpu_rolling_mean",
    "cpu_rolling_std",
    "cpu_change",
    "latency_change"
]

X = df[features]

# Train Isolation Forest
model = IsolationForest(
    contamination=0.02,
    random_state=42
)

model.fit(X)

# Predictions
predictions = model.predict(X)

# Convert predictions
df["iforest_prediction"] = predictions

# Save results
df.to_csv(
    "data/processed/isolation_forest_results.csv",
    index=False
)

print("Isolation Forest Training Complete")

print("\nPrediction Counts:")
print(df["iforest_prediction"].value_counts())