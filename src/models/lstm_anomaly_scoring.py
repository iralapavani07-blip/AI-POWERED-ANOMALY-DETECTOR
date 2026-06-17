import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv(
    "data/processed/server_metrics_features.csv"
)

features = [
    "cpu_percent",
    "memory_percent",
    "latency_ms",
    "error_rate",
    "requests_per_sec"
]

data = df[features]

# Scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
sequence_length = 10

X = []

for i in range(sequence_length, len(scaled_data)):
    X.append(
        scaled_data[i-sequence_length:i]
    )

X = np.array(X)

# Load trained model
model = load_model(
    "models/lstm_autoencoder.keras"
)

# Predict
predictions = model.predict(X)

# Reconstruction error
errors = np.mean(
    np.square(
        scaled_data[sequence_length:] - predictions
    ),
    axis=1
)

# Threshold
threshold = np.percentile(
    errors,
    98
)

anomalies = (
    errors > threshold
).astype(int)

# Align with original dataframe
df = df.iloc[sequence_length:].copy()

df["lstm_error"] = errors
df["lstm_anomaly"] = anomalies

# Save
df.to_csv(
    "data/processed/lstm_results.csv",
    index=False
)

print("LSTM Anomaly Scoring Complete")

print("\nThreshold:")
print(threshold)

print("\nAnomaly Count:")
print(df["lstm_anomaly"].sum())