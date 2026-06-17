"""
LSTM Autoencoder — Anomaly Detection
-------------------------------------
An autoencoder learns to reconstruct NORMAL sequences.
When it sees an anomaly (never trained on), reconstruction error is HIGH.
High error = anomaly. This is why we train ONLY on normal data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv("data/processed/server_metrics_features.csv")

FEATURES = [
    "cpu_percent", "memory_percent", "latency_ms",
    "error_rate", "requests_per_sec"
]
SEQUENCE_LENGTH = 10

# ── IMPORTANT: Train ONLY on normal data ─────────────────────────────────────
normal_df = df[df["is_anomaly"] == 0][FEATURES]
print(f"Training on {len(normal_df)} normal rows (excluded {(df['is_anomaly']==1).sum()} anomalies)")

# Scale
scaler = MinMaxScaler()
scaled_normal = scaler.fit_transform(normal_df)

# Create sequences from NORMAL data only
X_train = []
for i in range(SEQUENCE_LENGTH, len(scaled_normal)):
    X_train.append(scaled_normal[i - SEQUENCE_LENGTH:i])
X_train = np.array(X_train)
print(f"Training sequence shape: {X_train.shape}")

# ── LSTM Autoencoder architecture ─────────────────────────────────────────────
# Encoder compresses sequence → Decoder reconstructs it
# High reconstruction error on anomalies = detected
model = Sequential([
    LSTM(64, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2]),
         return_sequences=False),
    RepeatVector(X_train.shape[1]),               # bridge encoder→decoder
    LSTM(64, activation="relu", return_sequences=True),
    TimeDistributed(Dense(X_train.shape[2]))       # reconstruct each timestep
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# Train with early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
history = model.fit(
    X_train, X_train,                             # autoencoder: input = target
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Save model
model.save("models/lstm_autoencoder.keras")
print("\n✅  LSTM Autoencoder saved to models/lstm_autoencoder.keras")
print(f"   Final training loss:   {history.history['loss'][-1]:.6f}")
print(f"   Final validation loss: {history.history['val_loss'][-1]:.6f}")
