import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/server_metrics.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# CPU Usage Plot
plt.figure(figsize=(12,5))
plt.plot(df["cpu_percent"])
plt.title("CPU Usage Over Time")
plt.xlabel("Index")
plt.ylabel("CPU %")
plt.show()

# Memory Usage Plot
plt.figure(figsize=(12,5))
plt.plot(df["memory_percent"])
plt.title("Memory Usage Over Time")
plt.xlabel("Index")
plt.ylabel("Memory %")
plt.show()

# Latency Histogram
plt.figure(figsize=(8,5))
plt.hist(df["latency_ms"], bins=30)
plt.title("Latency Distribution")
plt.xlabel("Latency")
plt.ylabel("Frequency")
plt.show()

