import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/processed/isolation_forest_results.csv"
)

plt.figure(figsize=(14,6))

normal = df[df["iforest_prediction"] == 1]
anomaly = df[df["iforest_prediction"] == -1]

plt.scatter(
    normal.index,
    normal["cpu_percent"],
    label="Normal",
    s=10
)

plt.scatter(
    anomaly.index,
    anomaly["cpu_percent"],
    label="Anomaly",
    s=20
)

plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Index")
plt.ylabel("CPU Usage")
plt.legend()

plt.show()