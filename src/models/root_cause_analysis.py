import pandas as pd

# Load ensemble results
df = pd.read_csv(
    "data/processed/ensemble_results.csv"
)

root_causes = []

for _, row in df.iterrows():

    reasons = []

    if row["cpu_percent"] > 85:
        reasons.append("High CPU Usage")

    if row["memory_percent"] > 80:
        reasons.append("High Memory Usage")

    if row["latency_ms"] > 200:
        reasons.append("High Latency")

    if row["error_rate"] > 5:
        reasons.append("High Error Rate")

    if row["requests_per_sec"] > 1000:
        reasons.append("Traffic Spike")

    if len(reasons) == 0:
        reasons.append("Unknown Pattern")

    root_causes.append(
        ", ".join(reasons)
    )

df["root_cause"] = root_causes

# Severity
severity = []

for _, row in df.iterrows():

    score = 0

    if row["cpu_percent"] > 85:
        score += 1

    if row["memory_percent"] > 80:
        score += 1

    if row["latency_ms"] > 200:
        score += 1

    if row["error_rate"] > 5:
        score += 1

    if score >= 3:
        severity.append("Critical")

    elif score >= 2:
        severity.append("High")

    elif score >= 1:
        severity.append("Medium")

    else:
        severity.append("Low")

df["severity"] = severity

df.to_csv(
    "data/processed/root_cause_results.csv",
    index=False
)

print("Root Cause Analysis Complete")

print("\nSeverity Counts:")

print(
    df["severity"].value_counts()
)