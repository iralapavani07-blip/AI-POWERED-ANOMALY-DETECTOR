"""
run_pipeline.py
---------------
Runs the complete anomaly detection pipeline end-to-end.
Usage: python run_pipeline.py
"""
import subprocess
import sys
import time

STEPS = [
    ("Step 1/6 — Generating synthetic server metrics",
     "src/data/generate_data.py"),
    ("Step 2/6 — Feature engineering",
     "src/features/feature_engineering.py"),
    ("Step 3/6 — Training Isolation Forest",
     "src/models/isolation_forest.py"),
    ("Step 4/6 — Training LSTM Autoencoder",
     "src/models/lstm_autoencoder.py"),
    ("Step 5/6 — LSTM anomaly scoring",
     "src/models/lstm_anomaly_scoring.py"),
    ("Step 6/6 — Ensemble detection + root cause analysis",
     "src/models/ensemble_detector.py"),
    ("Step 6b  — Root cause + severity scoring",
     "src/models/root_cause_analysis.py"),
]

def run_step(description, script):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = round(time.time() - start, 1)
    if result.returncode != 0:
        print(f"\n❌  FAILED: {script}")
        sys.exit(1)
    print(f"✅  Done in {elapsed}s")

if __name__ == "__main__":
    print("\n🚀  AI-Powered Production Anomaly Detector — Full Pipeline")
    print("   Author: Irala Pavani | B.Tech Data Science 2026\n")

    for desc, script in STEPS:
        run_step(desc, script)

    print("\n" + "="*60)
    print("  ✅  Pipeline complete!")
    print("  📊  Launch dashboard: streamlit run dashboard/app.py")
    print("  🔌  Launch API:       uvicorn src.api.main:app --reload")
    print("  🧪  Run tests:        pytest tests/ -v")
    print("="*60 + "\n")
