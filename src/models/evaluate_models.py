"""
Model Evaluation
-----------------
Computes precision, recall, and F1 score for each model
against ground truth is_anomaly labels.

Run: python src/models/evaluate_models.py
Put these numbers in your resume and README!
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# Load all results
iforest  = pd.read_csv("data/processed/isolation_forest_results.csv")
lstm     = pd.read_csv("data/processed/lstm_results.csv")
ensemble = pd.read_csv("data/processed/ensemble_results.csv")

# Align ground truth
y_true_if  = iforest.iloc[10:]["is_anomaly"].reset_index(drop=True).astype(int)
y_true_seq = lstm["is_anomaly"].astype(int)

# Isolation Forest: -1 = anomaly → convert to 1
y_pred_if  = (iforest.iloc[10:]["iforest_prediction"].reset_index(drop=True) == -1).astype(int)
y_pred_lstm = lstm["lstm_anomaly"].astype(int)
y_pred_ens  = ensemble["ensemble_anomaly"].astype(int)

print("=" * 60)
print("  MODEL EVALUATION REPORT")
print("  AI-Powered Production Anomaly Detector")
print("=" * 60)

for name, y_true, y_pred in [
    ("Isolation Forest",   y_true_if,  y_pred_if),
    ("LSTM Autoencoder",   y_true_seq, y_pred_lstm),
    ("Ensemble (IF+LSTM)", y_true_seq, y_pred_ens),
]:
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n── {name}")
    print(f"   Precision : {p:.4f}")
    print(f"   Recall    : {r:.4f}")
    print(f"   F1 Score  : {f1:.4f}")
    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}")

print("\n" + "=" * 60)
print("  Copy these numbers into your resume and README!")
print("=" * 60)
