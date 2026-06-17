"""
FastAPI REST Endpoint — AI-Powered Production Anomaly Detector
--------------------------------------------------------------
Run: uvicorn src.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd

from src.utils.data_loader import (
    load_root_cause_results,
    get_summary_stats,
    get_anomalies_only,
)

app = FastAPI(
    title="AI-Powered Production Anomaly Detector API",
    description=(
        "Real-time anomaly detection using Isolation Forest + LSTM Autoencoder ensemble. "
        "GenAI root cause explanation via LangChain + Gemini RAG pipeline."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response models ───────────────────────────────────────────────────────────
class SummaryResponse(BaseModel):
    total_records: int
    total_anomalies: int
    anomaly_rate: float
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    top_root_cause: str
    avg_cpu_during_anomaly: float
    avg_latency_during_anomaly: float


class AnomalyRecord(BaseModel):
    timestamp: str
    cpu_percent: float
    memory_percent: float
    latency_ms: float
    error_rate: float
    requests_per_sec: float
    ensemble_anomaly: int
    severity: str
    root_cause: str


class ExplainRequest(BaseModel):
    timestamp: str = Field(..., example="2025-01-15 02:14:00")
    cpu_percent: float = Field(..., example=87.3)
    memory_percent: float = Field(..., example=72.1)
    latency_ms: float = Field(..., example=940.0)
    error_rate: float = Field(..., example=14.2)
    requests_per_sec: float = Field(..., example=1840.0)
    severity: Optional[str] = Field("High", example="High")
    root_cause: Optional[str] = Field("High CPU Usage", example="High CPU Usage")


class ExplainResponse(BaseModel):
    timestamp: str
    severity: str
    explanation: str
    model: str = "gemini-1.5-flash + FAISS RAG"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "project": "AI-Powered Production Anomaly Detector",
        "version": "1.0.0",
        "author": "Irala Pavani",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "pipeline": "Isolation Forest + LSTM + GenAI RAG"}


@app.get("/summary", response_model=SummaryResponse, tags=["Analytics"])
def get_summary():
    """Get overall summary statistics from the anomaly detection pipeline."""
    try:
        df = load_root_cause_results()
        stats = get_summary_stats(df)
        return SummaryResponse(
            total_records=stats["total_records"],
            total_anomalies=stats["total_anomalies"],
            anomaly_rate=stats["anomaly_rate"],
            critical_count=stats["critical_count"],
            high_count=stats["high_count"],
            medium_count=stats["medium_count"],
            low_count=stats["low_count"],
            top_root_cause=stats["top_root_cause"],
            avg_cpu_during_anomaly=stats["avg_cpu"],
            avg_latency_during_anomaly=stats["avg_latency"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies", tags=["Analytics"])
def get_anomalies(limit: int = 50, severity: Optional[str] = None):
    """
    Get detected anomalies.
    - **limit**: max number of records to return (default 50)
    - **severity**: filter by severity (Critical / High / Medium / Low)
    """
    try:
        df = load_root_cause_results()
        anomalies = get_anomalies_only(df)
        if severity:
            anomalies = anomalies[anomalies["severity"] == severity]
        result = anomalies.head(limit)[
            ["timestamp", "cpu_percent", "memory_percent",
             "latency_ms", "error_rate", "requests_per_sec",
             "ensemble_anomaly", "severity", "root_cause"]
        ]
        return {
            "count": len(result),
            "severity_filter": severity,
            "anomalies": result.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies/severity-breakdown", tags=["Analytics"])
def severity_breakdown():
    """Get count of anomalies per severity level."""
    try:
        df = load_root_cause_results()
        anomalies = get_anomalies_only(df)
        breakdown = anomalies["severity"].value_counts().to_dict()
        return {"severity_breakdown": breakdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomalies/root-causes", tags=["Analytics"])
def top_root_causes(top_n: int = 10):
    """Get top N most common root causes."""
    try:
        df = load_root_cause_results()
        anomalies = get_anomalies_only(df)
        counts = anomalies["root_cause"].value_counts().head(top_n).to_dict()
        return {"top_root_causes": counts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse, tags=["GenAI"])
def explain_anomaly(request: ExplainRequest):
    """
    Generate a GenAI root cause explanation for a given anomaly using
    LangChain RAG + Gemini API + FAISS vector search over 50 past incidents.
    """
    try:
        from src.rag.explainer import AnomalyExplainer
        explainer = AnomalyExplainer()
        explanation = explainer.explain(request.dict())
        return ExplainResponse(
            timestamp=request.timestamp,
            severity=request.severity or "Unknown",
            explanation=explanation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GenAI explanation failed: {str(e)}")
