"""
GenAI Root Cause Explainer
--------------------------
Uses LangChain + Gemini API + FAISS RAG to generate plain-English
explanations for detected anomalies, grounded in past incident history.
"""

import os
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# ── 50 past incident knowledge base ─────────────────────────────────────────
PAST_INCIDENTS = [
    {"id": "INC-001", "title": "CPU spike on checkout service",
     "metrics": "cpu=89%, latency=840ms, error_rate=12%",
     "root_cause": "3x traffic surge on /checkout endpoint due to flash sale. Autoscaling lag caused CPU to spike before new pods came online.",
     "resolution": "Scaled checkout pods from 3 to 8. Implemented predictive autoscaling.", "duration_min": 23},
    {"id": "INC-002", "title": "Memory leak in user-service",
     "metrics": "memory=94%, latency=1200ms, error_rate=8%",
     "root_cause": "Newly deployed version had unclosed database connection pool. Memory grew linearly over 4 hours.",
     "resolution": "Rolled back to previous version. Fixed connection pool closure.", "duration_min": 67},
    {"id": "INC-003", "title": "Latency surge on payment API",
     "metrics": "latency=950ms, cpu=72%, error_rate=6%",
     "root_cause": "Third-party payment gateway timeout causing requests to queue up. Thread pool exhaustion.",
     "resolution": "Added circuit breaker pattern. Set 2s timeout on payment gateway calls.", "duration_min": 41},
    {"id": "INC-004", "title": "High error rate on login endpoint",
     "metrics": "error_rate=18%, cpu=55%, latency=300ms",
     "root_cause": "Database certificate expired at midnight, causing all auth queries to fail.",
     "resolution": "Renewed SSL certificate. Added certificate expiry monitoring alert.", "duration_min": 19},
    {"id": "INC-005", "title": "Traffic spike causing cascading failures",
     "metrics": "requests=1800/s, cpu=91%, memory=88%, error_rate=15%",
     "root_cause": "Viral social media post drove unexpected traffic. Rate limiting not configured.",
     "resolution": "Enabled rate limiting. Added CDN caching for static resources.", "duration_min": 55},
    {"id": "INC-006", "title": "Disk I/O bottleneck on logging service",
     "metrics": "cpu=45%, latency=600ms, error_rate=3%",
     "root_cause": "Log rotation failed, disk 98% full. All write operations slowed.",
     "resolution": "Cleared old logs. Fixed log rotation cron job.", "duration_min": 30},
    {"id": "INC-007", "title": "CPU spike from background job",
     "metrics": "cpu=88%, memory=65%, latency=200ms",
     "root_cause": "Nightly batch report job scheduled incorrectly during peak hours.",
     "resolution": "Rescheduled batch job to 3 AM. Added CPU usage alert for batch jobs.", "duration_min": 25},
    {"id": "INC-008", "title": "Memory spike from cache not expiring",
     "metrics": "memory=92%, cpu=60%, latency=180ms",
     "root_cause": "Redis cache TTL not set. Cache grew unbounded over 12 hours.",
     "resolution": "Set TTL on all cache keys. Added memory usage alert at 80%.", "duration_min": 35},
    {"id": "INC-009", "title": "Network partition between microservices",
     "metrics": "error_rate=22%, latency=2000ms, cpu=40%",
     "root_cause": "Firewall rule change blocked inter-service communication on port 8080.",
     "resolution": "Reverted firewall change. Added network connectivity health checks.", "duration_min": 48},
    {"id": "INC-010", "title": "Database connection pool exhaustion",
     "metrics": "error_rate=9%, latency=1500ms, cpu=35%",
     "root_cause": "Sudden spike in concurrent users exhausted the DB connection pool of 100 connections.",
     "resolution": "Increased pool size to 250. Added connection wait timeout.", "duration_min": 20},
    {"id": "INC-011", "title": "API gateway CPU overload",
     "metrics": "cpu=93%, requests=2100/s, latency=400ms",
     "root_cause": "Bot traffic scraping the product catalog at high frequency.",
     "resolution": "Blocked bot IPs. Added rate limiting per IP: 100 req/min.", "duration_min": 15},
    {"id": "INC-012", "title": "High latency from DNS resolution failure",
     "metrics": "latency=1800ms, error_rate=5%, cpu=30%",
     "root_cause": "Internal DNS server overloaded, causing slow resolution for service discovery.",
     "resolution": "Added secondary DNS server. Implemented DNS caching at service level.", "duration_min": 38},
    {"id": "INC-013", "title": "Memory leak from websocket connections",
     "metrics": "memory=90%, cpu=50%, latency=250ms",
     "root_cause": "WebSocket connections not being closed after user disconnects. 50k orphaned connections.",
     "resolution": "Added connection cleanup on disconnect. Implemented connection health ping.", "duration_min": 60},
    {"id": "INC-014", "title": "Error spike from bad deployment",
     "metrics": "error_rate=25%, cpu=60%, latency=350ms",
     "root_cause": "New deployment had missing environment variable for payment service URL.",
     "resolution": "Rolled back deployment. Added env var validation in CI/CD pipeline.", "duration_min": 12},
    {"id": "INC-015", "title": "CPU overload from recursive API calls",
     "metrics": "cpu=97%, requests=5000/s, latency=800ms",
     "root_cause": "Bug in recommendation service caused circular API calls between two services.",
     "resolution": "Hotfix deployed to break circular dependency. Added call depth limiting.", "duration_min": 22},
    {"id": "INC-016", "title": "Latency from slow third-party analytics",
     "metrics": "latency=700ms, cpu=40%, error_rate=2%",
     "root_cause": "Synchronous calls to analytics API added 500ms to every user request.",
     "resolution": "Made analytics calls asynchronous. Requests no longer wait for analytics.", "duration_min": 45},
    {"id": "INC-017", "title": "High error rate from SSL certificate mismatch",
     "metrics": "error_rate=30%, latency=100ms, cpu=25%",
     "root_cause": "Load balancer SSL cert renewed but not deployed to all nodes. Mixed cert versions.",
     "resolution": "Deployed cert to all nodes. Added cert consistency check to health probe.", "duration_min": 18},
    {"id": "INC-018", "title": "Memory pressure from large file uploads",
     "metrics": "memory=88%, cpu=70%, latency=900ms",
     "root_cause": "Large file upload feature loaded entire file into memory instead of streaming.",
     "resolution": "Refactored to streaming upload. Added file size limit of 100MB.", "duration_min": 50},
    {"id": "INC-019", "title": "Traffic spike from email campaign",
     "metrics": "requests=1600/s, cpu=85%, memory=75%",
     "root_cause": "Marketing email sent to 500k users simultaneously. All clicked within 2 minutes.",
     "resolution": "Pre-scaled before next campaign. Added email delivery rate limiting.", "duration_min": 33},
    {"id": "INC-020", "title": "Slow queries from missing database index",
     "metrics": "latency=1100ms, cpu=45%, error_rate=4%",
     "root_cause": "New feature added filter on unindexed column. Full table scan on 10M rows.",
     "resolution": "Added composite index. Query time reduced from 1100ms to 12ms.", "duration_min": 40},
    {"id": "INC-021", "title": "CPU spike from image processing",
     "metrics": "cpu=95%, memory=80%, latency=600ms",
     "root_cause": "New image resize feature ran synchronously on web workers. CPU-intensive.",
     "resolution": "Moved image processing to background queue. Web workers freed instantly.", "duration_min": 28},
    {"id": "INC-022", "title": "Error surge from downstream service timeout",
     "metrics": "error_rate=12%, latency=3000ms, cpu=35%",
     "root_cause": "Inventory service deployed slow version. Caused order service to timeout waiting.",
     "resolution": "Rolled back inventory service. Added fallback response for timeouts.", "duration_min": 36},
    {"id": "INC-023", "title": "Memory spike from log accumulation",
     "metrics": "memory=87%, cpu=55%, latency=220ms",
     "root_cause": "Debug logging left on in production after testing. 10GB logs in 6 hours.",
     "resolution": "Disabled debug logging. Implemented log level configuration per environment.", "duration_min": 17},
    {"id": "INC-024", "title": "Latency from N+1 database query bug",
     "metrics": "latency=1400ms, cpu=65%, error_rate=1%",
     "root_cause": "ORM lazy loading caused 1 DB query per item in a 500-item list response.",
     "resolution": "Fixed with eager loading. Queries reduced from 501 to 2.", "duration_min": 52},
    {"id": "INC-025", "title": "CPU and memory spike at month end",
     "metrics": "cpu=90%, memory=85%, latency=500ms",
     "root_cause": "Month-end billing report job ran unexpectedly during business hours due to timezone misconfiguration.",
     "resolution": "Fixed timezone in scheduler. Added job isolation to prevent resource contention.", "duration_min": 44},
]
# Add 25 more lightweight incidents to reach 50
for i in range(26, 51):
    PAST_INCIDENTS.append({
        "id": f"INC-{i:03d}",
        "title": f"Production incident pattern {i}",
        "metrics": "cpu=80%, memory=75%, latency=400ms, error_rate=7%",
        "root_cause": f"Resource contention pattern {i}: Combination of traffic surge and insufficient capacity caused service degradation.",
        "resolution": "Scaled horizontally and added monitoring alert for early detection.",
        "duration_min": 30
    })


class AnomalyExplainer:
    """
    GenAI RAG pipeline for anomaly root cause explanation.
    Uses sentence-transformers + FAISS for retrieval,
    Gemini API for natural language explanation generation.
    """

    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.incidents = PAST_INCIDENTS
        self._build_index()
        self.llm = genai.GenerativeModel("models/gemini-2.5-flash")

    def _build_index(self):
        """Embed all past incidents into FAISS index."""
        texts = [
            f"Incident: {inc['title']}. "
            f"Metrics: {inc['metrics']}. "
            f"Root cause: {inc['root_cause']}. "
            f"Resolution: {inc['resolution']}."
            for inc in self.incidents
        ]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def _retrieve_similar(self, anomaly_description: str, top_k: int = 3) -> list:
        """Retrieve top-k most similar past incidents using cosine similarity."""
        q_vec = self.embedder.encode([anomaly_description]).astype(np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "incident": self.incidents[idx],
                    "similarity": round(float(score), 3)
                })
        return results

    def explain(self, anomaly_row: dict) -> str:
        """
        Generate plain-English root cause explanation for a detected anomaly.

        Parameters
        ----------
        anomaly_row : dict
            Keys: timestamp, cpu_percent, memory_percent, latency_ms,
                  error_rate, requests_per_sec, severity, root_cause
        """
        # Build anomaly description for retrieval
        anomaly_desc = (
            f"Anomaly: cpu={anomaly_row.get('cpu_percent', 0):.1f}%, "
            f"memory={anomaly_row.get('memory_percent', 0):.1f}%, "
            f"latency={anomaly_row.get('latency_ms', 0):.0f}ms, "
            f"error_rate={anomaly_row.get('error_rate', 0):.1f}%, "
            f"requests={anomaly_row.get('requests_per_sec', 0):.0f}/s"
        )

        # Retrieve similar past incidents
        similar = self._retrieve_similar(anomaly_desc, top_k=3)
        similar_text = ""
        for i, s in enumerate(similar, 1):
            inc = s["incident"]
            similar_text += (
                f"\nPast Incident {i} (similarity: {s['similarity']}):\n"
                f"  Title: {inc['title']}\n"
                f"  Root Cause: {inc['root_cause']}\n"
                f"  Resolution: {inc['resolution']}\n"
                f"  Duration: {inc['duration_min']} minutes\n"
            )

        prompt = f"""You are a senior Site Reliability Engineer at a top tech company.
A production anomaly has been detected. Analyze it and provide a clear, actionable explanation.

CURRENT ANOMALY:
- Timestamp: {anomaly_row.get('timestamp', 'N/A')}
- CPU Usage: {anomaly_row.get('cpu_percent', 0):.1f}% (normal: 30-55%)
- Memory Usage: {anomaly_row.get('memory_percent', 0):.1f}% (normal: 40-70%)
- Latency: {anomaly_row.get('latency_ms', 0):.0f}ms (normal: 80-150ms)
- Error Rate: {anomaly_row.get('error_rate', 0):.1f}% (normal: <1%)
- Requests/sec: {anomaly_row.get('requests_per_sec', 0):.0f} (normal: 300-700)
- Severity: {anomaly_row.get('severity', 'Unknown')}
- Rule-based Root Cause: {anomaly_row.get('root_cause', 'Unknown')}

SIMILAR PAST INCIDENTS FROM KNOWLEDGE BASE:
{similar_text}

Respond in this exact format (keep under 180 words total):

**ANOMALY SUMMARY**
[One sentence: what metric is abnormal and by how much]

**MOST LIKELY ROOT CAUSE**
[2-3 sentences. Be specific. Reference which metrics point to this cause.]

**CONFIDENCE**
[High / Medium / Low — one sentence why]

**RECOMMENDED ACTIONS**
1. [Do this in the next 5 minutes]
2. [Do this in the next hour]
3. [Long-term prevention]

**SIMILAR PAST INCIDENT**
[Reference the most relevant past incident and how it was resolved]"""

        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return (
                f"**ANOMALY SUMMARY**\n"
                f"Anomaly detected — CPU: {anomaly_row.get('cpu_percent',0):.1f}%, "
                f"Latency: {anomaly_row.get('latency_ms',0):.0f}ms, "
                f"Error Rate: {anomaly_row.get('error_rate',0):.1f}%\n\n"
                f"**NOTE**: GenAI explanation unavailable (set GEMINI_API_KEY in .env). "
                f"Rule-based root cause: {anomaly_row.get('root_cause', 'Unknown')}\n\n"
                f"**RECOMMENDED ACTIONS**\n"
                f"1. Check metrics that are above normal threshold\n"
                f"2. Review recent deployments and config changes\n"
                f"3. Add GEMINI_API_KEY to .env for AI-powered explanations"
            )


# ── Standalone usage ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    print("Building incident knowledge base and FAISS index...")
    explainer = AnomalyExplainer()

    # Load a real anomaly from processed data
    df = pd.read_csv("data/processed/root_cause_results.csv")
    anomalies = df[df["ensemble_anomaly"] == 1]

    if len(anomalies) == 0:
        print("No anomalies found in ensemble results.")
    else:
        sample = anomalies.iloc[0].to_dict()
        print(f"\nExplaining anomaly at {sample.get('timestamp', 'N/A')}...")
        print("-" * 60)
        explanation = explainer.explain(sample)
        print(explanation)
        print("-" * 60)
        print(f"\nTotal anomalies in dataset: {len(anomalies)}")
