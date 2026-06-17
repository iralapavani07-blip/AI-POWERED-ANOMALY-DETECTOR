
"""
Streamlit Dashboard — AI-Powered Production Anomaly Detector
-------------------------------------------------------------
Run: streamlit run dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from src.utils.data_loader import load_root_cause_results, get_summary_stats, get_anomalies_only

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Powered Anomaly Detector",
    page_icon="🚨",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return load_root_cause_results()

df = load_data()
stats = get_summary_stats(df)
anomalies = get_anomalies_only(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚨 AI-Powered Production Anomaly Detector")
st.caption("Isolation Forest + LSTM Autoencoder + GenAI Root Cause Explainer | Irala Pavani")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records",    f"{stats['total_records']:,}")
c2.metric("Total Anomalies",  f"{stats['total_anomalies']:,}", f"{stats['anomaly_rate']}%")
c3.metric("Critical",         stats["critical_count"], delta_color="inverse")
c4.metric("High",             stats["high_count"],     delta_color="inverse")
c5.metric("Top Root Cause",   stats["top_root_cause"][:20] + "...")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Time Series",
    "🔥 Heatmap",
    "🧠 GenAI Explainer",
    "📄 Export"
])

# ────────────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Severity Distribution")
        fig = px.pie(
            anomalies, names="severity",
            color="severity",
            color_discrete_map={
                "Critical": "#E24B4A", "High": "#F97316",
                "Medium":   "#EAB308", "Low":  "#22C55E"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Root Causes")
        rc = anomalies["root_cause"].value_counts().head(10).reset_index()
        rc.columns = ["root_cause", "count"]
        fig = px.bar(rc, x="count", y="root_cause", orientation="h",
                     color="count", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 Critical Incidents")
    critical = df[df["severity"] == "Critical"][
        ["timestamp", "cpu_percent", "memory_percent",
         "latency_ms", "error_rate", "root_cause", "severity"]
    ].head(20)
    st.dataframe(critical, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Server Metrics Over Time — Anomalies Highlighted")

    metric = st.selectbox(
        "Select metric",
        ["cpu_percent", "memory_percent", "latency_ms", "error_rate", "requests_per_sec"]
    )
    n = st.slider("Records to display", 500, len(df), 1000, step=500)
    plot_df = df.tail(n).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp"], y=plot_df[metric],
        name=metric, line=dict(color="#378ADD", width=1.2)
    ))

    # Overlay anomaly points in red
    anom_plot = plot_df[plot_df["ensemble_anomaly"] == 1]
    fig.add_trace(go.Scatter(
        x=anom_plot["timestamp"], y=anom_plot[metric],
        mode="markers", name="Anomaly",
        marker=dict(color="#E24B4A", size=6, symbol="x")
    ))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Correlation Heatmap — Server Metrics")
    corr = df[["cpu_percent", "memory_percent", "latency_ms",
               "error_rate", "requests_per_sec"]].corr()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    st.pyplot(fig)

    st.subheader("Severity Breakdown by Root Cause")
    pivot = anomalies.groupby(["root_cause", "severity"]).size().reset_index(name="count")
    fig2 = px.bar(pivot, x="root_cause", y="count", color="severity",
                  color_discrete_map={"Critical": "#E24B4A", "High": "#F97316",
                                      "Medium": "#EAB308", "Low": "#22C55E"})
    st.plotly_chart(fig2, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🧠 GenAI Root Cause Explainer")
    st.caption("Uses LangChain + Gemini API + FAISS RAG — retrieves similar past incidents and generates plain-English explanation")

    # Pick an anomaly to explain
    st.markdown("**Select an anomaly to explain:**")
    anom_display = anomalies[
        ["timestamp", "cpu_percent", "memory_percent",
         "latency_ms", "error_rate", "severity", "root_cause"]
    ].head(20)
    selected_idx = st.selectbox(
        "Anomaly",
        anom_display.index,
        format_func=lambda i: f"{anom_display.loc[i,'timestamp']} | {anom_display.loc[i,'severity']} | CPU:{anom_display.loc[i,'cpu_percent']:.1f}%"
    )
    selected_row = anomalies.loc[selected_idx].to_dict()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("CPU %",       f"{selected_row['cpu_percent']:.1f}%")
    col_b.metric("Latency ms",  f"{selected_row['latency_ms']:.0f}")
    col_c.metric("Error Rate",  f"{selected_row['error_rate']:.1f}%")

    if st.button("🧠 Explain with GenAI", type="primary"):
        with st.spinner("Retrieving similar incidents from FAISS knowledge base... Generating explanation with Gemini..."):
            try:
                from src.rag.explainer import AnomalyExplainer
                explainer = AnomalyExplainer()
                explanation = explainer.explain(selected_row)
                st.success("GenAI explanation generated!")
                st.markdown(explanation)
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure GEMINI_API_KEY is set in your .env file.\nGet a free key at: https://aistudio.google.com/app/apikey")

    with st.expander("ℹ️ How the GenAI Explainer works"):
        st.markdown("""
        1. **Knowledge base**: 50 past incident reports are embedded using `sentence-transformers/all-MiniLM-L6-v2`
        2. **FAISS retrieval**: The detected anomaly metrics are embedded and the 3 most similar past incidents are retrieved using cosine similarity
        3. **Grounded prompt**: The current anomaly + retrieved incidents are passed as context to Gemini API
        4. **Output**: A specific, grounded root cause explanation — not hallucination
        
        This is called **RAG (Retrieval-Augmented Generation)** — industry standard for enterprise LLM applications.
        """)

# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📄 Export Reports")

    col1, col2 = st.columns(2)

    with col1:
        # CSV download
        csv = anomalies.to_csv(index=False)
        st.download_button(
            "⬇️ Download Anomalies CSV",
            data=csv,
            file_name="anomaly_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # PDF download
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer)
        styles = getSampleStyleSheet()
        content = [
            Paragraph("AI-Powered Production Anomaly Detector — Report", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"Total Records: {stats['total_records']:,}", styles["BodyText"]),
            Paragraph(f"Total Anomalies: {stats['total_anomalies']:,}", styles["BodyText"]),
            Paragraph(f"Anomaly Rate: {stats['anomaly_rate']}%", styles["BodyText"]),
            Paragraph(f"Critical: {stats['critical_count']}", styles["BodyText"]),
            Paragraph(f"High: {stats['high_count']}", styles["BodyText"]),
            Paragraph(f"Top Root Cause: {stats['top_root_cause']}", styles["BodyText"]),
            Spacer(1, 12),
            Paragraph("Built by Irala Pavani | B.Tech Data Science | CGPA 9.4", styles["Italic"]),
        ]
        doc.build(content)
        pdf_buffer.seek(0)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_buffer,
            file_name="anomaly_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
