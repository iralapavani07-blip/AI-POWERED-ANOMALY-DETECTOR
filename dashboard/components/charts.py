import streamlit as st
import plotly.express as px

def severity_pie(df):

    fig = px.pie(
        df,
        names="severity",
        title="Severity Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def rootcause_pie(df):

    fig = px.pie(
        df,
        names="root_cause",
        title="Root Cause Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def cpu_trend(df):

    fig = px.line(
        df,
        y="cpu_percent",
        title="CPU Usage Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def memory_trend(df):

    fig = px.line(
        df,
        y="memory_percent",
        title="Memory Usage Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def latency_trend(df):

    fig = px.line(
        df,
        y="latency_ms",
        title="Latency Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

def severity_bar(df):

    fig = px.bar(
        df["severity"].value_counts(),
        title="Severity Breakdown"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )