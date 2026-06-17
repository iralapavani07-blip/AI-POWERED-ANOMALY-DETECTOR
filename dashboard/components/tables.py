import streamlit as st

def critical_incidents(df):

    st.subheader(
        "Top Critical Incidents"
    )

    critical = df[
        df["severity"] == "Critical"
    ]

    st.dataframe(
        critical.head(20)
    )

def anomaly_table(df):

    st.subheader(
        "Anomaly Explorer"
    )

    anomalies = df[
        df["ensemble_anomaly"] == 1
    ]

    st.dataframe(
        anomalies
    )