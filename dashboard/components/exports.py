import streamlit as st

def csv_download(df):

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        "Download CSV",
        csv,
        "anomaly_results.csv",
        "text/csv"
    )