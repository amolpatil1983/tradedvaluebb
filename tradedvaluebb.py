import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stock %BB Visualizer", layout="wide")

st.title("📈 Price and %BB Visualizer (Based on Daily Traded Value)")
st.markdown("""
This app pulls historical stock data from Yahoo Finance (NSE India) and computes:
- **Traded Value = Close × Volume**
- **%B (Percent Bollinger Band)** from Traded Value  
""")

# --- User Inputs ---
symbol = st.text_input("Enter Stock Symbol (NSE):", "HDFCBANK").strip().upper()
lookback_period = st.selectbox("Lookback Period:", ["6mo", "1y", "2y", "3y", "5y"], index=1)
interval = st.selectbox("Data Interval:", ["1d", "1wk", "1mo"], index=0)

bb_window = st.slider("Bollinger Band Window:", 10, 60, 20)
bb_std = st.slider("BB Standard Deviations:", 1.0, 3.0, 2.0, step=0.1)

if st.button("Fetch & Analyze"):
    ticker = symbol + ".NS"
    with st.spinner(f"Fetching {lookback_period} data for {ticker}..."):
        data = yf.download(ticker, period=lookback_period, interval=interval, progress=False)
    
    if data.empty:
        st.error("No data found. Please check the symbol or try another period.")
    else:
        # --- Compute Traded Value and %BB ---
        data["Traded_Value"] = data["Close"] * data["Volume"]

        # Bollinger Bands on Traded Value
        data["MA"] = data["Traded_Value"].rolling(bb_window).mean()
        data["STD"] = data["Traded_Value"].rolling(bb_window).std()
        data["Upper"] = data["MA"] + bb_std * data["STD"]
        data["Lower"] = data["MA"] - bb_std * data["STD"]
        data["%B"] = (data["Traded_Value"] - data["Lower"]) / (data["Upper"] - data["Lower"]) * 100

        # Drop NA for smooth chart
        data = data.dropna()

        # --- Plot 1: Price Chart ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="blue")))
        fig1.update_layout(
            title=f"{symbol} - Price Chart ({lookback_period})",
            xaxis_title="Date", yaxis_title="Price (INR)", template="plotly_dark", height=400
        )

        # --- Plot 2: %BB Chart ---
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data["%B"], name="%B", line=dict(color="orange")))
        fig2.add_hline(y=100, line_dash="dot", annotation_text="Upper Band")
        fig2.add_hline(y=0, line_dash="dot", annotation_text="Lower Band")
        fig2.add_hline(y=50, line_dash="dot", annotation_text="Midpoint (MA)")
        fig2.update_layout(
            title=f"{symbol} - %B (Bollinger from Traded Value)",
            xaxis_title="Date", yaxis_title="%B", template="plotly_dark", height=400
        )

        # --- Display Charts ---
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        # --- Optional Data Table ---
        with st.expander("Show Raw Data"):
            st.dataframe(data.tail(20))
