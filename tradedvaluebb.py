import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stock %BB Comparison", layout="wide")

st.title("📊 %BB Comparison — Price / Volume / Traded Value")
st.markdown("""
This app compares **%B (Percent Bollinger Band)** computed separately for:
- **Close Price**
- **Volume**
- **Traded Value = Close × Volume**

Data is fetched from Yahoo Finance (NSE India).
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
        # --- Compute derived series ---
        data["Traded_Value"] = data["Close"] * data["Volume"]

        def compute_bb(df, column, window, std):
            ma = df[column].rolling(window).mean()
            s = df[column].rolling(window).std()
            lower = ma - std * s
            upper = ma + std * s
            bb = (df[column] - lower) / (upper - lower) * 100
            return bb

        # Compute %B for all three metrics
        data["%B_Price"] = compute_bb(data, "Close", bb_window, bb_std)
        data["%B_Volume"] = compute_bb(data, "Volume", bb_window, bb_std)
        data["%B_TradedVal"] = compute_bb(data, "Traded_Value", bb_window, bb_std)

        data = data.dropna()

        # --- Plot 1: Price Chart ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close", line=dict(color="cyan")))
        fig1.update_layout(
            title=f"{symbol} - Price Chart ({lookback_period})",
            xaxis_title="Date", yaxis_title="Price (INR)", template="plotly_dark", height=400
        )

        # --- Plot 2: %BB Combined Chart ---
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data["%B_Price"], name="%B (Price)", line=dict(color="yellow")))
        fig2.add_trace(go.Scatter(x=data.index, y=data["%B_Volume"], name="%B (Volume)", line=dict(color="lightgreen")))
        fig2.add_trace(go.Scatter(x=data.index, y=data["%B_TradedVal"], name="%B (Traded Value)", line=dict(color="orange")))
        fig2.add_hline(y=100, line_dash="dot", annotation_text="Upper Band")
        fig2.add_hline(y=0, line_dash="dot", annotation_text="Lower Band")
        fig2.add_hline(y=50, line_dash="dot", annotation_text="Midpoint (MA)")
        fig2.update_layout(
            title=f"{symbol} - %B Comparison (Price / Volume / Traded Value)",
            xaxis_title="Date", yaxis_title="%B", template="plotly_dark", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )

        # --- Display Charts ---
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        # --- Optional Data Table ---
        with st.expander("Show Raw Data"):
            st.dataframe(data[["Close", "Volume", "Traded_Value", "%B_Price", "%B_Volume", "%B_TradedVal"]].tail(20))
