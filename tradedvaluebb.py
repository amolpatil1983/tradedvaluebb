import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# --- Streamlit Setup ---
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
        # --- Compute Derived Columns ---
        data["Traded_Value"] = data["Close"] * data["Volume"]

        def compute_bb(df, column, window, std):
            ma = df[column].rolling(window).mean()
            s = df[column].rolling(window).std()
            lower = ma - std * s
            upper = ma + std * s
            bb = (df[column] - lower) / (upper - lower) * 100
            return bb

        # Compute %B for each metric
        data["%B_Price"] = compute_bb(data, "Close", bb_window, bb_std)
        data["%B_Volume"] = compute_bb(data, "Volume", bb_window, bb_std)
        data["%B_TradedVal"] = compute_bb(data, "Traded_Value", bb_window, bb_std)
        data = data.dropna()

        # --- Plot 1: Price Chart ---
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"], name="Close Price", line=dict(color="deepskyblue", width=2)
        ))
        price_fig.update_layout(
            title=f"{symbol} — Daily Close Price ({lookback_period})",
            xaxis_title="Date", yaxis_title="Price (INR)",
            template="plotly_dark", height=400
        )
        st.plotly_chart(price_fig, use_container_width=True)

        # --- Plot 2: %B Chart with Accumulation & Distribution Zones ---
        bb_fig = go.Figure()

        # %B lines
        bb_fig.add_trace(go.Scatter(
            x=data.index, y=data["%B_Price"], name="%B (Price)", line=dict(color="yellow", width=1.5)))
        bb_fig.add_trace(go.Scatter(
            x=data.index, y=data["%B_Volume"], name="%B (Volume)", line=dict(color="lightgreen", width=1.2)))
        bb_fig.add_trace(go.Scatter(
            x=data.index, y=data["%B_TradedVal"], name="%B (Traded Value)", line=dict(color="orange", width=1.5)))

        # Shaded Accumulation/Distribution zones
        bb_fig.add_hrect(
            y0=0, y1=20, fillcolor="green", opacity=0.15, line_width=0,
            annotation_text="Accumulation Zone", annotation_position="top left"
        )
        bb_fig.add_hrect(
            y0=80, y1=100, fillcolor="red", opacity=0.15, line_width=0,
            annotation_text="Distribution Zone", annotation_position="bottom left"
        )

        # Reference lines
        bb_fig.add_hline(y=100, line_dash="dot", annotation_text="Upper Band", annotation_position="top left")
        bb_fig.add_hline(y=0, line_dash="dot", annotation_text="Lower Band", annotation_position="bottom left")
        bb_fig.add_hline(y=50, line_dash="dot", annotation_text="Midpoint", annotation_position="top left")

        # Layout
        bb_fig.update_layout(
            title=f"{symbol} — %B Comparison (Price / Volume / Traded Value)",
            xaxis_title="Date", yaxis_title="%B",
            template="plotly_dark", height=600,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(bb_fig, use_container_width=True)

        # --- Optional Data Table ---
        with st.expander("Show Raw Data"):
            st.dataframe(data[["Close", "Volume", "Traded_Value", "%B_Price", "%B_Volume", "%B_TradedVal"]].tail(20))
