import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Streamlit Config ---
st.set_page_config(page_title="Stock %BB Analyzer", layout="wide")

st.title("📊 %B (Percent Bollinger Band) Analyzer — Price / Volume / Traded Value")
st.markdown("""
Analyze how **Price**, **Volume**, and **Traded Value (Price × Volume)** behave 
relative to their respective Bollinger Bands over different timeframes.
""")

# --- User Inputs ---
symbol = st.text_input("Enter Stock Symbol (NSE):", "HDFCBANK").strip().upper()
lookback_period = st.selectbox("Lookback Period:", ["6mo", "1y", "2y", "3y", "5y"], index=1)
interval = st.selectbox("Data Interval:", ["1d", "1wk", "1mo"], index=0)

bb_window = st.slider("Bollinger Band Window:", 10, 60, 20)
bb_std = st.slider("BB Standard Deviations:", 1.0, 3.0, 2.0, step=0.1)

# --- Convert lookback period to explicit start date (ensures full data fetch) ---
period_map = {"6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825}
start_date = datetime.now() - timedelta(days=period_map[lookback_period])
end_date = datetime.now()

# --- Fetch Data ---
if st.button("Fetch & Analyze"):
    ticker = symbol + ".NS"
    with st.spinner(f"Fetching {lookback_period} data for {ticker}..."):
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False
        )

    if data.empty:
        st.error("⚠️ No data found. Please check the symbol or try another period.")
    else:
        st.success(f"✅ Fetched data from {data.index.min().date()} to {data.index.max().date()} "
                   f"({len(data)} points)")

        # --- Compute Derived Data ---
        data["Traded_Value"] = data["Close"] * data["Volume"]

        def compute_bb(df, column, window, std):
            ma = df[column].rolling(window).mean()
            s = df[column].rolling(window).std()
            lower = ma - std * s
            upper = ma + std * s
            bb = (df[column] - lower) / (upper - lower) * 100
            return bb

        # Compute %B for all metrics
        data["%B_Price"] = compute_bb(data, "Close", bb_window, bb_std)
        data["%B_Volume"] = compute_bb(data, "Volume", bb_window, bb_std)
        data["%B_TradedVal"] = compute_bb(data, "Traded_Value", bb_window, bb_std)
        data = data.dropna()

        # --- Plot 1: Price Chart ---
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=data.index, y=data["Close"], mode="lines",
            name="Close Price", line=dict(color="deepskyblue", width=2)
        ))
        price_fig.update_layout(
            title=f"{symbol} — Daily Close Price",
            xaxis_title="Date", yaxis_title="Price (INR)",
            template="plotly_dark", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(price_fig, use_container_width=True)

        # --- Plot 2: %B Chart ---
        bb_fig = go.Figure()
        bb_fig.add_trace(go.Scatter(x=data.index, y=data["%B_Price"],
                                    name="%B (Price)", line=dict(color="yellow")))
        bb_fig.add_trace(go.Scatter(x=data.index, y=data["%B_Volume"],
                                    name="%B (Volume)", line=dict(color="lightgreen")))
        bb_fig.add_trace(go.Scatter(x=data.index, y=data["%B_TradedVal"],
                                    name="%B (Traded Value)", line=dict(color="orange")))

        bb_fig.add_hline(y=100, line_dash="dot", annotation_text="Upper Band", annotation_position="top left")
        bb_fig.add_hline(y=0, line_dash="dot", annotation_text="Lower Band", annotation_position="bottom left")
        bb_fig.add_hline(y=50, line_dash="dot", annotation_text="Midpoint", annotation_position="top left")

        bb_fig.update_layout(
            title=f"{symbol} — %B Comparison (Price / Volume / Traded Value)",
            xaxis_title="Date", yaxis_title="%B",
            template="plotly_dark", height=600,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )

        st.plotly_chart(bb_fig, use_container_width=True)

        # --- Optional Data Table ---
        with st.expander("Show Raw Data"):
            st.dataframe(
                data[["Close", "Volume", "Traded_Value", "%B_Price", "%B_Volume", "%B_TradedVal"]].tail(20)
            )
