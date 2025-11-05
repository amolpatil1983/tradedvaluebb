import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Phase Analyzer", layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------
def compute_bb(series, window=20, num_std=2):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bb_percent = 100 * (series - lower) / (upper - lower)
    return bb_percent.clip(0, 100)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(data, period=14):
    high, low, close = data["High"], data["Low"], data["Close"]
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = (-minus_dm).where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period).mean()
    return adx

def classify_phase(bb_price, bb_value, bb_vol, bb_rsi, bb_adx):
    avg_bb = (bb_price + bb_value + bb_vol + bb_rsi + bb_adx) / 5
    if bb_adx < 25 and avg_bb < 40:
        return "Accumulation"
    elif bb_adx > 25 and bb_rsi > 55 and bb_price > 60:
        return "Trending Up"
    elif bb_adx > 25 and bb_rsi < 45 and bb_price < 40:
        return "Trending Down"
    elif bb_adx < 25 and avg_bb > 60:
        return "Distribution"
    else:
        return "Transition"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Intelligent Stock Phase Analyzer")
symbol = st.text_input("Enter Stock Symbol (e.g. HDFCBANK):", "HDFCBANK")
period = st.selectbox("Select Lookback Period:", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.selectbox("Select Interval:", ["1d", "1wk", "1mo"], index=0)

if st.button("Analyze"):
    symbol_full = symbol.strip().upper() + ".NS"
    data = yf.download(symbol_full, period=period, interval=interval, auto_adjust=True)

    if data.empty:
        st.error("No data found. Check the symbol or try another range.")
    else:
        data["Value_Traded"] = data["Close"] * data["Volume"]

        # Compute technical indicators
        data["RSI"] = compute_rsi(data["Close"])
        data["ADX"] = compute_adx(data)

        # Compute %BB for each
        data["%BB_Price"] = compute_bb(data["Close"])
        data["%BB_Volume"] = compute_bb(data["Volume"])
        data["%BB_Value"] = compute_bb(data["Value_Traded"])
        data["%BB_RSI"] = compute_bb(data["RSI"].dropna())
        data["%BB_ADX"] = compute_bb(data["ADX"].dropna())

        # Forward-fill to align shapes
        data = data.fillna(method="ffill")

        # Classify phase row-wise
        data["Phase"] = data.apply(lambda row: classify_phase(
            row["%BB_Price"], row["%BB_Value"], row["%BB_Volume"], row["%BB_RSI"], row["%BB_ADX"]), axis=1)

        # -----------------------------
        # Visualization
        # -----------------------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index, y=data["%BB_Price"], mode="lines", name="%BB_Price", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(
            x=data.index, y=data["%BB_Value"], mode="lines", name="%BB_Value", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(
            x=data.index, y=data["%BB_Volume"], mode="lines", name="%BB_Volume", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(
            x=data.index, y=data["%BB_RSI"], mode="lines", name="%BB_RSI", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(
            x=data.index, y=data["%BB_ADX"], mode="lines", name="%BB_ADX", line=dict(width=1.5)))

        # Add vertical color-coded regions
        phase_colors = {
            "Accumulation": "rgba(0, 200, 100, 0.1)",
            "Trending Up": "rgba(0, 100, 250, 0.1)",
            "Distribution": "rgba(250, 150, 0, 0.1)",
            "Trending Down": "rgba(250, 0, 0, 0.1)",
            "Transition": "rgba(200, 200, 200, 0.1)"
        }

        prev_phase = None
        start_idx = None
        for i in range(len(data)):
            phase = data["Phase"].iloc[i]
            if phase != prev_phase:
                if prev_phase is not None and start_idx is not None:
                    fig.add_vrect(
                        x0=data.index[start_idx], x1=data.index[i],
                        fillcolor=phase_colors.get(prev_phase, "rgba(0,0,0,0)"),
                        opacity=0.3, layer="below", line_width=0)
                start_idx = i
                prev_phase = phase

        fig.update_layout(
            title=f"Stock Phase Analysis for {symbol_full}",
            xaxis_title="Date",
            yaxis_title="% Bollinger Band (0–100)",
            legend_title="Indicators",
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📌 Current Stock Phase:")
        st.success(f"{symbol_full} is currently in **{data['Phase'].iloc[-1]}** phase.")

