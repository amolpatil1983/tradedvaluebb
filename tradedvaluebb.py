import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Market Phases Analyzer", layout="wide")

st.title("📊 Market Phase Analyzer using %BB, RSI & ADX")

st.markdown("""
This app computes **%B (Percent Bollinger Band)** for:
- **Price**, **Volume**, **Traded Value**, **RSI**, and **ADX**  
It then detects the **current market phase** — Accumulation, Markup, Distribution, or Markdown —  
based on combined signals and visualizes them clearly.
""")

# --- User Input ---
symbol = st.text_input("Enter Stock Symbol (NSE):", "HDFCBANK").strip().upper()
lookback_period = st.selectbox("Lookback Period:", ["6mo", "1y", "2y", "3y", "5y"], index=1)
interval = st.selectbox("Data Interval:", ["1d", "1wk", "1mo"], index=0)
bb_window = st.slider("Bollinger Band Window:", 10, 60, 20)
bb_std = st.slider("BB Standard Deviations:", 1.0, 3.0, 2.0, step=0.1)

# --- Helper Functions ---
def compute_bb(df, column, window, std):
    ma = df[column].rolling(window).mean()
    s = df[column].rolling(window).std()
    upper = ma + std * s
    lower = ma - std * s
    bb = (df[column] - lower) / (upper - lower) * 100
    return bb

def compute_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    roll_up = gain.rolling(window=window).mean()
    roll_down = loss.rolling(window=window).mean()
    
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_adx(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    # Calculate True Range as maximum of the three values
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    
    # Ensure we return a Series, not a DataFrame
    return pd.Series(adx, index=df.index, name='ADX')

def get_zone(row):
    p, v, t, r, a = row["%B_Close"], row["%B_Volume"], row["%B_Traded_Value"], row["%B_RSI"], row["%B_ADX"]
    
    # Check for NaN values
    if pd.isna([p, v, t, r, a]).any():
        return "Neutral"
    
    if p < 30 and v < 30 and t < 30 and r < 40 and a < 50:
        return "Accumulation"
    elif p > 70 and v > 70 and t > 70 and r > 60 and a < 50:
        return "Distribution"
    elif p > 50 and r > 60 and a > 50:
        return "Markup"
    elif p < 50 and r < 40 and a < 40:
        return "Markdown"
    else:
        return "Neutral"

# --- Fetch and Analyze ---
if st.button("Fetch & Analyze"):
    ticker = symbol + ".NS"
    with st.spinner(f"Fetching {lookback_period} data for {ticker}..."):
        try:
            data = yf.download(ticker, period=lookback_period, interval=interval, progress=False)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    if data.empty:
        st.error("No data found. Please check the symbol or period.")
    else:
        try:
            data["Traded_Value"] = data["Close"] * data["Volume"]
            data["RSI"] = compute_rsi(data)
            data["ADX"] = compute_adx(data)

            for col in ["Close", "Volume", "Traded_Value", "RSI", "ADX"]:
                data[f"%B_{col}"] = compute_bb(data, col, bb_window, bb_std)

            data.dropna(inplace=True)
            
            if data.empty:
                st.error("Insufficient data after calculations. Try a longer lookback period.")
                st.stop()
                
            data["Zone"] = data.apply(get_zone, axis=1)

            # --- Visualization ---
            st.subheader(f"{symbol} - Market Phase Overview")

            # Current Phase Display
            current_phase = data["Zone"].iloc[-1]
            phase_emoji = {
                "Accumulation": "🟢",
                "Markup": "🔵",
                "Distribution": "🟠",
                "Markdown": "🔴",
                "Neutral": "⚪"
            }
            st.metric("Current Market Phase", f"{phase_emoji.get(current_phase, '⚪')} {current_phase}")

            # Price Chart with Phase Zones
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Price", line=dict(color="cyan")))

            # --- Color zones vertically (Fixed) ---
            zone_colors = {
                "Accumulation": "rgba(0,255,0,0.1)",
                "Markup": "rgba(0,128,255,0.1)",
                "Distribution": "rgba(255,165,0,0.15)",
                "Markdown": "rgba(255,0,0,0.1)",
                "Neutral": "rgba(255,255,255,0.05)"
            }

            current_zone = data["Zone"].iloc[0]
            start_idx = 0
            
            for i in range(1, len(data)):
                zone = data["Zone"].iloc[i]
                if zone != current_zone:
                    # Draw rectangle for previous zone
                    price_fig.add_vrect(
                        x0=data.index[start_idx],
                        x1=data.index[i-1],
                        fillcolor=zone_colors.get(current_zone, "rgba(255,255,255,0.05)"),
                        opacity=0.3, layer="below", line_width=0
                    )
                    current_zone = zone
                    start_idx = i
            
            # Draw final zone
            price_fig.add_vrect(
                x0=data.index[start_idx],
                x1=data.index[-1],
                fillcolor=zone_colors.get(current_zone, "rgba(255,255,255,0.05)"),
                opacity=0.3, layer="below", line_width=0
            )

            price_fig.update_layout(
                title=f"{symbol} Price with Phase Zones",
                yaxis_title="Price (INR)",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(price_fig, use_container_width=True)

            # --- %BB Overlay Chart ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["%B_Close"], name="%B Price", line=dict(color="yellow")))
            fig.add_trace(go.Scatter(x=data.index, y=data["%B_Volume"], name="%B Volume", line=dict(color="lightgreen")))
            fig.add_trace(go.Scatter(x=data.index, y=data["%B_Traded_Value"], name="%B Value", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=data.index, y=data["%B_RSI"], name="%B RSI", line=dict(color="magenta")))
            fig.add_trace(go.Scatter(x=data.index, y=data["%B_ADX"], name="%B ADX", line=dict(color="cyan")))

            fig.add_hline(y=100, line_dash="dot", annotation_text="Upper Band")
            fig.add_hline(y=0, line_dash="dot", annotation_text="Lower Band")
            fig.add_hline(y=50, line_dash="dot", annotation_text="Midpoint")

            fig.update_layout(
                title="%BB Indicators Comparison",
                xaxis_title="Date",
                yaxis_title="%B Value",
                template="plotly_dark",
                height=700,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show Latest Data"):
                st.dataframe(data[["Close", "RSI", "ADX", "%B_Close", "%B_RSI", "%B_ADX", "Zone"]].tail(20))
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.exception(e)
