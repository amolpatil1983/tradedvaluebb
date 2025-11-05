import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Stock %BB + RSI + ADX Zones", layout="wide")

st.title("📊 %BB Comparison with RSI & ADX — Multi-Signal Market Phases")

st.markdown("""
This app computes **%B (Percent Bollinger Band)** for:
- **Price**, **Volume**, **Traded Value**, **RSI**, and **ADX**  
and visually classifies **market phases** (Accumulation, Markup, Distribution, Markdown)
based on a combined %BB logic.
""")

# --- Inputs ---
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
        # --- Derived Columns ---
        data["Traded_Value"] = data["Close"] * data["Volume"]

        # --- %B Computation ---
        def compute_bb(df, column, window, std):
            ma = df[column].rolling(window).mean()
            s = df[column].rolling(window).std()
            lower = ma - std * s
            upper = ma + std * s
            return (df[column] - lower) / (upper - lower) * 100

        # --- RSI ---
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        roll_up = gain.ewm(span=14, adjust=False).mean()
        roll_down = loss.ewm(span=14, adjust=False).mean()
        rs = roll_up / roll_down
        data["RSI"] = 100 - (100 / (1 + rs))

        # --- ADX ---
        high, low, close = data["High"], data["Low"], data["Close"]

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (low.diff() < 0), minus_dm, 0)

        tr = np.maximum.reduce([
            (high - low).abs(),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ])
        atr = pd.Series(tr, index=data.index).rolling(14).mean()

        plus_di = 100 * pd.Series(plus_dm, index=data.index).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=data.index).rolling(14).mean() / atr

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        data["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()

        # --- %B for all ---
        for col in ["Close", "Volume", "Traded_Value", "RSI", "ADX"]:
            data[f"%B_{col}"] = compute_bb(data, col, bb_window, bb_std)

        data.dropna(inplace=True)

        # --- Zone Classification ---
        def get_zone(row):
            p, v, t, r, a = row["%B_Close"], row["%B_Volume"], row["%B_Traded_Value"], row["%B_RSI"], row["%B_ADX"]
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

        data["Zone"] = data.apply(get_zone, axis=1)

        # --- Price Chart ---
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=data.index, y=data["Close"],
                                       name="Close Price", line=dict(color="cyan")))
        price_fig.update_layout(title=f"{symbol} - Price Chart",
                                yaxis_title="Price (INR)",
                                template="plotly_dark", height=400)
        st.plotly_chart(price_fig, use_container_width=True)

        # --- %BB Chart with Zones ---
        fig = go.Figure()
        zone_colors = {
            "Accumulation": "rgba(0,255,0,0.1)",
            "Markup": "rgba(0,128,255,0.1)",
            "Distribution": "rgba(255,165,0,0.15)",
            "Markdown": "rgba(255,0,0,0.1)",
            "Neutral": "rgba(255,255,255,0.05)"
        }

        current_zone = None
        start_idx = None
        for i in range(len(data)):
            zone = data["Zone"].iloc[i]
            if current_zone is None:
                current_zone, start_idx = zone, i
            elif zone != current_zone or i == len(data) - 1:
                end_idx = i if i < len(data) - 1 else i
                fig.add_vrect(
                    x0=data.index[start_idx],
                    x1=data.index[end_idx],
                    fillcolor=zone_colors.get(current_zone, "rgba(255,255,255,0.05)"),
                    opacity=0.3, layer="below", line_width=0)
                current_zone, start_idx = zone, i

        fig.add_trace(go.Scatter(x=data.index, y=data["%B_Close"], name="%B (Price)", line=dict(color="yellow")))
        fig.add_trace(go.Scatter(x=data.index, y=data["%B_Volume"], name="%B (Volume)", line=dict(color="lightgreen")))
        fig.add_trace(go.Scatter(x=data.index, y=data["%B_Traded_Value"], name="%B (Traded Value)", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=data.index, y=data["%B_RSI"], name="%B (RSI)", line=dict(color="magenta")))
        fig.add_trace(go.Scatter(x=data.index, y=data["%B_ADX"], name="%B (ADX)", line=dict(color="cyan")))

        fig.add_hline(y=100, line_dash="dot", annotation_text="Upper Band", annotation_position="top left")
        fig.add_hline(y=0, line_dash="dot", annotation_text="Lower Band", annotation_position="bottom left")
        fig.add_hline(y=50, line_dash="dot", annotation_text="Midpoint", annotation_position="top left")

        fig.update_layout(
            title=f"{symbol} — %B Comparison (Price, Volume, Traded Value, RSI, ADX)",
            xaxis_title="Date",
            yaxis_title="%B",
            template="plotly_dark",
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show Data with Zones"):
            st.dataframe(data[["Close", "%B_Close", "%B_RSI", "%B_ADX", "Zone"]].tail(25))
