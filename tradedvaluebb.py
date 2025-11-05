import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Smart Money Phase Detector", layout="wide")

st.title("🎯 Smart Money Accumulation & Distribution Detector")

st.markdown("""
This app identifies **accumulation zones before bullish breakouts** and **distribution zones before bearish breakdowns**:

**Accumulation Signals** (Buy Zone):
- Volume rising while price stable/falling (smart money accumulating)
- Traded value surge at support levels
- RSI bullish divergence (hidden strength)
- ADX compression (coiling spring)
- Bollinger Band squeeze

**Distribution Signals** (Sell Zone):
- Volume rising while price stable/rising (smart money distributing)
- Traded value surge at resistance levels
- RSI bearish divergence (hidden weakness)
- ADX at extremes (trend exhaustion)
- Price at resistance with decreasing momentum
""")

# --- User Input ---
col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Enter Stock Symbol (NSE):", "HDFCBANK").strip().upper()
    lookback_period = st.selectbox("Lookback Period:", ["6mo", "1y", "2y", "3y", "5y"], index=1)
with col2:
    interval = st.selectbox("Data Interval:", ["1d", "1wk"], index=0)
    bb_window = st.slider("Bollinger Band Window:", 10, 40, 20)

# --- Helper Functions ---
def compute_bb(df, column, window, std=2.0):
    ma = df[column].rolling(window).mean()
    s = df[column].rolling(window).std()
    upper = ma + std * s
    lower = ma - std * s
    bb = (df[column] - lower) / (upper - lower) * 100
    bandwidth = ((upper - lower) / ma) * 100
    return bb, bandwidth

def compute_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(window=window).mean()
    roll_down = loss.rolling(window=window).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=df.index, name="RSI")

def compute_adx(df, window=14):
    high, low, close = df["High"], df["Low"], df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return pd.Series(adx, index=df.index, name='ADX')

def detect_accumulation_signals(data):
    signals = pd.DataFrame(index=data.index)
    vol_ma = data['Volume'].rolling(20).mean()
    price_change = data['Close'].pct_change(20)
    signals['volume_accumulation'] = (data['Volume'] > vol_ma * 1.2) & (price_change < 0.05)
    tv_ma = data['Traded_Value'].rolling(20).mean()
    signals['value_surge'] = (data['Traded_Value'] > tv_ma * 1.3) & (data['%B_Close'] < 40)
    price_slope = data['Close'].diff(5)
    rsi_slope = data['RSI'].diff(5)
    signals['rsi_divergence'] = (price_slope < 0) & (rsi_slope > 0) & (data['RSI'] < 45)
    signals['adx_compression'] = (data['ADX'] < 20) & (data['ADX'].diff() < 0)
    signals['bb_squeeze'] = data['BB_Bandwidth'] < data['BB_Bandwidth'].rolling(50).quantile(0.2)
    signals['at_support'] = (data['%B_Close'] > 0) & (data['%B_Close'] < 30)

    acc_score = signals.astype(int).sum(axis=1)
    return acc_score, signals

def detect_distribution_signals(data):
    signals = pd.DataFrame(index=data.index)
    vol_ma = data['Volume'].rolling(20).mean()
    price_change = data['Close'].pct_change(20)
    signals['volume_distribution'] = (data['Volume'] > vol_ma * 1.2) & (price_change > -0.05) & (data['%B_Close'] > 60)
    tv_ma = data['Traded_Value'].rolling(20).mean()
    signals['value_surge_high'] = (data['Traded_Value'] > tv_ma * 1.3) & (data['%B_Close'] > 70)
    price_slope = data['Close'].diff(5)
    rsi_slope = data['RSI'].diff(5)
    signals['rsi_divergence'] = (price_slope > 0) & (rsi_slope < 0) & (data['RSI'] > 60)
    signals['adx_exhaustion'] = (data['ADX'] > 30) & (data['ADX'].diff() < -0.5)
    signals['at_resistance'] = data['%B_Close'] > 70
    signals['momentum_loss'] = (data['%B_Close'] > 65) & (data['RSI'].diff(3) < -2)

    dist_score = signals.astype(int).sum(axis=1)
    return dist_score, signals

# --- Fetch and Analyze ---
if st.button("🔍 Scan for Smart Money Activity", type="primary"):
    ticker = symbol + ".NS"
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            data = yf.download(ticker, period=lookback_period, interval=interval, progress=False)
            if data.empty:
                st.error("No data found. Please check the symbol.")
                st.stop()

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data["Traded_Value"] = data["Close"] * data["Volume"]
            data["RSI"] = compute_rsi(data)
            data["ADX"] = compute_adx(data)
            data["%B_Close"], data["BB_Bandwidth"] = compute_bb(data, "Close", bb_window)
            data.dropna(inplace=True)

            data['Accumulation_Score'], acc_signals = detect_accumulation_signals(data)
            data['Distribution_Score'], dist_signals = detect_distribution_signals(data)
            data['Net_Flow'] = data['Accumulation_Score'] - data['Distribution_Score']

            data['Strong_Accumulation'] = data['Accumulation_Score'] >= 4
            data['Strong_Distribution'] = data['Distribution_Score'] >= 4

            data['Phase'] = 'Neutral'
            data.loc[data['Strong_Accumulation'], 'Phase'] = 'Accumulation'
            data.loc[data['Strong_Distribution'], 'Phase'] = 'Distribution'

            # --- Dashboard Metrics ---
            current_phase = data['Phase'].iloc[-1]
            current_acc = data['Accumulation_Score'].iloc[-1]
            current_dist = data['Distribution_Score'].iloc[-1]
            current_net = data['Net_Flow'].iloc[-1]

            st.subheader("📊 Current Market Status")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Phase", current_phase)
            with col2:
                st.metric("Accumulation", f"{current_acc}/6")
            with col3:
                st.metric("Distribution", f"{current_dist}/6")
            with col4:
                st.metric("Net Smart Money Flow", f"{current_net:+}")

            # --- Charts ---
            st.subheader("📈 Accumulation vs Distribution Scores and Net Effect")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4])

            # Score chart
            fig.add_trace(go.Scatter(x=data.index, y=data['Accumulation_Score'], 
                                     name='Accumulation', line=dict(color='lime', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=-data['Distribution_Score'], 
                                     name='Distribution', line=dict(color='red', width=2)), row=1, col=1)
            fig.add_hline(y=4, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_hline(y=-4, line_dash="dash", line_color="red", row=1, col=1)

            # Net Flow chart
            fig.add_trace(go.Bar(x=data.index, y=data['Net_Flow'], 
                                 name='Net Smart Money Flow', 
                                 marker_color=['green' if v > 0 else 'red' for v in data['Net_Flow']]), row=2, col=1)

            fig.update_layout(
                template="plotly_dark",
                height=600,
                title_text=f"Smart Money Flow Analysis: {symbol}",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Data Table ---
            with st.expander("📋 Latest Data Snapshot"):
                st.dataframe(data.tail(20))

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
