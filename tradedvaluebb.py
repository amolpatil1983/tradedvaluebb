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
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(window=window).mean()
    roll_down = loss.rolling(window=window).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_adx(df, window=14):
    high = df["High"].copy()
    low = df["Low"].copy()
    close = df["Close"].copy()

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr_array = np.maximum(np.maximum(tr1.values, tr2.values), tr3.values)
    tr = pd.Series(tr_array, index=df.index)

    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return pd.Series(adx.values, index=df.index, name='ADX')

def detect_accumulation_signals(data):
    """Detect pre-breakout accumulation patterns"""
    signals = pd.DataFrame(index=data.index)
    
    # 1. Volume Accumulation: Volume rising while price stable/falling
    vol_ma = data['Volume'].rolling(20).mean()
    price_change = data['Close'].pct_change(20)
    signals['volume_accumulation'] = (data['Volume'] > vol_ma * 1.2) & (price_change < 0.05)
    
    # 2. Traded Value Surge at Low Prices
    tv_ma = data['Traded_Value'].rolling(20).mean()
    signals['value_surge'] = (data['Traded_Value'] > tv_ma * 1.3) & (data['%B_Close'] < 40)
    
    # 3. RSI Bullish Divergence (price down, RSI up)
    price_slope = data['Close'].diff(5)
    rsi_slope = data['RSI'].diff(5)
    signals['rsi_divergence'] = (price_slope < 0) & (rsi_slope > 0) & (data['RSI'] < 45)
    
    # 4. ADX Compression (low volatility, coiling)
    signals['adx_compression'] = (data['ADX'] < 20) & (data['ADX'].diff() < 0)
    
    # 5. Bollinger Band Squeeze (bandwidth contraction)
    signals['bb_squeeze'] = data['BB_Bandwidth'] < data['BB_Bandwidth'].rolling(50).quantile(0.2)
    
    # 6. Price at Support (%B between 0-30)
    signals['at_support'] = (data['%B_Close'] > 0) & (data['%B_Close'] < 30)
    
    # Accumulation Score (0-6)
    acc_score = (
        signals['volume_accumulation'].astype(int) +
        signals['value_surge'].astype(int) +
        signals['rsi_divergence'].astype(int) +
        signals['adx_compression'].astype(int) +
        signals['bb_squeeze'].astype(int) +
        signals['at_support'].astype(int)
    )
    
    return acc_score, signals

def detect_distribution_signals(data):
    """Detect pre-breakdown distribution patterns"""
    signals = pd.DataFrame(index=data.index)
    
    # 1. Volume Distribution: Volume rising while price stalling at highs
    vol_ma = data['Volume'].rolling(20).mean()
    price_change = data['Close'].pct_change(20)
    signals['volume_distribution'] = (data['Volume'] > vol_ma * 1.2) & (price_change > -0.05) & (data['%B_Close'] > 60)
    
    # 2. Traded Value Surge at High Prices (selling into strength)
    tv_ma = data['Traded_Value'].rolling(20).mean()
    signals['value_surge_high'] = (data['Traded_Value'] > tv_ma * 1.3) & (data['%B_Close'] > 70)
    
    # 3. RSI Bearish Divergence (price up, RSI down)
    price_slope = data['Close'].diff(5)
    rsi_slope = data['RSI'].diff(5)
    signals['rsi_divergence'] = (price_slope > 0) & (rsi_slope < 0) & (data['RSI'] > 60)
    
    # 4. ADX Exhaustion (high ADX declining = trend weakening)
    signals['adx_exhaustion'] = (data['ADX'] > 30) & (data['ADX'].diff() < -0.5)
    
    # 5. Price at Resistance (%B > 70)
    signals['at_resistance'] = data['%B_Close'] > 70
    
    # 6. Momentum Loss (price high but RSI falling)
    signals['momentum_loss'] = (data['%B_Close'] > 65) & (data['RSI'].diff(3) < -2)
    
    # Distribution Score (0-6)
    dist_score = (
        signals['volume_distribution'].astype(int) +
        signals['value_surge_high'].astype(int) +
        signals['rsi_divergence'].astype(int) +
        signals['adx_exhaustion'].astype(int) +
        signals['at_resistance'].astype(int) +
        signals['momentum_loss'].astype(int)
    )
    
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
            
            # Flatten multi-level columns if they exist
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Calculate indicators
            data["Traded_Value"] = data["Close"] * data["Volume"]
            data["RSI"] = compute_rsi(data)
            data["ADX"] = compute_adx(data)
            
            data["%B_Close"], data["BB_Bandwidth"] = compute_bb(data, "Close", bb_window)
            data["%B_Volume"], _ = compute_bb(data, "Volume", bb_window)
            data["%B_Traded_Value"], _ = compute_bb(data, "Traded_Value", bb_window)
            data["%B_RSI"], _ = compute_bb(data, "RSI", bb_window)
            data["%B_ADX"], _ = compute_bb(data, "ADX", bb_window)
            
            data.dropna(inplace=True)
            
            if data.empty:
                st.error("Insufficient data. Try a longer lookback period.")
                st.stop()
            
            # Detect both accumulation and distribution
            data['Accumulation_Score'], acc_signals = detect_accumulation_signals(data)
            data['Distribution_Score'], dist_signals = detect_distribution_signals(data)
            
            data['Strong_Accumulation'] = data['Accumulation_Score'] >= 4
            data['Strong_Distribution'] = data['Distribution_Score'] >= 4
            
            # Determine phase
            data['Phase'] = 'Neutral'
            data.loc[data['Strong_Accumulation'], 'Phase'] = 'Accumulation'
            data.loc[data['Strong_Distribution'], 'Phase'] = 'Distribution'
            
            # --- Current Status ---
            current_acc = data['Accumulation_Score'].iloc[-1]
            current_dist = data['Distribution_Score'].iloc[-1]
            current_phase = data['Phase'].iloc[-1]
            
            st.subheader("📊 Current Market Status")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if current_phase == 'Accumulation':
                    st.metric("Phase", "🟢 ACCUMULATION", delta="Bullish Setup")
                elif current_phase == 'Distribution':
                    st.metric("Phase", "🔴 DISTRIBUTION", delta="Bearish Setup")
                else:
                    st.metric("Phase", "⚪ Neutral", delta="Wait")
            with col2:
                color = "🟢" if current_acc >= 4 else "🟡" if current_acc >= 2 else "⚪"
                st.metric("Accumulation", f"{color} {current_acc}/6")
            with col3:
                color = "🔴" if current_dist >= 4 else "🟡" if current_dist >= 2 else "⚪"
                st.metric("Distribution", f"{color} {current_dist}/6")
            with col4:
                st.metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
            with col5:
                st.metric("ADX", f"{data['ADX'].iloc[-1]:.1f}")
            
            # Alert messages
            if current_phase == 'Accumulation':
                st.success("🎯 **STRONG ACCUMULATION DETECTED** - Potential pre-breakout zone! Smart money accumulating.")
                st.info("✅ Strategy: Watch for breakout above resistance with volume expansion. Consider entry on confirmation.")
            elif current_phase == 'Distribution':
                st.error("⚠️ **STRONG DISTRIBUTION DETECTED** - Potential pre-breakdown zone! Smart money distributing.")
                st.warning("❌ Strategy: Avoid buying. Consider exit if holding. Watch for breakdown below support.")
            elif current_acc >= 2 or current_dist >= 2:
                st.warning("⚠️ Mixed signals detected. Monitor closely for confirmation.")
            else:
                st.info("No significant accumulation or distribution currently.")
            
            # --- Main Chart ---
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} Price with Smart Money Zones', 'Volume & Traded Value', 'RSI', 'ADX'),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            
            # Price with accumulation/distribution zones
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', 
                                    line=dict(color='white', width=2)), row=1, col=1)
            
            # Highlight zones
            for i in range(len(data)):
                if data['Strong_Accumulation'].iloc[i]:
                    fig.add_vrect(
                        x0=data.index[i], x1=data.index[min(i+1, len(data)-1)],
                        fillcolor="rgba(0,255,0,0.3)",
                        layer="below", line_width=0,
                        row=1, col=1
                    )
                elif data['Strong_Distribution'].iloc[i]:
                    fig.add_vrect(
                        x0=data.index[i], x1=data.index[min(i+1, len(data)-1)],
                        fillcolor="rgba(255,0,0,0.3)",
                        layer="below", line_width=0,
                        row=1, col=1
                    )
            
            # Bollinger Bands
            ma = data['Close'].rolling(bb_window).mean()
            std = data['Close'].rolling(bb_window).std()
            fig.add_trace(go.Scatter(x=data.index, y=ma + 2*std, name='Upper BB',
                                    line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=ma - 2*std, name='Lower BB',
                                    line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
            
            # Volume & Traded Value
            colors = ['green' if data['Strong_Accumulation'].iloc[i] else 'red' if data['Strong_Distribution'].iloc[i] else 'lightblue' 
                      for i in range(len(data))]
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume',
                                marker_color=colors, opacity=0.6), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=data.index, y=data['Traded_Value']/1e6, name='Traded Value (M)',
                                    line=dict(color='orange', width=1), yaxis='y2'), row=2, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                                    line=dict(color='magenta', width=2)), row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            
            # ADX
            fig.add_trace(go.Scatter(x=data.index, y=data['ADX'], name='ADX',
                                    line=dict(color='cyan', width=2)), row=4, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="yellow", row=4, col=1)
            
            fig.update_layout(
                height=1000,
                template="plotly_dark",
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="ADX", row=4, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Score Comparison Chart ---
            st.subheader("📈 Accumulation vs Distribution Scores")
            score_fig = go.Figure()
            score_fig.add_trace(go.Scatter(x=data.index, y=data['Accumulation_Score'],
                                          name='Accumulation', fill='tozeroy',
                                          line=dict(color='lime', width=2)))
            score_fig.add_trace(go.Scatter(x=data.index, y=-data['Distribution_Score'],
                                          name='Distribution', fill='tozeroy',
                                          line=dict(color='red', width=2)))
            score_fig.add_hline(y=4, line_dash="dash", line_color="green", 
                               annotation_text="Strong Accumulation")
            score_fig.add_hline(y=-4, line_dash="dash", line_color="red", 
                               annotation_text="Strong Distribution")
            score_fig.update_layout(
                yaxis_title="Score (Acc +, Dist -)",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(score_fig, use_container_width=True)
            
            # --- Signal Breakdown ---
            with st.expander("🔍 View Signal Breakdown"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🟢 Accumulation Signals:")
                    st.write(f"✅ Volume Accumulation: {acc_signals['volume_accumulation'].iloc[-1]}")
                    st.write(f"✅ Value Surge at Support: {acc_signals['value_surge'].iloc[-1]}")
                    st.write(f"✅ RSI Bullish Divergence: {acc_signals['rsi_divergence'].iloc[-1]}")
                    st.write(f"✅ ADX Compression: {acc_signals['adx_compression'].iloc[-1]}")
                    st.write(f"✅ Bollinger Band Squeeze: {acc_signals['bb_squeeze'].iloc[-1]}")
                    st.write(f"✅ Price at Support: {acc_signals['at_support'].iloc[-1]}")
                
                with col2:
                    st.markdown("### 🔴 Distribution Signals:")
                    st.write(f"⚠️ Volume Distribution: {dist_signals['volume_distribution'].iloc[-1]}")
                    st.write(f"⚠️ Value Surge at Resistance: {dist_signals['value_surge_high'].iloc[-1]}")
                    st.write(f"⚠️ RSI Bearish Divergence: {dist_signals['rsi_divergence'].iloc[-1]}")
                    st.write(f"⚠️ ADX Exhaustion: {dist_signals['adx_exhaustion'].iloc[-1]}")
                    st.write(f"⚠️ Price at Resistance: {dist_signals['at_resistance'].iloc[-1]}")
                    st.write(f"⚠️ Momentum Loss: {dist_signals['momentum_loss'].iloc[-1]}")
            
            # --- Historical Performance ---
            with st.expander("📊 Historical Phase Statistics"):
                acc_count = (data['Strong_Accumulation']).sum()
                dist_count = (data['Strong_Distribution']).sum()
                
                st.write(f"Total Accumulation Zones: **{acc_count}**")
                st.write(f"Total Distribution Zones: **{dist_count}**")
                st.write(f"Current Lookback: **{lookback_period}**")
            
            # --- Data Table ---
            with st.expander("📋 Recent Data"):
                display_cols = ['Close', 'Volume', 'RSI', 'ADX', '%B_Close', 
                               'Accumulation_Score', 'Distribution_Score', 'Phase']
                st.dataframe(data[display_cols].tail(30))
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

st.markdown("---")
st.markdown("""
### 🎓 How to Use This Tool:
**🟢 Accumulation (Buy Setup)**:
- Score 4-6: Strong pre-breakout signal
- Green zones = Smart money accumulating
- Strategy: Wait for breakout confirmation, enter on volume spike

**🔴 Distribution (Sell Setup)**:
- Score 4-6: Strong pre-breakdown signal  
- Red zones = Smart money distributing
- Strategy: Avoid buying, exit positions, watch for breakdown

**📊 Volume Colors**:
- Green bars = Accumulation phase
- Red bars = Distribution phase
- Blue bars = Neutral
""")
