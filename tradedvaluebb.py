# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go

# --- App Title ---
st.set_page_config(page_title="Stock Traded Value %BB Tracker", layout="wide")
st.title("📈 Stock %BB Analysis on Traded Value (Price × Volume)")

# --- Sidebar Inputs ---
symbol = st.text_input("Enter Stock Symbol (e.g., HBLENGIN):", "HBLENGIN").upper()
granularity = st.selectbox("Select Granularity", ["1d", "1wk"])
lookback_days = st.number_input("Lookback Period (days):", min_value=30, max_value=2000, value=365)
bb_window = st.slider("Bollinger Band Window:", 10, 60, 20)
bb_std = st.slider("BB Standard Deviations:", 1.0, 3.0, 2.0, step=0.1)

# --- Date Range ---
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days=lookback_days)

# --- Fetch Data ---
ticker = f"{symbol}.NS"
st.write(f"Fetching data for **{ticker}** from {start_date} to {end_date}...")
data = yf.download(ticker, start=start_date, end=end_date, interval=granularity, progress=False)

if data.empty:
    st.error("No data found. Check the symbol or try a longer date range.")
    st.stop()

# --- Ensure Continuous Dates ---
data = data.asfreq('D')  # daily frequency even if missing
data[['Open','High','Low','Close','Adj Close']] = data[['Open','High','Low','Close','Adj Close']].ffill()
data['Volume'] = data['Volume'].fillna(0)

# --- Compute Traded Value and %BB ---
data['Traded_Value'] = data['Close'] * data['Volume']

# Bollinger Band Calculations
rolling_mean = data['Traded_Value'].rolling(bb_window).mean()
rolling_std = data['Traded_Value'].rolling(bb_window).std()

data['BB_Upper'] = rolling_mean + (bb_std * rolling_std)
data['BB_Lower'] = rolling_mean - (bb_std * rolling_std)
data['%BB'] = (data['Traded_Value'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

# --- Layout ---
col1, col2 = st.columns([2, 1])

# --- Plot Price Chart ---
with col1:
    st.subheader("Stock Price & Bollinger Bands (%BB of Traded Value)")
    fig = go.Figure()

    # Price chart
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'], name='Close Price',
        mode='lines', line=dict(color='royalblue', width=2)
    ))

    # %BB chart (as secondary axis)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['%BB'] * data['Close'].max(), 
        name='%BB (scaled)', mode='lines',
        line=dict(color='orange', width=2, dash='dot'),
        yaxis='y2'
    ))

    fig.update_layout(
        yaxis=dict(title="Price (₹)", side='left'),
        yaxis2=dict(title="%BB (scaled)", overlaying='y', side='right', showgrid=False),
        title=f"{symbol} — Price vs Traded Value %BB",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Summary Stats ---
with col2:
    st.subheader("Recent Statistics")
    latest = data.iloc[-1]
    st.metric("Last Close (₹)", f"{latest['Close']:.2f}")
    st.metric("%BB of Traded Value", f"{latest['%BB']*100:.1f}%")
    st.metric("Avg Traded Value (₹ Cr)", f"{data['Traded_Value'].rolling(bb_window).mean().iloc[-1]/1e7:.2f}")
    st.write("— %BB near 0 ⇒ low activity; near 1 ⇒ high activity/band top")

# --- Option to Show Data ---
if st.checkbox("Show raw data table"):
    st.dataframe(data.tail(50))
