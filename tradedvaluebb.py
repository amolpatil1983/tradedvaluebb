import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------
# Technical indicator functions
# -------------------------
def compute_bbands(series, window=20):
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    percent_b = (series - lower) / (upper - lower)
    return percent_b

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)
    roll_up = gain_series.rolling(window).mean()
    roll_down = loss_series.rolling(window).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_adx(df, window=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window).mean()
    return adx

# -------------------------
# Phase classification
# -------------------------
def classify_phase(bb_price, bb_vol, bb_val, bb_adx, bb_rsi):
    avg_bb = np.nanmean([bb_price, bb_vol, bb_val, bb_adx, bb_rsi])
    if avg_bb > 0.8:
        return "Distribution", 1
    elif avg_bb > 0.6:
        return "Trending Down", 2
    elif avg_bb > 0.4:
        return "Sideways", 3
    elif avg_bb > 0.2:
        return "Trending Up", 4
    else:
        return "Accumulation", 5

# -------------------------
# Data processing
# -------------------------
def analyze_stock(symbol):
    data = yf.download(symbol, period="6mo", interval="1d", progress=False)
    if data.empty:
        return None

    data["ValueTraded"] = data["Close"] * data["Volume"]
    data["RSI"] = compute_rsi(data["Close"])
    data["ADX"] = compute_adx(data)

    data["%BB_Price"] = compute_bbands(data["Close"])
    data["%BB_Volume"] = compute_bbands(data["Volume"])
    data["%BB_Value"] = compute_bbands(data["ValueTraded"])
    data["%BB_ADX"] = compute_bbands(data["ADX"])
    data["%BB_RSI"] = compute_bbands(data["RSI"])

    latest = data.iloc[-1]
    phase, rating = classify_phase(
        latest["%BB_Price"], latest["%BB_Volume"], latest["%BB_Value"],
        latest["%BB_ADX"], latest["%BB_RSI"]
    )
    return symbol, phase, rating, data

# -------------------------
# Streamlit UI
# -------------------------
st.title("📊 Multi-Indicator Stock Phase Scanner")

uploaded_file = st.file_uploader("Upload Stock List (CSV with 'Symbol' and 'SERIES')", type=["csv"])

if uploaded_file:
    stocks_df = pd.read_csv(uploaded_file)
    stocks_df.columns = stocks_df.columns.str.strip()
    stocks_df = stocks_df[stocks_df["SERIES"] == "EQ"]
    symbols = stocks_df["Symbol"].unique().tolist()

    st.write(f"✅ Found {len(symbols)} EQ symbols.")

    results = []
    progress = st.progress(0)
    for i, symbol in enumerate(symbols):
        result = analyze_stock(symbol + ".NS")  # Assuming NSE
        if result:
            results.append(result)
        progress.progress((i + 1) / len(symbols))

    summary = pd.DataFrame(results, columns=["Symbol", "Phase", "Accumulation_Rating", "Data"])
    filtered = summary[summary["Accumulation_Rating"] >= 4]

    st.subheader("📈 High Accumulation Stocks (Rating ≥ 4)")
    st.dataframe(filtered[["Symbol", "Phase", "Accumulation_Rating"]])

    if not filtered.empty:
        selected = st.selectbox("Select a stock to visualize:", filtered["Symbol"])
        sel_data = filtered[filtered["Symbol"] == selected].iloc[0]["Data"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sel_data.index, sel_data["Close"], label="Price", color="black")
        ax.set_title(f"{selected} Phase Visualization: {filtered[filtered['Symbol']==selected].iloc[0]['Phase']}")
        ax.set_ylabel("Price")
        ax.grid(True)
        st.pyplot(fig)
else:
    st.info("⬆️ Upload a CSV file to begin scanning.")
