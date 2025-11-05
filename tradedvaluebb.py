import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Phase Analyzer", layout="wide")

st.title("📊 Stock Phase Analyzer using %BB, RSI, ADX, and Value Metrics")

# --- INPUTS ---
symbol = st.text_input("Enter stock symbol (e.g. TCS.NS, INFY.NS, RELIANCE.NS):", "TCS.NS")
start_date = st.date_input("Start date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End date", pd.to_datetime("today"))
period = st.number_input("Bollinger Bands period", 14, 100, 20)

# --- FETCH DATA ---
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data["ValueTraded"] = data["Close"] * data["Volume"]
    return data

data = load_data(symbol, start_date, end_date)

if data.empty:
    st.error("No data found for the symbol.")
    st.stop()

# --- HELPER FUNCTIONS ---
def percent_bb(series, window=20):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return ((series - lower) / (upper - lower)) * 100

def compute_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(window).mean()
    roll_down = pd.Series(loss).rolling(window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def compute_adx(df, n=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = np.where((high.diff() > low.diff()) & (high.diff() > 0), high.diff(), 0)
    minus_dm = np.where((low.diff() > high.diff()) & (low.diff() > 0), low.diff(), 0)
    tr = np.maximum.reduce([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ])
    atr = pd.Series(tr).rolling(n).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(n).mean()
    return pd.Series(adx.values.flatten(), index=df.index, name="ADX")

# --- COMPUTE INDICATORS ---
data["%BB_Price"] = percent_bb(data["Close"], period)
data["%BB_Volume"] = percent_bb(data["Volume"], period)
data["%BB_Value"] = percent_bb(data["ValueTraded"], period)
data["RSI"] = compute_rsi(data)
data["ADX"] = compute_adx(data)
data["%BB_RSI"] = percent_bb(data["RSI"], period)
data["%BB_ADX"] = percent_bb(data["ADX"], period)

# --- DEFINE PHASE ---
def determine_phase(row):
    price_bb = row["%BB_Price"]
    adx = row["ADX"]
    rsi = row["RSI"]

    if adx < 20 and 40 <= price_bb <= 60:
        return "Accumulation"
    elif adx < 20 and (price_bb > 70):
        return "Distribution"
    elif adx >= 20 and rsi > 55:
        return "Trending Up"
    elif adx >= 20 and rsi < 45:
        return "Trending Down"
    else:
        return "Neutral"

data["Phase"] = data.apply(determine_phase, axis=1)

# --- PLOT ---
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(data.index, data["Close"], color="black", label="Close Price")
ax[0].set_title(f"{symbol} Price Chart")
ax[0].set_ylabel("Price")

# Add vertical color bands for phases
colors = {
    "Accumulation": "#66ff99",
    "Distribution": "#ffcc99",
    "Trending Up": "#99ccff",
    "Trending Down": "#ff9999",
    "Neutral": "#cccccc"
}

for i in range(len(data)):
    phase = data["Phase"].iloc[i]
    ax[0].axvspan(data.index[i], data.index[i], color=colors[phase], alpha=0.4, lw=0)

ax[1].plot(data.index, data["%BB_Price"], label="%BB Price", color="blue")
ax[1].plot(data.index, data["%BB_Volume"], label="%BB Volume", color="orange")
ax[1].plot(data.index, data["%BB_Value"], label="%BB ValueTraded", color="green")
ax[1].plot(data.index, data["%BB_RSI"], label="%BB RSI", color="red")
ax[1].plot(data.index, data["%BB_ADX"], label="%BB ADX", color="purple")
ax[1].set_title("Bollinger Band % Values")
ax[1].set_ylabel("%BB")
ax[1].legend(loc="upper left")

st.pyplot(fig)

# --- CURRENT STATUS ---
st.subheader("📈 Current Market Phase")
latest = data.iloc[-1]
st.metric("Phase", latest["Phase"])
st.write(f"Price %BB: {latest['%BB_Price']:.2f}")
st.write(f"Volume %BB: {latest['%BB_Volume']:.2f}")
st.write(f"ValueTraded %BB: {latest['%BB_Value']:.2f}")
st.write(f"RSI: {latest['RSI']:.2f}")
st.write(f"ADX: {latest['ADX']:.2f}")
