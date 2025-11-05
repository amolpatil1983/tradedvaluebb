import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import StringIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Money Analyzer", layout="wide")
st.title("🎯 Smart Money Accumulation & Distribution Detector")

# --- CORE INDICATOR FUNCTIONS ---
def compute_bb(df, column, window=20, std=2.0):
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
    roll_up = gain.rolling(window).mean()
    roll_down = loss.rolling(window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def compute_adx(df, window=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = (high.diff()).where((high.diff() > low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window).mean()

def detect_accumulation_signals(data):
    signals = pd.DataFrame(index=data.index)
    vol_ma = data["Volume"].rolling(20).mean()
    price_change = data["Close"].pct_change(20)
    signals["volume_accumulation"] = (data["Volume"] > vol_ma * 1.2) & (price_change < 0.05)
    tv_ma = data["Traded_Value"].rolling(20).mean()
    signals["value_surge"] = (data["Traded_Value"] > tv_ma * 1.3) & (data["%B_Close"] < 40)
    price_slope = data["Close"].diff(5)
    rsi_slope = data["RSI"].diff(5)
    signals["rsi_divergence"] = (price_slope < 0) & (rsi_slope > 0) & (data["RSI"] < 45)
    signals["adx_compression"] = (data["ADX"] < 20) & (data["ADX"].diff() < 0)
    signals["bb_squeeze"] = data["BB_Bandwidth"] < data["BB_Bandwidth"].rolling(50).quantile(0.2)
    signals["at_support"] = (data["%B_Close"] > 0) & (data["%B_Close"] < 30)
    score = signals.astype(int).sum(axis=1)
    return score, signals

def detect_distribution_signals(data):
    signals = pd.DataFrame(index=data.index)
    vol_ma = data["Volume"].rolling(20).mean()
    price_change = data["Close"].pct_change(20)
    signals["volume_distribution"] = (data["Volume"] > vol_ma * 1.2) & (price_change > -0.05) & (data["%B_Close"] > 60)
    tv_ma = data["Traded_Value"].rolling(20).mean()
    signals["value_surge_high"] = (data["Traded_Value"] > tv_ma * 1.3) & (data["%B_Close"] > 70)
    price_slope = data["Close"].diff(5)
    rsi_slope = data["RSI"].diff(5)
    signals["rsi_divergence"] = (price_slope > 0) & (rsi_slope < 0) & (data["RSI"] > 60)
    signals["adx_exhaustion"] = (data["ADX"] > 30) & (data["ADX"].diff() < -0.5)
    signals["at_resistance"] = data["%B_Close"] > 70
    signals["momentum_loss"] = (data["%B_Close"] > 65) & (data["RSI"].diff(3) < -2)
    score = signals.astype(int).sum(axis=1)
    return score, signals

# --- STOCK ANALYZER FUNCTION ---
def analyze_stock(symbol, lookback="6mo", interval="1d"):
    data = yf.download(symbol, period=lookback, interval=interval, progress=False)
    if data.empty:
        return None

    data["Traded_Value"] = data["Close"] * data["Volume"]
    data["RSI"] = compute_rsi(data)
    data["ADX"] = compute_adx(data)
    data["%B_Close"], data["BB_Bandwidth"] = compute_bb(data, "Close", 20)
    data.dropna(inplace=True)

    acc_score, _ = detect_accumulation_signals(data)
    dist_score, _ = detect_distribution_signals(data)

    data["Accumulation_Score"] = acc_score
    data["Distribution_Score"] = dist_score

    return {
        "Symbol": symbol.replace(".NS", ""),
        "Close": round(data["Close"].iloc[-1], 2),
        "RSI": round(data["RSI"].iloc[-1], 1),
        "ADX": round(data["ADX"].iloc[-1], 1),
        "Accumulation_Score": int(acc_score.iloc[-1]),
        "Distribution_Score": int(dist_score.iloc[-1])
    }

# --- SINGLE STOCK MODE ---
st.sidebar.header("🔍 Single Stock Analysis")
symbol = st.sidebar.text_input("Enter Stock Symbol (NSE):", "HDFCBANK").strip().upper()
if st.sidebar.button("Analyze Stock"):
    result = analyze_stock(symbol + ".NS")
    if result:
        st.success(f"✅ {symbol}: Acc={result['Accumulation_Score']}, Dist={result['Distribution_Score']}, RSI={result['RSI']}, ADX={result['ADX']}")
    else:
        st.error("No data found for symbol.")

# --- BATCH MODE ---
st.sidebar.header("📂 Batch Stock Scan (from CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV with SYMBOL and SERIES columns", type=["csv"])

if uploaded_file is not None:
    try:
        stocks_df = pd.read_csv(uploaded_file)
        stocks_df.columns = [c.strip().upper() for c in stocks_df.columns]
        eq_symbols = stocks_df[stocks_df["SERIES"].astype(str).str.strip().str.upper() == "EQ"]["SYMBOL"].unique().tolist()
        st.sidebar.info(f"Found {len(eq_symbols)} EQ symbols in uploaded file.")

        if st.sidebar.button("🚀 Scan for Breakout Candidates"):
            results = []
            progress = st.progress(0)
            for i, sym in enumerate(eq_symbols):
                progress.progress((i + 1) / len(eq_symbols))
                try:
                    res = analyze_stock(sym + ".NS")
                    if res and res["Accumulation_Score"] >= 4:
                        results.append(res)
                except Exception:
                    continue
            progress.empty()

            if results:
                df = pd.DataFrame(results).sort_values(by="Accumulation_Score", ascending=False)
                st.subheader("📈 Strong Breakout Potential Stocks (Accumulation ≥ 4)")
                st.dataframe(df)
            else:
                st.warning("No strong accumulation candidates found.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
