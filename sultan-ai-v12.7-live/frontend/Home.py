import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
current_file = Path(__file__).resolve()
frontend_dir = current_file.parent
project_dir = frontend_dir.parent
backend_dir = project_dir / 'backend'
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_dir))

# Page config
st.set_page_config(
    page_title="Sultan AI Trading",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header, .stDeployButton { display: none !important; }

    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
    }

    .main .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
    }

    /* Hero Section */
    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #8892b0;
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-label {
        color: #8892b0;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e6f1ff;
    }

    .metric-value.green { color: #64ffda; }
    .metric-value.red { color: #ff6b6b; }
    .metric-value.purple { color: #bd93f9; }
    .metric-value.blue { color: #8be9fd; }

    /* Signal Card */
    .signal-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .signal-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }

    .signal-card.buy::before { background: linear-gradient(90deg, #64ffda, #50fa7b); }
    .signal-card.sell::before { background: linear-gradient(90deg, #ff6b6b, #ff5555); }
    .signal-card.hold::before { background: linear-gradient(90deg, #f1fa8c, #ffb86c); }

    .signal-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
    }

    .signal-title.buy { color: #64ffda; }
    .signal-title.sell { color: #ff6b6b; }
    .signal-title.hold { color: #f1fa8c; }

    /* Level Cards */
    .level-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .level-label { color: #8892b0; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
    .level-value { font-size: 1.2rem; font-weight: 600; margin-top: 0.3rem; }
    .level-value.entry { color: #8be9fd; }
    .level-value.sl { color: #ff6b6b; }
    .level-value.tp { color: #64ffda; }
    .level-value.rr { color: #bd93f9; }

    /* Confidence Bar */
    .confidence-bar {
        background: linear-gradient(135deg, rgba(100, 255, 218, 0.2) 0%, rgba(139, 233, 253, 0.2) 100%);
        border: 2px solid rgba(100, 255, 218, 0.3);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .confidence-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #64ffda, #8be9fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }

    /* Divider */
    hr { border-color: rgba(255, 255, 255, 0.1); margin: 2rem 0; }

    /* Section titles */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6f1ff;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Hide streamlit elements */
    div[data-testid="stToolbar"] { display: none; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# Forex pairs
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY"]

YAHOO_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "XAUUSD": "GC=F", "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",
    "USDCHF": "CHF=X", "NZDUSD": "NZDUSD=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X"
}

# Hero Section
st.markdown("""
<div class="hero">
    <h1 class="hero-title">Sultan AI</h1>
    <p class="hero-subtitle">AI-Powered Trading Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Controls
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_symbol = st.selectbox("Select Trading Pair", FOREX_PAIRS, index=0, label_visibility="collapsed")

# Fetch data from Yahoo Finance
@st.cache_data(ttl=300)
def fetch_data(symbol):
    try:
        ticker = YAHOO_MAP.get(symbol, f"{symbol}=X")
        data = yf.download(ticker, period="60d", interval="1h", progress=False)
        if data.empty:
            data = yf.download(ticker, period="30d", interval="1d", progress=False)
        return data
    except:
        return pd.DataFrame()

with st.spinner(f"Loading {selected_symbol}..."):
    df = fetch_data(selected_symbol)

if df.empty:
    st.error(f"Unable to load data for {selected_symbol}. Please try another pair.")
    st.stop()

# Flatten multi-index columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Calculate indicators
def calculate_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']

    return df.dropna()

df = calculate_indicators(df)

if len(df) < 20:
    st.error("Insufficient data to analyze.")
    st.stop()

# Generate signals
def generate_signal(df):
    last = df.iloc[-1]
    signals = 0

    # RSI
    if last['RSI'] < 30: signals += 2
    elif last['RSI'] > 70: signals -= 2
    elif last['RSI'] < 40: signals += 1
    elif last['RSI'] > 60: signals -= 1

    # MACD
    if last['MACD'] > last['MACD_Signal']: signals += 1
    else: signals -= 1

    # MA Cross
    if last['SMA_20'] > last['SMA_50']: signals += 1
    else: signals -= 1

    # Price vs MA
    if last['Close'] > last['SMA_20']: signals += 1
    else: signals -= 1

    # Determine action
    if signals >= 2:
        action = "BUY"
        confidence = min(90, 60 + signals * 5)
    elif signals <= -2:
        action = "SELL"
        confidence = min(90, 60 + abs(signals) * 5)
    else:
        action = "HOLD"
        confidence = 50

    return action, confidence, signals

action, confidence, signal_score = generate_signal(df)

# Current data
current_price = float(df['Close'].iloc[-1])
prev_price = float(df['Close'].iloc[-2])
change_pct = ((current_price - prev_price) / prev_price) * 100
high_24h = float(df['High'].tail(24).max())
low_24h = float(df['Low'].tail(24).min())
rsi = float(df['RSI'].iloc[-1])

# Metrics Row
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Price</div>
        <div class="metric-value blue">{current_price:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    color_class = "green" if change_pct >= 0 else "red"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">24h Change</div>
        <div class="metric-value {color_class}">{change_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">24h High</div>
        <div class="metric-value green">{high_24h:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">24h Low</div>
        <div class="metric-value red">{low_24h:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    rsi_color = "green" if rsi < 40 else "red" if rsi > 60 else "purple"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RSI</div>
        <div class="metric-value {rsi_color}">{rsi:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

# Signal Card
st.markdown("<br>", unsafe_allow_html=True)
signal_class = action.lower()
st.markdown(f"""
<div class="signal-card {signal_class}">
    <div style="color: #8892b0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">AI Signal</div>
    <div class="signal-title {signal_class}">{action}</div>
    <div style="color: #e6f1ff; font-size: 1.2rem;">Confidence: <strong>{confidence}%</strong></div>
</div>
""", unsafe_allow_html=True)

# Trading Levels
st.markdown("<br>", unsafe_allow_html=True)
atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]

if action == "BUY":
    entry = current_price
    sl = entry - (2 * atr)
    tp = entry + (3 * atr)
elif action == "SELL":
    entry = current_price
    sl = entry + (2 * atr)
    tp = entry - (3 * atr)
else:
    entry = current_price
    sl = entry - atr
    tp = entry + atr

rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 1

l1, l2, l3, l4 = st.columns(4)
with l1:
    st.markdown(f"""<div class="level-card"><div class="level-label">Entry</div><div class="level-value entry">{entry:.5f}</div></div>""", unsafe_allow_html=True)
with l2:
    st.markdown(f"""<div class="level-card"><div class="level-label">Stop Loss</div><div class="level-value sl">{sl:.5f}</div></div>""", unsafe_allow_html=True)
with l3:
    st.markdown(f"""<div class="level-card"><div class="level-label">Take Profit</div><div class="level-value tp">{tp:.5f}</div></div>""", unsafe_allow_html=True)
with l4:
    st.markdown(f"""<div class="level-card"><div class="level-label">Risk:Reward</div><div class="level-value rr">1:{rr:.1f}</div></div>""", unsafe_allow_html=True)

# Chart
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Price Chart</div>', unsafe_allow_html=True)

# Create chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.7, 0.3], subplot_titles=('', 'RSI'))

# Candlestick
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    increasing_line_color='#64ffda', decreasing_line_color='#ff6b6b',
    increasing_fillcolor='#64ffda', decreasing_fillcolor='#ff6b6b', name='Price'
), row=1, col=1)

# SMAs
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
    line=dict(color='#bd93f9', width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
    line=dict(color='#ffb86c', width=1.5)), row=1, col=1)

# Entry lines
fig.add_hline(y=entry, line_dash="dash", line_color="#8be9fd", annotation_text="Entry", row=1, col=1)
fig.add_hline(y=sl, line_dash="dash", line_color="#ff6b6b", annotation_text="SL", row=1, col=1)
fig.add_hline(y=tp, line_dash="dash", line_color="#64ffda", annotation_text="TP", row=1, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
    line=dict(color='#bd93f9', width=2)), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#64ffda", row=2, col=1)

fig.update_layout(
    height=600,
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=30, b=0)
)

fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', showgrid=True)
fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', showgrid=True)

st.plotly_chart(fig, use_container_width=True)

# Confidence Summary
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">🎯 AI Confidence</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="confidence-bar">
    <div style="color: #8892b0; font-size: 0.9rem; margin-bottom: 0.5rem;">Overall Confidence Score</div>
    <div class="confidence-value">{confidence}%</div>
    <div style="color: #8892b0; font-size: 0.8rem; margin-top: 0.5rem;">Based on RSI, MACD, Moving Averages & Price Action</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.8rem; padding: 1rem 0;">
    Sultan AI v12.7 • AI-Powered Trading Analysis • Not Financial Advice
</div>
""", unsafe_allow_html=True)
