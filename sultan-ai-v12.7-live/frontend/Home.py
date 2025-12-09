import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import os
import sys
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from analytics import calculate_advanced_indicators, calculate_strength_score, analyze_news_sentiment
from ml_model import predict_price_movement, calculate_entry_exit_levels, get_trading_recommendation
from fetch_news import get_news_for_symbol, load_news_cache, save_news_cache
from ai_analysis import generate_trading_insight
from data_loader import load_csv_robust
from smart_data_fetcher import smart_load_data, fast_load_only

# ============================================================
# 🎨 APP CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Trading Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 UNIQUE MODERN DESIGN
# ============================================================
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Style sidebar for navigation - make it visible and beautiful */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 100%) !important;
        border-right: 2px solid rgba(0, 245, 255, 0.3);
        min-width: 250px;
    }
    
    /* Sidebar navigation list */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        padding: 1rem 0.5rem;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li {
        margin: 0.5rem 0;
    }
    
    /* Navigation links */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        color: #00f5ff !important;
        background: rgba(0, 245, 255, 0.1) !important;
        border: 1px solid rgba(0, 245, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem 0 !important;
        display: block !important;
        text-decoration: none !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: rgba(0, 245, 255, 0.2) !important;
        border-color: #00f5ff !important;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0, 245, 255, 0.4) !important;
    }
    
    /* Active page */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] [aria-current="page"] {
        background: rgba(0, 245, 255, 0.3) !important;
        border-left: 4px solid #00f5ff !important;
        box-shadow: 0 4px 12px rgba(0, 245, 255, 0.5) !important;
        color: #ffffff !important;
    }
    
    /* Clean background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }
    
    /* Header */
    .dashboard-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .dashboard-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f5ff 0%, #00d4ff 50%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
    }
    
    .dashboard-subtitle {
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Control bar - clean and centered */
    .control-bar {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 2rem auto;
        max-width: 1200px;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 1.5rem;
    }
    
    /* Signal card - modern glassmorphism */
    .signal-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .signal-buy {
        border-left: 5px solid #00ff88;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);
    }
    
    .signal-sell {
        border-left: 5px solid #ff3366;
        box-shadow: 0 8px 32px rgba(255, 51, 102, 0.2);
    }
    
    .signal-hold {
        border-left: 5px solid #ffaa00;
        box-shadow: 0 8px 32px rgba(255, 170, 0, 0.2);
    }
    
    /* Metric cards */
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 245, 255, 0.2);
    }
    
    /* Level cards */
    .level-card {
        background: rgba(0, 245, 255, 0.1);
        border: 2px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Custom selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Custom checkbox */
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(0, 245, 255, 0.4);
    }
    
    /* Info text */
    .info-text {
        color: #a0a0a0;
        font-size: 0.9rem;
    }
    
    /* Navigation bar */
    .nav-bar {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem 2rem;
        margin: 1rem auto 2rem auto;
        max-width: 1200px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .nav-btn {
        background: rgba(0, 245, 255, 0.1);
        border: 2px solid rgba(0, 245, 255, 0.3);
        color: #00f5ff;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: pointer;
        display: inline-block;
        font-size: 1rem;
        border: none;
        outline: none;
    }
    
    .nav-btn:hover {
        background: rgba(0, 245, 255, 0.2);
        border: 2px solid #00f5ff;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 245, 255, 0.3);
    }
    
    .nav-btn.active {
        background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
        color: white;
        border: 2px solid #00f5ff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 📊 DATA SETUP
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

# Comprehensive forex pairs list - always available
COMMON_FOREX_PAIRS = [
    "XAUUSD",  # Gold
    "EURUSD",  # Euro/US Dollar
    "GBPUSD",  # British Pound/US Dollar
    "USDJPY",  # US Dollar/Japanese Yen
    "AUDUSD",  # Australian Dollar/US Dollar
    "USDCAD",  # US Dollar/Canadian Dollar
    "USDCHF",  # US Dollar/Swiss Franc
    "NZDUSD",  # New Zealand Dollar/US Dollar
    "EURGBP",  # Euro/British Pound
    "EURJPY",  # Euro/Japanese Yen
    "GBPJPY",  # British Pound/Japanese Yen
    "AUDJPY",  # Australian Dollar/Japanese Yen
    "EURCHF",  # Euro/Swiss Franc
    "GBPCHF",  # British Pound/Swiss Franc
    "AUDNZD",  # Australian Dollar/New Zealand Dollar
    "EURAUD",  # Euro/Australian Dollar
    "EURCAD",  # Euro/Canadian Dollar
    "GBPAUD",  # British Pound/Australian Dollar
    "XAGUSD",  # Silver
    # XAUGBP and XAUJPY not available on Yahoo Finance - removed
]

# Yahoo Finance mapping - comprehensive list
yahoo_map = {
    "XAUUSD": "GC=F",      # Gold futures
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "EURCHF": "EURCHF=X",
    "GBPCHF": "GBPCHF=X",
    "AUDNZD": "AUDNZD=X",
    "EURAUD": "EURAUD=X",
    "EURCAD": "EURCAD=X",
    "GBPAUD": "GBPAUD=X",
    "XAGUSD": "SI=F",      # Silver futures
    # XAUGBP and XAUJPY not available as direct tickers - removed
}

# Get available symbols from CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith(".")]
symbols_from_files = [f.split(".")[0] for f in csv_files]

# Combine common forex pairs with symbols from files
# Priority: Show all common forex pairs first, then add any other symbols from files
all_available_symbols = list(set(COMMON_FOREX_PAIRS + symbols_from_files))
all_available_symbols.sort()  # Sort alphabetically

# Filter forex symbols - include all common pairs plus any from files that look like forex
forex_symbols = list(set(COMMON_FOREX_PAIRS + [s for s in symbols_from_files if any(s.startswith(fx) for fx in ["EUR", "USD", "GBP", "AUD", "JPY", "CAD", "XAU", "XAG", "CHF", "NZD"])]))
forex_symbols.sort()

# Stock symbols are everything else
stock_symbols = [s for s in all_available_symbols if s not in forex_symbols]
stock_symbols.sort()

# Don't require CSV files - allow fetching data on the fly
if not symbols_from_files:
    st.warning("⚠️ No local data found. Data will be fetched in real-time. Run `python backend/fetch_data.py` for faster loading.")

# ============================================================
# 🎨 HEADER WITH NAVIGATION
# ============================================================
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">🚀 AI Trading Home</h1>
    <p class="dashboard-subtitle">Real-time Market Analysis & Trading Signals</p>
</div>
""", unsafe_allow_html=True)

# Top Navigation Bar - Centered buttons
st.markdown("""
<div class="nav-bar">
    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; width: 100%;">
        <span style="color: #00f5ff; font-weight: bold; margin-right: 1rem; font-size: 1.1rem;">📑 Pages:</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation buttons using columns
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns(6)

with nav_col1:
    if st.button("🏠 Home", use_container_width=True, type="secondary"):
        st.switch_page("Home.py")

with nav_col2:
    if st.button("🎯 Trading Signals", use_container_width=True, type="secondary"):
        st.switch_page("pages/Trading_Signals.py")

with nav_col3:
    if st.button("📈 Advanced Charts", use_container_width=True, type="secondary"):
        st.switch_page("pages/Advanced_Charts.py")

with nav_col4:
    if st.button("📊 Backtesting", use_container_width=True, type="secondary"):
        st.switch_page("pages/Backtesting.py")

with nav_col5:
    if st.button("📰 News", use_container_width=True, type="secondary"):
        st.switch_page("pages/News.py")

with nav_col6:
    if st.button("📋 Methodology", use_container_width=True, type="secondary"):
        st.switch_page("pages/Methodology.py")

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar note
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(0, 245, 255, 0.2); margin-bottom: 1rem;">
    <h3 style="color: #00f5ff; margin: 0; font-size: 1.3rem;">📑 Navigation</h3>
    <p style="color: #a0a0a0; font-size: 0.8rem; margin: 0.5rem 0 0 0;">Use buttons above or sidebar</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 🎛️ CONTROL BAR - CENTERED
# ============================================================
st.markdown('<div class="control-bar">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 3, 2])

with col2:
    # Symbol selection
    symbol_category = st.radio("", ["Forex", "Stocks", "All"], index=2, horizontal=True, label_visibility="collapsed")
    
    if symbol_category == "Forex":
        available_symbols = forex_symbols if forex_symbols else COMMON_FOREX_PAIRS
    elif symbol_category == "Stocks":
        available_symbols = stock_symbols if stock_symbols else []
    else:
        available_symbols = all_available_symbols if all_available_symbols else COMMON_FOREX_PAIRS
    
    selected_symbol = st.selectbox("Select Symbol", available_symbols if available_symbols else COMMON_FOREX_PAIRS, key="symbol_select")

symbol_ticker = yahoo_map.get(selected_symbol, selected_symbol)
csv_path = os.path.join(data_dir, f"{selected_symbol}.csv")

# Settings row
col_set1, col_set2, col_set3 = st.columns(3)
with col_set1:
    use_real_time = st.checkbox("🔄 Real-time", value=False)
with col_set2:
    show_news = st.checkbox("📰 News", value=False)
with col_set3:
    show_ai = st.checkbox("🤖 AI Analysis", value=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ⚡ DATA LOADING
# ============================================================
@st.cache_data(ttl=300)
def cached_smart_load(path, symbol, use_rt, max_age=15):
    return smart_load_data(path, symbol, use_rt, max_age)

status_placeholder = st.empty()

with status_placeholder:
    st.info(f"⏳ Loading {selected_symbol}...")

if use_real_time:
    with st.spinner("⚡ Updating data..."):
        df_cached = fast_load_only(csv_path)
        if not df_cached.empty:
            try:
                df = cached_smart_load(csv_path, symbol_ticker, use_real_time, max_age=15)
                if df.empty:
                    df = df_cached
            except:
                df = df_cached
        else:
            df = cached_smart_load(csv_path, symbol_ticker, use_real_time, max_age=15)
else:
    df = fast_load_only(csv_path)

if df.empty:
    status_placeholder.error(f"⚠️ No data for {selected_symbol}")
    st.stop()

status_placeholder.success(f"✅ Loaded {len(df)} records")

# ============================================================
# 📊 CALCULATE INDICATORS
# ============================================================
@st.cache_data
def compute_indicators(df):
    df_copy = df.copy()
    df_copy = calculate_advanced_indicators(df_copy)
    strength = calculate_strength_score(df_copy)
    return df_copy, strength

try:
    with st.spinner("📊 Calculating indicators..."):
        df, strength_score = compute_indicators(df)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ============================================================
# 🤖 AI PREDICTIONS
# ============================================================
@st.cache_data(ttl=300)
def get_predictions(df_data, symbol):
    try:
        # Pass symbol to use pre-trained models
        ml_pred = predict_price_movement(df_data, symbol=symbol)
        trading_rec = get_trading_recommendation(df_data, ml_pred)
        entry_levels = calculate_entry_exit_levels(df_data, trading_rec)
        return ml_pred, trading_rec, entry_levels
    except Exception as e:
        current_price = df_data['Close'].iloc[-1] if len(df_data) > 0 else 0
        ml_pred = {'direction': 'NEUTRAL', 'confidence': 0.5, 'predicted_change': 0.0, 'method': 'Rule-based'}
        trading_rec = {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Analysis in progress', 'risk_level': 'Medium'}
        entry_levels = {'entry': current_price, 'stop_loss': current_price * 0.99, 'take_profit': current_price * 1.01, 'risk_reward_ratio': 1.0}
        return ml_pred, trading_rec, entry_levels

with st.spinner("🤖 AI Analysis..."):
    ml_prediction, trading_recommendation, entry_levels = get_predictions(df, selected_symbol)

# News sentiment
news_sentiment = {'compound': 0.0, 'count': 0}
if show_news:
    try:
        news_items = get_news_for_symbol(selected_symbol, max_items=5)
        news_sentiment = analyze_news_sentiment(news_items)
    except:
        pass

# ============================================================
# 🎨 MAIN DASHBOARD
# ============================================================
current_price = df['Close'].iloc[-1]
change_pct = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0

# Key Metrics Row
st.markdown("---")
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">Current Price</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #00f5ff;">{current_price:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    delta_color = "#00ff88" if change_pct >= 0 else "#ff3366"
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">24h Change</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: {delta_color};">{change_pct:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">Strength</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #00f5ff;">{strength_score:.0f}/100</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    conf_pct = trading_recommendation.get('confidence', 0.5) * 100
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">Confidence</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #00f5ff;">{conf_pct:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    risk_level = trading_recommendation.get('risk_level', 'Medium')
    risk_color = "#00ff88" if risk_level == "Low" else "#ffaa00" if risk_level == "Medium" else "#ff3366"
    st.markdown(f"""
    <div class="metric-box">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">Risk</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: {risk_color};">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

# Trading Signal Card
action = trading_recommendation.get('action', 'HOLD')
signal_class = f"signal-{action.lower()}" if action in ['BUY', 'SELL'] else "signal-hold"
confidence = trading_recommendation.get('confidence', 0.5) * 100
reason = trading_recommendation.get('reason', 'Analysis in progress')

st.markdown(f"""
<div class="signal-card {signal_class}">
    <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">🎯 {action}</h2>
    <p style="font-size: 1.3rem; margin: 0.5rem 0;"><strong>Confidence:</strong> {confidence:.1f}%</p>
    <p style="font-size: 1rem; color: rgba(255,255,255,0.8); margin-top: 1rem;">{reason}</p>
</div>
""", unsafe_allow_html=True)

# Entry, Stop Loss, Take Profit
st.markdown("---")
st.markdown("<h3 style='text-align: center; color: #00f5ff; margin: 2rem 0;'>📊 Trading Levels</h3>", unsafe_allow_html=True)

col_l1, col_l2, col_l3, col_l4 = st.columns(4)

entry = entry_levels.get('entry', current_price)
stop_loss = entry_levels.get('stop_loss', current_price * 0.99)
take_profit = entry_levels.get('take_profit', current_price * 1.01)
risk_reward = entry_levels.get('risk_reward_ratio', 1.0)

with col_l1:
    st.markdown(f"""
    <div class="level-card">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">Entry Price</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #00f5ff;">{entry:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_l2:
    st.markdown(f"""
    <div class="level-card">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">Stop Loss</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #ff3366;">{stop_loss:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_l3:
    st.markdown(f"""
    <div class="level-card">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">Take Profit</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #00ff88;">{take_profit:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_l4:
    st.markdown(f"""
    <div class="level-card">
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">Risk:Reward</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #00f5ff;">{risk_reward:.2f}:1</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 📈 CHART
# ============================================================
st.markdown("---")
st.markdown("<h3 style='text-align: center; color: #00f5ff; margin: 2rem 0;'>📈 Price Chart</h3>", unsafe_allow_html=True)

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=('Price Action', 'RSI', 'Volume')
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff3366'
    ), row=1, col=1
)

# Moving averages
if "SMA_20" in df.columns:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA_20"], mode="lines", name="SMA 20",
                  line=dict(color="#00f5ff", width=1.5)), row=1, col=1
    )

if "SMA_50" in df.columns:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["SMA_50"], mode="lines", name="SMA 50",
                  line=dict(color="#ffaa00", width=1.5)), row=1, col=1
    )

# Entry/Stop/Take Profit lines
fig.add_hline(y=entry, line_dash="dash", line_color="#00f5ff", 
             annotation_text=f"Entry: {entry:.5f}", row=1, col=1, opacity=0.7)
fig.add_hline(y=stop_loss, line_dash="dash", line_color="#ff3366", 
             annotation_text=f"Stop: {stop_loss:.5f}", row=1, col=1, opacity=0.7)
fig.add_hline(y=take_profit, line_dash="dash", line_color="#00ff88", 
             annotation_text=f"Target: {take_profit:.5f}", row=1, col=1, opacity=0.7)

# RSI
if "RSI" in df.columns:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI",
                  line=dict(color="#9b59b6", width=1.5)), row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ff3366", row=2, col=1, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", row=2, col=1, opacity=0.5)

# Volume
if "Volume" in df.columns and df["Volume"].sum() > 0:
    colors = ['#ff3366' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#00ff88' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors, opacity=0.6), row=3, col=1
    )

fig.update_layout(
    template="plotly_dark",
    height=800,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 🤖 AI INSIGHTS
# ============================================================
if show_ai:
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #00f5ff; margin: 2rem 0;'>🤖 AI Analysis</h3>", unsafe_allow_html=True)
    
    try:
        insight = generate_trading_insight(
            df, ml_prediction, strength_score, news_sentiment, 
            trading_recommendation, entry_levels
        )
        st.markdown(f"""
        <div class="signal-card" style="text-align: left; padding: 2rem; color: #ffffff;">
            {insight.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        import traceback
        # Show actual error for debugging, but make it user-friendly
        st.warning(f"⚠️ AI Analysis: Generating enhanced insights...")
        # Fallback: show basic insight from trading recommendation
        try:
            action = trading_recommendation.get('action', 'HOLD')
            confidence = trading_recommendation.get('confidence', 0.5) * 100
            reason = trading_recommendation.get('reason', 'Analysis complete')
            entry = entry_levels.get('entry', df['Close'].iloc[-1])
            stop_loss = entry_levels.get('stop_loss', entry * 0.99)
            take_profit = entry_levels.get('take_profit', entry * 1.01)
            
            st.markdown(f"""
            <div class="signal-card" style="text-align: left; padding: 2rem; color: #ffffff;">
                <h3>🤖 AI Recommendation: {action}</h3>
                <p><strong>Confidence:</strong> {confidence:.0f}%</p>
                <p><strong>Reason:</strong> {reason}</p>
                <p><strong>Entry:</strong> {entry:.5f} | <strong>Stop Loss:</strong> {stop_loss:.5f} | <strong>Take Profit:</strong> {take_profit:.5f}</p>
                <p><strong>Technical Strength:</strong> {strength_score:.0f}/100</p>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.info("📊 Loading comprehensive AI analysis...")
    
    # ============================================================
    # 📊 CONFIDENCE SUMMARY
    # ============================================================
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calculate confidence percentages
    # 1. AI Confidence (from ML prediction)
    ai_confidence = ml_prediction.get('confidence', 0.5) * 100
    
    # 2. News Confidence (convert sentiment to percentage)
    sentiment_val = news_sentiment.get('compound', 0.0)
    # Convert sentiment (-1 to 1) to confidence (0-100%)
    # Positive sentiment = higher confidence, negative = lower confidence
    news_confidence = max(0, min(100, 50 + (sentiment_val * 50)))  # Scale -1 to 1 -> 0 to 100
    
    # 3. Analysis/Technical Confidence (from strength score)
    analysis_confidence = strength_score  # Already 0-100
    
    # Combined total (weighted average)
    # Weights: AI 45%, Analysis 35%, News 20%
    combined_confidence = (ai_confidence * 0.45) + (analysis_confidence * 0.35) + (news_confidence * 0.20)
    
    # Determine color for combined confidence
    if combined_confidence >= 75:
        combined_color = "#00ff88"  # Green (high confidence)
    elif combined_confidence >= 60:
        combined_color = "#00f5ff"  # Cyan (medium-high)
    elif combined_confidence >= 45:
        combined_color = "#ffaa00"  # Orange (medium)
    else:
        combined_color = "#ff3366"  # Red (low)
    
    # Individual colors
    ai_color = "#00f5ff"
    analysis_color = "#00f5ff"
    news_color = "#00ff88" if sentiment_val > 0.1 else "#ff3366" if sentiment_val < -0.1 else "#ffaa00"
    
    # Display confidence summary using columns for better rendering
    st.markdown("### 📊 Combined Confidence Summary")
    
    col_conf1, col_conf2, col_conf3 = st.columns(3)
    
    with col_conf1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: rgba(0, 0, 0, 0.3); border-radius: 15px; border: 2px solid {ai_color};">
            <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">🤖 AI Confidence</div>
            <div style="font-size: 2.5rem; font-weight: bold; color: {ai_color};">{ai_confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_conf2:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: rgba(0, 0, 0, 0.3); border-radius: 15px; border: 2px solid {analysis_color};">
            <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">📈 Analysis Confidence</div>
            <div style="font-size: 2.5rem; font-weight: bold; color: {analysis_color};">{analysis_confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_conf3:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: rgba(0, 0, 0, 0.3); border-radius: 15px; border: 2px solid {news_color};">
            <div style="font-size: 0.9rem; color: #a0a0a0; margin-bottom: 0.5rem;">📰 News Confidence</div>
            <div style="font-size: 2.5rem; font-weight: bold; color: {news_color};">{news_confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(0, 245, 255, 0.15) 0%, rgba(0, 212, 255, 0.15) 100%); border-radius: 20px; border: 3px solid {combined_color}; margin: 1rem 0;">
        <div style="font-size: 1.1rem; color: #a0a0a0; margin-bottom: 0.5rem;">🎯 Overall Combined Confidence</div>
        <div style="font-size: 4rem; font-weight: bold; color: {combined_color}; text-shadow: 0 0 25px {combined_color};">{combined_confidence:.1f}%</div>
        <div style="font-size: 0.9rem; color: #a0a0a0; margin-top: 0.5rem;">(Weighted: AI 45% | Analysis 35% | News 20%)</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 📰 NEWS (if enabled)
# ============================================================
if show_news and news_sentiment.get('count', 0) > 0:
    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #00f5ff; margin: 2rem 0;'>📰 Market News</h3>", unsafe_allow_html=True)
    
    sentiment_val = news_sentiment.get('compound', 0)
    sentiment_text = "Positive" if sentiment_val > 0.1 else "Negative" if sentiment_val < -0.1 else "Neutral"
    sentiment_color = "#00ff88" if sentiment_val > 0.1 else "#ff3366" if sentiment_val < -0.1 else "#ffaa00"
    
    st.markdown(f"""
    <div class="metric-box" style="text-align: center;">
        <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">News Sentiment: <span style="color: {sentiment_color};">{sentiment_text}</span></div>
        <div style="font-size: 0.9rem; color: #a0a0a0;">Score: {sentiment_val:+.2f} | Articles: {news_sentiment.get('count', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
