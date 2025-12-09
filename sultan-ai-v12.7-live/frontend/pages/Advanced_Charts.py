import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add backend to path - works for both local and Streamlit Cloud
current_file = Path(__file__).resolve()
pages_dir = current_file.parent
frontend_dir = pages_dir.parent
project_dir = frontend_dir.parent
backend_dir = project_dir / 'backend'
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_dir))

from analytics import calculate_advanced_indicators, calculate_strength_score
from ml_model import predict_price_movement, calculate_entry_exit_levels, get_trading_recommendation
from fetch_news import get_news_for_symbol
from analytics import analyze_news_sentiment
from ai_analysis import generate_trading_insight
from smc_analysis import comprehensive_smc_analysis, compare_user_vs_ai_smc
from data_loader import load_csv_robust, clean_dataframe
from smart_data_fetcher import smart_load_data, fast_load_only

st.set_page_config(page_title="📈 Advanced Charts & Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Style sidebar for navigation */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 100%) !important;
        border-right: 2px solid rgba(0, 245, 255, 0.3);
        min-width: 250px;
    }
    
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
    
    .chart-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .comparison-box {
        background: rgba(0, 245, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .accuracy-high {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .accuracy-medium {
        color: #ffaa00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .accuracy-low {
        color: #ff3366;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    p {
        color: rgba(255, 255, 255, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# Navigation bar
st.markdown("""
<div class="nav-bar">
    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; width: 100%;">
        <span style="color: #00f5ff; font-weight: bold; margin-right: 1rem; font-size: 1.1rem;">📑 Pages:</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation buttons using columns
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

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
    if st.button("📰 News", use_container_width=True, type="secondary"):
        st.switch_page("pages/News.py")

with nav_col5:
    if st.button("📊 Methodology", use_container_width=True, type="secondary"):
        st.switch_page("pages/Methodology.py")

st.markdown("<br>", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">📈 Advanced Charts & AI Analysis Comparison</h1>
    <p class="dashboard-subtitle">Interactive charts with SMC analysis and drawing tools</p>
</div>
""", unsafe_allow_html=True)

# Data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

# Symbol selection
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith(".")]
symbols = [f.split(".")[0] for f in csv_files]

if not symbols:
    st.error("⚠️ No datasets found. Please run fetch_data.py first.")
    st.stop()

forex_symbols = [s for s in symbols if any(s.startswith(fx) for fx in ["EUR", "USD", "GBP", "AUD", "JPY", "CAD", "XAU"])]

col_select, col_mode, col_chart, col_tf = st.columns([2, 1, 1, 1])
with col_select:
    selected_symbol = st.selectbox("📊 Select Symbol", forex_symbols if forex_symbols else symbols, index=0 if forex_symbols else 0)
with col_mode:
    chart_mode = st.selectbox("Mode", ["Live Analysis", "Drawing Tools", "Comparison"], index=2)
with col_chart:
    chart_type = st.selectbox("📈 Chart Type", ["Plotly Chart", "TradingView Chart", "Both"], index=0)
with col_tf:
    # Timeframe selection
    timeframe_options = {
        "1 Minute": "1m",
        "5 Minutes": "5m",
        "15 Minutes": "15m",
        "30 Minutes": "30m",
        "1 Hour": "1h",
        "4 Hours": "4h",
        "1 Day": "1d",
        "1 Week": "1wk",
        "1 Month": "1mo"
    }
    selected_timeframe_label = st.selectbox(
        "⏱️ Timeframe",
        list(timeframe_options.keys()),
        index=4,  # Default to 1 Hour
        help="Select the chart timeframe/interval"
    )
    selected_timeframe = timeframe_options[selected_timeframe_label]

# TradingView Account Settings in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 TradingView Integration")
tradingview_username = st.sidebar.text_input("TradingView Username", value="", help="Enter your TradingView username to sync your charts")
show_tv_drawings = st.sidebar.checkbox("Show My TradingView Drawings", value=False, help="Display your saved drawings from TradingView (requires account connection)")

# TradingView symbol mapping
tradingview_map = {
    "EURUSD": "FX:EURUSD",
    "XAUUSD": "TVC:GOLD",  # TradingView uses TVC:GOLD for gold
    "USDJPY": "FX:USDJPY",
    "GBPUSD": "FX:GBPUSD",
    "AUDUSD": "FX:AUDUSD",
    "USDCAD": "FX:USDCAD",
    "USDCHF": "FX:USDCHF",
    "NZDUSD": "FX:NZDUSD",
    "EURGBP": "FX:EURGBP",
    "EURJPY": "FX:EURJPY",
    "GBPJPY": "FX:GBPJPY",
    "AUDJPY": "FX:AUDJPY",
    "EURCHF": "FX:EURCHF",
    "GBPCHF": "FX:GBPCHF",
    "AUDNZD": "FX:AUDNZD",
    "EURAUD": "FX:EURAUD",
    "EURCAD": "FX:EURCAD",
    "GBPAUD": "FX:GBPAUD",
    "XAGUSD": "TVC:SILVER",  # Silver
    "AAPL": "NASDAQ:AAPL",
    "GOOGL": "NASDAQ:GOOGL",
    "MSFT": "NASDAQ:MSFT",
    "TSLA": "NASDAQ:TSLA",
    "AMZN": "NASDAQ:AMZN",
    "NVDA": "NASDAQ:NVDA",
    "META": "NASDAQ:META",
    "NFLX": "NASDAQ:NFLX",
    "SPY": "AMEX:SPY",
    "QQQ": "NASDAQ:QQQ",
    "BTC-USD": "BINANCE:BTCUSDT",
    "ETH-USD": "BINANCE:ETHUSDT",
}

# Yahoo Finance mapping
yahoo_map = {
    "EURUSD": "EURUSD=X",
    "XAUUSD": "GC=F",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
}
symbol_ticker = yahoo_map.get(selected_symbol, selected_symbol)
tradingview_symbol = tradingview_map.get(selected_symbol, f"FX:{selected_symbol}" if any(selected_symbol.startswith(fx) for fx in ["EUR", "USD", "GBP", "AUD", "JPY", "CAD", "CHF", "NZD"]) else selected_symbol)
csv_path = os.path.join(data_dir, f"{selected_symbol}.csv")

# Settings
use_real_time = st.sidebar.checkbox("🔄 Enable Real-time Updates", value=False, 
                                     help="Fast loading with background updates. Loads cached data instantly, then updates incrementally.")

# Load data with smart fetcher - updated to support timeframes
@st.cache_data(ttl=300)
def load_chart_data_smart(path, ticker, use_rt=False, interval="1h"):
    """Load chart data with smart caching - fast load + optional real-time updates"""
    if use_rt:
        return smart_load_data_with_timeframe(path, ticker, interval, use_real_time=True, max_age_minutes=15)
    else:
        return fast_load_only(path)

def smart_load_data_with_timeframe(csv_path, symbol, interval="1h", use_real_time=False, max_age_minutes=15):
    """
    Load data with specific timeframe/interval support
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    from data_loader import load_csv_robust, save_csv_clean, clean_dataframe
    
    # Check if cached data exists and matches timeframe
    cached_df = load_csv_robust(csv_path)
    
    # If no cache or real-time requested, fetch fresh data with selected timeframe
    if cached_df.empty or use_real_time:
        try:
            # Determine period based on interval
            period_map = {
                "1m": "5d",   # 1 minute - last 5 days max
                "5m": "60d",  # 5 minutes - last 60 days
                "15m": "60d", # 15 minutes - last 60 days
                "30m": "60d", # 30 minutes - last 60 days
                "1h": "730d", # 1 hour - 2 years
                "4h": "730d", # 4 hours - 2 years
                "1d": "2y",   # 1 day - 2 years
                "1wk": "5y",  # 1 week - 5 years
                "1mo": "10y"  # 1 month - 10 years
            }
            period = period_map.get(interval, "2y")
            
            df = yf.download(symbol, period=period, interval=interval, progress=False, timeout=15)
            if not df.empty:
                df = clean_dataframe(df)
                if not df.empty:
                    save_csv_clean(df, csv_path)
                    return df
        except Exception as e:
            st.warning(f"⚠️ Error fetching {interval} data: {e}. Using cached data if available.")
            if not cached_df.empty:
                return cached_df
    
    return cached_df if not cached_df.empty else pd.DataFrame()

try:
    if use_real_time:
        with st.spinner(f"⚡ Loading {selected_timeframe_label} data, updating in background..."):
            df = load_chart_data_smart(csv_path, symbol_ticker, use_real_time, selected_timeframe)
    else:
        with st.spinner(f"📊 Loading {selected_timeframe_label} data..."):
            df = load_chart_data_smart(csv_path, symbol_ticker, False, selected_timeframe)
except Exception as e:
    st.error(f"Error loading data: {e}")
    import traceback
    st.code(traceback.format_exc())
    df = pd.DataFrame()

if df.empty:
    st.error(f"⚠️ No data available for {selected_symbol}")
    st.info("💡 Please run `python backend/fetch_data.py` to download data first")
    st.info(f"📁 Looking for file: {csv_path}")
    if os.path.exists(csv_path):
        st.warning(f"File exists but has no data")
    st.stop()
elif len(df) < 10:
    st.warning(f"⚠️ Limited data available ({len(df)} rows). Charts may not display optimally.")
    st.info("💡 For better results, run `python backend/fetch_data.py` to download more historical data")

# Calculate indicators with error handling
try:
    with st.spinner("Calculating indicators..."):
        # Clean data first using validator
        from data_validator import clean_dataframe_for_analysis, validate_dataframe
        df = clean_dataframe_for_analysis(df)
        is_valid, error_msg = validate_dataframe(df, min_rows=10)  # Reduced minimum
        if not is_valid:
            st.warning(f"⚠️ Data validation warning: {error_msg}")
            st.info("💡 Charts will still be displayed, but analysis may be limited. Run fetch_data.py for better results.")
        
        df = calculate_advanced_indicators(df)
        if df.empty:
            st.error("❌ Error: Data became empty after processing. Please check your data file.")
            st.stop()
        elif len(df) < 10:
            st.warning(f"⚠️ Limited data ({len(df)} rows). Some indicators may not be available.")
        
        strength_score = calculate_strength_score(df) if len(df) >= 10 else 50.0
except Exception as e:
    st.warning(f"⚠️ Error calculating some indicators: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.info("💡 Charts will still be displayed with basic data. Please refresh the page or try a different symbol for full analysis.")
    # Continue with basic data instead of stopping
    if df.empty:
        st.stop()

# SMC Analysis with error handling
try:
    with st.spinner("Running SMC analysis..."):
        try:
            # Data is already cleaned in indicator calculation step above
            # Just ensure it's valid for SMC (needs at least 20 rows)
            from data_validator import validate_dataframe, clean_dataframe_for_analysis
            df = clean_dataframe_for_analysis(df)  # Clean again to be safe
            is_valid, error_msg = validate_dataframe(df, min_rows=20)
            if not is_valid:
                st.warning(f"⚠️ SMC data validation warning: {error_msg}. Using simplified analysis.")
                smc_data = {'support_resistance': {'support': [], 'resistance': []}, 
                           'trend': {'trend_direction': 'UNKNOWN', 'strength': 0.5},
                           'market_structure': {'structure': 'UNKNOWN'},
                           'fibonacci': {'levels': {}, 'nearest_level': None},
                           'nearest_support': None, 'nearest_resistance': None,
                           'current_price': float(df['Close'].iloc[-1]) if len(df) > 0 else 0.0}
            else:
                smc_data = comprehensive_smc_analysis(df)
                
                # Validate SMC data structure
                if not isinstance(smc_data, dict):
                    raise ValueError("SMC analysis returned invalid data structure")
                
                # Ensure all numeric values are floats
                if smc_data.get('nearest_support') is not None:
                    smc_data['nearest_support'] = float(smc_data['nearest_support'])
                if smc_data.get('nearest_resistance') is not None:
                    smc_data['nearest_resistance'] = float(smc_data['nearest_resistance'])
                if smc_data.get('current_price') is not None:
                    smc_data['current_price'] = float(smc_data['current_price'])
        except Exception as e:
            import traceback
            st.warning(f"⚠️ SMC analysis encountered an error. Using simplified analysis.")
            st.info(f"Error details: {str(e)}")
            # Create fallback SMC data
            try:
                current_price = float(df['Close'].iloc[-1]) if len(df) > 0 else 0.0
            except:
                current_price = 0.0
            
            smc_data = {
                'support_resistance': {'support': [], 'resistance': []},
                'trend': {'trend_direction': 'UNKNOWN', 'strength': 0.5},
                'market_structure': {'structure': 'UNKNOWN'},
                'fibonacci': {'levels': {}, 'nearest_level': None},
                'nearest_support': None,
                'nearest_resistance': None,
                'current_price': current_price
            }
except Exception as e:
    import traceback
    error_msg = f"SMC analysis error: {str(e)}"
    st.warning(f"{error_msg}. Using simplified analysis.")
    st.code(traceback.format_exc())
    try:
        current_price = float(df['Close'].iloc[-1]) if len(df) > 0 else 0.0
    except:
        current_price = 0.0
    smc_data = {
        'support_resistance': {'support': [], 'resistance': []},
        'nearest_support': current_price * 0.99 if current_price > 0 else None,
        'nearest_resistance': current_price * 1.01 if current_price > 0 else None,
        'trend': {'trend_direction': 'UNKNOWN', 'strength': 0.5},
        'market_structure': {'structure': 'UNKNOWN'},
        'fibonacci': {'levels': {}, 'nearest_level': None},
        'current_price': current_price
    }

# AI Analysis with error handling
try:
    current_price = df['Close'].iloc[-1] if len(df) > 0 else 0
except:
    current_price = 0

ml_prediction = {'direction': 'NEUTRAL', 'confidence': 0.5, 'predicted_change': 0.0, 'method': 'Rule-based'}
trading_recommendation = {'action': 'HOLD', 'confidence': 0.5, 'reason': 'Analysis in progress', 'risk_level': 'Medium'}
entry_levels = {'entry': current_price, 'stop_loss': current_price * 0.99, 'take_profit': current_price * 1.01, 'risk_reward_ratio': 1.0}

try:
    if len(df) >= 10:
        with st.spinner("Running AI analysis..."):
            ml_prediction = predict_price_movement(df)
            trading_recommendation = get_trading_recommendation(df, ml_prediction)
            entry_levels = calculate_entry_exit_levels(df, trading_recommendation)
except Exception as e:
    st.warning(f"⚠️ ML analysis warning: {e}. Using rule-based analysis.")

# ------------------------------------------------------------------
# Post-processing: make signals safer and more user-friendly
# ------------------------------------------------------------------
MIN_CONFIDENCE = 0.70          # Require at least 70% confidence to trade
MIN_RR = 1.8                   # Require minimum risk/reward
TREND_TF = "4H"                # Higher timeframe for trend filter
ATR_STOP_MULT = 2.0            # ATR stop multiplier
ATR_TP_MULT = 3.0              # ATR target multiplier

def compute_higher_tf_trend(df, tf="4H"):
    """Very lightweight higher timeframe trend detection."""
    try:
        hi = df["Close"].resample(tf).last().dropna()
        if len(hi) < 30:
            return "UNKNOWN"
        sma = hi.rolling(50).mean()
        if hi.iloc[-1] > sma.iloc[-1]:
            return "BULL"
        if hi.iloc[-1] < sma.iloc[-1]:
            return "BEAR"
        return "FLAT"
    except Exception:
        return "UNKNOWN"

def apply_safety_filters(tr_rec, levels, df, strength_score):
    """Apply confidence, RR, trend and ATR-based safety filters."""
    action = tr_rec.get("action", "HOLD")
    conf = tr_rec.get("confidence", 0.5)
    reason = tr_rec.get("reason", "Analysis in progress")
    rr = levels.get("risk_reward_ratio", 1.0)

    # Confidence penalties on thin data
    if len(df) < 50:
        conf *= 0.8
    if len(df) < 25:
        conf *= 0.7

    # Penalize weak technical strength
    if strength_score < 50:
        conf *= 0.85

    # Higher timeframe trend filter
    ht_trend = compute_higher_tf_trend(df, TREND_TF)
    if ht_trend == "BULL" and action == "SELL":
        conf *= 0.7
        reason = f"Bearish signal blocked by {TREND_TF} uptrend"
    if ht_trend == "BEAR" and action == "BUY":
        conf *= 0.7
        reason = f"Bullish signal blocked by {TREND_TF} downtrend"

    # ATR-based levels to enforce sane stops/targets
    try:
        if "ATR" in df.columns:
            atr = float(df["ATR"].iloc[-1])
            if atr > 0 and levels:
                entry = levels.get("entry", current_price)
                if action == "BUY":
                    levels["stop_loss"] = entry - ATR_STOP_MULT * atr
                    levels["take_profit"] = entry + ATR_TP_MULT * atr
                elif action == "SELL":
                    levels["stop_loss"] = entry + ATR_STOP_MULT * atr
                    levels["take_profit"] = entry - ATR_TP_MULT * atr
                if levels["stop_loss"] and levels["take_profit"]:
                    rr = abs(levels["take_profit"] - entry) / max(
                        abs(entry - levels["stop_loss"]), 1e-9
                    )
                    levels["risk_reward_ratio"] = rr
    except Exception:
        pass

    # Enforce RR threshold
    if rr < MIN_RR:
        action = "HOLD"
        reason = f"Risk/Reward too low ({rr:.2f} < {MIN_RR})"

    # Enforce confidence threshold
    if conf < MIN_CONFIDENCE:
        action = "HOLD"
        reason = f"Confidence too low ({conf*100:.0f}% < {MIN_CONFIDENCE*100:.0f}%)"

    tr_rec["action"] = action
    tr_rec["confidence"] = conf
    tr_rec["reason"] = reason
    levels["risk_reward_ratio"] = rr
    return tr_rec, levels

# Apply safety filters
trading_recommendation, entry_levels = apply_safety_filters(
    trading_recommendation, entry_levels, df, strength_score
)

# Get news sentiment (non-blocking)
news_sentiment = {'compound': 0.0, 'count': 0}
try:
    news_items = get_news_for_symbol(selected_symbol, max_items=5)
    news_sentiment = analyze_news_sentiment(news_items)
except Exception as e:
    pass  # News is optional

# TradingView Chart Embed Function with Timeframe Support
def get_tradingview_widget(symbol, theme="dark", height=600, show_drawings=False, username="", interval="1h"):
    """Generate TradingView widget HTML with timeframe support"""
    # Map yfinance intervals to TradingView intervals
    tv_interval_map = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "D",
        "1wk": "W",
        "1mo": "M"
    }
    tv_interval = tv_interval_map.get(interval, "D")
    
    # TradingView Advanced Chart Widget (Free Embed)
    # Note: For full drawing sync, TradingView Charting Library (paid) is required
    # This widget allows basic drawing but drawings are saved locally in browser
    widget_html = f"""
    <div class="tradingview-widget-container" style="height:{height}px;width:100%;border-radius:15px;overflow:hidden;">
        <div id="tradingview_chart_{symbol.replace(':', '_').replace('/', '_')}" style="height:100%;width:100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
            "autosize": true,
            "symbol": "{symbol}",
            "interval": "{tv_interval}",
            "timezone": "Etc/UTC",
            "theme": "{theme}",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#1e1e1e",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_chart_{symbol.replace(':', '_').replace('/', '_')}",
            "height": {height},
            "width": "100%",
            "studies": [
                "RSI@tv-basicstudies",
                "MACD@tv-basicstudies",
                "Volume@tv-basicstudies"
            ],
            "drawings_access": {{
                "type": "all",
                "tools": [
                    {{"name": "Regression Trend"}},
                    {{"name": "Trend Angle"}},
                    {{"name": "Trend Line"}},
                    {{"name": "Horizontal Line"}},
                    {{"name": "Vertical Line"}},
                    {{"name": "Rectangle"}},
                    {{"name": "Ellipse"}},
                    {{"name": "Arrow"}},
                    {{"name": "Text"}},
                    {{"name": "Fibonacci Retracement"}},
                    {{"name": "Fibonacci Extension"}}
                ]
            }},
            "show_popup_button": true,
            "popup_width": "1000",
            "popup_height": "650",
            "hide_side_toolbar": false,
            "save_image": true,
            "no_referral_id": true,
            "referral_id": "{username}" if username else ""
        }});
        </script>
    </div>
    """
    return widget_html

# Main Chart with SMC and Trend Analysis
# Always show status info
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    st.metric("Chart Type", chart_type)
with status_col2:
    st.metric("Symbol", selected_symbol)
with status_col3:
    st.metric("Timeframe", selected_timeframe_label)
with status_col4:
    st.metric("Data Rows", len(df))

# Quick test - always show a simple chart if data exists
if not df.empty and len(df) > 0:
    st.info("✅ Data loaded successfully. Rendering charts...")
    
    # Show a simple test chart first to verify rendering works
    try:
        if 'Close' in df.columns:
            test_fig = go.Figure()
            # Use last 100 rows or all if less
            display_data = df.tail(100) if len(df) > 100 else df
            test_fig.add_trace(go.Scatter(
                x=display_data.index,
                y=display_data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#00f5ff', width=2)
            ))
            test_fig.update_layout(
                title=f"Quick Preview - {selected_symbol}",
                template="plotly_dark",
                height=400,
                showlegend=True,
                xaxis_title="Date",
                yaxis_title="Price"
            )
            st.plotly_chart(test_fig, use_container_width=True, key="test_chart")
            st.success("✅ Chart rendering works! Loading full chart...")
        else:
            st.warning(f"⚠️ 'Close' column not found. Available columns: {list(df.columns)}")
    except Exception as e:
        st.error(f"❌ Test chart failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.write("DataFrame info:")
        st.write(df.info() if hasattr(df, 'info') else str(df))
else:
    st.error("❌ No data available. Cannot render charts.")
    st.info(f"📁 Looking for file: {csv_path}")
    if os.path.exists(csv_path):
        st.warning("File exists but appears empty. Try: python backend/fetch_data.py")

# Always show at least a basic chart if data exists
if not df.empty and len(df) > 0:
    if chart_type in ["Plotly Chart", "Both"]:
        st.subheader("📊 Interactive Chart with Smart Money Concepts (SMC) & Trend Analysis")
    else:
        st.subheader("📊 Basic Price Chart")
    
    # Create advanced chart with error handling
    try:
        # Verify we have valid data
        if df.empty:
            st.error("❌ No data available to display chart.")
            st.info("💡 Please ensure data files exist in the data/ directory")
            st.info(f"📁 Expected file: {csv_path}")
            if os.path.exists(csv_path):
                st.warning("File exists but appears to be empty. Try running: python backend/fetch_data.py")
            st.stop()
        elif len(df) < 5:
            st.warning("⚠️ Very limited data. Chart may not display properly.")
        
        # Show data preview
        with st.expander("📋 Data Preview (first 5 rows)"):
            st.dataframe(df.head())
        
        # If not Plotly Chart mode, show simple chart
        if chart_type not in ["Plotly Chart", "Both"]:
            simple_chart = go.Figure()
            if 'Close' in df.columns:
                simple_chart.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#00f5ff', width=2)
                ))
                simple_chart.update_layout(
                    title=f"{selected_symbol} Price Chart",
                    template="plotly_dark",
                    height=600,
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(simple_chart, use_container_width=True)
                st.stop()  # Stop here if not showing advanced chart
        
        # Create advanced chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price Action with SMC Levels', 'RSI', 'Volume')
        )
        
        # Candlestick - ensure all columns are numeric
        try:
            # Clean index - ensure it's datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
            
            # Ensure all OHLC values are numeric
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN in critical columns
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if df.empty:
                st.error("❌ No valid price data found after cleaning. Please check your data source.")
                st.stop()
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index, 
                    open=df["Open"], 
                    high=df["High"], 
                    low=df["Low"], 
                    close=df["Close"],
                    name="Price",
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ), row=1, col=1
            )
        except Exception as e:
            st.error(f"❌ Error creating candlestick chart: {str(e)}")
            # Fallback to line chart
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name="Price",
                    line=dict(color="#00f5ff", width=2)
                ), row=1, col=1
            )
        
        # SMC Levels from comprehensive analysis
        current_price = df['Close'].iloc[-1]
        
        # Add SMC levels with error handling
        try:
            # Support Levels
            if smc_data.get('nearest_support'):
                main_support = smc_data['nearest_support']
                if isinstance(main_support, (int, float)) and not np.isnan(main_support):
                    fig.add_hline(y=float(main_support), line_dash="dash", line_color="green", 
                                 annotation_text=f"SMC Support: {float(main_support):.5f}", 
                                 annotation_position="right", row=1, col=1, opacity=0.7, line_width=2)
            
            # Show additional support levels
            try:
                for sup in smc_data.get('support_resistance', {}).get('support', [])[:2]:
                    if isinstance(sup, dict) and 'price' in sup:
                        sup_price = sup['price']
                        if isinstance(sup_price, (int, float)) and not np.isnan(sup_price):
                            if sup_price != smc_data.get('nearest_support'):
                                fig.add_hline(y=float(sup_price), line_dash="dot", line_color="lightgreen", 
                                             row=1, col=1, opacity=0.4)
            except:
                pass
            
            # Resistance Levels
            if smc_data.get('nearest_resistance'):
                main_resistance = smc_data['nearest_resistance']
                if isinstance(main_resistance, (int, float)) and not np.isnan(main_resistance):
                    fig.add_hline(y=float(main_resistance), line_dash="dash", line_color="red", 
                                 annotation_text=f"SMC Resistance: {float(main_resistance):.5f}", 
                                 annotation_position="right", row=1, col=1, opacity=0.7, line_width=2)
            
            # Show additional resistance levels
            try:
                for res in smc_data.get('support_resistance', {}).get('resistance', [])[:2]:
                    if isinstance(res, dict) and 'price' in res:
                        res_price = res['price']
                        if isinstance(res_price, (int, float)) and not np.isnan(res_price):
                            if res_price != smc_data.get('nearest_resistance'):
                                fig.add_hline(y=float(res_price), line_dash="dot", line_color="lightcoral", 
                                             row=1, col=1, opacity=0.4)
            except:
                pass
            
            # Fibonacci Levels
            try:
                fib_levels = smc_data.get('fibonacci', {}).get('levels', {})
                if fib_levels:
                    # Show key Fibonacci levels
                    for level_name, level_price in [('0.618', fib_levels.get('0.618')), 
                                                     ('0.5', fib_levels.get('0.5')), 
                                                     ('0.382', fib_levels.get('0.382'))]:
                        if level_price and isinstance(level_price, (int, float)) and not np.isnan(level_price):
                            fig.add_hline(y=float(level_price), line_dash="dot", line_color="gold", 
                                         annotation_text=f"Fib {level_name}: {float(level_price):.5f}", 
                                         annotation_position="left", row=1, col=1, opacity=0.5)
            except:
                pass
            
            # Trend Lines from SMC analysis
            try:
                trend_info = smc_data.get('trend', {})
                if trend_info and 'current_trend_line' in trend_info:
                    recent_df = df.tail(30)
                    trend_line_values = trend_info['current_trend_line']
                    
                    if isinstance(trend_line_values, (list, np.ndarray)) and len(trend_line_values) == len(recent_df):
                        fig.add_trace(
                            go.Scatter(
                                x=recent_df.index,
                                y=trend_line_values,
                                mode='lines',
                                name=f"Trend Line ({trend_info.get('trend_direction', 'UNKNOWN')})",
                                line=dict(color='yellow', width=2, dash='dash'),
                                opacity=0.8
                            ),
                            row=1, col=1
                        )
                        
                        # Show trend strength
                        if trend_info.get('strength', 0) > 0.7:
                            st.info(f"📈 Strong {trend_info.get('trend_direction', 'trend')} detected (Strength: {trend_info.get('strength', 0)*100:.0f}%)")
            except Exception as e:
                pass  # Trend line optional
        except Exception as e:
            st.warning(f"⚠️ Some SMC levels may not be displayed: {str(e)}")
        
        # Entry, Stop Loss, Take Profit from AI
        try:
            entry = entry_levels.get('entry', current_price)
            stop_loss = entry_levels.get('stop_loss', current_price * 0.99)
            take_profit = entry_levels.get('take_profit', current_price * 1.01)
            
            if isinstance(entry, (int, float)) and not np.isnan(entry):
                fig.add_hline(y=float(entry), line_dash="dash", line_color="blue", 
                             annotation_text=f"AI Entry: {float(entry):.5f}", 
                             annotation_position="right", row=1, col=1, opacity=0.8)
            if isinstance(stop_loss, (int, float)) and not np.isnan(stop_loss):
                fig.add_hline(y=float(stop_loss), line_dash="dash", line_color="red", 
                             annotation_text=f"AI Stop Loss: {float(stop_loss):.5f}", 
                             annotation_position="right", row=1, col=1, opacity=0.8)
            if isinstance(take_profit, (int, float)) and not np.isnan(take_profit):
                fig.add_hline(y=float(take_profit), line_dash="dash", line_color="green", 
                             annotation_text=f"AI Take Profit: {float(take_profit):.5f}", 
                             annotation_position="right", row=1, col=1, opacity=0.8)
        except:
            pass  # AI levels optional
        
        # Moving Averages
        try:
            if 'SMA_20' in df.columns and df['SMA_20'].notna().any():
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["SMA_20"], mode="lines",
                              name="SMA 20", line=dict(color="cyan", width=1.5)),
                    row=1, col=1
                )
            if 'SMA_50' in df.columns and df['SMA_50'].notna().any():
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["SMA_50"], mode="lines",
                              name="SMA 50", line=dict(color="orange", width=1.5)),
                    row=1, col=1
                )
        except:
            pass  # MAs optional
        
        # RSI
        try:
            if "RSI" in df.columns and df["RSI"].notna().any():
                fig.add_trace(
                    go.Scatter(x=df.index, y=df["RSI"], mode="lines",
                              name="RSI", line=dict(color="purple", width=1.5)),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, opacity=0.5)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, opacity=0.5)
        except:
            pass  # RSI optional
        
        # Volume
        try:
            if "Volume" in df.columns and df["Volume"].sum() > 0:
                colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                          for i in range(min(len(df), len(df)))]
                fig.add_trace(
                    go.Bar(x=df.index, y=df["Volume"], name="Volume",
                           marker_color=colors, opacity=0.6),
                    row=3, col=1
                )
        except:
            pass  # Volume optional
        
        # Enable drawing tools and annotations
        try:
            fig.update_layout(
                template="plotly_dark",
                height=900,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
                # Enable drawing tools
                dragmode='pan',  # Default to pan, but user can switch
                modebar_add=[
                    'drawline',
                    'drawopenpath',
                    'drawclosedpath',
                    'drawcircle',
                    'drawrect',
                    'eraseshape'
                ],
                # Enable annotations
                xaxis=dict(
                    type='date',
                    rangeslider=dict(visible=False)
                ),
                # Add modebar buttons for drawing
                modebar_orientation='v'
            )
        except Exception as e:
            # Fallback layout without drawing tools
            fig.update_layout(
                template="plotly_dark",
                height=900,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            st.warning(f"⚠️ Some chart features may not be available: {str(e)}")
        
        # Add custom CSS for better drawing tools visibility
        st.markdown("""
    <style>
        .stPlotlyChart {
            background-color: #1e1e1e;
        }
        .modebar-group {
            background-color: rgba(255, 255, 255, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
        
        # Display chart with config for drawing tools
        try:
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'modeBarButtonsToAdd': [
                        'drawline',
                        'drawopenpath',
                        'drawclosedpath',
                        'drawcircle',
                        'drawrect',
                        'eraseshape'
                    ],
                    'modeBarButtonsToRemove': [],
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'{selected_symbol}_chart',
                        'height': 900,
                        'width': 1400,
                        'scale': 1
                    },
                    'doubleClick': 'reset',
                    'editable': True,  # Enable editing for drawing
                    'edits': {
                        'shapePosition': True,
                        'annotationPosition': True,
                        'annotationTail': True,
                        'annotationText': True
                    }
                }
            )
        except Exception as e:
            # Fallback: simple chart display
            st.plotly_chart(fig, use_container_width=True)
            st.warning(f"⚠️ Chart displayed with limited features: {str(e)}")

    except Exception as e:
        import traceback
        st.error(f"❌ Error creating Plotly chart: {str(e)}")
        st.code(traceback.format_exc())
        st.info("💡 Please try refreshing the page or selecting a different symbol.")
        
        # Try to show a simple chart as fallback
        try:
            st.info("🔄 Attempting to display basic chart...")
            if not df.empty and 'Close' in df.columns:
                simple_fig = go.Figure()
                simple_fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Price'
                ))
                simple_fig.update_layout(
                    title=f"Basic Price Chart - {selected_symbol}",
                    template="plotly_dark",
                    height=600
                )
                st.plotly_chart(simple_fig, use_container_width=True)
        except Exception as e2:
            st.error(f"❌ Even basic chart failed: {str(e2)}")

# TradingView Chart Section
if chart_type in ["TradingView Chart", "Both"]:
    st.markdown("---")
    st.subheader("📈 TradingView Advanced Chart")
    
    if chart_type == "Both":
        st.info("💡 **Below is your TradingView chart. You can draw directly on it and your drawings will be saved to your TradingView account.**")
    
    # TradingView chart container
    tv_chart_html = get_tradingview_widget(
        symbol=tradingview_symbol,
        theme="dark",
        height=700,
        show_drawings=show_tv_drawings,
        username=tradingview_username if tradingview_username else "",
        interval=selected_timeframe
    )
    
    st.markdown(tv_chart_html, unsafe_allow_html=True)
    
    # TradingView Account Connection Info
    if not tradingview_username:
        st.info("""
        🔗 **Connect Your TradingView Account:**
        - Enter your TradingView username in the sidebar to sync your charts
        - Your drawings and saved analyses will appear when enabled
        - You can draw directly on the TradingView chart above
        - All drawings are automatically saved to your TradingView account
        """)
    else:
        st.success(f"✅ Connected to TradingView account: **{tradingview_username}**")
        if show_tv_drawings:
            st.info("📝 Your saved drawings from TradingView are now visible on the chart above. Draw new lines, shapes, or annotations and they'll be saved to your account.")
        else:
            st.info("💡 Enable 'Show My TradingView Drawings' in the sidebar to display your saved analyses.")
    
    # TradingView Features Info
    with st.expander("📚 TradingView Chart Features"):
        st.markdown("""
        **TradingView Chart Capabilities:**
        - ✅ Real-time price updates
        - ✅ Professional drawing tools (trend lines, shapes, annotations)
        - ✅ 100+ technical indicators
        - ✅ Multiple chart types (Candles, Line, Area, etc.)
        - ✅ Customizable timeframes
        - ✅ Save drawings to your TradingView account
        - ✅ Access your saved charts from any device
        
        **How to Use:**
        1. Click the drawing tools icon in the chart toolbar
        2. Draw trend lines, support/resistance, or any shapes
        3. Your drawings are automatically saved to your TradingView account
        4. Enable "Show My TradingView Drawings" to see your saved analyses
        5. Switch between timeframes using the buttons at the top
        
        **Note:** To fully sync your drawings, make sure you're logged into TradingView in your browser.
        """)

# Drawing mode selector (only for Plotly charts)
if chart_type in ["Plotly Chart", "Both"]:
    st.markdown("---")
    st.info("💡 **Tip**: Use the chart toolbar above to draw trend lines, shapes, and annotations. Click the drawing icons in the toolbar.")

# User Drawing Interface
st.markdown("---")
st.subheader("✏️ Your Analysis Drawing Tools")

col_draw1, col_draw2, col_draw3 = st.columns(3)

with col_draw1:
    user_entry = st.number_input(
        "📥 Your Entry Price",
        value=float(current_price),
        min_value=0.0,
        step=0.00001,
        format="%.5f"
    )

with col_draw2:
    user_stop_loss = st.number_input(
        "🛑 Your Stop Loss",
        value=float(current_price * 0.99),
        min_value=0.0,
        step=0.00001,
        format="%.5f"
    )

with col_draw3:
    user_take_profit = st.number_input(
        "🎯 Your Take Profit",
        value=float(current_price * 1.01),
        min_value=0.0,
        step=0.00001,
        format="%.5f"
    )

user_action = st.radio("Your Action", ["BUY", "SELL", "HOLD"], horizontal=True)
user_confidence = st.slider("Your Confidence %", 0, 100, 50)

# Comparison Analysis - Always show results
st.markdown("---")
st.subheader("🤖 AI vs Your Analysis Comparison")

# Always run comparison automatically
# Calculate differences
entry_diff = abs(user_entry - entry) / entry * 100 if entry > 0 else 0
sl_diff = abs(user_stop_loss - stop_loss) / stop_loss * 100 if stop_loss > 0 else 0
tp_diff = abs(user_take_profit - take_profit) / take_profit * 100 if take_profit > 0 else 0

# Action comparison
ai_action = trading_recommendation.get('action', 'HOLD')
action_match = (user_action == ai_action)

# Calculate overall similarity
avg_diff = (entry_diff + sl_diff + tp_diff) / 3
similarity_score = 100 - avg_diff

# Risk-Reward comparison
user_risk = abs(user_entry - user_stop_loss)
user_reward = abs(user_take_profit - user_entry)
user_rr = user_reward / user_risk if user_risk > 0 else 0

ai_risk = abs(entry - stop_loss)
ai_reward = abs(take_profit - entry)
ai_rr = entry_levels.get('risk_reward_ratio', 1.0)

# Display comparison
col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    st.markdown(f"""
    <div class="comparison-box">
        <h3>🤖 AI Analysis</h3>
        <p><strong>Action:</strong> {ai_action}</p>
        <p><strong>Entry:</strong> {entry:.5f}</p>
        <p><strong>Stop Loss:</strong> {stop_loss:.5f}</p>
        <p><strong>Take Profit:</strong> {take_profit:.5f}</p>
        <p><strong>Risk:Reward:</strong> {ai_rr:.2f}:1</p>
        <p><strong>Confidence:</strong> {trading_recommendation.get('confidence', 0.5)*100:.0f}%</p>
        <p><strong>Technical Strength:</strong> {strength_score:.0f}/100</p>
    </div>
    """, unsafe_allow_html=True)

with col_comp2:
    st.markdown(f"""
    <div class="comparison-box">
        <h3>✏️ Your Analysis</h3>
        <p><strong>Action:</strong> {user_action}</p>
        <p><strong>Entry:</strong> {user_entry:.5f}</p>
        <p><strong>Stop Loss:</strong> {user_stop_loss:.5f}</p>
        <p><strong>Take Profit:</strong> {user_take_profit:.5f}</p>
        <p><strong>Risk:Reward:</strong> {user_rr:.2f}:1</p>
        <p><strong>Confidence:</strong> {user_confidence}%</p>
    </div>
    """, unsafe_allow_html=True)

# Differences
st.markdown("### 📊 Differences")
col_diff1, col_diff2, col_diff3 = st.columns(3)

with col_diff1:
    st.metric("Entry Difference", f"{entry_diff:.2f}%", 
             delta=f"{'Similar' if entry_diff < 1 else 'Different'}")
with col_diff2:
    st.metric("Stop Loss Difference", f"{sl_diff:.2f}%",
             delta=f"{'Similar' if sl_diff < 1 else 'Different'}")
with col_diff3:
    st.metric("Take Profit Difference", f"{tp_diff:.2f}%",
             delta=f"{'Similar' if tp_diff < 1 else 'Different'}")

# Overall Analysis using SMC and AI/LLM logic
st.markdown("### 🧠 AI Evaluation (Using ML Model + SMC + News)")

# Use comprehensive SMC comparison
comparison_result = compare_user_vs_ai_smc(
    user_entry, user_stop_loss, user_take_profit,
    entry, stop_loss, take_profit,
    smc_data, news_sentiment, trading_recommendation.get('confidence', 0.5)
)

total_score = comparison_result['total_score']
max_score = 100
scores = comparison_result['scores']

# Display detailed scores
factors = []

# Action match
if action_match:
    factors.append(f"✅ Action matches AI recommendation ({scores.get('entry_score', 0) + 10}/35)")
else:
    factors.append(f"❌ Action differs (AI: {ai_action}, You: {user_action})")

factors.append(f"📥 Entry Price Score: {scores.get('entry_score', 0)}/30")
factors.append(f"🛑 Stop Loss Score: {scores.get('stop_loss_score', 0)}/25")
factors.append(f"🎯 Take Profit Score: {scores.get('take_profit_score', 0)}/25")
factors.append(f"⚖️ Risk-Reward Score: {scores.get('risk_reward_score', 0)}/20")

# Display evaluation
for factor in factors:
    st.write(factor)

st.markdown("---")

# Overall recommendation from SMC comparison
winner = comparison_result['winner']
recommendation_ai = comparison_result['recommendation']

if total_score >= 80:
    accuracy_class = "accuracy-high"
    recommendation = "🎯 EXCELLENT! Your analysis closely matches AI recommendations."
elif total_score >= 60:
    accuracy_class = "accuracy-medium"
    recommendation = "✅ GOOD! Your analysis is reasonably aligned with AI."
else:
    accuracy_class = "accuracy-low"
    recommendation = "⚠️ Your analysis differs significantly from AI recommendations."

# Add SMC insights
market_structure = smc_data.get('market_structure', {}).get('structure', 'UNKNOWN')
if market_structure != 'UNKNOWN':
    factors.append(f"📊 Market Structure: {market_structure}")

nearest_fib = smc_data.get('fibonacci', {}).get('nearest_level', {})
if nearest_fib:
    factors.append(f"📐 Nearest Fibonacci Level: {nearest_fib.get('level', 'N/A')} at {nearest_fib.get('price', 0):.5f}")

similarity_pct = comparison_result.get('similarity_pct', total_score)

st.markdown(f"""
<div class="comparison-box">
    <h2>📊 Overall Score: <span class="{accuracy_class}">{total_score}/{max_score} ({similarity_pct:.1f}% Similarity)</span></h2>
    <h3>{recommendation}</h3>
    <h4>🏆 Best Choice: {winner}</h4>
    <p><strong>AI Analysis Based On:</strong></p>
    <ul>
        <li>ML Model Prediction: {ml_prediction.get('direction', 'NEUTRAL')} ({ml_prediction.get('confidence', 0.5)*100:.0f}% confidence)</li>
        <li>Technical Analysis: Strength {strength_score}/100</li>
        <li>SMC Levels: Support {smc_data.get('nearest_support', 'N/A')}, Resistance {smc_data.get('nearest_resistance', 'N/A')}</li>
        <li>Market Structure: {market_structure}</li>
        <li>News Sentiment: {news_sentiment.get('compound', 0):+.2f} ({news_sentiment.get('count', 0)} articles)</li>
    </ul>
    <p><strong>Recommendation:</strong> {trading_recommendation.get('reason', 'Multiple factors considered')}</p>
</div>
""", unsafe_allow_html=True)

# Show percentage difference
st.markdown("### 📈 Percentage Difference Analysis")
col_pct1, col_pct2, col_pct3 = st.columns(3)
with col_pct1:
    st.metric("Entry Difference", f"{entry_diff:.2f}%", 
             delta=f"{'Very Close' if entry_diff < 1 else 'Moderate' if entry_diff < 3 else 'Different'}")
with col_pct2:
    st.metric("Stop Loss Difference", f"{sl_diff:.2f}%",
             delta=f"{'Very Close' if sl_diff < 1 else 'Moderate' if sl_diff < 3 else 'Different'}")
with col_pct3:
    st.metric("Take Profit Difference", f"{tp_diff:.2f}%",
             delta=f"{'Very Close' if tp_diff < 1 else 'Moderate' if tp_diff < 3 else 'Different'}")

avg_difference = (entry_diff + sl_diff + tp_diff) / 3
st.metric("Average Difference", f"{avg_difference:.2f}%", 
         delta=f"{'Excellent Match' if avg_difference < 1 else 'Good Match' if avg_difference < 2 else 'Needs Improvement'}")

# SMC Explanation
with st.expander("📚 About Smart Money Concepts (SMC)"):
    st.markdown("""
    **Smart Money Concepts include:**
    - **Support/Resistance Levels:** Key price levels where price has reversed
    - **Trend Lines:** Linear regression showing price direction
    - **Entry/Exit Zones:** Optimal levels for entering/exiting trades
    
    **How to use:**
    1. Identify support (green line) and resistance (red line) levels
    2. Draw trend lines following price action
    3. Compare your levels with AI predictions above
    4. Use the comparison tool to evaluate your analysis
    """)

st.markdown("---")
st.info("💡 **Tip:** Draw your analysis based on the chart, then compare with AI predictions to improve your trading skills!")

