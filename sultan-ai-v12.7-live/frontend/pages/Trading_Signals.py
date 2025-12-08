import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import yfinance as yf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
from analytics import calculate_advanced_indicators, calculate_strength_score
from data_validator import clean_dataframe_for_analysis
from ml_model import predict_price_movement, calculate_entry_exit_levels, get_trading_recommendation
from fetch_news import get_news_for_symbol
from analytics import analyze_news_sentiment
from data_loader import load_csv_robust, clean_dataframe

st.set_page_config(page_title="🎯 Trading Signals", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# 🎨 UNIQUE MODERN DESIGN (Master Dashboard Style)
# ============================================================
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
    
    /* Signal cards */
    .signal-buy-card {
        background: rgba(0, 255, 136, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);
        color: #ffffff;
    }
    
    .signal-sell-card {
        background: rgba(255, 51, 102, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 51, 102, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(255, 51, 102, 0.2);
        color: #ffffff;
    }
    
    .signal-hold-card {
        background: rgba(255, 170, 0, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 170, 0, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(255, 170, 0, 0.2);
        color: #ffffff;
    }
    
    .level-box {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Custom components */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stCheckbox label {
        color: #ffffff !important;
    }
    
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
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    p {
        color: rgba(255, 255, 255, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🎨 HEADER WITH NAVIGATION
# ============================================================
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
    <h1 class="dashboard-title">🎯 Trading Signals & Entry Recommendations</h1>
    <p class="dashboard-subtitle">Get precise entry, stop loss, and take profit levels for all major forex pairs</p>
</div>
""", unsafe_allow_html=True)

# Forex pairs to analyze
forex_pairs = {
    "XAUUSD": {"name": "Gold (XAU/USD)", "ticker": "GC=F", "icon": "🥇"},
    "EURUSD": {"name": "Euro/US Dollar", "ticker": "EURUSD=X", "icon": "💶"},
    "GBPUSD": {"name": "British Pound/US Dollar", "ticker": "GBPUSD=X", "icon": "💷"},
    "USDJPY": {"name": "US Dollar/Japanese Yen", "ticker": "JPY=X", "icon": "💴"},
    "AUDUSD": {"name": "Australian Dollar/US Dollar", "ticker": "AUDUSD=X", "icon": "💰"},
    "USDCAD": {"name": "US Dollar/Canadian Dollar", "ticker": "CAD=X", "icon": "💵"},
}

# Data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(project_root, "data")

# Filter to only show pairs with data
available_pairs = {}
for pair, info in forex_pairs.items():
    csv_path = os.path.join(data_dir, f"{pair}.csv")
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
        # Verify data is loadable
        try:
            test_df = load_csv_robust(csv_path)
            if len(test_df) > 10:
                available_pairs[pair] = info
        except:
            pass

if not available_pairs:
    st.error("⚠️ No forex data found. Please run fetch_data.py first.")
    st.stop()

# Quick selection
st.markdown("### 📊 Quick Select Pair")
selected_pairs = st.multiselect(
    "Choose pairs to analyze (leave empty for all):",
    list(available_pairs.keys()),
    default=list(available_pairs.keys())[:3] if len(available_pairs) >= 3 else list(available_pairs.keys())
)

pairs_to_show = selected_pairs if selected_pairs else list(available_pairs.keys())

# Analyze each pair
st.markdown("---")

for pair in pairs_to_show:
    if pair not in available_pairs:
        continue
    
    info = available_pairs[pair]
    pair_name = info["name"]
    ticker = info["ticker"]
    icon = info["icon"]
    
    csv_path = os.path.join(data_dir, f"{pair}.csv")
    
    with st.expander(f"{icon} {pair} - {pair_name}", expanded=True):
        try:
            # Load data using robust loader
            try:
                df = load_csv_robust(csv_path)
                
                if df.empty or len(df) < 10:
                    continue  # Skip this pair
            except Exception as e:
                continue  # Skip this pair if error
            
            # Fetch latest data
            try:
                latest_df = yf.download(ticker, period="5d", interval="30m", progress=False)
                if not latest_df.empty:
                    df = latest_df
            except:
                pass
            
            # Clean data before analysis
            try:
                df = clean_dataframe_for_analysis(df)
            except Exception as e:
                st.error(f"Error cleaning data for {pair}: {str(e)}")
                continue
            
            # Calculate indicators with better error handling
            try:
                df = calculate_advanced_indicators(df)
                if df.empty or len(df) < 50:
                    st.warning(f"Insufficient data for {pair} (only {len(df)} rows)")
                    continue
            except Exception as e:
                import traceback
                st.error(f"Error calculating indicators for {pair}: {str(e)}")
                st.code(traceback.format_exc())
                continue
            
            try:
                strength_score = calculate_strength_score(df)
                current_price = float(df['Close'].iloc[-1])
            except Exception as e:
                st.error(f"Error calculating strength score for {pair}: {str(e)}")
                continue
            
            # ML Prediction
            try:
                ml_prediction = predict_price_movement(df)
                trading_recommendation = get_trading_recommendation(df, ml_prediction)
                entry_levels = calculate_entry_exit_levels(df, trading_recommendation)
            except Exception as e:
                st.warning(f"ML analysis unavailable: {e}")
                trading_recommendation = {
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reason': 'Analysis in progress',
                    'risk_level': 'Medium'
                }
                entry_levels = {
                    'entry': current_price,
                    'stop_loss': current_price * 0.99,
                    'take_profit': current_price * 1.01,
                    'risk_reward_ratio': 1.0
                }
            
            action = trading_recommendation.get('action', 'HOLD')
            confidence = trading_recommendation.get('confidence', 0.5) * 100
            
            # Determine card style
            if action == 'BUY':
                card_class = 'signal-buy-card'
                action_emoji = "🟢"
            elif action == 'SELL':
                card_class = 'signal-sell-card'
                action_emoji = "🔴"
            else:
                card_class = 'signal-hold-card'
                action_emoji = "🟡"
            
            # Main signal card
            entry = entry_levels.get('entry', current_price)
            stop_loss = entry_levels.get('stop_loss', current_price * 0.99)
            take_profit = entry_levels.get('take_profit', current_price * 1.01)
            risk_reward = entry_levels.get('risk_reward_ratio', 1.0)
            
            # Price change
            change_pct = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
            
            st.markdown(f"""
            <div class="{card_class}">
                <h2>{action_emoji} {action} Signal for {pair}</h2>
                <p><strong>Current Price:</strong> {current_price:.5f} ({change_pct:+.2f}%)</p>
                <p><strong>Confidence:</strong> {confidence:.0f}% | <strong>Risk Level:</strong> {trading_recommendation.get('risk_level', 'Medium')}</p>
                <p><strong>Reason:</strong> {trading_recommendation.get('reason', 'Analysis complete')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Trading levels in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="level-box" style="background: rgba(102, 126, 234, 0.2);">
                    <h3 style="color: white; margin: 0;">📥 Entry Price</h3>
                    <h2 style="color: white; margin: 10px 0;">{entry:.5f}</h2>
                    <p style="color: rgba(255,255,255,0.8);">Enter market at this price</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                sl_pct = ((stop_loss - entry) / entry) * 100 if entry > 0 else 0
                st.markdown(f"""
                <div class="level-box" style="background: rgba(220, 53, 69, 0.2);">
                    <h3 style="color: white; margin: 0;">🛑 Stop Loss</h3>
                    <h2 style="color: white; margin: 10px 0;">{stop_loss:.5f}</h2>
                    <p style="color: rgba(255,255,255,0.8);">{sl_pct:+.2f}% from entry</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                tp_pct = ((take_profit - entry) / entry) * 100 if entry > 0 else 0
                st.markdown(f"""
                <div class="level-box" style="background: rgba(40, 167, 69, 0.2);">
                    <h3 style="color: white; margin: 0;">🎯 Take Profit</h3>
                    <h2 style="color: white; margin: 10px 0;">{take_profit:.5f}</h2>
                    <p style="color: rgba(255,255,255,0.8);">{tp_pct:+.2f}% from entry</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="level-box" style="background: rgba(255, 193, 7, 0.2);">
                    <h3 style="color: white; margin: 0;">⚖️ Risk:Reward</h3>
                    <h2 style="color: white; margin: 10px 0;">{risk_reward:.2f}:1</h2>
                    <p style="color: rgba(255,255,255,0.8);">Target ratio</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("💪 Technical Strength", f"{strength_score:.0f}/100")
            with col_met2:
                ml_dir = ml_prediction.get('direction', 'NEUTRAL') if 'ml_prediction' in locals() else 'NEUTRAL'
                st.metric("🤖 ML Prediction", ml_dir)
            with col_met3:
                rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                st.metric("📊 RSI", f"{rsi:.2f}")
            
            # Action instructions
            st.markdown("---")
            if action == 'BUY':
                st.success(f"""
                **📋 Action Plan:**
                1. Wait for price to reach **{entry:.5f}** (or enter now at current price {current_price:.5f})
                2. Set stop loss at **{stop_loss:.5f}** ({abs(sl_pct):.2f}% risk)
                3. Set take profit at **{take_profit:.5f}** ({tp_pct:.2f}% gain)
                4. Risk only 1-2% of your account per trade
                5. Monitor the trade and adjust stop loss if price moves favorably
                """)
            elif action == 'SELL':
                st.error(f"""
                **📋 Action Plan:**
                1. Wait for price to reach **{entry:.5f}** (or enter now at current price {current_price:.5f})
                2. Set stop loss at **{stop_loss:.5f}** ({abs(sl_pct):.2f}% risk)
                3. Set take profit at **{take_profit:.5f}** ({tp_pct:.2f}% gain)
                4. Risk only 1-2% of your account per trade
                5. Monitor the trade and adjust stop loss if price moves favorably
                """)
            else:
                st.info(f"""
                **📋 Action Plan:**
                - **HOLD** position - Mixed signals detected
                - Wait for clearer direction before entering
                - Current price: **{current_price:.5f}**
                - Monitor for better entry opportunities
                """)
            
        except Exception as e:
            st.error(f"Error analyzing {pair}: {str(e)}")
            continue
        
        st.markdown("<br>", unsafe_allow_html=True)

# Summary table
st.markdown("---")
st.markdown("### 📊 Summary Table")

summary_data = []
for pair in pairs_to_show:
    if pair not in available_pairs:
        continue
    
    try:
        csv_path = os.path.join(data_dir, f"{pair}.csv")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if df.empty:
            continue
        
        df = calculate_advanced_indicators(df)
        current_price = df['Close'].iloc[-1]
        
        try:
            ml_prediction = predict_price_movement(df)
            trading_recommendation = get_trading_recommendation(df, ml_prediction)
            entry_levels = calculate_entry_exit_levels(df, trading_recommendation)
            action = trading_recommendation.get('action', 'HOLD')
            confidence = trading_recommendation.get('confidence', 0.5) * 100
            entry = entry_levels.get('entry', current_price)
        except:
            action = 'HOLD'
            confidence = 50
            entry = current_price
        
        summary_data.append({
            'Pair': pair,
            'Current Price': f"{current_price:.5f}",
            'Action': action,
            'Entry': f"{entry:.5f}",
            'Confidence': f"{confidence:.0f}%",
            'Strength': f"{calculate_strength_score(df):.0f}/100"
        })
    except:
        continue

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h3>⚠️ Important Notes:</h3>
    <ul>
        <li>These signals are generated using AI/ML predictions and technical analysis</li>
        <li>Always use stop loss and proper risk management</li>
        <li>Signals are predictions, not guarantees</li>
        <li>Monitor market conditions and news before entering trades</li>
        <li>Past performance does not guarantee future results</li>
    </ul>
</div>
""", unsafe_allow_html=True)

