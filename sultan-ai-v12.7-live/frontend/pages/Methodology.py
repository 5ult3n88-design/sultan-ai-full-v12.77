import streamlit as st
from datetime import datetime

st.set_page_config(page_title="📊 Methodology", layout="wide", initial_sidebar_state="expanded")

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
    
    .method-box {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .method-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00f5ff;
        margin-bottom: 15px;
    }
    
    .code-block {
        background: rgba(0, 0, 0, 0.3);
        color: #f8f8f2;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
    <h1 class="dashboard-title">📊 Trading Methodology & Analysis Framework</h1>
    <p class="dashboard-subtitle">Comprehensive guide to our AI-powered trading system</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="method-box">
    <div class="method-title">🎯 Overview</div>
    <p>Our trading system combines Machine Learning predictions, Technical Analysis, and News Sentiment Analysis 
    to provide comprehensive trading recommendations with precise entry, stop loss, and take profit levels.</p>
</div>
""", unsafe_allow_html=True)

# Methodology Sections
with st.expander("🤖 Machine Learning Model", expanded=True):
    st.markdown("""
    ### Model Architecture
    
    Our ML model uses a **Random Forest Classifier** combined with **Gradient Boosting Regressor**:
    
    **Features Used:**
    - Price data (Open, High, Low, Close)
    - Technical indicators (RSI, MACD, SMA, Bollinger Bands)
    - Volatility metrics (ATR, rolling standard deviation)
    - Momentum indicators
    - Volume analysis (when available)
    - Price momentum over multiple timeframes
    
    **Prediction Process:**
    1. Extract 100+ technical features from historical data
    2. Train Random Forest for direction prediction (UP/DOWN/NEUTRAL)
    3. Use Gradient Boosting for precise price change prediction
    4. Combine predictions with confidence scores
    
    **Model Retraining:**
    - Model automatically retrains when new data is available
    - Uses latest 100 periods for training
    - Saves trained model for future use
    """)

with st.expander("📈 Technical Analysis Methods"):
    st.markdown("""
    ### Indicators Used
    
    **1. Relative Strength Index (RSI)**
    - Measures momentum and overbought/oversold conditions
    - Values above 70 indicate overbought (potential sell)
    - Values below 30 indicate oversold (potential buy)
    - Weight: High
    
    **2. Moving Average Convergence Divergence (MACD)**
    - Shows relationship between two moving averages
    - Bullish: MACD line crosses above signal line
    - Bearish: MACD line crosses below signal line
    - Weight: High
    
    **3. Moving Averages (SMA 20 & SMA 50)**
    - Trend identification
    - Golden Cross: SMA 20 > SMA 50 (Bullish)
    - Death Cross: SMA 20 < SMA 50 (Bearish)
    - Weight: Medium-High
    
    **4. Bollinger Bands**
    - Volatility and price position
    - Price near lower band: Potential bounce (Buy)
    - Price near upper band: Potential reversal (Sell)
    - Weight: Medium
    
    **5. Stochastic Oscillator**
    - Momentum indicator
    - Overbought: > 80
    - Oversold: < 20
    - Weight: Medium
    
    **6. Average True Range (ATR)**
    - Volatility measurement
    - Used for calculating stop loss and take profit levels
    - Weight: High (for risk management)
    
    ### Technical Strength Score (0-100)
    
    Calculated by combining:
    - RSI position and momentum
    - MACD crossover signals
    - Moving average trends
    - Bollinger Band position
    - Stochastic readings
    - Price momentum
    
    **Interpretation:**
    - 70-100: Strong bullish signals
    - 30-70: Neutral/mixed signals
    - 0-30: Strong bearish signals
    """)

with st.expander("📰 News Sentiment Analysis"):
    st.markdown("""
    ### Sentiment Processing
    
    **Data Sources:**
    - Reuters Financial News RSS feeds
    - Yahoo Finance News API
    - Bloomberg Market News
    - CNN Money Latest
    - Multiple forex/finance news sources
    
    **Analysis Method:**
    - **VADER Sentiment**: Specialized for social media and financial text
    - **TextBlob Sentiment**: Additional validation
    - Combines title and summary for comprehensive analysis
    
    **Sentiment Scores:**
    - Compound score: -1 (very negative) to +1 (very positive)
    - Positive/Negative/Neutral classification
    - Weight in decision: 30%
    
    **Impact on Trading:**
    - Positive sentiment + Technical signals = Strong BUY
    - Negative sentiment + Technical signals = Strong SELL
    - Conflicting signals = HOLD recommendation
    """)

with st.expander("🎯 Entry, Stop Loss & Take Profit Calculation"):
    st.markdown("""
    ### Level Calculation Methodology
    
    **Entry Price:**
    - **BUY Signal**: Current price or slightly above recent resistance
    - **SELL Signal**: Current price or slightly below recent support
    - Entry executes at market price when signal is generated
    
    **Stop Loss Calculation:**
    Uses two methods and takes the more conservative:
    
    1. **ATR-Based Stop Loss:**
       - Stop Loss = Entry ± (2 × ATR)
       - Accounts for normal volatility
       - Prevents being stopped out by noise
    
    2. **Support/Resistance-Based:**
       - BUY: Below recent 20-period low
       - SELL: Above recent 20-period high
       - Respects key price levels
    
    **Take Profit Calculation:**
    - Based on **Risk-Reward Ratio**: Minimum 2:1, Target 2.5:1
    - Risk = |Entry - Stop Loss|
    - Take Profit = Entry ± (Risk × 2.5)
    
    **Example (BUY Signal):**
    - Entry: 2000.00
    - Stop Loss: 1990.00 (Risk = 10 points)
    - Take Profit: 2025.00 (Reward = 25 points, Ratio = 2.5:1)
    
    **Safety Limits:**
    - Maximum stop loss: 3% of entry price
    - Maximum take profit: 5% of entry price
    - Prevents unrealistic targets
    """)

with st.expander("🔬 Combined Analysis Framework"):
    st.markdown("""
    ### Decision Making Process
    
    **Step 1: Technical Analysis (70% weight)**
    - Calculate all technical indicators
    - Compute Technical Strength Score
    - Identify trend direction
    
    **Step 2: ML Prediction (25% weight)**
    - Run ML model with latest features
    - Get direction prediction (UP/DOWN/NEUTRAL)
    - Get predicted price change percentage
    - Get model confidence score
    
    **Step 3: News Sentiment (5% weight)**
    - Fetch relevant news articles
    - Analyze sentiment (VADER + TextBlob)
    - Classify as positive/negative/neutral
    
    **Step 4: Signal Generation**
    
    **BUY Signal Generated When:**
    - ML predicts UP direction (confidence > 60%)
    - Technical Strength Score > 60
    - OR: Strong ML signal (confidence > 70%) even with mixed technicals
    
    **SELL Signal Generated When:**
    - ML predicts DOWN direction (confidence > 60%)
    - Technical Strength Score < 40
    - OR: Strong ML signal (confidence > 70%) even with mixed technicals
    
    **HOLD Signal Generated When:**
    - ML direction is NEUTRAL
    - Technical signals are mixed (40-60)
    - Conflicting ML and technical signals
    
    **Step 5: Confidence Calculation**
    - Weighted average of:
      - ML confidence: 50%
      - Technical strength: 30%
      - News sentiment alignment: 20%
    
    **Step 6: Risk Assessment**
    - Calculate annualized volatility
    - High volatility (>30%): High Risk
    - Medium volatility (15-30%): Medium Risk
    - Low volatility (<15%): Low Risk
    """)

with st.expander("⚙️ Best Practices & Tips"):
    st.markdown("""
    ### Trading Recommendations
    
    **1. Always Use Stop Loss:**
    - Never enter a trade without a stop loss
    - Our calculated stop loss accounts for volatility
    - Adjust if you have different risk tolerance
    
    **2. Risk Management:**
    - Risk only 1-2% of your account per trade
    - Calculate position size: (Account × Risk%) / (Entry - Stop Loss)
    - Never risk more than you can afford to lose
    
    **3. Entry Execution:**
    - Enter at calculated entry price or better
    - Use limit orders for better fills
    - Avoid market orders during high volatility
    
    **4. Take Profit Strategies:**
    - **Conservative**: Take 50% profit at first TP, let rest run
    - **Aggressive**: Hold for full TP target
    - **Scalping**: Take quick profits at 1:1 risk-reward
    
    **5. Signal Confidence:**
    - **High Confidence (>75%)**: Larger position size
    - **Medium Confidence (50-75%)**: Normal position size
    - **Low Confidence (<50%)**: Smaller position or skip trade
    
    **6. Timeframes:**
    - Our analysis uses 30-minute candles
    - Best for swing trading (hours to days)
    - Not suitable for scalping (seconds/minutes)
    
    **7. Market Conditions:**
    - Works best in trending markets
    - Less reliable in choppy/sideways markets
    - Check news for major events before trading
    
    **8. Continuous Monitoring:**
    - Re-evaluate positions as new data arrives
    - Update stop loss if price moves favorably
    - Exit early if fundamentals change
    """)

with st.expander("📚 Additional Resources"):
    st.markdown("""
    ### Learning Resources
    
    - **Technical Analysis**: Study RSI, MACD, Moving Averages
    - **Risk Management**: Position sizing, stop loss strategies
    - **Market Psychology**: Understanding market sentiment
    - **Backtesting**: Test strategies on historical data
    
    ### Disclaimer
    
    **⚠️ Important Notice:**
    
    This trading system is for **educational purposes** only. Trading forex and stocks involves substantial risk of loss.
    
    - Past performance does not guarantee future results
    - Always do your own research
    - Never trade with money you cannot afford to lose
    - Consider consulting a financial advisor
    - Markets can move against predictions
    - Use proper risk management always
    
    **No Guarantees:**
    - Signals are predictions, not certainties
    - Market conditions can change rapidly
    - External factors (news, events) can override predictions
    - Always use stop losses
    """)

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #00f5ff; font-weight: bold;">
    📊 Built with Advanced AI & Machine Learning | Updated: {datetime.now().strftime("%Y-%m-%d")}
</div>
""", unsafe_allow_html=True)




