# 📊 Sultan AI Live Trading Dashboard v12.7

A comprehensive, real-time trading analytics dashboard with advanced AI-powered recommendations, news sentiment analysis, and cross-platform support.

## ✨ Features

### 🎯 Core Features
- **Real-time Stock & Forex Data**: Live updates for stocks (AAPL, TSLA, GOOGL, etc.) and forex pairs (EURUSD, XAUUSD, etc.)
- **Advanced Technical Analysis**: 
  - RSI, MACD, Bollinger Bands, Stochastic Oscillator
  - Multiple moving averages (SMA, EMA)
  - ATR for volatility measurement
  - Momentum and ROC indicators

### 📰 News & Sentiment
- **Real-time Financial News**: Automatically fetches relevant news for selected symbols
- **Sentiment Analysis**: AI-powered sentiment analysis using VADER and TextBlob
- **News Sidebar**: Curated news feed with summaries and impact analysis

### 🤖 AI Trading Recommendations
- **Smart Signals**: BUY/SELL/HOLD recommendations based on:
  - Technical indicators (70% weight)
  - News sentiment (30% weight)
- **Risk Assessment**: Automatic risk level calculation
- **Target Prices & Stop Loss**: Suggested entry and exit points

### 💻 Cross-Platform Support
- **macOS**: Use `start.sh`
- **Windows**: Use `start.bat`
- Automatic virtual environment setup

## 🚀 Quick Start

### For macOS/Linux:
```bash
chmod +x start.sh
./start.sh
```

### For Windows:
Double-click `start.bat` or run from command prompt:
```cmd
start.bat
```

The script will:
1. Create a Python virtual environment
2. Install all required dependencies
3. Download historical data for stocks and forex pairs
4. Launch the Streamlit dashboard

## 📦 Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv .venv  # macOS/Linux
python -m venv .venv   # Windows

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Fetch initial data
python backend/fetch_data.py

# Run dashboard
streamlit run frontend/master_dashboard.py
```

## 📊 Available Symbols

### Forex Pairs:
- EURUSD, XAUUSD (Gold), USDJPY, GBPUSD, AUDUSD, USDCAD

### Stocks:
- AAPL (Apple), GOOGL (Google), MSFT (Microsoft)
- TSLA (Tesla), AMZN (Amazon), NVDA (NVIDIA)
- META (Meta/Facebook), NFLX (Netflix)
- SPY (S&P 500 ETF), QQQ (NASDAQ ETF)
- BTC-USD (Bitcoin), ETH-USD (Ethereum)

## 🎯 How to Use

1. **Select a Symbol**: Choose from forex, stocks, or all categories in the sidebar
2. **View Real-time Charts**: Interactive candlestick charts with technical indicators
3. **Check Trading Signal**: AI-powered BUY/SELL/HOLD recommendation with confidence score
4. **Read News**: Latest relevant news in the sidebar with sentiment analysis
5. **Analyze Indicators**: View RSI, MACD, Bollinger Bands, and more

## 🔧 Configuration

### Auto-refresh Settings
- Adjust refresh rate in sidebar (1-30 minutes)
- Enable/disable auto-refresh toggle
- Manual refresh button for news

### Data Refresh
- Historical data: Run `python backend/fetch_data.py`
- News data: Click "Refresh News Now" in sidebar
- Auto-refreshes based on your settings

## 📈 Trading Recommendations Explained

The AI recommendation system combines:

1. **Technical Analysis (70% weight)**:
   - RSI levels (oversold/overbought)
   - MACD crossover signals
   - Moving average trends
   - Bollinger Band position
   - Momentum indicators

2. **Sentiment Analysis (30% weight)**:
   - News sentiment scores
   - Positive/negative article count
   - Market sentiment trends

**Signal Types**:
- 🟢 **BUY**: Strong positive signals, confidence > 70%
- 🔴 **SELL**: Strong negative signals, confidence > 70%
- 🟡 **HOLD**: Mixed signals, wait for clearer direction

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Data**: yfinance (Yahoo Finance API)
- **News**: RSS feeds + Yahoo Finance News API
- **Analytics**: NumPy, Pandas, custom indicators
- **Sentiment**: VADER Sentiment, TextBlob
- **Visualization**: Plotly

## 📝 Notes

- **First Run**: Initial data download may take a few minutes
- **Internet Required**: Real-time data and news require active internet connection
- **Data Storage**: Historical data cached in `data/` directory
- **News Cache**: News articles cached for 1 hour to reduce API calls

## 🔒 Security & Privacy

- No API keys required for basic functionality
- All data fetched from public APIs
- News data is cached locally
- No user data is collected or transmitted

## 🐛 Troubleshooting

**Issue**: No datasets found
- **Solution**: Run `python backend/fetch_data.py` manually

**Issue**: News not loading
- **Solution**: Check internet connection, click "Refresh News Now"

**Issue**: Installation errors
- **Solution**: Ensure Python 3.8+ is installed, try `pip install --upgrade pip` first

**Issue**: Charts not displaying
- **Solution**: Ensure you have at least 50 data points (wait for data download)

## 📄 License

This project is for educational and personal use.

## 🤝 Support

For issues or questions, please check:
1. Ensure all dependencies are installed
2. Verify internet connection
3. Check that data files exist in `data/` directory

---

**Happy Trading! 📈🚀**






