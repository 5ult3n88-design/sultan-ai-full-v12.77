# 🤖 Autonomous Trading Robot Guide

## Overview

The **Autonomous AI Trading Robot** is a fully automated trading system that makes trading decisions based on:

- 🧠 **Machine Learning Models** - Ensemble ML predictions with 70-95% confidence
- 📊 **Technical Analysis** - RSI, MACD, Bollinger Bands, Moving Averages, Stochastic, ATR
- 📰 **News Sentiment** - Real-time news analysis using VADER and TextBlob
- ⚖️ **Risk Management** - Advanced position sizing, stop-loss, and take-profit calculations
- 📈 **Pattern Recognition** - MACD crossovers, support/resistance, trend strength

## Key Features

### ✅ Autonomous Decision Making
- Makes BUY/SELL/HOLD decisions automatically
- Combines ML predictions with technical indicators
- Adjusts confidence based on signal convergence
- Only trades when confidence exceeds threshold (default: 65%)

### ✅ Strong Confidence System
The robot calculates confidence using **6 factors**:

1. **ML Model Confidence (42%)** - Enhanced ensemble ML prediction
2. **Technical Strength Alignment (28%)** - How well indicators align with ML
3. **Signal Convergence (25%)** - Agreement across multiple indicators
4. **News Sentiment (10%)** - Positive/negative news alignment
5. **Risk Adjustment (5%)** - Lower risk increases confidence
6. **Predicted Change Magnitude** - Strong predictions boost confidence

**Confidence Levels:**
- 🟢 **80%+ (Excellent)** - Very strong signal, all factors aligned
- 🔵 **70-80% (Very Good)** - Strong signal, good alignment
- 🟡 **60-70% (Good)** - Moderate confidence, acceptable risk
- 🟠 **50-60% (Fair)** - Mixed signals, proceed with caution
- 🔴 **<50% (Low)** - Weak signals, HOLD recommended

### ✅ Advanced Risk Management

**Position Sizing:**
- Automatically calculates position size based on risk % (default: 1%)
- Maximum 20% of equity per position
- Maximum 5 positions simultaneously
- Risk-reward ratio minimum: 1.5:1

**Stop Loss & Take Profit:**
- Dynamic stop-loss based on ATR (Average True Range)
- Take profit targets 2.5:1 risk-reward ratio
- Automatic position closure when targets hit
- Volatility-adjusted levels

**Safety Features:**
- Won't trade high-risk setups with low confidence (<75%)
- Requires strong technical alignment for entry
- Monitors all positions in real-time
- Automatic stop-loss/take-profit execution

### ✅ Paper Trading (Simulated)
- **100% SIMULATED** - No real money at risk
- Start with virtual balance (default: $10,000)
- Full position tracking and P&L calculation
- Realistic trade execution and slippage

## How to Use

### Method 1: Dashboard (Recommended)

1. **Start the Streamlit app:**
   ```bash
   streamlit run frontend/Home.py
   ```

2. **Navigate to "Auto Trading Robot" page**

3. **Configure the robot** (sidebar):
   - Select trading symbols
   - Set minimum confidence (recommended: 65-75%)
   - Set risk per trade (recommended: 0.5-2%)
   - Set check interval (recommended: 5-15 minutes)

4. **Start the robot:**
   - Click "▶️ Start" to run continuously
   - Or click "🔄 Run Once" to test a single cycle

5. **Monitor performance:**
   - Dashboard: Real-time metrics and status
   - Open Positions: Live P&L tracking
   - Performance: Equity curve and statistics
   - Recent Signals: All trading decisions with confidence
   - Trade History: Complete transaction log

### Method 2: Command Line

Run the robot standalone:

```bash
# Basic usage (default settings)
python run_trading_robot.py

# Custom settings
python run_trading_robot.py \
  --balance 10000 \
  --risk 1.0 \
  --confidence 70 \
  --interval 300 \
  --symbols EURUSD=X GBPUSD=X XAUUSD=X AAPL GOOGL

# Test run (single cycle)
python run_trading_robot.py --test

# Run for 10 cycles then stop
python run_trading_robot.py --cycles 10
```

**Command line options:**
- `--balance`: Initial balance (default: 10000)
- `--risk`: Risk per trade % (default: 1.0)
- `--confidence`: Minimum confidence % (default: 65)
- `--interval`: Check interval in seconds (default: 300)
- `--symbols`: Symbols to trade (space-separated)
- `--cycles`: Maximum cycles (default: unlimited)
- `--test`: Run one cycle and exit

## Understanding Robot Decisions

### Signal Generation Process

For each symbol, the robot:

1. **Loads latest price data** (30-minute candles, 2 years history)
2. **Calculates 25+ technical indicators**
3. **Runs ML ensemble prediction** (Random Forest + Gradient Boosting + Extra Trees)
4. **Fetches recent news** and analyzes sentiment
5. **Calculates advanced confidence** (6-factor model)
6. **Generates BUY/SELL/HOLD signal**

### Entry Criteria (ALL must be met)

✅ **Confidence** >= Minimum threshold (default: 65%)
✅ **Action** is BUY or SELL (not HOLD)
✅ **Position slot available** (max 5 positions)
✅ **Risk-Reward ratio** >= 1.5:1
✅ **Technical alignment** supports the direction
✅ **If high risk**, confidence must be >= 75%
✅ **Sufficient balance** available

### Example Decision Log

```
[EURUSD=X] Analyzing...
[EURUSD=X] Signal: BUY | Confidence: 78.5% | ML: UP (82%) | Tech: 72/100 | Risk: Medium
[EURUSD=X] Trade Decision: Position opened - BUY 0.5234 units at 1.08450

Decision Details:
  - ML predicts: +0.82% movement (82% confidence)
  - Technical strength: 72/100 (strong bullish)
  - Signal convergence: 85% (RSI, MACD, MA all aligned)
  - News sentiment: +0.15 (5 articles analyzed)
  - Risk level: Medium (volatility: 12.5%)
  - Entry: 1.08450, Stop: 1.08200, Target: 1.08750
  - Risk-Reward: 2.4:1
  - Final confidence: 78.5% ✅
```

## Dashboard Features

### 📊 Main Dashboard
- **Account metrics**: Balance, Equity, ROI, Win Rate
- **Runtime stats**: Hours running, signals generated, trades executed
- **Latest signals**: Real-time BUY/SELL/HOLD decisions for all symbols
- **Confidence visualization**: Color-coded confidence levels

### 💼 Open Positions
- Live P&L tracking for all positions
- Entry price, current price, unrealized profit
- Stop-loss and take-profit levels
- Holding time tracking
- Manual close option

### 📈 Performance
- **Equity curve**: Visual chart of account growth
- **Trade statistics**: Win rate, profit factor, avg win/loss
- **Risk metrics**: Sharpe ratio, max drawdown

### 🔔 Recent Signals
- Last 50 trading signals with full details
- Filter by action (BUY/SELL/HOLD)
- Filter by confidence level
- Filter by symbol
- Shows execution status (✅/❌)

### 📜 Trade History
- Complete log of all trades (open & close)
- P&L for each trade
- Holding time, exit reason
- Downloadable CSV export

## Performance Metrics Explained

### Win Rate
Percentage of profitable trades out of total closed trades.
- **Good**: 55%+
- **Excellent**: 65%+

### Profit Factor
Ratio of gross profit to gross loss.
- **Breakeven**: 1.0
- **Good**: 1.5+
- **Excellent**: 2.0+

### ROI (Return on Investment)
Percentage return on initial balance.
- Formula: (Current Equity - Initial Balance) / Initial Balance × 100

### Average Win/Loss
Average profit from winning trades vs average loss from losing trades.
- Target: Avg Win should be > Avg Loss

## Best Practices

### ✅ Recommended Settings

**For Conservative Trading:**
- Minimum Confidence: 70-75%
- Risk per Trade: 0.5-1%
- Check Interval: 15-30 minutes

**For Balanced Trading:**
- Minimum Confidence: 65-70%
- Risk per Trade: 1-2%
- Check Interval: 5-15 minutes

**For Aggressive Trading:**
- Minimum Confidence: 60-65%
- Risk per Trade: 2-3%
- Check Interval: 5-10 minutes

### ✅ Trading Tips

1. **Start with test runs**: Use `--test` flag or "Run Once" button
2. **Monitor first trades**: Watch the robot's decisions before leaving it alone
3. **Don't overtrade**: Limit to 3-5 symbols initially
4. **Review performance**: Check win rate and profit factor regularly
5. **Adjust confidence**: If too many trades, increase min confidence
6. **Be patient**: Good setups take time - don't lower standards
7. **Check news**: Major news events can affect performance

### ✅ Risk Management Rules

1. **Never risk more than 1-2% per trade**
2. **Maximum 5 open positions**
3. **Always use stop-loss**
4. **Respect the robot's decisions** (don't override manually too often)
5. **Monitor high-risk trades** closely
6. **Close losing positions** if fundamentals change

## Supported Symbols

### Forex Pairs (24/5 trading)
- EURUSD=X, GBPUSD=X, USDJPY=X
- AUDUSD=X, USDCAD=X, NZDUSD=X
- EURGBP=X, XAUUSD=X (Gold), XAGUSD=X (Silver)

### Stocks (market hours only)
- AAPL, GOOGL, MSFT, TSLA, AMZN
- NVDA, META, NFLX, SPY, QQQ

### Crypto (24/7 trading)
- BTC-USD, ETH-USD

**Note**: Add '=X' suffix for forex pairs in yfinance format

## Troubleshooting

### Robot not generating signals?
- Check data availability: Run `python backend/fetch_data.py`
- Verify symbols are correct format
- Check internet connection for news fetching

### No trades being executed?
- Confidence threshold may be too high - lower it slightly
- Technical requirements may be too strict
- Check if max positions reached (5 limit)

### Performance is poor?
- Review min confidence setting (may be too low)
- Check if trading during high volatility events
- Analyze trade history for patterns
- Consider adding/removing symbols

### Robot stopped unexpectedly?
- Check logs in terminal
- Verify data files in `data/` directory
- Ensure sufficient disk space for logging

## Data Storage

The robot saves state to:
- `data/trading_robot_state.json` - Positions and trade history
- `data/trading_robot_config.json` - Robot configuration

**State includes:**
- Current balance and equity
- Open positions
- Trade history (last 100 decisions)
- Equity curve data

State is automatically saved:
- After each analysis cycle
- When robot is stopped
- After each trade execution

## Technical Details

### ML Model Stack
- **Ensemble combination** of 3 models:
  - Random Forest Classifier (400 trees)
  - Gradient Boosting Classifier (200 estimators)
  - Extra Trees Classifier (300 trees)
- **Feature engineering**: 25+ technical features
- **Prediction**: Direction probability + magnitude

### Confidence Calculation
```python
confidence = (
    ml_confidence * 0.42 +           # ML model output
    tech_alignment * 0.28 +          # Technical indicators alignment
    signal_convergence * 0.25 +      # Cross-indicator agreement
    news_alignment * 0.10 +          # Sentiment alignment
    risk_adjustment * 0.05           # Risk-based adjustment
)

# Apply bonuses/penalties
if convergence > 85% and tech > 80%:
    confidence *= 1.20  # 20% bonus
if risk_level == "Low":
    confidence *= 1.15  # 15% bonus
if risk_level == "High":
    confidence *= 0.85  # 15% penalty
```

### Position Sizing
```python
risk_amount = equity * risk_percent
stop_loss_distance = abs(entry - stop_loss)
position_size = risk_amount / stop_loss_distance

# Cap at 20% of equity
max_size = equity * 0.20
position_size = min(position_size, max_size)
```

## Safety & Disclaimers

⚠️ **IMPORTANT DISCLAIMERS:**

1. **Paper Trading Only**: This robot is designed for **SIMULATED TRADING ONLY**. It does NOT connect to real brokers or execute real trades.

2. **Educational Purpose**: This is an educational project to demonstrate algorithmic trading concepts. Not financial advice.

3. **No Guarantees**: Past performance does not guarantee future results. The robot may lose money.

4. **Use at Own Risk**: You are responsible for any decisions made based on robot recommendations.

5. **Test Thoroughly**: Always test extensively before considering any real trading.

6. **Market Risks**: Markets are unpredictable. Even high-confidence signals can lose.

## Future Enhancements

Potential improvements (not yet implemented):

- [ ] Multi-timeframe analysis
- [ ] Support for more asset classes
- [ ] Real broker integration (with proper safeguards)
- [ ] Advanced portfolio optimization
- [ ] Machine learning model retraining
- [ ] Backtesting framework
- [ ] Email/SMS notifications
- [ ] Advanced charting in dashboard

## Support

For issues or questions:

1. Check this guide first
2. Review the troubleshooting section
3. Check code comments in source files
4. Review recent decisions log for patterns

---

**Happy (Paper) Trading! 🤖📈**

Remember: The best trader is a disciplined trader. Let the robot do what it does best - consistent, emotion-free analysis and execution.
