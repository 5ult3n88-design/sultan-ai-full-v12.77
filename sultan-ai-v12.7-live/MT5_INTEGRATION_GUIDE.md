# 🔌 MetaTrader 5 Integration Guide

## Overview

Connect your autonomous trading robot to MetaTrader 5 for **real live/demo trading** instead of paper trading!

**⚠️ IMPORTANT:** This connects to REAL trading platforms. Always test on **DEMO accounts** first!

---

## 📋 What You Need

### 1. MetaTrader 5 Terminal
- **Download:** https://www.metatrader5.com/en/download
- **Platforms:** Windows, Mac, Linux (via Wine)
- **Free:** Yes, completely free

### 2. MT5 Python Package
```bash
pip install MetaTrader5
```

### 3. Trading Account
- **Demo Account:** Free, practice money (recommended for testing)
- **Live Account:** Real money (use ONLY after thorough testing!)

---

## 🚀 Quick Setup (Windows)

### Step 1: Install MT5
1. Download MT5 from your broker's website or https://www.metatrader5.com
2. Install MT5 terminal
3. Open MT5 and create a **DEMO account** (File → Open Account → Demo)

### Step 2: Install Python Package
```powershell
pip install MetaTrader5
```

### Step 3: Test Connection
```powershell
cd C:\Users\user\Desktop\sultan-ai-full-v12.77-main\sultan-ai-v12.7-live
python backend\mt5_connector.py
```

You should see:
```
✅ Connected to MT5
   Account: 12345678
   Server: BrokerServer-Demo
   Balance: $10000.00
   Equity: $10000.00
```

### Step 4: Run Robot with MT5
```powershell
python backend\mt5_robot.py --test --symbols EURUSD GBPUSD
```

---

## 🎮 Usage

### Method 1: Use Already Logged-In MT5

**Easiest way!** Just log into MT5 terminal, then:

```bash
# Test with MT5 (1 cycle)
python backend/mt5_robot.py --test

# Run continuously
python backend/mt5_robot.py --symbols EURUSD GBPUSD XAUUSD

# Custom settings
python backend/mt5_robot.py \
  --symbols EURUSD GBPUSD \
  --confidence 70 \
  --risk 1.0 \
  --interval 300
```

### Method 2: Provide Login Credentials

```bash
python backend/mt5_robot.py \
  --account 12345678 \
  --password "YourPassword" \
  --server "BrokerServer-Demo" \
  --symbols EURUSD GBPUSD
```

### Method 3: Paper Trading Mode (No MT5)

```bash
# Use --paper flag for paper trading without MT5
python backend/mt5_robot.py --paper --test
```

---

## ⚙️ Configuration Options

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--symbols` | Symbols to trade | EURUSD, GBPUSD, XAUUSD | `--symbols EURUSD USDJPY` |
| `--account` | MT5 account number | None (uses logged-in account) | `--account 12345678` |
| `--password` | MT5 password | None | `--password "MyPass123"` |
| `--server` | Broker server | None | `--server "ICMarkets-Demo"` |
| `--confidence` | Min confidence % | 70 | `--confidence 75` |
| `--risk` | Risk per trade % | 1.0 | `--risk 2.0` |
| `--interval` | Check interval (sec) | 300 | `--interval 600` |
| `--paper` | Paper trading mode | False | `--paper` |
| `--test` | Run 1 cycle & exit | False | `--test` |
| `--cycles` | Max cycles | Unlimited | `--cycles 10` |

---

## 🔍 Symbol Mapping

The robot automatically handles symbol name differences:

| Your Robot | MT5 Broker | Auto-Detected |
|------------|------------|---------------|
| EURUSD=X | EURUSD | ✅ Yes |
| EURUSD=X | EURUSDm | ✅ Yes |
| EURUSD=X | EURUSD.raw | ✅ Yes |
| GBPUSD=X | GBPUSD | ✅ Yes |
| XAUUSD=X | XAUUSD | ✅ Yes |
| AAPL | AAPL | ✅ Yes |

The robot tries multiple variations automatically!

---

## 📊 What the Robot Does with MT5

### 1. Analyzes Markets
- Loads data from MT5 (not CSV files)
- Calculates technical indicators
- Runs ML predictions
- Analyzes news sentiment

### 2. Makes Decisions
- Calculates 6-factor confidence
- Validates entry criteria
- Checks risk management

### 3. Executes Trades on MT5
- Opens positions with proper lot sizing
- Sets stop-loss automatically
- Sets take-profit automatically
- Tracks all trades

### 4. Manages Positions
- Monitors open trades
- MT5 automatically closes at SL/TP
- Robot can manually close if needed
- Updates P&L in real-time

---

## 💰 Risk Management with MT5

### Automatic Position Sizing

The robot calculates lot sizes based on:
- Your account equity
- Risk percentage (default: 1%)
- Stop-loss distance
- Symbol contract size

**Example:**
```
Account Equity: $10,000
Risk per Trade: 1% = $100
Stop Loss: 50 pips
EURUSD Lot Size: 0.20 lots (calculated automatically)
```

### Safety Limits

- ✅ Maximum risk per trade: 1-2% of equity
- ✅ Minimum lot size: 0.01 (micro lots)
- ✅ Stop-loss: Always set automatically
- ✅ Take-profit: 2.5:1 risk-reward
- ✅ Maximum positions: 5 simultaneously

---

## 🧪 Testing on Demo Account

**ALWAYS test on demo first!**

### Get a Demo Account

Most brokers offer free demo accounts:

1. **IC Markets:** https://www.icmarkets.com/demo-trading-account
2. **Pepperstone:** https://pepperstone.com/demo
3. **OANDA:** https://www.oanda.com/demo
4. **XM:** https://www.xm.com/demo

### Steps:

1. Open MT5 → File → Open Account
2. Select "Demo Account"
3. Fill in details (use real email!)
4. Get account number & password
5. Test robot with these credentials

---

## 🔧 Troubleshooting

### "Failed to connect to MT5"

**Solution:**
1. Make sure MT5 terminal is **running**
2. Make sure you're **logged in** to an account
3. Try restarting MT5
4. Check firewall isn't blocking MT5

### "Symbol not found"

**Solution:**
1. Open MT5 → View → Market Watch
2. Right-click → Symbols
3. Search for your symbol
4. Click "Show" to enable it
5. Try again

### "Order failed: Invalid volume"

**Solution:**
- Check minimum lot size (usually 0.01)
- Increase risk amount
- Check account has enough margin

### "Not connected" error

**Solution:**
```bash
# Test connection separately
python backend/mt5_connector.py
```

If this fails, MT5 isn't properly configured.

---

## 📈 Example Session

```powershell
PS> python backend\mt5_robot.py --test --symbols EURUSD GBPUSD

============================================================
MT5 Trading Robot
============================================================

🔌 Connecting to MetaTrader 5...
✅ Connected to MT5 successfully!
   Account: 12345678
   Balance: $10000.00
   Equity: $10000.00
   Leverage: 1:500

🧪 Running test cycle...

============================================================
[2024-12-10 10:30:00] Starting analysis cycle
============================================================

[EURUSD] Analyzing...
[EURUSD] Signal: BUY | Confidence: 78.5% | ML: UP (82%) | Tech: 72/100 | Risk: Medium
   💰 MT5 Trade: Position opened: BUY 0.20 lots of EURUSD at 1.08450

[GBPUSD] Analyzing...
[GBPUSD] Signal: HOLD | Confidence: 58.2% | ML: UP (63%) | Tech: 55/100 | Risk: High
[GBPUSD] Trade Decision: Confidence below threshold (65%)

============================================================
Account Summary:
  MT5 Balance: $10000.00
  MT5 Equity: $10010.50
  Open Positions: 1
  Total Trades: 1
============================================================

Test complete!
```

---

## 🛡️ Safety Features

### Built-in Protection

1. **Risk Limits**
   - Never risk more than 1-2% per trade
   - Position sizing based on account equity
   - Maximum 5 positions open

2. **Stop Loss**
   - Always set on every trade
   - ATR-based (2x Average True Range)
   - Can't be disabled

3. **Take Profit**
   - 2.5:1 risk-reward minimum
   - Automatically calculated
   - Protects profits

4. **Confidence Filter**
   - Only trades high-confidence setups
   - Default: 70% minimum
   - Adjustable per your preference

---

## 🌍 Broker Compatibility

### Tested Brokers

✅ **Works with:**
- IC Markets
- Pepperstone
- OANDA
- XM
- FBS
- Most ECN/STP brokers

❌ **May not work with:**
- Brokers that block automated trading
- Brokers with strict API restrictions

### Check Your Broker

Ask your broker:
1. "Do you allow automated trading (Expert Advisors)?"
2. "Is the MT5 API accessible?"

Most reputable brokers support this!

---

## 📝 Important Notes

### Demo vs Live

| Feature | Demo Account | Live Account |
|---------|-------------|--------------|
| **Money** | Virtual | Real |
| **Risk** | None | Real risk |
| **Execution** | Instant | May have slippage |
| **Cost** | Free | Spreads & commissions |
| **Testing** | ✅ Recommended | ❌ Not for testing! |

**Always test thoroughly on demo before going live!**

### Recommended Testing Period

- ✅ **Minimum:** 2 weeks on demo
- ✅ **Better:** 1 month on demo
- ✅ **Best:** 2-3 months on demo

Track:
- Win rate
- Profit factor
- Maximum drawdown
- Number of trades

---

## 🔐 Security

### Protecting Your Credentials

**Never share:**
- MT5 account number
- MT5 password
- API keys
- Account details

**Best practices:**
1. Use demo account for testing
2. Use strong passwords
3. Enable 2FA if available
4. Don't share your trading strategy

---

## 📊 Monitoring Your Robot

### Check These Regularly

1. **MT5 Terminal**
   - Open positions
   - Account equity
   - Margin level

2. **Robot Output**
   - Confidence levels
   - Entry reasons
   - Risk levels

3. **Performance Metrics**
   - Win rate (should be 55%+)
   - Profit factor (should be >1.5)
   - ROI trend

---

## 🎯 Tips for Success

### 1. Start Small
- Use minimum lot sizes
- Trade 1-2 symbols initially
- Set low risk (0.5-1%)

### 2. Monitor Closely
- Check robot every few hours (first week)
- Review all trades
- Adjust settings if needed

### 3. Keep Records
- Screenshot trades
- Log performance
- Note any issues

### 4. Be Patient
- Don't overtrade
- Let the robot work
- Trust the confidence system

### 5. Continuous Improvement
- Adjust confidence threshold
- Add/remove symbols
- Optimize risk settings

---

## 🆘 Getting Help

### Common Questions

**Q: Can I run 24/7?**
A: Yes, but monitor regularly. Use VPS for continuous operation.

**Q: What if MT5 closes?**
A: Robot stops. Restart both MT5 and robot.

**Q: Can I trade stocks?**
A: If your broker offers stocks in MT5, yes!

**Q: Minimum account size?**
A: Recommended: $500+ for demo, $2000+ for live

**Q: Can I use multiple robots?**
A: Yes, but each needs its own MT5 instance.

---

## 📚 Additional Resources

- **MT5 Official Docs:** https://www.metatrader5.com/en/terminal/help
- **Python MT5 Docs:** https://www.mql5.com/en/docs/python_metatrader5
- **Trading Robot Guide:** `TRADING_ROBOT_GUIDE.md`

---

## ⚠️ Disclaimer

**IMPORTANT:**

This robot is for **educational and research purposes**. Trading involves substantial risk of loss.

- ✅ Test thoroughly on demo
- ✅ Never risk money you can't afford to lose
- ✅ Past performance doesn't guarantee future results
- ✅ Understand the risks before trading
- ✅ Consider professional financial advice

**You are responsible for your trading decisions and results.**

---

## ✅ Checklist Before Live Trading

- [ ] Tested on demo for 1+ month
- [ ] Win rate above 55%
- [ ] Profit factor above 1.5
- [ ] Understand how robot works
- [ ] Know how to stop robot
- [ ] Can afford to lose entire amount
- [ ] Broker allows automated trading
- [ ] Stop-loss always enabled
- [ ] Risk per trade ≤ 1%
- [ ] Have emergency plan

**If all checked ✅ you're ready to consider live trading!**

---

**Happy Trading!** 🤖📈

Remember: **Start with DEMO, stay safe, and trade responsibly!**
