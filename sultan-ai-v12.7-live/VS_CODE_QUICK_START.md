# ⚡ VS Code Quick Start - 30 Seconds to Running

## 🚀 Super Fast Start

### 1️⃣ Open Project (5 seconds)
```bash
cd /home/user/sultan-ai-full-v12.77/sultan-ai-v12.7-live
code .
```

### 2️⃣ Run Robot (2 clicks)
1. Press **`Ctrl+Shift+D`** (Debug panel)
2. Click **Green Play Button ▶️**

**DONE!** Robot is running! 🤖

---

## 🎯 What You'll See

```
======================================================================
  🤖 TRADING ROBOT DEMO
======================================================================

----------------------------------------------------------------------
Symbol: EURUSD=X @ $1.08450
----------------------------------------------------------------------
📈 Action: HOLD
🟢 Confidence: 76.2%

📊 ML Analysis:
   Direction: UP (70.0% confidence)
   Predicted Change: +0.39%

💰 Account Summary:
  Balance: $10,000.00
  Equity: $10,000.00
  ROI: 0.00%
======================================================================
```

---

## 🎮 Pre-Configured Options

In Debug Panel dropdown, choose:

| Option | What it Does |
|--------|--------------|
| **🤖 Run Trading Robot Demo** | ← **START HERE!** Visual demo |
| 🧪 Test Robot | Quick test (1 cycle) |
| 🚀 Run Robot (5 Cycles) | Watch over time |
| 📊 Start Dashboard | Web interface |

---

## ⌨️ Essential Shortcuts

| Action | Shortcut |
|--------|----------|
| **Run** | `F5` |
| **Stop** | `Shift+F5` |
| **Debug Panel** | `Ctrl+Shift+D` |
| **Terminal** | `` Ctrl+` `` |
| **Command Palette** | `Ctrl+Shift+P` |

---

## 📊 View Dashboard

### Start:
1. Terminal: `streamlit run frontend/Home.py`
2. Click notification: **"Open in Browser"**
3. Or open: `http://localhost:8501`

### Access in VS Code:
- Click **PORTS** tab (bottom panel)
- Right-click **8501**
- Select **"Preview in Editor"**

---

## 💻 Alternative: Terminal

```bash
# Visual demo
python demo_robot.py

# Quick test
python run_trading_robot.py --test

# Custom run
python run_trading_robot.py --confidence 65 --cycles 3
```

---

## 🆘 Help

**Commands not working?**
- Open terminal: `` Ctrl+` ``
- Check you're in: `sultan-ai-v12.7-live`

**Module not found?**
- Terminal: `pip install -r requirements.txt`

**More help:**
- Read `VSCODE_GUIDE.md` for complete guide

---

## ✅ That's It!

**Press `Ctrl+Shift+D` → Click ▶️ → Watch it work!**

🎉 **Your robot is ready to trade!** 🤖📈
