# 🚀 Running Trading Robot in VS Code

Complete guide to running your autonomous trading robot in Visual Studio Code.

---

## 📂 Step 1: Open Project in VS Code

### Option A: From Command Line
```bash
cd /home/user/sultan-ai-full-v12.77/sultan-ai-v12.7-live
code .
```

### Option B: From VS Code
1. Open VS Code
2. File → Open Folder
3. Navigate to `/home/user/sultan-ai-full-v12.77/sultan-ai-v12.7-live`
4. Click "Open"

---

## ⚡ Quick Start - Run Immediately

### Method 1: Using Debug Panel (Easiest)

1. **Open Debug Panel:**
   - Click the "Run and Debug" icon in sidebar (▶️ with bug icon)
   - Or press `Ctrl+Shift+D` (Windows/Linux) or `Cmd+Shift+D` (Mac)

2. **Select a Configuration:**
   At the top, you'll see a dropdown with these options:
   - **🤖 Run Trading Robot Demo** ← Start here!
   - 🧪 Test Robot (Single Cycle)
   - 🚀 Run Robot (5 Cycles)
   - 🔧 Generate Demo Data
   - 📊 Start Streamlit Dashboard

3. **Click the Green Play Button (▶️)**
   - Robot will run in the integrated terminal below

4. **Watch the Output!**
   - See detailed analysis for each symbol
   - View confidence scores
   - Monitor trades

### Method 2: Using Tasks (Command Palette)

1. **Open Command Palette:**
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)

2. **Type:** `Tasks: Run Task`

3. **Select:**
   - **🤖 Run Trading Robot Demo** ← Recommended first!
   - 🧪 Test Robot
   - 🚀 Run Robot (3 Cycles)
   - 📊 Start Dashboard
   - 🔧 Generate Data
   - 📦 Install Dependencies

### Method 3: Using Integrated Terminal

1. **Open Terminal in VS Code:**
   - Menu: Terminal → New Terminal
   - Or press `` Ctrl+` `` (backtick)

2. **Run Commands:**
   ```bash
   # Visual demo (best for first run)
   python demo_robot.py

   # Test run (single cycle)
   python run_trading_robot.py --test

   # Run with custom settings
   python run_trading_robot.py --confidence 65 --cycles 3
   ```

---

## 🎯 Recommended First Run

**Run the Visual Demo:**

1. Open Debug Panel (`Ctrl+Shift+D`)
2. Select "🤖 Run Trading Robot Demo"
3. Click green play button ▶️
4. Watch the detailed analysis in terminal!

You'll see:
```
======================================================================
  🤖 TRADING ROBOT DEMO
======================================================================

📋 Configuration:
   Symbols: EURUSD=X, AAPL, GOOGL, MSFT
   Min Confidence: 65%
   Risk per Trade: 1%
   Starting Balance: $10,000

----------------------------------------------------------------------
Symbol: EURUSD=X @ $1.08450
----------------------------------------------------------------------
📈 Action: BUY
🟢 Confidence: 78.5%

📊 ML Analysis:
   Direction: UP (82.0% confidence)
   Predicted Change: +0.82%
   Model: Ensemble ML Model

📈 Technical Analysis:
   Strength: 72/100
   Risk Level: Medium
   Volatility: 12.50%
...
```

---

## 📊 View the Dashboard in VS Code

### Start the Dashboard:

**Option 1: Debug Panel**
1. Select "📊 Start Streamlit Dashboard"
2. Click play ▶️

**Option 2: Terminal**
```bash
streamlit run frontend/Home.py
```

### Access Dashboard:

**VS Code has built-in browser preview!**

1. **Look for the popup notification:**
   - VS Code will show: "Your application running on port 8501 is available"
   - Click **"Open in Browser"** or **"Open in Preview"**

2. **Or manually open:**
   - Press `Ctrl+Shift+P`
   - Type: `Simple Browser: Show`
   - Enter URL: `http://localhost:8501`

3. **Or use Ports View:**
   - Open Terminal panel
   - Click "PORTS" tab (next to TERMINAL)
   - Right-click port 8501
   - Select "Open in Browser" or "Preview in Editor"

### Navigate to Robot:
- Once dashboard opens, click **"Auto Trading Robot"** in the sidebar
- Click **"Run Once"** to test
- Click **"Start"** to run continuously

---

## ⌨️ Keyboard Shortcuts

| Action | Windows/Linux | Mac |
|--------|---------------|-----|
| **Open Command Palette** | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| **Run/Debug Panel** | `Ctrl+Shift+D` | `Cmd+Shift+D` |
| **New Terminal** | `` Ctrl+` `` | `` Cmd+` `` |
| **Start Debugging** | `F5` | `F5` |
| **Stop Debugging** | `Shift+F5` | `Shift+F5` |
| **Run Task** | `Ctrl+Shift+B` | `Cmd+Shift+B` |

---

## 🎮 Available Configurations

### 1. 🤖 Run Trading Robot Demo
**What it does:** Visual terminal demo with detailed analysis
**Best for:** First time users, seeing detailed breakdown
**Output:** Beautiful formatted analysis for each symbol

### 2. 🧪 Test Robot (Single Cycle)
**What it does:** Runs one analysis cycle for 3 symbols
**Best for:** Quick testing
**Symbols:** EURUSD=X, AAPL, GOOGL

### 3. 🚀 Run Robot (5 Cycles)
**What it does:** Runs 5 cycles, checks every 5 minutes
**Best for:** Watching robot over time
**Settings:** 65% min confidence, 4 symbols

### 4. 🔧 Generate Demo Data
**What it does:** Creates new demo data for 16 symbols
**Best for:** Regenerating data if needed
**Time:** ~10 seconds

### 5. 📊 Start Streamlit Dashboard
**What it does:** Launches web dashboard
**Best for:** Full visual interface
**Access:** http://localhost:8501

---

## 🔧 Customizing Runs

### Edit Launch Configuration:

1. Open `.vscode/launch.json`
2. Find the configuration you want to change
3. Edit the `args` array

**Example - Change symbols:**
```json
{
    "name": "🚀 Run Robot (5 Cycles)",
    "args": [
        "--cycles", "5",
        "--confidence", "70",  // ← Changed from 65
        "--symbols", "EURUSD=X", "TSLA", "NVDA"  // ← Changed symbols
    ]
}
```

### Common Arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `--test` | Run single cycle | `--test` |
| `--cycles` | Number of cycles | `--cycles 10` |
| `--confidence` | Min confidence % | `--confidence 70` |
| `--risk` | Risk per trade % | `--risk 1.5` |
| `--balance` | Starting balance | `--balance 20000` |
| `--interval` | Seconds between cycles | `--interval 600` |
| `--symbols` | Symbols to trade | `--symbols AAPL GOOGL` |

---

## 📁 Project Structure in VS Code

```
sultan-ai-v12.7-live/
├── 📂 backend/              # Core engine
│   ├── trading_robot.py     # Robot brain
│   ├── trade_executor.py    # Trade execution
│   ├── ml_model.py          # ML predictions
│   ├── analytics.py         # Technical indicators
│   └── fetch_news.py        # News sentiment
│
├── 📂 frontend/             # Dashboard
│   ├── Home.py              # Main page
│   └── pages/
│       └── Auto_Trading_Robot.py  # Robot dashboard
│
├── 📂 data/                 # Historical data (16 symbols)
├── 📂 models/               # ML models
│
├── 🐍 demo_robot.py         # Visual demo
├── 🐍 run_trading_robot.py  # CLI interface
├── 🐍 generate_demo_data.py # Data generator
│
├── 📄 requirements.txt      # Dependencies
└── 📂 .vscode/              # VS Code configs
    ├── launch.json          # Debug configurations
    ├── tasks.json           # Tasks
    └── settings.json        # Workspace settings
```

---

## 🐛 Debugging

### Set Breakpoints:

1. **Open any Python file** (e.g., `backend/trading_robot.py`)
2. **Click left of line number** - Red dot appears
3. **Start debugging** (F5)
4. **Code pauses** at breakpoint
5. **Inspect variables** in left panel

### Debug Controls:

- **Continue** (F5): Resume execution
- **Step Over** (F10): Execute current line
- **Step Into** (F11): Enter function
- **Step Out** (Shift+F11): Exit function
- **Stop** (Shift+F5): Stop debugging

### Useful Breakpoint Locations:

- `backend/trading_robot.py:60` - analyze_symbol function
- `backend/trading_robot.py:100` - should_enter_trade function
- `backend/trade_executor.py:50` - open_position function
- `backend/ml_model.py:252` - predict_price_movement function

---

## 💡 Tips & Tricks

### 1. Split Terminal
- Click the split terminal icon in terminal panel
- Run dashboard in one, robot in another

### 2. Watch Output
- Output appears in "TERMINAL" tab at bottom
- Scroll to see full analysis
- Use `Ctrl+C` to stop running programs

### 3. Multiple Terminals
- Terminal → New Terminal
- Run different commands simultaneously
- Example: Dashboard in Terminal 1, Robot in Terminal 2

### 4. Quick Open Files
- Press `Ctrl+P`
- Type filename
- Instantly jump to file

### 5. Search Across Files
- Press `Ctrl+Shift+F`
- Search entire project
- Great for finding specific code

### 6. Integrated Git
- Click Source Control icon (branch icon)
- See changes, commit, push
- All in VS Code!

---

## 🎨 Recommended Extensions

VS Code will suggest these (click "Install" when prompted):

1. **Python** (ms-python.python)
   - Python language support
   - Debugging, linting, IntelliSense

2. **Pylance** (ms-python.vscode-pylance)
   - Fast Python language server
   - Better autocomplete

3. **Jupyter** (ms-toolsai.jupyter)
   - If you want to use notebooks

4. **autopep8** (ms-python.autopep8)
   - Code formatting

---

## 🚨 Troubleshooting

### "Python not found"
**Solution:**
1. Install Python extension
2. Select interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter`
3. Choose Python 3.x

### "Module not found"
**Solution:**
```bash
# In VS Code terminal:
pip install -r requirements.txt
```

Or use task:
- `Ctrl+Shift+P` → `Tasks: Run Task` → `📦 Install Dependencies`

### Dashboard won't open
**Solution:**
1. Check PORTS tab in terminal panel
2. Verify port 8501 is running
3. Click "Open in Browser" next to port 8501

### Terminal commands not working
**Solution:**
- Make sure you're in the correct directory
- Terminal shows: `sultan-ai-v12.7-live`
- If not, run: `cd sultan-ai-v12.7-live`

---

## 🎯 Common Workflows

### Workflow 1: Quick Demo
```
1. Open VS Code
2. Ctrl+Shift+D (Debug panel)
3. Select "🤖 Run Trading Robot Demo"
4. F5 (Run)
5. Watch output!
```

### Workflow 2: Dashboard
```
1. Ctrl+` (Open terminal)
2. streamlit run frontend/Home.py
3. Click "Open in Browser" notification
4. Navigate to "Auto Trading Robot"
5. Click "Run Once"
```

### Workflow 3: Custom Run
```
1. Ctrl+` (Open terminal)
2. python run_trading_robot.py --confidence 60 --cycles 3
3. Monitor output
```

### Workflow 4: Development
```
1. Open file (e.g., backend/trading_robot.py)
2. Set breakpoints (click left of line numbers)
3. F5 (Start debugging)
4. Inspect variables when paused
5. F10 to step through code
```

---

## ✅ Quick Checklist

Before running for first time:

- [ ] Opened folder in VS Code
- [ ] Python extension installed
- [ ] Selected Python interpreter
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Demo data generated (should already be there)

Ready to run:

- [ ] Debug panel opened (`Ctrl+Shift+D`)
- [ ] Configuration selected
- [ ] Press F5 or click ▶️
- [ ] Watch the magic happen!

---

## 🎉 You're Ready!

**Easiest way to start:**

1. Press `Ctrl+Shift+D`
2. Select "🤖 Run Trading Robot Demo"
3. Press `F5`
4. Watch your autonomous trading robot work!

---

## 📚 More Help

- **ROBOT_README.md** - Feature overview
- **TRADING_ROBOT_GUIDE.md** - Complete manual
- **ACCESS_DASHBOARD.md** - Dashboard guide

---

**Happy coding in VS Code!** 🚀💻
