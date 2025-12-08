import yfinance as yf, os
import pandas as pd
from datetime import datetime, timedelta

# Forex pairs - comprehensive list with valid Yahoo Finance tickers
forex_symbols = {
    "XAUUSD": "GC=F",      # Gold futures (COMEX)
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
    "XAGUSD": "SI=F",      # Silver futures (COMEX)
    # Note: XAUGBP and XAUJPY are not available as direct tickers on Yahoo Finance
    # We'll use GC=F (gold) and calculate cross rates, or skip them
}

# Popular stocks for real-time trading
stock_symbols = {
    "AAPL": "AAPL",
    "GOOGL": "GOOGL",
    "MSFT": "MSFT",
    "TSLA": "TSLA",
    "AMZN": "AMZN",
    "NVDA": "NVDA",
    "META": "META",
    "NFLX": "NFLX",
    "SPY": "SPY",  # S&P 500 ETF
    "QQQ": "QQQ",  # NASDAQ ETF
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD"
}

# Combine all symbols
all_symbols = {**forex_symbols, **stock_symbols}

# Cross-platform path handling - works from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)

print(f"📥 Starting data fetch at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Total symbols: {len(all_symbols)}")
print(f"📁 Data directory: {data_dir}")

for name, ticker in all_symbols.items():
    try:
        print(f"📥 Downloading {name} ({ticker})...")
        # Add timeout and retry logic
        import time
        max_retries = 2
        df = None
        
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, period="2y", interval="1h", progress=False, timeout=30)
                if not df.empty:
                    break
                elif attempt < max_retries - 1:
                    print(f"   Retrying {name}...")
                    time.sleep(2)
            except Exception as retry_error:
                if attempt < max_retries - 1:
                    print(f"   Retry {attempt + 1} failed, retrying...")
                    time.sleep(2)
                else:
                    raise retry_error
        
        if df is not None and not df.empty:
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
            
            # Clean the dataframe - ensure proper numeric columns
            for col in df.columns:
                if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if df[col].dtype == 'object':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN in Close
            df = df.dropna(subset=['Close'])
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[df.index.notna()]
            
            # Only save if we have meaningful data
            if len(df) >= 10:
                csv_path = os.path.join(data_dir, f"{name}.csv")
                # Save with clean format - remove index name
                df.index.name = None
                df.to_csv(csv_path, index=True)
                print(f"✅ Saved {name} ({len(df)} records)")
            else:
                print(f"⚠️ Insufficient data for {name} (only {len(df)} records)")
        else:
            print(f"⚠️ No data available for {name} ({ticker})")
    except Exception as e:
        print(f"⚠️ Failed {name}: {str(e)[:100]}")  # Truncate long error messages

print(f"✅ Data fetch completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
