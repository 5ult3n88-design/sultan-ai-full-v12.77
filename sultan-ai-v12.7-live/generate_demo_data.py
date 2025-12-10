#!/usr/bin/env python3
"""
Generate demo data for testing the trading robot without yfinance
Creates realistic-looking price data for common symbols
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_ohlc(symbol, days=730, interval_minutes=30):
    """Generate realistic OHLC data"""

    # Base prices for different symbols
    base_prices = {
        'EURUSD=X': 1.08,
        'GBPUSD=X': 1.26,
        'USDJPY=X': 148.50,
        'XAUUSD=X': 2050.00,
        'AAPL': 195.00,
        'GOOGL': 140.00,
        'MSFT': 380.00,
        'TSLA': 240.00,
        'AMZN': 155.00,
        'NVDA': 490.00
    }

    base_price = base_prices.get(symbol, 100.0)

    # Calculate number of candles
    candles_per_day = (24 * 60) // interval_minutes
    total_candles = days * candles_per_day

    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=total_candles)

    # Generate price data with realistic patterns
    np.random.seed(hash(symbol) % (2**32))  # Consistent seed per symbol

    # Trend component (slow drift)
    trend = np.cumsum(np.random.randn(total_candles) * 0.0001)

    # Volatility (based on symbol type)
    if '=' in symbol or symbol.startswith('X'):  # Forex
        volatility = 0.0005
    else:  # Stocks
        volatility = 0.01

    # Generate returns
    returns = np.random.randn(total_candles) * volatility + trend

    # Calculate prices
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC
    data = []
    for i in range(total_candles):
        close = prices[i]

        # Generate realistic intrabar movement
        intrabar_vol = abs(np.random.randn()) * volatility * close
        high = close + intrabar_vol * np.random.rand()
        low = close - intrabar_vol * np.random.rand()

        # Open is between high and low
        open_price = low + (high - low) * np.random.rand()

        # Volume (higher for stocks)
        if '=' in symbol or symbol.startswith('X'):
            volume = int(np.random.uniform(1000, 5000))
        else:
            volume = int(np.random.uniform(50000000, 200000000))

        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })

    df = pd.DataFrame(data, index=timestamps)
    df.index.name = 'Datetime'

    return df

def main():
    """Generate demo data for all symbols"""

    symbols = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'XAUUSD=X',
        'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA',
        'META', 'NFLX', 'SPY', 'QQQ', 'BTC-USD', 'ETH-USD'
    ]

    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    print("🔄 Generating demo data for trading robot...\n")

    for symbol in symbols:
        print(f"  Generating {symbol}...", end=' ')

        # Generate 2 years of 30-minute data
        df = generate_realistic_ohlc(symbol, days=730, interval_minutes=30)

        # Save to CSV
        filename = f"{data_dir}/{symbol.replace('=', '').replace('/', '')}_data.csv"
        df.to_csv(filename)

        print(f"✅ {len(df)} candles saved to {filename}")

    print(f"\n✅ Demo data generation complete!")
    print(f"📊 Generated {len(symbols)} symbols with 2 years of historical data")
    print(f"💾 Data saved to '{data_dir}/' directory")
    print(f"\n🚀 You can now start the trading robot dashboard!")

if __name__ == "__main__":
    main()
