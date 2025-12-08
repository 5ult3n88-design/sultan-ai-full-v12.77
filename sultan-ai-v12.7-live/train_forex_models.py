"""
Train Pre-trained Models for All Forex Pairs
This script trains specialized ML models for each forex pair including XAUUSD
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from analytics import calculate_advanced_indicators
from data_validator import clean_dataframe_for_analysis
from pretrained_models import train_forex_model, FOREX_SYMBOLS

def train_all_forex_models():
    """Train models for all supported forex pairs"""
    
    print("=" * 60)
    print("🚀 Training Pre-trained Forex Models")
    print("=" * 60)
    print()
    
    # Use FOREX_SYMBOLS from pretrained_models - comprehensive list
    forex_pairs = FOREX_SYMBOLS.copy()
    
    trained_count = 0
    failed_count = 0
    
    for symbol_name, yahoo_symbol in forex_pairs.items():
        print(f"\n📊 Training model for {symbol_name} ({yahoo_symbol})...")
        print("-" * 60)
        
        try:
            # Fetch historical data using yfinance directly
            print(f"   📥 Fetching historical data for {yahoo_symbol}...")
            df = yf.download(yahoo_symbol, period='2y', interval='1h', progress=False)
            
            if df is None or df.empty or len(df) < 200:
                print(f"   ❌ Insufficient data for {symbol_name} ({len(df) if df is not None and not df.empty else 0} rows)")
                failed_count += 1
                continue
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
            
            print(f"   ✅ Loaded {len(df)} data points")
            
            # Clean data
            df = clean_dataframe_for_analysis(df)
            
            if len(df) < 200:
                print(f"   ❌ Insufficient data after cleaning for {symbol_name} ({len(df)} rows)")
                failed_count += 1
                continue
            
            # Calculate technical indicators
            print(f"   🔧 Calculating technical indicators...")
            df = calculate_advanced_indicators(df)
            
            # Add symbol attribute
            if not hasattr(df, 'attrs'):
                df.attrs = {}
            df.attrs['symbol'] = symbol_name
            
            # Train model
            print(f"   🤖 Training ML model...")
            model_data = train_forex_model(df, symbol_name)
            
            if model_data:
                trained_count += 1
                print(f"   ✅ Model trained successfully!")
                print(f"      Test Accuracy: {model_data.get('test_accuracy', 0):.2%}")
                print(f"      OOB Score: {model_data.get('oob_score', 0):.2%}" if model_data.get('oob_score') else "")
            else:
                print(f"   ❌ Failed to train model for {symbol_name}")
                failed_count += 1
                
        except Exception as e:
            print(f"   ❌ Error training {symbol_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    print()
    print("=" * 60)
    print(f"✅ Training Complete!")
    print(f"   Successfully trained: {trained_count} models")
    print(f"   Failed: {failed_count} models")
    print("=" * 60)

if __name__ == '__main__':
    train_all_forex_models()

