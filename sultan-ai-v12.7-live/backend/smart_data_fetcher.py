"""
Smart Data Fetcher - Fast loading with optional real-time updates
Loads cached data instantly, then updates incrementally in background if needed
"""
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
from data_loader import load_csv_robust, save_csv_clean, clean_dataframe

def get_latest_incremental_update(cached_df, symbol, interval="30m"):
    """
    Fetch only the latest candles (incremental update) instead of full history.
    This is much faster than downloading all historical data.
    """
    if cached_df.empty or len(cached_df) == 0:
        return pd.DataFrame()
    
    try:
        # Get the last timestamp from cached data
        last_timestamp = cached_df.index[-1]
        
        # Fetch only last 1 day of data (much faster than full history)
        # We'll merge only new candles
        latest = yf.download(
            symbol, 
            period="1d", 
            interval=interval, 
            progress=False, 
            timeout=5
        )
        
        if latest.empty:
            return pd.DataFrame()
        
        latest = clean_dataframe(latest)
        
        if latest.empty:
            return pd.DataFrame()
        
        # Filter only new data (after last cached timestamp)
        new_data = latest[latest.index > last_timestamp]
        
        return new_data
        
    except Exception as e:
        print(f"Incremental update error: {e}")
        return pd.DataFrame()

def merge_incremental_data(cached_df, new_data):
    """Merge new incremental data with cached data"""
    if new_data.empty:
        return cached_df
    
    if cached_df.empty:
        return new_data
    
    # Combine and remove duplicates (keep latest)
    combined = pd.concat([cached_df, new_data])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    
    # Keep only last 2 years of data to manage size
    if len(combined) > 0:
        two_years_ago = combined.index[-1] - timedelta(days=730)
        combined = combined[combined.index >= two_years_ago]
    
    return combined

def smart_load_data(csv_path, symbol, use_real_time=False, max_age_minutes=15):
    """
    Smart data loader: Fast cache-first with optional real-time updates
    
    Args:
        csv_path: Path to cached CSV file
        symbol: Yahoo Finance symbol (e.g., "EURUSD=X")
        use_real_time: If True, check for updates
        max_age_minutes: If cached data is older than this, fetch updates
        
    Returns:
        DataFrame with cached data (fast) + optional updates
    """
    # Step 1: Load cached data instantly (fast path)
    cached_df = load_csv_robust(csv_path)
    
    if cached_df.empty:
        # No cache exists, fetch full data
        try:
            df = yf.download(symbol, period="2y", interval="1h", progress=False, timeout=10)
            if not df.empty:
                df = clean_dataframe(df)
                if not df.empty:
                    save_csv_clean(df, csv_path)
                    return df
        except Exception as e:
            print(f"Initial fetch error: {e}")
        return pd.DataFrame()
    
    # Step 2: Check if update is needed
    if not use_real_time:
        # Fast mode: return cached data immediately
        return cached_df
    
    # Step 3: Check cache age
    if len(cached_df) > 0:
        last_update = cached_df.index[-1]
        now = datetime.now(last_update.tz) if hasattr(last_update, 'tz') else datetime.now()
        age_minutes = (now - last_update).total_seconds() / 60
        
        if age_minutes < max_age_minutes:
            # Cache is fresh, return cached data
            return cached_df
    
    # Step 4: Fetch incremental update (only latest candles)
    try:
        new_data = get_latest_incremental_update(cached_df, symbol)
        
        if not new_data.empty:
            # Merge new data with cached
            updated_df = merge_incremental_data(cached_df, new_data)
            
            # Save updated data to cache
            save_csv_clean(updated_df, csv_path)
            
            return updated_df
        else:
            # No new data, return cached
            return cached_df
            
    except Exception as e:
        print(f"Update error: {e}, using cached data")
        return cached_df

def fast_load_only(csv_path):
    """Ultra-fast loading - cache only, no network calls"""
    return load_csv_robust(csv_path)





