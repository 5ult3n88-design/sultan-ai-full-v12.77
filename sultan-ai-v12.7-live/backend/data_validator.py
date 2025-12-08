"""
Data validation and cleaning utility
Ensures DataFrames are in the correct format before analysis
"""
import pandas as pd
import numpy as np

def clean_dataframe_for_analysis(df):
    """
    Comprehensive data cleaning for analysis functions
    Handles MultiIndex columns, ensures numeric types, removes invalid data
    """
    if df.empty:
        return df
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Handle MultiIndex columns if present (from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Drop second level if it exists
        if df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        # Flatten tuple column names
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Required columns for analysis
    required_cols = ['Open', 'High', 'Low', 'Close']
    
    # Check and clean each required column
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        
        # Handle if column is a DataFrame (shouldn't happen, but just in case)
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
        
        # Convert to numeric, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in Close (essential column)
    df = df.dropna(subset=['Close'])
    
    # Ensure High >= Low for all rows (data integrity check)
    if 'High' in df.columns and 'Low' in df.columns:
        invalid_mask = df['High'] < df['Low']
        if invalid_mask.any():
            df = df[~invalid_mask]
    
    # Ensure Close is between Low and High (if both exist)
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        invalid_mask = (df['Close'] < df['Low']) | (df['Close'] > df['High'])
        if invalid_mask.any():
            # Fix invalid closes by clamping to Low/High range
            df.loc[df['Close'] < df['Low'], 'Close'] = df.loc[df['Close'] < df['Low'], 'Low']
            df.loc[df['Close'] > df['High'], 'Close'] = df.loc[df['Close'] > df['High'], 'High']
    
    # Sort by index (should be datetime)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    elif hasattr(df.index, 'sort_values'):
        df = df.sort_index()
    
    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='last')]
    
    return df

def validate_dataframe(df, min_rows=50):
    """
    Validate that DataFrame is suitable for analysis
    Returns (is_valid, error_message)
    """
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"Insufficient data: {len(df)} rows (minimum {min_rows} required)"
    
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for NaN in Close
    if df['Close'].isna().any():
        return False, f"NaN values found in Close column ({df['Close'].isna().sum()} rows)"
    
    # Check data types
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column {col} is not numeric type"
    
    return True, "OK"





