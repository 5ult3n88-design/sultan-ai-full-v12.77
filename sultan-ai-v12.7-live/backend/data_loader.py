"""
Robust data loading utility with CSV cleaning
"""
import pandas as pd
import os

def clean_dataframe(df):
    """Clean dataframe - remove header rows and ensure numeric types"""
    if df.empty:
        return df.copy()
    
    df = df.copy()  # Avoid SettingWithCopyWarning
    
    # Remove rows where index is not a datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()].copy()
    
    # Clean numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with non-numeric data in essential columns
    df = df.dropna(subset=['Close']).copy()
    
    # Filter rows where Close is numeric
    mask = df['Close'].apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))
    df = df[mask].copy()
    
    return df

def load_csv_robust(path):
    """Load CSV file with robust error handling"""
    if not os.path.exists(path):
        return pd.DataFrame()
    
    try:
        # Read CSV file
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        
        # Remove any rows where the index is a string header (like 'Ticker', 'Datetime')
        if len(df) > 0:
            # Convert index to datetime and filter out invalid ones
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]
            
            # Remove rows where index is before year 2000 (likely header/data errors)
            df = df[df.index >= pd.Timestamp('2000-01-01')]
        
        # Clean numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                # Convert to numeric, coercing errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in Close (essential column)
        df = df.dropna(subset=['Close'])
        
        # Remove any rows where price columns contain text values
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                # Check if values are numeric
                df = df[pd.to_numeric(df[col], errors='coerce').notna()]
        
        # Ensure we have valid data
        if len(df) == 0:
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        # Try reading with skiprows to skip header row
        try:
            df = pd.read_csv(path, skiprows=[1], index_col=0, parse_dates=True)
            df = clean_dataframe(df)
            return df
        except Exception as e2:
            return pd.DataFrame()

def save_csv_clean(df, path):
    """Save dataframe to CSV in clean format"""
    try:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]

        # Remove any index name to avoid header issues
        df.index.name = None

        # Save
        df.to_csv(path, index=True)
        return True
    except Exception as e:
        print(f"Error saving CSV: {e}")
        return False

def load_symbol_data(symbol):
    """Load data for a specific symbol"""
    # Map symbol to filename
    symbol_clean = symbol.replace('=', '').replace('/', '').replace('-', '-')

    # Try different filename patterns
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    possible_files = [
        f"{data_dir}/{symbol_clean}_data.csv",
        f"{data_dir}/{symbol}_data.csv",
        f"{data_dir}/{symbol.replace('=X', 'X')}_data.csv",
    ]

    for filepath in possible_files:
        if os.path.exists(filepath):
            df = load_csv_robust(filepath)
            if df is not None and len(df) > 0:
                return df

    return None

