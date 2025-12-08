"""
Smart Money Concepts (SMC) Analysis Module
Includes support/resistance, trend lines, and market structure analysis
"""

import pandas as pd
import numpy as np

def identify_support_resistance(df, window=20, num_levels=3):
    """Identify key support and resistance levels using SMC principles"""
    
    if len(df) < window * 2:
        return {'support': [], 'resistance': []}
    
    # Import data validator for cleaning
    try:
        from data_validator import clean_dataframe_for_analysis
        df = clean_dataframe_for_analysis(df)
    except:
        pass
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
        except:
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else str(col) for col in df.columns]
            # Rename common columns
            rename_map = {}
            for col in df.columns:
                if 'Low' in col and 'Low' not in rename_map:
                    rename_map[col] = 'Low'
                elif 'High' in col and 'High' not in rename_map:
                    rename_map[col] = 'High'
                elif 'Close' in col and 'Close' not in rename_map:
                    rename_map[col] = 'Close'
                elif 'Open' in col and 'Open' not in rename_map:
                    rename_map[col] = 'Open'
            df = df.rename(columns=rename_map)
    
    # Ensure we have numeric Series - robust extraction
    try:
        # Direct extraction and conversion
        if 'Low' not in df.columns or 'High' not in df.columns or 'Close' not in df.columns:
            return {'support': [], 'resistance': []}
        
        # Extract as Series and convert to numeric
        low_series = df['Low'].copy()
        high_series = df['High'].copy()
        close_series = df['Close'].copy()
        
        # Handle if they're DataFrames (shouldn't happen, but safety check)
        if isinstance(low_series, pd.DataFrame):
            low_series = low_series.iloc[:, 0]
        if isinstance(high_series, pd.DataFrame):
            high_series = high_series.iloc[:, 0]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        
        # Convert to numeric, coercing errors
        low_series = pd.to_numeric(low_series, errors='coerce')
        high_series = pd.to_numeric(high_series, errors='coerce')
        close_series = pd.to_numeric(close_series, errors='coerce')
        
        # Drop any NaN values
        valid_mask = low_series.notna() & high_series.notna() & close_series.notna()
        low_series = low_series[valid_mask]
        high_series = high_series[valid_mask]
        close_series = close_series[valid_mask]
        
        if len(low_series) < window * 2:
            return {'support': [], 'resistance': []}
            
    except Exception as e:
        # Complete fallback - return empty
        return {'support': [], 'resistance': []}
    
    supports = []
    resistances = []
    
    # Look for local minima (support) and maxima (resistance)
    # Use len(low_series) instead of len(df) since we filtered
    series_len = len(low_series)
    if series_len < window * 2:
        return {'support': [], 'resistance': []}
    
    for i in range(window, series_len - window):
        # Support: local low with higher lows on both sides
        try:
            # Safe extraction - ensure we get a scalar float
            low_val_raw = low_series.iloc[i]
            if isinstance(low_val_raw, (pd.Series, pd.DataFrame, dict)):
                continue  # Skip if not a scalar
            
            low_val = float(low_val_raw)
            if pd.isna(low_val) or not np.isfinite(low_val):
                continue
            
            # Get min value as float - ensure we get a scalar
            window_slice = low_series.iloc[i-window:i+window]
            if len(window_slice) == 0:
                continue
                
            window_min_raw = window_slice.min()
            if isinstance(window_min_raw, (pd.Series, pd.DataFrame, dict)):
                continue
            
            window_min = float(window_min_raw)
            if pd.isna(window_min) or not np.isfinite(window_min):
                continue
            
            if abs(low_val - window_min) < 0.000001:  # Use tolerance for float comparison
                # Check if it's a significant level
                if len(supports) == 0:
                    supports.append({
                        'price': float(low_val),
                        'index': int(i),
                        'strength': 1.0
                    })
                else:
                    # Extract last price safely - ensure it's always a float
                    last_item = supports[-1]
                    last_price = None
                    
                    if isinstance(last_item, dict):
                        price_val = last_item.get('price')
                        if price_val is not None:
                            try:
                                last_price = float(price_val)
                                if not np.isfinite(last_price) or isinstance(last_price, (dict, list)):
                                    last_price = float(low_val)
                            except (TypeError, ValueError):
                                last_price = float(low_val)
                        else:
                            last_price = float(low_val)
                    elif isinstance(last_item, (int, float)):
                        try:
                            last_price = float(last_item)
                            if not np.isfinite(last_price) or isinstance(last_price, (dict, list)):
                                last_price = float(low_val)
                        except (TypeError, ValueError):
                            last_price = float(low_val)
                    else:
                        last_price = float(low_val)
                    
                    # Ensure last_price is valid before comparison
                    if last_price is None or not np.isfinite(last_price) or isinstance(last_price, (dict, list)):
                        last_price = float(low_val)
                    
                    try:
                        close_val_raw = close_series.iloc[i]
                        if isinstance(close_val_raw, (pd.Series, pd.DataFrame, dict, list)):
                            close_val = float(low_val)
                        else:
                            close_val = float(close_val_raw)
                        if pd.isna(close_val) or not np.isfinite(close_val):
                            close_val = float(low_val)
                    except:
                        close_val = float(low_val)
                    
                    # Final safety check - ensure all values are floats before arithmetic
                    if isinstance(last_price, (int, float)) and np.isfinite(last_price) and \
                       isinstance(low_val, (int, float)) and np.isfinite(low_val) and \
                       isinstance(close_val, (int, float)) and np.isfinite(close_val):
                        if abs(float(low_val) - float(last_price)) > float(close_val) * 0.001:
                            supports.append({
                                'price': float(low_val),
                                'index': int(i),
                                'strength': 1.0
                            })
        except (ValueError, TypeError, IndexError) as e:
            continue  # Skip this iteration if there's an error
        
        # Resistance: local high with lower highs on both sides
        try:
            # Safe extraction - ensure we get a scalar float
            high_val_raw = high_series.iloc[i]
            if isinstance(high_val_raw, (pd.Series, pd.DataFrame, dict)):
                continue  # Skip if not a scalar
            
            high_val = float(high_val_raw)
            if pd.isna(high_val) or not np.isfinite(high_val):
                continue
            
            # Get max value as float - ensure we get a scalar
            window_slice = high_series.iloc[i-window:i+window]
            if len(window_slice) == 0:
                continue
                
            window_max_raw = window_slice.max()
            if isinstance(window_max_raw, (pd.Series, pd.DataFrame, dict)):
                continue
            
            window_max = float(window_max_raw)
            if pd.isna(window_max) or not np.isfinite(window_max):
                continue
            
            if abs(high_val - window_max) < 0.000001:  # Use tolerance for float comparison
                # Check if it's a significant level
                if len(resistances) == 0:
                    resistances.append({
                        'price': float(high_val),
                        'index': int(i),
                        'strength': 1.0
                    })
                else:
                    # Extract last price safely - ensure it's always a float
                    last_item = resistances[-1]
                    last_price = None
                    
                    if isinstance(last_item, dict):
                        price_val = last_item.get('price')
                        if price_val is not None:
                            try:
                                last_price = float(price_val)
                                if not np.isfinite(last_price) or isinstance(last_price, (dict, list)):
                                    last_price = float(high_val)
                            except (TypeError, ValueError):
                                last_price = float(high_val)
                        else:
                            last_price = float(high_val)
                    elif isinstance(last_item, (int, float)):
                        try:
                            last_price = float(last_item)
                            if not np.isfinite(last_price) or isinstance(last_price, (dict, list)):
                                last_price = float(high_val)
                        except (TypeError, ValueError):
                            last_price = float(high_val)
                    else:
                        last_price = float(high_val)
                    
                    # Ensure last_price is valid before comparison
                    if last_price is None or not np.isfinite(last_price) or isinstance(last_price, (dict, list)):
                        last_price = float(high_val)
                    
                    try:
                        close_val_raw = close_series.iloc[i]
                        if isinstance(close_val_raw, (pd.Series, pd.DataFrame, dict, list)):
                            close_val = float(high_val)
                        else:
                            close_val = float(close_val_raw)
                        if pd.isna(close_val) or not np.isfinite(close_val):
                            close_val = float(high_val)
                    except:
                        close_val = float(high_val)
                    
                    # Final safety check - ensure all values are floats before arithmetic
                    if isinstance(last_price, (int, float)) and np.isfinite(last_price) and \
                       isinstance(high_val, (int, float)) and np.isfinite(high_val) and \
                       isinstance(close_val, (int, float)) and np.isfinite(close_val):
                        if abs(float(high_val) - float(last_price)) > float(close_val) * 0.001:
                            resistances.append({
                                'price': float(high_val),
                                'index': int(i),
                                'strength': 1.0
                            })
        except (ValueError, TypeError, IndexError) as e:
            continue  # Skip this iteration if there's an error
    
    # Sort by recency and return top levels
    supports = sorted(supports, key=lambda x: x['index'] if isinstance(x, dict) else 0, reverse=True)[:num_levels]
    resistances = sorted(resistances, key=lambda x: x['index'] if isinstance(x, dict) else 0, reverse=True)[:num_levels]
    
    return {
        'support': supports,
        'resistance': resistances
    }

def calculate_trend_line(df, period=30):
    """Calculate trend line using linear regression"""
    
    if len(df) < period:
        period = len(df)
    
    recent_df = df.tail(period)
    x_numeric = np.arange(len(recent_df))
    
    # Linear regression
    z = np.polyfit(x_numeric, recent_df['Close'].values, 1)
    trend_line_func = np.poly1d(z)
    
    # Calculate trend strength (R-squared)
    y_pred = trend_line_func(x_numeric)
    y_actual = recent_df['Close'].values
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Determine trend direction
    slope = z[0]
    trend_direction = 'UP' if slope > 0 else 'DOWN' if slope < 0 else 'SIDEWAYS'
    
    # Project future trend line
    future_x = np.arange(len(recent_df), len(recent_df) + 10)
    future_trend = trend_line_func(future_x)
    
    return {
        'trend_direction': trend_direction,
        'slope': slope,
        'strength': abs(r_squared),
        'current_trend_line': trend_line_func(np.arange(len(recent_df))),
        'future_projection': future_trend,
        'r_squared': r_squared
    }

def identify_market_structure(df, lookback=50):
    """Identify market structure (higher highs, lower lows, etc.)"""
    
    if len(df) < lookback:
        lookback = len(df)
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
    
    recent_df = df.tail(lookback)
    
    # Ensure columns are numeric
    for col in ['High', 'Low']:
        if col in recent_df.columns:
            recent_df[col] = pd.to_numeric(recent_df[col], errors='coerce')
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent_df) - 2):
        try:
            # Extract values as floats
            high_val = float(recent_df['High'].iloc[i])
            low_val = float(recent_df['Low'].iloc[i])
            
            if pd.isna(high_val) or pd.isna(low_val):
                continue
            
            # Swing high - ensure all comparisons are with floats
            if (i > 1 and i < len(recent_df) - 2):
                high_prev1 = float(recent_df['High'].iloc[i-1])
                high_prev2 = float(recent_df['High'].iloc[i-2]) if i >= 2 else high_val
                high_next1 = float(recent_df['High'].iloc[i+1]) if i < len(recent_df) - 1 else high_val
                high_next2 = float(recent_df['High'].iloc[i+2]) if i < len(recent_df) - 2 else high_val
                
                if not any(pd.isna([high_prev1, high_prev2, high_next1, high_next2])):
                    if (high_val > high_prev1 and 
                        high_val > high_prev2 and
                        high_val > high_next1 and
                        high_val > high_next2):
                        swing_highs.append({
                            'price': float(high_val),
                            'index': int(i)
                        })
            
            # Swing low - ensure all comparisons are with floats
            if (i > 1 and i < len(recent_df) - 2):
                low_prev1 = float(recent_df['Low'].iloc[i-1])
                low_prev2 = float(recent_df['Low'].iloc[i-2]) if i >= 2 else low_val
                low_next1 = float(recent_df['Low'].iloc[i+1]) if i < len(recent_df) - 1 else low_val
                low_next2 = float(recent_df['Low'].iloc[i+2]) if i < len(recent_df) - 2 else low_val
                
                if not any(pd.isna([low_prev1, low_prev2, low_next1, low_next2])):
                    if (low_val < low_prev1 and 
                        low_val < low_prev2 and
                        low_val < low_next1 and
                        low_val < low_next2):
                        swing_lows.append({
                            'price': float(low_val),
                            'index': int(i)
                        })
        except (ValueError, TypeError, IndexError):
            continue  # Skip this iteration
    
    # Determine market structure - ensure all prices are floats
    structure = 'UNKNOWN'
    current_swing_high = None
    current_swing_low = None
    
    try:
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Extract prices as floats
            recent_hh_prices = []
            for h in swing_highs[-2:]:
                if isinstance(h, dict):
                    price = h.get('price')
                    if price is not None:
                        try:
                            recent_hh_prices.append(float(price))
                        except (TypeError, ValueError):
                            pass
            
            previous_hh_prices = []
            if len(swing_highs) > 2:
                for h in swing_highs[:-2]:
                    if isinstance(h, dict):
                        price = h.get('price')
                        if price is not None:
                            try:
                                previous_hh_prices.append(float(price))
                            except (TypeError, ValueError):
                                pass
            
            recent_ll_prices = []
            for l in swing_lows[-2:]:
                if isinstance(l, dict):
                    price = l.get('price')
                    if price is not None:
                        try:
                            recent_ll_prices.append(float(price))
                        except (TypeError, ValueError):
                            pass
            
            previous_ll_prices = []
            if len(swing_lows) > 2:
                for l in swing_lows[:-2]:
                    if isinstance(l, dict):
                        price = l.get('price')
                        if price is not None:
                            try:
                                previous_ll_prices.append(float(price))
                            except (TypeError, ValueError):
                                pass
            
            if recent_hh_prices and recent_ll_prices:
                recent_hh = max(recent_hh_prices)
                previous_hh = max(previous_hh_prices) if previous_hh_prices else recent_hh
                
                recent_ll = min(recent_ll_prices)
                previous_ll = min(previous_ll_prices) if previous_ll_prices else recent_ll
                
                if recent_hh > previous_hh and recent_ll > previous_ll:
                    structure = 'BULLISH'  # Higher highs and higher lows
                elif recent_hh < previous_hh and recent_ll < previous_ll:
                    structure = 'BEARISH'  # Lower highs and lower lows
                else:
                    structure = 'MIXED'
        
        # Extract current swing high/low safely
        if swing_highs:
            last_high = swing_highs[-1]
            if isinstance(last_high, dict):
                price = last_high.get('price')
                if price is not None:
                    try:
                        current_swing_high = float(price)
                    except (TypeError, ValueError):
                        pass
        
        if swing_lows:
            last_low = swing_lows[-1]
            if isinstance(last_low, dict):
                price = last_low.get('price')
                if price is not None:
                    try:
                        current_swing_low = float(price)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass  # Keep defaults
    
    return {
        'structure': structure,
        'swing_highs': swing_highs[-5:],  # Last 5 swing highs
        'swing_lows': swing_lows[-5:],    # Last 5 swing lows
        'current_swing_high': current_swing_high,
        'current_swing_low': current_swing_low
    }

def calculate_fibonacci_levels(df, period=50):
    """Calculate Fibonacci retracement levels"""
    
    if len(df) < period:
        period = len(df)
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
    
    recent_df = df.tail(period)
    
    # Ensure we have Series, not DataFrames - convert to numeric
    try:
        high_series = pd.to_numeric(recent_df['High'], errors='coerce') if isinstance(recent_df['High'], pd.Series) else pd.to_numeric(recent_df['High'].iloc[:, 0], errors='coerce')
        low_series = pd.to_numeric(recent_df['Low'], errors='coerce') if isinstance(recent_df['Low'], pd.Series) else pd.to_numeric(recent_df['Low'].iloc[:, 0], errors='coerce')
    except Exception:
        # Fallback
        high_series = recent_df['High'].copy() if isinstance(recent_df['High'], pd.Series) else recent_df.iloc[:, recent_df.columns.get_loc('High')]
        low_series = recent_df['Low'].copy() if isinstance(recent_df['Low'], pd.Series) else recent_df.iloc[:, recent_df.columns.get_loc('Low')]
        high_series = pd.to_numeric(high_series, errors='coerce')
        low_series = pd.to_numeric(low_series, errors='coerce')
    
    # Get max/min as floats - ensure Series is numeric
    try:
        # Ensure high_series and low_series are Series and numeric
        if not isinstance(high_series, pd.Series):
            high_series = pd.Series(high_series) if hasattr(high_series, '__iter__') else pd.Series([float(high_series)])
        if not isinstance(low_series, pd.Series):
            low_series = pd.Series(low_series) if hasattr(low_series, '__iter__') else pd.Series([float(low_series)])
        
        # Get max/min values
        high_max = high_series.max()
        low_min = low_series.min()
        
        # Convert to float safely
        if isinstance(high_max, (pd.Series, pd.DataFrame)):
            high_max = float(high_max.iloc[0] if len(high_max) > 0 else recent_df['High'].iloc[-1])
        if isinstance(low_min, (pd.Series, pd.DataFrame)):
            low_min = float(low_min.iloc[0] if len(low_min) > 0 else recent_df['Low'].iloc[-1])
        
        high = float(high_max) if not pd.isna(high_max) else float(recent_df['High'].iloc[-1])
        low = float(low_min) if not pd.isna(low_min) else float(recent_df['Low'].iloc[-1])
    except Exception as e:
        # Last resort fallback - directly from DataFrame
        try:
            high_val = recent_df['High'].values
            low_val = recent_df['Low'].values
            high = float(np.nanmax(high_val))
            low = float(np.nanmin(low_val))
        except:
            # Ultimate fallback
            high = float(recent_df['High'].max()) if isinstance(recent_df['High'], pd.Series) else 2000.0
            low = float(recent_df['Low'].min()) if isinstance(recent_df['Low'], pd.Series) else 1800.0
    
    diff = high - low
    
    # Fibonacci levels
    fib_levels = {
        '0.0': high,
        '0.236': high - (diff * 0.236),
        '0.382': high - (diff * 0.382),
        '0.5': high - (diff * 0.5),
        '0.618': high - (diff * 0.618),
        '0.786': high - (diff * 0.786),
        '1.0': low
    }
    
    # Ensure Close is a Series
    close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    current_price = float(close_series.iloc[-1])
    
    # Find nearest Fibonacci level
    nearest_level = None
    min_distance = float('inf')
    for level, price in fib_levels.items():
        price_float = float(price) if price is not None else 0.0
        distance = abs(current_price - price_float)
        if distance < min_distance:
            min_distance = distance
            nearest_level = {'level': level, 'price': price_float}
    
    return {
        'levels': fib_levels,
        'range_high': high,
        'range_low': low,
        'nearest_level': nearest_level
    }

def comprehensive_smc_analysis(df):
    """Comprehensive SMC analysis combining all methods"""
    from data_validator import clean_dataframe_for_analysis, validate_dataframe
    
    # Clean and validate data first
    try:
        df = clean_dataframe_for_analysis(df)
        is_valid, error_msg = validate_dataframe(df, min_rows=20)
        if not is_valid:
            raise ValueError(f"Data validation failed: {error_msg}")
    except Exception as e:
        raise ValueError(f"Data cleaning error in SMC analysis: {str(e)}")
    
    # Ensure current_price is extracted safely
    try:
        current_price_raw = df['Close'].iloc[-1]
        if isinstance(current_price_raw, (pd.Series, pd.DataFrame, dict)):
            current_price = float(df['Close'].values[-1]) if len(df) > 0 else 0.0
        else:
            current_price = float(current_price_raw)
        if pd.isna(current_price) or not np.isfinite(current_price):
            current_price = float(df['Close'].values[-1]) if len(df) > 0 else 0.0
    except:
        current_price = float(df['Close'].values[-1]) if len(df) > 0 else 0.0
    
    # Run analyses with error handling
    try:
        support_resistance = identify_support_resistance(df)
    except Exception as e:
        support_resistance = {'support': [], 'resistance': []}
    
    try:
        trend_analysis = calculate_trend_line(df)
    except:
        trend_analysis = {'trend_direction': 'UNKNOWN', 'strength': 0.5}
    
    try:
        market_structure = identify_market_structure(df)
    except:
        market_structure = {'structure': 'UNKNOWN'}
    
    try:
        fibonacci = calculate_fibonacci_levels(df)
    except:
        fibonacci = {'levels': {}, 'nearest_level': None}
    
    # Ensure current_price is ALWAYS a float first, before any comparisons
    try:
        if isinstance(current_price, (pd.Series, pd.DataFrame, dict, list)):
            # If it's a Series/DataFrame, extract the value
            if isinstance(current_price, pd.Series) and len(current_price) > 0:
                current_price = float(current_price.iloc[-1])
            elif isinstance(current_price, pd.DataFrame) and len(current_price) > 0:
                current_price = float(current_price.iloc[-1, 0])
            else:
                current_price = 0.0
        else:
            current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0
    except (TypeError, ValueError, IndexError):
        current_price = 0.0
    
    # Ensure current_price is finite
    if not np.isfinite(current_price):
        current_price = 0.0
    
    # Nearest support - ensure it's a float, not dict
    nearest_support = None
    try:
        if support_resistance and isinstance(support_resistance, dict):
            support_list = support_resistance.get('support', [])
            if support_list and current_price > 0:
                supports_prices = []
                for s in support_list:
                    try:
                        if isinstance(s, dict):
                            price = s.get('price')
                            if price is not None:
                                price_float = float(price)
                                # Ensure price is a valid number before comparison
                                if np.isfinite(price_float) and isinstance(price_float, (int, float)) and price_float < current_price:
                                    supports_prices.append(price_float)
                        elif isinstance(s, (int, float)):
                            price_float = float(s)
                            if np.isfinite(price_float) and price_float < current_price:
                                supports_prices.append(price_float)
                    except (TypeError, ValueError, KeyError):
                        continue  # Skip invalid entries
                
                if supports_prices and len(supports_prices) > 0:
                    nearest_support = float(max(supports_prices))  # Closest support below price
    except Exception as e:
        nearest_support = None
    
    # Nearest resistance - ensure it's a float, not dict
    nearest_resistance = None
    try:
        if support_resistance and isinstance(support_resistance, dict):
            resistance_list = support_resistance.get('resistance', [])
            if resistance_list and current_price > 0:
                resistances_prices = []
                for r in resistance_list:
                    try:
                        if isinstance(r, dict):
                            price = r.get('price')
                            if price is not None:
                                price_float = float(price)
                                # Ensure price is a valid number before comparison
                                if np.isfinite(price_float) and isinstance(price_float, (int, float)) and price_float > current_price:
                                    resistances_prices.append(price_float)
                        elif isinstance(r, (int, float)):
                            price_float = float(r)
                            if np.isfinite(price_float) and price_float > current_price:
                                resistances_prices.append(price_float)
                    except (TypeError, ValueError, KeyError):
                        continue  # Skip invalid entries
                
                if resistances_prices and len(resistances_prices) > 0:
                    nearest_resistance = float(min(resistances_prices))  # Closest resistance above price
    except Exception as e:
        nearest_resistance = None
    
    return {
        'support_resistance': support_resistance if support_resistance else {'support': [], 'resistance': []},
        'trend': trend_analysis if trend_analysis else {'trend_direction': 'UNKNOWN', 'strength': 0.5},
        'market_structure': market_structure if market_structure else {'structure': 'UNKNOWN'},
        'fibonacci': fibonacci if fibonacci else {'levels': {}, 'nearest_level': None},
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'current_price': current_price
    }

def compare_user_vs_ai_smc(user_entry, user_sl, user_tp, ai_entry, ai_sl, ai_tp, smc_data, news_sentiment, ml_confidence):
    """Compare user analysis with AI analysis using SMC principles"""
    
    scores = {
        'entry_score': 0,
        'stop_loss_score': 0,
        'take_profit_score': 0,
        'smc_alignment': 0,
        'risk_reward_score': 0
    }
    
    total_max_score = 100
    
    # Entry score (30 points)
    entry_diff_pct = abs(user_entry - ai_entry) / ai_entry * 100 if ai_entry > 0 else 100
    if entry_diff_pct < 0.5:
        scores['entry_score'] = 30
    elif entry_diff_pct < 1.0:
        scores['entry_score'] = 25
    elif entry_diff_pct < 2.0:
        scores['entry_score'] = 20
    elif entry_diff_pct < 5.0:
        scores['entry_score'] = 10
    else:
        scores['entry_score'] = 5
    
    # Check if entry is near SMC levels
    nearest_support = smc_data.get('nearest_support')
    nearest_resistance = smc_data.get('nearest_resistance')
    
    if nearest_support and abs(user_entry - nearest_support) / nearest_support < 0.005:
        scores['entry_score'] += 5  # Bonus for SMC alignment
    if nearest_resistance and abs(user_entry - nearest_resistance) / nearest_resistance < 0.005:
        scores['entry_score'] += 5  # Bonus for SMC alignment
    
    # Stop Loss score (25 points)
    sl_diff_pct = abs(user_sl - ai_sl) / ai_sl * 100 if ai_sl > 0 else 100
    if sl_diff_pct < 1.0:
        scores['stop_loss_score'] = 25
    elif sl_diff_pct < 2.0:
        scores['stop_loss_score'] = 20
    elif sl_diff_pct < 5.0:
        scores['stop_loss_score'] = 15
    else:
        scores['stop_loss_score'] = 5
    
    # Take Profit score (25 points)
    tp_diff_pct = abs(user_tp - ai_tp) / ai_tp * 100 if ai_tp > 0 else 100
    if tp_diff_pct < 1.0:
        scores['take_profit_score'] = 25
    elif tp_diff_pct < 2.0:
        scores['take_profit_score'] = 20
    elif tp_diff_pct < 5.0:
        scores['take_profit_score'] = 15
    else:
        scores['take_profit_score'] = 5
    
    # Risk-Reward score (20 points)
    user_risk = abs(user_entry - user_sl)
    user_reward = abs(user_tp - user_entry)
    user_rr = user_reward / user_risk if user_risk > 0 else 0
    
    ai_risk = abs(ai_entry - ai_sl)
    ai_reward = abs(ai_tp - ai_entry)
    ai_rr = ai_reward / ai_risk if ai_risk > 0 else 0
    
    rr_diff = abs(user_rr - ai_rr)
    if rr_diff < 0.3:
        scores['risk_reward_score'] = 20
    elif user_rr >= 2.0:  # Good R:R ratio
        scores['risk_reward_score'] = 15
    else:
        scores['risk_reward_score'] = 10
    
    total_score = sum(scores.values())
    
    # Cap total score at 100 (in case bonuses push it over)
    total_score = min(100, total_score)
    
    # Calculate actual similarity percentage based on differences
    avg_diff = (entry_diff_pct + sl_diff_pct + tp_diff_pct) / 3
    # Convert difference to similarity (inverse relationship)
    # 0% diff = 100% similarity, 10% diff = 0% similarity
    actual_similarity_pct = max(0, 100 - (avg_diff * 10))
    
    # Use weighted average: 70% score-based, 30% actual difference-based
    similarity_pct = (total_score * 0.7) + (actual_similarity_pct * 0.3)
    similarity_pct = min(100, max(0, similarity_pct))  # Clamp 0-100
    
    # Determine winner
    if total_score >= 80:
        winner = "Both analyses are excellent and similar"
        recommendation = "AI"
    elif total_score >= 60:
        if ml_confidence > 0.7 and news_sentiment.get('compound', 0) != 0:
            winner = "AI (high confidence + news alignment)"
            recommendation = "AI"
        else:
            winner = "Both are reasonable"
            recommendation = "Similar"
    else:
        winner = "AI (based on ML model + technical + news)"
        recommendation = "AI"
    
    return {
        'total_score': total_score,
        'scores': scores,
        'winner': winner,
        'recommendation': recommendation,
        'similarity_pct': similarity_pct
    }


