"""
Pre-trained Model Manager - Loads and manages forex-specific ML models
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Forex symbols mapping - comprehensive list matching fetch_data.py
FOREX_SYMBOLS = {
    'XAUUSD': 'GC=F',      # Gold futures
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'JPY=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCAD': 'CAD=X',
    'USDCHF': 'CHF=X',
    'NZDUSD': 'NZDUSD=X',
    'EURGBP': 'EURGBP=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'AUDJPY': 'AUDJPY=X',
    'EURCHF': 'EURCHF=X',
    'GBPCHF': 'GBPCHF=X',
    'AUDNZD': 'AUDNZD=X',
    'EURAUD': 'EURAUD=X',
    'EURCAD': 'EURCAD=X',
    'GBPAUD': 'GBPAUD=X',
    'XAGUSD': 'SI=F',      # Silver futures
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

def get_model_path(symbol):
    """Get the path for a symbol-specific model"""
    symbol_clean = symbol.replace('=', '').replace('X', '').replace('F', '').replace('/', '')
    return os.path.join(MODELS_DIR, f'{symbol_clean}_model.pkl')

def load_pretrained_model(symbol):
    """Load a pre-trained model for a specific symbol"""
    model_path = get_model_path(symbol)
    
    if not os.path.exists(model_path):
        # Try to load generic forex model
        generic_path = os.path.join(MODELS_DIR, 'forex_model.pkl')
        if os.path.exists(generic_path):
            try:
                with open(generic_path, 'rb') as f:
                    generic_model = pickle.load(f)
                    # Check if it's old format (just model) or new format (dict)
                    if isinstance(generic_model, dict):
                        return generic_model
                    else:
                        # Old format - return None
                        return None
            except:
                pass
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            # Check if model_data is the old format (just a model) or new format (dict)
            if isinstance(model_data, dict):
                return model_data
            else:
                # Old format - return None to use new training
                return None
    except Exception as e:
        print(f"Error loading model for {symbol}: {e}")
        return None

def save_pretrained_model(symbol, model_data):
    """Save a trained model for a specific symbol"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = get_model_path(symbol)
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        return True
    except Exception as e:
        print(f"Error saving model for {symbol}: {e}")
        return False

def train_forex_model(df, symbol, retrain=False):
    """
    Train a comprehensive model specifically for forex pairs
    Enhanced with more features and better architecture
    """
    if not SKLEARN_AVAILABLE:
        return None
    
    if len(df) < 200:  # Need at least 200 data points
        return None
    
    try:
        # Import calculate_features dynamically to avoid circular imports
        import sys
        import os
        backend_path = os.path.join(os.path.dirname(__file__))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        from ml_model import calculate_features
        
        # Calculate comprehensive features
        features = calculate_features(df)
        if features is None or len(features) < 150:
            return None
        
        # Prepare targets
        # Direction prediction (binary)
        returns = df['Close'].pct_change(1).shift(-1).fillna(0).values[-len(features):]
        target_direction = (returns > 0).astype(int)
        
        # Magnitude prediction (regression)
        target_magnitude = returns
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx].values
        X_test = features.iloc[split_idx:].values
        y_train_dir = target_direction[:split_idx]
        y_test_dir = target_direction[split_idx:]
        y_train_mag = target_magnitude[:split_idx]
        y_test_mag = target_magnitude[split_idx:]
        
        # Train direction classifier (enhanced Random Forest)
        direction_model = RandomForestClassifier(
            n_estimators=400,  # More trees for better accuracy
            max_depth=20,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True  # Out-of-bag score for validation
        )
        direction_model.fit(X_train, y_train_dir)
        
        # Calculate accuracy
        train_acc = direction_model.score(X_train, y_train_dir)
        test_acc = direction_model.score(X_test, y_test_dir)
        oob_score = direction_model.oob_score_ if hasattr(direction_model, 'oob_score_') else None
        
        # Train magnitude regressor (enhanced Gradient Boosting)
        magnitude_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=10,
            learning_rate=0.08,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            loss='huber',  # Robust to outliers
            subsample=0.8  # Stochastic gradient boosting
        )
        magnitude_model.fit(X_train, y_train_mag)
        
        # Calculate regression metrics
        train_pred_mag = magnitude_model.predict(X_train)
        test_pred_mag = magnitude_model.predict(X_test)
        train_mae = np.mean(np.abs(train_pred_mag - y_train_mag))
        test_mae = np.mean(np.abs(test_pred_mag - y_test_mag))
        
        # Create model package
        model_data = {
            'direction_model': direction_model,
            'magnitude_model': magnitude_model,
            'symbol': symbol,
            'trained_date': datetime.now().isoformat(),
            'feature_names': list(features.columns),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'oob_score': oob_score,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'n_features': len(features.columns),
            'n_samples': len(features),
            'model_version': '2.0'  # Enhanced version
        }
        
        # Save model
        if save_pretrained_model(symbol, model_data):
            print(f"✅ Model trained and saved for {symbol}")
            print(f"   Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}")
            if oob_score:
                print(f"   OOB Score: {oob_score:.2%}")
            print(f"   Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            return model_data
        
        return None
        
    except Exception as e:
        print(f"Error training model for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_with_pretrained_model(df, symbol):
    """
    Make predictions using a pre-trained model or fallback to generic training
    """
    # Try to load pre-trained model
    model_data = load_pretrained_model(symbol)
    
    if model_data is None:
        # Fallback to on-the-fly training
        model_data = train_forex_model(df, symbol)
        if model_data is None:
            return None
    
    try:
        # Validate model_data format
        if not isinstance(model_data, dict):
            return None
        
        if 'direction_model' not in model_data or 'magnitude_model' not in model_data:
            return None
        
        # Import calculate_features dynamically to avoid circular imports
        import sys
        import os
        backend_path = os.path.join(os.path.dirname(__file__))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        from ml_model import calculate_features
        
        features = calculate_features(df)
        if features is None or len(features) == 0:
            return None
        
        # Get latest features
        X_latest = features.iloc[-1:].values
        
        # Get models from dict
        direction_model = model_data['direction_model']
        magnitude_model = model_data['magnitude_model']
        
        # Predict direction
        direction_proba = direction_model.predict_proba(X_latest)[0]
        direction_prob = direction_proba[1] if len(direction_proba) > 1 else 0.5  # Probability of UP
        
        # Predict magnitude
        predicted_change = magnitude_model.predict(X_latest)[0]
        
        # Determine direction
        if direction_prob > 0.6:
            direction = 'UP'
            confidence = direction_prob
        elif direction_prob < 0.4:
            direction = 'DOWN'
            confidence = 1 - direction_prob
        else:
            direction = 'NEUTRAL'
            confidence = 0.5
        
        # Enhance confidence based on model quality
        model_accuracy = model_data.get('test_accuracy', 0.5)
        confidence_boost = min(0.15, (model_accuracy - 0.5) * 0.3)  # Boost up to 15%
        enhanced_confidence = min(0.97, confidence + confidence_boost)
        
        return {
            'direction': direction,
            'predicted_change': predicted_change,
            'confidence': enhanced_confidence,
            'method': f'Pre-trained Model ({symbol})',
            'direction_probability': direction_prob,
            'model_accuracy': model_accuracy,
            'model_version': model_data.get('model_version', '1.0')
        }
        
    except Exception as e:
        print(f"Error making prediction with pretrained model: {e}")
        return None

def list_available_models():
    """List all available pre-trained models"""
    if not os.path.exists(MODELS_DIR):
        return []
    
    models = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('_model.pkl'):
            symbol = filename.replace('_model.pkl', '')
            model_path = os.path.join(MODELS_DIR, filename)
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict):
                        models.append({
                            'symbol': symbol,
                            'trained_date': model_data.get('trained_date', 'Unknown'),
                            'test_accuracy': model_data.get('test_accuracy', 0),
                            'model_version': model_data.get('model_version', '1.0'),
                            'n_features': model_data.get('n_features', 0)
                        })
            except:
                pass
    
    return models

