"""
Sultan AI Enhanced ML Model
===========================
Improved ML model with:
- Larger training data (2000+ candles)
- Time-series cross-validation
- Better feature engineering
- Regime detection
- Calibrated confidence scores
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configuration
DEFAULT_TRAINING_SAMPLES = 2000  # 20x more than before
MIN_TRAINING_SAMPLES = 500
VALIDATION_SPLIT = 0.2
N_CV_FOLDS = 5


def calculate_features_enhanced(df: pd.DataFrame, n_samples: int = None) -> Optional[pd.DataFrame]:
    """
    Calculate comprehensive features for ML model with configurable sample size.

    Args:
        df: DataFrame with OHLCV and indicator data
        n_samples: Number of samples to use (default: all available)

    Returns:
        DataFrame with features or None if insufficient data
    """
    if n_samples is None:
        n_samples = len(df)

    min_required = max(100, n_samples)
    if len(df) < min_required:
        return None

    # Use the last n_samples
    df_subset = df.tail(n_samples).copy()
    n = len(df_subset)

    features = pd.DataFrame(index=df_subset.index)

    # ============================================================
    # PRICE-BASED FEATURES
    # ============================================================

    # Normalized price features (relative to recent range)
    recent_high = df_subset['High'].rolling(50).max()
    recent_low = df_subset['Low'].rolling(50).min()
    price_range = recent_high - recent_low

    features['price_position'] = ((df_subset['Close'] - recent_low) / price_range).fillna(0.5)
    features['high_position'] = ((df_subset['High'] - recent_low) / price_range).fillna(0.5)
    features['low_position'] = ((df_subset['Low'] - recent_low) / price_range).fillna(0.5)

    # Returns at multiple timeframes
    for period in [1, 2, 3, 5, 10, 20]:
        features[f'return_{period}'] = df_subset['Close'].pct_change(period).fillna(0)

    # Cumulative returns
    features['return_cumulative_5'] = df_subset['Close'].pct_change(5).rolling(5).sum().fillna(0)
    features['return_cumulative_20'] = df_subset['Close'].pct_change(20).rolling(20).sum().fillna(0)

    # ============================================================
    # VOLATILITY FEATURES
    # ============================================================

    # Rolling volatility at multiple windows
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = df_subset['Close'].pct_change().rolling(window).std().fillna(0)

    # Volatility ratio (short-term vs long-term)
    features['volatility_ratio'] = (features['volatility_5'] / features['volatility_20']).fillna(1).replace([np.inf, -np.inf], 1)

    # ATR-based volatility
    if 'ATR' in df_subset.columns:
        features['atr_pct'] = (df_subset['ATR'] / df_subset['Close']).fillna(0)
        features['atr_ratio'] = (df_subset['ATR'] / df_subset['ATR'].rolling(20).mean()).fillna(1)
    else:
        # Calculate ATR manually
        high_low = df_subset['High'] - df_subset['Low']
        high_close = abs(df_subset['High'] - df_subset['Close'].shift())
        low_close = abs(df_subset['Low'] - df_subset['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean()
        features['atr_pct'] = (atr / df_subset['Close']).fillna(0)
        features['atr_ratio'] = (atr / atr.rolling(20).mean()).fillna(1)

    # Bollinger Band width (volatility indicator)
    if 'BB_Upper' in df_subset.columns and 'BB_Lower' in df_subset.columns:
        bb_width = (df_subset['BB_Upper'] - df_subset['BB_Lower']) / df_subset['Close']
        features['bb_width'] = bb_width.fillna(0)
        features['bb_width_ratio'] = (bb_width / bb_width.rolling(20).mean()).fillna(1)

        bb_mid = (df_subset['BB_Upper'] + df_subset['BB_Lower']) / 2
        features['bb_position'] = ((df_subset['Close'] - df_subset['BB_Lower']) /
                                   (df_subset['BB_Upper'] - df_subset['BB_Lower'])).fillna(0.5)
    else:
        features['bb_width'] = 0
        features['bb_width_ratio'] = 1
        features['bb_position'] = 0.5

    # ============================================================
    # TREND FEATURES
    # ============================================================

    # Moving average features
    if 'SMA_20' in df_subset.columns:
        features['price_sma20_ratio'] = (df_subset['Close'] / df_subset['SMA_20']).fillna(1)
        features['sma20_slope'] = df_subset['SMA_20'].pct_change(5).fillna(0)
    else:
        sma20 = df_subset['Close'].rolling(20).mean()
        features['price_sma20_ratio'] = (df_subset['Close'] / sma20).fillna(1)
        features['sma20_slope'] = sma20.pct_change(5).fillna(0)

    if 'SMA_50' in df_subset.columns:
        features['price_sma50_ratio'] = (df_subset['Close'] / df_subset['SMA_50']).fillna(1)
        features['sma50_slope'] = df_subset['SMA_50'].pct_change(5).fillna(0)
    else:
        sma50 = df_subset['Close'].rolling(50).mean()
        features['price_sma50_ratio'] = (df_subset['Close'] / sma50).fillna(1)
        features['sma50_slope'] = sma50.pct_change(5).fillna(0)

    # MA crossover features
    if 'SMA_20' in df_subset.columns and 'SMA_50' in df_subset.columns:
        features['ma_spread'] = ((df_subset['SMA_20'] - df_subset['SMA_50']) / df_subset['Close']).fillna(0)
        features['ma_cross'] = (df_subset['SMA_20'] > df_subset['SMA_50']).astype(int)
    else:
        features['ma_spread'] = 0
        features['ma_cross'] = 0

    # EMA features
    if 'EMA_12' in df_subset.columns and 'EMA_26' in df_subset.columns:
        features['ema_spread'] = ((df_subset['EMA_12'] - df_subset['EMA_26']) / df_subset['Close']).fillna(0)
        features['ema_cross'] = (df_subset['EMA_12'] > df_subset['EMA_26']).astype(int)
    else:
        ema12 = df_subset['Close'].ewm(span=12).mean()
        ema26 = df_subset['Close'].ewm(span=26).mean()
        features['ema_spread'] = ((ema12 - ema26) / df_subset['Close']).fillna(0)
        features['ema_cross'] = (ema12 > ema26).astype(int)

    # ============================================================
    # MOMENTUM FEATURES
    # ============================================================

    # RSI
    if 'RSI' in df_subset.columns:
        features['rsi'] = df_subset['RSI'].fillna(50) / 100  # Normalize to 0-1
        features['rsi_oversold'] = (df_subset['RSI'] < 30).astype(int)
        features['rsi_overbought'] = (df_subset['RSI'] > 70).astype(int)
    else:
        # Calculate RSI
        delta = df_subset['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features['rsi'] = rsi.fillna(50) / 100
        features['rsi_oversold'] = (rsi < 30).astype(int)
        features['rsi_overbought'] = (rsi > 70).astype(int)

    # MACD
    if 'MACD' in df_subset.columns and 'MACD_Signal' in df_subset.columns:
        features['macd_normalized'] = (df_subset['MACD'] / df_subset['Close']).fillna(0)
        features['macd_signal_normalized'] = (df_subset['MACD_Signal'] / df_subset['Close']).fillna(0)
        features['macd_histogram'] = ((df_subset['MACD'] - df_subset['MACD_Signal']) / df_subset['Close']).fillna(0)
        features['macd_cross'] = (df_subset['MACD'] > df_subset['MACD_Signal']).astype(int)
    else:
        ema12 = df_subset['Close'].ewm(span=12).mean()
        ema26 = df_subset['Close'].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        features['macd_normalized'] = (macd / df_subset['Close']).fillna(0)
        features['macd_signal_normalized'] = (macd_signal / df_subset['Close']).fillna(0)
        features['macd_histogram'] = ((macd - macd_signal) / df_subset['Close']).fillna(0)
        features['macd_cross'] = (macd > macd_signal).astype(int)

    # Stochastic
    if 'Stoch_K' in df_subset.columns:
        features['stoch_k'] = df_subset['Stoch_K'].fillna(50) / 100
        features['stoch_d'] = df_subset['Stoch_D'].fillna(50) / 100 if 'Stoch_D' in df_subset.columns else features['stoch_k']
    else:
        low_14 = df_subset['Low'].rolling(14).min()
        high_14 = df_subset['High'].rolling(14).max()
        stoch_k = 100 * (df_subset['Close'] - low_14) / (high_14 - low_14)
        features['stoch_k'] = stoch_k.fillna(50) / 100
        features['stoch_d'] = stoch_k.rolling(3).mean().fillna(50) / 100

    # Momentum indicators
    for period in [5, 10, 20]:
        features[f'momentum_{period}'] = (df_subset['Close'] / df_subset['Close'].shift(period) - 1).fillna(0)
        features[f'roc_{period}'] = ((df_subset['Close'] - df_subset['Close'].shift(period)) /
                                      df_subset['Close'].shift(period)).fillna(0)

    # ============================================================
    # VOLUME FEATURES (if available)
    # ============================================================

    if 'Volume' in df_subset.columns and df_subset['Volume'].sum() > 0:
        volume_ma = df_subset['Volume'].rolling(20).mean()
        features['volume_ratio'] = (df_subset['Volume'] / volume_ma).fillna(1)
        features['volume_trend'] = df_subset['Volume'].pct_change(5).fillna(0)

        # Volume-price relationship
        price_change = df_subset['Close'].pct_change()
        features['volume_price_corr'] = price_change.rolling(20).corr(df_subset['Volume'].pct_change()).fillna(0)
    else:
        features['volume_ratio'] = 1
        features['volume_trend'] = 0
        features['volume_price_corr'] = 0

    # ============================================================
    # REGIME FEATURES
    # ============================================================

    # Trend regime (bull/bear/sideways)
    returns_20 = df_subset['Close'].pct_change(20).fillna(0)
    features['trend_regime'] = np.where(returns_20 > 0.02, 1, np.where(returns_20 < -0.02, -1, 0))

    # Volatility regime (high/normal/low)
    vol_20 = df_subset['Close'].pct_change().rolling(20).std()
    vol_ma = vol_20.rolling(50).mean()
    features['vol_regime'] = np.where(vol_20 > vol_ma * 1.5, 1, np.where(vol_20 < vol_ma * 0.5, -1, 0)).astype(float).fillna(0)

    # Range-bound detection
    price_range_pct = (recent_high - recent_low) / df_subset['Close']
    features['range_bound'] = (price_range_pct < price_range_pct.rolling(50).mean()).astype(int)

    # ============================================================
    # CANDLESTICK PATTERNS
    # ============================================================

    # Body size relative to range
    body_size = abs(df_subset['Close'] - df_subset['Open'])
    candle_range = df_subset['High'] - df_subset['Low']
    features['body_ratio'] = (body_size / candle_range).fillna(0.5)

    # Bullish/bearish candle
    features['candle_direction'] = (df_subset['Close'] > df_subset['Open']).astype(int)

    # Upper/lower wick ratios
    upper_wick = df_subset['High'] - df_subset[['Close', 'Open']].max(axis=1)
    lower_wick = df_subset[['Close', 'Open']].min(axis=1) - df_subset['Low']
    features['upper_wick_ratio'] = (upper_wick / candle_range).fillna(0)
    features['lower_wick_ratio'] = (lower_wick / candle_range).fillna(0)

    # Handle infinities and NaNs
    features = features.replace([np.inf, -np.inf], 0)
    features = features.fillna(0)

    return features


def train_enhanced_model(df: pd.DataFrame,
                         n_samples: int = DEFAULT_TRAINING_SAMPLES,
                         use_cv: bool = True,
                         calibrate: bool = True) -> Optional[Dict]:
    """
    Train enhanced ensemble model with time-series cross-validation.

    Args:
        df: DataFrame with OHLCV and indicator data
        n_samples: Number of samples to use for training
        use_cv: Use time-series cross-validation
        calibrate: Calibrate probability outputs

    Returns:
        Dictionary with trained model components and metrics
    """
    if not SKLEARN_AVAILABLE:
        return None

    # Calculate features
    features = calculate_features_enhanced(df, n_samples)
    if features is None or len(features) < MIN_TRAINING_SAMPLES:
        return None

    try:
        # Prepare target (next period return direction)
        # Use forward-looking target (no leakage)
        future_returns = df['Close'].pct_change(1).shift(-1)
        target = (future_returns > 0).astype(int)

        # Align features and target
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx].values
        y = target.loc[common_idx].values

        if len(X) < MIN_TRAINING_SAMPLES:
            return None

        # Train/validation split (time-series aware)
        split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Initialize models
        models = []
        model_weights = []
        cv_scores = []

        # Model 1: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            oob_score=True
        )

        # Model 2: Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )

        # Model 3: Extra Trees
        et_model = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=18,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=123,
            n_jobs=-1,
            class_weight='balanced'
        )

        # Time-series cross-validation
        if use_cv:
            tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)

            for name, model in [('rf', rf_model), ('gb', gb_model), ('et', et_model)]:
                fold_scores = []

                for train_idx, val_idx in tscv.split(X_train_scaled):
                    X_fold_train = X_train_scaled[train_idx]
                    y_fold_train = y_train[train_idx]
                    X_fold_val = X_train_scaled[val_idx]
                    y_fold_val = y_train[val_idx]

                    model_copy = type(model)(**model.get_params())
                    model_copy.fit(X_fold_train, y_fold_train)
                    score = model_copy.score(X_fold_val, y_fold_val)
                    fold_scores.append(score)

                cv_scores.append({
                    'name': name,
                    'mean_cv_score': np.mean(fold_scores),
                    'std_cv_score': np.std(fold_scores),
                    'fold_scores': fold_scores
                })

        # Train final models on full training set
        rf_model.fit(X_train_scaled, y_train)
        rf_val_score = rf_model.score(X_val_scaled, y_val)
        models.append(('rf', rf_model, scaler))
        model_weights.append(max(0.3, rf_val_score))

        gb_model.fit(X_train_scaled, y_train)
        gb_val_score = gb_model.score(X_val_scaled, y_val)
        models.append(('gb', gb_model, scaler))
        model_weights.append(max(0.25, gb_val_score))

        et_model.fit(X_train_scaled, y_train)
        et_val_score = et_model.score(X_val_scaled, y_val)
        models.append(('et', et_model, scaler))
        model_weights.append(max(0.2, et_val_score))

        # Normalize weights
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]

        # Calculate validation metrics
        ensemble_probs = np.zeros(len(X_val))
        for (name, model, _), weight in zip(models, model_weights):
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_val_scaled)[:, 1]
            else:
                probs = model.predict(X_val_scaled)
            ensemble_probs += probs * weight

        ensemble_preds = (ensemble_probs > 0.5).astype(int)

        val_metrics = {
            'accuracy': accuracy_score(y_val, ensemble_preds),
            'precision': precision_score(y_val, ensemble_preds, zero_division=0),
            'recall': recall_score(y_val, ensemble_preds, zero_division=0),
            'f1': f1_score(y_val, ensemble_preds, zero_division=0)
        }

        # Train magnitude regressor
        regressor = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            min_samples_split=5,
            random_state=42,
            loss='huber'
        )

        # Target: actual return magnitude
        y_reg = future_returns.loc[common_idx[:split_idx]].fillna(0).values
        regressor.fit(X_train_scaled, y_reg)

        # Feature importance
        feature_names = features.columns.tolist()
        rf_importance = rf_model.feature_importances_
        top_features = sorted(zip(feature_names, rf_importance), key=lambda x: x[1], reverse=True)[:10]

        return {
            'models': models,
            'weights': model_weights,
            'scaler': scaler,
            'regressor': regressor,
            'type': 'enhanced_ensemble',
            'feature_names': feature_names,
            'cv_scores': cv_scores if use_cv else None,
            'val_metrics': val_metrics,
            'top_features': top_features,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'trained_at': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Enhanced model training error: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_with_enhanced_model(df: pd.DataFrame,
                                 model_dict: Dict,
                                 symbol: str = None) -> Dict:
    """
    Make prediction using enhanced model.

    Args:
        df: DataFrame with current market data
        model_dict: Trained model dictionary
        symbol: Trading symbol

    Returns:
        Prediction dictionary with direction, confidence, etc.
    """
    if model_dict is None:
        return {
            'direction': 'NEUTRAL',
            'confidence': 0.5,
            'predicted_change': 0.0,
            'method': 'No model available'
        }

    try:
        # Calculate features for latest data
        features = calculate_features_enhanced(df, n_samples=100)
        if features is None or len(features) == 0:
            return {
                'direction': 'NEUTRAL',
                'confidence': 0.5,
                'predicted_change': 0.0,
                'method': 'Insufficient features'
            }

        # Get latest features
        X_latest = features.iloc[-1:].values

        # Scale features
        scaler = model_dict['scaler']
        X_scaled = scaler.transform(X_latest)

        # Ensemble prediction
        ensemble_prob = 0.0
        models = model_dict['models']
        weights = model_dict['weights']

        individual_probs = []

        for (name, model, _), weight in zip(models, weights):
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_scaled)[0]
                prob_up = prob[1] if len(prob) > 1 else 0.5
            else:
                pred = model.predict(X_scaled)[0]
                prob_up = 1.0 if pred > 0 else 0.0

            individual_probs.append({'model': name, 'prob': prob_up, 'weight': weight})
            ensemble_prob += prob_up * weight

        # Predict magnitude
        regressor = model_dict.get('regressor')
        if regressor is not None:
            predicted_change = regressor.predict(X_scaled)[0]
            predicted_change = np.clip(predicted_change, -0.05, 0.05)
        else:
            predicted_change = (ensemble_prob - 0.5) * 0.02

        # Calculate direction
        if ensemble_prob > 0.6:
            direction = 'UP'
        elif ensemble_prob < 0.4:
            direction = 'DOWN'
        else:
            direction = 'NEUTRAL'

        # Calculate calibrated confidence
        base_confidence = max(ensemble_prob, 1 - ensemble_prob)

        # Model agreement bonus
        prob_std = np.std([p['prob'] for p in individual_probs])
        agreement_factor = 1.0 - min(prob_std * 2, 0.2)  # Lower std = higher agreement

        # Validation accuracy adjustment
        val_accuracy = model_dict.get('val_metrics', {}).get('accuracy', 0.5)
        accuracy_factor = 0.8 + (val_accuracy - 0.5) * 0.4  # Scale 0.5-0.7 -> 0.8-0.88

        # Final confidence
        confidence = base_confidence * agreement_factor * accuracy_factor
        confidence = min(0.90, max(0.15, confidence))  # Cap at 90% max

        return {
            'direction': direction,
            'direction_probability': ensemble_prob,
            'confidence': confidence,
            'predicted_change': predicted_change,
            'method': 'Enhanced Ensemble ML',
            'individual_predictions': individual_probs,
            'model_agreement': 1.0 - prob_std,
            'val_accuracy': val_accuracy
        }

    except Exception as e:
        print(f"Enhanced prediction error: {e}")
        return {
            'direction': 'NEUTRAL',
            'confidence': 0.5,
            'predicted_change': 0.0,
            'method': f'Error: {str(e)}'
        }


def detect_market_regime(df: pd.DataFrame) -> Dict:
    """
    Detect current market regime (trend, volatility, etc.)

    Returns:
        Dictionary with regime information
    """
    if len(df) < 50:
        return {
            'trend': 'unknown',
            'volatility': 'unknown',
            'strength': 0.5
        }

    close = df['Close']

    # Trend detection
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    current = close.iloc[-1]

    return_20d = (current / close.iloc[-20] - 1) if len(close) >= 20 else 0

    if return_20d > 0.03 and current > sma20 > sma50:
        trend = 'strong_bull'
        trend_strength = min(1.0, return_20d * 10)
    elif return_20d > 0.01 and current > sma20:
        trend = 'bull'
        trend_strength = 0.6 + return_20d * 5
    elif return_20d < -0.03 and current < sma20 < sma50:
        trend = 'strong_bear'
        trend_strength = min(1.0, abs(return_20d) * 10)
    elif return_20d < -0.01 and current < sma20:
        trend = 'bear'
        trend_strength = 0.6 + abs(return_20d) * 5
    else:
        trend = 'sideways'
        trend_strength = 0.3

    # Volatility detection
    vol_20 = close.pct_change().rolling(20).std().iloc[-1]
    vol_50_avg = close.pct_change().rolling(50).std().mean()

    if vol_20 > vol_50_avg * 1.5:
        volatility = 'high'
        vol_strength = min(1.0, vol_20 / vol_50_avg - 1)
    elif vol_20 < vol_50_avg * 0.5:
        volatility = 'low'
        vol_strength = 0.3
    else:
        volatility = 'normal'
        vol_strength = 0.5

    return {
        'trend': trend,
        'trend_strength': trend_strength,
        'volatility': volatility,
        'volatility_level': vol_20,
        'volatility_ratio': vol_20 / vol_50_avg if vol_50_avg > 0 else 1,
        'regime_confidence': (trend_strength + vol_strength) / 2
    }


def save_enhanced_model(model_dict: Dict, filepath: str):
    """Save enhanced model to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)


def load_enhanced_model(filepath: str) -> Optional[Dict]:
    """Load enhanced model from file"""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except:
        return None


# Integration with existing system
def get_or_train_enhanced_model(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    Get existing enhanced model or train new one.

    This function integrates with the existing model system while providing
    the enhanced training capabilities.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models', 'enhanced')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f'{symbol}_enhanced.pkl')

    # Check for existing model
    model = load_enhanced_model(model_path)

    # Retrain if model is old (> 7 days) or doesn't exist
    should_retrain = model is None
    if model and 'trained_at' in model:
        try:
            trained_at = datetime.fromisoformat(model['trained_at'])
            days_old = (datetime.now() - trained_at).days
            if days_old > 7:
                should_retrain = True
        except:
            should_retrain = True

    if should_retrain and len(df) >= MIN_TRAINING_SAMPLES:
        print(f"Training enhanced model for {symbol}...")
        model = train_enhanced_model(df, n_samples=min(len(df), DEFAULT_TRAINING_SAMPLES))

        if model:
            save_enhanced_model(model, model_path)
            print(f"Enhanced model saved for {symbol}")
            print(f"  - Training samples: {model['training_samples']}")
            print(f"  - Validation accuracy: {model['val_metrics']['accuracy']:.2%}")

    return model


if __name__ == '__main__':
    # Test the enhanced model
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loader import load_csv_robust
    from analytics import calculate_advanced_indicators

    # Load test data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    test_file = os.path.join(data_dir, 'EURUSD.csv')

    if os.path.exists(test_file):
        print("Loading test data...")
        df = load_csv_robust(test_file)
        df = calculate_advanced_indicators(df)

        print(f"Data shape: {df.shape}")

        # Train model
        print("\nTraining enhanced model...")
        model = train_enhanced_model(df, n_samples=2000)

        if model:
            print("\nModel trained successfully!")
            print(f"Training samples: {model['training_samples']}")
            print(f"Validation metrics: {model['val_metrics']}")
            print(f"\nTop features:")
            for name, importance in model['top_features']:
                print(f"  {name}: {importance:.4f}")

            # Test prediction
            print("\nTesting prediction...")
            prediction = predict_with_enhanced_model(df, model, 'EURUSD')
            print(f"Prediction: {prediction}")

            # Detect regime
            print("\nMarket regime:")
            regime = detect_market_regime(df)
            print(f"  Trend: {regime['trend']} (strength: {regime['trend_strength']:.2f})")
            print(f"  Volatility: {regime['volatility']}")
        else:
            print("Model training failed!")
    else:
        print(f"Test file not found: {test_file}")
