import pandas as pd
import numpy as np
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
import pickle
import os
from datetime import datetime

def calculate_features(df):
    """Calculate comprehensive features for ML model"""
    if len(df) < 100:
        return None
    
    features = pd.DataFrame()
    
    # Price features
    features['close'] = df['Close'].values[-100:]
    features['high'] = df['High'].values[-100:]
    features['low'] = df['Low'].values[-100:]
    features['open'] = df['Open'].values[-100:]
    
    # Returns
    features['returns_1'] = df['Close'].pct_change(1).fillna(0).values[-100:]
    features['returns_5'] = df['Close'].pct_change(5).fillna(0).values[-100:]
    features['returns_10'] = df['Close'].pct_change(10).fillna(0).values[-100:]
    
    # Volatility
    features['volatility_5'] = df['Close'].rolling(5).std().fillna(0).values[-100:]
    features['volatility_20'] = df['Close'].rolling(20).std().fillna(0).values[-100:]
    
    # Technical indicators
    if 'RSI' in df.columns:
        features['rsi'] = df['RSI'].fillna(50).values[-100:]
    else:
        features['rsi'] = np.full(100, 50)
    
    if 'MACD' in df.columns:
        features['macd'] = df['MACD'].fillna(0).values[-100:]
        features['macd_signal'] = df['MACD_Signal'].fillna(0).values[-100:]
    else:
        features['macd'] = np.zeros(100)
        features['macd_signal'] = np.zeros(100)
    
    if 'SMA_20' in df.columns:
        features['sma_20'] = df['SMA_20'].fillna(df['Close']).values[-100:]
    else:
        features['sma_20'] = df['Close'].values[-100:]
    
    if 'SMA_50' in df.columns:
        features['sma_50'] = df['SMA_50'].fillna(df['Close']).values[-100:]
    else:
        features['sma_50'] = df['Close'].values[-100:]
    
    # Price position relative to moving averages
    features['price_sma20_ratio'] = (df['Close'] / df['SMA_20']).fillna(1).values[-100:] if 'SMA_20' in df.columns else np.ones(100)
    features['price_sma50_ratio'] = (df['Close'] / df['SMA_50']).fillna(1).values[-100:] if 'SMA_50' in df.columns else np.ones(100)
    
    # Bollinger Bands features
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        bb_mid = (df['BB_Upper'] + df['BB_Lower']) / 2
        bb_width = df['BB_Upper'] - df['BB_Lower']
        features['bb_position'] = ((df['Close'] - df['BB_Lower']) / bb_width).fillna(0.5).values[-100:]
        features['bb_width'] = (bb_width / df['Close']).fillna(0).values[-100:]
    else:
        features['bb_position'] = np.full(100, 0.5)
        features['bb_width'] = np.zeros(100)
    
    # Volume features (if available)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        features['volume'] = df['Volume'].fillna(0).values[-100:]
        features['volume_ma'] = df['Volume'].rolling(20).mean().fillna(0).values[-100:]
        features['volume_ratio'] = (df['Volume'] / features['volume_ma']).fillna(1).values[-100:]
    else:
        features['volume'] = np.zeros(100)
        features['volume_ma'] = np.zeros(100)
        features['volume_ratio'] = np.ones(100)
    
    # Momentum
    features['momentum_5'] = (df['Close'] / df['Close'].shift(5) - 1).fillna(0).values[-100:]
    features['momentum_10'] = (df['Close'] / df['Close'].shift(10) - 1).fillna(0).values[-100:]
    features['momentum_20'] = (df['Close'] / df['Close'].shift(20) - 1).fillna(0).values[-100:]
    
    # Additional advanced features
    # Rate of Change
    features['roc_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).fillna(0).values[-100:]
    features['roc_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)).fillna(0).values[-100:]
    
    # Stochastic if available
    if 'Stoch_K' in df.columns:
        features['stoch_k'] = df['Stoch_K'].fillna(50).values[-100:]
        features['stoch_d'] = df['Stoch_D'].fillna(50).values[-100:]
    else:
        features['stoch_k'] = np.full(100, 50)
        features['stoch_d'] = np.full(100, 50)
    
    # ATR-based volatility
    if 'ATR' in df.columns:
        atr_pct = (df['ATR'] / df['Close']).fillna(0).values[-100:]
        features['atr_pct'] = atr_pct
    else:
        features['atr_pct'] = np.zeros(100)
    
    # Price position within recent range
    recent_high = df['High'].rolling(20).max()
    recent_low = df['Low'].rolling(20).min()
    price_position = ((df['Close'] - recent_low) / (recent_high - recent_low)).fillna(0.5)
    features['price_position'] = price_position.values[-100:]
    
    # Trend strength (distance between MAs)
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        ma_spread = abs(df['SMA_20'] - df['SMA_50']) / df['Close']
        features['ma_spread'] = ma_spread.fillna(0).values[-100:]
    else:
        features['ma_spread'] = np.zeros(100)
    
    return features

def train_ensemble_model(df):
    """Train ensemble model combining multiple algorithms for better accuracy"""
    if not SKLEARN_AVAILABLE:
        return None
    
    features = calculate_features(df)
    if features is None:
        return None
    
    try:
        # Prepare target (next period return)
        target = df['Close'].pct_change(1).shift(-1).fillna(0).values[-100:]
        target_direction = (target > 0).astype(int)  # Binary classification
        
        # Use last 80 for training, last 20 for validation
        X_train = features.iloc[:-20].values
        y_train = target_direction[:-20]
        X_test = features.iloc[-20:].values
        y_test = target_direction[-20:]
        
        # Normalize features for better convergence
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble of models
        models = []
        model_weights = []
        
        # Model 1: Enhanced Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=400,  # Increased from 300 to 400
            max_depth=18,  # Deeper trees for complex patterns
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True  # Out-of-bag scoring for better validation
        )
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test_scaled, y_test) if len(X_test) > 0 else 0.55
        models.append(('rf', rf_model, scaler))
        model_weights.append(max(0.3, rf_score))  # Weight based on validation accuracy
        
        # Model 2: Enhanced Gradient Boosting Classifier
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            gb_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,  # Lower learning rate for better generalization
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                subsample=0.8  # Stochastic gradient boosting
            )
            gb_model.fit(X_train_scaled, y_train)
            gb_score = gb_model.score(X_test_scaled, y_test) if len(X_test) > 0 else 0.53
            models.append(('gb', gb_model, scaler))
            model_weights.append(max(0.25, gb_score))
        except:
            pass
        
        # Model 3: Extra Trees Classifier (more randomness)
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            et_model = ExtraTreesClassifier(
                n_estimators=300,
                max_depth=16,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=123,  # Different seed for diversity
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True
            )
            et_model.fit(X_train, y_train)
            et_score = et_model.score(X_test_scaled, y_test) if len(X_test) > 0 else 0.52
            models.append(('et', et_model, scaler))
            model_weights.append(max(0.2, et_score))
        except:
            pass
        
        # Normalize weights
        total_weight = sum(model_weights)
        if total_weight > 0:
            model_weights = [w / total_weight for w in model_weights]
        
        # Return ensemble (dict with models, weights, and scaler)
        return {
            'models': models,
            'weights': model_weights,
            'scaler': scaler,
            'type': 'ensemble'
        }
        
    except Exception as e:
        print(f"Ensemble model training error: {e}")
        # Fallback to simple Random Forest
        return train_simple_model_fallback(df, features, target_direction)

def train_simple_model_fallback(df, features, target_direction):
    """Fallback to simple Random Forest if ensemble fails"""
    try:
        X_train = features.iloc[:-20].values
        y_train = target_direction[:-20]
        
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        return {'models': [('rf', model, None)], 'weights': [1.0], 'scaler': None, 'type': 'single'}
    except:
        return None

def train_simple_model(df):
    """Train a simple model if no pretrained model exists - now uses ensemble"""
    return train_ensemble_model(df)

def predict_price_movement(df, symbol=None):
    """Predict price movement using ML model with pre-trained forex models"""
    if len(df) < 100:
        return {
            'direction': 'NEUTRAL',
            'predicted_change': 0.0,
            'confidence': 0.5,
            'method': 'Insufficient data'
        }
    
    # Try to get symbol from df attributes or parameter
    if symbol is None:
        symbol = df.attrs.get('symbol', 'EURUSD') if hasattr(df, 'attrs') else 'EURUSD'
    
    # Try pre-trained model first (enhanced)
    try:
        from pretrained_models import predict_with_pretrained_model, FOREX_SYMBOLS
        
        # Map symbol to standard format
        symbol_clean = symbol.replace('=', '').replace('X', '').replace('F', '').replace('/', '')
        
        # Try pre-trained model
        pretrained_result = predict_with_pretrained_model(df, symbol_clean)
        if pretrained_result:
            return pretrained_result
    except Exception as e:
        # Fallback to old method
        pass
    
    # Fallback: Check for old pretrained model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'models', 'forex_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model = None
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except:
            pass
    
    # If no model, train a simple one
    if model is None:
        model = train_simple_model(df)
        if model is not None:
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            except:
                pass
    
    if model is None:
        # Fallback to rule-based prediction
        return predict_rule_based(df)
    
    # Prepare features
    features = calculate_features(df)
    if features is None:
        return predict_rule_based(df)
    
    # Get latest features
    X_latest = features.iloc[-1:].values
    
    # Predict using ensemble or single model
    try:
        # Check if model is ensemble or single
        if isinstance(model, dict) and model.get('type') == 'ensemble':
            # Ensemble prediction: weighted average of all models
            ensemble_probs = []
            models_list = model['models']
            weights = model['weights']
            scaler = model['scaler']
            
            for (model_name, single_model, model_scaler), weight in zip(models_list, weights):
                try:
                    if model_scaler is not None:
                        X_scaled = model_scaler.transform(X_latest)
                    else:
                        X_scaled = X_latest
                    
                    # Get prediction from this model
                    if hasattr(single_model, 'predict_proba'):
                        prob = single_model.predict_proba(X_scaled)[0]
                        prob_up = prob[1] if len(prob) > 1 else 0.5
                    else:
                        pred = single_model.predict(X_scaled)[0]
                        prob_up = 1.0 if pred > 0 else 0.0
                    
                    ensemble_probs.append(prob_up * weight)
                except Exception as e:
                    # If one model fails, use its weight for average
                    ensemble_probs.append(0.5 * weight)
            
            # Weighted average of probabilities
            direction_prob = sum(ensemble_probs) if ensemble_probs else 0.5
        elif hasattr(model, 'predict_proba'):
            # Single model prediction
            prediction = model.predict_proba(X_latest)[0]
            direction_prob = prediction[1] if len(prediction) > 1 else 0.5
        else:
            # Fallback to rule-based
            return predict_rule_based(df)
        
        # Predict actual change using enhanced regression
        if SKLEARN_AVAILABLE:
            try:
                # Enhanced Gradient Boosting Regressor
                regressor = GradientBoostingRegressor(
                    n_estimators=100,  # Increased from 30 to 100
                    max_depth=8,
                    learning_rate=0.1,
                    min_samples_split=5,
                    random_state=42,
                    loss='huber'  # More robust to outliers
                )
                y_reg = df['Close'].pct_change(1).shift(-1).fillna(0).values[-80:]
                X_reg = features.iloc[:-20].values
                if len(y_reg) > 0 and len(X_reg) > 0:
                    regressor.fit(X_reg, y_reg)
                    predicted_change = regressor.predict(X_latest)[0]
                    # Clip extreme predictions
                    predicted_change = np.clip(predicted_change, -0.05, 0.05)
                else:
                    predicted_change = 0.0
            except Exception as e:
                predicted_change = (direction_prob - 0.5) * 0.02  # Simple estimate
        else:
            predicted_change = (direction_prob - 0.5) * 0.02  # Simple estimate
        
        # Enhanced confidence calculation with ensemble boost
        base_confidence = max(direction_prob, 1 - direction_prob)
        
        # Ensemble models provide more reliable predictions
        is_ensemble = isinstance(model, dict) and model.get('type') == 'ensemble'
        ensemble_boost = 1.08 if is_ensemble else 1.0  # 8% boost for ensemble
        
        # Boost confidence if probability is very high/low
        if direction_prob > 0.80 or direction_prob < 0.20:
            confidence = min(0.95, base_confidence * 1.20 * ensemble_boost)  # Strong signals
        elif direction_prob > 0.75 or direction_prob < 0.25:
            confidence = min(0.93, base_confidence * 1.15 * ensemble_boost)  # Boost strong signals
        elif direction_prob > 0.65 or direction_prob < 0.35:
            confidence = min(0.90, base_confidence * 1.10 * ensemble_boost)  # Moderate boost
        else:
            confidence = base_confidence * ensemble_boost
        
        # Additional confidence boost based on predicted change magnitude
        if abs(predicted_change) > 0.015:  # Very strong predicted movement (>1.5%)
            confidence = min(0.95, confidence * 1.08)
        elif abs(predicted_change) > 0.01:  # Strong predicted movement (>1%)
            confidence = min(0.93, confidence * 1.05)
        
        return {
            'direction': 'UP' if direction_prob > 0.6 else 'DOWN' if direction_prob < 0.4 else 'NEUTRAL',
            'predicted_change': predicted_change,
            'confidence': min(0.95, max(0.15, confidence)),  # Clamp between 0.15-0.95
            'method': 'Ensemble ML Model' if is_ensemble else 'Enhanced ML Model',
            'direction_probability': direction_prob
        }
    except Exception as e:
        return predict_rule_based(df)

def predict_rule_based(df):
    """Rule-based prediction as fallback"""
    if len(df) < 20:
        return {
            'direction': 'NEUTRAL',
            'predicted_change': 0.0,
            'confidence': 0.5,
            'method': 'Rule-based'
        }
    
    signals = 0
    
    # RSI signal
    if 'RSI' in df.columns:
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            signals += 1  # Buy signal
        elif rsi > 70:
            signals -= 1  # Sell signal
    
    # MACD signal
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]:
            signals += 1
        else:
            signals -= 1
    
    # Moving average signal
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
            signals += 1
        else:
            signals -= 1
    
    # Price momentum
    momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) if len(df) >= 5 else 0
    if momentum > 0.01:
        signals += 1
    elif momentum < -0.01:
        signals -= 1
    
    # Determine direction
    if signals >= 2:
        direction = 'UP'
        confidence = min(0.9, 0.5 + abs(signals) * 0.1)
    elif signals <= -2:
        direction = 'DOWN'
        confidence = min(0.9, 0.5 + abs(signals) * 0.1)
    else:
        direction = 'NEUTRAL'
        confidence = 0.5
    
    # Estimate change (simple)
    avg_volatility = df['Close'].pct_change().std() if len(df) > 1 else 0.01
    predicted_change = signals * avg_volatility * 0.5
    
    return {
        'direction': direction,
        'predicted_change': predicted_change,
        'confidence': confidence,
        'method': 'Rule-based'
    }

def calculate_entry_exit_levels(df, recommendation):
    """Calculate precise entry, stop loss, and take profit levels"""
    if len(df) < 20:
        current_price = df['Close'].iloc[-1]
        return {
            'entry': current_price,
            'stop_loss': current_price * 0.99,
            'take_profit': current_price * 1.01
        }
    
    current_price = df['Close'].iloc[-1]
    action = recommendation.get('action', 'HOLD')
    
    # Calculate ATR for volatility-based stops
    if 'ATR' in df.columns:
        atr = df['ATR'].iloc[-1]
    else:
        # Calculate ATR manually
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
    
    # If ATR is 0 or NaN, use percentage
    if pd.isna(atr) or atr == 0:
        atr_multiplier = current_price * 0.01  # 1% default
    else:
        atr_multiplier = atr
    
    if action == 'BUY':
        # Entry: Current price or slightly above resistance
        entry = current_price
        
        # Stop loss: Below recent low, or 2*ATR below entry
        if 'Low' in df.columns:
            recent_low = df['Low'].rolling(20).min().iloc[-1]
            sl_by_low = recent_low * 0.999
            sl_by_atr = entry - (2 * atr_multiplier)
            stop_loss = min(sl_by_low, sl_by_atr)
        else:
            stop_loss = entry - (2 * atr_multiplier)
        
        # Take profit: 2:1 or 3:1 risk-reward ratio
        risk = entry - stop_loss
        take_profit = entry + (risk * 2.5)  # 2.5:1 ratio
        
        # Ensure TP is reasonable (not too far)
        if take_profit > current_price * 1.05:  # Max 5% gain
            take_profit = current_price * 1.03
        
    elif action == 'SELL':
        # Entry: Current price or slightly below support
        entry = current_price
        
        # Stop loss: Above recent high, or 2*ATR above entry
        if 'High' in df.columns:
            recent_high = df['High'].rolling(20).max().iloc[-1]
            sl_by_high = recent_high * 1.001
            sl_by_atr = entry + (2 * atr_multiplier)
            stop_loss = max(sl_by_high, sl_by_atr)
        else:
            stop_loss = entry + (2 * atr_multiplier)
        
        # Take profit: 2:1 or 3:1 risk-reward ratio
        risk = stop_loss - entry
        take_profit = entry - (risk * 2.5)  # 2.5:1 ratio
        
        # Ensure TP is reasonable
        if take_profit < current_price * 0.95:  # Max 5% drop
            take_profit = current_price * 0.97
    else:
        # HOLD - conservative levels
        entry = current_price
        stop_loss = current_price - atr_multiplier
        take_profit = current_price + atr_multiplier
    
    return {
        'entry': round(entry, 5),
        'stop_loss': round(stop_loss, 5),
        'take_profit': round(take_profit, 5),
        'risk_reward_ratio': abs((take_profit - entry) / (stop_loss - entry)) if abs(stop_loss - entry) > 0 else 0
    }

def calculate_risk_metrics(df):
    """Calculate comprehensive risk metrics"""
    if len(df) < 20:
        return {
            'volatility': 0.2,
            'risk_score': 0.5,
            'liquidity_risk': 0.5,
            'trend_risk': 0.5,
            'overall_risk': 'Medium'
        }
    
    import numpy as np
    
    # Calculate volatility (annualized)
    returns = df['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.2
    
    # Calculate ATR-based volatility
    if 'ATR' in df.columns:
        atr_pct = (df['ATR'].iloc[-1] / df['Close'].iloc[-1]) * 100
        volatility = max(volatility, atr_pct * np.sqrt(252) / 100)
    
    # Liquidity risk (based on volume consistency)
    liquidity_risk = 0.5
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        volume_cv = df['Volume'].tail(20).std() / df['Volume'].tail(20).mean() if df['Volume'].tail(20).mean() > 0 else 1
        if volume_cv > 1.5:
            liquidity_risk = 0.8  # High variation = low liquidity
        elif volume_cv < 0.5:
            liquidity_risk = 0.2  # Stable volume = good liquidity
    
    # Trend risk (how clear is the trend)
    trend_risk = 0.5
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        sma_diff = abs(df['SMA_20'].iloc[-1] - df['SMA_50'].iloc[-1]) / df['Close'].iloc[-1]
        if sma_diff < 0.01:  # MAs close together = choppy market
            trend_risk = 0.7
        elif sma_diff > 0.05:  # Clear trend
            trend_risk = 0.3
    
    # Calculate overall risk score (0-1, lower = less risk)
    risk_score = (
        min(volatility / 0.5, 1.0) * 0.4 +  # Volatility weight: 40%
        liquidity_risk * 0.2 +                # Liquidity weight: 20%
        trend_risk * 0.2 +                    # Trend clarity: 20%
        (1 - min(volatility / 0.5, 1.0)) * 0.2  # Inverse volatility bonus: 20%
    )
    
    # Determine risk level
    if risk_score > 0.7:
        risk_level = "High"
    elif risk_score > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    return {
        'volatility': volatility,
        'risk_score': risk_score,
        'liquidity_risk': liquidity_risk,
        'trend_risk': trend_risk,
        'overall_risk': risk_level,
        'atr_volatility': atr_pct if 'ATR' in df.columns else volatility
    }

def calculate_advanced_confidence(df, ml_prediction, strength_score, news_sentiment, risk_metrics):
    """Calculate advanced confidence score considering multiple factors with enhanced weighting"""
    
    ml_direction = ml_prediction.get('direction', 'NEUTRAL')
    ml_confidence = ml_prediction.get('confidence', 0.5)
    ml_change = ml_prediction.get('predicted_change', 0)
    direction_prob = ml_prediction.get('direction_probability', 0.5)  # Raw ML probability
    
    # Factor 1: Enhanced ML Model Confidence with Dynamic Scaling (42% weight - increased)
    # Boost confidence if ML is very certain (high/low direction_prob)
    # Also consider if using ensemble model (more reliable)
    is_ensemble = 'Ensemble' in ml_prediction.get('method', '')
    
    if direction_prob > 0.85 or direction_prob < 0.15:
        ml_confidence_boosted = min(0.97, ml_confidence * 1.25)  # Very strong signals
    elif direction_prob > 0.75 or direction_prob < 0.25:
        ml_confidence_boosted = min(0.95, ml_confidence * 1.20)
    elif direction_prob > 0.65 or direction_prob < 0.35:
        ml_confidence_boosted = min(0.92, ml_confidence * 1.12)
    else:
        ml_confidence_boosted = ml_confidence * 1.05  # Small boost even for neutral
    
    # Additional boost for ensemble models (more reliable)
    if is_ensemble:
        ml_confidence_boosted = min(0.97, ml_confidence_boosted * 1.05)
    
    ml_factor = ml_confidence_boosted * 0.42  # Increased weight from 0.40 to 0.42
    
    # Factor 2: Enhanced Technical Strength Alignment with Dynamic Multipliers (28% weight)
    # Check if technicals align with ML direction, with strength-based scaling
    if ml_direction == 'UP':
        tech_alignment = strength_score / 100  # Higher strength = better alignment
        # Progressive bonus based on strength level
        if strength_score > 85:
            tech_alignment = min(1.0, tech_alignment * 1.25)  # Very strong bullish
        elif strength_score > 75:
            tech_alignment = min(1.0, tech_alignment * 1.18)  # Strong bullish
        elif strength_score > 65:
            tech_alignment = min(1.0, tech_alignment * 1.10)  # Moderate bullish
    elif ml_direction == 'DOWN':
        tech_alignment = (100 - strength_score) / 100  # Lower strength = better alignment
        # Progressive bonus based on weakness level
        if strength_score < 15:
            tech_alignment = min(1.0, tech_alignment * 1.25)  # Very strong bearish
        elif strength_score < 25:
            tech_alignment = min(1.0, tech_alignment * 1.18)  # Strong bearish
        elif strength_score < 35:
            tech_alignment = min(1.0, tech_alignment * 1.10)  # Moderate bearish
    else:
        tech_alignment = 0.5  # Neutral
    
    tech_factor = tech_alignment * 0.28  # Slightly reduced to accommodate increased convergence weight
    
    # Factor 3: Enhanced Signal Convergence with Dynamic Multipliers (25% weight - increased)
    # How many indicators agree with the direction? Weighted by indicator importance
    convergence_score = 0.5
    signal_weights = {}  # Track individual signal strengths
    weighted_agreement = 0.0
    total_weight = 0.0
    
    if len(df) >= 20:
        close = df['Close'].iloc[-1]
        
        # RSI alignment (weight: 0.18 - high importance)
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if ml_direction == 'UP':
                rsi_support = 1.0 if rsi < 70 else (0.5 if rsi < 80 else 0.2)
            elif ml_direction == 'DOWN':
                rsi_support = 1.0 if rsi > 30 else (0.5 if rsi > 20 else 0.2)
            else:
                rsi_support = 0.5
            weighted_agreement += rsi_support * 0.18
            total_weight += 0.18
            signal_weights['RSI'] = rsi_support
        
        # MACD alignment (weight: 0.20 - high importance)
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            macd_bullish = macd > macd_signal
            macd_strength = abs(macd - macd_signal) / abs(macd_signal) if abs(macd_signal) > 0 else 0
            
            if (ml_direction == 'UP' and macd_bullish) or (ml_direction == 'DOWN' and not macd_bullish):
                # Strong MACD signal with significant divergence
                macd_support = min(1.0, 0.7 + macd_strength * 0.3)
            else:
                macd_support = 0.3
            weighted_agreement += macd_support * 0.20
            total_weight += 0.20
            signal_weights['MACD'] = macd_support
        
        # Moving Average alignment (weight: 0.18 - high importance)
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma20 = df['SMA_20'].iloc[-1]
            sma50 = df['SMA_50'].iloc[-1]
            sma_bullish = sma20 > sma50
            price_above_sma20 = close > sma20
            
            if ml_direction == 'UP':
                sma_support = 0.9 if sma_bullish else 0.3
                if price_above_sma20:
                    sma_support = min(1.0, sma_support + 0.1)
            elif ml_direction == 'DOWN':
                sma_support = 0.9 if not sma_bullish else 0.3
                if not price_above_sma20:
                    sma_support = min(1.0, sma_support + 0.1)
            else:
                sma_support = 0.5
            
            weighted_agreement += sma_support * 0.18
            total_weight += 0.18
            signal_weights['SMA'] = sma_support
        
        # Momentum alignment (weight: 0.15)
        if 'Momentum' in df.columns:
            momentum = df['Momentum'].iloc[-1]
            momentum_bullish = momentum > 0
            momentum_strength = abs(momentum) / close if close > 0 else 0
            
            if (ml_direction == 'UP' and momentum_bullish) or (ml_direction == 'DOWN' and not momentum_bullish):
                momentum_support = min(1.0, 0.6 + momentum_strength * 200)  # Scale momentum strength
            else:
                momentum_support = 0.4
            weighted_agreement += momentum_support * 0.15
            total_weight += 0.15
            signal_weights['Momentum'] = momentum_support
        
        # Bollinger Bands alignment (weight: 0.12)
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            bb_mid = (bb_upper + bb_lower) / 2
            
            if (ml_direction == 'UP' and close > bb_mid) or (ml_direction == 'DOWN' and close < bb_mid):
                # Check how far from middle band (stronger signal if further)
                bb_position = (close - bb_mid) / (bb_upper - bb_mid) if (bb_upper - bb_mid) > 0 else 0
                bb_support = min(1.0, 0.6 + abs(bb_position) * 0.4)
            else:
                bb_support = 0.4
            weighted_agreement += bb_support * 0.12
            total_weight += 0.12
            signal_weights['BB'] = bb_support
        
        # Stochastic alignment (weight: 0.10)
        if 'Stoch_K' in df.columns:
            stoch = df['Stoch_K'].iloc[-1]
            if ml_direction == 'UP':
                stoch_support = 1.0 if stoch < 70 else (0.6 if stoch < 80 else 0.3)
            elif ml_direction == 'DOWN':
                stoch_support = 1.0 if stoch > 30 else (0.6 if stoch > 20 else 0.3)
            else:
                stoch_support = 0.5
            weighted_agreement += stoch_support * 0.10
            total_weight += 0.10
            signal_weights['Stoch'] = stoch_support
        
        # EMA alignment (weight: 0.07) - Additional trend confirmation
        if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
            ema12 = df['EMA_12'].iloc[-1]
            ema26 = df['EMA_26'].iloc[-1]
            ema_bullish = ema12 > ema26
            
            if (ml_direction == 'UP' and ema_bullish) or (ml_direction == 'DOWN' and not ema_bullish):
                ema_support = 0.8
            else:
                ema_support = 0.4
            weighted_agreement += ema_support * 0.07
            total_weight += 0.07
            signal_weights['EMA'] = ema_support
        
        # Calculate weighted convergence score
        if total_weight > 0:
            convergence_score = weighted_agreement / total_weight
            
            # Dynamic multiplier based on agreement strength
            # Very high agreement (85%+) gets significant boost
            if convergence_score > 0.85:
                convergence_multiplier = 1.25  # 25% boost
            elif convergence_score > 0.75:
                convergence_multiplier = 1.15  # 15% boost
            elif convergence_score > 0.65:
                convergence_multiplier = 1.08  # 8% boost
            elif convergence_score < 0.35:
                convergence_multiplier = 0.80  # 20% penalty
            elif convergence_score < 0.45:
                convergence_multiplier = 0.90  # 10% penalty
            else:
                convergence_multiplier = 1.0
            
            convergence_score = min(1.0, convergence_score * convergence_multiplier)
        else:
            convergence_score = 0.5
    
    convergence_factor = convergence_score * 0.25  # Increased weight from 0.20 to 0.25
    
    # Factor 4: News Sentiment Alignment (10% weight)
    news_compound = news_sentiment.get('compound', 0)
    if ml_direction == 'UP':
        news_alignment = max(0, news_compound)  # Positive news for BUY
    elif ml_direction == 'DOWN':
        news_alignment = max(0, -news_compound)  # Negative news for SELL
    else:
        news_alignment = 0.5
    
    news_factor = news_alignment * 0.10
    
    # Factor 5: Enhanced Risk-Adjusted Bonus/Penalty (5% weight)
    # Lower risk = higher confidence, higher risk = lower confidence
    risk_score = risk_metrics.get('risk_score', 0.5)
    risk_adjustment = (1 - risk_score) * 0.05  # Lower risk increases confidence
    
    # Factor 6: ML Predicted Change Magnitude Bonus (5% weight - NEW)
    # Strong predicted movements increase confidence
    change_magnitude = abs(ml_change)
    if change_magnitude > 0.015:  # > 1.5% predicted change
        change_bonus = min(0.05, change_magnitude * 2.0)
    elif change_magnitude > 0.01:  # > 1% predicted change
        change_bonus = min(0.03, change_magnitude * 2.0)
    else:
        change_bonus = 0.0
    
    # Calculate base confidence
    base_confidence = ml_factor + tech_factor + convergence_factor + news_factor + risk_adjustment + change_bonus
    
    # Risk-based confidence adjustment
    # High confidence with low risk = very high confidence
    # High confidence with high risk = moderate confidence
    risk_level = risk_metrics.get('overall_risk', 'Medium')
    
    if risk_level == "Low":
        # Low risk boosts confidence
        confidence = min(0.95, base_confidence * 1.15)
    elif risk_level == "High":
        # High risk reduces confidence
        confidence = max(0.1, base_confidence * 0.85)
    else:
        confidence = base_confidence
    
    # Enhanced bonuses for strong signal alignment
    if convergence_score > 0.85 and tech_alignment > 0.8:
        confidence = min(0.97, confidence * 1.20)  # 20% bonus for very strong alignment
    elif convergence_score > 0.75 and tech_alignment > 0.7:
        confidence = min(0.95, confidence * 1.15)  # 15% bonus for strong alignment
    
    # Additional bonus if ML predicted change aligns with direction
    if ml_direction == 'UP' and ml_change > 0.01:
        confidence = min(0.95, confidence * 1.05)  # 5% bonus for strong UP prediction
    elif ml_direction == 'DOWN' and ml_change < -0.01:
        confidence = min(0.95, confidence * 1.05)  # 5% bonus for strong DOWN prediction
    
    # Penalty for conflicting signals
    if convergence_score < 0.3 or tech_alignment < 0.3:
        confidence = max(0.15, confidence * 0.75)  # 25% penalty for weak alignment
    
    # Final confidence boost if all factors align strongly
    if (ml_confidence_boosted > 0.8 and 
        tech_alignment > 0.75 and 
        convergence_score > 0.7 and 
        risk_score < 0.4):
        confidence = min(0.97, confidence * 1.10)  # Final 10% boost for perfect alignment
    
    # Ensure confidence is within bounds (increased max to 0.97)
    confidence = min(0.97, max(0.15, confidence))
    
    return confidence

def get_trading_recommendation(df, ml_prediction):
    """Generate comprehensive trading recommendation combining ML and technical analysis"""
    from analytics import calculate_strength_score, generate_trading_signal
    from analytics import analyze_news_sentiment
    from fetch_news import get_news_for_symbol
    
    # Get technical strength
    strength_score = calculate_strength_score(df)
    
    # Get ML prediction
    ml_direction = ml_prediction.get('direction', 'NEUTRAL')
    ml_confidence = ml_prediction.get('confidence', 0.5)
    ml_change = ml_prediction.get('predicted_change', 0)
    
    # Get news sentiment (try to fetch if not provided)
    try:
        symbol = df.attrs.get('symbol', 'EURUSD') if hasattr(df, 'attrs') else 'EURUSD'
        news_items = get_news_for_symbol(symbol, max_items=5)
        news_sentiment = analyze_news_sentiment(news_items)
    except:
        news_sentiment = {'compound': 0.0, 'count': 0}
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(df)
    
    # Enhanced action determination with better thresholds
    ml_direction_prob = ml_prediction.get('direction_probability', 0.5)
    
    # Strong signals: require higher confidence
    if ml_direction == 'UP' and strength_score > 60 and ml_direction_prob > 0.65:
        action = 'BUY'
        base_reason = f"Strong bullish signals: ML predicts {ml_change*100:.2f}% gain ({(ml_direction_prob*100):.0f}% confidence), Technical strength {strength_score}/100"
    elif ml_direction == 'DOWN' and strength_score < 40 and ml_direction_prob < 0.35:
        action = 'SELL'
        base_reason = f"Strong bearish signals: ML predicts {ml_change*100:.2f}% drop ({(ml_direction_prob*100):.0f}% confidence), Technical strength {strength_score}/100"
    elif ml_direction == 'UP' and ml_direction_prob > 0.55:
        action = 'BUY'
        base_reason = f"ML suggests upward movement ({ml_change*100:.2f}%, {(ml_direction_prob*100):.0f}% confidence), technical signals supporting"
    elif ml_direction == 'DOWN' and ml_direction_prob < 0.45:
        action = 'SELL'
        base_reason = f"ML suggests downward movement ({ml_change*100:.2f}%, {(ml_direction_prob*100):.0f}% confidence), technical signals supporting"
    else:
        action = 'HOLD'
        base_reason = "Mixed signals - waiting for clearer direction"
    
    # Calculate advanced confidence
    confidence = calculate_advanced_confidence(
        df, ml_prediction, strength_score, news_sentiment, risk_metrics
    )
    
    # Add risk information to reason
    risk_level = risk_metrics.get('overall_risk', 'Medium')
    volatility = risk_metrics.get('volatility', 0.2)
    
    reason = f"{base_reason}. Risk: {risk_level} (Volatility: {volatility*100:.1f}%). "
    
    if news_sentiment.get('count', 0) > 0:
        news_compound = news_sentiment.get('compound', 0)
        if news_compound > 0.1:
            reason += f"News sentiment: Positive. "
        elif news_compound < -0.1:
            reason += f"News sentiment: Negative. "
        else:
            reason += f"News sentiment: Neutral. "
    
    return {
        'action': action,
        'confidence': confidence,
        'reason': reason,
        'risk_level': risk_level,
        'risk_score': risk_metrics.get('risk_score', 0.5),
        'volatility': volatility,
        'ml_direction': ml_direction,
        'technical_strength': strength_score,
        'risk_metrics': risk_metrics,
        'news_sentiment_score': news_sentiment.get('compound', 0)
    }

