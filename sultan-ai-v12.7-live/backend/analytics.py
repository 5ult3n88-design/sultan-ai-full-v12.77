import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

def calculate_advanced_indicators(df):
    """Calculate comprehensive technical indicators"""
    from data_validator import clean_dataframe_for_analysis, validate_dataframe
    
    # Clean and validate data first
    try:
        df = clean_dataframe_for_analysis(df)
        is_valid, error_msg = validate_dataframe(df, min_rows=50)
        if not is_valid:
            raise ValueError(f"Data validation failed: {error_msg}")
    except Exception as e:
        raise ValueError(f"Data cleaning error: {str(e)}")
    
    # Ensure we have a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Basic moving averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    
    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    # Handle MultiIndex columns if present (from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
    
    # Ensure Close is a Series, not DataFrame
    close_series = df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    
    # Calculate Bollinger Bands components as Series
    bb_middle = close_series.rolling(window=20).mean()
    bb_std = close_series.rolling(window=20).std()
    
    # Assign as Series to avoid DataFrame issues
    df["BB_Middle"] = bb_middle
    df["BB_Upper"] = bb_middle + (bb_std * 2)
    df["BB_Lower"] = bb_middle - (bb_std * 2)
    
    # Stochastic Oscillator
    low_min = df["Low"].rolling(window=14).min()
    high_max = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    
    # ATR (Average True Range) for volatility
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()
    
    # Volume indicators (if volume exists)
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
    
    # Price momentum
    df["Momentum"] = df["Close"] / df["Close"].shift(10) - 1
    
    # Rate of Change (ROC)
    df["ROC"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100
    
    return df

def calculate_strength_score(df):
    """Calculate overall technical strength score (0-100) with enhanced multi-indicator scoring + trend confirmation"""
    if len(df) < 50:
        return 50  # Neutral if not enough data
    
    scores = []
    weights = []
    last_idx = -1
    
    try:
        close = df["Close"].iloc[last_idx]
        
        # ============================================================
        # TREND CONFIRMATION (Multi-timeframe + Alignment Check)
        # ============================================================
        trend_signals = []
        trend_strength = 0.5  # Default neutral
        
        # 1. Short-term trend (SMA 20 vs Price)
        if 'SMA_20' in df.columns:
            sma20 = df["SMA_20"].iloc[last_idx]
            price_above_sma20 = close > sma20
            trend_signals.append(1.0 if price_above_sma20 else 0.0)
            
            # Trend strength: distance from MA
            ma_distance = abs(close - sma20) / sma20
            trend_strength = max(trend_strength, min(1.0, ma_distance * 100))
        
        # 2. Medium-term trend (SMA 20 vs SMA 50)
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma20 = df["SMA_20"].iloc[last_idx]
            sma50 = df["SMA_50"].iloc[last_idx]
            sma_bullish = sma20 > sma50
            trend_signals.append(1.0 if sma_bullish else 0.0)
            
            # Enhanced trend strength calculation
            ma_spread = abs(sma20 - sma50) / sma50
            trend_strength = max(trend_strength, min(1.0, ma_spread * 200))
        
        # 3. MACD trend confirmation
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd = df["MACD"].iloc[last_idx]
            macd_signal = df["MACD_Signal"].iloc[last_idx]
            macd_bullish = macd > macd_signal
            trend_signals.append(1.0 if macd_bullish else 0.0)
            
            # MACD histogram strength
            macd_hist = macd - macd_signal
            if abs(macd_signal) > 0:
                hist_strength = min(1.0, abs(macd_hist) / abs(macd_signal))
                trend_strength = max(trend_strength, hist_strength)
        
        # 4. Momentum trend confirmation
        if 'Momentum' in df.columns:
            momentum = df["Momentum"].iloc[last_idx]
            momentum_bullish = momentum > 0
            trend_signals.append(1.0 if momentum_bullish else 0.0)
        
        # 5. EMA trend (if available)
        if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
            ema12 = df["EMA_12"].iloc[last_idx]
            ema26 = df["EMA_26"].iloc[last_idx]
            ema_bullish = ema12 > ema26
            trend_signals.append(1.0 if ema_bullish else 0.0)
        
        # Calculate trend alignment percentage
        trend_alignment = np.mean(trend_signals) if trend_signals else 0.5
        
        # ============================================================
        # INDIVIDUAL INDICATOR SCORES (Enhanced Weighting)
        # ============================================================
        
        # RSI signals (weight: 18%)
        rsi = df["RSI"].iloc[last_idx]
        if rsi < 30:
            rsi_score = 80 + (30 - rsi) / 30 * 15  # 80-95 for oversold (boosted)
        elif rsi > 70:
            rsi_score = 20 - (rsi - 70) / 30 * 10  # 20-10 for overbought
        else:
            # Normal range: 30-70, convert to 10-90 scale
            rsi_score = 10 + (rsi - 30) / 40 * 80
        scores.append(rsi_score)
        weights.append(0.18)
        
        # MACD signals (weight: 18%) - Enhanced with trend confirmation
        macd = df["MACD"].iloc[last_idx]
        macd_signal = df["MACD_Signal"].iloc[last_idx]
        macd_hist = macd - macd_signal
        
        if macd_hist > 0:
            base_macd_score = 50 + min(45, abs(macd_hist) / abs(macd_signal) * 50) if abs(macd_signal) > 0 else 65
            # Boost if MACD aligns with trend
            if trend_alignment > 0.6:
                base_macd_score = min(100, base_macd_score * 1.15)
        else:
            base_macd_score = 50 - min(45, abs(macd_hist) / abs(macd_signal) * 50) if abs(macd_signal) > 0 else 35
            # Boost if MACD aligns with trend
            if trend_alignment < 0.4:
                base_macd_score = max(0, base_macd_score * 0.85)
        
        scores.append(base_macd_score)
        weights.append(0.18)
        
        # Moving average system (weight: 16%) - Enhanced with trend confirmation
        sma20 = df["SMA_20"].iloc[last_idx]
        sma50 = df["SMA_50"].iloc[last_idx]
        
        if sma20 > sma50:
            sma_base_score = 65 + min(30, (sma20 - sma50) / sma50 * 100 * 6)
            # Bonus for price above both MAs
            if close > sma20:
                sma_base_score += 8
            # Extra bonus for strong trend alignment
            if trend_alignment > 0.7:
                sma_base_score = min(100, sma_base_score * 1.1)
        else:
            sma_base_score = 35 - min(30, (sma50 - sma20) / sma50 * 100 * 6)
            # Penalty for price below both MAs
            if close < sma20:
                sma_base_score -= 8
            # Extra penalty for strong bearish alignment
            if trend_alignment < 0.3:
                sma_base_score = max(0, sma_base_score * 0.9)
        
        scores.append(sma_base_score)
        weights.append(0.16)
        
        # Bollinger Bands position (weight: 14%) - Enhanced interpretation
        bb_upper = df["BB_Upper"].iloc[last_idx]
        bb_lower = df["BB_Lower"].iloc[last_idx]
        bb_mid = (bb_upper + bb_lower) / 2
        bb_range = bb_upper - bb_lower
        
        if bb_range > 0:
            bb_position = (close - bb_lower) / bb_range
            if bb_position < 0.2:
                bb_score = 78  # Near lower band, potential bounce (boosted)
            elif bb_position > 0.8:
                bb_score = 22  # Near upper band, potential reversal
            else:
                # Middle zone: consider trend
                if trend_alignment > 0.6:
                    bb_score = 55 + bb_position * 35  # Slight bullish bias
                elif trend_alignment < 0.4:
                    bb_score = 45 - (1 - bb_position) * 35  # Slight bearish bias
                else:
                    bb_score = 25 + bb_position * 50  # Neutral
        else:
            bb_score = 50
        
        scores.append(bb_score)
        weights.append(0.14)
        
        # Stochastic (weight: 12%) - Enhanced
        stoch_k = df["Stoch_K"].iloc[last_idx] if 'Stoch_K' in df.columns else 50
        if stoch_k < 20:
            stoch_score = 82 + (20 - stoch_k) / 20 * 13  # 82-95 for oversold
        elif stoch_k > 80:
            stoch_score = 18 - (stoch_k - 80) / 20 * 13  # 18-5 for overbought
        else:
            stoch_score = 20 + (stoch_k - 20) / 60 * 60
        
        scores.append(stoch_score)
        weights.append(0.12)
        
        # Momentum (weight: 10%) - Enhanced with ROC
        momentum = df["Momentum"].iloc[last_idx] if 'Momentum' in df.columns else 0
        roc = df["ROC"].iloc[last_idx] if 'ROC' in df.columns else 0
        
        # Combined momentum score
        momentum_pct = momentum * 100 if abs(momentum) < 1 else momentum / close * 100
        roc_pct = roc  # Already in percentage
        
        # Average momentum indicators
        avg_momentum = (momentum_pct + roc_pct) / 2
        momentum_score = 50 + min(45, max(-45, avg_momentum * 2))
        
        # Boost if momentum aligns with trend
        if (momentum_score > 50 and trend_alignment > 0.6) or (momentum_score < 50 and trend_alignment < 0.4):
            momentum_score = 50 + (momentum_score - 50) * 1.15
        
        scores.append(momentum_score)
        weights.append(0.10)
        
        # Volume trend (weight: 7%) - Enhanced
        volume_score = 50  # Default neutral
        if "Volume" in df.columns and df["Volume"].iloc[last_idx] > 0:
            volume_ma = df["Volume"].tail(20).mean()
            current_volume = df["Volume"].iloc[last_idx]
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
            
            # Volume confirmation of trend
            price_direction = 1 if close > df["Close"].iloc[max(last_idx-1, 0)] else -1
            
            if volume_ratio > 1.3:  # High volume
                volume_score = 65 if price_direction > 0 else 35
                # Extra boost if volume confirms trend
                if (price_direction > 0 and trend_alignment > 0.6) or (price_direction < 0 and trend_alignment < 0.4):
                    volume_score = 70 if price_direction > 0 else 30
            elif volume_ratio < 0.7:  # Low volume
                volume_score = 45  # Uncertainty
            else:
                volume_score = 50  # Normal volume
        
        scores.append(volume_score)
        weights.append(0.07)
        
        # Trend Strength Bonus (weight: 5%) - NEW
        # Strong trend alignment multiplies confidence
        trend_bonus_score = 50 + (trend_alignment - 0.5) * 60  # Convert 0-1 to 20-80
        # Extra boost for strong trend strength
        if trend_strength > 0.7:
            trend_bonus_score = min(100, trend_bonus_score * 1.2)
        elif trend_strength < 0.3:
            trend_bonus_score = max(0, trend_bonus_score * 0.8)
        
        scores.append(trend_bonus_score)
        weights.append(0.05)
        
        # ============================================================
        # CALCULATE WEIGHTED AVERAGE WITH TREND CONFIRMATION MULTIPLIER
        # ============================================================
        total_weight = sum(weights)
        base_score = sum(s * w for s, w in zip(scores, weights)) / total_weight if total_weight > 0 else 50
        
        # Apply trend confirmation multiplier
        # Strong trend alignment (high agreement) increases confidence
        if trend_alignment > 0.75:  # 75%+ signals agree
            trend_multiplier = 1.15  # 15% boost
        elif trend_alignment > 0.65:  # 65%+ signals agree
            trend_multiplier = 1.10  # 10% boost
        elif trend_alignment < 0.25:  # <25% signals agree (strong bearish)
            trend_multiplier = 0.85  # 15% reduction
        elif trend_alignment < 0.35:  # <35% signals agree
            trend_multiplier = 0.90  # 10% reduction
        else:
            trend_multiplier = 1.0  # No change
        
        final_score = base_score * trend_multiplier
        
    except Exception as e:
        # If any indicator fails, return neutral
        final_score = 50
    
    return max(0, min(100, final_score))  # Clamp between 0-100

def analyze_news_sentiment(news_items):
    """Analyze sentiment of news articles"""
    if not news_items:
        return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'count': 0}
    
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    
    for item in news_items:
        # Combine title and summary
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        # Clean text
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        
        # VADER sentiment
        vader_scores = analyzer.polarity_scores(text)
        sentiments.append(vader_scores['compound'])
        
        # TextBlob sentiment (backup)
        try:
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        except:
            pass
    
    if not sentiments:
        return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'count': 0}
    
    avg_sentiment = np.mean(sentiments)
    
    # Classify
    positive = sum(1 for s in sentiments if s > 0.1)
    negative = sum(1 for s in sentiments if s < -0.1)
    neutral = len(sentiments) - positive - negative
    
    return {
        'compound': avg_sentiment,
        'positive': positive / len(sentiments),
        'negative': negative / len(sentiments),
        'neutral': neutral / len(sentiments),
        'count': len(news_items)
    }

def generate_trading_signal(df, news_sentiment, strength_score):
    """Generate trading recommendation based on technical and sentiment analysis"""
    if len(df) < 50:
        return {
            'action': 'HOLD',
            'confidence': 0.5,
            'reason': 'Insufficient data for analysis'
        }
    
    last_close = df["Close"].iloc[-1]
    rsi = df["RSI"].iloc[-1]
    macd_cross = df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]
    sma_trend = df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1]
    
    # Combine technical and sentiment
    technical_score = (strength_score - 50) / 50  # -1 to 1
    sentiment_score = news_sentiment.get('compound', 0)
    
    # Weighted decision
    combined_score = (technical_score * 0.7) + (sentiment_score * 0.3)
    
    # Generate signal
    if combined_score > 0.3:
        action = 'BUY'
        confidence = min(0.95, abs(combined_score))
        reasons = []
        if technical_score > 0.2:
            reasons.append("Strong technical indicators")
        if sentiment_score > 0.1:
            reasons.append("Positive news sentiment")
        if rsi < 40:
            reasons.append("RSI suggests oversold condition")
        if macd_cross and sma_trend:
            reasons.append("Bullish trend confirmed")
        reason = "; ".join(reasons) if reasons else "Positive market signals"
        
    elif combined_score < -0.3:
        action = 'SELL'
        confidence = min(0.95, abs(combined_score))
        reasons = []
        if technical_score < -0.2:
            reasons.append("Weak technical indicators")
        if sentiment_score < -0.1:
            reasons.append("Negative news sentiment")
        if rsi > 60:
            reasons.append("RSI suggests overbought condition")
        if not macd_cross and not sma_trend:
            reasons.append("Bearish trend confirmed")
        reason = "; ".join(reasons) if reasons else "Negative market signals"
    else:
        action = 'HOLD'
        confidence = 0.5
        reason = "Mixed signals - wait for clearer direction"
    
    # Risk assessment
    volatility = df["Close"].pct_change().std() * np.sqrt(252)
    risk_level = "High" if volatility > 0.3 else "Medium" if volatility > 0.15 else "Low"
    
    return {
        'action': action,
        'confidence': confidence,
        'reason': reason,
        'risk_level': risk_level,
        'target_price': last_close * (1 + combined_score * 0.02) if combined_score > 0.1 else last_close * (1 + combined_score * 0.02),
        'stop_loss': last_close * (1 - abs(combined_score) * 0.015)
    }

def get_market_summary(df, news_sentiment, strength_score, signal):
    """Generate comprehensive market summary"""
    if len(df) < 50:
        return "Insufficient data for analysis."
    
    last_close = df["Close"].iloc[-1]
    change_pct = ((last_close - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100 if len(df) > 1 else 0
    volatility = df["Close"].pct_change().std() * np.sqrt(252) * 100
    
    summary = f"""
**Current Price:** {last_close:.4f} ({change_pct:+.2f}%)
**Technical Strength:** {strength_score}/100
**Sentiment:** {news_sentiment.get('compound', 0):.2f} (Compound)
**Volatility:** {volatility:.2f}%

**Recommendation:** {signal['action']} (Confidence: {signal['confidence']*100:.0f}%)
**Risk Level:** {signal['risk_level']}
**Reason:** {signal['reason']}
"""
    
    return summary

