"""
AI Analysis Module - Provides Enhanced LLM-like intelligent trading insights
Enhanced with advanced pattern recognition and multi-factor analysis
"""

def generate_trading_insight(df, ml_prediction, technical_score, news_sentiment, trading_recommendation, entry_levels):
    """
    Generate enhanced human-readable trading insight using advanced rule-based AI
    Incorporates multiple analysis factors for comprehensive market insights
    """
    
    current_price = float(df['Close'].iloc[-1])
    
    # Get action from trading_recommendation if available, otherwise from ml_prediction
    if trading_recommendation and isinstance(trading_recommendation, dict):
        action = trading_recommendation.get('action', 'HOLD')
        rec_confidence = trading_recommendation.get('confidence', 0.5) * 100
        rec_reason = trading_recommendation.get('reason', 'Based on technical and sentiment analysis')
    else:
        action = ml_prediction.get('action', 'HOLD') if isinstance(ml_prediction, dict) and 'action' in ml_prediction else ml_prediction.get('direction', 'NEUTRAL')
        rec_confidence = ml_prediction.get('confidence', 0.5) * 100 if isinstance(ml_prediction, dict) else 50
        rec_reason = 'Based on ML prediction and technical analysis'
    
    # Determine market condition
    volatility = df['Close'].pct_change().std() * 100 if len(df) > 1 else 1
    
    if volatility > 2:
        market_condition = "highly volatile"
        caution_level = "high"
    elif volatility > 1:
        market_condition = "moderately volatile"
        caution_level = "medium"
    else:
        market_condition = "stable"
        caution_level = "low"
    
    # Trend analysis
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
            trend = "bullish"
        else:
            trend = "bearish"
    else:
        trend = "neutral"
    
    # RSI analysis
    if 'RSI' in df.columns:
        rsi = df['RSI'].iloc[-1]
        if rsi > 70:
            rsi_condition = "overbought, suggesting potential downward pressure"
        elif rsi < 30:
            rsi_condition = "oversold, suggesting potential upward bounce"
        else:
            rsi_condition = "neutral, no extreme conditions"
    else:
        rsi_condition = "not available"
    
    # Generate comprehensive insight
    insight_parts = []
    
    # Opening statement
    insight_parts.append(f"**Market Analysis for Current Session:**")
    insight_parts.append("")
    
    # Price action
    change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
    insight_parts.append(f"📊 **Price Action:** Current price is {current_price:.5f}, showing a {abs(change):.2f}% {'gain' if change > 0 else 'loss'} in the recent period.")
    
    # Market condition
    insight_parts.append(f"🌊 **Market Condition:** The market is {market_condition} (volatility: {volatility:.2f}%), requiring {caution_level} caution.")
    
    # Trend
    insight_parts.append(f"📈 **Trend Analysis:** The overall trend appears {trend}, with moving averages supporting this direction.")
    
    # RSI
    insight_parts.append(f"⚖️ **Momentum (RSI):** The RSI indicator shows the market is {rsi_condition}.")
    
    # Trading Recommendation with enhanced details
    ml_direction = ml_prediction.get('direction', 'NEUTRAL') if isinstance(ml_prediction, dict) else 'NEUTRAL'
    ml_conf = ml_prediction.get('confidence', 0.5) * 100 if isinstance(ml_prediction, dict) else 50
    ml_change = ml_prediction.get('predicted_change', 0) * 100 if isinstance(ml_prediction, dict) else 0
    
    insight_parts.append(f"🤖 **AI Recommendation:** {action} signal with {rec_confidence:.0f}% confidence.")
    insight_parts.append(f"📈 **ML Model Prediction:** {ml_direction} direction ({ml_conf:.0f}% confidence, {ml_change:+.2f}% expected movement).")
    insight_parts.append(f"💡 **Analysis:** {rec_reason}")
    
    # Technical strength with enhanced description
    if technical_score > 75:
        tech_desc = "very strong bullish signals"
    elif technical_score > 60:
        tech_desc = "strong bullish signals"
    elif technical_score > 40:
        tech_desc = "moderate/mixed signals"
    elif technical_score > 25:
        tech_desc = "strong bearish signals"
    else:
        tech_desc = "very strong bearish signals"
    
    insight_parts.append(f"💪 **Technical Strength:** Score of {technical_score:.0f}/100 indicates {tech_desc}.")
    
    # News sentiment
    news_compound = news_sentiment.get('compound', 0)
    news_count = news_sentiment.get('count', 0)
    if news_count > 0:
        if news_compound > 0.2:
            sentiment_desc = "strongly positive"
        elif news_compound < -0.2:
            sentiment_desc = "strongly negative"
        else:
            sentiment_desc = "mixed/neutral"
        insight_parts.append(f"📰 **News Sentiment:** Based on {news_count} recent articles, market sentiment is {sentiment_desc}.")
    
    # Trading recommendation
    insight_parts.append("")
    insight_parts.append("**🎯 Trading Recommendation:**")
    
    entry = entry_levels.get('entry', current_price)
    stop_loss = entry_levels.get('stop_loss', entry * 0.99)
    take_profit = entry_levels.get('take_profit', entry * 1.01)
    risk_reward = entry_levels.get('risk_reward_ratio', 1.0)
    
    if action in ['BUY', 'SELL']:
        insight_parts.append(f"- **Action:** {action} signal is recommended")
        insight_parts.append(f"- **Entry Price:** Consider entering at {entry:.5f}")
        insight_parts.append(f"- **Stop Loss:** Set stop loss at {stop_loss:.5f} to limit risk")
        insight_parts.append(f"- **Take Profit:** Target take profit at {take_profit:.5f}")
        insight_parts.append(f"- **Risk-Reward Ratio:** {risk_reward:.2f}:1 (favorable ratio)")
    else:
        insight_parts.append(f"- **Action:** HOLD - Wait for clearer signals")
        insight_parts.append(f"- **Reason:** Mixed signals detected, better to wait for confirmation")
    
    # Enhanced risk assessment
    risk_level = trading_recommendation.get('risk_level', 'Medium') if isinstance(trading_recommendation, dict) else 'Medium'
    volatility_pct = volatility
    # Enhanced Pattern Recognition Analysis
    insight_parts.append("")
    insight_parts.append("**🔍 Advanced Pattern Recognition:**")
    
    # MACD Analysis
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        try:
            macd = float(df['MACD'].iloc[-1])
            macd_signal = float(df['MACD_Signal'].iloc[-1])
            macd_hist = macd - macd_signal
            
            if macd_hist > 0 and macd > 0:
                macd_signal_desc = "bullish momentum with MACD above signal line"
            elif macd_hist < 0 and macd < 0:
                macd_signal_desc = "bearish momentum with MACD below signal line"
            elif macd_hist > 0:
                macd_signal_desc = "potential bullish crossover forming"
            else:
                macd_signal_desc = "potential bearish crossover forming"
            
            insight_parts.append(f"- **MACD:** {macd_signal_desc}")
        except:
            pass
    
    # Bollinger Bands Analysis
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        try:
            bb_upper = float(df['BB_Upper'].iloc[-1])
            bb_lower = float(df['BB_Lower'].iloc[-1])
            bb_mid = (bb_upper + bb_lower) / 2
            bb_width = (bb_upper - bb_lower) / bb_mid * 100 if bb_mid > 0 else 0
            
            price_pos_bb = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            if current_price > bb_upper:
                bb_signal = "price above upper band - potentially overbought"
            elif current_price < bb_lower:
                bb_signal = "price below lower band - potentially oversold"
            elif price_pos_bb > 0.8:
                bb_signal = "price near upper band - watch for reversal"
            elif price_pos_bb < 0.2:
                bb_signal = "price near lower band - potential bounce"
            else:
                bb_signal = "price in middle range - neutral"
            
            insight_parts.append(f"- **Bollinger Bands:** {bb_signal} (Band width: {bb_width:.2f}% indicating {'high' if bb_width > 2 else 'low'} volatility)")
        except:
            pass
    
    # Support/Resistance Analysis
    try:
        recent_high_20 = float(df['High'].rolling(20).max().iloc[-1])
        recent_low_20 = float(df['Low'].rolling(20).min().iloc[-1])
        
        distance_to_resistance_20 = (recent_high_20 - current_price) / current_price * 100
        distance_to_support_20 = (current_price - recent_low_20) / current_price * 100
        
        if distance_to_resistance_20 < 0.5:
            sr_signal = f"near resistance at {recent_high_20:.5f} ({distance_to_resistance_20:.2f}% away)"
        elif distance_to_support_20 < 0.5:
            sr_signal = f"near support at {recent_low_20:.5f} ({distance_to_support_20:.2f}% away)"
        else:
            sr_signal = f"mid-range (Support: {recent_low_20:.5f}, Resistance: {recent_high_20:.5f})"
        
        insight_parts.append(f"- **Key Levels:** {sr_signal}")
    except:
        pass
    
    # Enhanced Risk Assessment
    insight_parts.append("")
    insight_parts.append(f"⚠️ **Risk Assessment:** {risk_level} risk level (Volatility: {volatility_pct:.2f}%).")
    try:
        entry = float(entry_levels.get('entry', current_price))
        stop_loss = float(entry_levels.get('stop_loss', entry * 0.99))
        take_profit = float(entry_levels.get('take_profit', entry * 1.01))
        
        stop_loss_pct = abs((current_price - stop_loss) / current_price * 100)
        take_profit_pct = abs((take_profit - current_price) / current_price * 100)
        
        insight_parts.append(f"   - **Recommended Position Size:** {'Conservative (0.5-1%)' if risk_level == 'High' else 'Moderate (1-2%)' if risk_level == 'Medium' else 'Standard (1-2%)'}")
        insight_parts.append(f"   - **Stop Loss:** Always set at {stop_loss_pct:.2f}% below entry ({stop_loss:.5f})")
        insight_parts.append(f"   - **Take Profit:** Target {take_profit_pct:.2f}% gain ({take_profit:.5f})")
    except:
        pass
    insight_parts.append(f"   - **Risk Management:** Never risk more than 1-2% of account per trade. Monitor positions closely.")
    
    # Enhanced Performance note with model information
    model_method = ml_prediction.get('method', 'ML Model') if isinstance(ml_prediction, dict) else 'ML Model'
    model_version = ml_prediction.get('model_version', '1.0') if isinstance(ml_prediction, dict) else '1.0'
    model_accuracy = ml_prediction.get('model_accuracy', None) if isinstance(ml_prediction, dict) else None
    
    insight_parts.append("")
    insight_parts.append(f"📊 **Analysis Quality:** This comprehensive analysis combines:")
    insight_parts.append(f"   - {model_method} (v{model_version}) with {ml_conf:.0f}% confidence")
    if model_accuracy:
        insight_parts.append(f"   - Model accuracy: {model_accuracy:.2%} on historical data")
    insight_parts.append(f"   - Technical analysis ({technical_score:.0f}/100 strength)")
    insight_parts.append(f"   - News sentiment analysis ({news_count} articles analyzed)")
    insight_parts.append(f"   - Advanced pattern recognition and multi-factor analysis")
    
    return "\n\n".join(insight_parts)

def generate_summary_text(df, recommendation, entry_levels):
    """Generate a brief summary text"""
    current_price = df['Close'].iloc[-1]
    action = recommendation.get('action', 'HOLD')
    confidence = recommendation.get('confidence', 0.5) * 100
    
    summary = f"Current price: {current_price:.5f}. {action} signal ({confidence:.0f}% confidence). "
    
    if action != 'HOLD':
        entry = entry_levels.get('entry', current_price)
        summary += f"Enter at {entry:.5f}, stop loss: {entry_levels.get('stop_loss', 0):.5f}, take profit: {entry_levels.get('take_profit', 0):.5f}."
    
    return summary

