#!/usr/bin/env python3
"""
Trading Robot Demo - Shows detailed analysis in terminal
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.trading_robot import TradingRobot

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_signal(decision):
    """Print a trading signal with details"""
    symbol = decision['symbol']
    action = decision['action']
    confidence = decision['confidence'] * 100
    ml_dir = decision['ml_direction']
    ml_conf = decision['ml_confidence'] * 100
    tech = decision['technical_strength']
    risk = decision['risk_level']
    price = decision['current_price']

    # Color coding
    if confidence >= 80:
        conf_emoji = "🟢"  # Green
    elif confidence >= 70:
        conf_emoji = "🔵"  # Blue
    elif confidence >= 60:
        conf_emoji = "🟡"  # Yellow
    elif confidence >= 50:
        conf_emoji = "🟠"  # Orange
    else:
        conf_emoji = "🔴"  # Red

    if action == "BUY":
        action_emoji = "📈"
    elif action == "SELL":
        action_emoji = "📉"
    else:
        action_emoji = "⏸️"

    print(f"\n{'-'*70}")
    print(f"Symbol: {symbol} @ ${price:.5f}")
    print(f"{'-'*70}")
    print(f"{action_emoji} Action: {action}")
    print(f"{conf_emoji} Confidence: {confidence:.1f}%")
    print(f"")
    print(f"📊 ML Analysis:")
    print(f"   Direction: {ml_dir} ({ml_conf:.1f}% confidence)")
    print(f"   Predicted Change: {decision['predicted_change']:+.2f}%")
    print(f"   Model: {decision['model_method']}")
    print(f"")
    print(f"📈 Technical Analysis:")
    print(f"   Strength: {tech:.0f}/100")
    print(f"   Risk Level: {risk}")
    print(f"   Volatility: {decision['volatility']*100:.2f}%")
    print(f"")
    print(f"📰 News Sentiment:")
    print(f"   Score: {decision['news_sentiment']:+.2f}")
    print(f"   Articles: {decision['news_count']}")
    print(f"")
    print(f"💰 Trading Levels:")
    print(f"   Entry: ${decision['entry_price']:.5f}")
    print(f"   Stop Loss: ${decision['stop_loss']:.5f}")
    print(f"   Take Profit: ${decision['take_profit']:.5f}")
    print(f"   Risk/Reward: {decision['risk_reward_ratio']:.2f}:1")
    print(f"")

    if decision.get('should_trade') is not None:
        if decision['should_trade']:
            print(f"✅ TRADE EXECUTED: {decision.get('trade_reason', 'N/A')}")
        else:
            print(f"❌ NO TRADE: {decision.get('trade_reason', 'N/A')}")

    print(f"{'-'*70}")

def main():
    """Run the demo"""

    print_header("🤖 TRADING ROBOT DEMO")

    print("\n📋 Configuration:")
    print("   Symbols: EURUSD=X, AAPL, GOOGL, MSFT")
    print("   Min Confidence: 65%")
    print("   Risk per Trade: 1%")
    print("   Starting Balance: $10,000")

    # Create robot
    robot = TradingRobot(
        symbols=['EURUSD=X', 'AAPL', 'GOOGL', 'MSFT'],
        initial_balance=10000,
        risk_per_trade=0.01,
        min_confidence=0.65
    )

    print_header("🔍 ANALYZING SYMBOLS...")

    # Analyze each symbol
    for symbol in robot.symbols:
        try:
            decision, error = robot.analyze_symbol(symbol)

            if error:
                print(f"\n❌ {symbol}: {error}")
                continue

            if decision:
                print_signal(decision)

                # Try to execute
                if decision['action'] in ['BUY', 'SELL']:
                    success, message = robot.execute_trade_decision(decision)

        except Exception as e:
            print(f"\n❌ {symbol}: Error - {str(e)}")

    # Show account summary
    account = robot.executor.get_account_info()

    print_header("💰 ACCOUNT SUMMARY")
    print(f"")
    print(f"  Balance: ${account['balance']:,.2f}")
    print(f"  Equity: ${account['equity']:,.2f}")
    print(f"  Open P&L: ${account['open_pl']:+,.2f}")
    print(f"  Open Positions: {account['open_positions']}")
    print(f"  Total Trades: {account['total_trades']}")
    print(f"  Win Rate: {account['win_rate']:.1f}%")
    print(f"  ROI: {account['roi']:+.2f}%")
    print(f"")

    # Show open positions
    if robot.executor.positions:
        print_header("📊 OPEN POSITIONS")
        for symbol, pos in robot.executor.positions.items():
            print(f"\n  {symbol}:")
            print(f"    Action: {pos['action']}")
            print(f"    Entry: ${pos['entry_price']:.5f}")
            print(f"    Current: ${pos['current_price']:.5f}")
            print(f"    P&L: ${pos['unrealized_pl']:+.2f} ({pos['unrealized_pl_pct']:+.2f}%)")
            print(f"    Confidence: {pos['confidence']:.1%}")

    print_header("✅ DEMO COMPLETE")

    print("\n💡 What you just saw:")
    print("   - Robot analyzed each symbol using ML + Technical + News")
    print("   - Calculated confidence using 6-factor system")
    print("   - Generated BUY/SELL/HOLD signals")
    print("   - Executed trades when criteria met")
    print("   - Tracked account performance")

    print("\n🚀 To run continuously:")
    print("   python run_trading_robot.py")

    print("\n📊 To see in dashboard:")
    print("   streamlit run frontend/Home.py")
    print("   Then click 'Auto Trading Robot' in sidebar")

    print("\n")

if __name__ == "__main__":
    main()
