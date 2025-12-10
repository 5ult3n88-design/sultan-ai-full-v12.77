"""
Autonomous Trading Robot - Core Engine
Makes trading decisions based on ML predictions, technical analysis, and news sentiment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from datetime import datetime, timedelta
import time
import pandas as pd
from trade_executor import TradeExecutor
from data_loader import load_symbol_data
from analytics import calculate_advanced_indicators, calculate_strength_score
from ml_model import predict_price_movement, get_trading_recommendation, calculate_entry_exit_levels
from fetch_news import get_news_for_symbol
from analytics import analyze_news_sentiment
import json
from pathlib import Path


class TradingRobot:
    """Autonomous Trading Robot with ML and Technical Analysis"""

    def __init__(self, symbols=None, initial_balance=10000.0, risk_per_trade=0.01,
                 min_confidence=0.65, check_interval=300):
        """
        Initialize Trading Robot

        Args:
            symbols: List of symbols to trade (default: major forex pairs)
            initial_balance: Starting balance in USD
            risk_per_trade: Risk per trade as % of equity (default: 1%)
            min_confidence: Minimum confidence to enter trade (default: 65%)
            check_interval: How often to check for signals in seconds (default: 5 min)
        """
        self.symbols = symbols or [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'XAUUSD=X',
            'AAPL', 'GOOGL', 'TSLA', 'MSFT'
        ]
        self.risk_per_trade = risk_per_trade
        self.min_confidence = min_confidence
        self.check_interval = check_interval

        # Initialize trade executor
        self.executor = TradeExecutor(initial_balance=initial_balance, max_positions=5)

        # Robot state
        self.is_running = False
        self.last_check = {}
        self.decisions_log = []
        self.last_signals = {}

        # Performance tracking
        self.start_time = None
        self.total_signals = 0
        self.total_trades = 0

    def analyze_symbol(self, symbol):
        """Analyze a symbol and generate trading decision"""
        try:
            # Load data
            df = load_symbol_data(symbol)
            if df is None or len(df) < 100:
                return None, f"Insufficient data for {symbol}"

            # Calculate indicators
            df = calculate_advanced_indicators(df)

            # Set symbol attribute for model
            df.attrs['symbol'] = symbol

            # Get ML prediction
            ml_prediction = predict_price_movement(df, symbol=symbol)

            # Get trading recommendation with advanced confidence
            recommendation = get_trading_recommendation(df, ml_prediction)

            # Calculate entry/exit levels
            entry_levels = calculate_entry_exit_levels(df, recommendation)

            # Get news sentiment
            try:
                news_items = get_news_for_symbol(symbol, max_items=5)
                news_sentiment = analyze_news_sentiment(news_items)
            except:
                news_sentiment = {'compound': 0.0, 'count': 0}

            # Get current price
            current_price = float(df['Close'].iloc[-1])

            # Build decision data
            decision = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'action': recommendation['action'],
                'confidence': recommendation['confidence'],
                'ml_direction': ml_prediction.get('direction', 'NEUTRAL'),
                'ml_confidence': ml_prediction.get('confidence', 0.5),
                'predicted_change': ml_prediction.get('predicted_change', 0) * 100,
                'technical_strength': recommendation['technical_strength'],
                'risk_level': recommendation['risk_level'],
                'volatility': recommendation['volatility'],
                'news_sentiment': news_sentiment.get('compound', 0),
                'news_count': news_sentiment.get('count', 0),
                'entry_price': entry_levels['entry'],
                'stop_loss': entry_levels['stop_loss'],
                'take_profit': entry_levels['take_profit'],
                'risk_reward_ratio': entry_levels.get('risk_reward_ratio', 0),
                'reason': recommendation.get('reason', 'N/A'),
                'model_method': ml_prediction.get('method', 'ML Model')
            }

            return decision, None

        except Exception as e:
            return None, f"Error analyzing {symbol}: {str(e)}"

    def should_enter_trade(self, decision):
        """Determine if robot should enter a trade based on decision"""
        # Check minimum confidence
        if decision['confidence'] < self.min_confidence:
            return False, f"Confidence {decision['confidence']:.1%} < minimum {self.min_confidence:.1%}"

        # Only trade on BUY/SELL signals (not HOLD)
        if decision['action'] not in ['BUY', 'SELL']:
            return False, "Action is HOLD"

        # Check if position already exists
        can_open, message = self.executor.can_open_position(decision['symbol'])
        if not can_open:
            return False, message

        # Risk management: avoid high risk trades
        if decision['risk_level'] == 'High' and decision['confidence'] < 0.75:
            return False, "High risk with insufficient confidence"

        # Ensure good risk-reward ratio
        if decision['risk_reward_ratio'] < 1.5:
            return False, f"Poor risk-reward ratio: {decision['risk_reward_ratio']:.2f}"

        # Check technical alignment
        if decision['action'] == 'BUY' and decision['technical_strength'] < 40:
            return False, "Weak technical support for BUY"
        if decision['action'] == 'SELL' and decision['technical_strength'] > 60:
            return False, "Weak technical support for SELL"

        return True, "All checks passed"

    def execute_trade_decision(self, decision):
        """Execute a trading decision"""
        should_trade, reason = self.should_enter_trade(decision)

        decision['should_trade'] = should_trade
        decision['trade_reason'] = reason

        if not should_trade:
            self.decisions_log.append(decision)
            return False, reason

        # Open position
        success, message = self.executor.open_position(
            symbol=decision['symbol'],
            action=decision['action'],
            current_price=decision['current_price'],
            stop_loss=decision['stop_loss'],
            take_profit=decision['take_profit'],
            confidence=decision['confidence'],
            risk_percent=self.risk_per_trade
        )

        decision['trade_executed'] = success
        decision['execution_message'] = message
        self.decisions_log.append(decision)

        if success:
            self.total_trades += 1

        return success, message

    def manage_open_positions(self):
        """Update and manage all open positions"""
        positions = self.executor.get_all_positions()

        for symbol, position in list(positions.items()):
            try:
                # Load latest data
                df = load_symbol_data(symbol)
                if df is None or len(df) == 0:
                    continue

                current_price = float(df['Close'].iloc[-1])

                # Update position with current price
                self.executor.update_position(symbol, current_price)

                # Check stop loss / take profit
                exit_reason = self.executor.check_stop_take(symbol)

                if exit_reason:
                    success, message = self.executor.close_position(
                        symbol, current_price, reason=exit_reason
                    )
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol}: {message}")

            except Exception as e:
                print(f"Error managing position {symbol}: {e}")

    def run_single_cycle(self):
        """Run one cycle of analysis for all symbols"""
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting analysis cycle")
        print(f"{'='*60}")

        # Manage existing positions
        self.manage_open_positions()

        # Analyze each symbol
        for symbol in self.symbols:
            try:
                print(f"\n[{symbol}] Analyzing...")

                # Analyze symbol
                decision, error = self.analyze_symbol(symbol)

                if error:
                    print(f"[{symbol}] Error: {error}")
                    continue

                self.total_signals += 1

                # Store last signal
                self.last_signals[symbol] = decision

                # Print decision
                print(f"[{symbol}] Signal: {decision['action']} | "
                      f"Confidence: {decision['confidence']:.1%} | "
                      f"ML: {decision['ml_direction']} ({decision['ml_confidence']:.1%}) | "
                      f"Tech: {decision['technical_strength']}/100 | "
                      f"Risk: {decision['risk_level']}")

                # Attempt to execute trade
                if decision['action'] in ['BUY', 'SELL']:
                    success, message = self.execute_trade_decision(decision)
                    print(f"[{symbol}] Trade Decision: {message}")

            except Exception as e:
                print(f"[{symbol}] Error: {str(e)}")

        # Print account summary
        account_info = self.executor.get_account_info()
        print(f"\n{'='*60}")
        print(f"Account Summary:")
        print(f"  Balance: ${account_info['balance']:.2f}")
        print(f"  Equity: ${account_info['equity']:.2f}")
        print(f"  Open P&L: ${account_info['open_pl']:.2f}")
        print(f"  Open Positions: {account_info['open_positions']}")
        print(f"  Total Trades: {account_info['total_trades']}")
        print(f"  Win Rate: {account_info['win_rate']:.1f}%")
        print(f"  ROI: {account_info['roi']:.2f}%")
        print(f"{'='*60}\n")

    def run(self, max_cycles=None):
        """Run the trading robot"""
        self.is_running = True
        self.start_time = datetime.now()
        cycle_count = 0

        print(f"\n{'='*60}")
        print(f"Trading Robot Started")
        print(f"Initial Balance: ${self.executor.initial_balance:.2f}")
        print(f"Risk per Trade: {self.risk_per_trade:.1%}")
        print(f"Min Confidence: {self.min_confidence:.1%}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Check Interval: {self.check_interval}s")
        print(f"{'='*60}\n")

        try:
            while self.is_running:
                # Run analysis cycle
                self.run_single_cycle()

                cycle_count += 1

                # Check if max cycles reached
                if max_cycles and cycle_count >= max_cycles:
                    print(f"\nMax cycles ({max_cycles}) reached. Stopping...")
                    break

                # Save state
                self.save_state()

                # Wait for next cycle
                if self.is_running:
                    print(f"Next check in {self.check_interval} seconds...")
                    time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\nRobot stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the trading robot"""
        self.is_running = False
        self.save_state()
        print("\nRobot stopped and state saved")

    def get_status(self):
        """Get current robot status"""
        account_info = self.executor.get_account_info()
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0

        return {
            'is_running': self.is_running,
            'runtime_hours': runtime,
            'total_signals': self.total_signals,
            'total_trades': self.total_trades,
            'account_info': account_info,
            'open_positions': len(self.executor.positions),
            'last_signals': self.last_signals,
            'trade_stats': self.executor.get_trade_statistics()
        }

    def get_decisions_log(self, limit=50):
        """Get recent decisions log"""
        return self.decisions_log[-limit:] if self.decisions_log else []

    def save_state(self):
        """Save robot state and executor state"""
        # Save executor state
        self.executor.save_state()

        # Save robot state
        data_dir = Path(__file__).parent.parent / 'data'
        data_dir.mkdir(exist_ok=True)

        state = {
            'symbols': self.symbols,
            'risk_per_trade': self.risk_per_trade,
            'min_confidence': self.min_confidence,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'total_signals': self.total_signals,
            'total_trades': self.total_trades,
            'decisions_log': [
                {
                    **decision,
                    'timestamp': decision['timestamp'].isoformat()
                }
                for decision in self.decisions_log[-100:]  # Keep last 100
            ]
        }

        with open(data_dir / 'trading_robot_config.json', 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load robot state"""
        # Load executor state
        self.executor.load_state()

        # Load robot state
        data_dir = Path(__file__).parent.parent / 'data'
        config_file = data_dir / 'trading_robot_config.json'

        if not config_file.exists():
            return False

        with open(config_file, 'r') as f:
            state = json.load(f)

        self.symbols = state.get('symbols', self.symbols)
        self.risk_per_trade = state.get('risk_per_trade', self.risk_per_trade)
        self.min_confidence = state.get('min_confidence', self.min_confidence)
        self.total_signals = state.get('total_signals', 0)
        self.total_trades = state.get('total_trades', 0)

        if state.get('start_time'):
            self.start_time = datetime.fromisoformat(state['start_time'])

        # Load decisions log
        self.decisions_log = [
            {
                **decision,
                'timestamp': datetime.fromisoformat(decision['timestamp'])
            }
            for decision in state.get('decisions_log', [])
        ]

        return True


# Singleton instance for use in Streamlit
_robot_instance = None

def get_robot_instance():
    """Get or create trading robot singleton"""
    global _robot_instance
    if _robot_instance is None:
        _robot_instance = TradingRobot()
        _robot_instance.load_state()
    return _robot_instance
