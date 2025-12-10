"""
Trade Executor - Simulated/Paper Trading Engine
Handles trade execution, position management, and P&L tracking
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class TradeExecutor:
    """Simulates trade execution and manages positions"""

    def __init__(self, initial_balance=10000.0, max_positions=5):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.positions = {}  # {symbol: position_info}
        self.trade_history = []
        self.equity_curve = []

    def get_account_info(self):
        """Get current account information"""
        open_pl = sum(pos['unrealized_pl'] for pos in self.positions.values())
        total_equity = self.balance + open_pl

        return {
            'balance': self.balance,
            'equity': total_equity,
            'open_positions': len(self.positions),
            'open_pl': open_pl,
            'total_trades': len(self.trade_history),
            'win_rate': self._calculate_win_rate(),
            'total_profit': total_equity - self.initial_balance,
            'roi': ((total_equity - self.initial_balance) / self.initial_balance) * 100
        }

    def can_open_position(self, symbol):
        """Check if we can open a new position"""
        if symbol in self.positions:
            return False, "Position already open for this symbol"
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"
        return True, "OK"

    def calculate_position_size(self, current_price, risk_percent=0.01, stop_loss_pct=0.02):
        """Calculate position size based on risk management rules"""
        account_info = self.get_account_info()
        risk_amount = account_info['equity'] * risk_percent

        # Calculate units based on stop loss
        if stop_loss_pct > 0:
            units = risk_amount / (current_price * stop_loss_pct)
        else:
            units = risk_amount / current_price

        position_value = units * current_price

        # Ensure we don't use more than 20% of equity per position
        max_position_value = account_info['equity'] * 0.20
        if position_value > max_position_value:
            units = max_position_value / current_price
            position_value = max_position_value

        return units, position_value

    def open_position(self, symbol, action, current_price, stop_loss, take_profit,
                     confidence, risk_percent=0.01):
        """Open a new trading position"""
        can_open, message = self.can_open_position(symbol)
        if not can_open:
            return False, message

        # Calculate position size
        stop_loss_pct = abs(current_price - stop_loss) / current_price
        units, position_value = self.calculate_position_size(
            current_price, risk_percent, stop_loss_pct
        )

        # Check if we have enough balance
        if position_value > self.balance * 0.9:  # Max 90% of balance
            return False, "Insufficient balance for position"

        # Create position
        position = {
            'symbol': symbol,
            'action': action,
            'entry_price': current_price,
            'current_price': current_price,
            'units': units,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'entry_time': datetime.now(),
            'unrealized_pl': 0.0,
            'unrealized_pl_pct': 0.0
        }

        self.positions[symbol] = position
        self.balance -= position_value  # Reserve balance

        # Log trade
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': symbol,
            'type': 'OPEN',
            'action': action,
            'price': current_price,
            'units': units,
            'value': position_value,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })

        return True, f"Position opened: {action} {units:.4f} units at {current_price:.5f}"

    def update_position(self, symbol, current_price):
        """Update position with current price and calculate P&L"""
        if symbol not in self.positions:
            return False, "Position not found"

        pos = self.positions[symbol]
        pos['current_price'] = current_price

        # Calculate unrealized P&L
        if pos['action'] == 'BUY':
            pl = (current_price - pos['entry_price']) * pos['units']
            pl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
        else:  # SELL
            pl = (pos['entry_price'] - current_price) * pos['units']
            pl_pct = ((pos['entry_price'] - current_price) / pos['entry_price']) * 100

        pos['unrealized_pl'] = pl
        pos['unrealized_pl_pct'] = pl_pct

        return True, f"Position updated: P&L = ${pl:.2f} ({pl_pct:.2f}%)"

    def check_stop_take(self, symbol):
        """Check if stop loss or take profit is hit"""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        current_price = pos['current_price']

        if pos['action'] == 'BUY':
            # Check stop loss (price went down)
            if current_price <= pos['stop_loss']:
                return 'STOP_LOSS'
            # Check take profit (price went up)
            if current_price >= pos['take_profit']:
                return 'TAKE_PROFIT'
        else:  # SELL
            # Check stop loss (price went up)
            if current_price >= pos['stop_loss']:
                return 'STOP_LOSS'
            # Check take profit (price went down)
            if current_price <= pos['take_profit']:
                return 'TAKE_PROFIT'

        return None

    def close_position(self, symbol, current_price, reason='MANUAL'):
        """Close an open position"""
        if symbol not in self.positions:
            return False, "Position not found"

        pos = self.positions[symbol]

        # Update with final price
        self.update_position(symbol, current_price)

        # Calculate final P&L
        realized_pl = pos['unrealized_pl']
        realized_pl_pct = pos['unrealized_pl_pct']

        # Return position value + P&L to balance
        self.balance += pos['position_value'] + realized_pl

        # Log trade
        self.trade_history.append({
            'time': datetime.now(),
            'symbol': symbol,
            'type': 'CLOSE',
            'action': pos['action'],
            'price': current_price,
            'units': pos['units'],
            'value': pos['position_value'],
            'pl': realized_pl,
            'pl_pct': realized_pl_pct,
            'reason': reason,
            'holding_time': (datetime.now() - pos['entry_time']).total_seconds() / 3600
        })

        # Record equity
        account_info = self.get_account_info()
        self.equity_curve.append({
            'time': datetime.now(),
            'equity': account_info['equity'],
            'balance': self.balance
        })

        # Remove position
        del self.positions[symbol]

        return True, f"Position closed: {reason} - P&L = ${realized_pl:.2f} ({realized_pl_pct:.2f}%)"

    def get_position(self, symbol):
        """Get position information for a symbol"""
        return self.positions.get(symbol, None)

    def get_all_positions(self):
        """Get all open positions"""
        return self.positions.copy()

    def _calculate_win_rate(self):
        """Calculate win rate from closed trades"""
        closed_trades = [t for t in self.trade_history if t['type'] == 'CLOSE']
        if len(closed_trades) == 0:
            return 0.0

        winning_trades = sum(1 for t in closed_trades if t.get('pl', 0) > 0)
        return (winning_trades / len(closed_trades)) * 100

    def get_trade_statistics(self):
        """Calculate detailed trade statistics"""
        closed_trades = [t for t in self.trade_history if t['type'] == 'CLOSE']

        if len(closed_trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_holding_time': 0.0
            }

        wins = [t for t in closed_trades if t.get('pl', 0) > 0]
        losses = [t for t in closed_trades if t.get('pl', 0) < 0]

        total_wins = sum(t.get('pl', 0) for t in wins)
        total_losses = abs(sum(t.get('pl', 0) for t in losses))

        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': (len(wins) / len(closed_trades)) * 100,
            'avg_win': total_wins / len(wins) if len(wins) > 0 else 0,
            'avg_loss': total_losses / len(losses) if len(losses) > 0 else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0,
            'largest_win': max([t.get('pl', 0) for t in wins]) if wins else 0,
            'largest_loss': min([t.get('pl', 0) for t in losses]) if losses else 0,
            'avg_holding_time': sum(t.get('holding_time', 0) for t in closed_trades) / len(closed_trades)
        }

    def save_state(self, filename='trading_robot_state.json'):
        """Save robot state to file"""
        data_dir = Path(__file__).parent.parent / 'data'
        data_dir.mkdir(exist_ok=True)
        filepath = data_dir / filename

        # Convert datetime objects to strings
        state = {
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'positions': {
                symbol: {
                    **pos,
                    'entry_time': pos['entry_time'].isoformat()
                }
                for symbol, pos in self.positions.items()
            },
            'trade_history': [
                {
                    **trade,
                    'time': trade['time'].isoformat()
                }
                for trade in self.trade_history
            ],
            'equity_curve': [
                {
                    **eq,
                    'time': eq['time'].isoformat()
                }
                for eq in self.equity_curve
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filename='trading_robot_state.json'):
        """Load robot state from file"""
        data_dir = Path(__file__).parent.parent / 'data'
        filepath = data_dir / filename

        if not filepath.exists():
            return False

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.balance = state['balance']
        self.initial_balance = state['initial_balance']

        # Convert string timestamps back to datetime
        self.positions = {
            symbol: {
                **pos,
                'entry_time': datetime.fromisoformat(pos['entry_time'])
            }
            for symbol, pos in state['positions'].items()
        }

        self.trade_history = [
            {
                **trade,
                'time': datetime.fromisoformat(trade['time'])
            }
            for trade in state['trade_history']
        ]

        self.equity_curve = [
            {
                **eq,
                'time': datetime.fromisoformat(eq['time'])
            }
            for eq in state['equity_curve']
        ]

        return True
