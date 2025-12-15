"""
Trading Robot with MT5 Integration
Connects the autonomous robot to MetaTrader 5 for live/demo trading
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from trading_robot import TradingRobot
from mt5_connector import MT5Connector
from datetime import datetime


class MT5TradingRobot(TradingRobot):
    """Trading Robot with MT5 live trading capability"""

    def __init__(self, symbols=None, mt5_account=None, mt5_password=None,
                 mt5_server=None, risk_per_trade=0.01, min_confidence=0.65,
                 check_interval=300, use_mt5=True):
        """
        Initialize MT5 Trading Robot

        Args:
            symbols: List of symbols to trade
            mt5_account: MT5 account number (optional if already logged in)
            mt5_password: MT5 password
            mt5_server: MT5 broker server
            risk_per_trade: Risk per trade as decimal (0.01 = 1%)
            min_confidence: Minimum confidence to trade (0.65 = 65%)
            check_interval: Seconds between checks
            use_mt5: Use MT5 for real trading (False = paper trading only)
        """
        # Initialize parent robot (paper trading)
        super().__init__(
            symbols=symbols,
            initial_balance=0,  # Will use MT5 balance
            risk_per_trade=risk_per_trade,
            min_confidence=min_confidence,
            check_interval=check_interval
        )

        self.use_mt5 = use_mt5
        self.mt5 = None
        self.mt5_account = mt5_account
        self.mt5_password = mt5_password
        self.mt5_server = mt5_server

        if use_mt5:
            self._connect_mt5()

    def _connect_mt5(self):
        """Connect to MT5"""
        print("\n🔌 Connecting to MetaTrader 5...")

        self.mt5 = MT5Connector(
            account=self.mt5_account,
            password=self.mt5_password,
            server=self.mt5_server
        )

        if self.mt5.connect():
            print("✅ Connected to MT5 successfully!")

            # Get account info
            account = self.mt5.get_account_info()
            if account:
                print(f"   Account: {account['login']}")
                print(f"   Balance: ${account['balance']:.2f}")
                print(f"   Equity: ${account['equity']:.2f}")
                print(f"   Leverage: 1:{account['leverage']}")

            return True
        else:
            print("❌ Failed to connect to MT5")
            print("   Falling back to paper trading mode")
            self.use_mt5 = False
            return False

    def execute_trade_decision(self, decision):
        """Execute trade - MT5 if connected, paper trading if not"""

        if self.use_mt5 and self.mt5 and self.mt5.connected:
            return self._execute_mt5_trade(decision)
        else:
            # Use paper trading from parent class
            return super().execute_trade_decision(decision)

    def _execute_mt5_trade(self, decision):
        """Execute trade on MT5"""
        should_trade, reason = self.should_enter_trade(decision)

        decision['should_trade'] = should_trade
        decision['trade_reason'] = reason

        if not should_trade:
            self.decisions_log.append(decision)
            return False, reason

        # Get account info for risk calculation
        account = self.mt5.get_account_info()
        if not account:
            return False, "Failed to get account info"

        risk_amount = account['equity'] * self.risk_per_trade

        # Open position on MT5
        success, message = self.mt5.open_position(
            symbol=decision['symbol'],
            action=decision['action'],
            risk_amount=risk_amount,
            stop_loss=decision['stop_loss'],
            take_profit=decision['take_profit'],
            comment=f"Robot {decision['confidence']:.1%}"
        )

        decision['trade_executed'] = success
        decision['execution_message'] = message
        decision['mt5_trade'] = True
        self.decisions_log.append(decision)

        if success:
            self.total_trades += 1
            print(f"   💰 MT5 Trade: {message}")

        return success, message

    def manage_open_positions(self):
        """Manage positions - MT5 or paper trading"""
        if self.use_mt5 and self.mt5 and self.mt5.connected:
            self._manage_mt5_positions()
        else:
            # Use paper trading position management
            super().manage_open_positions()

    def _manage_mt5_positions(self):
        """Manage MT5 positions"""
        positions = self.mt5.get_open_positions()

        for pos in positions:
            # Check if we should close based on robot logic
            # (MT5 will automatically handle SL/TP)

            print(f"   📊 {pos['symbol']}: {pos['type']} {pos['volume']} lots, "
                  f"P&L: ${pos['profit']:+.2f}")

    def get_status(self):
        """Get robot status including MT5 info"""
        status = super().get_status()

        if self.use_mt5 and self.mt5 and self.mt5.connected:
            account = self.mt5.get_account_info()
            if account:
                status['mt5_connected'] = True
                status['mt5_account'] = {
                    'balance': account['balance'],
                    'equity': account['equity'],
                    'profit': account['profit'],
                    'margin_free': account['margin_free']
                }

                # Get MT5 positions
                positions = self.mt5.get_open_positions()
                status['mt5_positions'] = positions
        else:
            status['mt5_connected'] = False

        return status

    def stop(self):
        """Stop robot and disconnect from MT5"""
        super().stop()

        if self.mt5 and self.mt5.connected:
            self.mt5.disconnect()
            print("Disconnected from MT5")


def main():
    """Run MT5 Trading Robot"""
    import argparse

    parser = argparse.ArgumentParser(description='MT5 Trading Robot')

    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['EURUSD', 'GBPUSD', 'XAUUSD'],
                       help='Symbols to trade')

    parser.add_argument('--account', type=int,
                       help='MT5 account number (optional if already logged in)')

    parser.add_argument('--password', type=str,
                       help='MT5 password')

    parser.add_argument('--server', type=str,
                       help='MT5 broker server')

    parser.add_argument('--confidence', type=float, default=70.0,
                       help='Minimum confidence % (default: 70)')

    parser.add_argument('--risk', type=float, default=1.0,
                       help='Risk per trade % (default: 1.0)')

    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (default: 300)')

    parser.add_argument('--paper', action='store_true',
                       help='Use paper trading mode (no MT5)')

    parser.add_argument('--test', action='store_true',
                       help='Run one cycle and exit')

    parser.add_argument('--cycles', type=int,
                       help='Max cycles to run')

    args = parser.parse_args()

    # Create robot
    print("\n" + "="*60)
    print("MT5 Trading Robot")
    print("="*60)

    robot = MT5TradingRobot(
        symbols=args.symbols,
        mt5_account=args.account,
        mt5_password=args.password,
        mt5_server=args.server,
        risk_per_trade=args.risk / 100,
        min_confidence=args.confidence / 100,
        check_interval=args.interval,
        use_mt5=not args.paper
    )

    # Run robot
    if args.test:
        print("\n🧪 Running test cycle...\n")
        robot.run_single_cycle()
        print("\nTest complete!")
    else:
        robot.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
