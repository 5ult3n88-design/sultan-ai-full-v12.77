#!/usr/bin/env python3
"""
Standalone Trading Robot Runner
Run the autonomous trading robot from command line
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.trading_robot import TradingRobot
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run Autonomous Trading Robot')

    parser.add_argument(
        '--balance',
        type=float,
        default=10000.0,
        help='Initial balance in USD (default: 10000)'
    )

    parser.add_argument(
        '--risk',
        type=float,
        default=1.0,
        help='Risk per trade in percent (default: 1.0)'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=65.0,
        help='Minimum confidence to enter trade in percent (default: 65)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Check interval in seconds (default: 300 = 5 minutes)'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['EURUSD=X', 'GBPUSD=X', 'XAUUSD=X', 'AAPL', 'GOOGL'],
        help='Symbols to trade (default: EURUSD=X GBPUSD=X XAUUSD=X AAPL GOOGL)'
    )

    parser.add_argument(
        '--cycles',
        type=int,
        default=None,
        help='Maximum number of cycles to run (default: unlimited)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run one test cycle and exit'
    )

    args = parser.parse_args()

    # Create robot
    print("\n" + "="*60)
    print("Autonomous Trading Robot")
    print("="*60)

    robot = TradingRobot(
        symbols=args.symbols,
        initial_balance=args.balance,
        risk_per_trade=args.risk / 100,
        min_confidence=args.confidence / 100,
        check_interval=args.interval
    )

    # Try to load saved state
    if robot.load_state():
        print("\n✅ Loaded previous robot state")
        account_info = robot.executor.get_account_info()
        print(f"   Balance: ${account_info['balance']:.2f}")
        print(f"   Equity: ${account_info['equity']:.2f}")
        print(f"   Total Trades: {account_info['total_trades']}")
        print(f"   Win Rate: {account_info['win_rate']:.1f}%")
    else:
        print("\n🆕 Starting fresh robot")

    # Run robot
    if args.test:
        print("\n🧪 Running test cycle...\n")
        robot.run_single_cycle()
        print("\nTest cycle completed!")
    else:
        print("\n🚀 Starting robot...\n")
        robot.run(max_cycles=args.cycles)

    # Final summary
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)

    account_info = robot.executor.get_account_info()
    trade_stats = robot.executor.get_trade_statistics()

    print(f"\nAccount:")
    print(f"  Initial Balance: ${robot.executor.initial_balance:.2f}")
    print(f"  Final Balance: ${account_info['balance']:.2f}")
    print(f"  Final Equity: ${account_info['equity']:.2f}")
    print(f"  Total Profit: ${account_info['total_profit']:.2f}")
    print(f"  ROI: {account_info['roi']:.2f}%")

    print(f"\nTrading:")
    print(f"  Total Trades: {trade_stats['total_trades']}")
    print(f"  Winning Trades: {trade_stats['winning_trades']}")
    print(f"  Losing Trades: {trade_stats['losing_trades']}")
    print(f"  Win Rate: {trade_stats['win_rate']:.1f}%")
    print(f"  Profit Factor: {trade_stats['profit_factor']:.2f}")
    print(f"  Avg Win: ${trade_stats['avg_win']:.2f}")
    print(f"  Avg Loss: ${trade_stats['avg_loss']:.2f}")

    print(f"\nSignals:")
    print(f"  Total Signals Generated: {robot.total_signals}")
    print(f"  Signals Converted to Trades: {robot.total_trades}")
    print(f"  Conversion Rate: {(robot.total_trades / robot.total_signals * 100) if robot.total_signals > 0 else 0:.1f}%")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
