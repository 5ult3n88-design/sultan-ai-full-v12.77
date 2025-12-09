"""
Sultan AI Backtesting Framework
================================
Comprehensive backtesting system for evaluating trading strategy performance.

Features:
- Historical signal replay
- Realistic slippage and spread modeling
- Performance metrics (Win Rate, Sharpe, Drawdown, etc.)
- Trade logging and analysis
- Visual performance reports
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

# Import ML and analytics modules
from analytics import calculate_advanced_indicators, calculate_strength_score
from ml_model import predict_price_movement, get_trading_recommendation, calculate_entry_exit_levels


class Trade:
    """Represents a single trade with entry, exit, and performance data"""

    def __init__(self, trade_id: int, symbol: str, direction: str,
                 entry_time: datetime, entry_price: float,
                 stop_loss: float, take_profit: float,
                 confidence: float, position_size: float = 1.0):
        self.trade_id = trade_id
        self.symbol = symbol
        self.direction = direction  # 'BUY' or 'SELL'
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence = confidence
        self.position_size = position_size

        # Exit details (filled when trade closes)
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None  # 'take_profit', 'stop_loss', 'signal_change', 'timeout'

        # Performance
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.is_winner = False
        self.duration_hours = 0

    def close(self, exit_time: datetime, exit_price: float, exit_reason: str):
        """Close the trade and calculate performance"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calculate P&L
        if self.direction == 'BUY':
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:  # SELL
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100

        self.pnl = self.pnl_pct * self.position_size
        self.is_winner = self.pnl > 0

        # Calculate duration
        if isinstance(exit_time, datetime) and isinstance(self.entry_time, datetime):
            self.duration_hours = (exit_time - self.entry_time).total_seconds() / 3600
        else:
            self.duration_hours = 0

    def to_dict(self) -> dict:
        """Convert trade to dictionary for logging"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_time': str(self.entry_time),
            'entry_price': self.entry_price,
            'exit_time': str(self.exit_time),
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'confidence': self.confidence,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'is_winner': self.is_winner,
            'duration_hours': self.duration_hours
        }


class BacktestResult:
    """Contains all backtesting results and metrics"""

    def __init__(self, symbol: str, start_date: datetime, end_date: datetime):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.metrics: Dict = {}

    def calculate_metrics(self, initial_capital: float = 10000.0):
        """Calculate all performance metrics"""
        if not self.trades:
            self.metrics = self._empty_metrics()
            return

        # Basic counts
        total_trades = len(self.trades)
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]

        # Win rate
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        # P&L calculations
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0

        # Average win/loss
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0

        # Profit factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Win/Loss ratio (average win size / average loss size)
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0

        # Expectancy (average profit per trade)
        expectancy = total_pnl / total_trades if total_trades > 0 else 0

        # Calculate equity curve and drawdown
        equity = initial_capital
        peak_equity = equity
        max_drawdown = 0
        max_drawdown_pct = 0
        self.equity_curve = [equity]

        for trade in self.trades:
            equity += trade.pnl * (initial_capital / 100)  # Convert % to $
            self.equity_curve.append(equity)

            if equity > peak_equity:
                peak_equity = equity

            drawdown = peak_equity - equity
            drawdown_pct = (drawdown / peak_equity) * 100 if peak_equity > 0 else 0

            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
                max_drawdown = drawdown

        # Total return
        total_return = (equity - initial_capital) / initial_capital * 100

        # Sharpe Ratio (assuming daily returns, annualized)
        returns = [t.pnl_pct for t in self.trades]
        if len(returns) > 1 and np.std(returns) > 0:
            # Approximate annualization based on trade frequency
            avg_duration = np.mean([t.duration_hours for t in self.trades]) if self.trades else 24
            trades_per_year = 8760 / avg_duration if avg_duration > 0 else 252  # 8760 hours in year
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(min(trades_per_year, 252))
        else:
            sharpe = 0

        # Sortino Ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if len(negative_returns) > 1 and np.std(negative_returns) > 0:
            avg_duration = np.mean([t.duration_hours for t in self.trades]) if self.trades else 24
            trades_per_year = 8760 / avg_duration if avg_duration > 0 else 252
            sortino = (np.mean(returns) / np.std(negative_returns)) * np.sqrt(min(trades_per_year, 252))
        else:
            sortino = sharpe  # Fall back to Sharpe if no negative returns

        # Calmar Ratio (return / max drawdown)
        calmar = total_return / max_drawdown_pct if max_drawdown_pct > 0 else float('inf') if total_return > 0 else 0

        # Consecutive wins/losses
        max_consecutive_wins = self._max_consecutive(True)
        max_consecutive_losses = self._max_consecutive(False)

        # Average trade duration
        avg_duration = np.mean([t.duration_hours for t in self.trades]) if self.trades else 0

        # Confidence accuracy (how accurate are high-confidence predictions?)
        high_conf_trades = [t for t in self.trades if t.confidence > 0.7]
        high_conf_accuracy = len([t for t in high_conf_trades if t.is_winner]) / len(high_conf_trades) * 100 if high_conf_trades else 0

        low_conf_trades = [t for t in self.trades if t.confidence <= 0.7]
        low_conf_accuracy = len([t for t in low_conf_trades if t.is_winner]) / len(low_conf_trades) * 100 if low_conf_trades else 0

        self.metrics = {
            # Trade counts
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),

            # Win rate
            'win_rate': round(win_rate, 2),

            # P&L
            'total_pnl_pct': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),

            # Ratios
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            'win_loss_ratio': round(win_loss_ratio, 2) if win_loss_ratio != float('inf') else 999.99,
            'expectancy': round(expectancy, 4),

            # Risk metrics
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'max_drawdown': round(max_drawdown, 2),
            'total_return': round(total_return, 2),

            # Risk-adjusted returns
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'calmar_ratio': round(calmar, 2) if calmar != float('inf') else 999.99,

            # Streaks
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,

            # Timing
            'avg_trade_duration_hours': round(avg_duration, 1),

            # Confidence analysis
            'high_confidence_accuracy': round(high_conf_accuracy, 2),
            'low_confidence_accuracy': round(low_conf_accuracy, 2),
            'high_confidence_trades': len(high_conf_trades),
            'low_confidence_trades': len(low_conf_trades),

            # Final equity
            'final_equity': round(equity, 2),
            'initial_capital': initial_capital
        }

    def _max_consecutive(self, is_winner: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_streak = 0
        current_streak = 0

        for trade in self.trades:
            if trade.is_winner == is_winner:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _empty_metrics(self) -> dict:
        """Return empty metrics when no trades"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl_pct': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'win_loss_ratio': 0,
            'expectancy': 0,
            'max_drawdown_pct': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_trade_duration_hours': 0,
            'high_confidence_accuracy': 0,
            'low_confidence_accuracy': 0,
            'high_confidence_trades': 0,
            'low_confidence_trades': 0,
            'final_equity': 10000,
            'initial_capital': 10000
        }

    def get_summary(self) -> str:
        """Generate human-readable summary"""
        m = self.metrics

        # Determine performance rating
        if m['win_rate'] >= 55 and m['profit_factor'] >= 1.5 and m['sharpe_ratio'] >= 1.0:
            rating = "EXCELLENT"
            rating_emoji = "🌟"
        elif m['win_rate'] >= 50 and m['profit_factor'] >= 1.2:
            rating = "GOOD"
            rating_emoji = "✅"
        elif m['win_rate'] >= 45 and m['profit_factor'] >= 1.0:
            rating = "FAIR"
            rating_emoji = "⚠️"
        else:
            rating = "NEEDS IMPROVEMENT"
            rating_emoji = "❌"

        summary = f"""
{'='*60}
BACKTEST RESULTS: {self.symbol}
{'='*60}
Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
Overall Rating: {rating_emoji} {rating}

📊 TRADE STATISTICS
-------------------
Total Trades: {m['total_trades']}
Winners: {m['winning_trades']} | Losers: {m['losing_trades']}
Win Rate: {m['win_rate']}%

💰 PROFIT & LOSS
----------------
Total P&L: {m['total_pnl_pct']:+.2f}%
Gross Profit: +{m['gross_profit']:.2f}% | Gross Loss: -{m['gross_loss']:.2f}%
Avg Win: +{m['avg_win']:.2f}% | Avg Loss: -{m['avg_loss']:.2f}%
Profit Factor: {m['profit_factor']}
Expectancy: {m['expectancy']:.4f}% per trade

📈 RISK METRICS
---------------
Max Drawdown: {m['max_drawdown_pct']:.2f}%
Total Return: {m['total_return']:+.2f}%
Final Equity: ${m['final_equity']:,.2f} (from ${m['initial_capital']:,.2f})

📐 RISK-ADJUSTED RETURNS
------------------------
Sharpe Ratio: {m['sharpe_ratio']} {'(Good)' if m['sharpe_ratio'] >= 1 else '(Needs work)'}
Sortino Ratio: {m['sortino_ratio']}
Calmar Ratio: {m['calmar_ratio']}

🎯 CONFIDENCE ANALYSIS
----------------------
High Confidence (>70%) Trades: {m['high_confidence_trades']} | Accuracy: {m['high_confidence_accuracy']}%
Low Confidence (<=70%) Trades: {m['low_confidence_trades']} | Accuracy: {m['low_confidence_accuracy']}%

📉 STREAKS
----------
Max Consecutive Wins: {m['max_consecutive_wins']}
Max Consecutive Losses: {m['max_consecutive_losses']}
Avg Trade Duration: {m['avg_trade_duration_hours']:.1f} hours
{'='*60}
"""
        return summary

    def to_dict(self) -> dict:
        """Convert result to dictionary"""
        return {
            'symbol': self.symbol,
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'metrics': self.metrics,
            'trades': [t.to_dict() for t in self.trades],
            'equity_curve': self.equity_curve
        }


class Backtester:
    """
    Main backtesting engine for Sultan AI trading strategies.

    Features:
    - Walk-forward testing (no lookahead bias)
    - Realistic execution with slippage
    - Multiple exit conditions (SL, TP, timeout, signal change)
    - Comprehensive performance tracking
    """

    def __init__(self,
                 spread_pips: float = 2.0,
                 slippage_pips: float = 1.0,
                 commission_pct: float = 0.0,
                 min_confidence: float = 0.5,
                 max_trades_per_day: int = 5,
                 trade_timeout_hours: int = 48):
        """
        Initialize backtester with trading parameters.

        Args:
            spread_pips: Bid-ask spread in pips
            slippage_pips: Expected slippage in pips
            commission_pct: Commission as percentage of trade
            min_confidence: Minimum confidence to take trade
            max_trades_per_day: Maximum trades allowed per day
            trade_timeout_hours: Auto-close trade after X hours
        """
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_pct = commission_pct
        self.min_confidence = min_confidence
        self.max_trades_per_day = max_trades_per_day
        self.trade_timeout_hours = trade_timeout_hours

        # Pip values for different symbols
        self.pip_values = {
            'XAUUSD': 0.1,    # Gold: 0.1 = 1 pip
            'XAGUSD': 0.01,   # Silver: 0.01 = 1 pip
            'USDJPY': 0.01,   # JPY pairs: 0.01 = 1 pip
            'EURJPY': 0.01,
            'GBPJPY': 0.01,
            'AUDJPY': 0.01,
            'DEFAULT': 0.0001  # Standard forex pairs: 0.0001 = 1 pip
        }

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value for a symbol"""
        for key in self.pip_values:
            if key in symbol.upper():
                return self.pip_values[key]
        return self.pip_values['DEFAULT']

    def apply_costs(self, price: float, direction: str, symbol: str, is_entry: bool = True) -> float:
        """Apply spread and slippage costs to price"""
        pip_value = self.get_pip_value(symbol)
        total_cost_pips = self.spread_pips / 2 + self.slippage_pips

        if is_entry:
            if direction == 'BUY':
                return price + (total_cost_pips * pip_value)  # Worse entry for buy
            else:
                return price - (total_cost_pips * pip_value)  # Worse entry for sell
        else:
            if direction == 'BUY':
                return price - (total_cost_pips * pip_value)  # Worse exit for buy
            else:
                return price + (total_cost_pips * pip_value)  # Worse exit for sell

    def run_backtest(self,
                     df: pd.DataFrame,
                     symbol: str,
                     lookback_periods: int = 100,
                     initial_capital: float = 10000.0,
                     verbose: bool = False) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol (e.g., 'EURUSD')
            lookback_periods: Periods needed for indicators
            initial_capital: Starting capital
            verbose: Print progress updates

        Returns:
            BacktestResult with all metrics and trades
        """
        if len(df) < lookback_periods + 50:
            raise ValueError(f"Insufficient data: need at least {lookback_periods + 50} periods, got {len(df)}")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            elif 'Datetime' in df.columns:
                df = df.set_index('Datetime')
            df.index = pd.to_datetime(df.index)

        # Initialize result
        start_date = df.index[lookback_periods]
        end_date = df.index[-1]
        result = BacktestResult(symbol, start_date, end_date)

        # Track state
        current_trade: Optional[Trade] = None
        trade_count = 0
        daily_trades = {}  # Track trades per day

        if verbose:
            print(f"Starting backtest for {symbol}")
            print(f"Period: {start_date} to {end_date}")
            print(f"Data points: {len(df)}")
            print("-" * 50)

        # Walk through each bar (skip lookback period)
        for i in range(lookback_periods, len(df)):
            current_time = df.index[i]
            current_bar = df.iloc[i]
            current_price = current_bar['Close']
            current_high = current_bar['High']
            current_low = current_bar['Low']

            # Get historical data up to current point (no lookahead)
            historical_df = df.iloc[:i+1].copy()

            # Calculate indicators
            try:
                historical_df = calculate_advanced_indicators(historical_df)
            except Exception as e:
                continue  # Skip if indicator calculation fails

            # Check if we have an open trade
            if current_trade is not None:
                # Check exit conditions
                exit_price = None
                exit_reason = None

                # 1. Check stop loss
                if current_trade.direction == 'BUY':
                    if current_low <= current_trade.stop_loss:
                        exit_price = self.apply_costs(current_trade.stop_loss, 'BUY', symbol, is_entry=False)
                        exit_reason = 'stop_loss'
                else:  # SELL
                    if current_high >= current_trade.stop_loss:
                        exit_price = self.apply_costs(current_trade.stop_loss, 'SELL', symbol, is_entry=False)
                        exit_reason = 'stop_loss'

                # 2. Check take profit
                if exit_reason is None:
                    if current_trade.direction == 'BUY':
                        if current_high >= current_trade.take_profit:
                            exit_price = self.apply_costs(current_trade.take_profit, 'BUY', symbol, is_entry=False)
                            exit_reason = 'take_profit'
                    else:  # SELL
                        if current_low <= current_trade.take_profit:
                            exit_price = self.apply_costs(current_trade.take_profit, 'SELL', symbol, is_entry=False)
                            exit_reason = 'take_profit'

                # 3. Check timeout
                if exit_reason is None and current_trade.entry_time is not None:
                    trade_duration = (current_time - current_trade.entry_time).total_seconds() / 3600
                    if trade_duration >= self.trade_timeout_hours:
                        exit_price = self.apply_costs(current_price, current_trade.direction, symbol, is_entry=False)
                        exit_reason = 'timeout'

                # 4. Check signal change (optional - more aggressive)
                if exit_reason is None:
                    try:
                        # Get new prediction
                        ml_pred = predict_price_movement(historical_df, symbol=symbol)
                        new_direction = ml_pred.get('direction', 'NEUTRAL')

                        # Exit if signal completely reverses with high confidence
                        if (current_trade.direction == 'BUY' and new_direction == 'DOWN' and
                            ml_pred.get('confidence', 0) > 0.7):
                            exit_price = self.apply_costs(current_price, 'BUY', symbol, is_entry=False)
                            exit_reason = 'signal_reversal'
                        elif (current_trade.direction == 'SELL' and new_direction == 'UP' and
                              ml_pred.get('confidence', 0) > 0.7):
                            exit_price = self.apply_costs(current_price, 'SELL', symbol, is_entry=False)
                            exit_reason = 'signal_reversal'
                    except:
                        pass

                # Close trade if exit condition met
                if exit_reason is not None:
                    current_trade.close(current_time, exit_price, exit_reason)
                    result.trades.append(current_trade)

                    if verbose and len(result.trades) % 10 == 0:
                        print(f"Trade #{len(result.trades)}: {current_trade.direction} | "
                              f"Exit: {exit_reason} | P&L: {current_trade.pnl_pct:+.2f}%")

                    current_trade = None

            # Look for new trade entry (only if no open trade)
            if current_trade is None:
                # Check daily trade limit
                day_key = current_time.strftime('%Y-%m-%d')
                if daily_trades.get(day_key, 0) >= self.max_trades_per_day:
                    continue

                try:
                    # Get prediction
                    ml_pred = predict_price_movement(historical_df, symbol=symbol)
                    direction = ml_pred.get('direction', 'NEUTRAL')
                    confidence = ml_pred.get('confidence', 0.5)

                    # Only trade if confidence meets threshold
                    if confidence < self.min_confidence:
                        continue

                    # Only trade if we have a directional signal
                    if direction not in ['UP', 'DOWN']:
                        continue

                    # Get recommendation and levels
                    recommendation = get_trading_recommendation(historical_df, ml_pred)
                    action = recommendation.get('action', 'HOLD')

                    if action not in ['BUY', 'SELL']:
                        continue

                    levels = calculate_entry_exit_levels(historical_df, recommendation)

                    # Create trade
                    trade_count += 1
                    entry_price = self.apply_costs(current_price, action, symbol, is_entry=True)

                    current_trade = Trade(
                        trade_id=trade_count,
                        symbol=symbol,
                        direction=action,
                        entry_time=current_time,
                        entry_price=entry_price,
                        stop_loss=levels.get('stop_loss', entry_price * 0.99 if action == 'BUY' else entry_price * 1.01),
                        take_profit=levels.get('take_profit', entry_price * 1.02 if action == 'BUY' else entry_price * 0.98),
                        confidence=confidence
                    )

                    # Update daily trade count
                    daily_trades[day_key] = daily_trades.get(day_key, 0) + 1

                except Exception as e:
                    continue

        # Close any remaining open trade at end
        if current_trade is not None:
            exit_price = self.apply_costs(df['Close'].iloc[-1], current_trade.direction, symbol, is_entry=False)
            current_trade.close(df.index[-1], exit_price, 'end_of_data')
            result.trades.append(current_trade)

        # Calculate metrics
        result.calculate_metrics(initial_capital)

        if verbose:
            print("-" * 50)
            print(f"Backtest complete! Total trades: {len(result.trades)}")
            print(result.get_summary())

        return result

    def run_walk_forward(self,
                         df: pd.DataFrame,
                         symbol: str,
                         train_periods: int = 500,
                         test_periods: int = 100,
                         step_size: int = 50,
                         verbose: bool = False) -> List[BacktestResult]:
        """
        Run walk-forward analysis with rolling training/testing windows.

        This is more realistic as it simulates model retraining over time.

        Args:
            df: Full historical DataFrame
            symbol: Trading symbol
            train_periods: Size of training window
            test_periods: Size of testing window
            step_size: How many periods to step forward
            verbose: Print progress

        Returns:
            List of BacktestResult for each walk-forward window
        """
        results = []
        total_periods = len(df)
        min_required = train_periods + test_periods

        if total_periods < min_required:
            raise ValueError(f"Insufficient data: need {min_required} periods, got {total_periods}")

        # Walk forward through data
        start_idx = 0
        window_num = 0

        while start_idx + min_required <= total_periods:
            window_num += 1
            train_end = start_idx + train_periods
            test_end = min(train_end + test_periods, total_periods)

            # Get test window data (training data is used internally by predictions)
            test_df = df.iloc[start_idx:test_end].copy()

            if verbose:
                print(f"\nWindow {window_num}: Training {start_idx}-{train_end}, Testing {train_end}-{test_end}")

            try:
                result = self.run_backtest(
                    test_df,
                    symbol,
                    lookback_periods=train_periods,
                    verbose=False
                )
                results.append(result)

                if verbose:
                    print(f"  Trades: {result.metrics['total_trades']} | "
                          f"Win Rate: {result.metrics['win_rate']}% | "
                          f"P&L: {result.metrics['total_pnl_pct']:+.2f}%")
            except Exception as e:
                if verbose:
                    print(f"  Window failed: {e}")

            start_idx += step_size

        return results

    def aggregate_results(self, results: List[BacktestResult]) -> Dict:
        """Aggregate metrics across multiple backtest results"""
        if not results:
            return {}

        all_trades = []
        for r in results:
            all_trades.extend(r.trades)

        if not all_trades:
            return {'error': 'No trades across all windows'}

        # Aggregate metrics
        total_trades = len(all_trades)
        winners = [t for t in all_trades if t.is_winner]

        aggregated = {
            'total_windows': len(results),
            'total_trades': total_trades,
            'overall_win_rate': round(len(winners) / total_trades * 100, 2) if total_trades > 0 else 0,
            'avg_win_rate': round(np.mean([r.metrics['win_rate'] for r in results]), 2),
            'std_win_rate': round(np.std([r.metrics['win_rate'] for r in results]), 2),
            'avg_profit_factor': round(np.mean([min(r.metrics['profit_factor'], 10) for r in results]), 2),
            'total_pnl': round(sum(r.metrics['total_pnl_pct'] for r in results), 2),
            'avg_sharpe': round(np.mean([r.metrics['sharpe_ratio'] for r in results]), 2),
            'worst_drawdown': round(max(r.metrics['max_drawdown_pct'] for r in results), 2),
            'best_window_pnl': round(max(r.metrics['total_pnl_pct'] for r in results), 2),
            'worst_window_pnl': round(min(r.metrics['total_pnl_pct'] for r in results), 2),
        }

        return aggregated


def save_backtest_results(result: BacktestResult, filepath: str):
    """Save backtest results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def load_backtest_results(filepath: str) -> Dict:
    """Load backtest results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


# Quick test function
def quick_backtest(symbol: str = 'EURUSD',
                   days: int = 90,
                   verbose: bool = True) -> BacktestResult:
    """
    Quick backtest for a symbol using available data.

    Args:
        symbol: Trading symbol
        days: Number of days to backtest
        verbose: Print results

    Returns:
        BacktestResult
    """
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')

    # Load data
    csv_path = os.path.join(data_dir, f'{symbol}.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No data file found for {symbol} at {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse dates
    if 'Datetime' in df.columns:
        df['Date'] = pd.to_datetime(df['Datetime'])
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    df = df.set_index('Date')

    # Get last N days
    if days and len(df) > days:
        df = df.tail(days * 24)  # Assuming hourly data

    # Run backtest
    backtester = Backtester(
        spread_pips=2.0,
        slippage_pips=1.0,
        min_confidence=0.55,
        max_trades_per_day=3,
        trade_timeout_hours=48
    )

    result = backtester.run_backtest(df, symbol, lookback_periods=100, verbose=verbose)

    return result


if __name__ == '__main__':
    # Run quick test
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else 'EURUSD'
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90

    try:
        result = quick_backtest(symbol, days, verbose=True)

        # Save results
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        results_dir = os.path.join(project_root, 'data', 'backtest_results')

        save_backtest_results(
            result,
            os.path.join(results_dir, f'{symbol}_backtest_{datetime.now().strftime("%Y%m%d")}.json')
        )
        print(f"\nResults saved to {results_dir}/")

    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
