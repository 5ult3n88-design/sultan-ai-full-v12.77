"""
Autonomous Trading Robot Dashboard
Monitor and control the AI trading robot with real-time confidence metrics
"""

import streamlit as st
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading_robot import get_robot_instance, TradingRobot
import time

# Page config
st.set_page_config(
    page_title="AI Trading Robot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .big-metric {
        font-size: 28px !important;
        font-weight: bold !important;
    }
    .confidence-high {
        color: #00ff00;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff4444;
        font-weight: bold;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Autonomous AI Trading Robot")
st.markdown("---")

# Initialize robot
if 'robot' not in st.session_state:
    st.session_state.robot = get_robot_instance()

robot = st.session_state.robot

# Sidebar - Robot Controls
with st.sidebar:
    st.header("🎛️ Robot Controls")

    # Robot configuration
    st.subheader("Configuration")

    symbols_input = st.text_area(
        "Trading Symbols (one per line)",
        value="\n".join(robot.symbols),
        height=150
    )

    min_confidence = st.slider(
        "Minimum Confidence (%)",
        min_value=50,
        max_value=95,
        value=int(robot.min_confidence * 100),
        step=5
    )

    risk_per_trade = st.slider(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=robot.risk_per_trade * 100,
        step=0.5
    )

    check_interval = st.select_slider(
        "Check Interval",
        options=[60, 120, 300, 600, 900, 1800],
        value=robot.check_interval,
        format_func=lambda x: f"{x//60} min" if x >= 60 else f"{x}s"
    )

    if st.button("💾 Update Configuration", use_container_width=True):
        robot.symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
        robot.min_confidence = min_confidence / 100
        robot.risk_per_trade = risk_per_trade / 100
        robot.check_interval = check_interval
        robot.save_state()
        st.success("Configuration updated!")

    st.markdown("---")

    # Robot status
    status = robot.get_status()

    if robot.is_running:
        st.success("✅ Robot is RUNNING")
        if st.button("⏸️ Stop Robot", use_container_width=True, type="primary"):
            robot.stop()
            st.rerun()
    else:
        st.info("⏸️ Robot is STOPPED")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True, type="primary"):
                # Start robot in background thread
                import threading
                robot.is_running = True
                robot.start_time = datetime.now()
                st.success("Robot started!")
                st.rerun()
        with col2:
            if st.button("🔄 Run Once", use_container_width=True):
                robot.run_single_cycle()
                st.success("Cycle completed!")
                st.rerun()

    st.markdown("---")

    # Reset
    if st.button("🔄 Reset Robot", use_container_width=True):
        if st.checkbox("Confirm reset (will lose all data)"):
            st.session_state.robot = TradingRobot()
            robot = st.session_state.robot
            st.success("Robot reset!")
            st.rerun()

# Main content
col1, col2, col3, col4 = st.columns(4)

account_info = status['account_info']

with col1:
    st.metric(
        "💰 Balance",
        f"${account_info['balance']:.2f}",
        delta=None
    )

with col2:
    equity_delta = account_info['equity'] - robot.executor.initial_balance
    st.metric(
        "📊 Total Equity",
        f"${account_info['equity']:.2f}",
        delta=f"${equity_delta:.2f}"
    )

with col3:
    st.metric(
        "📈 ROI",
        f"{account_info['roi']:.2f}%",
        delta=None
    )

with col4:
    st.metric(
        "🎯 Win Rate",
        f"{account_info['win_rate']:.1f}%",
        delta=None
    )

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "💼 Open Positions",
    "📈 Performance",
    "🔔 Recent Signals",
    "📜 Trade History"
])

# TAB 1: Dashboard
with tab1:
    st.header("Robot Status & Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⏱️ Runtime")
        runtime_hours = status['runtime_hours']
        st.write(f"**{runtime_hours:.1f} hours**")

        st.subheader("📊 Activity")
        st.write(f"Total Signals: **{status['total_signals']}**")
        st.write(f"Total Trades: **{status['total_trades']}**")
        st.write(f"Open Positions: **{status['open_positions']}**")

    with col2:
        st.subheader("💵 Account")
        st.write(f"Initial: **${robot.executor.initial_balance:.2f}**")
        st.write(f"Balance: **${account_info['balance']:.2f}**")
        st.write(f"Equity: **${account_info['equity']:.2f}**")
        st.write(f"Open P&L: **${account_info['open_pl']:.2f}**")

    with col3:
        st.subheader("📈 Performance")
        trade_stats = status['trade_stats']
        st.write(f"Win Rate: **{trade_stats['win_rate']:.1f}%**")
        st.write(f"Profit Factor: **{trade_stats['profit_factor']:.2f}**")
        st.write(f"Avg Win: **${trade_stats['avg_win']:.2f}**")
        st.write(f"Avg Loss: **${trade_stats['avg_loss']:.2f}**")

    st.markdown("---")

    # Latest signals for each symbol
    st.subheader("🎯 Latest Signals by Symbol")

    if status['last_signals']:
        signals_df = pd.DataFrame([
            {
                'Symbol': sig['symbol'],
                'Action': sig['action'],
                'Confidence': f"{sig['confidence']:.1%}",
                'ML Direction': sig['ml_direction'],
                'ML Conf': f"{sig['ml_confidence']:.1%}",
                'Tech Score': f"{sig['technical_strength']}/100",
                'Risk': sig['risk_level'],
                'Price': f"${sig['current_price']:.5f}",
                'Time': sig['timestamp'].strftime('%H:%M:%S')
            }
            for sig in status['last_signals'].values()
        ])

        # Color code by action
        def color_action(val):
            if val == 'BUY':
                return 'background-color: #90EE90'
            elif val == 'SELL':
                return 'background-color: #FFB6C6'
            return ''

        styled_df = signals_df.style.applymap(color_action, subset=['Action'])
        st.dataframe(styled_df, use_container_width=True, height=300)
    else:
        st.info("No signals yet. Run a cycle to generate signals.")

# TAB 2: Open Positions
with tab2:
    st.header("💼 Open Positions")

    positions = robot.executor.get_all_positions()

    if positions:
        positions_data = []
        for symbol, pos in positions.items():
            # Calculate holding time
            holding_time = (datetime.now() - pos['entry_time']).total_seconds() / 3600

            positions_data.append({
                'Symbol': symbol,
                'Action': pos['action'],
                'Entry': f"${pos['entry_price']:.5f}",
                'Current': f"${pos['current_price']:.5f}",
                'Units': f"{pos['units']:.4f}",
                'Value': f"${pos['position_value']:.2f}",
                'P&L': f"${pos['unrealized_pl']:.2f}",
                'P&L %': f"{pos['unrealized_pl_pct']:.2f}%",
                'Stop Loss': f"${pos['stop_loss']:.5f}",
                'Take Profit': f"${pos['take_profit']:.5f}",
                'Confidence': f"{pos['confidence']:.1%}",
                'Holding': f"{holding_time:.1f}h"
            })

        positions_df = pd.DataFrame(positions_data)

        # Color code P&L
        def color_pl(val):
            if 'P&L' in val.name:
                if val.startswith('$-') or val.startswith('-'):
                    return 'color: red; font-weight: bold'
                elif val.startswith('$'):
                    return 'color: green; font-weight: bold'
            return ''

        styled_positions = positions_df.style.applymap(color_pl)
        st.dataframe(styled_positions, use_container_width=True)

        # Close position controls
        st.subheader("Close Position")
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol_to_close = st.selectbox("Select Position to Close", list(positions.keys()))
        with col2:
            if st.button("🔴 Close Position", use_container_width=True):
                # Get current price
                from data_loader import load_symbol_data
                df = load_symbol_data(symbol_to_close)
                if df is not None:
                    current_price = float(df['Close'].iloc[-1])
                    success, message = robot.executor.close_position(
                        symbol_to_close, current_price, reason='MANUAL'
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                    st.rerun()
    else:
        st.info("No open positions")

# TAB 3: Performance
with tab3:
    st.header("📈 Performance Analysis")

    # Equity curve
    if robot.executor.equity_curve:
        st.subheader("Equity Curve")

        equity_df = pd.DataFrame(robot.executor.equity_curve)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['balance'],
            mode='lines',
            name='Balance',
            line=dict(color='green', width=2, dash='dash')
        ))

        fig.update_layout(
            title="Account Equity Over Time",
            xaxis_title="Time",
            yaxis_title="Value ($)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Trade statistics
    st.subheader("Trade Statistics")

    trade_stats = robot.executor.get_trade_statistics()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Trades", trade_stats['total_trades'])
        st.metric("Winning Trades", trade_stats['winning_trades'])
        st.metric("Losing Trades", trade_stats['losing_trades'])

    with col2:
        st.metric("Win Rate", f"{trade_stats['win_rate']:.1f}%")
        st.metric("Profit Factor", f"{trade_stats['profit_factor']:.2f}")
        st.metric("Avg Holding Time", f"{trade_stats['avg_holding_time']:.1f}h")

    with col3:
        st.metric("Average Win", f"${trade_stats['avg_win']:.2f}")
        st.metric("Average Loss", f"${trade_stats['avg_loss']:.2f}")
        st.metric("Largest Win", f"${trade_stats['largest_win']:.2f}")

# TAB 4: Recent Signals
with tab4:
    st.header("🔔 Recent Trading Signals")

    decisions = robot.get_decisions_log(limit=50)

    if decisions:
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            action_filter = st.multiselect(
                "Filter by Action",
                ['BUY', 'SELL', 'HOLD'],
                default=['BUY', 'SELL', 'HOLD']
            )
        with col2:
            min_conf_filter = st.slider(
                "Min Confidence %",
                0, 100, 0, 5
            )
        with col3:
            symbol_filter = st.multiselect(
                "Filter by Symbol",
                list(set([d['symbol'] for d in decisions])),
                default=list(set([d['symbol'] for d in decisions]))
            )

        # Filter decisions
        filtered_decisions = [
            d for d in decisions
            if d['action'] in action_filter
            and d['confidence'] * 100 >= min_conf_filter
            and d['symbol'] in symbol_filter
        ]

        # Display as table
        if filtered_decisions:
            decisions_data = []
            for dec in reversed(filtered_decisions[-20:]):  # Last 20

                # Confidence color
                conf_pct = dec['confidence'] * 100
                if conf_pct >= 75:
                    conf_class = "confidence-high"
                elif conf_pct >= 60:
                    conf_class = "confidence-medium"
                else:
                    conf_class = "confidence-low"

                decisions_data.append({
                    'Time': dec['timestamp'].strftime('%m-%d %H:%M'),
                    'Symbol': dec['symbol'],
                    'Action': dec['action'],
                    'Confidence': f"{dec['confidence']:.1%}",
                    'ML Dir': dec['ml_direction'],
                    'Pred Chg': f"{dec['predicted_change']:.2f}%",
                    'Tech': f"{dec['technical_strength']}/100",
                    'Risk': dec['risk_level'],
                    'Price': f"${dec['current_price']:.5f}",
                    'Executed': '✅' if dec.get('trade_executed', False) else '❌',
                    'Reason': dec.get('trade_reason', 'N/A')[:30]
                })

            st.dataframe(pd.DataFrame(decisions_data), use_container_width=True, height=500)
        else:
            st.info("No signals match the filters")
    else:
        st.info("No signals yet. Run the robot to generate trading signals.")

# TAB 5: Trade History
with tab5:
    st.header("📜 Complete Trade History")

    if robot.executor.trade_history:
        # Separate open and close trades
        open_trades = [t for t in robot.executor.trade_history if t['type'] == 'OPEN']
        close_trades = [t for t in robot.executor.trade_history if t['type'] == 'CLOSE']

        st.subheader(f"Closed Trades ({len(close_trades)})")

        if close_trades:
            closed_data = []
            for trade in reversed(close_trades):
                closed_data.append({
                    'Time': trade['time'].strftime('%m-%d %H:%M'),
                    'Symbol': trade['symbol'],
                    'Action': trade['action'],
                    'Price': f"${trade['price']:.5f}",
                    'Units': f"{trade['units']:.4f}",
                    'P&L': f"${trade['pl']:.2f}",
                    'P&L %': f"{trade['pl_pct']:.2f}%",
                    'Reason': trade['reason'],
                    'Holding': f"{trade['holding_time']:.1f}h"
                })

            closed_df = pd.DataFrame(closed_data)

            # Color P&L
            def highlight_pl(row):
                if row['P&L'].startswith('$-'):
                    return [''] * len(row) + ['background-color: #ffcccc'] * 2 + ['']
                else:
                    return [''] * len(row) + ['background-color: #ccffcc'] * 2 + ['']

            st.dataframe(closed_df, use_container_width=True, height=400)

            # Download trades
            csv = closed_df.to_csv(index=False)
            st.download_button(
                "📥 Download Trade History (CSV)",
                csv,
                "trade_history.csv",
                "text/csv"
            )

        st.markdown("---")
        st.subheader(f"All Transactions ({len(robot.executor.trade_history)})")

        all_trades_data = []
        for trade in reversed(robot.executor.trade_history[-50:]):
            all_trades_data.append({
                'Time': trade['time'].strftime('%m-%d %H:%M:%S'),
                'Type': trade['type'],
                'Symbol': trade['symbol'],
                'Action': trade['action'],
                'Price': f"${trade['price']:.5f}",
                'Units': f"{trade['units']:.4f}",
                'Value': f"${trade['value']:.2f}",
                'P&L': f"${trade.get('pl', 0):.2f}" if 'pl' in trade else 'N/A'
            })

        st.dataframe(pd.DataFrame(all_trades_data), use_container_width=True, height=300)
    else:
        st.info("No trades executed yet")

# Auto-refresh
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    auto_refresh = st.checkbox("🔄 Auto-refresh dashboard", value=False)

with col2:
    refresh_interval = st.select_slider(
        "Refresh every",
        options=[5, 10, 15, 30, 60],
        value=10,
        format_func=lambda x: f"{x}s"
    )

with col3:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
