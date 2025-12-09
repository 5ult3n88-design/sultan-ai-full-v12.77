import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path - works for both local and Streamlit Cloud
current_file = Path(__file__).resolve()
pages_dir = current_file.parent
frontend_dir = pages_dir.parent
project_dir = frontend_dir.parent
backend_dir = project_dir / 'backend'
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_dir))

from backtester import Backtester, BacktestResult, quick_backtest, save_backtest_results
from analytics import calculate_advanced_indicators

# Page config
st.set_page_config(
    page_title="Backtesting - Sultan AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }

    .good { color: #00ff88; }
    .bad { color: #ff3366; }
    .neutral { color: #ffaa00; }

    .result-card {
        background: rgba(0, 245, 255, 0.1);
        border: 2px solid rgba(0, 245, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }

    .trade-row {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📊 Strategy Backtesting
    </h1>
    <p style="color: #a0a0a0; font-size: 1.1rem;">Test your trading strategy on historical data</p>
</div>
""", unsafe_allow_html=True)

# Data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_dir = os.path.join(project_root, "data")

# Get available symbols
csv_files = [f.replace('.csv', '') for f in os.listdir(data_dir) if f.endswith('.csv')]
csv_files.sort()

if not csv_files:
    st.error("No data files found. Please run `python backend/fetch_data.py` first.")
    st.stop()

# Sidebar settings
st.sidebar.markdown("## Backtest Settings")

symbol = st.sidebar.selectbox("Symbol", csv_files, index=0)

col1, col2 = st.sidebar.columns(2)
with col1:
    days = st.number_input("Days", min_value=30, max_value=365, value=90)
with col2:
    lookback = st.number_input("Lookback", min_value=50, max_value=200, value=100)

st.sidebar.markdown("### Trading Parameters")

min_confidence = st.sidebar.slider("Min Confidence", 0.4, 0.8, 0.55, 0.05)
max_trades_day = st.sidebar.slider("Max Trades/Day", 1, 10, 3)
timeout_hours = st.sidebar.slider("Trade Timeout (hrs)", 12, 96, 48)

st.sidebar.markdown("### Costs")
spread = st.sidebar.number_input("Spread (pips)", 0.0, 10.0, 2.0, 0.5)
slippage = st.sidebar.number_input("Slippage (pips)", 0.0, 5.0, 1.0, 0.5)

# Run backtest button
run_backtest = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")

# Main content
if run_backtest:
    with st.spinner(f"Running backtest for {symbol}..."):
        try:
            # Load data
            csv_path = os.path.join(data_dir, f'{symbol}.csv')
            df = pd.read_csv(csv_path)

            # Parse dates
            if 'Datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['Datetime'])
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])

            df = df.set_index('Date')

            # Limit to requested days
            if len(df) > days * 24:
                df = df.tail(days * 24)

            # Initialize backtester
            backtester = Backtester(
                spread_pips=spread,
                slippage_pips=slippage,
                min_confidence=min_confidence,
                max_trades_per_day=max_trades_day,
                trade_timeout_hours=timeout_hours
            )

            # Run backtest
            result = backtester.run_backtest(
                df,
                symbol,
                lookback_periods=lookback,
                verbose=False
            )

            # Store in session state
            st.session_state['backtest_result'] = result
            st.session_state['backtest_df'] = df

            st.success(f"Backtest complete! {result.metrics['total_trades']} trades analyzed.")

        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display results if available
if 'backtest_result' in st.session_state:
    result = st.session_state['backtest_result']
    m = result.metrics

    # Rating
    if m['win_rate'] >= 55 and m['profit_factor'] >= 1.5:
        rating = "EXCELLENT"
        rating_color = "#00ff88"
    elif m['win_rate'] >= 50 and m['profit_factor'] >= 1.2:
        rating = "GOOD"
        rating_color = "#00f5ff"
    elif m['win_rate'] >= 45:
        rating = "FAIR"
        rating_color = "#ffaa00"
    else:
        rating = "NEEDS WORK"
        rating_color = "#ff3366"

    # Overall rating card
    st.markdown(f"""
    <div class="result-card" style="text-align: center;">
        <h2 style="color: {rating_color}; font-size: 2.5rem; margin: 0;">{rating}</h2>
        <p style="color: #a0a0a0;">Strategy Performance Rating</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    st.markdown("### Key Performance Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        wr_color = "good" if m['win_rate'] >= 50 else "bad"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Win Rate</div>
            <div class="{wr_color}" style="font-size: 2rem; font-weight: bold;">{m['win_rate']}%</div>
            <div style="font-size: 0.8rem; color: #666;">{m['winning_trades']}W / {m['losing_trades']}L</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pf_color = "good" if m['profit_factor'] >= 1.5 else "neutral" if m['profit_factor'] >= 1.0 else "bad"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Profit Factor</div>
            <div class="{pf_color}" style="font-size: 2rem; font-weight: bold;">{m['profit_factor']}</div>
            <div style="font-size: 0.8rem; color: #666;">Target: > 1.5</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        pnl_color = "good" if m['total_pnl_pct'] > 0 else "bad"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Total P&L</div>
            <div class="{pnl_color}" style="font-size: 2rem; font-weight: bold;">{m['total_pnl_pct']:+.1f}%</div>
            <div style="font-size: 0.8rem; color: #666;">{m['total_trades']} trades</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        sr_color = "good" if m['sharpe_ratio'] >= 1.0 else "neutral" if m['sharpe_ratio'] >= 0.5 else "bad"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Sharpe Ratio</div>
            <div class="{sr_color}" style="font-size: 2rem; font-weight: bold;">{m['sharpe_ratio']}</div>
            <div style="font-size: 0.8rem; color: #666;">Target: > 1.0</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        dd_color = "good" if m['max_drawdown_pct'] < 10 else "neutral" if m['max_drawdown_pct'] < 20 else "bad"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Max Drawdown</div>
            <div class="{dd_color}" style="font-size: 2rem; font-weight: bold;">{m['max_drawdown_pct']:.1f}%</div>
            <div style="font-size: 0.8rem; color: #666;">Target: < 15%</div>
        </div>
        """, unsafe_allow_html=True)

    # Second row of metrics
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Avg Win</div>
            <div class="good" style="font-size: 1.5rem; font-weight: bold;">+{m['avg_win']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Avg Loss</div>
            <div class="bad" style="font-size: 1.5rem; font-weight: bold;">-{m['avg_loss']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Win/Loss Ratio</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #00f5ff;">{m['win_loss_ratio']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #a0a0a0;">Expectancy</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {'#00ff88' if m['expectancy'] > 0 else '#ff3366'};">{m['expectancy']:.3f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Equity Curve
    st.markdown("### Equity Curve")

    if result.equity_curve:
        fig_equity = go.Figure()

        fig_equity.add_trace(go.Scatter(
            y=result.equity_curve,
            mode='lines',
            name='Equity',
            line=dict(color='#00f5ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 245, 255, 0.1)'
        ))

        # Add starting capital line
        fig_equity.add_hline(y=m['initial_capital'], line_dash="dash",
                           line_color="#ffaa00", annotation_text="Starting Capital")

        fig_equity.update_layout(
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Trade Number",
            yaxis_title="Equity ($)",
            showlegend=False
        )

        st.plotly_chart(fig_equity, use_container_width=True)

    # Confidence Analysis
    st.markdown("### Confidence Accuracy Analysis")

    col1, col2 = st.columns(2)

    with col1:
        hc_color = "good" if m['high_confidence_accuracy'] >= 55 else "neutral" if m['high_confidence_accuracy'] >= 45 else "bad"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1rem; color: #a0a0a0;">High Confidence Trades (>70%)</div>
            <div class="{hc_color}" style="font-size: 2.5rem; font-weight: bold;">{m['high_confidence_accuracy']}%</div>
            <div style="font-size: 0.9rem; color: #666;">{m['high_confidence_trades']} trades</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        lc_color = "good" if m['low_confidence_accuracy'] >= 50 else "neutral" if m['low_confidence_accuracy'] >= 40 else "bad"
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1rem; color: #a0a0a0;">Low Confidence Trades (<=70%)</div>
            <div class="{lc_color}" style="font-size: 2.5rem; font-weight: bold;">{m['low_confidence_accuracy']}%</div>
            <div style="font-size: 0.9rem; color: #666;">{m['low_confidence_trades']} trades</div>
        </div>
        """, unsafe_allow_html=True)

    # Trade Distribution Charts
    st.markdown("### Trade Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Win/Loss Pie
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Winners', 'Losers'],
            values=[m['winning_trades'], m['losing_trades']],
            hole=0.6,
            marker_colors=['#00ff88', '#ff3366']
        )])

        fig_pie.update_layout(
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title="Win/Loss Distribution",
            showlegend=True
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Exit Reason Distribution
        if result.trades:
            exit_reasons = {}
            for t in result.trades:
                reason = t.exit_reason or 'unknown'
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

            fig_exit = go.Figure(data=[go.Bar(
                x=list(exit_reasons.keys()),
                y=list(exit_reasons.values()),
                marker_color=['#00f5ff', '#00ff88', '#ff3366', '#ffaa00'][:len(exit_reasons)]
            )])

            fig_exit.update_layout(
                template="plotly_dark",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title="Exit Reasons",
                xaxis_title="Reason",
                yaxis_title="Count"
            )

            st.plotly_chart(fig_exit, use_container_width=True)

    # P&L Distribution
    st.markdown("### P&L Distribution")

    if result.trades:
        pnls = [t.pnl_pct for t in result.trades]

        fig_dist = go.Figure()

        fig_dist.add_trace(go.Histogram(
            x=pnls,
            nbinsx=30,
            marker_color='#00f5ff',
            opacity=0.7
        ))

        fig_dist.add_vline(x=0, line_dash="dash", line_color="#ffaa00")
        fig_dist.add_vline(x=np.mean(pnls), line_dash="solid", line_color="#00ff88",
                          annotation_text=f"Mean: {np.mean(pnls):.2f}%")

        fig_dist.update_layout(
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="P&L (%)",
            yaxis_title="Frequency"
        )

        st.plotly_chart(fig_dist, use_container_width=True)

    # Trade Log
    st.markdown("### Recent Trades")

    if result.trades:
        # Show last 20 trades
        trades_data = []
        for t in result.trades[-20:]:
            trades_data.append({
                'ID': t.trade_id,
                'Direction': t.direction,
                'Entry': f"{t.entry_price:.5f}",
                'Exit': f"{t.exit_price:.5f}" if t.exit_price else '-',
                'P&L': f"{t.pnl_pct:+.2f}%",
                'Confidence': f"{t.confidence*100:.0f}%",
                'Exit Reason': t.exit_reason or '-',
                'Duration': f"{t.duration_hours:.1f}h"
            })

        trades_df = pd.DataFrame(trades_data)

        # Style the dataframe
        def color_pnl(val):
            if '+' in str(val):
                return 'color: #00ff88'
            elif '-' in str(val):
                return 'color: #ff3366'
            return ''

        st.dataframe(
            trades_df.style.applymap(color_pnl, subset=['P&L']),
            use_container_width=True,
            height=400
        )

    # Download results
    st.markdown("### Export Results")

    col1, col2 = st.columns(2)

    with col1:
        # Save to file
        if st.button("💾 Save Results", use_container_width=True):
            results_dir = os.path.join(project_root, 'data', 'backtest_results')
            os.makedirs(results_dir, exist_ok=True)

            filepath = os.path.join(
                results_dir,
                f'{result.symbol}_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )

            save_backtest_results(result, filepath)
            st.success(f"Saved to {filepath}")

    with col2:
        # Download as CSV
        if result.trades:
            trades_export = []
            for t in result.trades:
                trades_export.append(t.to_dict())

            csv_data = pd.DataFrame(trades_export).to_csv(index=False)

            st.download_button(
                "📥 Download Trades CSV",
                csv_data,
                f"{result.symbol}_trades.csv",
                "text/csv",
                use_container_width=True
            )

else:
    # No results yet - show instructions
    st.markdown("""
    <div class="result-card" style="text-align: center;">
        <h3 style="color: #00f5ff;">Welcome to Strategy Backtesting</h3>
        <p style="color: #a0a0a0;">
            Test your AI trading strategy on historical data to measure its real performance.
        </p>
        <br>
        <h4 style="color: #ffffff;">How to use:</h4>
        <ol style="text-align: left; color: #a0a0a0; max-width: 600px; margin: 0 auto;">
            <li>Select a symbol from the sidebar</li>
            <li>Choose the number of days to backtest</li>
            <li>Adjust trading parameters (confidence, costs)</li>
            <li>Click "Run Backtest" to see results</li>
        </ol>
        <br>
        <h4 style="color: #ffffff;">Key Metrics Explained:</h4>
        <ul style="text-align: left; color: #a0a0a0; max-width: 600px; margin: 0 auto;">
            <li><strong>Win Rate</strong> - % of profitable trades (target: >50%)</li>
            <li><strong>Profit Factor</strong> - Gross profit / Gross loss (target: >1.5)</li>
            <li><strong>Sharpe Ratio</strong> - Risk-adjusted returns (target: >1.0)</li>
            <li><strong>Max Drawdown</strong> - Largest peak-to-trough decline (target: <15%)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Sultan AI Backtesting Engine v1.0 | Results are based on historical data and do not guarantee future performance</p>
</div>
""", unsafe_allow_html=True)
