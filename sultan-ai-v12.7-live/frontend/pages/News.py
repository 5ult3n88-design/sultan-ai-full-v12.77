import streamlit as st
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
from fetch_news import get_news_for_symbol, load_news_cache, save_news_cache
from analytics import analyze_news_sentiment

st.set_page_config(page_title="📰 News & Analysis", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Style sidebar for navigation */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 100%) !important;
        border-right: 2px solid rgba(0, 245, 255, 0.3);
        min-width: 250px;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        padding: 1rem 0.5rem;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li {
        margin: 0.5rem 0;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        color: #00f5ff !important;
        background: rgba(0, 245, 255, 0.1) !important;
        border: 1px solid rgba(0, 245, 255, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        margin: 0.25rem 0 !important;
        display: block !important;
        text-decoration: none !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: rgba(0, 245, 255, 0.2) !important;
        border-color: #00f5ff !important;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0, 245, 255, 0.4) !important;
    }
    
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] [aria-current="page"] {
        background: rgba(0, 245, 255, 0.3) !important;
        border-left: 4px solid #00f5ff !important;
        box-shadow: 0 4px 12px rgba(0, 245, 255, 0.5) !important;
        color: #ffffff !important;
    }
    
    /* Clean background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }
    
    /* Header */
    .dashboard-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .dashboard-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f5ff 0%, #00d4ff 50%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
    }
    
    .dashboard-subtitle {
        color: #a0a0a0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Navigation bar */
    .nav-bar {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem 2rem;
        margin: 1rem auto 2rem auto;
        max-width: 1200px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .news-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .news-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #00f5ff;
        margin-bottom: 10px;
    }
    
    .news-summary {
        color: rgba(255, 255, 255, 0.8);
        margin: 10px 0;
        line-height: 1.6;
    }
    
    .sentiment-positive {
        color: #00ff88;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #ff3366;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #ffaa00;
        font-weight: bold;
    }
    
    /* Scrollable news container */
    .news-scroll-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        max-height: 600px;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    .news-scroll-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .news-scroll-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    .news-scroll-container::-webkit-scrollbar-thumb {
        background: rgba(0, 245, 255, 0.5);
        border-radius: 10px;
    }
    
    .news-scroll-container::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 245, 255, 0.7);
    }
    
    /* Filter buttons */
    .filter-button {
        background: rgba(0, 245, 255, 0.1);
        border: 2px solid rgba(0, 245, 255, 0.3);
        color: #00f5ff;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.25rem;
        display: inline-block;
    }
    
    .filter-button:hover {
        background: rgba(0, 245, 255, 0.2);
        border-color: #00f5ff;
        transform: translateY(-2px);
    }
    
    .filter-button.active {
        background: linear-gradient(135deg, #00f5ff 0%, #0099ff 100%);
        color: white;
        border-color: #00f5ff;
        box-shadow: 0 4px 12px rgba(0, 245, 255, 0.4);
    }
    
    .filter-button.positive {
        border-color: #00ff88;
        color: #00ff88;
    }
    
    .filter-button.positive.active {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: white;
    }
    
    .filter-button.negative {
        border-color: #ff3366;
        color: #ff3366;
    }
    
    .filter-button.negative.active {
        background: linear-gradient(135deg, #ff3366 0%, #cc1a44 100%);
        color: white;
    }
    
    .filter-button.neutral {
        border-color: #ffaa00;
        color: #ffaa00;
    }
    
    .filter-button.neutral.active {
        background: linear-gradient(135deg, #ffaa00 0%, #cc8800 100%);
        color: white;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    p {
        color: rgba(255, 255, 255, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# Navigation bar
st.markdown("""
<div class="nav-bar">
    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; width: 100%;">
        <span style="color: #00f5ff; font-weight: bold; margin-right: 1rem; font-size: 1.1rem;">📑 Pages:</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation buttons using columns
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

with nav_col1:
    if st.button("🏠 Home", use_container_width=True, type="secondary"):
        st.switch_page("Home.py")

with nav_col2:
    if st.button("🎯 Trading Signals", use_container_width=True, type="secondary"):
        st.switch_page("pages/Trading_Signals.py")

with nav_col3:
    if st.button("📈 Advanced Charts", use_container_width=True, type="secondary"):
        st.switch_page("pages/Advanced_Charts.py")

with nav_col4:
    if st.button("📰 News", use_container_width=True, type="secondary"):
        st.switch_page("pages/News.py")

with nav_col5:
    if st.button("📊 Methodology", use_container_width=True, type="secondary"):
        st.switch_page("pages/Methodology.py")

st.markdown("<br>", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">📰 Financial News & Sentiment Analysis</h1>
    <p class="dashboard-subtitle">Real-time market news and sentiment analysis for informed trading decisions</p>
</div>
""", unsafe_allow_html=True)

# Get selected symbol from session state or use default
try:
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = "XAUUSD"
except:
    # Session state not initialized yet (during import)
    pass

# Symbol selection
col1, col2 = st.columns([3, 1])

default_symbol = st.session_state.get('selected_symbol', 'XAUUSD') if hasattr(st, 'session_state') else 'XAUUSD'

with col1:
    symbol = st.text_input("🔍 Search for Symbol (e.g., XAUUSD, AAPL, EURUSD)", value=default_symbol)
    if hasattr(st, 'session_state'):
        st.session_state.selected_symbol = symbol

with col2:
    refresh_news = st.button("🔄 Refresh News", type="primary")

st.markdown("---")

# Fetch news
@st.cache_data(ttl=3600)
def get_cached_news(symbol, force_refresh=False):
    if force_refresh:
        return None
    cached = load_news_cache(symbol, max_age_hours=1)
    return cached

cached_news = get_cached_news(symbol, force_refresh=refresh_news)

if cached_news is None or refresh_news:
    with st.spinner(f"📥 Fetching latest news for {symbol}..."):
        try:
            news_items = get_news_for_symbol(symbol, max_items=20)
            save_news_cache(news_items, symbol)
            cached_news = news_items
        except Exception as e:
            st.error(f"Error fetching news: {e}")
            cached_news = []

if cached_news:
    news_items = cached_news
else:
    news_items = cached_news or []

# Sentiment Analysis
if news_items:
    sentiment = analyze_news_sentiment(news_items)
    
    st.markdown("### 📊 Overall Market Sentiment")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        compound = sentiment.get('compound', 0)
        if compound > 0.1:
            st.markdown(f'<p class="sentiment-positive">📈 Positive: {compound:.2f}</p>', unsafe_allow_html=True)
        elif compound < -0.1:
            st.markdown(f'<p class="sentiment-negative">📉 Negative: {compound:.2f}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="sentiment-neutral">➡️ Neutral: {compound:.2f}</p>', unsafe_allow_html=True)
    
    with col_s2:
        st.metric("Positive News", f"{sentiment.get('positive', 0)*100:.0f}%")
    with col_s3:
        st.metric("Negative News", f"{sentiment.get('negative', 0)*100:.0f}%")
    with col_s4:
        st.metric("Total Articles", sentiment.get('count', 0))
    
    st.markdown("---")
    
    # Initialize filter in session state
    if 'news_filter' not in st.session_state:
        st.session_state.news_filter = 'All'
    
    # Analyze sentiment for each article
    articles_with_sentiment = []
    for article in news_items:
        article_sentiment = analyze_news_sentiment([article])
        art_compound = article_sentiment.get('compound', 0)
        
        # Classify sentiment
        if art_compound > 0.1:
            sentiment_type = 'Positive'
        elif art_compound < -0.1:
            sentiment_type = 'Negative'
        else:
            sentiment_type = 'Neutral'
        
        articles_with_sentiment.append({
            'article': article,
            'sentiment': art_compound,
            'sentiment_type': sentiment_type
        })
    
    # Filter buttons
    st.markdown("### 🔍 Filter News by Sentiment")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        if st.button("📰 All News", use_container_width=True, 
                    type="primary" if st.session_state.news_filter == 'All' else "secondary"):
            st.session_state.news_filter = 'All'
    
    with col_f2:
        positive_count = sum(1 for item in articles_with_sentiment if item['sentiment_type'] == 'Positive')
        if st.button(f"📈 Positive ({positive_count})", use_container_width=True,
                    type="primary" if st.session_state.news_filter == 'Positive' else "secondary"):
            st.session_state.news_filter = 'Positive'
    
    with col_f3:
        negative_count = sum(1 for item in articles_with_sentiment if item['sentiment_type'] == 'Negative')
        if st.button(f"📉 Negative ({negative_count})", use_container_width=True,
                    type="primary" if st.session_state.news_filter == 'Negative' else "secondary"):
            st.session_state.news_filter = 'Negative'
    
    with col_f4:
        neutral_count = sum(1 for item in articles_with_sentiment if item['sentiment_type'] == 'Neutral')
        if st.button(f"➡️ Neutral ({neutral_count})", use_container_width=True,
                    type="primary" if st.session_state.news_filter == 'Neutral' else "secondary"):
            st.session_state.news_filter = 'Neutral'
    
    # Filter articles based on selected sentiment
    filtered_articles = articles_with_sentiment
    if st.session_state.news_filter != 'All':
        filtered_articles = [item for item in articles_with_sentiment 
                           if item['sentiment_type'] == st.session_state.news_filter]
    
    st.markdown("---")
    st.markdown(f"### 📰 Latest News for {symbol} ({len(filtered_articles)} of {len(news_items)} articles)")
    
    # Scrollable news container
    st.markdown("""
    <div class="news-scroll-container">
    """, unsafe_allow_html=True)
    
    # Display filtered news articles
    if filtered_articles:
        for i, item in enumerate(filtered_articles, 1):
            article = item['article']
            art_compound = item['sentiment']
            sentiment_type = item['sentiment_type']
            
            title = article.get('title', 'No title')
            summary = article.get('summary', article.get('description', 'No summary available'))
            link = article.get('link', '')
            source = article.get('source', 'Unknown')
            published = article.get('published', '')
            
            sentiment_emoji = "📈" if sentiment_type == 'Positive' else "📉" if sentiment_type == 'Negative' else "➡️"
            sentiment_color = "#00ff88" if sentiment_type == 'Positive' else "#ff3366" if sentiment_type == 'Negative' else "#ffaa00"
            
            st.markdown(f"""
            <div class="news-card">
                <div class="news-header">
                    {sentiment_emoji} {title}
                    <span style="font-size: 0.9rem; color: {sentiment_color}; margin-left: 10px;">
                        ({sentiment_type}: {art_compound:+.2f})
                    </span>
                </div>
                <p style="color: rgba(255, 255, 255, 0.7);"><strong>Source:</strong> {source} | <strong>Published:</strong> {published}</p>
                <div class="news-summary">{summary[:300]}{'...' if len(summary) > 300 else ''}</div>
                {f'<a href="{link}" target="_blank" style="color: #00f5ff; text-decoration: none;">Read full article →</a>' if link else ''}
            </div>
            """, unsafe_allow_html=True)
            
            if i < len(filtered_articles):
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info(f"📰 No {st.session_state.news_filter.lower()} news articles found. Try selecting a different filter.")
    
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info(f"📰 No news articles found for {symbol}. Try a different symbol or check your internet connection.")

# Auto-refresh option
st.markdown("---")
auto_refresh = st.checkbox("🔄 Auto-refresh news every 5 minutes", value=False)
if auto_refresh:
    st.rerun()




