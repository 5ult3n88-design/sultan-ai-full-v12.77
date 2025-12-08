import requests
import feedparser
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

load_dotenv()

def fetch_rss_news(query="finance", max_items=20):
    """Fetch news from RSS feeds (free, no API key required)"""
    news_items = []
    
    # Financial news RSS feeds
    rss_feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={query}&region=US&lang=en-US",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/marketsNews",
        "https://rss.cnn.com/rss/money_latest.rss",
        "https://feeds.bloomberg.com/markets/news.rss"
    ]
    
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_items//len(rss_feeds) + 1]:
                news_items.append({
                    'title': entry.get('title', 'No title'),
                    'summary': entry.get('summary', entry.get('description', 'No summary')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': feed.feed.get('title', 'Unknown')
                })
        except Exception as e:
            print(f"⚠️ Error fetching RSS feed {feed_url}: {e}")
            continue
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_news = []
    for item in news_items:
        if item['title'] not in seen_titles:
            seen_titles.add(item['title'])
            unique_news.append(item)
    
    return unique_news[:max_items]

def fetch_yahoo_news(query="forex OR stock OR trading", max_items=10):
    """Fetch news using Yahoo Finance API (free)"""
    news_items = []
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for item in data.get('news', [])[:max_items]:
                news_items.append({
                    'title': item.get('title', 'No title'),
                    'summary': item.get('summary', 'No summary'),
                    'link': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat() if item.get('providerPublishTime') else '',
                    'source': item.get('publisher', 'Yahoo Finance')
                })
    except Exception as e:
        print(f"⚠️ Error fetching Yahoo news: {e}")
    
    return news_items

def get_news_for_symbol(symbol, max_items=15):
    """Get relevant news for a trading symbol"""
    # Map symbols to search queries - comprehensive forex pairs
    query_map = {
        # Major forex pairs
        'XAUUSD': 'gold price OR XAUUSD OR gold trading OR gold futures',
        'XAGUSD': 'silver price OR XAGUSD OR silver trading OR silver futures',
        'EURUSD': 'EURUSD OR euro dollar OR EUR/USD OR euro usd',
        'GBPUSD': 'GBPUSD OR british pound dollar OR GBP/USD OR pound usd',
        'USDJPY': 'USDJPY OR dollar yen OR USD/JPY OR usd jpy',
        'AUDUSD': 'AUDUSD OR australian dollar OR AUD/USD OR aussie dollar',
        'USDCAD': 'USDCAD OR dollar canadian OR USD/CAD OR loonie',
        'USDCHF': 'USDCHF OR dollar swiss franc OR USD/CHF OR swissie',
        'NZDUSD': 'NZDUSD OR new zealand dollar OR NZD/USD OR kiwi dollar',
        # Cross pairs
        'EURGBP': 'EURGBP OR euro pound OR EUR/GBP OR euro gbp',
        'EURJPY': 'EURJPY OR euro yen OR EUR/JPY OR euro jpy',
        'GBPJPY': 'GBPJPY OR pound yen OR GBP/JPY OR gbp jpy',
        'AUDJPY': 'AUDJPY OR aussie yen OR AUD/JPY OR aud jpy',
        'EURCHF': 'EURCHF OR euro swiss OR EUR/CHF OR euro chf',
        'GBPCHF': 'GBPCHF OR pound swiss OR GBP/CHF OR gbp chf',
        'AUDNZD': 'AUDNZD OR aussie kiwi OR AUD/NZD OR aud nzd',
        'EURAUD': 'EURAUD OR euro aussie OR EUR/AUD OR euro aud',
        'EURCAD': 'EURCAD OR euro loonie OR EUR/CAD OR euro cad',
        'GBPAUD': 'GBPAUD OR pound aussie OR GBP/AUD OR gbp aud',
        # Stocks & Crypto
        'BTC-USD': 'bitcoin BTC cryptocurrency',
        'ETH-USD': 'ethereum ETH cryptocurrency',
        'AAPL': 'Apple AAPL stock',
        'TSLA': 'Tesla TSLA stock',
        'GOOGL': 'Google GOOGL stock',
        'SPY': 'S&P 500 stock market',
        'QQQ': 'NASDAQ QQQ stock market'
    }
    
    query = query_map.get(symbol, f"{symbol} forex OR {symbol} trading OR {symbol} currency")
    
    # Fetch from multiple sources
    news_rss = fetch_rss_news(query, max_items//2)
    news_yahoo = fetch_yahoo_news(query, max_items//2)
    
    # Combine and deduplicate
    all_news = news_rss + news_yahoo
    seen = set()
    unique_news = []
    for item in all_news:
        title_key = item['title'].lower()
        if title_key not in seen:
            seen.add(title_key)
            unique_news.append(item)
    
    return unique_news[:max_items]

def save_news_cache(news_data, symbol):
    """Cache news data to file"""
    # Cross-platform path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "news")
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, f"{symbol}_news.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({
            'symbol': symbol,
            'fetched_at': datetime.now().isoformat(),
            'news': news_data
        }, f, indent=2, ensure_ascii=False)

def load_news_cache(symbol, max_age_hours=1):
    """Load cached news if fresh"""
    # Cross-platform path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "news")
    cache_file = os.path.join(data_dir, f"{symbol}_news.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            fetched_at = datetime.fromisoformat(cache['fetched_at'])
            if (datetime.now() - fetched_at).total_seconds() < max_age_hours * 3600:
                return cache['news']
        except Exception as e:
            print(f"⚠️ Error loading news cache: {e}")
    return None

if __name__ == "__main__":
    # Test news fetching
    test_symbols = ["XAUUSD", "AAPL", "BTC-USD"]
    for symbol in test_symbols:
        print(f"\n📰 Fetching news for {symbol}...")
        news = get_news_for_symbol(symbol, max_items=10)
        print(f"✅ Found {len(news)} news items")
        save_news_cache(news, symbol)
        for i, item in enumerate(news[:3], 1):
            print(f"  {i}. {item['title'][:80]}...")

