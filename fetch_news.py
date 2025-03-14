# fetch_news.py
from newsapi import NewsApiClient
from gnews import GNews
import sqlite3
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import tweepy
import time
import feedparser
from transformers import pipeline

# API Keys (replace with your own)
NEWSAPI_KEY = '23ccac7d02af4646bfc72f18a5dfacdb'
GNEWS_KEY = 'b5d2fdc194a113fabc6bbe75536a5340'
X_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAALRpzwEAAAAA3ipZ7NqyElDPD0ieQ98bxx31BD0%3DCubMIh6jXMcgIG78zPyWEP0al8JXbOM01CZBbg5orlnLJXXFIQ'

# Initialize APIs and AI model
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
google_news = GNews(max_results=20)
client = tweepy.Client(bearer_token=X_BEARER_TOKEN)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Database connection with timeout
def get_db_connection():
    conn = sqlite3.connect('/Users/yashmandaviya/Newsfetcher/NewsFetcher/news.db', timeout=10)
    return conn

# News fetching function
def fetch_news(language='en'):
    topics = ['artificial intelligence', 'blockchain', 'cryptocurrency', 'cybersecurity']
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            print("Creating or verifying news table...")
            cursor.execute('''CREATE TABLE IF NOT EXISTS news 
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                             title TEXT, 
                             url TEXT UNIQUE, 
                             date TEXT, 
                             topic TEXT, 
                             source TEXT, 
                             language TEXT, 
                             rating REAL DEFAULT 0.0)''')
            conn.commit()

            # Fetch from NewsAPI
            for topic in topics:
                try:
                    articles = newsapi.get_everything(q=topic, language=language, sort_by='publishedAt', page_size=20)
                    print(f"NewsAPI fetched {len(articles['articles'])} articles for {topic} in {language}")
                    for article in articles['articles']:
                        title = article['title']
                        url = article['url']
                        date = article['publishedAt']
                        inferred_topic = classifier(title, candidate_labels=topics)['labels'][0]
                        cursor.execute("INSERT OR IGNORE INTO news (title, url, date, topic, source, language) VALUES (?, ?, ?, ?, ?, ?)", 
                                      (title, url, date, inferred_topic, 'NewsAPI', language))
                except Exception as e:
                    print(f"NewsAPI error for {topic}: {e}")

            # Fetch from GNews
            google_news.language = language
            for topic in topics:
                try:
                    articles = google_news.get_news(topic)
                    print(f"GNews fetched {len(articles)} articles for {topic} in {language}")
                    for article in articles:
                        title = article['title']
                        url = article['url']
                        date = article['published date']
                        inferred_topic = classifier(title, candidate_labels=topics)['labels'][0]
                        cursor.execute("INSERT OR IGNORE INTO news (title, url, date, topic, source, language) VALUES (?, ?, ?, ?, ?, ?)", 
                                      (title, url, date, inferred_topic, 'GNews', language))
                except Exception as e:
                    print(f"GNews error for {topic}: {e}")

            # Fetch from X with retry on rate limit
            for topic in topics:
                try:
                    tweets = client.search_recent_tweets(query=f"{topic} lang:{language}", max_results=10, tweet_fields=['created_at'])
                    if tweets.data:
                        print(f"X fetched {len(tweets.data)} tweets for {topic} in {language}")
                        for tweet in tweets.data:
                            title = tweet.text[:100]
                            url = f"https://twitter.com/i/status/{tweet.id}"
                            date = tweet.created_at.strftime('%Y-%m-%dT%H:%M:%SZ')
                            inferred_topic = classifier(title, candidate_labels=topics)['labels'][0]
                            cursor.execute("INSERT OR IGNORE INTO news (title, url, date, topic, source, language) VALUES (?, ?, ?, ?, ?, ?)", 
                                          (title, url, date, inferred_topic, 'X', language))
                    else:
                        print(f"No X tweets found for {topic}")
                    time.sleep(15)
                except tweepy.TooManyRequests:
                    print(f"X rate limit hit for {topic}. Waiting 15 minutes...")
                    time.sleep(900)
                except Exception as e:
                    print(f"X error for {topic}: {e}")

            # Web scraping with fallback
            def scrape_site(url, topic):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 403 or response.status_code == 401:
                        print(f"Skipping {url} - Access denied ({response.status_code})")
                        return
                    soup = BeautifulSoup(response.content, 'html.parser')
                    articles = soup.select('h2 a') if 'techcrunch' in url else soup.find_all('a', href=True)
                    print(f"Scraped {len(articles)} articles from {url} for {topic}")
                    for article in articles[:20]:
                        title = article.get_text().strip()
                        href = article['href']
                        url_link = href if href.startswith('http') else f"https://{url.split('/')[2]}{href}"
                        date = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                        inferred_topic = classifier(title, candidate_labels=topics)['labels'][0]
                        if title and len(title) > 10:
                            cursor.execute("INSERT OR IGNORE INTO news (title, url, date, topic, source, language) VALUES (?, ?, ?, ?, ?, ?)", 
                                          (title, url_link, date, inferred_topic, 'Scraped', language))
                except Exception as e:
                    print(f"Scraping error for {url}: {e}")

            # Adjusted URLs (fixed typo and removed invalid ones)
            scrape_site('https://techcrunch.com/tag/artificial-intelligence/', 'artificial intelligence')
            scrape_site('https://techcrunch.com/tag/blockchain/', 'blockchain')
            scrape_site('https://www.coindesk.com/', 'cryptocurrency')  # Replaced invalid identif.org
            scrape_site('https://techcrunch.com/tag/cybersecurity/', 'cybersecurity')

            # Fetch from RSS feeds
            rss_feeds = ["https://techcrunch.com/feed/", "https://www.coindesk.com/feed/"]
            try:
                with open("/Users/yashmandaviya/Newsfetcher/NewsFetcher/rss_feeds.txt", "r") as f:
                    rss_feeds.extend([line.strip() for line in f if line.strip()])
            except FileNotFoundError:
                print("rss_feeds.txt not found, using default feeds.")
            for rss_url in rss_feeds:
                try:
                    feed = feedparser.parse(rss_url)
                    print(f"RSS fetched {len(feed.entries)} articles from {rss_url}")
                    for entry in feed.entries[:20]:
                        title = entry.title
                        url = entry.link
                        date = entry.published if 'published' in entry else datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
                        inferred_topic = classifier(title, candidate_labels=topics)['labels'][0]
                        cursor.execute("INSERT OR IGNORE INTO news (title, url, date, topic, source, language) VALUES (?, ?, ?, ?, ?, ?)", 
                                      (title, url, date, inferred_topic, 'RSS', language))
                except Exception as e:
                    print(f"RSS error for {rss_url}: {e}")

            conn.commit()
            total_rows = cursor.execute("SELECT COUNT(*) FROM news").fetchone()[0]
            print(f"News fetched at {datetime.now()}. Total rows in database: {total_rows}")
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}. Retrying in 10 seconds...")
        time.sleep(10)
        fetch_news(language)  # Retry once

# Scheduler setup
def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: fetch_news('en'), 'interval', hours=1)
    scheduler.add_job(lambda: fetch_news('es'), 'interval', hours=1, start_date='2025-03-11 01:00:00')
    scheduler.start()

if __name__ == "__main__":
    fetch_news('en')  # Initial run
    start_scheduler()
    try:
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")