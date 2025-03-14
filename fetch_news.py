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
from dotenv import load_dotenv  # Add this import
import os

# Load environment variables from .env
load_dotenv()

# API Keys from environment variables
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

# Initialize APIs and AI model with checks
if not NEWSAPI_KEY:
    raise ValueError("NEWSAPI_KEY not set in environment variables")
if not GNEWS_KEY:
    raise ValueError("GNEWS_KEY not set in environment variables")
if not X_BEARER_TOKEN:
    raise ValueError("X_BEARER_TOKEN not set in environment variables")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
google_news = GNews(max_results=20, api_key=GNEWS_KEY)  # Assuming GNews supports an API key
client = tweepy.Client(bearer_token=X_BEARER_TOKEN)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Database connection with relative path
def get_db_connection():
    conn = sqlite3.connect('news.db', timeout=10)  # Updated to relative path
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

            scrape_site('https://techcrunch.com/tag/artificial-intelligence/', 'artificial intelligence')
            scrape_site('https://techcrunch.com/tag/blockchain/', 'blockchain')
            scrape_site('https://www.coindesk.com/', 'cryptocurrency')
            scrape_site('https://techcrunch.com/tag/cybersecurity/', 'cybersecurity')

            # Fetch from RSS feeds
            rss_feeds = ["https://techcrunch.com/feed/", "https://www.coindesk.com/feed/"]
            try:
                with open("rss_feeds.txt", "r") as f:  # Updated to relative path
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