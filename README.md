# NewsFetcher: AI Boom

Welcome to **NewsFetcher**, a sleek, AI-powered web application built with Streamlit that aggregates and analyzes news on cutting-edge topics like artificial intelligence, blockchain, cryptocurrency, and cybersecurity. With features like real-time news fetching, sentiment analysis, hot takes, daily quizzes, and more, it’s your one-stop shop for staying ahead in the tech world.

## Features
- **Real-Time News Feed**: Aggregates articles from NewsAPI, GNews, X, RSS feeds, and web scraping.
- **AI-Powered Hot Takes**: Generates bold, witty opinions on news articles using `distilgpt2`.
- **Daily Quiz**: Test your knowledge with a gamified quiz based on recent articles.
- **Sentiment Analysis**: Analyzes article titles with VADER sentiment.
- **Text-to-Speech**: Listen to article summaries with gTTS.
- **Personalized Feed**: Filter by topic, language, and user preferences.
- **Crypto Prices**: Live Bitcoin and Ethereum prices from CoinGecko.
- **Chatbot**: AI assistant powered by `facebook/blenderbot-400M-distill`.
- **Clustering**: Groups similar articles using sentence embeddings and KMeans.
- **Gamification**: Earn points for ratings, comments, and hot takes.
- **Social Sharing**: Tweet or share articles on LinkedIn.

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python 3.12
- **Database**: SQLite (`news.db`)
- **AI Models**: Transformers (`facebook/bart-large-cnn`, `distilgpt2`, etc.), Sentence-Transformers
- **APIs**: NewsAPI, GNews, Tweepy (X), CoinGecko
- **Libraries**: Pandas, NumPy, Plotly, Scikit-learn, Requests, VADER Sentiment, gTTS

## Installation

### Prerequisites
- Python 3.12+
- Git
- A GitHub account (for deployment)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/NewsFetcher.git
   cd NewsFetcher