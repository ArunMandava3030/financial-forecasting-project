import yfinance as yf
import pandas as pd
import requests
from textblob import TextBlob

def get_stock_data(ticker, start_date, end_date):
    try:
        # Append .NS for NSE stocks
        ticker_symbol = f"{ticker}.NS"
        
        # Fetch data using yfinance
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        
        # Ensure the index is a DatetimeIndex
        data.index = pd.to_datetime(data.index)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker_symbol}")
        
        return data
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()
    
def compute_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI'] = compute_RSI(data['Close'])
    return data

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def get_latest_news(ticker):
    api_key = "YOUR_NEWS_API_KEY"  # Replace with your own News API key
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

def get_news_sentiment(articles):
    sentiments = [TextBlob(article['title']).sentiment.polarity for article in articles]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment
