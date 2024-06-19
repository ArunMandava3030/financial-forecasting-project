# streamlit_app.py

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from data_fetch import get_stock_data
from technical_indicators import compute_technical_indicators
from ratings import generate_rating, load_model, predict_next_session
from news import get_latest_news

st.title('Indian Stock Market Analysis')

# Sidebar for user input
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., RELIANCE, TCS):')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

def fetch_data_and_compute_indicators(ticker, start_date, end_date):
    data = get_stock_data(ticker, start_date, end_date)
    if not data.empty:
        data = compute_technical_indicators(data)
        return data
    else:
        return None

def display_fundamentals(ticker):
    stock = yf.Ticker(f"{ticker}.NS")
    info = stock.info
    fundamentals = {
        'Market Cap': info.get('marketCap', 'N/A'),
        'P/E Ratio': info.get('trailingPE', 'N/A'),
        'EPS': info.get('trailingEps', 'N/A'),
        'Dividend Yield': info.get('dividendYield', 'N/A'),
        '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
        '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A')
    }
    return fundamentals

if ticker and start_date and end_date:
    # Fetch stock data
    data = fetch_data_and_compute_indicators(ticker, start_date, end_date)

    if data is None or data.empty:
        st.warning(f'No data available for {ticker}. Please enter a valid stock ticker.')
    else:
        # Ensure the index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Display stock data
        st.subheader(f'Stock Data for {ticker}')
        st.write(data.tail())

        # Generate rating
        model_path = 'models/random_forest_model.pkl'  # Path to your trained model
        model = load_model(model_path)
        rating = generate_rating(model, data)
        st.subheader('Rating')
        st.write(f'Based on historical data, the stock is rated as: {rating}')

        # Predict next session's OHLC prices
        next_session_prediction = predict_next_session(model, data)
        st.subheader('Prediction for Next Trading Session')
        st.write(f"Open: {next_session_prediction['Open']}")
        st.write(f"High: {next_session_prediction['High']}")
        st.write(f"Low: {next_session_prediction['Low']}")
        st.write(f"Close: {next_session_prediction['Close']}")

        # Display news
        st.subheader('Latest News')
        news_data = get_latest_news(ticker)
        if 'articles' in news_data:
            for article in news_data['articles']:
                st.write(f"- {article['title']} ({article['url']})")
        else:
            st.write("No news available.")

        # Display fundamentals
        st.subheader('Fundamentals')
        fundamentals = display_fundamentals(ticker)
        for key, value in fundamentals.items():
            st.write(f"{key}: {value}")

        # Stock price graph with different intervals
        st.subheader('Stock Price Visualization')

        fig = go.Figure()

        # 1 day
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='1 Day Close Price'))

        # 1 week
        weekly_data = data.resample('W').mean()  # Weekly resampling
        fig.add_trace(go.Scatter(x=weekly_data.index, y=weekly_data['Close'], mode='lines', name='Weekly Close Price'))

        # 1 month
        monthly_data = data.resample('M').mean()  # Monthly resampling
        fig.add_trace(go.Scatter(x=monthly_data.index, y=monthly_data['Close'], mode='lines', name='Monthly Close Price'))

        # 1 year
        yearly_data = data.resample('Y').mean()  # Yearly resampling
        fig.add_trace(go.Scatter(x=yearly_data.index, y=yearly_data['Close'], mode='lines', name='Yearly Close Price'))

        # All time
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='All Time Close Price'))

        # Update layout
        fig.update_layout(title=f'{ticker} Stock Price',
                          xaxis_title='Date',
                          yaxis_title='Price (INR)',  # Assuming prices are in Indian Rupees
                          )

        # Display the graph
        st.plotly_chart(fig)

# Compare multiple stocks
st.sidebar.subheader('Compare Multiple Stocks')
compare_tickers = st.sidebar.text_input('Enter tickers separated by commas (e.g., RELIANCE, TCS):')
if compare_tickers:
    tickers_list = [ticker.strip() for ticker in compare_tickers.split(',')]
    for ticker in tickers_list:
        data = fetch_data_and_compute_indicators(ticker, start_date, end_date)
        if data is not None and not data.empty:
            st.subheader(f'Stock Data for {ticker}')
            st.write(data.tail())
        else:
            st.warning(f'No data available for {ticker}. Please enter a valid stock ticker.')
