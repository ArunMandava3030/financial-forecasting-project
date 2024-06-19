from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
from data_fetch import get_stock_data, compute_technical_indicators, get_latest_news, get_news_sentiment
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model_path = 'models/random_forest_model.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except EOFError:
        raise EOFError(f"Failed to load model from {model_path}. The file might be corrupted.")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

def generate_rating(model, data):
    features = ['SMA_20', 'EMA_20', 'RSI', 'Volume']
    latest_data = data[features].iloc[-1]
    prediction = model.predict([latest_data])[0]
    if prediction > latest_data['Close']:
        return "Buy"
    else:
        return "Hold"

@app.route('/stock', methods=['GET'])
def get_stock_info():
    tickers = request.args.get('tickers').split(',')
    investment_horizon = request.args.get('horizon')  # 'shortterm' or 'longterm'
    
    response = {}
    for ticker in tickers:
        try:
            data = get_stock_data(ticker)
            if data.empty:
                response[ticker] = {'error': 'No data available for this ticker.'}
                continue

            data = compute_technical_indicators(data)
            if data.dropna().empty:
                response[ticker] = {'error': 'Not enough data to compute technical indicators.'}
                continue

            rating = generate_rating(model, data)
            news_articles = get_latest_news(ticker)
            sentiment = get_news_sentiment(news_articles)
            
            # Generate and save the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            data['Close'].plot(ax=ax, label='Close Price')
            data['SMA_20'].plot(ax=ax, label='SMA 20')
            data['EMA_20'].plot(ax=ax, label='EMA 20')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'{ticker} Stock Performance')
            ax.legend()
            plot_data = BytesIO()
            plt.savefig(plot_data, format='png')
            plot_data.seek(0)
            plot_base64 = base64.b64encode(plot_data.read()).decode('utf-8')

            response[ticker] = {
                'current_price': data['Close'].iloc[-1],
                'rating': rating,
                'news_sentiment': sentiment,
                'latest_news': news_articles,
                'plot': plot_base64
            }
        except Exception as e:
            response[ticker] = {'error': str(e)}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
