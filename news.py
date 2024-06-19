# news.py

import requests

def get_latest_news(company_name):
    url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey=f565e63bcf4846868f92b716e0968ad6"
    response = requests.get(url)
    news_data = response.json()
    return news_data
