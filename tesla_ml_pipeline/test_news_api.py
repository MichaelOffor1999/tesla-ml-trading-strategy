# Create a new file: test_api.py
import os
from dotenv import load_dotenv
import requests

env_location = r"c:\Users\offor\Downloads\tesla-ml-trading-strategy\.env"
load_dotenv(env_location)
news_api_key = os.getenv("NEWS_API_KEY")

print(f"API Key: {news_api_key}")

# Simple test request
url = "https://newsapi.org/v2/everything"
params = {
    'q': 'tesla',
    'apiKey': news_api_key,
    'pageSize': 5
}

response = requests.get(url, params=params)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:300]}")