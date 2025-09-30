import os
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import time
import json

def collect_news(news_api_key, days_back=365):

    print(f"API Key loaded: {news_api_key[:10]}..." if news_api_key else "No API key found!")
    
    if not news_api_key:
        print(" No NewsAPI key found in .env file!")
        return
    os.makedirs("data/raw", exist_ok=True)

    tesla_terms = [
        "Tesla",
        "TSLA", 
        "Elon Musk Tesla",
        "Tesla earnings",
        "Tesla recall",
        "Tesla delivery",
        "Tesla Model",
        "Tesla stock"
    ]

    #try multiple time periods (last 3 years didnt work)

    date_ranges = [
        30,   # Last 30 days (guaranteed to work)
        90,   # Try 90 days  
        180,  # Try 6 months
        365   # Try 1 year
    ]
    all_articles = []

    for days in date_ranges:

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days) 
        range_failed = False

        print(f"Collecting news for: {days} days")
       
        for search_term in tesla_terms:
            print(f"Searching for: {search_term}")

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{search_term}"',  # Exact phrase matching (more relevant results),
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': news_api_key,
                'pageSize': 100
            }

            try:
                response = requests.get(url, params=params)
                response.raise_for_status() #automatically throw error if bad status
                data = response.json()
                articles = data.get("articles", [])

                if articles:
                    all_articles.extend(articles)
                    print(f"Theres {len(articles)} articles")      
                else:
                    print(f"Theres no articles found")

                time.sleep(1)
            except requests.exceptions.RequestException as e:
                print(f" Error fetching: {search_term}, {e}")
                if "426" in str(e):
                    print(f"    Date range {days} days not allowed")
                    range_failed = True
                    break 
                continue

        # Check if we need to stop trying longer date ranges
        if range_failed:
            print(f" Stopping at {days} days due to API limitation")
            break  # Break outer loop (date ranges)
        else:
            print(f" Successfully collected from {days} days range")

        


    seen_urls = set()
    unique_articles = []

    for article in all_articles:
        url = article.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_articles.append(article)

    filename = "data/raw/tesla_news_comprehensive.jsonl"
    with open(filename, 'w', encoding='utf-8') as f:
        for article in unique_articles:
            json.dump(article, f, ensure_ascii=False) 
            f.write("\n")
    
    print(f" Saved {len(unique_articles)} unique articles to {filename}")
    print()
    return unique_articles



if __name__ == "__main__":
    env_location = r"c:\Users\offor\Downloads\tesla-ml-trading-strategy\.env"
    load_dotenv(env_location)
    news_api_key = os.getenv("NEWS_API_KEY")

    if news_api_key:
        articles = collect_news(news_api_key)

        # Quick analysis of what we collected
        if articles:
            dates = [article.get('publishedAt', '')[:10] for article in articles]
            print(dates)
            print(f"\n Date range: {min(dates)} to {max(dates)}")
            print(f" Articles per day: {len(articles) / len(set(dates)):.1f}")
    else:
        print("No keys found")
