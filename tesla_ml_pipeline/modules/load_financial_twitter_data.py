# Create: hybrid_sentiment_analysis.py
from datasets import load_dataset
import pandas as pd
import json
from transformers import pipeline
import os

def load_financial_twitter_data():
    """Load the financial Twitter sentiment dataset"""
    print("Loading financial Twitter sentiment dataset...")
    
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
    
    # Get both training and validation data
    train_df = dataset['train'].to_pandas()
    val_df = dataset['validation'].to_pandas()
    
    # Combine for larger dataset
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Map labels to readable format
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    combined_df['sentiment'] = combined_df['label'].map(label_map)
    
    print(f" Loaded {len(combined_df)} financial examples")
    print(f" Sentiment distribution:")
    print(combined_df['sentiment'].value_counts())
    
    return combined_df

def load_tesla_news():
    """Load your Tesla NewsAPI data"""
    print(" Loading Tesla NewsAPI data...")
    
    try:
        articles = []
        with open("data/raw/tesla_news_comprehensive.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                article = json.loads(line.strip())
                articles.append({
                    'text': f"{article['title']}. {article.get('description', '')}",
                    'date': article['publishedAt'],
                    'source': 'tesla_newsapi',
                    'url': article.get('url', ''),
                    'title': article['title']
                })
        
        tesla_df = pd.DataFrame(articles)
        print(f" Loaded {len(tesla_df)} Tesla articles")
        return tesla_df
        
    except FileNotFoundError:
        print(" Tesla data not found - run NewsAPI collector first")
        return None

def analyze_with_pretrained_labels():
    """Use the labeled financial data to analyze Tesla articles"""
    print(" HYBRID SENTIMENT ANALYSIS PIPELINE")
    
    # Step 1: Load labeled financial data (for training/validation)
    financial_df = load_financial_twitter_data()
    
    # Step 2: Load Tesla data (for prediction)
    tesla_df = load_tesla_news()
    if tesla_df is None:
        return
    
    # Step 3: Setup FinBERT model
    print("\n Loading FinBERT model...")
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
        print(" FinBERT loaded successfully!")
    except Exception as e:
        print(f" FinBERT failed: {e}")
        print(" Using general sentiment model...")
        sentiment_analyzer = pipeline("sentiment-analysis")
    
    # Step 4: Analyze Tesla articles
    print(f"\n Analyzing {len(tesla_df)} Tesla articles...")
    results = []
    
    for i, row in tesla_df.iterrows():
        try:
            text = row['text'][:512]  # Truncate for model
            
            if len(text.strip()) > 10:
                sentiment = sentiment_analyzer(text)
                
                result = {
                    'date': row['date'],
                    'title': row['title'],
                    'text': text,
                    'sentiment_label': sentiment[0]['label'],
                    'sentiment_score': sentiment[0]['score'],
                    'source': row['source']
                }
                results.append(result)
                
                if i % 50 == 0:
                    print(f"   Processed {i+1}/{len(tesla_df)} articles...")
        
        except Exception as e:
            print(f"   Error on article {i}: {e}")
    
    # Step 5: Save results
    print(f"\n Saving analysis results...")
    os.makedirs("data/processed", exist_ok=True)
    
    # Save Tesla sentiment results
    tesla_results_df = pd.DataFrame(results)
    tesla_results_df.to_csv("data/processed/tesla_sentiment_analysis.csv", index=False)
    
    # Save labeled financial data for reference
    financial_df.to_csv("data/processed/financial_twitter_labeled.csv", index=False)
    
    # Analysis summary
    print(f"\n ANALYSIS COMPLETE!")
    print(f" Tesla articles analyzed: {len(results)}")
    print(f" Labeled financial examples: {len(financial_df)}")
    
    if len(results) > 0:
        print(f"\n TESLA SENTIMENT DISTRIBUTION:")
        tesla_sentiment = tesla_results_df['sentiment_label'].value_counts()
        for sentiment, count in tesla_sentiment.items():
            percentage = (count / len(results)) * 100
            print(f"  {sentiment}: {count} articles ({percentage:.1f}%)")
    
    print(f"\n FILES SAVED:")
    print(f"   data/processed/tesla_sentiment_analysis.csv")
    print(f"   data/processed/financial_twitter_labeled.csv")
    
    return tesla_results_df, financial_df

def quick_analysis_preview():
    """Show a quick preview of both datasets"""
    print(" QUICK DATASET PREVIEW")
    
    # Financial Twitter examples
    financial_df = load_financial_twitter_data()
    print(f"\n FINANCIAL TWITTER EXAMPLES:")
    for i, row in financial_df.head(3).iterrows():
        print(f"  {row['sentiment'].upper()}: {row['text'][:80]}...")
    
    # Tesla examples
    tesla_df = load_tesla_news()
    if tesla_df is not None:
        print(f"\n TESLA NEWS EXAMPLES:")
        for i, row in tesla_df.head(3).iterrows():
            print(f"  {row['date']}: {row['title'][:60]}...")

if __name__ == "__main__":
    # Option 1: Quick preview
    # quick_analysis_preview()
    
    # Option 2: Full analysis
    tesla_results, financial_data = analyze_with_pretrained_labels()
    
    print(f"\n SUCCESS!")
    print(f" You now have both labeled financial data AND Tesla sentiment analysis")
    print(f" Ready for stock price correlation analysis!")