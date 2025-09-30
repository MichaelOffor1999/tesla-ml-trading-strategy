# Create: stock_correlation_analysis.py
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_sentiment_data():
    """Load your Tesla sentiment analysis results"""
    print(" Loading Tesla sentiment analysis...")
    
    df = pd.read_csv("data/processed/tesla_sentiment_analysis.csv")
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    
    print(f" Loaded {len(df)} sentiment-analyzed articles")
    print(f" Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df

def get_tesla_stock_data(start_date, end_date):
    """Get Tesla stock prices for the same period"""
    print(" Fetching Tesla stock data...")
    
    try:
        tesla = yf.Ticker("TSLA")
        stock_data = tesla.history(start=start_date, end=end_date)
        
        # Reset index to make Date a column
        stock_data.reset_index(inplace=True)
        stock_data['date_only'] = stock_data['Date'].dt.date
        
        # Calculate daily returns
        stock_data['daily_return'] = stock_data['Close'].pct_change() * 100
        stock_data['price_change'] = stock_data['Close'].diff()
        
        print(f" Retrieved {len(stock_data)} trading days")
        print(f" Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")
        
        return stock_data
        
    except Exception as e:
        print(f" Error fetching stock data: {e}")
        return None

def aggregate_daily_sentiment(sentiment_df):
    """Aggregate sentiment scores by day"""
    print(" Aggregating daily sentiment...")
    
    # Group by date and calculate daily sentiment metrics
    daily_sentiment = sentiment_df.groupby('date_only').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Flatten column names
    daily_sentiment.columns = ['date_only', 'avg_sentiment_score', 'sentiment_std', 'article_count', 'sentiment_breakdown']
    
    # Calculate sentiment ratios
    def extract_sentiment_ratios(breakdown_dict):
        total = sum(breakdown_dict.values())
        return {
            'positive_ratio': breakdown_dict.get('positive', 0) / total,
            'negative_ratio': breakdown_dict.get('negative', 0) / total,
            'neutral_ratio': breakdown_dict.get('neutral', 0) / total
        }
    
    sentiment_ratios = daily_sentiment['sentiment_breakdown'].apply(extract_sentiment_ratios)
    ratio_df = pd.DataFrame(sentiment_ratios.tolist())
    
    # Combine with daily sentiment
    daily_sentiment = pd.concat([daily_sentiment.drop('sentiment_breakdown', axis=1), ratio_df], axis=1)
    
    print(f" Created daily sentiment for {len(daily_sentiment)} days")
    
    return daily_sentiment

def correlate_sentiment_stock(sentiment_df, stock_df):
    """Correlate daily sentiment with stock performance"""
    print(" Correlating sentiment with stock performance...")
    
    # Merge sentiment and stock data by date
    merged_df = pd.merge(sentiment_df, stock_df, on='date_only', how='inner')
    
    print(f" Merged data: {len(merged_df)} days with both sentiment and stock data")
    
    if len(merged_df) == 0:
        print(" No overlapping dates found!")
        return None
    
    # Calculate correlations
    correlations = {
        'sentiment_vs_daily_return': merged_df['avg_sentiment_score'].corr(merged_df['daily_return']),
        'sentiment_vs_price_change': merged_df['avg_sentiment_score'].corr(merged_df['price_change']),
        'positive_ratio_vs_return': merged_df['positive_ratio'].corr(merged_df['daily_return']),
        'negative_ratio_vs_return': merged_df['negative_ratio'].corr(merged_df['daily_return']),
    }
    
    print(f"\n CORRELATION ANALYSIS:")
    for metric, corr in correlations.items():
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        print(f"  {metric}: {corr:.3f} ({strength} {direction})")
    
    return merged_df, correlations

def create_visualizations(merged_df):
    """Create visualizations of sentiment vs stock performance"""
    print(" Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tesla Sentiment vs Stock Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Sentiment Score vs Daily Return
    axes[0,0].scatter(merged_df['avg_sentiment_score'], merged_df['daily_return'], alpha=0.6, color='blue')
    axes[0,0].set_xlabel('Average Daily Sentiment Score')
    axes[0,0].set_ylabel('Daily Stock Return (%)')
    axes[0,0].set_title('Sentiment Score vs Daily Return')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(merged_df['avg_sentiment_score'], merged_df['daily_return'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(merged_df['avg_sentiment_score'], p(merged_df['avg_sentiment_score']), "r--", alpha=0.8)
    
    # Plot 2: Positive Ratio vs Daily Return
    axes[0,1].scatter(merged_df['positive_ratio'], merged_df['daily_return'], alpha=0.6, color='green')
    axes[0,1].set_xlabel('Positive News Ratio')
    axes[0,1].set_ylabel('Daily Stock Return (%)')
    axes[0,1].set_title('Positive News Ratio vs Daily Return')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Time series of sentiment and stock price
    ax3 = axes[1,0]
    ax3_twin = ax3.twinx()
    
    ax3.plot(merged_df['date_only'], merged_df['avg_sentiment_score'], color='orange', label='Avg Sentiment')
    ax3_twin.plot(merged_df['date_only'], merged_df['Close'], color='black', label='Stock Price')
    
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sentiment Score', color='orange')
    ax3_twin.set_ylabel('Stock Price ($)', color='black')
    ax3.set_title('Sentiment and Stock Price Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Sentiment distribution
    sentiment_counts = [
        merged_df['positive_ratio'].mean() * 100,
        merged_df['neutral_ratio'].mean() * 100, 
        merged_df['negative_ratio'].mean() * 100
    ]
    
    axes[1,1].pie(sentiment_counts, labels=['Positive', 'Neutral', 'Negative'], 
                  autopct='%1.1f%%', colors=['green', 'gray', 'red'])
    axes[1,1].set_title('Average Daily Sentiment Distribution')
    
    plt.tight_layout()
    plt.savefig('data/processed/tesla_sentiment_stock_analysis.png', dpi=300, bbox_inches='tight')
    print(" Visualization saved: data/processed/tesla_sentiment_stock_analysis.png")
    
    plt.show()

def main():
    """Main correlation analysis pipeline"""
    print(" TESLA SENTIMENT-STOCK CORRELATION ANALYSIS")
    print("="*50)
    
    # Step 1: Load sentiment data
    sentiment_df = load_sentiment_data()
    
    # Step 2: Get stock data for same period
    start_date = sentiment_df['date'].min().date()
    end_date = sentiment_df['date'].max().date() + timedelta(days=1)
    
    stock_df = get_tesla_stock_data(start_date, end_date)
    if stock_df is None:
        return
    
    # Step 3: Aggregate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)
    
    # Step 4: Correlate with stock performance
    merged_df, correlations = correlate_sentiment_stock(daily_sentiment, stock_df)
    if merged_df is None:
        return
    
    # Step 5: Create visualizations
    create_visualizations(merged_df)
    
    # Step 6: Save results
    merged_df.to_csv('data/processed/tesla_sentiment_stock_correlation.csv', index=False)
    
    print(f"\n CORRELATION ANALYSIS COMPLETE!")
    print(f" Analyzed {len(merged_df)} days of sentiment + stock data")
    print(f" Results saved: data/processed/tesla_sentiment_stock_correlation.csv")
    print(f" Visualization: data/processed/tesla_sentiment_stock_analysis.png")
    
    # Summary insights
    avg_sentiment = merged_df['avg_sentiment_score'].mean()
    avg_return = merged_df['daily_return'].mean()
    
    print(f"\n KEY INSIGHTS:")
    print(f"  Average daily sentiment score: {avg_sentiment:.3f}")
    print(f"  Average daily stock return: {avg_return:.2f}%")
    print(f"  Days with positive sentiment: {(merged_df['positive_ratio'] > 0.3).sum()}")
    print(f"  Days with negative sentiment: {(merged_df['negative_ratio'] > 0.3).sum()}")
    
    return merged_df, correlations

if __name__ == "__main__":
    results = main()