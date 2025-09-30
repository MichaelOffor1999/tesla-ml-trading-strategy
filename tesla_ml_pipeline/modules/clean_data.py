# Create: data_cleaning.py
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def clean_tesla_sentiment_data():
    """Basic data cleaning for Tesla sentiment analysis"""
    print(" BASIC DATA CLEANING FOR TESLA SENTIMENT")
    print("="*50)
    
    # Load raw Tesla sentiment data
    print(" Loading Tesla sentiment data...")
    try:
        df = pd.read_csv('data/processed/tesla_sentiment_analysis.csv')
        print(f" Loaded {len(df)} Tesla articles")
    except FileNotFoundError:
        print(" Tesla sentiment data not found. Run Phase 2 first!")
        return None
    
    # Display initial data info
    print(f"\n INITIAL DATA OVERVIEW:")
    print(f"  Total articles: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Columns: {list(df.columns)}")
    
    # 1. Remove duplicates
    print(f"\n STEP 1: Removing duplicates...")
    initial_count = len(df)
    
    # Remove exact duplicates based on title and text
    df_dedupe = df.drop_duplicates(subset=['title', 'text'], keep='first')
    duplicates_removed = initial_count - len(df_dedupe)
    
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Remaining articles: {len(df_dedupe)}")
    
    # 2. Filter by content quality
    print(f"\n STEP 2: Filtering by content quality...")
    
    # Remove articles with very short titles or texts
    quality_filter = (
        (df_dedupe['title'].str.len() >= 10) &  # Title at least 10 chars
        (df_dedupe['text'].str.len() >= 20) &  # text at least 20 chars
        (df_dedupe['title'].notna()) &  # Title not null
        (df_dedupe['text'].notna())  # text not null
    )
    
    df_quality = df_dedupe[quality_filter].copy()
    quality_removed = len(df_dedupe) - len(df_quality)
    
    print(f"  Low quality articles removed: {quality_removed}")
    print(f"  Remaining articles: {len(df_quality)}")
    
    # 3. Clean text content
    print(f"\n STEP 3: Basic text cleaning...")
    
    def basic_text_clean(text):
        """Basic text cleaning function"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    # Apply basic cleaning to title and text
    df_quality['title_clean'] = df_quality['title'].apply(basic_text_clean)
    df_quality['text_clean'] = df_quality['text'].apply(basic_text_clean)
    
    print(f"   Text cleaning applied to titles and text")
    
    # 4. Handle sentiment score outliers
    print(f"\n STEP 4: Handling sentiment score outliers...")
    
    # Calculate outlier bounds using IQR method
    Q1 = df_quality['sentiment_score'].quantile(0.25)
    Q3 = df_quality['sentiment_score'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"  Sentiment score range: {df_quality['sentiment_score'].min():.3f} to {df_quality['sentiment_score'].max():.3f}")
    print(f"  IQR bounds: {lower_bound:.3f} to {upper_bound:.3f}")
    
    # Identify outliers
    outliers = (
        (df_quality['sentiment_score'] < lower_bound) | 
        (df_quality['sentiment_score'] > upper_bound)
    )
    outlier_count = outliers.sum()
    
    # For portfolio project, we'll flag but not remove outliers
    df_quality['is_outlier'] = outliers
    
    print(f"  Sentiment outliers identified: {outlier_count}")
    print(f"  Outliers flagged but retained for analysis")
    
    # 5. Standardize date formats
    print(f"\n STEP 5: Standardizing dates...")
    
    # Ensure consistent date format
    df_quality['date'] = pd.to_datetime(df_quality['date'])
    df_quality['date_only'] = df_quality['date'].dt.date
    
    print(f"   Date formats standardized")
    print(f"  Date range: {df_quality['date_only'].min()} to {df_quality['date_only'].max()}")
    
    # 6. Add data quality flags
    print(f"\n STEP 6: Adding quality indicators...")
    
    # Title length category
    df_quality['title_length'] = df_quality['title'].str.len()
    df_quality['title_length_category'] = pd.cut(
        df_quality['title_length'], 
        bins=[0, 30, 60, 100, np.inf], 
        labels=['short', 'medium', 'long', 'very_long']
    )
    
    # text length category
    df_quality['text_length'] = df_quality['text'].str.len()
    df_quality['text_length_category'] = pd.cut(
        df_quality['text_length'], 
        bins=[0, 100, 200, 400, np.inf], 
        labels=['short', 'medium', 'long', 'very_long']
    )
    
    
    # 7. Generate cleaning report
    print(f"\n DATA CLEANING SUMMARY:")
    print(f"="*30)
    print(f"  Original articles: {initial_count}")
    print(f"  After deduplication: {len(df_dedupe)} (-{duplicates_removed})")
    print(f"  After quality filter: {len(df_quality)} (-{quality_removed})")
    print(f"  Final dataset: {len(df_quality)} articles")
    print(f"  Retention rate: {len(df_quality)/initial_count:.1%}")
    
    # Sentiment distribution after cleaning
    print(f"\n CLEANED DATA DISTRIBUTION:")
    sentiment_dist = df_quality['sentiment_label'].value_counts(normalize=True)
    for label, pct in sentiment_dist.items():
        print(f"  {label}: {pct:.1%}")
    
    print(f"\n SENTIMENT SCORE STATISTICS:")
    print(f"  Mean: {df_quality['sentiment_score'].mean():.3f}")
    print(f"  Std:  {df_quality['sentiment_score'].std():.3f}")
    print(f"  Min:  {df_quality['sentiment_score'].min():.3f}")
    print(f"  Max:  {df_quality['sentiment_score'].max():.3f}")
    
    # Save cleaned data
    output_file = 'data/processed/tesla_sentiment_cleaned.csv'
    df_quality.to_csv(output_file, index=False)
    print(f"\n Cleaned data saved: {output_file}")
    
    return df_quality

def create_cleaning_visualizations(df_clean):
    """Create visualizations showing data cleaning impact"""
    print(f"\n Creating data quality visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tesla Sentiment Data Quality Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Sentiment distribution
    sentiment_counts = df_clean['sentiment_label'].value_counts()
    axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    axes[0,0].set_title('Sentiment Distribution After Cleaning')
    
    # Plot 2: Article length distribution
    axes[0,1].hist(df_clean['title_length'], bins=20, alpha=0.7, color='blue', label='Title Length')
    axes[0,1].hist(df_clean['text_length'], bins=20, alpha=0.7, color='red', label='text Length')
    axes[0,1].set_xlabel('Character Count')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Article Length Distribution')
    axes[0,1].legend()
    
    # Plot 3: Sentiment scores over time
    daily_sentiment = df_clean.groupby('date_only')['sentiment_score'].mean()
    axes[1,0].plot(daily_sentiment.index, daily_sentiment.values, marker='o')
    axes[1,0].set_xlabel('Date')
    axes[1,0].set_ylabel('Average Sentiment Score')
    axes[1,0].set_title('Daily Sentiment Trend')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    
    plt.tight_layout()
    plt.savefig('data/processed/tesla_data_quality_analysis.png', dpi=300, bbox_inches='tight')
    print(f" Visualization saved: data/processed/tesla_data_quality_analysis.png")
    plt.show()

def clean_financial_twitter_data():
    """Basic cleaning for financial Twitter dataset"""
    print(f"\n CLEANING FINANCIAL TWITTER DATA:")
    print("="*40)
    
    try:
        df = pd.read_csv('data/processed/financial_twitter_labeled.csv')
        print(f" Loaded {len(df)} financial examples")
    except FileNotFoundError:
        print(" Financial Twitter data not found. Run Phase 2 first!")
        return None
    
    # Remove duplicates
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=['text'], keep='first')
    print(f"  Duplicates removed: {initial_count - len(df_clean)}")
    
    # Filter by text length
    df_clean = df_clean[df_clean['text'].str.len() >= 10]
    print(f"  Short texts removed: {len(df) - len(df_clean)}")
    
    # Check class balance
    print(f"\n CLEANED FINANCIAL DATA DISTRIBUTION:")
    class_dist = df_clean['label'].value_counts(normalize=True)
    for label, pct in class_dist.items():
        print(f"  {label}: {pct:.1%}")
    
    # Save cleaned financial data
    output_file = 'data/processed/financial_twitter_cleaned.csv'
    df_clean.to_csv(output_file, index=False)
    print(f" Cleaned financial data saved: {output_file}")
    
    return df_clean

def main():
    """Main data cleaning pipeline"""
    print(" TESLA SENTIMENT DATA CLEANING PIPELINE")
    print("="*50)
    
    # Step 1: Clean Tesla sentiment data
    tesla_clean = clean_tesla_sentiment_data()
    
    if tesla_clean is not None:
        # Step 2: Create quality visualizations
        create_cleaning_visualizations(tesla_clean)
        
        # Step 3: Clean financial Twitter data
        financial_clean = clean_financial_twitter_data()
        
        print(f"\n DATA CLEANING COMPLETE!")
        print(f" Tesla articles cleaned: {len(tesla_clean) if tesla_clean is not None else 0}")
        print(f" Financial examples cleaned: {len(financial_clean) if financial_clean is not None else 0}")
        print(f" Quality analysis charts created")
        print(f" Cleaned datasets ready for model training")
        
        return tesla_clean, financial_clean
    
    else:
        print(" Data cleaning failed. Check that Phase 2 sentiment analysis completed successfully.")
        return None, None

if __name__ == "__main__":
    results = main()