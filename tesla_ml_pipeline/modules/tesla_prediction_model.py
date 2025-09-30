import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load cleaned data and prepare for model training"""
    print(" LOADING AND PREPARING CLEAN DATA")
    print("="*50)
    
    # Load cleaned Tesla sentiment data
    print(" Loading Tesla sentiment data...")
    try:
        tesla_df = pd.read_csv('data/processed/tesla_sentiment_cleaned.csv')
        print(f" Tesla articles: {len(tesla_df)}")
    except FileNotFoundError:
        print(" Tesla cleaned data not found. Run data cleaning first!")
        return None, None
    
    # Load Tesla stock data
    print(" Loading Tesla stock data...")
    tesla_stock = yf.download('TSLA', start='2025-08-01', end='2025-10-01', progress=False)
    tesla_stock.reset_index(inplace=True)
    tesla_stock['daily_return'] = tesla_stock['Close'].pct_change()
    tesla_stock['direction'] = (tesla_stock['daily_return'] > 0).astype(int)
    print(f" Tesla stock data: {len(tesla_stock)} days")
    
    return tesla_df, tesla_stock

def create_tesla_features(tesla_df, tesla_stock):
    """Create features from Tesla sentiment data"""
    print("\n🔧 CREATING TESLA FEATURES")
    print("="*30)
    
    # FIX: Flatten tesla_stock MultiIndex columns first
    if isinstance(tesla_stock.columns, pd.MultiIndex):
        tesla_stock.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in tesla_stock.columns]
    
    # Convert Tesla dates and aggregate daily sentiment
    tesla_df['date'] = pd.to_datetime(tesla_df['date'])
    tesla_df['date_only'] = tesla_df['date'].dt.date
    
    # Daily sentiment aggregation
    daily_sentiment = tesla_df.groupby('date_only').agg({
        'sentiment_score': ['mean', 'std', 'min', 'max', 'count'],
        'sentiment_label': [
            lambda x: (x == 'positive').mean(),
            lambda x: (x == 'negative').mean(),
            lambda x: (x == 'neutral').mean()
        ]
    }).round(4)
    
    # Flatten column names
    daily_sentiment.columns = [
        'avg_sentiment_score', 'sentiment_std', 'min_sentiment', 'max_sentiment', 'article_count',
        'positive_ratio', 'negative_ratio', 'neutral_ratio'
    ]
    
    # Reset index to make date_only a regular column
    daily_sentiment = daily_sentiment.reset_index()
    
    # Fill NaN standard deviation with 0 (single article days)
    daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
    
    # Prepare stock data for merging
    tesla_stock['date_only'] = tesla_stock['Date'].dt.date
    
    # Merge sentiment with stock data
    merged_df = pd.merge(tesla_stock, daily_sentiment, on='date_only', how='inner')
    
    print(f"📊 Merged dataset: {len(merged_df)} days with both sentiment and stock data")
    
    # Rest stays the same...
    print("🔧 Engineering advanced features...")
    
    # Sentiment extreme flags
    merged_df['sentiment_extreme_positive'] = (merged_df['avg_sentiment_score'] > 0.85).astype(int)
    merged_df['sentiment_extreme_negative'] = (merged_df['avg_sentiment_score'] < 0.75).astype(int)
    merged_df['sentiment_extreme'] = np.where(
        merged_df['avg_sentiment_score'] > 0.85, 1,
        np.where(merged_df['avg_sentiment_score'] < 0.75, -1, 0)
    )
    
    # Sentiment volatility and momentum
    merged_df['sentiment_volatility'] = merged_df['sentiment_std']
    merged_df['high_article_volume'] = (merged_df['article_count'] > merged_df['article_count'].median()).astype(int)
    
    # Lagged features (previous day)
    merged_df = merged_df.sort_values('date_only')
    merged_df['prev_return'] = merged_df['daily_return'].shift(1)
    merged_df['prev_sentiment'] = merged_df['avg_sentiment_score'].shift(1)
    
    # Rolling features (3-day windows)
    merged_df['sentiment_ma3'] = merged_df['avg_sentiment_score'].rolling(window=3, min_periods=1).mean()
    merged_df['return_ma3'] = merged_df['daily_return'].rolling(window=3, min_periods=1).mean()
    
    # Sentiment-return interaction
    merged_df['sentiment_return_interaction'] = merged_df['avg_sentiment_score'] * merged_df['prev_return']
    
    # Next day direction target (what we want to predict)
    merged_df['next_day_direction'] = merged_df['direction'].shift(-1)
    
    # Remove rows with NaN target
    merged_df = merged_df.dropna(subset=['next_day_direction'])
    
    print(f" Feature engineering complete: {merged_df.shape[1]} features created")
    print(f" Final dataset for Tesla: {len(merged_df)} days")
    
    return merged_df

def train_tesla_prediction_models(tesla_features):
    """Train Tesla-specific prediction models"""
    print("\n TRAINING TESLA PREDICTION MODELS")
    print("="*40)
    
    # Define feature columns for Tesla model
    feature_columns = [
        'avg_sentiment_score', 'sentiment_std', 'min_sentiment', 'max_sentiment',
        'positive_ratio', 'negative_ratio', 'neutral_ratio', 'article_count',
        'sentiment_extreme_positive', 'sentiment_extreme_negative', 'sentiment_extreme',
        'sentiment_volatility', 'high_article_volume', 'prev_return', 'prev_sentiment',
        'sentiment_ma3', 'return_ma3', 'sentiment_return_interaction'
    ]
    
    # Prepare features and target
    X = tesla_features[feature_columns]
    y = tesla_features['next_day_direction']
    
    # FIX: Handle NaN values
    print(" Handling missing values...")
    X = X.fillna(0)  # Fill NaN with 0
    
    print(f" Tesla training samples: {len(X)}")
    print(f" Features: {len(feature_columns)}")
    
    # Check if we have enough data
    if len(X) < 10:
        print("  WARNING: Very small dataset! Results may not be reliable.")
        print("  Consider collecting more data for production use.")
    
    print(" Target distribution:")
    target_dist = y.value_counts(normalize=True)
    for direction, pct in target_dist.items():
        direction_label = 'UP' if direction == 1 else 'DOWN'
        print(f"   {direction_label}: {pct:.1%}")
    
    # For small datasets, use simple train-test split instead of CV
    if len(X) < 15:
        print(" Using simple train-test split (dataset too small for CV)")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        # Time series split for larger datasets
        print(" Using TimeSeriesSplit for validation...")
        tscv = TimeSeriesSplit(n_splits=3)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    models = {}
    
    # 1. Logistic Regression
    print(" Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    
    if len(X) < 15:
        # Simple fit for small data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr_model.fit(X_train_scaled, y_train)
        lr_score = lr_model.score(X_test_scaled, y_test)
        lr_scores = np.array([lr_score])  # Single score
    else:
        # Cross-validation for larger data
        lr_scores = cross_val_score(lr_model, X_scaled, y, cv=tscv, scoring='accuracy')
        lr_model.fit(X_scaled, y)
    
    models['Logistic Regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'cv_scores': lr_scores,
        'mean_cv': lr_scores.mean()
    }
    
    # 2. Random Forest
    print(" Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=50,  # Reduced for small data
        random_state=42, 
        class_weight='balanced'
    )
    
    if len(X) < 15:
        # Simple fit for small data
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_test, y_test)
        rf_scores = np.array([rf_score])  # Single score
    else:
        # Cross-validation for larger data
        rf_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='accuracy')
        rf_model.fit(X, y)
    
    models['Random Forest'] = {
        'model': rf_model,
        'scaler': None,  # RF doesn't need scaling
        'cv_scores': rf_scores,
        'mean_cv': rf_scores.mean()
    }
    
    # Print model performance
    print("\n📊 MODEL PERFORMANCE SUMMARY:")
    for name, model_info in models.items():
        cv_mean = model_info['mean_cv']
        if len(model_info['cv_scores']) > 1:
            cv_std = model_info['cv_scores'].std()
            print(f"   {name}: {cv_mean:.3f} ± {cv_std:.3f}")
        else:
            print(f"   {name}: {cv_mean:.3f} (single validation)")
    
    # Select best model
    best_model_name = max(models.keys(), key=lambda k: models[k]['mean_cv'])
    best_model_info = models[best_model_name]
    
    print(f"\n Best model: {best_model_name} (Score: {best_model_info['mean_cv']:.3f})")
    
    return models, best_model_name, feature_columns

def analyze_feature_importance(models, feature_columns):
    """Analyze which features are most important"""
    print("\n FEATURE IMPORTANCE ANALYSIS")
    print("="*35)
    
    # Get Random Forest feature importance
    rf_model = models['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("📊 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Create feature importance visualization
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Tesla Stock Direction Prediction - Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('data/processed/tesla_feature_importance.png', dpi=300, bbox_inches='tight')
    print(" Feature importance chart saved: data/processed/tesla_feature_importance.png")
    plt.show()
    
    return feature_importance

def backtest_trading_strategy(tesla_features, models, best_model_name, feature_columns):
    """Backtest the sentiment-driven trading strategy"""
    print("\n BACKTESTING TRADING STRATEGY")
    print("="*35)
    
    # Get the best model
    best_model_info = models[best_model_name]
    model = best_model_info['model']
    scaler = best_model_info['scaler']
    
    # Prepare features - FIX: Handle NaN values here too
    X = tesla_features[feature_columns]
    X = X.fillna(0)  # Fill NaN with 0, same as training
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        pred_proba = model.predict_proba(X_scaled)
    else:
        predictions = model.predict(X)
        pred_proba = model.predict_proba(X)
    
    # Add predictions to dataframe
    tesla_features = tesla_features.copy()
    tesla_features['predicted_direction'] = predictions
    tesla_features['prediction_confidence'] = pred_proba.max(axis=1)
    
    # Calculate strategy returns
    tesla_features['actual_next_return'] = tesla_features['daily_return'].shift(-1)
    
    # Strategy: Buy if predict UP (1), Sell if predict DOWN (0)
    tesla_features['strategy_signal'] = tesla_features['predicted_direction']
    tesla_features['strategy_return'] = np.where(
        tesla_features['strategy_signal'] == 1,
        tesla_features['actual_next_return'],  # Go long
        -tesla_features['actual_next_return']  # Go short
    )
    
    # Calculate performance metrics
    total_days = len(tesla_features.dropna(subset=['strategy_return']))
    strategy_returns = tesla_features['strategy_return'].dropna()
    actual_returns = tesla_features['actual_next_return'].dropna()
    
    # Handle edge case of very small data
    if len(strategy_returns) == 0 or len(actual_returns) == 0:
        print(" No valid returns for backtesting!")
        return {'accuracy': 0, 'strategy_total_return': 0, 'buy_hold_total_return': 0}, tesla_features
    
    # Basic performance metrics
    strategy_total_return = (1 + strategy_returns).prod() - 1
    buy_hold_total_return = (1 + actual_returns).prod() - 1
    
    if total_days > 0:
        strategy_annual_return = (1 + strategy_total_return) ** (252/total_days) - 1
        buy_hold_annual_return = (1 + buy_hold_total_return) ** (252/total_days) - 1
    else:
        strategy_annual_return = 0
        buy_hold_annual_return = 0
    
    # Risk metrics - handle division by zero
    if strategy_returns.std() > 0:
        strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    else:
        strategy_sharpe = 0
        
    if actual_returns.std() > 0:
        buy_hold_sharpe = actual_returns.mean() / actual_returns.std() * np.sqrt(252)
    else:
        buy_hold_sharpe = 0
    
    # Accuracy metrics - handle mismatched lengths
    valid_predictions = len(tesla_features['predicted_direction'][:-1])
    valid_actuals = len(tesla_features['next_day_direction'].dropna())
    
    if valid_predictions > 0 and valid_actuals > 0:
        # Match lengths for accuracy calculation
        min_length = min(valid_predictions, valid_actuals)
        accuracy = accuracy_score(
            tesla_features['next_day_direction'].dropna().iloc[:min_length], 
            tesla_features['predicted_direction'].iloc[:min_length]
        )
    else:
        accuracy = 0
    
    # Win rate
    win_rate = (strategy_returns > 0).mean() if len(strategy_returns) > 0 else 0
    
    print(f" BACKTESTING RESULTS ({total_days} days):")
    print(f"="*40)
    print(f" Model Accuracy: {accuracy:.1%}")
    print(f" Strategy Total Return: {strategy_total_return:.1%}")
    print(f" Buy & Hold Total Return: {buy_hold_total_return:.1%}")
    print(f" Strategy Annualized Return: {strategy_annual_return:.1%}")
    print(f" Buy & Hold Annualized Return: {buy_hold_annual_return:.1%}")
    print(f" Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f" Buy & Hold Sharpe Ratio: {buy_hold_sharpe:.2f}")
    print(f" Strategy Win Rate: {win_rate:.1%}")
    
    # Create simple performance visualization for small data
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Cumulative returns
    plt.subplot(2, 2, 1)
    if len(strategy_returns) > 0 and len(actual_returns) > 0:
        strategy_cumulative = (1 + strategy_returns).cumprod()
        buyhold_cumulative = (1 + actual_returns).cumprod()
        
        plt.plot(strategy_cumulative.values, label='Sentiment Strategy', linewidth=2)
        plt.plot(buyhold_cumulative.values, label='Buy & Hold', linewidth=2)
    plt.xlabel('Days')
    plt.ylabel('Cumulative Return')
    plt.title('Strategy vs Buy & Hold Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Feature importance (from Random Forest)
    plt.subplot(2, 2, 2)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']['model']
        feature_imp = rf_model.feature_importances_
        top_5_idx = np.argsort(feature_imp)[-5:]
        plt.barh(range(5), feature_imp[top_5_idx])
        plt.yticks(range(5), [feature_columns[i] for i in top_5_idx])
        plt.xlabel('Importance')
        plt.title('Top 5 Feature Importance')
    
    # Subplot 3: Prediction confidence
    plt.subplot(2, 2, 3)
    if len(tesla_features['prediction_confidence']) > 1:
        plt.plot(tesla_features['prediction_confidence'][:-1], alpha=0.7)
    plt.xlabel('Days')
    plt.ylabel('Model Confidence')
    plt.title('Model Prediction Confidence Over Time')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Simple accuracy display
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f'Model Accuracy\n{accuracy:.1%}', 
             ha='center', va='center', fontsize=20, 
             transform=plt.gca().transAxes)
    plt.title('Overall Accuracy')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/processed/tesla_backtest_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Backtesting analysis saved: data/processed/tesla_backtest_analysis.png")
    plt.show()
    
    # Save results
    results = {
        'accuracy': accuracy,
        'strategy_total_return': strategy_total_return,
        'buy_hold_total_return': buy_hold_total_return,
        'strategy_sharpe': strategy_sharpe,
        'buy_hold_sharpe': buy_hold_sharpe,
        'win_rate': win_rate,
        'total_days': total_days
    }
    
    return results, tesla_features

def main():
    """Main model training and backtesting pipeline"""
    print(" TESLA SENTIMENT PREDICTION MODEL PIPELINE")
    print("="*55)
    
    # Step 1: Load and prepare data
    tesla_df, tesla_stock = load_and_prepare_data()
    
    if tesla_df is None or tesla_stock is None:
        print(" Data loading failed. Make sure you've run data cleaning first!")
        return None, None, None
    
    # Step 2: Create Tesla features
    tesla_features = create_tesla_features(tesla_df, tesla_stock)
    
    # Step 3: Train Tesla-specific models
    models, best_model_name, feature_columns = train_tesla_prediction_models(tesla_features)
    
    # Step 4: Analyze feature importance
    feature_importance = analyze_feature_importance(models, feature_columns)
    
    # Step 5: Backtest trading strategy
    results, final_data = backtest_trading_strategy(tesla_features, models, best_model_name, feature_columns)
    
    # Step 6: Final summary
    print(f"\n MODEL TRAINING COMPLETE!")
    print(f"="*30)
    print(f" Tesla model trained on {len(tesla_features)} days of data")
    print(f" Best model: {best_model_name}")
    print(f" Final accuracy: {results['accuracy']:.1%}")
    print(f" Strategy outperformance: {results['strategy_total_return'] - results['buy_hold_total_return']:.1%}")
    print(f" All results and visualizations saved to data/processed/")
    
    return models, results, final_data

if __name__ == "__main__":
    models, results, data = main()