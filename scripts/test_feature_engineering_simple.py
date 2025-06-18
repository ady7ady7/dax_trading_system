#!/usr/bin/env python3
"""
Simple test script for the FeatureEngineer class.

Location: Save this as scripts/test_feature_engineering_simple.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

print(f"🔍 Project root: {project_root}")

# Import the FeatureEngineer
try:
    from src.features.engineering import FeatureEngineer
    print("✅ Successfully imported FeatureEngineer")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_historical_data():
    """Load historical DAX data from parquet file."""
    
    data_file = project_root / "data" / "processed" / "historical_data.parquet"
    
    if not data_file.exists():
        print(f"❌ Historical data file not found: {data_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(data_file)
        print(f"✅ Loaded historical data: {len(df):,} records")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        return pd.DataFrame()

def create_simple_test_data():
    """Create simple test data with proper pandas operations."""
    
    print("📊 Creating simple test data...")
    
    # Create time index
    dates = pd.date_range('2024-01-15 09:00', '2024-01-15 17:30', freq='1min', tz='Europe/Berlin')
    
    # Create base price series (pandas Series, not numpy array)
    np.random.seed(42)
    base_price = 18000
    n_periods = len(dates)
    
    # Generate returns as pandas Series
    returns = pd.Series(np.random.normal(0, 0.0005, n_periods), index=dates)
    returns.iloc[0] = 0
    
    # Calculate prices (cumulative product)
    prices = (1 + returns).cumprod() * base_price
    
    # Generate OHLC data using pandas operations
    opens = prices.shift(1).fillna(base_price)
    closes = prices
    
    # Generate highs and lows
    volatility = np.abs(returns) * prices
    highs = pd.concat([opens, closes], axis=1).max(axis=1) + np.random.exponential(volatility.abs() * 0.5)
    lows = pd.concat([opens, closes], axis=1).min(axis=1) - np.random.exponential(volatility.abs() * 0.5)
    
    # Ensure OHLC relationships
    highs = pd.concat([highs, opens, closes], axis=1).max(axis=1)
    lows = pd.concat([lows, opens, closes], axis=1).min(axis=1)
    
    # Generate volume
    volume = pd.Series(np.random.poisson(1000, n_periods), index=dates)
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volume
    }, index=dates)
    
    print(f"✅ Test data created: {len(test_df)} records")
    return test_df

def main():
    """Main test function."""
    
    print("🧪 Simple Feature Engineering Test")
    print("=" * 50)
    
    # Try to load real data first
    print("\n1️⃣ Loading historical data...")
    df = load_historical_data()
    
    if df.empty:
        print("📊 No historical data found, creating test data...")
        df = create_simple_test_data()
    else:
        # Use a smaller sample for testing
        sample_size = 5000  # ~3.5 days of 1-minute data
        if len(df) > sample_size:
            df = df.tail(sample_size)
            print(f"📊 Using last {sample_size:,} records for testing")
    
    if df.empty:
        print("❌ No data available for testing")
        return
    
    # Initialize and test feature engineering
    print(f"\n2️⃣ Testing feature engineering...")
    
    try:
        feature_engineer = FeatureEngineer()
        print("✅ FeatureEngineer initialized")
        
        # Run feature engineering
        start_time = datetime.now()
        features_df = feature_engineer.engineer_features(df)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Calculate results
        original_cols = len(df.columns)
        total_cols = len(features_df.columns)
        new_features = total_cols - original_cols
        
        print(f"\n🎉 Feature Engineering Successful!")
        print(f"   Processing time: {duration:.2f} seconds")
        print(f"   Records processed: {len(df):,}")
        print(f"   Original columns: {original_cols}")
        print(f"   Total columns: {total_cols}")
        print(f"   New features generated: {new_features}")
        
        # Show some sample features
        feature_cols = [col for col in features_df.columns if col not in df.columns]
        print(f"\n📊 Sample of generated features:")
        
        # Group features by category
        momentum_features = [f for f in feature_cols if any(x in f for x in ['RSI', 'MACD', 'Stoch', 'CCI', 'Williams'])]
        trend_features = [f for f in feature_cols if any(x in f for x in ['EMA', 'ADX', 'DI_'])]
        volatility_features = [f for f in feature_cols if any(x in f for x in ['ATR', 'BB_', 'Volatility'])]
        volume_features = [f for f in feature_cols if any(x in f for x in ['OBV', 'CMF', 'Volume'])]
        dax_features = [f for f in feature_cols if any(x in f for x in ['Gap', 'Opening', 'EU_Session'])]
        session_features = [f for f in feature_cols if any(x in f for x in ['Is_', 'Hour', 'London'])]
        
        categories = [
            ("Momentum", momentum_features),
            ("Trend", trend_features), 
            ("Volatility", volatility_features),
            ("Volume", volume_features),
            ("DAX-Specific", dax_features),
            ("Session", session_features)
        ]
        
        for category, features in categories:
            if features:
                print(f"   {category} ({len(features)}): {', '.join(features[:3])}{'...' if len(features) > 3 else ''}")
        
        # Data quality check
        total_cells = len(features_df) * len(feature_cols)
        non_null_cells = features_df[feature_cols].count().sum()
        completeness = (non_null_cells / total_cells) * 100
        
        print(f"\n📈 Data Quality:")
        print(f"   Feature completeness: {completeness:.1f}%")
        
        # Show a few sample values
        print(f"\n🔍 Sample feature values (last row):")
        sample_features = ['RSI_14', 'MACD', 'ATR', 'EMA_Cross_5_10', 'Volume_Ratio']
        available_sample = [f for f in sample_features if f in features_df.columns]
        
        if available_sample:
            last_values = features_df[available_sample].iloc[-1]
            for feature in available_sample:
                value = last_values[feature]
                if pd.notna(value):
                    print(f"   {feature}: {value:.4f}")
                else:
                    print(f"   {feature}: NaN")
        
        print(f"\n✅ Feature engineering test completed successfully!")
        print(f"🚀 Ready for strategy development with {new_features} features!")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()