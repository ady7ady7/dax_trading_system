#!/usr/bin/env python3
"""
Test script for the FeatureEngineer class using real historical data.

This script demonstrates how to use the FeatureEngineer with your actual
DAX historical data from the historical_data.parquet file.

Usage: python test_feature_engineering.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import the FeatureEngineer
try:
    from src.features.engineering import FeatureEngineer
except ImportError:
    print("❌ Could not import FeatureEngineer. Ensure the module is in src/features/engineering.py")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_historical_data() -> pd.DataFrame:
    """Load historical DAX data from parquet file."""
    
    data_file = Path("data/processed/historical_data.parquet")
    
    if not data_file.exists():
        logger.error(f"Historical data file not found: {data_file}")
        logger.error("Run 'python main.py' first to generate historical_data.parquet")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(data_file)
        logger.info(f"Loaded historical data: {len(df):,} records")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return pd.DataFrame()


def test_feature_engineering_performance(df: pd.DataFrame, sample_size: int = 10080) -> None:
    """Test feature engineering performance on different data sizes."""
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PERFORMANCE TEST")
    print("="*60)
    
    # Test with different sample sizes
    test_sizes = [1440, 5040, sample_size]  # 1 day, ~3.5 days, 1 week
    
    for size in test_sizes:
        if len(df) < size:
            continue
            
        print(f"\n🧪 Testing with {size:,} records ({size/1440:.1f} days)...")
        
        # Sample data
        sample_df = df.tail(size).copy()
        
        # Time the feature engineering
        start_time = datetime.now()
        
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.engineer_features(sample_df)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        original_cols = len(sample_df.columns)
        feature_cols = len(features_df.columns)
        new_features = feature_cols - original_cols
        
        records_per_sec = size / duration if duration > 0 else 0
        
        print(f"  ⏱️  Processing time: {duration:.2f} seconds")
        print(f"  📊 Records/second: {records_per_sec:,.0f}")
        print(f"  🔧 Features generated: {new_features}")
        print(f"  💾 Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


def analyze_feature_quality(features_df: pd.DataFrame) -> None:
    """Analyze the quality and characteristics of generated features."""
    
    print("\n" + "="*60)
    print("FEATURE QUALITY ANALYSIS")
    print("="*60)
    
    # Get feature columns (excluding original OHLCV)
    original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in features_df.columns if col not in original_cols]
    
    print(f"\n📈 Feature Overview:")
    print(f"  Total features generated: {len(feature_cols)}")
    
    # Analyze missing values
    missing_analysis = []
    for col in feature_cols:
        missing_count = features_df[col].isnull().sum()
        missing_pct = (missing_count / len(features_df)) * 100
        missing_analysis.append((col, missing_count, missing_pct))
    
    # Sort by missing percentage
    missing_analysis.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n🔍 Missing Value Analysis:")
    high_missing = [item for item in missing_analysis if item[2] > 5]
    if high_missing:
        print(f"  Features with >5% missing values: {len(high_missing)}")
        for col, count, pct in high_missing[:5]:
            print(f"    • {col}: {pct:.1f}% missing")
    else:
        print(f"  ✅ No features with >5% missing values")
    
    # Analyze feature distributions
    print(f"\n📊 Feature Distribution Analysis:")
    
    # Find features with potential issues
    problematic_features = []
    
    for col in feature_cols:
        series = features_df[col].dropna()
        if len(series) == 0:
            continue
            
        # Check for constant features
        if series.nunique() == 1:
            problematic_features.append((col, "constant"))
            continue
        
        # Check for extreme outliers (>6 standard deviations)
        if series.std() > 0:
            outliers = np.abs((series - series.mean()) / series.std()) > 6
            outlier_pct = (outliers.sum() / len(series)) * 100
            if outlier_pct > 1:  # More than 1% outliers
                problematic_features.append((col, f"{outlier_pct:.1f}% extreme outliers"))
        
        # Check for infinite values
        inf_count = np.isinf(series).sum()
        if inf_count > 0:
            problematic_features.append((col, f"{inf_count} infinite values"))
    
    if problematic_features:
        print(f"  ⚠️  Potential issues found in {len(problematic_features)} features:")
        for col, issue in problematic_features[:5]:
            print(f"    • {col}: {issue}")
    else:
        print(f"  ✅ No major distribution issues detected")
    
    # Show statistics for key features
    key_features = ['RSI_14', 'MACD', 'ATR', 'BB_Width_20', 'Opening_Gap_Normalized']
    available_key_features = [f for f in key_features if f in features_df.columns]
    
    if available_key_features:
        print(f"\n📋 Key Feature Statistics:")
        stats = features_df[available_key_features].describe()
        print(stats.round(4))


def test_dax_specific_features(features_df: pd.DataFrame) -> None:
    """Test and analyze DAX-specific features."""
    
    print("\n" + "="*60)
    print("DAX-SPECIFIC FEATURES ANALYSIS")
    print("="*60)
    
    # Opening gap analysis
    if 'Opening_Gap' in features_df.columns:
        gaps = features_df['Opening_Gap'].dropna()
        gap_days = (gaps != 0).sum()
        
        print(f"\n📊 Opening Gap Analysis:")
        print(f"  Days with gaps: {gap_days}")
        if gap_days > 0:
            print(f"  Average gap: €{gaps[gaps != 0].mean():.2f}")
            print(f"  Largest gap: €{gaps.max():.2f}")
            print(f"  Smallest gap: €{gaps.min():.2f}")
    
    # Session pattern analysis
    session_features = ['Is_Opening_Hour', 'Is_Lunch_Time', 'Is_Closing_Hour']
    available_session = [f for f in session_features if f in features_df.columns]
    
    if available_session:
        print(f"\n🕐 European Session Patterns:")
        for feature in available_session:
            active_periods = features_df[feature].sum()
            total_periods = len(features_df)
            pct = (active_periods / total_periods) * 100
            print(f"  {feature}: {active_periods:,} periods ({pct:.1f}%)")
    
    # Volume patterns during different sessions
    if 'Volume' in features_df.columns and 'Is_Market_Hours' in features_df.columns:
        market_hours_vol = features_df[features_df['Is_Market_Hours']]['Volume'].mean()
        after_hours_vol = features_df[~features_df['Is_Market_Hours']]['Volume'].mean()
        
        print(f"\n📈 Volume Patterns:")
        print(f"  Average market hours volume: {market_hours_vol:,.0f}")
        print(f"  Average after hours volume: {after_hours_vol:,.0f}")
        print(f"  Market/After hours ratio: {market_hours_vol/after_hours_vol:.2f}x")


def save_feature_sample(features_df: pd.DataFrame, output_path: str = "data/processed/feature_sample.csv") -> None:
    """Save a sample of features for inspection."""
    
    try:
        # Save last 1000 records as sample
        sample_df = features_df.tail(1000)
        sample_df.to_csv(output_path)
        
        print(f"\n💾 Feature sample saved: {output_path}")
        print(f"  Records: {len(sample_df)}")
        print(f"  Features: {len(sample_df.columns)}")
        
    except Exception as e:
        logger.error(f"Error saving feature sample: {e}")


def main():
    """Main test function."""
    
    print("🧪 DAX Feature Engineering - Comprehensive Test")
    print("="*60)
    
    # Load historical data
    print("\n1️⃣ Loading historical data...")
    df = load_historical_data()
    
    if df.empty:
        print("❌ No historical data available for testing")
        print("💡 Run 'python main.py' to generate historical_data.parquet first")
        return
    
    # Limit data size for testing (last week of data)
    max_records = 10080  # 1 week of 1-minute data
    if len(df) > max_records:
        df = df.tail(max_records)
        print(f"📊 Using last {max_records:,} records for testing")
    
    # Initialize and run feature engineering
    print("\n2️⃣ Running feature engineering...")
    
    try:
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.engineer_features(df)
        
        original_cols = len(df.columns)
        total_cols = len(features_df.columns)
        new_features = total_cols - original_cols
        
        print(f"✅ Feature engineering completed!")
        print(f"  Original columns: {original_cols}")
        print(f"  Total columns: {total_cols}")
        print(f"  New features: {new_features}")
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run analysis tests
    print("\n3️⃣ Analyzing feature quality...")
    analyze_feature_quality(features_df)
    
    print("\n4️⃣ Testing DAX-specific features...")
    test_dax_specific_features(features_df)
    
    print("\n5️⃣ Testing performance...")
    test_feature_engineering_performance(df)
    
    # Generate comprehensive report
    print("\n6️⃣ Generating feature report...")
    report = feature_engineer.generate_feature_report(features_df)
    print(report)
    
    # Save sample for inspection
    print("\n7️⃣ Saving feature sample...")
    save_feature_sample(features_df)
    
    # Show feature descriptions
    print("\n8️⃣ Feature descriptions available:")
    descriptions = feature_engineer.get_feature_descriptions()
    print(f"  Total feature descriptions: {len(descriptions)}")
    print(f"  Access with: feature_engineer.get_feature_descriptions()")
    
    print("\n✅ Comprehensive feature engineering test completed!")
    print(f"📈 Ready for strategy development with {new_features} engineered features")


if __name__ == "__main__":
    main()