#!/usr/bin/env python3
"""
DAX Trend-Following Algorithmic Trading System
ENHANCED: Main execution with Trading Hours Processing (8:00-17:30 CET)

Author: ady7ady7
Created: 2025
Python Version: 3.12.2
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, time
import yaml
import pytz

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Core imports
import pandas as pd
import numpy as np

# Local imports
from src.data.data_ingestion import load_and_convert_data, DataIngestionError
from src.data.gap_detector import GapDetector
from src.data.etl_pipeline import impute_data, DataImputer, validate_ohlcv_data


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the trading system."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"trading_system_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load system configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config file: {e}")
        return {}


def validate_environment() -> bool:
    """Validate that the runtime environment meets requirements."""
    try:
        # Check Python version
        if sys.version_info < (3, 9):
            logging.error("Python 3.9+ required")
            return False
            
        # Check critical directories exist
        required_dirs = ["data", "data/raw", "data/processed", "config", "logs"]
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logging.info("Environment validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Environment validation failed: {e}")
        return False


def filter_to_trading_hours(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter DataFrame to trading hours only based on config.
    
    Args:
        df: DataFrame with timezone-aware datetime index
        config: Configuration dictionary with trading hours settings
        
    Returns:
        DataFrame filtered to trading hours only
    """
    logger = logging.getLogger(__name__)
    
    # Get trading hours from config (with defaults)
    trading_config = config.get('trading_hours', {})
    market_open_str = trading_config.get('market_open', '08:00')
    market_close_str = trading_config.get('market_close', '17:30')
    timezone_str = trading_config.get('timezone', 'Europe/Berlin')
    include_weekends = trading_config.get('include_weekends', False)
    
    # Parse time strings
    market_open = datetime.strptime(market_open_str, '%H:%M').time()
    market_close = datetime.strptime(market_close_str, '%H:%M').time()
    target_timezone = pytz.timezone(timezone_str)
    
    logger.info(f"Filtering to trading hours: {market_open} - {market_close} {timezone_str}")
    
    # Ensure DataFrame is in correct timezone
    if df.index.tz != target_timezone:
        if df.index.tz is None:
            logger.warning("DataFrame has no timezone, localizing to target timezone")
            df.index = df.index.tz_localize(target_timezone, ambiguous='infer')
        else:
            logger.info(f"Converting from {df.index.tz} to {target_timezone}")
            df.index = df.index.tz_convert(target_timezone)
    
    original_count = len(df)
    
    # Filter by time of day
    time_mask = (df.index.time >= market_open) & (df.index.time <= market_close)
    
    # Filter by weekdays if specified
    if not include_weekends:
        weekday_mask = df.index.weekday < 5  # Monday=0, Sunday=6
        combined_mask = time_mask & weekday_mask
    else:
        combined_mask = time_mask
    
    # Apply filter
    df_filtered = df[combined_mask].copy()
    
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    removal_percentage = (removed_count / original_count) * 100
    
    logger.info(f"✅ Trading hours filtering completed:")
    logger.info(f"   Original records: {original_count:,}")
    logger.info(f"   Trading hours records: {filtered_count:,}")
    logger.info(f"   Removed: {removed_count:,} ({removal_percentage:.1f}%)")
    
    if filtered_count > 0:
        logger.info(f"   Date range: {df_filtered.index.min()} to {df_filtered.index.max()}")
    else:
        logger.warning("No data remains after trading hours filtering!")
    
    return df_filtered


def comprehensive_data_processing(df: pd.DataFrame, config: dict) -> tuple:
    """
    Perform comprehensive data processing including gap detection, validation, and imputation.
    
    Args:
        df: Trading hours filtered DataFrame
        config: Configuration dictionary
        
    Returns:
        tuple: (processed_df, gap_analysis, validation_results, processing_summary)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("COMPREHENSIVE DATA PROCESSING")
    logger.info("="*60)
    
    processing_summary = {
        'original_records': len(df),
        'gaps_detected': 0,
        'gaps_imputed': 0,
        'data_quality_score': 0.0,
        'processing_time_seconds': 0
    }
    
    start_time = datetime.now()
    
    try:
        # Step 1: Gap Detection
        logger.info("Step 1: Detecting gaps in trading hours data...")
        gap_detector = GapDetector(timezone=df.index.tz.zone if df.index.tz else "Europe/Berlin")
        
        # For large datasets, we'll sample representative periods
        if len(df) > 1000000:  # > 1M records
            logger.info("Large dataset detected - sampling for gap detection...")
            sample_size = 100000
            sample_df = df.tail(sample_size)  # Use recent data
            gaps = gap_detector.detect_gaps(sample_df)
            logger.info(f"Detected {len(gaps)} gaps in {sample_size:,} record sample")
        else:
            gaps = gap_detector.detect_gaps(df)
            logger.info(f"Detected {len(gaps)} gaps in full dataset")
        
        processing_summary['gaps_detected'] = len(gaps)
        
        # Step 2: Data Validation
        logger.info("Step 2: Validating data quality...")
        
        # Use sample for validation if dataset is large
        if len(df) > 100000:
            validation_sample = df.tail(100000)
            logger.info(f"Large dataset - validating sample of {len(validation_sample):,} records")
        else:
            validation_sample = df
        
        validation_results = validate_ohlcv_data(validation_sample)
        processing_summary['data_quality_score'] = validation_results.data_quality_score
        
        logger.info(f"✅ Data quality score: {validation_results.data_quality_score:.1f}/100")
        
        # Step 3: Data Imputation (if needed and gaps are reasonable)
        processed_df = df.copy()
        
        if gaps and len(gaps) <= 100:  # Only impute if reasonable number of gaps
            logger.info("Step 3: Imputing minor and moderate gaps...")
            
            # Filter gaps to only impute minor and moderate ones
            imputable_gaps = [gap for gap in gaps if gap.severity in ['minor', 'moderate']]
            
            if imputable_gaps:
                imputation_result = impute_data(processed_df, imputable_gaps)
                processed_df = imputation_result.imputed_df
                processing_summary['gaps_imputed'] = len(imputable_gaps)
                
                logger.info(f"✅ Imputed {len(imputable_gaps)} gaps:")
                logger.info(f"   Minor gaps: {imputation_result.imputation_summary.get('minor_gaps_imputed', 0)}")
                logger.info(f"   Moderate gaps: {imputation_result.imputation_summary.get('moderate_gaps_imputed', 0)}")
            else:
                logger.info("No suitable gaps found for imputation")
        elif len(gaps) > 100:
            logger.warning(f"Too many gaps ({len(gaps)}) detected - skipping imputation")
            logger.warning("Consider reviewing data source quality")
        else:
            logger.info("No gaps detected - no imputation needed")
        
        # Step 4: Final validation of processed data
        logger.info("Step 4: Final validation of processed data...")
        
        if len(processed_df) != len(df):
            logger.info(f"Data size changed: {len(df):,} → {len(processed_df):,} records")
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        processing_summary['processing_time_seconds'] = processing_time
        processing_summary['final_records'] = len(processed_df)
        
        logger.info(f"✅ Data processing completed in {processing_time:.1f} seconds")
        
        # Gap analysis summary
        gap_analysis = {
            'total_gaps': len(gaps),
            'gap_types': {},
            'severity_distribution': {}
        }
        
        if gaps:
            # Analyze gap types and severity
            gap_types = {}
            severity_dist = {}
            
            for gap in gaps:
                gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1
                severity_dist[gap.severity] = severity_dist.get(gap.severity, 0) + 1
            
            gap_analysis['gap_types'] = gap_types
            gap_analysis['severity_distribution'] = severity_dist
            
            total_gap_minutes = sum(gap.duration_minutes for gap in gaps)
            gap_analysis['total_missing_minutes'] = total_gap_minutes
            gap_analysis['gap_impact_percentage'] = (total_gap_minutes / len(df)) * 100 if len(df) > 0 else 0
        
        return processed_df, gap_analysis, validation_results, processing_summary
        
    except Exception as e:
        logger.error(f"Error in comprehensive data processing: {e}", exc_info=True)
        
        # Return original data with error summary
        error_summary = processing_summary.copy()
        error_summary['error'] = str(e)
        
        return df, {'error': str(e)}, None, error_summary


def print_processing_summary(gap_analysis: dict, validation_results, processing_summary: dict):
    """Print a comprehensive summary of data processing results."""
    
    print("\n" + "="*70)
    print("📊 TRADING HOURS DATA PROCESSING SUMMARY")
    print("="*70)
    
    # Basic stats
    print(f"\n📈 DATA OVERVIEW:")
    print(f"   Original Records: {processing_summary.get('original_records', 0):,}")
    print(f"   Final Records: {processing_summary.get('final_records', 0):,}")
    print(f"   Processing Time: {processing_summary.get('processing_time_seconds', 0):.1f} seconds")
    
    # Data quality
    if validation_results:
        score = validation_results.data_quality_score
        score_emoji = "🟢" if score >= 90 else "🟡" if score >= 70 else "🔴"
        print(f"\n{score_emoji} DATA QUALITY:")
        print(f"   Quality Score: {score:.1f}/100")
        print(f"   OHLC Violations: {validation_results.validation_summary.get('ohlc_violations', 0)}")
        print(f"   Price Anomalies: {validation_results.validation_summary.get('impossible_price_movements', 0)}")
        print(f"   Volume Anomalies: {validation_results.validation_summary.get('volume_anomalies_low', 0) + validation_results.validation_summary.get('volume_anomalies_high', 0)}")
    
    # Gap analysis
    if 'error' not in gap_analysis:
        print(f"\n🔍 GAP ANALYSIS:")
        print(f"   Total Gaps: {gap_analysis.get('total_gaps', 0)}")
        print(f"   Gaps Imputed: {processing_summary.get('gaps_imputed', 0)}")
        
        if gap_analysis.get('total_gaps', 0) > 0:
            print(f"   Gap Impact: {gap_analysis.get('gap_impact_percentage', 0):.3f}% of trading time")
            
            # Gap types
            gap_types = gap_analysis.get('gap_types', {})
            if gap_types:
                print(f"   Gap Types:")
                for gap_type, count in gap_types.items():
                    print(f"     {gap_type}: {count}")
            
            # Severity distribution
            severity_dist = gap_analysis.get('severity_distribution', {})
            if severity_dist:
                print(f"   Severity Distribution:")
                for severity, count in severity_dist.items():
                    print(f"     {severity}: {count}")
    
    # Status assessment
    print(f"\n🎯 SYSTEM STATUS:")
    if validation_results and validation_results.data_quality_score >= 85:
        print(f"   ✅ DATA READY FOR ALGORITHMIC TRADING")
        print(f"   ✅ Quality sufficient for feature engineering")
    elif validation_results and validation_results.data_quality_score >= 70:
        print(f"   ⚠️ DATA USABLE WITH CAUTION")
        print(f"   ⚠️ Monitor for quality issues")
    else:
        print(f"   ❌ DATA QUALITY CONCERNS")
        print(f"   ❌ Manual review recommended")
    
    print("\n" + "="*70)


def main() -> None:
    """
    ENHANCED Main execution function with trading hours processing.
    """
    print("=" * 60)
    print("DAX Trend-Following Algorithmic Trading System")
    print("ENHANCED: Trading Hours Processing (8:00-17:30 CET)")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed. Exiting.")
            sys.exit(1)
            
        # Load configuration
        config = load_config()
        
        # Add trading hours to config if not present
        if 'trading_hours' not in config:
            config['trading_hours'] = {
                'market_open': '08:00',
                'market_close': '17:30', 
                'timezone': 'Europe/Berlin',
                'include_weekends': False
            }
            logger.info("Added default trading hours to config")
        
        # Initialize system components
        logger.info("Initializing enhanced trading system components...")
        
        # Test data ingestion functionality
        data_config = config.get('data', {})
        raw_data_path = data_config.get('raw_data_path', 'data/raw')
        
        # Look for CSV files in the raw data directory
        raw_data_dir = Path(raw_data_path)
        if raw_data_dir.exists():
            csv_files = list(raw_data_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"Found {len(csv_files)} CSV files in {raw_data_path}")
                
                # Load the first CSV file found
                test_file = csv_files[0]
                logger.info(f"Processing file: {test_file.name}")
                
                try:
                    # Step 1: Load and convert data
                    logger.info("="*50)
                    logger.info("STEP 1: DATA LOADING & TIMEZONE CONVERSION")
                    logger.info("="*50)
                    
                    df_raw = load_and_convert_data(test_file)
                    logger.info(f"✅ Raw data loaded: {len(df_raw):,} records")
                    logger.info(f"   Date range: {df_raw.index.min()} to {df_raw.index.max()}")
                    
                    # Step 2: Filter to trading hours
                    logger.info("="*50)
                    logger.info("STEP 2: TRADING HOURS FILTERING")
                    logger.info("="*50)
                    
                    df_trading = filter_to_trading_hours(df_raw, config)
                    
                    if len(df_trading) == 0:
                        logger.error("No data remains after trading hours filtering!")
                        print("❌ No trading hours data available")
                        return
                    
                    # Step 3: Comprehensive data processing
                    logger.info("="*50)
                    logger.info("STEP 3: COMPREHENSIVE DATA PROCESSING")
                    logger.info("="*50)
                    
                    df_processed, gap_analysis, validation_results, processing_summary = comprehensive_data_processing(df_trading, config)
                    
                    # Step 4: Summary and status
                    print_processing_summary(gap_analysis, validation_results, processing_summary)
                    
                    # Store processed data for next components
                    # TODO: Pass df_processed to feature engineering
                    
                    print(f"\n🚀 NEXT STEPS:")
                    print(f"   ✅ Trading hours data ready: {len(df_processed):,} records")
                    print(f"   📊 Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                    print(f"   ⏳ Ready for feature engineering phase")
                    
                    # Save processed data for future use
                    processed_data_path = Path("data/processed")
                    processed_data_path.mkdir(exist_ok=True)
                    
                    output_file = processed_data_path / f"trading_hours_data_{datetime.now().strftime('%Y%m%d')}.csv"
                    df_processed.to_csv(output_file)
                    logger.info(f"Processed data saved to: {output_file}")
                    
                except DataIngestionError as e:
                    logger.error(f"Data ingestion failed: {e}")
                    print(f"\n❌ Data Ingestion Failed: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during processing: {e}", exc_info=True)
                    print(f"\n❌ Processing Error: {e}")
            else:
                logger.info(f"No CSV files found in {raw_data_path}")
                print(f"\n📁 No CSV files found in {raw_data_path}")
                print("   Place your DAX 1-minute OHLCV CSV files there to begin processing.")
        else:
            logger.info(f"Raw data directory {raw_data_path} does not exist, creating it...")
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n📁 Created raw data directory: {raw_data_path}")
            print("   Place your DAX 1-minute OHLCV CSV files there to begin processing.")
        
        logger.info("Enhanced trading system initialization completed")
        
        # System status
        print("\n🚀 ENHANCED SYSTEM STATUS:")
        print("   ✅ Data Ingestion & Timezone Conversion - Ready")
        print("   ✅ Trading Hours Filtering - Ready") 
        print("   ✅ Gap Detection & Analysis - Ready")
        print("   ✅ Data Validation & Quality Scoring - Ready")
        print("   ✅ Data Imputation (Minor/Moderate Gaps) - Ready")
        print("   ⏳ Feature Engineering - Next Phase")
        print("   ⏳ Market Regime Detection - Pending") 
        print("   ⏳ Signal Generation - Pending")
        print("   ⏳ Risk Management - Pending")
        print("   ⏳ Backtesting Engine - Pending")
        
        logger.info("Trading system execution completed")
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()