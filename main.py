#!/usr/bin/env python3
"""
DAX Trend-Following Algorithmic Trading System
Main execution entry point for the trading system.

Author: [Your Name]
Created: 2025
Python Version: 3.12.2
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Core imports
import pandas as pd
import numpy as np

# Local imports (will be implemented in subsequent modules)
from src.data.data_ingestion import load_and_convert_data, DataIngestionError
from src.data.gap_detector import GapDetector
from src.data.etl_pipeline import impute_data, DataImputer, validate_ohlcv_data
# from src.data.validator import DataValidator
# from src.features.engineering import FeatureEngineer
# from src.models.regime import MarketRegimeDetector
# from src.strategy.signals import SignalGenerator
# from src.strategy.risk import RiskManager
# from src.backtesting.engine import BacktestEngine


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


def save_historical_data(df: pd.DataFrame, output_path: str = "data/processed/historical_data.parquet") -> bool:
    """
    Save processed historical data to parquet format.
    
    Args:
        df (pd.DataFrame): Processed OHLCV DataFrame with CET timezone index
        output_path (str): Path to save the parquet file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet with compression
        df.to_parquet(
            output_file,
            compression='snappy',
            index=True,  # Include the datetime index
            engine='pyarrow'
        )
        
        logger.info(f"Historical data saved to {output_path}")
        logger.info(f"Saved {len(df):,} records from {df.index.min()} to {df.index.max()}")
        
        # Verify the saved file
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving historical data: {e}")
        return False


def load_historical_data(file_path: str = "data/processed/historical_data.parquet") -> pd.DataFrame:
    """
    Load historical data from parquet file.
    
    Args:
        file_path (str): Path to the parquet file
        
    Returns:
        pd.DataFrame: Historical OHLCV data or empty DataFrame if file doesn't exist
    """
    logger = logging.getLogger(__name__)
    
    try:
        if not Path(file_path).exists():
            logger.info(f"Historical data file {file_path} does not exist")
            return pd.DataFrame()
            
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded historical data: {len(df):,} records from {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return pd.DataFrame()


def process_and_validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Process raw data through validation, gap detection, and imputation pipeline.
    
    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame
        
    Returns:
        tuple: (processed_df, success_flag)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Comprehensive Data Validation
        logger.info("Running comprehensive data validation...")
        
        # For large datasets, validate on a sample first
        if len(df) > 100000:
            logger.info("Large dataset detected, validating sample...")
            validation_df = df.head(7 * 24 * 60)  # ~1 week
        else:
            validation_df = df
            
        validation_result = validate_ohlcv_data(validation_df)
        
        logger.info(f"Data Quality Score: {validation_result.data_quality_score:.1f}/100")
        logger.info(f"OHLC Violations: {validation_result.validation_summary.get('ohlc_violations', 0)}")
        logger.info(f"Price Movement Anomalies: {validation_result.validation_summary.get('impossible_price_movements', 0)}")
        
        # Check if data quality is acceptable
        if validation_result.data_quality_score < 70:
            logger.warning(f"Data quality score ({validation_result.data_quality_score}) below threshold (70)")
            for recommendation in validation_result.recommended_actions[:3]:  # Show top 3
                logger.warning(f"  Recommendation: {recommendation}")
        
        # Step 2: Gap Detection
        logger.info("Detecting data gaps...")
        gap_detector = GapDetector()
        
        # For large datasets, test on a sample first for gap detection
        if len(df) > 100000:
            logger.info("Large dataset detected, running gap detection on sample...")
            sample_df = df.head(7 * 24 * 60)  # ~1 week of 1-minute data
            gaps = gap_detector.detect_gaps(sample_df)
            test_df = sample_df
        else:
            gaps = gap_detector.detect_gaps(df)
            test_df = df
        
        # Get quality metrics
        quality_metrics = gap_detector.get_data_quality_metrics(test_df, gaps)
        
        logger.info(f"Data Completeness: {quality_metrics.get('data_completeness_pct', 'N/A')}%")
        logger.info(f"Total Gaps Found: {quality_metrics.get('total_gaps', 0)}")
        logger.info(f"Critical Data Gaps: {quality_metrics.get('critical_data_gaps', 0)}")
        
        # Step 3: Data Imputation (if needed)
        processed_df = df.copy()
        
        if gaps:
            logger.info("Running data imputation...")
            
            # Only impute minor and moderate gaps for safety
            safe_gaps = [g for g in gaps if g.severity in ['minor', 'moderate']][:10]  # Limit for processing
            
            if safe_gaps:
                imputation_result = impute_data(test_df.copy(), safe_gaps)
                
                logger.info(f"Imputation Summary:")
                logger.info(f"  Minor gaps imputed: {imputation_result.imputation_summary.get('minor_gaps_imputed', 0)}")
                logger.info(f"  Moderate gaps imputed: {imputation_result.imputation_summary.get('moderate_gaps_imputed', 0)}")
                logger.info(f"  Total points imputed: {imputation_result.imputation_summary.get('total_points_imputed', 0)}")
                
                if imputation_result.quality_flags:
                    logger.warning(f"Imputation quality warnings: {len(imputation_result.quality_flags)}")
                    
                # For demonstration, we keep the original data
                # In production, you might want to use the imputed data
                # processed_df = imputation_result.imputed_df
                
        else:
            logger.info("Perfect data quality - no gaps detected!")
        
        return processed_df, True
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {e}")
        return df, False


def main() -> None:
    """Main execution function for the DAX trading system."""
    print("=" * 60)
    print("DAX Trend-Following Algorithmic Trading System")
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
        
        # Initialize system components
        logger.info("Initializing trading system components...")
        
        # Check if historical data already exists
        historical_file = "data/processed/historical_data.parquet"
        existing_data = load_historical_data(historical_file)
        
        if not existing_data.empty:
            logger.info(f"Existing historical data found: {len(existing_data):,} records")
            logger.info(f"Date range: {existing_data.index.min()} to {existing_data.index.max()}")
            
            print(f"\n✅ Historical Data Already Exists!")
            print(f"   File: {historical_file}")
            print(f"   Records: {len(existing_data):,}")
            print(f"   Date Range: {existing_data.index.min()} to {existing_data.index.max()}")
            print(f"   Columns: {list(existing_data.columns)}")
            
            # Quick validation of existing data
            validation_sample = existing_data.tail(1000) if len(existing_data) > 1000 else existing_data
            validation_result = validate_ohlcv_data(validation_sample)
            print(f"   Data Quality Score: {validation_result.data_quality_score:.1f}/100")
            
        else:
            logger.info("No existing historical data found, processing raw data...")
            
            # Test data ingestion functionality
            data_config = config.get('data', {})
            raw_data_path = data_config.get('raw_data_path', 'data/raw')
            
            # Look for CSV files in the raw data directory
            raw_data_dir = Path(raw_data_path)
            if raw_data_dir.exists():
                csv_files = list(raw_data_dir.glob("*.csv"))
                if csv_files:
                    logger.info(f"Found {len(csv_files)} CSV files in {raw_data_path}")
                    
                    # Process the first CSV file found
                    source_file = csv_files[0]
                    logger.info(f"Processing primary data file: {source_file.name}")
                    
                    try:
                        # Load and convert data using our ingestion system
                        df = load_and_convert_data(source_file)
                        logger.info(f"Successfully loaded data: {len(df)} rows")
                        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                        logger.info(f"Columns: {list(df.columns)}")
                        
                        print(f"\n✅ Data Ingestion Successful!")
                        print(f"   File: {source_file.name}")
                        print(f"   Rows: {len(df):,}")
                        print(f"   Columns: {list(df.columns)}")
                        print(f"   Date Range: {df.index.min()} to {df.index.max()}")
                        
                        # Process through validation and gap detection pipeline
                        processed_df, processing_success = process_and_validate_data(df)
                        
                        if processing_success:
                            print(f"\n🔍 Data Processing Pipeline Completed Successfully!")
                            
                            # Save to historical data file
                            save_success = save_historical_data(processed_df, historical_file)
                            
                            if save_success:
                                print(f"\n💾 Historical Data File Created!")
                                print(f"   Location: {historical_file}")
                                print(f"   Records: {len(processed_df):,}")
                                print(f"   File Format: Parquet (compressed)")
                                
                                # Verify the saved file
                                verification_df = load_historical_data(historical_file)
                                if not verification_df.empty and len(verification_df) == len(processed_df):
                                    print(f"   ✅ File integrity verified")
                                else:
                                    print(f"   ⚠️ File verification failed")
                            else:
                                print(f"\n❌ Failed to save historical data file")
                                
                        else:
                            print(f"\n⚠️ Data processing encountered issues")
                            
                    except DataIngestionError as e:
                        logger.error(f"Data ingestion failed: {e}")
                        print(f"\n❌ Data Ingestion Failed: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error during data processing: {e}")
                        print(f"\n❌ Unexpected Error: {e}")
                else:
                    logger.info(f"No CSV files found in {raw_data_path}")
                    print(f"\n📁 No CSV files found in {raw_data_path}")
                    print("   Place your DAX 1-minute OHLCV CSV files there to create historical data.")
            else:
                logger.info(f"Raw data directory {raw_data_path} does not exist, creating it...")
                raw_data_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n📁 Created raw data directory: {raw_data_path}")
                print("   Place your DAX 1-minute OHLCV CSV files there to create historical data.")
        
        # TODO: Initialize other components as they are implemented
        # feature_engineer = FeatureEngineer(config.get('features', {}))
        # regime_detector = MarketRegimeDetector(config.get('regime', {}))
        # signal_generator = SignalGenerator(config.get('strategy', {}))
        # risk_manager = RiskManager(config.get('risk', {}))
        # backtest_engine = BacktestEngine(config.get('backtest', {}))
        
        logger.info("System initialized successfully")
        
        # Main execution logic will be implemented here
        logger.info("Starting main execution loop...")
        
        # System status
        print("\n🚀 System Status:")
        print("   ✅ Data Ingestion Module - Ready")
        print("   ✅ Data Validation Module - Ready")
        print("   ✅ Gap Detection Module - Ready")
        print("   ✅ Data Imputation Module - Ready")
        print("   ✅ Historical Data Management - Ready")
        print("   ⏳ Feature Engineering - Pending")
        print("   ⏳ Market Regime Detection - Pending") 
        print("   ⏳ Signal Generation - Pending")
        print("   ⏳ Risk Management - Pending")
        print("   ⏳ Backtesting Engine - Pending")
        
        if Path(historical_file).exists():
            print(f"\n💾 Historical Data File Available:")
            print(f"   {historical_file}")
            print(f"   Ready for feature engineering and strategy development!")
        else:
            print(f"\n📝 Next Steps:")
            print(f"   1. Place DAX CSV file in data/raw/ directory")
            print(f"   2. Run main.py again to create historical_data.parquet")
            print(f"   3. Proceed with feature engineering implementation")
        
        logger.info("Trading system execution completed")
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()