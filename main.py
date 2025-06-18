#!/usr/bin/env python3
"""
DAX Trend-Following Algorithmic Trading System
Main execution entry point for the trading system.

Author: [Your Name]
Created: 2025
Python Version: 3.9+
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
        
        # Test data ingestion functionality
        data_config = config.get('data', {})
        raw_data_path = data_config.get('raw_data_path', 'data/raw')
        
        # Look for CSV files in the raw data directory
        raw_data_dir = Path(raw_data_path)
        if raw_data_dir.exists():
            csv_files = list(raw_data_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"Found {len(csv_files)} CSV files in {raw_data_path}")
                
                # Test loading the first CSV file found
                test_file = csv_files[0]
                logger.info(f"Testing data ingestion with: {test_file.name}")
                
                try:
                    # Load and convert data using our new function
                    df = load_and_convert_data(test_file)
                    logger.info(f"Successfully loaded data: {len(df)} rows")
                    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                    logger.info(f"Columns: {list(df.columns)}")
                    
                    print(f"\n✅ Data Ingestion Test Successful!")
                    print(f"   File: {test_file.name}")
                    print(f"   Rows: {len(df):,}")
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Date Range: {df.index.min()} to {df.index.max()}")
                    
                except DataIngestionError as e:
                    logger.error(f"Data ingestion failed: {e}")
                    print(f"\n❌ Data Ingestion Test Failed: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during data ingestion: {e}")
                    print(f"\n❌ Unexpected Error: {e}")
            else:
                logger.info(f"No CSV files found in {raw_data_path}")
                print(f"\n📁 No CSV files found in {raw_data_path}")
                print("   Place your DAX 1-minute OHLCV CSV files there to test data ingestion.")
        else:
            logger.info(f"Raw data directory {raw_data_path} does not exist, creating it...")
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n📁 Created raw data directory: {raw_data_path}")
            print("   Place your DAX 1-minute OHLCV CSV files there to test data ingestion.")
        
        # TODO: Initialize other components as they are implemented
        # feature_engineer = FeatureEngineer(config.get('features', {}))
        # regime_detector = MarketRegimeDetector(config.get('regime', {}))
        # signal_generator = SignalGenerator(config.get('strategy', {}))
        # risk_manager = RiskManager(config.get('risk', {}))
        # backtest_engine = BacktestEngine(config.get('backtest', {}))
        
        logger.info("System initialized successfully")
        
        # Main execution logic will be implemented here
        logger.info("Starting main execution loop...")
        
        # System is now ready with data ingestion capability
        print("\n🚀 System Status:")
        print("   ✅ Data Ingestion Module - Ready")
        print("   ⏳ Feature Engineering - Pending")
        print("   ⏳ Market Regime Detection - Pending") 
        print("   ⏳ Signal Generation - Pending")
        print("   ⏳ Risk Management - Pending")
        print("   ⏳ Backtesting Engine - Pending")
        print("\nNext: Implement data validation and gap detection modules.")
        
        logger.info("Trading system execution completed")
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()