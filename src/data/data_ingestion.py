"""
Data Ingestion Module

This module handles the ingestion of raw DAX 1-minute OHLCV data from CSV files,
with timezone conversion from Chicago time to CET, including proper DST handling.

Author: ady7ady7
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List
import pytz
from datetime import datetime
import warnings

# Configure logging
logger = logging.getLogger(__name__)


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


def load_and_convert_data(
    file_path: Union[str, Path],
    timestamp_col: str = "timestamp",
    expected_columns: Optional[List[str]] = None,
    validate_ohlc: bool = True
) -> pd.DataFrame:
    """
    Load DAX 1-minute OHLCV data from CSV and convert timezone from Chicago to CET.
    
    This function performs robust data loading with timezone conversion, handling
    Daylight Saving Time transitions for both Chicago (CDT/CST) and Berlin (CEST/CET).
    
    Args:
        file_path (Union[str, Path]): Path to the CSV file containing OHLCV data
        timestamp_col (str): Name of the timestamp column (default: "timestamp")
        expected_columns (Optional[List[str]]): Expected column names for validation
        validate_ohlc (bool): Whether to validate OHLC relationships (default: True)
        
    Returns:
        pd.DataFrame: DataFrame with CET timestamps as index and OHLCV columns
        
    Raises:
        DataIngestionError: If file not found, invalid format, or data validation fails
        
    Example:
        >>> df = load_and_convert_data("data/raw/DAX_1min_2024.csv")
        >>> print(df.head())
        >>> print(f"Data shape: {df.shape}")
        >>> print(f"Date range: {df.index.min()} to {df.index.max()}")
    """
    
    # Default expected columns for OHLCV data
    if expected_columns is None:
        expected_columns = [timestamp_col, "Open", "High", "Low", "Close", "Volume"]
    
    logger.info(f"Loading data from: {file_path}")
    
    try:
        # Convert to Path object for robust file handling
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise DataIngestionError(f"File not found: {file_path}")
            
        # Check file size (warn if too large)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:  # Warn for files larger than 500MB
            logger.warning(f"Large file detected: {file_size_mb:.1f}MB")
            
        # Read CSV file
        logger.info("Reading CSV file...")
        df = pd.read_csv(file_path)
        
        # Validate columns exist
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            available_cols = list(df.columns)
            raise DataIngestionError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {available_cols}"
            )
            
        # Log initial data info
        logger.info(f"Raw data loaded: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Date range (raw): {df[timestamp_col].iloc[0]} to {df[timestamp_col].iloc[-1]}")
        
        # Convert timestamp column to datetime
        logger.info("Converting timestamps...")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Set timezone to Chicago (source timezone)
        chicago_tz = pytz.timezone('America/Chicago')
        berlin_tz = pytz.timezone('Europe/Berlin')
        
        # Localize to Chicago timezone (handle naive timestamps)
        if df[timestamp_col].dt.tz is None:
            logger.info("Localizing naive timestamps to Chicago timezone...")
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(chicago_tz, ambiguous='infer')
        else:
            logger.info("Converting aware timestamps to Chicago timezone...")
            df[timestamp_col] = df[timestamp_col].dt.tz_convert(chicago_tz)
            
        # Convert to Berlin timezone (CET/CEST)
        logger.info("Converting to Berlin timezone (CET/CEST)...")
        df[timestamp_col] = df[timestamp_col].dt.tz_convert(berlin_tz)
        
        # Set timestamp as index
        df.set_index(timestamp_col, inplace=True)
        df.index.name = 'timestamp_cet'
        
        # Sort by timestamp to ensure chronological order
        df.sort_index(inplace=True)
        
        # Validate OHLC relationships if requested
        if validate_ohlc and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            logger.info("Validating OHLC relationships...")
            validation_results = _validate_ohlc_data(df)
            if validation_results['errors'] > 0:
                logger.warning(f"OHLC validation found {validation_results['errors']} errors")
                if validation_results['errors'] > len(df) * 0.01:  # More than 1% errors
                    logger.error("Too many OHLC validation errors - data quality issue")
                    
        # Remove any duplicate timestamps
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} duplicate timestamps")
            
        # Log final data info
        logger.info(f"Final data: {len(df)} rows")
        logger.info(f"Date range (CET): {df.index.min()} to {df.index.max()}")
        logger.info(f"Timezone: {df.index.tz}")
        
        # Check for reasonable data range (warn if data is too old or too recent)
        _validate_date_range(df.index)
        
        return df
        
    except pd.errors.EmptyDataError:
        raise DataIngestionError(f"Empty CSV file: {file_path}")
    except pd.errors.ParserError as e:
        raise DataIngestionError(f"CSV parsing error: {e}")
    except pytz.exceptions.AmbiguousTimeError as e:
        raise DataIngestionError(f"Ambiguous timezone conversion: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise DataIngestionError(f"Failed to load data from {file_path}: {e}")


def _validate_ohlc_data(df: pd.DataFrame) -> dict:
    """
    Validate OHLC data relationships.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC columns
        
    Returns:
        dict: Validation results with error counts and details
    """
    results = {
        'errors': 0,
        'warnings': 0,
        'details': []
    }
    
    try:
        # Check High >= max(Open, Close) and High >= Low
        high_errors = (df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)).sum()
        if high_errors > 0:
            results['errors'] += high_errors
            results['details'].append(f"High price violations: {high_errors}")
            
        # Check Low <= min(Open, Close) and Low <= High
        low_errors = (df['Low'] > df[['Open', 'Close', 'High']].min(axis=1)).sum()
        if low_errors > 0:
            results['errors'] += low_errors
            results['details'].append(f"Low price violations: {low_errors}")
            
        # Check for negative prices
        negative_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1).sum()
        if negative_prices > 0:
            results['errors'] += negative_prices
            results['details'].append(f"Negative/zero prices: {negative_prices}")
            
        # Check for unrealistic price movements (>20% in 1 minute)
        if len(df) > 1:
            price_changes = df['Close'].pct_change().abs()
            extreme_moves = (price_changes > 0.20).sum()
            if extreme_moves > 0:
                results['warnings'] += extreme_moves
                results['details'].append(f"Extreme price movements (>20%): {extreme_moves}")
                
        # Check volume
        if 'Volume' in df.columns:
            negative_volume = (df['Volume'] < 0).sum()
            if negative_volume > 0:
                results['errors'] += negative_volume
                results['details'].append(f"Negative volume: {negative_volume}")
                
    except Exception as e:
        logger.warning(f"OHLC validation failed: {e}")
        results['details'].append(f"Validation error: {e}")
        
    return results


def _validate_date_range(index: pd.DatetimeIndex) -> None:
    """
    Validate that the date range is reasonable for financial data.
    
    Args:
        index (pd.DatetimeIndex): Datetime index to validate
    """
    current_date = datetime.now(pytz.timezone('Europe/Berlin'))
    min_date = index.min()
    max_date = index.max()
    
    # Warn if data is too old (>5 years)
    if (current_date - min_date).days > 5 * 365:
        logger.warning(f"Data contains very old records: {min_date}")
        
    # Warn if data is from the future
    if max_date > current_date:
        logger.warning(f"Data contains future timestamps: {max_date}")
        
    # Check for reasonable market hours (DAX trades 9:00-17:30 CET)
    berlin_tz = pytz.timezone('Europe/Berlin')
    market_hours = index.tz_convert(berlin_tz).hour
    unusual_hours = ((market_hours < 8) | (market_hours > 18)).sum()
    
    if unusual_hours > len(index) * 0.1:  # More than 10% outside normal hours
        logger.info(f"Note: {unusual_hours} records outside typical market hours (8-18 CET)")


# Example usage and testing functions
def main():
    """Example usage of the data ingestion module."""
    
    # Example file path (adjust as needed)
    example_file = "data/raw/DAX_sample.csv"
    
    try:
        # Load and convert data
        df = load_and_convert_data(example_file)
        
        print("\n" + "="*60)
        print("DAX Data Ingestion - Example Results")
        print("="*60)
        
        print(f"\nData Shape: {df.shape}")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        print(f"Timezone: {df.index.tz}")
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nData Info:")
        print(df.info())
        
        print("\nBasic Statistics:")
        print(df.describe())
        
    except DataIngestionError as e:
        print(f"Data ingestion error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()




    # Basic usage
#df = load_and_convert_data("data/raw/DAX_1min_2024.csv")

# Advanced usage with custom validation
'''df = load_and_convert_data(
    file_path="data/raw/DAX_data.csv",
    timestamp_col="datetime",
    expected_columns=["datetime", "Open", "High", "Low", "Close", "Volume"],
    validate_ohlc=True
)'''