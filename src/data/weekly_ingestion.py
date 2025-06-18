#!/usr/bin/env python3
"""
DAX Trading System - Weekly Data Ingestion from TradingView

This module handles automated weekly data ingestion from TradingView CSV exports,
including data validation, deduplication, and seamless integration with historical data.

Dependencies:
    pip install tradingview-selenium helium selenium webdriver-manager

Author: DAX Trading System
Created: 2025-06-18
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import shutil
import yaml
from typing import Optional, Tuple, Dict, List
import sys

# For TradingView automation
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium dependencies not available. TradingView automation disabled.")

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
try:
    from src.data.data_ingestion import load_and_convert_data, DataIngestionError
    from src.data.etl_pipeline import validate_ohlcv_data, ValidationResult
    from src.data.gap_detector import GapDetector
except ImportError:
    logging.error("Local data modules not available. Ensure src/ directory is properly configured.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingViewDataDownloader:
    """
    Automated TradingView data downloader using Selenium.
    
    Requires TradingView Pro+ account for CSV export functionality.
    """
    
    def __init__(self, headless: bool = True, download_dir: Optional[str] = None):
        """
        Initialize TradingView downloader.
        
        Args:
            headless (bool): Run browser in headless mode
            download_dir (str): Directory for downloads (default: temp directory)
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium dependencies not available. Install with: pip install selenium webdriver-manager")
            
        self.headless = headless
        self.download_dir = Path(download_dir) if download_dir else Path("data/temp")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.driver = None
        
    def setup_driver(self) -> None:
        """Setup Chrome webdriver with appropriate options."""
        try:
            options = Options()
            
            if self.headless:
                options.add_argument("--headless")
                
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            
            # Set download directory
            prefs = {
                "download.default_directory": str(self.download_dir.absolute()),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True
            }
            options.add_experimental_option("prefs", prefs)
            
            # Install ChromeDriver automatically
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            
            logger.info("Chrome webdriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup webdriver: {e}")
            raise
    
    def login(self, username: str, password: str) -> bool:
        """
        Login to TradingView.
        
        Args:
            username (str): TradingView username
            password (str): TradingView password
            
        Returns:
            bool: True if login successful
        """
        try:
            if not self.driver:
                self.setup_driver()
                
            logger.info("Navigating to TradingView login page...")
            self.driver.get("https://www.tradingview.com/chart/")
            
            # Wait for and click sign in button
            wait = WebDriverWait(self.driver, 10)
            
            # Look for sign in button
            sign_in_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-name='header-user-menu-sign-in']"))
            )
            sign_in_button.click()
            
            # Enter credentials
            username_field = wait.until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            username_field.send_keys(username)
            
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(password)
            
            # Submit form
            submit_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_button.click()
            
            # Wait for login to complete
            time.sleep(5)
            
            # Check if login was successful
            if "chart" in self.driver.current_url:
                logger.info("Login successful")
                return True
            else:
                logger.error("Login failed")
                return False
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False
    
    def download_dax_data(self, timeframe: str = "1", days_back: int = 7) -> Optional[Path]:
        """
        Download DAX data from TradingView.
        
        Args:
            timeframe (str): Timeframe for data ("1" for 1-minute)
            days_back (int): How many days of data to download
            
        Returns:
            Optional[Path]: Path to downloaded CSV file
        """
        try:
            if not self.driver:
                raise RuntimeError("Driver not initialized. Call setup_driver() first.")
                
            logger.info(f"Downloading DAX {timeframe}-minute data for last {days_back} days...")
            
            # Navigate to DAX chart
            self.driver.get("https://www.tradingview.com/chart/?symbol=XETR%3ADAX")
            
            wait = WebDriverWait(self.driver, 10)
            
            # Set timeframe
            if timeframe != "1":
                timeframe_button = wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, f"[data-value='{timeframe}']"))
                )
                timeframe_button.click()
                time.sleep(2)
            
            # Load more historical data by scrolling left
            chart_area = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-name='legend-source-item']"))
            )
            
            # Scroll left to load more data
            for _ in range(days_back * 2):  # Scroll multiple times to ensure enough data
                chart_area.send_keys(Keys.ARROW_LEFT)
                time.sleep(0.1)
            
            # Open export menu
            menu_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-name='menu']"))
            )
            menu_button.click()
            
            # Click export data option
            export_option = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Export chart data')]"))
            )
            export_option.click()
            
            # Wait for download to complete
            time.sleep(10)
            
            # Find the downloaded file
            downloaded_files = list(self.download_dir.glob("*.csv"))
            if downloaded_files:
                # Get the most recent file
                latest_file = max(downloaded_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Downloaded file: {latest_file}")
                return latest_file
            else:
                logger.error("No CSV file found in download directory")
                return None
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    def cleanup(self) -> None:
        """Clean up webdriver resources."""
        if self.driver:
            self.driver.quit()
            logger.info("Webdriver cleaned up")


class WeeklyDataIngestion:
    """
    Main class for handling weekly data ingestion from TradingView exports.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize weekly data ingestion system.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.historical_file = Path("data/processed/historical_data.parquet")
        self.backup_dir = Path("data/processed/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Berlin timezone for consistency
        self.timezone = pytz.timezone('Europe/Berlin')
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            return {}
    
    def create_backup(self) -> bool:
        """
        Create backup of existing historical data.
        
        Returns:
            bool: True if backup successful or no existing data
        """
        try:
            if not self.historical_file.exists():
                logger.info("No existing historical data to backup")
                return True
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"historical_data_backup_{timestamp}.parquet"
            
            shutil.copy2(self.historical_file, backup_file)
            logger.info(f"Backup created: {backup_file}")
            
            # Keep only last 10 backups
            backups = sorted(self.backup_dir.glob("historical_data_backup_*.parquet"))
            if len(backups) > 10:
                for old_backup in backups[:-10]:
                    old_backup.unlink()
                    logger.info(f"Removed old backup: {old_backup}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def rollback_from_backup(self) -> bool:
        """
        Rollback to most recent backup.
        
        Returns:
            bool: True if rollback successful
        """
        try:
            backups = sorted(self.backup_dir.glob("historical_data_backup_*.parquet"))
            if not backups:
                logger.error("No backups available for rollback")
                return False
                
            latest_backup = backups[-1]
            shutil.copy2(latest_backup, self.historical_file)
            logger.info(f"Rolled back to backup: {latest_backup}")
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def parse_tradingview_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Parse TradingView CSV export format.
        
        TradingView exports typically have format: time,open,high,low,close,volume
        with timestamps in various formats.
        
        Args:
            csv_path (Path): Path to TradingView CSV file
            
        Returns:
            pd.DataFrame: Parsed OHLCV data with CET timezone
        """
        try:
            logger.info(f"Parsing TradingView CSV: {csv_path}")
            
            # Try different separators and encodings
            separators = [',', ';', '\t']
            encodings = ['utf-8', 'utf-8-sig', 'latin-1']
            
            df = None
            for encoding in encodings:
                for sep in separators:
                    try:
                        test_df = pd.read_csv(csv_path, sep=sep, encoding=encoding, nrows=5)
                        if len(test_df.columns) >= 5:  # Should have at least OHLCV columns
                            df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
                            logger.info(f"Successfully parsed with separator '{sep}' and encoding '{encoding}'")
                            break
                    except Exception:
                        continue
                if df is not None:
                    break
            
            if df is None:
                raise ValueError("Could not parse CSV with any combination of separators and encodings")
            
            # Clean column names (remove spaces, standardize case)
            df.columns = df.columns.str.strip().str.title()
            
            # Map common TradingView column variations
            column_mapping = {
                'Time': 'timestamp',
                'Datetime': 'timestamp', 
                'Date': 'timestamp',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                'Vol': 'Volume'
            }
            
            # Apply column mapping
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)
            
            # Ensure we have required columns
            required_cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                # Try to infer timestamp column
                if 'timestamp' in missing_cols and len(df.columns) >= 6:
                    # Assume first column is timestamp
                    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                    missing_cols.remove('timestamp')
                    
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert timestamp
            logger.info("Converting timestamps to CET...")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Handle timezone conversion
            if df['timestamp'].dt.tz is None:
                # Assume UTC if no timezone info
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                
            # Convert to Berlin timezone
            df['timestamp'] = df['timestamp'].dt.tz_convert(self.timezone)
            
            # Set as index
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Convert OHLCV to numeric
            ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in ohlcv_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values in OHLCV
            df.dropna(subset=ohlcv_cols, inplace=True)
            
            logger.info(f"Parsed {len(df)} records from {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing TradingView CSV: {e}")
            raise DataIngestionError(f"Failed to parse TradingView CSV: {e}")
    
    def load_existing_data(self) -> pd.DataFrame:
        """
        Load existing historical data.
        
        Returns:
            pd.DataFrame: Existing historical data or empty DataFrame
        """
        try:
            if not self.historical_file.exists():
                logger.info("No existing historical data found")
                return pd.DataFrame()
                
            df = pd.read_parquet(self.historical_file)
            logger.info(f"Loaded existing data: {len(df)} records from {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            return pd.DataFrame()
    
    def detect_overlaps_and_merge(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect overlaps between existing and new data, merge intelligently.
        
        Args:
            existing_df (pd.DataFrame): Existing historical data
            new_df (pd.DataFrame): New data from TradingView
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (merged_data, merge_stats)
        """
        try:
            if existing_df.empty:
                logger.info("No existing data, using new data as-is")
                return new_df, {'new_records': len(new_df), 'overlaps': 0, 'duplicates_removed': 0}
            
            # Find overlap period
            existing_start = existing_df.index.min()
            existing_end = existing_df.index.max()
            new_start = new_df.index.min()
            new_end = new_df.index.max()
            
            logger.info(f"Existing data: {existing_start} to {existing_end}")
            logger.info(f"New data: {new_start} to {new_end}")
            
            # Check for overlaps
            overlap_start = max(existing_start, new_start)
            overlap_end = min(existing_end, new_end)
            
            merge_stats = {
                'existing_records': len(existing_df),
                'new_records': len(new_df),
                'overlaps': 0,
                'duplicates_removed': 0,
                'final_records': 0
            }
            
            if overlap_start <= overlap_end:
                # There is overlap
                overlap_mask_existing = (existing_df.index >= overlap_start) & (existing_df.index <= overlap_end)
                overlap_mask_new = (new_df.index >= overlap_start) & (new_df.index <= overlap_end)
                
                overlapping_existing = existing_df[overlap_mask_existing]
                overlapping_new = new_df[overlap_mask_new]
                
                merge_stats['overlaps'] = len(overlapping_existing)
                
                logger.info(f"Found overlap period: {overlap_start} to {overlap_end}")
                logger.info(f"Overlapping records - Existing: {len(overlapping_existing)}, New: {len(overlapping_new)}")
                
                # Remove overlapping period from existing data (prioritize new data)
                existing_df_clean = existing_df[~overlap_mask_existing]
                
                # Combine non-overlapping existing data with all new data
                merged_df = pd.concat([existing_df_clean, new_df])
                
                merge_stats['duplicates_removed'] = len(overlapping_existing)
                
            else:
                # No overlap, simple concatenation
                logger.info("No overlap detected, concatenating data")
                merged_df = pd.concat([existing_df, new_df])
            
            # Sort by timestamp and remove any exact duplicates
            merged_df.sort_index(inplace=True)
            initial_len = len(merged_df)
            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
            final_len = len(merged_df)
            
            if initial_len != final_len:
                logger.info(f"Removed {initial_len - final_len} duplicate timestamps")
                merge_stats['duplicates_removed'] += (initial_len - final_len)
            
            merge_stats['final_records'] = final_len
            
            logger.info(f"Merge completed: {merge_stats}")
            
            return merged_df, merge_stats
            
        except Exception as e:
            logger.error(f"Error during merge: {e}")
            raise
    
    def validate_merged_data(self, df: pd.DataFrame) -> Tuple[bool, ValidationResult]:
        """
        Validate merged data using comprehensive validation.
        
        Args:
            df (pd.DataFrame): Merged OHLCV data
            
        Returns:
            Tuple[bool, ValidationResult]: (is_valid, validation_result)
        """
        try:
            logger.info("Validating merged data...")
            
            # Sample for validation if dataset is large
            if len(df) > 50000:
                # Validate recent data (last 2 weeks) and a random sample
                recent_data = df.tail(2 * 7 * 24 * 60)  # Last 2 weeks
                sample_size = min(10000, len(df) // 10)  # 10% sample, max 10k
                sample_data = df.sample(n=sample_size, random_state=42)
                validation_df = pd.concat([recent_data, sample_data]).drop_duplicates()
            else:
                validation_df = df
            
            validation_result = validate_ohlcv_data(validation_df)
            
            # Define validation criteria
            min_quality_score = 80  # Minimum acceptable quality score
            max_critical_issues = len(validation_df) * 0.001  # Max 0.1% critical issues
            
            is_valid = (
                validation_result.data_quality_score >= min_quality_score and
                validation_result.validation_summary.get('ohlc_violations', 0) <= max_critical_issues and
                validation_result.validation_summary.get('negative_prices', 0) == 0
            )
            
            logger.info(f"Validation result: Score={validation_result.data_quality_score:.1f}, Valid={is_valid}")
            
            if not is_valid:
                logger.warning("Data validation failed!")
                for recommendation in validation_result.recommended_actions[:3]:
                    logger.warning(f"  - {recommendation}")
            
            return is_valid, validation_result
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, None
    
    def save_updated_data(self, df: pd.DataFrame) -> bool:
        """
        Save updated historical data to parquet file.
        
        Args:
            df (pd.DataFrame): Updated historical data
            
        Returns:
            bool: True if successful
        """
        try:
            # Ensure directory exists
            self.historical_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with compression
            df.to_parquet(
                self.historical_file,
                compression='snappy',
                index=True,
                engine='pyarrow'
            )
            
            file_size_mb = self.historical_file.stat().st_size / (1024 * 1024)
            logger.info(f"Updated historical data saved: {len(df):,} records, {file_size_mb:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving updated data: {e}")
            return False
    
    def process_manual_csv(self, csv_path: str) -> bool:
        """
        Process manually provided TradingView CSV export.
        
        Args:
            csv_path (str): Path to TradingView CSV export
            
        Returns:
            bool: True if processing successful
        """
        try:
            csv_file = Path(csv_path)
            if not csv_file.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return False
            
            logger.info(f"Processing manual CSV: {csv_file}")
            
            # Step 1: Create backup
            if not self.create_backup():
                logger.error("Failed to create backup")
                return False
            
            # Step 2: Parse new data
            logger.info("Parsing TradingView CSV...")
            new_data = self.parse_tradingview_csv(csv_file)
            
            if new_data.empty:
                logger.error("No valid data found in CSV")
                return False
            
            # Step 3: Load existing data
            existing_data = self.load_existing_data()
            
            # Step 4: Merge data
            logger.info("Merging with existing data...")
            merged_data, merge_stats = self.detect_overlaps_and_merge(existing_data, new_data)
            
            # Step 5: Validate merged data
            is_valid, validation_result = self.validate_merged_data(merged_data)
            
            if not is_valid:
                logger.error("Merged data failed validation")
                if input("Continue anyway? (y/N): ").lower() != 'y':
                    logger.info("Processing aborted by user")
                    return False
            
            # Step 6: Save updated data
            if self.save_updated_data(merged_data):
                logger.info("Weekly ingestion completed successfully!")
                
                # Print summary
                print("\n" + "="*60)
                print("Weekly Data Ingestion Summary")
                print("="*60)
                print(f"Source File: {csv_file}")
                print(f"Processing Time: {datetime.now()}")
                print(f"\nMerge Statistics:")
                for key, value in merge_stats.items():
                    print(f"  {key}: {value:,}")
                    
                if validation_result:
                    print(f"\nData Quality:")
                    print(f"  Quality Score: {validation_result.data_quality_score:.1f}/100")
                    print(f"  Total Records: {validation_result.total_records:,}")
                    
                print(f"\nFinal Dataset:")
                print(f"  Records: {len(merged_data):,}")
                print(f"  Date Range: {merged_data.index.min()} to {merged_data.index.max()}")
                print(f"  File: {self.historical_file}")
                
                return True
            else:
                logger.error("Failed to save updated data, rolling back...")
                self.rollback_from_backup()
                return False
                
        except Exception as e:
            logger.error(f"Error in manual CSV processing: {e}")
            logger.info("Rolling back to previous version...")
            self.rollback_from_backup()
            return False
    
    def automated_download_and_process(self, username: str, password: str, days_back: int = 7) -> bool:
        """
        Automated download and processing of TradingView data.
        
        Args:
            username (str): TradingView username
            password (str): TradingView password
            days_back (int): Days of data to download
            
        Returns:
            bool: True if successful
        """
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available for automated download")
            return False
            
        downloader = None
        try:
            logger.info("Starting automated TradingView download...")
            
            # Step 1: Create backup
            if not self.create_backup():
                logger.error("Failed to create backup")
                return False
            
            # Step 2: Download data
            downloader = TradingViewDataDownloader(headless=True)
            
            if not downloader.login(username, password):
                logger.error("TradingView login failed")
                return False
            
            downloaded_file = downloader.download_dax_data(timeframe="1", days_back=days_back)
            
            if not downloaded_file:
                logger.error("Download failed")
                return False
            
            # Step 3: Process downloaded file
            success = self.process_manual_csv(str(downloaded_file))
            
            # Clean up downloaded file
            if downloaded_file.exists():
                downloaded_file.unlink()
                logger.info("Cleaned up temporary download file")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in automated download: {e}")
            return False
        finally:
            if downloader:
                downloader.cleanup()


def setup_cron_job_example():
    """
    Example of how to set up a cron job for weekly data ingestion.
    
    Add this to your crontab (crontab -e):
    
    # Run every Monday at 6 AM
    0 6 * * 1 cd /path/to/dax-trading-system && python weekly_ingestion.py --auto
    
    # Or run every Sunday at 11 PM
    0 23 * * 0 cd /path/to/dax-trading-system && python weekly_ingestion.py --auto
    """
    print("="*60)
    print("Cron Job Setup Instructions")
    print("="*60)
    print("To schedule weekly data ingestion, add one of these lines to your crontab:")
    print()
    print("# Run every Monday at 6:00 AM")
    print("0 6 * * 1 cd /path/to/dax-trading-system && python weekly_ingestion.py --auto")
    print()
    print("# Run every Sunday at 11:00 PM") 
    print("0 23 * * 0 cd /path/to/dax-trading-system && python weekly_ingestion.py --auto")
    print()
    print("To edit your crontab:")
    print("  crontab -e")
    print()
    print("To view your current crontab:")
    print("  crontab -l")
    print()
    print("Note: Ensure TradingView credentials are stored securely!")


def main():
    """Main CLI interface for weekly data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DAX Trading System - Weekly Data Ingestion")
    parser.add_argument("--csv", type=str, help="Path to TradingView CSV export file")
    parser.add_argument("--auto", action="store_true", help="Automated download from TradingView")
    parser.add_argument("--username", type=str, help="TradingView username (for auto mode)")
    parser.add_argument("--password", type=str, help="TradingView password (for auto mode)")
    parser.add_argument("--days", type=int, default=7, help="Days of data to download (default: 7)")
    parser.add_argument("--setup-cron", action="store_true", help="Show cron job setup instructions")
    
    args = parser.parse_args()
    
    if args.setup_cron:
        setup_cron_job_example()
        return
    
    ingestion = WeeklyDataIngestion()
    
    if args.csv:
        # Manual CSV processing
        success = ingestion.process_manual_csv(args.csv)
        if success:
            print("✅ Manual CSV processing completed successfully!")
        else:
            print("❌ Manual CSV processing failed!")
            sys.exit(1)
            
    elif args.auto:
        # Automated download and processing
        username = args.username or input("TradingView Username: ")
        password = args.password or input("TradingView Password: ")
        
        success = ingestion.automated_download_and_process(username, password, args.days)
        if success:
            print("✅ Automated data ingestion completed successfully!")
        else:
            print("❌ Automated data ingestion failed!")
            sys.exit(1)
    else:
        # Show usage
        parser.print_help()
        print("\nExamples:")
        print("  python weekly_ingestion.py --csv data/raw/tradingview_export.csv")
        print("  python weekly_ingestion.py --auto --username your_username --days 7")
        print("  python weekly_ingestion.py --setup-cron")


if __name__ == "__main__":
    main()