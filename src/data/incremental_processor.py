#!/usr/bin/env python3
"""
Incremental Data Processing System for DAX Trading System

This module efficiently manages processed data by checking for existing files
and only processing new/missing data when necessary.

File Location: src/data/incremental_processor.py

Author: DAX Trading System
Created: 2025-06-19
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
from typing import Optional, Tuple, Dict
import pytz

logger = logging.getLogger(__name__)


class IncrementalDataProcessor:
    """
    Intelligent data processor that avoids reprocessing unchanged data.
    
    Features:
    - Detects existing processed data files
    - Compares source file metadata (size, modification time, hash)
    - Incrementally adds new data if source is updated
    - Validates data continuity and quality
    - Saves processing time and storage space
    """
    
    def __init__(self, config: dict):
        """Initialize the incremental data processor."""
        self.config = config
        self.raw_data_path = Path(config.get('data', {}).get('raw_data_path', 'data/raw'))
        self.processed_data_path = Path(config.get('data', {}).get('processed_data_path', 'data/processed'))
        self.processed_data_path.mkdir(exist_ok=True)
        
        # Metadata file to track processing history
        self.metadata_file = self.processed_data_path / "processing_metadata.json"
        
        self.timezone = pytz.timezone(config.get('trading_hours', {}).get('timezone', 'Europe/Berlin'))
        
        logger.info("IncrementalDataProcessor initialized")
    
    def get_source_file_info(self, file_path: Path) -> Dict:
        """Get comprehensive information about the source file."""
        try:
            stat = file_path.stat()
            
            # Calculate file hash for change detection (sample-based for large files)
            file_hash = self._calculate_file_hash(file_path)
            
            return {
                'file_path': str(file_path),
                'file_size': stat.st_size,
                'modification_time': stat.st_mtime,
                'file_hash': file_hash,
                'last_checked': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate a hash of the file for change detection."""
        try:
            hasher = hashlib.md5()
            
            # For large files, sample first/middle/last chunks
            file_size = file_path.stat().st_size
            
            with open(file_path, 'rb') as f:
                # First 64KB
                hasher.update(f.read(65536))
                
                # Middle chunk (if file is large enough)
                if file_size > 200000:  # > 200KB
                    f.seek(file_size // 2)
                    hasher.update(f.read(65536))
                
                # Last chunk (if file is large enough)
                if file_size > 400000:  # > 400KB
                    f.seek(-65536, 2)  # Seek to 64KB from end
                    hasher.update(f.read(65536))
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return "unknown"
    
    def load_processing_metadata(self) -> Dict:
        """Load existing processing metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded processing metadata from {self.metadata_file}")
                return metadata
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        
        return {}
    
    def save_processing_metadata(self, metadata: Dict):
        """Save processing metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved processing metadata to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def find_latest_processed_file(self) -> Optional[Path]:
        """Find the most recent processed trading hours data file."""
        try:
            pattern = "trading_hours_data_*.csv"
            processed_files = list(self.processed_data_path.glob(pattern))
            
            if not processed_files:
                logger.info("No existing processed data files found")
                return None
            
            # Sort by modification time, get latest
            latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found latest processed file: {latest_file.name}")
            
            return latest_file
            
        except Exception as e:
            logger.error(f"Error finding processed files: {e}")
            return None
    
    def analyze_processed_data(self, processed_file: Path) -> Dict:
        """Analyze existing processed data to understand its scope and quality."""
        try:
            logger.info(f"Analyzing processed data: {processed_file.name}")
            
            # Get file stats first
            file_size = processed_file.stat().st_size
            
            # Count lines efficiently
            with open(processed_file, 'r') as f:
                record_count = sum(1 for line in f) - 1  # Subtract header
            
            # Load sample to check structure
            sample_df = pd.read_csv(processed_file, nrows=100, index_col=0, parse_dates=True)
            
            # Get date range efficiently
            first_date = sample_df.index[0]
            
            # Read last few lines to get end date
            with open(processed_file, 'rb') as f:
                f.seek(-2048, 2)  # Go to near end of file
                last_lines = f.read().decode().strip().split('\n')[-5:]
            
            # Parse the last valid timestamp
            end_date = first_date  # fallback
            for line in reversed(last_lines):
                try:
                    if ',' in line and not line.startswith('#'):
                        timestamp_str = line.split(',')[0]
                        end_date = pd.to_datetime(timestamp_str)
                        break
                except:
                    continue
            
            analysis = {
                'file_path': str(processed_file),
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'record_count': record_count,
                'start_date': first_date.isoformat(),
                'end_date': end_date.isoformat(),
                'columns': list(sample_df.columns),
                'date_range_days': (end_date - first_date).days,
                'timezone': str(sample_df.index.tz) if sample_df.index.tz else 'Unknown'
            }
            
            logger.info(f"Processed data analysis:")
            logger.info(f"  Records: {record_count:,}")
            logger.info(f"  Date range: {first_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            logger.info(f"  File size: {analysis['file_size_mb']} MB")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing processed data: {e}")
            return {}
    
    def check_data_freshness(self, source_file: Path, processed_analysis: Dict, 
                           source_metadata: Dict) -> Dict:
        """Check if processed data needs updating."""
        
        decision = {
            'action': 'unknown',
            'reason': '',
            'confidence': 0.0,
            'recommendations': []
        }
        
        try:
            # Check 1: Source file modification vs processed file
            processed_file = Path(processed_analysis['file_path'])
            source_mtime = source_file.stat().st_mtime
            processed_mtime = processed_file.stat().st_mtime
            
            if source_mtime > processed_mtime:
                decision['action'] = 'reprocess'
                decision['reason'] = 'Source file is newer than processed file'
                decision['confidence'] = 0.9
                return decision
            
            # Check 2: Compare with previous metadata
            metadata = self.load_processing_metadata()
            previous_source_info = metadata.get('last_source_file', {})
            
            if previous_source_info:
                # File size changed significantly
                prev_size = previous_source_info.get('file_size', 0)
                current_size = source_metadata['file_size']
                
                if prev_size > 0:
                    size_change_pct = abs(current_size - prev_size) / prev_size * 100
                    
                    if size_change_pct > 5:  # More than 5% change
                        decision['action'] = 'reprocess'
                        decision['reason'] = f'Source file size changed by {size_change_pct:.1f}%'
                        decision['confidence'] = 0.8
                        return decision
                
                # Hash changed
                if (previous_source_info.get('file_hash') != source_metadata.get('file_hash') and 
                    previous_source_info.get('file_hash') != 'unknown' and
                    source_metadata.get('file_hash') != 'unknown'):
                    decision['action'] = 'reprocess'
                    decision['reason'] = 'Source file content has changed'
                    decision['confidence'] = 0.9
                    return decision
            
            # Check 3: Data recency
            end_date = pd.to_datetime(processed_analysis['end_date'])
            if end_date.tz is None:
                end_date = end_date.tz_localize(self.timezone)
            
            days_old = (datetime.now(self.timezone) - end_date.tz_convert(self.timezone)).days
            
            if days_old > 7:  # More than a week old
                decision['action'] = 'update'
                decision['reason'] = f'Processed data is {days_old} days old'
                decision['confidence'] = 0.7
                decision['recommendations'].append('Consider updating for recent data')
                return decision
            
            # Check 4: File size seems too small
            expected_min_size = 50  # MB
            if processed_analysis.get('file_size_mb', 0) < expected_min_size:
                decision['action'] = 'reprocess'
                decision['reason'] = f'Processed file seems too small ({processed_analysis.get("file_size_mb", 0)} MB)'
                decision['confidence'] = 0.6
                return decision
            
            # All checks passed - reuse existing data
            decision['action'] = 'reuse'
            decision['reason'] = 'Processed data appears current and valid'
            decision['confidence'] = 0.8
            decision['recommendations'].append('Existing processed data is suitable for use')
            
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            decision['action'] = 'reprocess'
            decision['reason'] = f'Error during freshness check: {e}'
            decision['confidence'] = 0.5
        
        return decision
    
    def load_processed_data(self, processed_file: Path) -> pd.DataFrame:
        """Load processed data efficiently."""
        try:
            logger.info(f"Loading processed data from {processed_file.name}...")
            
            # Load with proper datetime parsing and timezone handling
            df = pd.read_csv(
                processed_file,
                index_col=0,
                parse_dates=True
            )
            
            # Ensure timezone is set correctly
            if df.index.tz is None:
                df.index = df.index.tz_localize(self.timezone)
            elif df.index.tz != self.timezone:
                df.index = df.index.tz_convert(self.timezone)
            
            logger.info(f"✅ Loaded {len(df):,} records from processed data")
            logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def smart_data_processing_decision(self, source_file: Path) -> Tuple[str, pd.DataFrame, Dict]:
        """
        Make intelligent decision about data processing.
        
        Returns:
            action: 'reuse', 'update', or 'reprocess'
            data: DataFrame with trading hours data (None if reprocess needed)
            metadata: Processing metadata
        """
        
        logger.info("="*60)
        logger.info("🧠 SMART DATA PROCESSING DECISION")
        logger.info("="*60)
        
        # Get source file information
        source_info = self.get_source_file_info(source_file)
        logger.info(f"Source file: {source_file.name} ({source_info.get('file_size', 0) / 1024 / 1024:.1f} MB)")
        
        # Look for existing processed data
        processed_file = self.find_latest_processed_file()
        
        if processed_file is None:
            logger.info("🔄 No processed data found - full processing required")
            return 'reprocess', None, {'reason': 'No existing processed data'}
        
        # Analyze existing processed data
        processed_analysis = self.analyze_processed_data(processed_file)
        
        if not processed_analysis:
            logger.warning("⚠️ Could not analyze processed data - reprocessing")
            return 'reprocess', None, {'reason': 'Could not analyze existing data'}
        
        # Check if we need to update
        freshness_check = self.check_data_freshness(source_file, processed_analysis, source_info)
        
        action = freshness_check['action']
        reason = freshness_check['reason']
        confidence = freshness_check['confidence']
        
        logger.info(f"📊 Processing Decision:")
        logger.info(f"   Action: {action.upper()}")
        logger.info(f"   Reason: {reason}")
        logger.info(f"   Confidence: {confidence:.1%}")
        
        if freshness_check.get('recommendations'):
            for rec in freshness_check['recommendations']:
                logger.info(f"   💡 {rec}")
        
        # Execute decision
        if action == 'reuse':
            logger.info("✅ Loading existing processed data...")
            df = self.load_processed_data(processed_file)
            
            # Update metadata
            metadata = self.load_processing_metadata()
            metadata['last_reuse'] = {
                'timestamp': datetime.now().isoformat(),
                'processed_file': str(processed_file),
                'record_count': len(df),
                'decision_reason': reason
            }
            self.save_processing_metadata(metadata)
            
            return 'reuse', df, metadata
        
        else:
            logger.info(f"🔄 {action.title()} required - will process data from source")
            return action, None, {'decision': freshness_check}
    
    def update_metadata_after_processing(self, source_file: Path, processed_file: Path, record_count: int):
        """Update metadata after successful processing."""
        try:
            metadata = self.load_processing_metadata()
            
            # Update processing history
            metadata['last_processing'] = {
                'timestamp': datetime.now().isoformat(),
                'source_file': self.get_source_file_info(source_file),
                'processed_file': str(processed_file),
                'processed_records': record_count,
                'action_taken': 'full_processing'
            }
            
            # Update source file tracking
            metadata['last_source_file'] = self.get_source_file_info(source_file)
            
            self.save_processing_metadata(metadata)
            logger.info("✅ Metadata updated after processing")
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")