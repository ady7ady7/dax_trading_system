"""
DAX Trading System - ETL Pipeline with Data Imputation

This module provides sophisticated data imputation strategies for financial time series,
handling different gap severities with OHLC-specific methods.

File Location: src/data/etl_pipeline.py

Author: DAX Trading System
Created: 2025-06-18
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
from scipy import stats

# Import our gap detection module
try:
    from .gap_detector import Gap, GapDetector
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.data.gap_detector import Gap, GapDetector

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from comprehensive OHLCV data validation."""
    total_records: int
    validation_summary: Dict[str, int]
    anomaly_details: Dict[str, List[Dict]]
    data_quality_score: float
    recommended_actions: List[str]


@dataclass
class ImputationResult:
    """Results from data imputation process."""
    imputed_df: pd.DataFrame
    imputation_summary: Dict[str, int]
    excluded_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    quality_flags: List[str]


class DataImputer:
    """
    Sophisticated data imputation for financial OHLCV time series.
    
    Handles different gap severities with appropriate strategies:
    - Minor gaps: OHLC-specific interpolation
    - Moderate gaps: Time-weighted interpolation with ATR noise
    - Major gaps: Flagged for exclusion
    """
    
    def __init__(self, atr_period: int = 14, noise_factor: float = 0.1):
        """
        Initialize the data imputer.
        
        Args:
            atr_period (int): Period for Average True Range calculation
            noise_factor (float): Factor for adding realistic noise (0.1 = 10% of ATR)
        """
        self.atr_period = atr_period
        self.noise_factor = noise_factor
        logger.info(f"DataImputer initialized with ATR period: {atr_period}")
    
    def impute_data(self, df: pd.DataFrame, gaps: List[Gap]) -> ImputationResult:
        """
        Apply sophisticated imputation strategies based on gap severity.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with datetime index
            gaps (List[Gap]): List of detected gaps from GapDetector
            
        Returns:
            ImputationResult: Imputed data with summary and excluded periods
        """
        logger.info(f"Starting imputation for {len(gaps)} gaps")
        
        # Create a copy to avoid modifying original data
        imputed_df = df.copy()
        
        # Initialize tracking variables
        imputation_summary = {
            'minor_gaps_imputed': 0,
            'moderate_gaps_imputed': 0,
            'major_gaps_flagged': 0,
            'total_points_imputed': 0
        }
        excluded_periods = []
        quality_flags = []
        
        # Calculate ATR for noise generation
        atr_series = self._calculate_atr(df)
        
        # Process gaps by severity
        for gap in gaps:
            try:
                if gap.severity == 'minor':
                    success = self._impute_minor_gap(imputed_df, gap, atr_series)
                    if success:
                        imputation_summary['minor_gaps_imputed'] += 1
                        imputation_summary['total_points_imputed'] += gap.duration_minutes
                        
                elif gap.severity == 'moderate':
                    success = self._impute_moderate_gap(imputed_df, gap, atr_series)
                    if success:
                        imputation_summary['moderate_gaps_imputed'] += 1
                        imputation_summary['total_points_imputed'] += gap.duration_minutes
                        
                else:  # major gaps
                    excluded_periods.append((gap.start_time, gap.end_time))
                    imputation_summary['major_gaps_flagged'] += 1
                    quality_flags.append(f"Major gap excluded: {gap.start_time} to {gap.end_time}")
                    logger.warning(f"Major gap flagged for exclusion: {gap.start_time} to {gap.end_time}")
                    
            except Exception as e:
                logger.error(f"Error imputing gap {gap.start_time}-{gap.end_time}: {e}")
                quality_flags.append(f"Imputation failed for gap: {gap.start_time}")
        
        # Validate imputed data
        validation_issues = self._validate_imputed_data(imputed_df)
        quality_flags.extend(validation_issues)
        
        logger.info(f"Imputation completed: {imputation_summary}")
        
        return ImputationResult(
            imputed_df=imputed_df,
            imputation_summary=imputation_summary,
            excluded_periods=excluded_periods,
            quality_flags=quality_flags
        )
    
    def _impute_minor_gap(self, df: pd.DataFrame, gap: Gap, atr_series: pd.Series) -> bool:
        """
        Impute minor gaps (<5 min) with OHLC-specific interpolation.
        
        Strategy:
        - Open: Use previous Close
        - High: max(prev_high, interpolated_open) 
        - Low: min(prev_low, interpolated_open)
        - Close: Linear interpolation
        - Volume: Set to 0 or very small value
        """
        try:
            # Create timestamp range for the gap
            gap_range = pd.date_range(
                start=gap.start_time,
                end=gap.end_time,
                freq='1min',
                tz=gap.start_time.tz
            )
            
            # Find surrounding data points
            prev_idx = df.index[df.index < gap.start_time][-1] if len(df.index[df.index < gap.start_time]) > 0 else None
            next_idx = df.index[df.index > gap.end_time][0] if len(df.index[df.index > gap.end_time]) > 0 else None
            
            if prev_idx is None or next_idx is None:
                logger.warning(f"Cannot impute gap {gap.start_time}-{gap.end_time}: missing surrounding data")
                return False
            
            prev_data = df.loc[prev_idx]
            next_data = df.loc[next_idx]
            
            # For 1-minute gaps, use simpler approach to avoid division by zero
            total_gap_duration = (gap.end_time - gap.start_time).total_seconds() / 60
            
            # Generate imputed values for each timestamp in gap
            for i, timestamp in enumerate(gap_range):
                # For single minute gaps or very short gaps
                if total_gap_duration <= 1 or len(gap_range) <= 1:
                    # Simple approach: use previous close as basis
                    close_price = prev_data['Close']
                    open_price = prev_data['Close']
                else:
                    # Linear interpolation weight
                    weight = i / (len(gap_range) - 1)
                    
                    # Open: Use previous Close
                    open_price = prev_data['Close']
                    
                    # Close: Linear interpolation between previous and next Close
                    close_price = prev_data['Close'] * (1 - weight) + next_data['Close'] * weight
                
                # High: max of recent highs and current open/close
                high_price = max(prev_data['High'], open_price, close_price)
                
                # Low: min of recent lows and current open/close  
                low_price = min(prev_data['Low'], open_price, close_price)
                
                # Volume: Very small value to indicate imputed data
                volume = max(1, prev_data['Volume'] * 0.01)  # 1% of previous volume
                
                # Add to DataFrame
                df.loc[timestamp] = {
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                }
            
            logger.debug(f"Minor gap imputed: {gap.start_time} to {gap.end_time}")
            return True
            
        except Exception as e:
            logger.error(f"Error in minor gap imputation: {e}")
            return False
    
    def _impute_moderate_gap(self, df: pd.DataFrame, gap: Gap, atr_series: pd.Series) -> bool:
        """
        Impute moderate gaps (5-30 min) with time-weighted interpolation + ATR noise.
        
        Strategy:
        - Time-weighted linear interpolation for OHLC
        - Add small random noise based on ATR to avoid perfect linearity
        - Volume: Smoothed average of surrounding volumes
        """
        try:
            # Create timestamp range for the gap
            gap_range = pd.date_range(
                start=gap.start_time,
                end=gap.end_time,
                freq='1min',
                tz=gap.start_time.tz
            )
            
            # Find surrounding data (wider window for moderate gaps)
            window_size = 10  # Look at 10 points before and after
            
            prev_indices = df.index[df.index < gap.start_time][-window_size:] if len(df.index[df.index < gap.start_time]) >= window_size else df.index[df.index < gap.start_time]
            next_indices = df.index[df.index > gap.end_time][:window_size] if len(df.index[df.index > gap.end_time]) >= window_size else df.index[df.index > gap.end_time]
            
            if len(prev_indices) == 0 or len(next_indices) == 0:
                logger.warning(f"Cannot impute gap {gap.start_time}-{gap.end_time}: insufficient surrounding data")
                return False
            
            # Get average ATR for noise calculation
            atr_value = atr_series.loc[prev_indices[-1]] if prev_indices[-1] in atr_series.index else atr_series.mean()
            
            # Calculate base interpolation values
            prev_close = df.loc[prev_indices[-1], 'Close']
            next_close = df.loc[next_indices[0], 'Close']
            
            # Average volume from surrounding periods
            surrounding_volumes = pd.concat([
                df.loc[prev_indices, 'Volume'],
                df.loc[next_indices, 'Volume']
            ])
            avg_volume = surrounding_volumes.median()
            
            # Generate imputed values with time-weighted interpolation + noise
            total_gap_duration = (gap.end_time - gap.start_time).total_seconds() / 60
            
            for i, timestamp in enumerate(gap_range):
                # Time weight (0 at start, 1 at end) - handle single point gaps
                if len(gap_range) <= 1:
                    weight = 0.5  # Middle interpolation for single point
                else:
                    weight = i / (len(gap_range) - 1)
                
                # Base interpolated close price
                base_close = prev_close * (1 - weight) + next_close * weight
                
                # Add realistic noise based on ATR
                noise = np.random.normal(0, atr_value * self.noise_factor)
                close_price = base_close + noise
                
                # Open: Previous close + small noise
                if i == 0:
                    open_price = prev_close + np.random.normal(0, atr_value * 0.05)
                else:
                    open_price = df.loc[gap_range[i-1], 'Close'] + np.random.normal(0, atr_value * 0.05)
                
                # High/Low: Based on open/close with some random variation
                mid_price = (open_price + close_price) / 2
                price_range = abs(close_price - open_price) + atr_value * 0.1
                
                high_price = mid_price + price_range * np.random.uniform(0.3, 0.7)
                low_price = mid_price - price_range * np.random.uniform(0.3, 0.7)
                
                # Ensure OHLC relationships
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                # Volume: Add some randomness to average
                volume = max(1, avg_volume * np.random.uniform(0.5, 1.5))
                
                # Add to DataFrame
                df.loc[timestamp] = {
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                }
            
            logger.debug(f"Moderate gap imputed: {gap.start_time} to {gap.end_time}")
            return True
            
        except Exception as e:
            logger.error(f"Error in moderate gap imputation: {e}")
            return False
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range for noise generation."""
        try:
            if len(df) < self.atr_period:
                logger.warning("Insufficient data for ATR calculation, using simplified version")
                return pd.Series(df['High'] - df['Low'], index=df.index).rolling(window=min(5, len(df))).mean()
            
            # True Range calculation
            df_shifted = df.shift(1)
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df_shifted['Close'])
            tr3 = abs(df['Low'] - df_shifted['Close'])
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.atr_period).mean()
            
            # Fill NaN values with median
            atr.fillna(atr.median(), inplace=True)
            
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            # Fallback: simple range
            return (df['High'] - df['Low']).rolling(window=5).mean().fillna(df['High'] - df['Low'])
    
    def _validate_imputed_data(self, df: pd.DataFrame) -> List[str]:
        """Validate that imputed data maintains realistic OHLC relationships."""
        issues = []
        
        try:
            # Check OHLC relationships
            invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
            invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
            
            if invalid_high > 0:
                issues.append(f"Found {invalid_high} rows where High < max(Open, Close)")
            
            if invalid_low > 0:
                issues.append(f"Found {invalid_low} rows where Low > min(Open, Close)")
            
            # Check for extreme values
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df.columns:
                    # Check for negative prices
                    negative_prices = (df[col] <= 0).sum()
                    if negative_prices > 0:
                        issues.append(f"Found {negative_prices} negative/zero prices in {col}")
                    
                    # Check for extreme price movements (>50% in one minute)
                    if len(df) > 1:
                        extreme_moves = (df[col].pct_change().abs() > 0.5).sum()
                        if extreme_moves > 0:
                            issues.append(f"Found {extreme_moves} extreme price movements (>50%) in {col}")
            
            return issues
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            return [f"Validation error: {e}"]


def validate_ohlcv_data(
    df: pd.DataFrame, 
    atr_period: int = 14,
    volume_window: int = 20,
    price_change_threshold: float = 5.0,
    volume_low_percentile: float = 5.0,
    volume_high_percentile: float = 95.0
) -> ValidationResult:
    """
    Comprehensive validation of OHLCV financial data.
    
    Performs multiple quality checks including OHLC relationships, price movement
    anomalies, and volume consistency analysis.
    
    Args:
        df (pd.DataFrame): OHLCV DataFrame with datetime index
        atr_period (int): Period for ATR calculation (default: 14)
        volume_window (int): Rolling window for volume percentiles (default: 20)
        price_change_threshold (float): Standard deviation threshold for price movements (default: 5.0)
        volume_low_percentile (float): Lower percentile for volume anomaly detection (default: 5.0)
        volume_high_percentile (float): Upper percentile for volume anomaly detection (default: 95.0)
        
    Returns:
        ValidationResult: Comprehensive validation results with anomaly details
        
    Example:
        >>> result = validate_ohlcv_data(df)
        >>> print(f"Data Quality Score: {result.data_quality_score:.2f}")
        >>> print(f"OHLC Violations: {result.validation_summary['ohlc_violations']}")
    """
    logger.info(f"Starting comprehensive validation of {len(df)} OHLCV records")
    
    if df.empty:
        logger.warning("Empty DataFrame provided for validation")
        return ValidationResult(
            total_records=0,
            validation_summary={},
            anomaly_details={},
            data_quality_score=0.0,
            recommended_actions=["No data to validate"]
        )
    
    # Initialize results tracking
    validation_summary = {
        'ohlc_violations': 0,
        'impossible_price_movements': 0,
        'volume_anomalies_low': 0,
        'volume_anomalies_high': 0,
        'missing_data_points': 0,
        'negative_prices': 0,
        'zero_volume_periods': 0
    }
    
    anomaly_details = {
        'ohlc_violations': [],
        'impossible_price_movements': [],
        'volume_anomalies': [],
        'data_quality_issues': []
    }
    
    recommended_actions = []
    
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error_msg = f"Missing required columns: {missing_cols}"
        logger.error(error_msg)
        return ValidationResult(
            total_records=len(df),
            validation_summary={'error': 1},
            anomaly_details={'error': [error_msg]},
            data_quality_score=0.0,
            recommended_actions=[f"Add missing columns: {missing_cols}"]
        )
    
    try:
        # 1. OHLC RELATIONSHIP VALIDATION
        logger.info("Validating OHLC relationships...")
        ohlc_violations = _validate_ohlc_relationships(df)
        validation_summary['ohlc_violations'] = len(ohlc_violations)
        anomaly_details['ohlc_violations'] = ohlc_violations
        
        # 2. IMPOSSIBLE PRICE MOVEMENTS DETECTION
        logger.info("Detecting impossible price movements...")
        price_anomalies = _detect_impossible_price_movements(
            df, atr_period, price_change_threshold
        )
        validation_summary['impossible_price_movements'] = len(price_anomalies)
        anomaly_details['impossible_price_movements'] = price_anomalies
        
        # 3. VOLUME CONSISTENCY ANALYSIS
        logger.info("Analyzing volume consistency...")
        volume_anomalies = _analyze_volume_consistency(
            df, volume_window, volume_low_percentile, volume_high_percentile
        )
        validation_summary['volume_anomalies_low'] = len(volume_anomalies['low'])
        validation_summary['volume_anomalies_high'] = len(volume_anomalies['high'])
        anomaly_details['volume_anomalies'] = volume_anomalies['low'] + volume_anomalies['high']
        
        # 4. ADDITIONAL DATA QUALITY CHECKS
        logger.info("Performing additional data quality checks...")
        quality_issues = _perform_additional_quality_checks(df)
        validation_summary.update(quality_issues['summary'])
        anomaly_details['data_quality_issues'] = quality_issues['details']
        
        # 5. CALCULATE DATA QUALITY SCORE
        quality_score = _calculate_data_quality_score(df, validation_summary)
        
        # 6. GENERATE RECOMMENDATIONS
        recommended_actions = _generate_recommendations(validation_summary, quality_score)
        
        logger.info(f"Validation completed. Quality score: {quality_score:.2f}")
        
        return ValidationResult(
            total_records=len(df),
            validation_summary=validation_summary,
            anomaly_details=anomaly_details,
            data_quality_score=quality_score,
            recommended_actions=recommended_actions
        )
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return ValidationResult(
            total_records=len(df),
            validation_summary={'validation_error': 1},
            anomaly_details={'error': [str(e)]},
            data_quality_score=0.0,
            recommended_actions=[f"Fix validation error: {e}"]
        )


def _validate_ohlc_relationships(df: pd.DataFrame) -> List[Dict]:
    """Validate OHLC price relationships."""
    violations = []
    
    try:
        # Check High >= max(Open, Close) and High >= Low
        high_violations = (
            (df['High'] < df['Open']) | 
            (df['High'] < df['Close']) | 
            (df['High'] < df['Low'])
        )
        
        # Check Low <= min(Open, Close) and Low <= High  
        low_violations = (
            (df['Low'] > df['Open']) | 
            (df['Low'] > df['Close']) | 
            (df['Low'] > df['High'])
        )
        
        # Combine violations
        all_violations = high_violations | low_violations
        
        # Create detailed violation records
        for timestamp in df.index[all_violations]:
            row = df.loc[timestamp]
            violation_types = []
            
            if row['High'] < max(row['Open'], row['Close'], row['Low']):
                violation_types.append("High < max(O,C,L)")
            if row['Low'] > min(row['Open'], row['Close'], row['High']):
                violation_types.append("Low > min(O,C,H)")
                
            violations.append({
                'timestamp': timestamp,
                'violation_type': 'OHLC_relationship',
                'details': violation_types,
                'values': {
                    'Open': row['Open'],
                    'High': row['High'], 
                    'Low': row['Low'],
                    'Close': row['Close']
                },
                'severity': 'high'
            })
            
    except Exception as e:
        logger.error(f"Error in OHLC validation: {e}")
        
    return violations


def _detect_impossible_price_movements(
    df: pd.DataFrame, 
    atr_period: int, 
    threshold: float
) -> List[Dict]:
    """Detect impossible price movements using ATR-normalized analysis."""
    anomalies = []
    
    try:
        if len(df) < atr_period + 1:
            logger.warning("Insufficient data for price movement analysis")
            return anomalies
            
        # Calculate ATR
        atr_series = _calculate_atr_for_validation(df, atr_period)
        
        # Calculate price changes
        price_changes = df['Close'].pct_change()
        
        # Calculate ATR-normalized price changes
        atr_normalized_changes = price_changes.abs() / (atr_series / df['Close'])
        
        # Detect outliers using statistical methods
        # Method 1: Standard deviation threshold
        mean_normalized = atr_normalized_changes.mean()
        std_normalized = atr_normalized_changes.std()
        
        outliers_std = atr_normalized_changes > (mean_normalized + threshold * std_normalized)
        
        # Method 2: Z-score approach for additional validation
        z_scores = np.abs(stats.zscore(price_changes.dropna()))
        outliers_zscore = z_scores > threshold
        
        # Combine both methods
        outlier_indices = df.index[outliers_std | outliers_zscore]
        
        # Create detailed anomaly records
        for timestamp in outlier_indices:
            if timestamp in price_changes.index and not pd.isna(price_changes.loc[timestamp]):
                current_price = df.loc[timestamp, 'Close']
                previous_timestamp = df.index[df.index < timestamp][-1]
                previous_price = df.loc[previous_timestamp, 'Close']
                
                change_pct = price_changes.loc[timestamp] * 100
                atr_value = atr_series.loc[timestamp] if timestamp in atr_series.index else np.nan
                
                anomalies.append({
                    'timestamp': timestamp,
                    'violation_type': 'impossible_price_movement',
                    'details': f"{change_pct:.2f}% price change",
                    'values': {
                        'current_price': current_price,
                        'previous_price': previous_price,
                        'change_percent': change_pct,
                        'atr_value': atr_value,
                        'z_score': z_scores[df.index.get_loc(timestamp)] if timestamp in df.index else np.nan
                    },
                    'severity': 'high' if abs(change_pct) > 10 else 'medium'
                })
                
    except Exception as e:
        logger.error(f"Error in price movement detection: {e}")
        
    return anomalies


def _analyze_volume_consistency(
    df: pd.DataFrame,
    window: int,
    low_percentile: float,
    high_percentile: float
) -> Dict[str, List[Dict]]:
    """Analyze volume consistency and detect anomalies."""
    volume_anomalies = {'low': [], 'high': []}
    
    try:
        if 'Volume' not in df.columns or len(df) < window:
            logger.warning("Insufficient data for volume analysis")
            return volume_anomalies
            
        # Calculate rolling percentiles
        volume_low_threshold = df['Volume'].rolling(window=window).quantile(low_percentile / 100)
        volume_high_threshold = df['Volume'].rolling(window=window).quantile(high_percentile / 100)
        
        # Detect low volume anomalies
        low_volume_mask = (df['Volume'] < volume_low_threshold) & (df['Volume'] > 0)
        
        # Detect high volume anomalies  
        high_volume_mask = df['Volume'] > volume_high_threshold
        
        # Create detailed anomaly records for low volume
        for timestamp in df.index[low_volume_mask]:
            current_volume = df.loc[timestamp, 'Volume']
            threshold = volume_low_threshold.loc[timestamp]
            
            # Calculate how far below threshold
            deviation_ratio = (threshold - current_volume) / threshold if threshold > 0 else 0
            
            volume_anomalies['low'].append({
                'timestamp': timestamp,
                'violation_type': 'volume_anomaly_low',
                'details': f"Volume {deviation_ratio:.1%} below {low_percentile}th percentile",
                'values': {
                    'current_volume': current_volume,
                    'threshold': threshold,
                    'deviation_ratio': deviation_ratio
                },
                'severity': 'high' if deviation_ratio > 0.8 else 'medium'
            })
            
        # Create detailed anomaly records for high volume
        for timestamp in df.index[high_volume_mask]:
            current_volume = df.loc[timestamp, 'Volume']
            threshold = volume_high_threshold.loc[timestamp]
            
            # Calculate how far above threshold
            deviation_ratio = (current_volume - threshold) / threshold if threshold > 0 else 0
            
            volume_anomalies['high'].append({
                'timestamp': timestamp,
                'violation_type': 'volume_anomaly_high', 
                'details': f"Volume {deviation_ratio:.1%} above {high_percentile}th percentile",
                'values': {
                    'current_volume': current_volume,
                    'threshold': threshold,
                    'deviation_ratio': deviation_ratio
                },
                'severity': 'high' if deviation_ratio > 5 else 'medium'
            })
            
    except Exception as e:
        logger.error(f"Error in volume analysis: {e}")
        
    return volume_anomalies


def _perform_additional_quality_checks(df: pd.DataFrame) -> Dict:
    """Perform additional data quality checks."""
    summary = {}
    details = []
    
    try:
        # Check for missing data
        missing_data = df.isnull().sum().sum()
        summary['missing_data_points'] = missing_data
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        negative_prices = 0
        for col in price_cols:
            if col in df.columns:
                negative_count = (df[col] <= 0).sum()
                negative_prices += negative_count
                
                if negative_count > 0:
                    details.append({
                        'issue_type': 'negative_prices',
                        'column': col,
                        'count': negative_count,
                        'severity': 'high'
                    })
                    
        summary['negative_prices'] = negative_prices
        
        # Check for zero volume periods
        if 'Volume' in df.columns:
            zero_volume = (df['Volume'] == 0).sum()
            summary['zero_volume_periods'] = zero_volume
            
            if zero_volume > 0:
                details.append({
                    'issue_type': 'zero_volume',
                    'count': zero_volume,
                    'percentage': (zero_volume / len(df)) * 100,
                    'severity': 'medium' if zero_volume < len(df) * 0.01 else 'high'
                })
                
        # Check for duplicate timestamps
        duplicate_timestamps = df.index.duplicated().sum()
        if duplicate_timestamps > 0:
            summary['duplicate_timestamps'] = duplicate_timestamps
            details.append({
                'issue_type': 'duplicate_timestamps',
                'count': duplicate_timestamps,
                'severity': 'high'
            })
            
    except Exception as e:
        logger.error(f"Error in additional quality checks: {e}")
        details.append({
            'issue_type': 'validation_error',
            'error': str(e),
            'severity': 'high'
        })
        
    return {'summary': summary, 'details': details}


def _calculate_atr_for_validation(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate ATR for validation purposes."""
    try:
        if len(df) < period:
            return pd.Series(df['High'] - df['Low'], index=df.index)
            
        # True Range calculation
        df_shifted = df.shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df_shifted['Close'])
        tr3 = abs(df['Low'] - df_shifted['Close'])
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # Fill NaN values
        atr.fillna(method='bfill', inplace=True)
        atr.fillna(true_range.mean(), inplace=True)
        
        return atr
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series(df['High'] - df['Low'], index=df.index)


def _calculate_data_quality_score(df: pd.DataFrame, summary: Dict[str, int]) -> float:
    """Calculate overall data quality score (0-100)."""
    try:
        total_records = len(df)
        if total_records == 0:
            return 0.0
            
        # Weight different types of issues
        weights = {
            'ohlc_violations': 10,  # High impact
            'impossible_price_movements': 8,
            'negative_prices': 10,
            'volume_anomalies_low': 3,
            'volume_anomalies_high': 5,
            'missing_data_points': 7,
            'zero_volume_periods': 2,
            'duplicate_timestamps': 9
        }
        
        # Calculate weighted penalty
        total_penalty = 0
        for issue_type, count in summary.items():
            if issue_type in weights:
                penalty = (count / total_records) * weights[issue_type]
                total_penalty += penalty
                
        # Convert to score (0-100)
        score = max(0, 100 - total_penalty)
        
        return round(score, 2)
        
    except Exception as e:
        logger.error(f"Error calculating quality score: {e}")
        return 0.0


def _generate_recommendations(summary: Dict[str, int], quality_score: float) -> List[str]:
    """Generate actionable recommendations based on validation results."""
    recommendations = []
    
    try:
        if quality_score >= 95:
            recommendations.append("✅ Excellent data quality - no major issues detected")
        elif quality_score >= 85:
            recommendations.append("✅ Good data quality - minor issues may need attention")
        elif quality_score >= 70:
            recommendations.append("⚠️ Moderate data quality - several issues need correction")
        else:
            recommendations.append("❌ Poor data quality - immediate attention required")
            
        # Specific recommendations
        if summary.get('ohlc_violations', 0) > 0:
            recommendations.append(f"🔧 Fix {summary['ohlc_violations']} OHLC relationship violations")
            
        if summary.get('impossible_price_movements', 0) > 0:
            recommendations.append(f"🔍 Investigate {summary['impossible_price_movements']} unusual price movements")
            
        if summary.get('negative_prices', 0) > 0:
            recommendations.append(f"💰 Correct {summary['negative_prices']} negative price entries")
            
        if summary.get('volume_anomalies_low', 0) + summary.get('volume_anomalies_high', 0) > 10:
            recommendations.append("📊 Review volume data quality and outlier detection thresholds")
            
        if summary.get('missing_data_points', 0) > 0:
            recommendations.append("📝 Address missing data points through imputation or data source fixes")
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        recommendations.append("❓ Unable to generate recommendations due to error")
        
    return recommendations


# Convenience functions for direct use
def impute_data(df: pd.DataFrame, gaps: List[Gap]) -> ImputationResult:
    """
    Convenience function for data imputation.
    
    Args:
        df (pd.DataFrame): OHLCV DataFrame with datetime index
        gaps (List[Gap]): List of detected gaps
        
    Returns:
        ImputationResult: Imputed data with summary and excluded periods
    """
    imputer = DataImputer()
    return imputer.impute_data(df, gaps)


def validate_data(df: pd.DataFrame, **kwargs) -> ValidationResult:
    """
    Convenience function for comprehensive OHLCV data validation.
    
    Args:
        df (pd.DataFrame): OHLCV DataFrame with datetime index
        **kwargs: Additional arguments for validation (atr_period, volume_window, etc.)
        
    Returns:
        ValidationResult: Comprehensive validation results
    """
    return validate_ohlcv_data(df, **kwargs)


# Example usage and testing
def main():
    """Example usage of the data imputation module."""
    
    # Test data validation
    print("\n" + "="*60)
    print("Data Validation - Example Usage")
    print("="*60)
    
    # Create sample data first
    dates = pd.date_range('2024-01-15 09:00', '2024-01-15 12:00', freq='1min', tz='Europe/Berlin')
    
    # Sample OHLCV data
    np.random.seed(42)
    base_price = 18000
    sample_data = {
        'Open': base_price + np.random.normal(0, 50, len(dates)),
        'High': base_price + 20 + np.random.normal(0, 30, len(dates)),
        'Low': base_price - 20 + np.random.normal(0, 30, len(dates)),
        'Close': base_price + np.random.normal(0, 50, len(dates)),
        'Volume': np.random.randint(100, 1000, len(dates))
    }
    
    # Ensure OHLC relationships
    for i in range(len(dates)):
        sample_data['High'][i] = max(sample_data['High'][i], sample_data['Open'][i], sample_data['Close'][i])
        sample_data['Low'][i] = min(sample_data['Low'][i], sample_data['Open'][i], sample_data['Close'][i])
    
    df = pd.DataFrame(sample_data, index=dates)
    
    # Validate the sample data
    validation_result = validate_ohlcv_data(df)
    
    print(f"\nValidation Results:")
    print(f"  Total Records: {validation_result.total_records}")
    print(f"  Data Quality Score: {validation_result.data_quality_score}/100")
    
    print(f"\nValidation Summary:")
    for check, count in validation_result.validation_summary.items():
        print(f"  {check}: {count}")
        
    if validation_result.anomaly_details:
        print(f"\nSample Anomalies:")
        for anomaly_type, anomalies in validation_result.anomaly_details.items():
            if anomalies:
                print(f"  {anomaly_type}: {len(anomalies)} detected")
                if len(anomalies) > 0:
                    print(f"    Example: {anomalies[0]['details']}")
                    
    print(f"\nRecommendations:")
    for recommendation in validation_result.recommended_actions:
        print(f"  {recommendation}")

    print("\n" + "="*60)
    print("Data Imputation - Example Usage")
    print("="*60)
    
    # Remove some data to create gaps for imputation testing
    gap_indices = [
        pd.date_range('2024-01-15 10:00', '2024-01-15 10:02', freq='1min', tz='Europe/Berlin'),  # Minor gap
        pd.date_range('2024-01-15 11:00', '2024-01-15 11:15', freq='1min', tz='Europe/Berlin'),  # Moderate gap
    ]
    
    df_with_gaps = df.copy()
    for gap_range in gap_indices:
        df_with_gaps = df_with_gaps.drop(gap_range, errors='ignore')
    
    print(f"Original data points: {len(dates)}")
    print(f"Data after creating gaps: {len(df_with_gaps)}")
    
    # Detect gaps
    gap_detector = GapDetector()
    gaps = gap_detector.detect_gaps(df_with_gaps)
    
    print(f"Detected {len(gaps)} gaps:")
    for gap in gaps:
        print(f"  {gap.severity.title()} gap: {gap.start_time} to {gap.end_time} ({gap.duration_minutes} min)")
    
    # Impute data
    result = impute_data(df_with_gaps, gaps)
    
    print(f"\nImputation Results:")
    print(f"  Original rows: {len(df_with_gaps)}")
    print(f"  Imputed rows: {len(result.imputed_df)}")
    print(f"  Summary: {result.imputation_summary}")
    
    if result.excluded_periods:
        print(f"  Excluded periods: {len(result.excluded_periods)}")
    
    if result.quality_flags:
        print(f"  Quality flags: {result.quality_flags}")
    
    # Show sample of imputed data
    print(f"\nSample of imputed data:")
    print(result.imputed_df.head(10))


if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()


    '''Key Features of the Validation Framework:
1. Comprehensive OHLC Relationship Validation

High >= max(Open, Close, Low) validation
Low <= min(Open, Close, High) validation
Detailed violation reporting with timestamps and exact values

2. Impossible Price Movement Detection

ATR-normalized price changes for context-aware analysis
5 standard deviation threshold for outlier detection
Z-score analysis for additional validation
Severity classification (high >10%, medium <10%)

3. Volume Consistency Analysis

Rolling percentile boundaries (5th and 95th percentiles)
20-period window for dynamic thresholds
Low and high volume anomaly detection
Deviation ratio calculations for severity assessment

4. Additional Quality Checks

Missing data detection
Negative/zero price validation
Zero volume period identification
Duplicate timestamp detection

5. Data Quality Scoring System

0-100 quality score with weighted penalties
Severity-based recommendations
Actionable improvement suggestions'''