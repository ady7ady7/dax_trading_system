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

# Import our gap detection module
from .gap_detector import Gap, GapDetector

logger = logging.getLogger(__name__)


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
            
            # Generate imputed values for each timestamp in gap
            for timestamp in gap_range:
                # Open: Use previous Close
                open_price = prev_data['Close']
                
                # Close: Linear interpolation between previous and next Close
                weight = (timestamp - gap.start_time) / (gap.end_time - gap.start_time)
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
                # Time weight (0 at start, 1 at end)
                weight = i / (len(gap_range) - 1) if len(gap_range) > 1 else 0.5
                
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


# Convenience function for direct use
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


# Example usage and testing
def main():
    """Example usage of the data imputation module."""
    
    print("\n" + "="*60)
    print("Data Imputation - Example Usage")
    print("="*60)
    
    # Create sample data with gaps for testing
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
    
    # Remove some data to create gaps
    gap_indices = [
        pd.date_range('2024-01-15 10:00', '2024-01-15 10:02', freq='1min', tz='Europe/Berlin'),  # Minor gap
        pd.date_range('2024-01-15 11:00', '2024-01-15 11:15', freq='1min', tz='Europe/Berlin'),  # Moderate gap
    ]
    
    for gap_range in gap_indices:
        df = df.drop(gap_range, errors='ignore')
    
    print(f"Original data points: {len(dates)}")
    print(f"Data after creating gaps: {len(df)}")
    
    # Detect gaps
    gap_detector = GapDetector()
    gaps = gap_detector.detect_gaps(df)
    
    print(f"Detected {len(gaps)} gaps:")
    for gap in gaps:
        print(f"  {gap.severity.title()} gap: {gap.start_time} to {gap.end_time} ({gap.duration_minutes} min)")
    
    # Impute data
    result = impute_data(df, gaps)
    
    print(f"\nImputation Results:")
    print(f"  Original rows: {len(df)}")
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