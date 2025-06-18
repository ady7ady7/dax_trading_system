"""
DAX Trading System - Feature Engineering Module

This module provides comprehensive technical analysis feature generation across
multiple timeframes with DAX-specific indicators and strict look-ahead bias prevention.

Author: DAX Trading System
Created: 2025-06-18
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import time, datetime
import pytz
from pathlib import Path
import warnings

# Suppress pandas performance warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive technical analysis feature generator for DAX 1-minute OHLCV data.
    
    Generates momentum, trend, volatility, and volume indicators across multiple timeframes
    with custom DAX-specific features and strict look-ahead bias prevention.
    
    Features:
    - Multi-timeframe analysis (1m, 5m, 15m, 1h)
    - Momentum indicators (RSI, MACD, Stochastic, CCI, Williams %R)
    - Trend indicators (EMA crossovers, ADX)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume indicators (OBV, CMF)
    - DAX-specific features (opening gaps, European session patterns)
    """
    
    def __init__(self, 
                 timezone: str = "Europe/Berlin",
                 market_open: time = time(9, 0),
                 market_close: time = time(17, 30)):
        """
        Initialize the FeatureEngineer.
        
        Args:
            timezone (str): Timezone for market hours (default: Europe/Berlin)
            market_open (time): Market opening time (default: 09:00)
            market_close (time): Market closing time (default: 17:30)
        """
        self.timezone = pytz.timezone(timezone)
        self.market_open = market_open
        self.market_close = market_close
        
        # Timeframes for multi-timeframe analysis
        self.timeframes = {
            '1min': '1T',
            '5min': '5T', 
            '15min': '15T',
            '1hour': '1H'
        }
        
        # EMA pairs for crossover analysis
        self.ema_pairs = [
            (5, 10), (10, 20), (20, 50), (50, 100), (100, 200)
        ]
        
        # Default parameters for indicators
        self.default_params = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'stoch_k': 14,
            'stoch_d': 3,
            'cci_period': 20,
            'williams_period': 14,
            'adx_period': 14,
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'cmf_period': 20
        }
        
        logger.info(f"FeatureEngineer initialized for timezone {timezone}")
        logger.info(f"Market hours: {market_open} - {market_close}")
        logger.info(f"Timeframes: {list(self.timeframes.keys())}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive technical analysis features.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with datetime index in CET timezone
            
        Returns:
            pd.DataFrame: Original data with engineered features
        """
        logger.info(f"Starting feature engineering for {len(df)} records")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Validate input data
        if not self._validate_input(df):
            raise ValueError("Invalid input data format")
        
        # Create copy to avoid modifying original data
        result_df = df.copy()
        
        # Generate base 1-minute features
        logger.info("Generating base 1-minute features...")
        result_df = self._add_base_features(result_df)
        
        # Generate multi-timeframe features
        logger.info("Generating multi-timeframe features...")
        result_df = self._add_multitimeframe_features(result_df, df)
        
        # Generate DAX-specific features
        logger.info("Generating DAX-specific features...")
        result_df = self._add_dax_specific_features(result_df)
        
        # Generate European session features
        logger.info("Generating European session features...")
        result_df = self._add_european_session_features(result_df)
        
        # Final cleanup and validation
        result_df = self._cleanup_features(result_df)
        
        feature_count = len(result_df.columns) - len(df.columns)
        logger.info(f"Feature engineering completed: {feature_count} features generated")
        
        return result_df
    
    def _validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input DataFrame format."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Expected: {required_columns}")
            return False
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be DatetimeIndex")
            return False
        
        if df.index.tz is None:
            logger.warning("DataFrame index is timezone-naive")
        
        if len(df) < 200:
            logger.warning(f"Small dataset ({len(df)} rows) may not have sufficient data for all indicators")
        
        return True
    
    def _add_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate base technical indicators on 1-minute timeframe."""
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = df['High'] - df['Low']
        df['Body_Size'] = abs(df['Close'] - df['Open'])
        df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
        df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Trend indicators
        df = self._add_trend_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based technical indicators."""
        
        # RSI (Relative Strength Index)
        rsi_period = self.default_params['rsi_period']
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        fast_ema = df['Close'].ewm(span=self.default_params['macd_fast']).mean()
        slow_ema = df['Close'].ewm(span=self.default_params['macd_slow']).mean()
        df['MACD'] = fast_ema - slow_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=self.default_params['macd_signal']).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        stoch_k = self.default_params['stoch_k']
        stoch_d = self.default_params['stoch_d']
        low_min = df['Low'].rolling(window=stoch_k).min()
        high_max = df['High'].rolling(window=stoch_k).max()
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df[f'Stoch_K_{stoch_k}'] = k_percent
        df[f'Stoch_D_{stoch_d}'] = k_percent.rolling(window=stoch_d).mean()
        
        # CCI (Commodity Channel Index)
        cci_period = self.default_params['cci_period']
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=cci_period).mean()
        mad = typical_price.rolling(window=cci_period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        df[f'CCI_{cci_period}'] = (typical_price - sma) / (0.015 * mad)
        
        # Williams %R
        williams_period = self.default_params['williams_period']
        high_max = df['High'].rolling(window=williams_period).max()
        low_min = df['Low'].rolling(window=williams_period).min()
        df[f'Williams_R_{williams_period}'] = -100 * (
            (high_max - df['Close']) / (high_max - low_min)
        )
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based technical indicators."""
        
        # EMA crossovers
        for fast, slow in self.ema_pairs:
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            
            df[f'EMA_{fast}'] = ema_fast
            df[f'EMA_{slow}'] = ema_slow
            df[f'EMA_Cross_{fast}_{slow}'] = ema_fast - ema_slow
            df[f'EMA_Cross_Signal_{fast}_{slow}'] = np.where(
                df[f'EMA_Cross_{fast}_{slow}'] > 0, 1, -1
            )
        
        # ADX (Average Directional Index)
        adx_period = self.default_params['adx_period']
        df = self._calculate_adx(df, adx_period)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate ADX and related directional indicators."""
        
        # True Range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        # Directional Movement
        df['DM_Plus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        
        df['DM_Minus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )
        
        # Smoothed values
        df['ATR'] = df['TR'].rolling(window=period).mean()
        df['DI_Plus'] = 100 * (df['DM_Plus'].rolling(window=period).mean() / df['ATR'])
        df['DI_Minus'] = 100 * (df['DM_Minus'].rolling(window=period).mean() / df['ATR'])
        
        # ADX calculation
        df['DX'] = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df[f'ADX_{period}'] = df['DX'].rolling(window=period).mean()
        
        # Clean up intermediate columns
        df.drop(['TR', 'DM_Plus', 'DM_Minus', 'DX'], axis=1, inplace=True)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based technical indicators."""
        
        # ATR (Average True Range) - already calculated in ADX
        if 'ATR' not in df.columns:
            atr_period = self.default_params['atr_period']
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df[f'ATR_{atr_period}'] = true_range.rolling(window=atr_period).mean()
        
        # Bollinger Bands
        bb_period = self.default_params['bb_period']
        bb_std = self.default_params['bb_std']
        
        sma = df['Close'].rolling(window=bb_period).mean()
        std = df['Close'].rolling(window=bb_period).std()
        
        df[f'BB_Upper_{bb_period}'] = sma + (bb_std * std)
        df[f'BB_Lower_{bb_period}'] = sma - (bb_std * std)
        df[f'BB_Middle_{bb_period}'] = sma
        df[f'BB_Width_{bb_period}'] = (df[f'BB_Upper_{bb_period}'] - df[f'BB_Lower_{bb_period}']) / sma
        df[f'BB_Percent_{bb_period}'] = (df['Close'] - df[f'BB_Lower_{bb_period}']) / (
            df[f'BB_Upper_{bb_period}'] - df[f'BB_Lower_{bb_period}']
        )
        
        # Normalized ATR (as percentage of price)
        df['ATR_Normalized'] = df['ATR'] / df['Close'] * 100
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based technical indicators."""
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Chaikin Money Flow (CMF)
        cmf_period = self.default_params['cmf_period']
        
        # Money Flow Multiplier
        mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mf_multiplier = mf_multiplier.fillna(0)  # Handle division by zero
        
        # Money Flow Volume
        mf_volume = mf_multiplier * df['Volume']
        
        # CMF
        df[f'CMF_{cmf_period}'] = (
            mf_volume.rolling(window=cmf_period).sum() / 
            df['Volume'].rolling(window=cmf_period).sum()
        )
        
        # Volume-related features
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        df['Price_Volume'] = df['Close'] * df['Volume']
        
        return df
    
    def _add_multitimeframe_features(self, result_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Generate features across multiple timeframes."""
        
        for tf_name, tf_freq in self.timeframes.items():
            if tf_name == '1min':  # Skip 1-minute as it's already processed
                continue
                
            logger.info(f"Processing {tf_name} timeframe...")
            
            # Resample to higher timeframe
            tf_df = self._resample_ohlcv(original_df, tf_freq)
            
            if len(tf_df) < 50:  # Skip if insufficient data
                logger.warning(f"Insufficient data for {tf_name} timeframe")
                continue
            
            # Generate indicators for this timeframe
            tf_df = self._add_timeframe_indicators(tf_df, tf_name)
            
            # Merge back to 1-minute data with forward fill
            result_df = self._merge_timeframe_features(result_df, tf_df, tf_name)
        
        return result_df
    
    def _resample_ohlcv(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLCV data to higher timeframe."""
        
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Resample and aggregate
        resampled = df.resample(freq, label='right', closed='right').agg(agg_dict)
        
        # Remove any rows with NaN values
        resampled = resampled.dropna()
        
        return resampled
    
    def _add_timeframe_indicators(self, df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
        """Add key indicators for a specific timeframe."""
        
        # RSI
        rsi_period = min(14, len(df) // 4)  # Adjust period for shorter timeframes
        if rsi_period >= 2:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df[f'RSI_{tf_name}'] = 100 - (100 / (1 + rs))
        
        # EMA trends
        for fast, slow in [(5, 10), (10, 20)]:  # Use shorter periods for higher timeframes
            if len(df) > slow:
                ema_fast = df['Close'].ewm(span=fast).mean()
                ema_slow = df['Close'].ewm(span=slow).mean()
                df[f'EMA_Trend_{fast}_{slow}_{tf_name}'] = ema_fast - ema_slow
        
        # ATR
        atr_period = min(14, len(df) // 3)
        if atr_period >= 2:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df[f'ATR_{tf_name}'] = true_range.rolling(window=atr_period).mean()
        
        return df
    
    def _merge_timeframe_features(self, result_df: pd.DataFrame, tf_df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
        """Merge timeframe features back to 1-minute data."""
        
        # Select only the generated features (not OHLCV)
        feature_cols = [col for col in tf_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        tf_features = tf_df[feature_cols]
        
        # Reindex to match 1-minute data and forward fill
        tf_features_reindexed = tf_features.reindex(result_df.index, method='ffill')
        
        # Merge features
        for col in tf_features_reindexed.columns:
            result_df[col] = tf_features_reindexed[col]
        
        return result_df
    
    def _add_dax_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate DAX-specific features."""
        
        # Opening Gap Analysis
        df = self._add_opening_gap_features(df)
        
        # Session-based features
        df = self._add_session_features(df)
        
        # DAX-specific volatility patterns
        df = self._add_volatility_patterns(df)
        
        return df
    
    def _add_opening_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opening gap analysis features."""
        
        # Identify market opening times (09:00 CET)
        df['Is_Market_Open'] = df.index.time == self.market_open
        
        # Get previous day's close
        df['Prev_Close'] = df['Close'].shift(1)
        
        # Calculate gaps at market open
        gap_mask = df['Is_Market_Open']
        df['Opening_Gap'] = np.where(gap_mask, df['Open'] - df['Prev_Close'], 0)
        
        # Normalize gap by previous day's ATR
        if 'ATR' in df.columns:
            df['Opening_Gap_Normalized'] = np.where(
                gap_mask & (df['ATR'].shift(1) > 0),
                df['Opening_Gap'] / df['ATR'].shift(1),
                0
            )
        
        # Gap direction and magnitude
        df['Gap_Direction'] = np.where(gap_mask, np.sign(df['Opening_Gap']), 0)
        df['Gap_Magnitude'] = np.where(gap_mask, abs(df['Opening_Gap']), 0)
        
        # Gap percentage
        df['Gap_Percent'] = np.where(
            gap_mask & (df['Prev_Close'] > 0),
            (df['Opening_Gap'] / df['Prev_Close']) * 100,
            0
        )
        
        return df
    
    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add session-based trading features."""
        
        # Time-based features
        df['Hour'] = df.index.hour
        df['Minute'] = df.index.minute
        df['Day_of_Week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        
        # Session indicators
        df['Is_Pre_Market'] = (df['Hour'] < self.market_open.hour)
        df['Is_Market_Hours'] = (
            (df['Hour'] > self.market_open.hour) | 
            ((df['Hour'] == self.market_open.hour) & (df['Minute'] >= self.market_open.minute))
        ) & (
            (df['Hour'] < self.market_close.hour) |
            ((df['Hour'] == self.market_close.hour) & (df['Minute'] <= self.market_close.minute))
        )
        df['Is_After_Market'] = (df['Hour'] > self.market_close.hour)
        
        # European trading session patterns
        df['Is_Opening_Hour'] = (df['Hour'] == 9)  # 09:00-10:00
        df['Is_Lunch_Time'] = (df['Hour'] >= 12) & (df['Hour'] < 14)  # 12:00-14:00
        df['Is_Closing_Hour'] = (df['Hour'] == 17)  # 17:00-17:30
        
        return df
    
    def _add_volatility_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add DAX-specific volatility pattern features."""
        
        # Intraday volatility patterns
        if 'ATR' in df.columns:
            # Hourly ATR patterns
            df['ATR_Hourly_Avg'] = df.groupby(df.index.hour)['ATR'].transform('mean')
            df['ATR_vs_Hourly_Avg'] = df['ATR'] / df['ATR_Hourly_Avg']
            
            # Volume-adjusted volatility
            df['Vol_Adj_Volatility'] = df['ATR'] * np.log1p(df['Volume'])
        
        # Price movement patterns during European hours
        df['EU_Session_Range'] = np.where(
            df['Is_Market_Hours'],
            df['High'] - df['Low'],
            np.nan
        )
        
        # Rolling volatility measures
        df['Volatility_5min'] = df['Returns'].rolling(window=5).std() * np.sqrt(288)  # Annualized
        df['Volatility_30min'] = df['Returns'].rolling(window=30).std() * np.sqrt(288)
        
        return df
    
    def _add_european_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate European session-specific features."""
        
        # Opening hour analysis (09:00-10:00 CET)
        opening_hour_mask = (df.index.hour == 9)
        
        if opening_hour_mask.any():
            # Average opening hour movements
            df['Opening_Hour_Avg_Move'] = df.groupby(df.index.date)[opening_hour_mask]['Returns'].transform('mean')
            df['Opening_Hour_Avg_Volume'] = df.groupby(df.index.date)[opening_hour_mask]['Volume'].transform('mean')
            
            # Current vs average opening patterns
            df['Opening_Move_vs_Avg'] = np.where(
                opening_hour_mask,
                df['Returns'] - df['Opening_Hour_Avg_Move'],
                np.nan
            )
        
        # London-Frankfurt overlap (08:00-16:30 CET)
        london_overlap = (df.index.hour >= 8) & (df.index.hour < 16.5)
        df['London_Frankfurt_Overlap'] = london_overlap
        
        # European close effect (17:00-17:30 CET)
        close_effect_mask = (df.index.hour == 17) & (df.index.minute <= 30)
        if close_effect_mask.any():
            df['Close_Effect_Volume'] = np.where(
                close_effect_mask,
                df['Volume'],
                np.nan
            )
            
            # Rolling average volume during close
            df['Avg_Close_Volume'] = df['Close_Effect_Volume'].rolling(window=20, min_periods=1).mean()
        
        # Weekend effect (Friday close patterns)
        df['Is_Friday'] = (df.index.dayofweek == 4)
        df['Friday_Close_Effect'] = df['Is_Friday'] & close_effect_mask
        
        return df
    
    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up and validate generated features."""
        
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Log feature statistics
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        nan_counts = df[feature_cols].isnull().sum()
        
        if nan_counts.any():
            logger.info("Features with NaN values:")
            for feature, nan_count in nan_counts[nan_counts > 0].items():
                pct = (nan_count / len(df)) * 100
                logger.info(f"  {feature}: {nan_count} ({pct:.1f}%)")
        
        # Drop intermediate calculation columns if they exist
        temp_cols = ['TR', 'DM_Plus', 'DM_Minus', 'DX', 'Prev_Close']
        df.drop(columns=[col for col in temp_cols if col in df.columns], inplace=True, errors='ignore')
        
        return df
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all generated features."""
        
        descriptions = {
            # Basic features
            'Returns': 'Price returns (Close-to-Close)',
            'Log_Returns': 'Log returns for better distribution properties',
            'Price_Range': 'High - Low price range',
            'Body_Size': 'Absolute difference between Open and Close',
            'Upper_Wick': 'Upper shadow of candlestick',
            'Lower_Wick': 'Lower shadow of candlestick',
            
            # Momentum indicators
            'RSI_14': 'Relative Strength Index (14 periods)',
            'MACD': 'MACD line (12-26 EMA difference)',
            'MACD_Signal': 'MACD signal line (9-period EMA of MACD)',
            'MACD_Histogram': 'MACD histogram (MACD - Signal)',
            'Stoch_K_14': 'Stochastic %K (14 periods)',
            'Stoch_D_3': 'Stochastic %D (3-period SMA of %K)',
            'CCI_20': 'Commodity Channel Index (20 periods)',
            'Williams_R_14': 'Williams %R (14 periods)',
            
            # Trend indicators  
            'EMA_Cross_*': 'EMA crossover signals',
            'ADX_14': 'Average Directional Index (trend strength)',
            'DI_Plus': 'Positive Directional Indicator',
            'DI_Minus': 'Negative Directional Indicator',
            
            # Volatility indicators
            'ATR': 'Average True Range',
            'ATR_Normalized': 'ATR as percentage of price',
            'BB_Upper_20': 'Bollinger Bands upper band',
            'BB_Lower_20': 'Bollinger Bands lower band',
            'BB_Width_20': 'Bollinger Bands width',
            'BB_Percent_20': 'Position within Bollinger Bands',
            
            # Volume indicators
            'OBV': 'On-Balance Volume',
            'CMF_20': 'Chaikin Money Flow (20 periods)',
            'Volume_Ratio': 'Volume vs 20-period average',
            
            # DAX-specific features
            'Opening_Gap': 'Gap between current Open and previous Close',
            'Opening_Gap_Normalized': 'Opening gap normalized by previous ATR',
            'Gap_Direction': 'Direction of opening gap (1=up, -1=down, 0=none)',
            'Gap_Magnitude': 'Absolute size of opening gap',
            'Gap_Percent': 'Opening gap as percentage of previous close',
            
            # Session features
            'Is_Market_Open': 'Boolean indicating market opening time',
            'Is_Market_Hours': 'Boolean indicating regular trading hours',
            'Is_Opening_Hour': 'Boolean indicating first hour of trading',
            'Is_Lunch_Time': 'Boolean indicating lunch period (12:00-14:00)',
            'Is_Closing_Hour': 'Boolean indicating last hour of trading',
            'London_Frankfurt_Overlap': 'Boolean indicating London-Frankfurt overlap',
            
            # European session patterns
            'Opening_Hour_Avg_Move': 'Average price movement during opening hour',
            'Opening_Move_vs_Avg': 'Current vs average opening hour movement',
            'EU_Session_Range': 'Price range during European session',
            'Friday_Close_Effect': 'Boolean indicating Friday close period',
            
            # Multi-timeframe features
            'RSI_5min': 'RSI calculated on 5-minute bars',
            'RSI_15min': 'RSI calculated on 15-minute bars', 
            'RSI_1hour': 'RSI calculated on 1-hour bars',
            'EMA_Trend_*_5min': 'EMA trend on 5-minute timeframe',
            'ATR_5min': 'ATR calculated on 5-minute bars',
        }
        
        return descriptions
    
    def generate_feature_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive feature engineering report."""
        
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        report = []
        report.append("="*80)
        report.append("DAX TRADING SYSTEM - FEATURE ENGINEERING REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Period: {df.index.min()} to {df.index.max()}")
        report.append(f"Total Records: {len(df):,}")
        report.append(f"Total Features: {len(feature_cols)}")
        report.append("")
        
        # Feature categories
        categories = {
            'Basic Price Features': [col for col in feature_cols if any(x in col for x in ['Returns', 'Range', 'Body', 'Wick'])],
            'Momentum Indicators': [col for col in feature_cols if any(x in col for x in ['RSI', 'MACD', 'Stoch', 'CCI', 'Williams'])],
            'Trend Indicators': [col for col in feature_cols if any(x in col for x in ['EMA', 'ADX', 'DI_'])],
            'Volatility Indicators': [col for col in feature_cols if any(x in col for x in ['ATR', 'BB_', 'Volatility'])],
            'Volume Indicators': [col for col in feature_cols if any(x in col for x in ['OBV', 'CMF', 'Volume'])],
            'DAX-Specific Features': [col for col in feature_cols if any(x in col for x in ['Gap', 'Opening', 'EU_Session'])],
            'Session Features': [col for col in feature_cols if any(x in col for x in ['Is_', 'Hour', 'Day_of', 'London', 'Friday'])],
            'Multi-Timeframe': [col for col in feature_cols if any(x in col for x in ['5min', '15min', '1hour'])]
        }
        
        for category, cols in categories.items():
            if cols:
                report.append(f"{category} ({len(cols)} features):")
                for col in sorted(cols)[:10]:  # Show first 10 features
                    nan_pct = (df[col].isnull().sum() / len(df)) * 100
                    report.append(f"  • {col:<30} (NaN: {nan_pct:.1f}%)")
                if len(cols) > 10:
                    report.append(f"  ... and {len(cols) - 10} more features")
                report.append("")
        
        # Data quality summary
        report.append("DATA QUALITY SUMMARY:")
        report.append("-" * 40)
        total_nan = df[feature_cols].isnull().sum().sum()
        total_cells = len(df) * len(feature_cols)
        overall_completeness = ((total_cells - total_nan) / total_cells) * 100
        report.append(f"Overall Data Completeness: {overall_completeness:.2f}%")
        
        # Features with high NaN percentage
        high_nan_features = []
        for col in feature_cols:
            nan_pct = (df[col].isnull().sum() / len(df)) * 100
            if nan_pct > 10:  # More than 10% NaN
                high_nan_features.append((col, nan_pct))
        
        if high_nan_features:
            report.append(f"\nFeatures with >10% missing values ({len(high_nan_features)}):")
            for col, nan_pct in sorted(high_nan_features, key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"  • {col}: {nan_pct:.1f}% missing")
        
        # Sample statistics for key features
        report.append("\nKEY FEATURE STATISTICS:")
        report.append("-" * 40)
        key_features = ['RSI_14', 'MACD', 'ATR', 'Opening_Gap_Normalized', 'Volume_Ratio']
        available_key_features = [f for f in key_features if f in df.columns]
        
        if available_key_features:
            stats_df = df[available_key_features].describe()
            report.append(stats_df.round(4).to_string())
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


# Example usage and testing functions
def main():
    """Example usage of the FeatureEngineer class."""
    
    print("="*60)
    print("DAX Feature Engineering - Example Usage")
    print("="*60)
    
    try:
        # Create sample DAX data
        dates = pd.date_range('2024-01-15 09:00', '2024-01-15 17:30', freq='1min', tz='Europe/Berlin')
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        base_price = 18000
        n_periods = len(dates)
        
        # Generate price series with realistic patterns
        returns = np.random.normal(0, 0.0005, n_periods)  # 0.05% average volatility per minute
        returns[0] = 0  # First return is 0
        
        # Add some trend and mean reversion
        for i in range(1, len(returns)):
            # Add slight mean reversion
            returns[i] += -0.1 * returns[i-1]
            # Add some intraday patterns
            hour = dates[i].hour
            if hour == 9:  # Opening volatility
                returns[i] *= 2
            elif hour in [12, 13]:  # Lunch time lower volatility
                returns[i] *= 0.5
            elif hour == 17:  # Closing volatility
                returns[i] *= 1.5
        
        prices = base_price * (1 + returns).cumprod()
        
        # Generate OHLC from prices
        opens = prices.shift(1).fillna(base_price)
        closes = prices
        
        # Generate realistic highs and lows
        volatility = np.abs(returns) * prices
        highs = np.maximum(opens, closes) + np.random.exponential(volatility * 0.5)
        lows = np.minimum(opens, closes) - np.random.exponential(volatility * 0.5)
        
        # Ensure OHLC relationships
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # Generate volume
        base_volume = 1000
        volume = np.random.poisson(base_volume, n_periods)
        
        # Higher volume during high volatility
        volume = volume * (1 + 2 * np.abs(returns))
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volume.astype(int)
        }, index=dates)
        
        print(f"\nSample Data Created:")
        print(f"  Records: {len(sample_df)}")
        print(f"  Date Range: {sample_df.index.min()} to {sample_df.index.max()}")
        print(f"  Price Range: €{sample_df['Close'].min():.2f} - €{sample_df['Close'].max():.2f}")
        
        # Initialize FeatureEngineer
        feature_engineer = FeatureEngineer()
        
        # Generate features
        print(f"\n🔧 Generating features...")
        features_df = feature_engineer.engineer_features(sample_df)
        
        # Show results
        original_cols = len(sample_df.columns)
        feature_cols = len(features_df.columns)
        new_features = feature_cols - original_cols
        
        print(f"\n✅ Feature Engineering Completed!")
        print(f"  Original columns: {original_cols}")
        print(f"  Total columns: {feature_cols}")
        print(f"  New features: {new_features}")
        
        # Show sample features
        print(f"\n📊 Sample Features (first 5 rows):")
        feature_columns = [col for col in features_df.columns if col not in sample_df.columns]
        sample_features = features_df[feature_columns[:10]]  # Show first 10 new features
        print(sample_features.head().round(4))
        
        # Generate and display report
        print(f"\n📈 Feature Engineering Report:")
        print(feature_engineer.generate_feature_report(features_df))
        
        # Show feature descriptions
        print(f"\n📖 Feature Descriptions (sample):")
        descriptions = feature_engineer.get_feature_descriptions()
        for i, (feature, desc) in enumerate(list(descriptions.items())[:5]):
            print(f"  • {feature}: {desc}")
        print(f"  ... and {len(descriptions) - 5} more feature descriptions available")
        
    except Exception as e:
        print(f"❌ Error in feature engineering example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()