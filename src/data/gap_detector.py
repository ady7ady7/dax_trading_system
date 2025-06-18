"""
DAX Trading System - Gap Detection Module

This module detects and classifies gaps in 1-minute OHLCV time series data,
distinguishing between expected market closures and actual data provider issues.

File Location: src/data/gap_detector.py

Author: DAX Trading System
Created: 2025-06-18
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Dict, Tuple, Optional
import pytz
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Gap:
    """Data class representing a detected gap in time series data."""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_minutes: int
    gap_type: str  # 'market_closure' or 'data_provider_issue'
    severity: str  # 'minor', 'moderate', 'major'
    severity_score: int  # 1=minor, 2=moderate, 3=major


class GapDetector:
    """
    Detects and classifies gaps in 1-minute DAX OHLCV time series data.
    
    Distinguishes between expected market closures and data provider issues,
    providing detailed gap analysis with severity scoring.
    """
    
    def __init__(self, timezone: str = "Europe/Berlin"):
        """
        Initialize the Gap Detector.
        
        Args:
            timezone (str): Timezone for market hours (default: Europe/Berlin)
        """
        self.timezone = pytz.timezone(timezone)
        
        # DAX market hours (09:00 - 17:30 CET/CEST)
        self.market_open = time(9, 0)
        self.market_close = time(17, 30)
        
        # German public holidays (major ones that affect DAX)
        self.german_holidays = self._get_german_holidays()
        
        logger.info(f"GapDetector initialized for timezone: {timezone}")
    
    def detect_gaps(self, df: pd.DataFrame, expected_frequency: str = "1min") -> List[Gap]:
        """
        Detect all gaps in the time series data.
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index
            expected_frequency (str): Expected data frequency (default: "1min")
            
        Returns:
            List[Gap]: List of detected gaps with classifications
        """
        logger.info(f"Starting gap detection for {len(df)} data points")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return []
            
        # Ensure the index is timezone-aware
        if df.index.tz is None:
            logger.warning("DataFrame index is timezone-naive, localizing to specified timezone")
            df.index = df.index.tz_localize(self.timezone)
        elif df.index.tz != self.timezone:
            logger.info(f"Converting timezone from {df.index.tz} to {self.timezone}")
            df.index = df.index.tz_convert(self.timezone)
        
        # Generate expected time range
        start_time = df.index.min()
        end_time = df.index.max()
        expected_times = pd.date_range(
            start=start_time,
            end=end_time,
            freq=expected_frequency,
            tz=self.timezone
        )
        
        logger.info(f"Expected {len(expected_times)} data points from {start_time} to {end_time}")
        
        # Find missing timestamps
        missing_times = expected_times.difference(df.index)
        
        if len(missing_times) == 0:
            logger.info("No gaps detected - perfect data continuity")
            return []
            
        logger.info(f"Found {len(missing_times)} missing timestamps")
        
        # Group consecutive missing times into gaps
        gaps = self._group_consecutive_gaps(missing_times)
        
        # Classify each gap
        classified_gaps = []
        for gap_start, gap_end in gaps:
            gap = self._classify_gap(gap_start, gap_end)
            classified_gaps.append(gap)
            
        logger.info(f"Detected {len(classified_gaps)} gaps total")
        self._log_gap_summary(classified_gaps)
        
        return classified_gaps
    
    def _group_consecutive_gaps(self, missing_times: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Group consecutive missing timestamps into gap periods."""
        if len(missing_times) == 0:
            return []
            
        # Sort missing times
        missing_times = missing_times.sort_values()
        
        gaps = []
        gap_start = missing_times[0]
        gap_end = missing_times[0]
        
        for i in range(1, len(missing_times)):
            current_time = missing_times[i]
            expected_next = gap_end + pd.Timedelta(minutes=1)
            
            if current_time == expected_next:
                # Consecutive gap continues
                gap_end = current_time
            else:
                # Gap ended, start new gap
                gaps.append((gap_start, gap_end))
                gap_start = current_time
                gap_end = current_time
        
        # Add the last gap
        gaps.append((gap_start, gap_end))
        
        return gaps
    
    def _classify_gap(self, gap_start: pd.Timestamp, gap_end: pd.Timestamp) -> Gap:
        """Classify a gap as market closure or data provider issue."""
        duration_minutes = int((gap_end - gap_start).total_seconds() / 60) + 1
        
        # Check if gap occurs during trading hours
        is_trading_hours_gap = self._is_during_trading_hours(gap_start, gap_end)
        
        # Determine gap type
        if is_trading_hours_gap:
            gap_type = "data_provider_issue"
        else:
            gap_type = "market_closure"
        
        # Determine severity
        severity, severity_score = self._calculate_severity(duration_minutes, gap_type)
        
        return Gap(
            start_time=gap_start,
            end_time=gap_end,
            duration_minutes=duration_minutes,
            gap_type=gap_type,
            severity=severity,
            severity_score=severity_score
        )
    
    def _is_during_trading_hours(self, gap_start: pd.Timestamp, gap_end: pd.Timestamp) -> bool:
        """Check if any part of the gap occurs during trading hours."""
        
        # Check each day in the gap period
        current_day = gap_start.date()
        end_day = gap_end.date()
        
        while current_day <= end_day:
            # Skip weekends
            if current_day.weekday() >= 5:  # Saturday=5, Sunday=6
                current_day += timedelta(days=1)
                continue
                
            # Skip German holidays
            if self._is_german_holiday(current_day):
                current_day += timedelta(days=1)
                continue
            
            # Check if any part of gap overlaps with trading hours on this day
            market_open_dt = datetime.combine(current_day, self.market_open)
            market_close_dt = datetime.combine(current_day, self.market_close)
            
            # Localize to timezone
            market_open_dt = self.timezone.localize(market_open_dt)
            market_close_dt = self.timezone.localize(market_close_dt)
            
            # Check for overlap
            gap_start_day = max(gap_start, market_open_dt)
            gap_end_day = min(gap_end, market_close_dt)
            
            if gap_start_day <= gap_end_day:
                return True  # Found overlap with trading hours
                
            current_day += timedelta(days=1)
        
        return False
    
    def _calculate_severity(self, duration_minutes: int, gap_type: str) -> Tuple[str, int]:
        """Calculate gap severity based on duration and type."""
        
        if gap_type == "data_provider_issue":
            # More strict for data provider issues
            if duration_minutes < 5:
                return "minor", 1
            elif duration_minutes <= 30:
                return "moderate", 2
            else:
                return "major", 3
        else:
            # More lenient for market closures
            if duration_minutes < 60:  # Less than 1 hour
                return "minor", 1
            elif duration_minutes <= 1440:  # Less than 1 day
                return "moderate", 2
            else:
                return "major", 3
    
    def _is_german_holiday(self, date: datetime.date) -> bool:
        """Check if a date is a German public holiday."""
        return date.strftime("%m-%d") in self.german_holidays
    
    def _get_german_holidays(self) -> set:
        """Get major German public holidays (month-day format)."""
        return {
            "01-01",  # New Year's Day
            "05-01",  # Labour Day
            "10-03",  # German Unity Day
            "12-25",  # Christmas Day
            "12-26",  # Boxing Day
            # Note: Easter dates vary by year - could be enhanced with proper calculation
        }
    
    def _log_gap_summary(self, gaps: List[Gap]) -> None:
        """Log a summary of detected gaps."""
        if not gaps:
            return
            
        # Count by type
        market_closures = sum(1 for g in gaps if g.gap_type == "market_closure")
        data_issues = sum(1 for g in gaps if g.gap_type == "data_provider_issue")
        
        # Count by severity
        minor = sum(1 for g in gaps if g.severity == "minor")
        moderate = sum(1 for g in gaps if g.severity == "moderate")
        major = sum(1 for g in gaps if g.severity == "major")
        
        # Total missing time
        total_minutes = sum(g.duration_minutes for g in gaps)
        
        logger.info("Gap Detection Summary:")
        logger.info(f"  Market Closures: {market_closures}")
        logger.info(f"  Data Provider Issues: {data_issues}")
        logger.info(f"  Severity - Minor: {minor}, Moderate: {moderate}, Major: {major}")
        logger.info(f"  Total Missing Time: {total_minutes:,} minutes ({total_minutes/1440:.1f} days)")
    
    def gaps_to_dataframe(self, gaps: List[Gap]) -> pd.DataFrame:
        """Convert gaps list to pandas DataFrame for analysis."""
        if not gaps:
            return pd.DataFrame()
            
        data = []
        for gap in gaps:
            data.append({
                'start_time': gap.start_time,
                'end_time': gap.end_time,
                'duration_minutes': gap.duration_minutes,
                'duration_hours': gap.duration_minutes / 60,
                'gap_type': gap.gap_type,
                'severity': gap.severity,
                'severity_score': gap.severity_score
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('start_time').reset_index(drop=True)
    
    def get_data_quality_metrics(self, df: pd.DataFrame, gaps: List[Gap]) -> Dict[str, float]:
        """Calculate overall data quality metrics."""
        if df.empty:
            return {}
        
        total_expected_points = len(pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq="1min",
            tz=self.timezone
        ))
        
        actual_points = len(df)
        missing_points = sum(g.duration_minutes for g in gaps)
        data_completeness = (actual_points / total_expected_points) * 100
        
        critical_gaps = sum(1 for g in gaps if g.gap_type == "data_provider_issue" and g.severity == "major")
        
        return {
            'data_completeness_pct': round(data_completeness, 2),
            'total_gaps': len(gaps),
            'critical_data_gaps': critical_gaps,
            'total_missing_minutes': missing_points,
            'avg_gap_duration_minutes': round(missing_points / len(gaps), 2) if gaps else 0
        }


# Example usage and testing
def main():
    """Example usage of the GapDetector."""
    
    # Example with sample data
    detector = GapDetector()
    
    # Create sample data with intentional gaps
    start_date = pd.Timestamp("2024-01-15 09:00:00", tz="Europe/Berlin")
    end_date = pd.Timestamp("2024-01-15 17:30:00", tz="Europe/Berlin")
    
    # Create complete time series
    full_range = pd.date_range(start=start_date, end=end_date, freq="1min")
    
    # Remove some data points to create gaps
    # Remove 10 minutes during trading hours (data provider issue)
    missing_trading = pd.date_range(start="2024-01-15 11:00:00", end="2024-01-15 11:10:00", 
                                   freq="1min", tz="Europe/Berlin")
    
    # Create sample data
    available_times = full_range.difference(missing_trading)
    sample_df = pd.DataFrame({
        'Close': np.random.normal(18000, 100, len(available_times))
    }, index=available_times)
    
    print("\n" + "="*60)
    print("Gap Detection - Example Results")
    print("="*60)
    
    # Detect gaps
    gaps = detector.detect_gaps(sample_df)
    
    # Convert to DataFrame for display
    gaps_df = detector.gaps_to_dataframe(gaps)
    print("\nDetected Gaps:")
    print(gaps_df)
    
    # Get quality metrics
    metrics = detector.get_data_quality_metrics(sample_df, gaps)
    print("\nData Quality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()



    '''
    Key Features of the GapDetector:
1. Smart Gap Classification:

Market Closures: Weekends, holidays, outside 09:00-17:30 CET
Data Provider Issues: Missing data during expected trading hours

2. Severity Scoring:

Minor: < 5 minutes (data issues) or < 1 hour (market closures)
Moderate: 5-30 minutes (data issues) or 1-24 hours (market closures)
Major: > 30 minutes (data issues) or > 1 day (market closures)

3. Comprehensive Analysis:

Gap grouping: Consecutive missing minutes grouped together
Quality metrics: Data completeness percentage, critical gaps count
DataFrame output: Easy analysis and visualization

4. DAX-Specific:

German timezone (Europe/Berlin)
DAX market hours (09:00-17:30)
German holidays recognition
    '''