#!/usr/bin/env python3
"""
DAX Trading System - Statistical Feature Validation Module

Standalone statistical validation framework for rigorous feature selection.
Implements comprehensive hypothesis testing, multiple testing correction,
and walk-forward validation for trading feature validation.

Author: DAX Trading System
Created: 2025-06-19
File: src/validation/statistical_validator.py
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, time
import warnings
from pathlib import Path

# Statistical testing imports
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, pearsonr, spearmanr, 
    chi2_contingency, ks_2samp, jarque_bera
)
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, kpss

# Handle sklearn imports with version compatibility
try:
    from sklearn.feature_selection import mutual_info_regression, f_regression, chi2
except ImportError:
    try:
        from sklearn.feature_selection import f_regression, chi2
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        # Fallback for older sklearn versions
        from sklearn.feature_selection import f_regression, chi2
        mutual_info_regression = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class StatisticalFeatureValidator:
    """
    Comprehensive statistical validation framework for trading features.
    
    This class provides rigorous statistical testing capabilities to identify
    features with genuine predictive power while controlling for multiple testing
    and ensuring temporal stability.
    
    Key Features:
    - Multiple hypothesis testing (6 different statistical tests)
    - Multiple testing correction (Bonferroni, FDR control)
    - Walk-forward temporal validation
    - Effect size requirements
    - Comprehensive reporting
    
    Usage:
        validator = StatisticalFeatureValidator()
        results = validator.validate_features(features_df, target_variable)
        validated_features = results['significant_features']
    """
    
    def __init__(self, 
                 min_window_size: int = 1000,
                 significance_level: float = 0.01,
                 min_effect_size: float = 0.3,
                 correction_method: str = 'fdr_bh',
                 min_temporal_consistency: float = 0.6):
        """
        Initialize the statistical validator.
        
        Args:
            min_window_size: Minimum observations per test window
            significance_level: Alpha level for hypothesis tests (p-value threshold)
            min_effect_size: Minimum effect size threshold (correlation)
            correction_method: Multiple testing correction method
                             ('bonferroni', 'fdr_bh', 'fdr_by')
            min_temporal_consistency: Minimum temporal consistency score (0-1)
        """
        self.min_window_size = min_window_size
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size
        self.correction_method = correction_method
        self.min_temporal_consistency = min_temporal_consistency
        
        # Statistical test configuration
        self.test_methods = {
            'parametric': ['pearson', 't_test', 'f_test'],
            'non_parametric': ['spearman', 'mannwhitney', 'ks_test'],
            'robust': ['mutual_info']
        }
        
        logger.info(f"StatisticalFeatureValidator initialized:")
        logger.info(f"  Min window size: {min_window_size}")
        logger.info(f"  Significance level: {significance_level}")
        logger.info(f"  Min effect size: {min_effect_size}")
        logger.info(f"  Correction method: {correction_method}")
        logger.info(f"  Min temporal consistency: {min_temporal_consistency}")
    
    
    def validate_features(self, 
                         features_df: pd.DataFrame, 
                         target_variable: pd.Series = None,
                         target_method: str = 'price_direction',
                         lookforward_periods: int = 5,
                         walk_forward_steps: int = 5) -> Dict:
        """
        Main feature validation method.
        
        Performs comprehensive statistical validation including screening,
        hypothesis testing, multiple testing correction, and temporal validation.
        
        Args:
            features_df: DataFrame containing engineered features
            target_variable: Binary target variable (optional, will be created if None)
            target_method: Method for target creation if target_variable is None
            lookforward_periods: Periods to look forward for target creation
            walk_forward_steps: Number of walk-forward validation windows
            
        Returns:
            Dictionary containing validation results:
            - significant_features: Validated features with statistics
            - validation_summary: Overall validation metrics
            - test_results: Detailed test results for each feature
            - walk_forward_results: Temporal validation results
            - screening_results: Initial screening results
        """
        
        logger.info("="*70)
        logger.info("STATISTICAL FEATURE VALIDATION")
        logger.info("="*70)
        
        start_time = datetime.now()
        
        # Input validation
        if not self._validate_inputs(features_df, target_variable):
            return self._create_error_result("Input validation failed")
        
        # Create target variable if not provided
        if target_variable is None:
            logger.info("Creating target variable...")
            target_variable = self.create_target_variable(
                features_df, target_method, lookforward_periods
            )
        
        # Ensure binary target
        target_binary = self._ensure_binary_target(target_variable)
        
        # Get feature columns (exclude OHLCV)
        feature_cols = self._get_feature_columns(features_df)
        
        logger.info(f"Validating {len(feature_cols)} features")
        logger.info(f"Dataset size: {len(features_df):,} observations")
        logger.info(f"Target distribution: {target_binary.value_counts().to_dict()}")
        
        # STEP 1: Feature Screening
        logger.info("Step 1: Feature screening...")
        screening_results = self._screen_features(features_df[feature_cols], target_binary)
        viable_features = screening_results['viable_features']
        
        logger.info(f"Features passing screening: {len(viable_features)}/{len(feature_cols)}")
        
        # STEP 2: Statistical Testing
        logger.info("Step 2: Statistical hypothesis testing...")
        test_results = self._perform_statistical_tests(
            features_df[viable_features], target_binary
        )
        
        logger.info(f"Completed testing on {len(test_results)} features")
        
        # STEP 3: Multiple Testing Correction
        logger.info("Step 3: Multiple testing correction...")
        correction_results = self._apply_multiple_testing_correction(test_results)
        
        # STEP 4: Walk-Forward Validation
        logger.info("Step 4: Walk-forward temporal validation...")
        walk_forward_results = self._perform_walk_forward_validation(
            features_df, target_binary, correction_results, walk_forward_steps
        )
        
        # STEP 5: Final Feature Selection
        logger.info("Step 5: Final feature selection...")
        significant_features = self._select_final_features(
            correction_results, walk_forward_results
        )
        
        # Compile results
        processing_time = (datetime.now() - start_time).total_seconds()
        
        validation_summary = self._create_validation_summary(
            feature_cols, viable_features, test_results, correction_results,
            significant_features, processing_time
        )
        
        logger.info(f"Validation completed in {processing_time:.2f} seconds")
        logger.info(f"Validated features: {len(significant_features)}/{len(feature_cols)}")
        
        return {
            'significant_features': significant_features,
            'validation_summary': validation_summary,
            'test_results': test_results,
            'walk_forward_results': walk_forward_results,
            'screening_results': screening_results,
            'correction_results': correction_results
        }
    
    
    def create_target_variable(self, 
                             features_df: pd.DataFrame,
                             method: str = 'price_direction',
                             lookforward_periods: int = 5,
                             threshold: float = 0.001) -> pd.Series:
        """
        Create target variable from feature data.
        
        Args:
            features_df: DataFrame with OHLCV data
            method: Target creation method
                   - 'price_direction': Future price direction (up/down)
                   - 'significant_move': Significant price moves above threshold
                   - 'volatility_breakout': High volatility periods
            lookforward_periods: Number of periods to look forward
            threshold: Threshold for significant moves (as decimal)
            
        Returns:
            Binary target variable (0/1)
        """
        
        if 'Close' not in features_df.columns:
            raise ValueError("'Close' column required for target variable creation")
        
        logger.info(f"Creating target variable: {method}")
        logger.info(f"Lookforward periods: {lookforward_periods}")
        
        close_prices = features_df['Close']
        
        if method == 'price_direction':
            # Future price direction
            future_prices = close_prices.shift(-lookforward_periods)
            target = (future_prices > close_prices).astype(int)
            
        elif method == 'significant_move':
            # Significant price movements
            future_prices = close_prices.shift(-lookforward_periods)
            price_change = (future_prices - close_prices) / close_prices
            target = (abs(price_change) > threshold).astype(int)
            
        elif method == 'volatility_breakout':
            # High volatility periods
            returns = close_prices.pct_change()
            rolling_vol = returns.rolling(20).std()
            vol_threshold = rolling_vol.quantile(0.8)
            target = (rolling_vol > vol_threshold).astype(int)
            
        else:
            raise ValueError(f"Unknown target creation method: {method}")
        
        target = target.dropna()
        
        logger.info(f"Target variable created: {len(target)} observations")
        logger.info(f"Class distribution: {target.value_counts().to_dict()}")
        
        return target
    
    
    def generate_validation_report(self, 
                                 validation_results: Dict,
                                 output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_results: Results from validate_features()
            output_path: Optional path to save report
            
        Returns:
            Formatted validation report string
        """
        
        if 'error' in validation_results:
            return f"❌ Validation Error: {validation_results['error']}"
        
        significant_features = validation_results['significant_features']
        summary = validation_results['validation_summary']
        
        # Generate report content
        report = self._generate_report_content(significant_features, summary)
        
        # Save if path provided
        if output_path:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Validation report saved: {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report
    
    
    def filter_features_to_validated(self, 
                                   features_df: pd.DataFrame,
                                   validation_results: Dict) -> pd.DataFrame:
        """
        Filter DataFrame to only include validated features.
        
        Args:
            features_df: Original features DataFrame
            validation_results: Results from validate_features()
            
        Returns:
            DataFrame with only validated features + original OHLCV
        """
        
        if 'error' in validation_results:
            logger.warning("Validation failed - returning original DataFrame")
            return features_df
        
        significant_features = validation_results['significant_features']
        
        if not significant_features:
            logger.warning("No validated features - returning only OHLCV columns")
            original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            return features_df[original_cols]
        
        # Keep original OHLCV + validated features
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        validated_feature_names = list(significant_features.keys())
        
        # Ensure all columns exist
        available_cols = [col for col in original_cols + validated_feature_names 
                         if col in features_df.columns]
        
        filtered_df = features_df[available_cols].copy()
        
        logger.info(f"Filtered to {len(validated_feature_names)} validated features")
        logger.info(f"Final DataFrame: {len(filtered_df.columns)} columns")
        
        return filtered_df
    
    
    # Private helper methods
    
    def _validate_inputs(self, features_df: pd.DataFrame, target_variable: pd.Series) -> bool:
        """Validate input data."""
        
        if not isinstance(features_df, pd.DataFrame):
            logger.error("features_df must be pandas DataFrame")
            return False
        
        if len(features_df) < self.min_window_size:
            logger.error(f"Dataset too small: {len(features_df)} < {self.min_window_size}")
            return False
        
        if target_variable is not None:
            if not isinstance(target_variable, pd.Series):
                logger.error("target_variable must be pandas Series")
                return False
            
            common_index = features_df.index.intersection(target_variable.index)
            if len(common_index) < self.min_window_size:
                logger.error("Insufficient aligned observations")
                return False
        
        return True
    
    
    def _ensure_binary_target(self, target_variable: pd.Series) -> pd.Series:
        """Ensure target variable is binary."""
        
        if target_variable.nunique() > 2:
            logger.warning("Converting non-binary target to binary using median split")
            return (target_variable > target_variable.median()).astype(int)
        
        return target_variable.astype(int)
    
    
    def _get_feature_columns(self, features_df: pd.DataFrame) -> List[str]:
        """Get feature columns (exclude OHLCV)."""
        
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        return [col for col in features_df.columns if col not in original_cols]
    
    
    def _screen_features(self, features_df: pd.DataFrame, target: pd.Series) -> Dict:
        """Screen features for basic quality requirements."""
        
        viable_features = []
        screening_stats = {}
        
        for feature in features_df.columns:
            feature_data = features_df[feature].dropna()
            
            # Basic requirements
            if len(feature_data) < 100:
                continue
            
            if feature_data.nunique() <= 1:
                continue
            
            # Check for quasi-constant features
            mode_frequency = feature_data.value_counts().iloc[0] / len(feature_data)
            if mode_frequency > 0.98:
                continue
            
            # Quick correlation check
            aligned_target = target.loc[feature_data.index]
            if len(aligned_target) < 100:
                continue
            
            try:
                quick_corr = abs(feature_data.corr(aligned_target))
                if pd.isna(quick_corr) or quick_corr < 0.005:
                    continue
            except:
                continue
            
            viable_features.append(feature)
            screening_stats[feature] = {
                'observations': len(feature_data),
                'unique_values': feature_data.nunique(),
                'missing_rate': (len(features_df) - len(feature_data)) / len(features_df),
                'quick_correlation': quick_corr
            }
        
        return {
            'viable_features': viable_features,
            'screening_stats': screening_stats
        }
    
    
    def _perform_statistical_tests(self, features_df: pd.DataFrame, target: pd.Series) -> Dict:
        """Perform battery of statistical tests on each feature."""
        
        test_results = {}
        
        for feature in features_df.columns:
            feature_data = features_df[feature].dropna()
            aligned_target = target.loc[feature_data.index]
            
            if len(feature_data) < self.min_window_size:
                continue
            
            # Align data and remove NaN
            common_idx = feature_data.index.intersection(aligned_target.index)
            X = feature_data.loc[common_idx].values
            y = aligned_target.loc[common_idx].values
            
            valid_mask = ~(pd.isna(X) | pd.isna(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:
                continue
            
            # Perform statistical tests
            test_result = self._run_statistical_battery(X, y, feature)
            test_results[feature] = test_result
        
        return test_results
    
    
    def _run_statistical_battery(self, X: np.ndarray, y: np.ndarray, feature_name: str) -> Dict:
        """Run comprehensive battery of statistical tests."""
        
        results = {
            'feature_name': feature_name,
            'n_observations': len(X)
        }
        
        try:
            # Test 1: Pearson correlation
            try:
                pearson_r, pearson_p = pearsonr(X, y)
                results['pearson_correlation'] = pearson_r
                results['pearson_p_value'] = pearson_p
            except:
                results['pearson_correlation'] = np.nan
                results['pearson_p_value'] = 1.0
            
            # Test 2: Spearman correlation
            try:
                spearman_r, spearman_p = spearmanr(X, y)
                results['spearman_correlation'] = spearman_r
                results['spearman_p_value'] = spearman_p
            except:
                results['spearman_correlation'] = np.nan
                results['spearman_p_value'] = 1.0
            
            # Test 3: T-test
            try:
                group_0 = X[y == 0]
                group_1 = X[y == 1]
                
                if len(group_0) >= 10 and len(group_1) >= 10:
                    t_stat, t_p = ttest_ind(group_1, group_0, equal_var=False)
                    results['t_statistic'] = t_stat
                    results['t_test_p_value'] = t_p
                else:
                    results['t_statistic'] = np.nan
                    results['t_test_p_value'] = 1.0
            except:
                results['t_statistic'] = np.nan
                results['t_test_p_value'] = 1.0
            
            # Test 4: Mann-Whitney U
            try:
                group_0 = X[y == 0]
                group_1 = X[y == 1]
                
                if len(group_0) >= 10 and len(group_1) >= 10:
                    mw_stat, mw_p = mannwhitneyu(group_1, group_0, alternative='two-sided')
                    results['mannwhitney_statistic'] = mw_stat
                    results['mannwhitney_p_value'] = mw_p
                else:
                    results['mannwhitney_statistic'] = np.nan
                    results['mannwhitney_p_value'] = 1.0
            except:
                results['mannwhitney_statistic'] = np.nan
                results['mannwhitney_p_value'] = 1.0
            
            # Test 5: F-statistic
            try:
                X_reshaped = X.reshape(-1, 1)
                f_stats, f_p_values = f_regression(X_reshaped, y)
                results['f_statistic'] = f_stats[0]
                results['f_test_p_value'] = f_p_values[0]
            except:
                results['f_statistic'] = np.nan
                results['f_test_p_value'] = 1.0
            
            # Test 6: Mutual Information
            try:
                X_reshaped = X.reshape(-1, 1)
                mi_scores = mutual_info_regression(X_reshaped, y, random_state=42)
                results['mutual_info_score'] = mi_scores[0]
            except:
                results['mutual_info_score'] = np.nan
            
            # Determine minimum p-value
            p_values = [
                results.get('pearson_p_value', 1.0),
                results.get('spearman_p_value', 1.0),
                results.get('t_test_p_value', 1.0),
                results.get('mannwhitney_p_value', 1.0),
                results.get('f_test_p_value', 1.0)
            ]
            
            valid_p_values = [p for p in p_values if not pd.isna(p)]
            results['raw_p_value'] = min(valid_p_values) if valid_p_values else 1.0
            
        except Exception as e:
            results['error'] = str(e)
            results['raw_p_value'] = 1.0
        
        return results
    
    
    def _apply_multiple_testing_correction(self, test_results: Dict) -> Dict:
        """Apply multiple testing correction."""
        
        features = []
        p_values = []
        
        for feature, result in test_results.items():
            if 'raw_p_value' in result:
                features.append(feature)
                p_values.append(result['raw_p_value'])
        
        if not p_values:
            return {'corrected_results': {}, 'significant_before_correction': []}
        
        try:
            rejected, p_corrected, _, _ = multipletests(
                p_values, alpha=self.significance_level, method=self.correction_method
            )
            
            corrected_results = {}
            significant_before = []
            
            for i, feature in enumerate(features):
                if p_values[i] < self.significance_level:
                    significant_before.append(feature)
                
                corrected_results[feature] = {
                    **test_results[feature],
                    'corrected_p_value': p_corrected[i],
                    'significant_after_correction': rejected[i]
                }
            
            return {
                'corrected_results': corrected_results,
                'significant_before_correction': significant_before
            }
            
        except Exception as e:
            logger.error(f"Multiple testing correction failed: {e}")
            return {'corrected_results': {}, 'significant_before_correction': []}
    
    
    def _perform_walk_forward_validation(self, features_df: pd.DataFrame,
                                       target: pd.Series, correction_results: Dict,
                                       n_steps: int) -> Dict:
        """Perform walk-forward validation for temporal stability."""
        
        walk_forward_results = {}
        
        candidates = [f for f, r in correction_results['corrected_results'].items() 
                     if r['significant_after_correction']]
        
        for feature in candidates:
            feature_data = features_df[feature].dropna()
            wf_result = self._walk_forward_single_feature(
                feature_data, target, n_steps
            )
            walk_forward_results[feature] = wf_result
        
        return walk_forward_results
    
    
    def _walk_forward_single_feature(self, feature_data: pd.Series,
                                   target: pd.Series, n_steps: int) -> Dict:
        """Walk-forward validation for single feature."""
        
        common_idx = feature_data.index.intersection(target.index)
        if len(common_idx) < self.min_window_size * 2:
            return {'error': 'Insufficient data', 'consistency_score': 0}
        
        sorted_idx = common_idx.sort_values()
        total_obs = len(sorted_idx)
        step_size = (total_obs - self.min_window_size) // n_steps
        
        if step_size < self.min_window_size // 4:
            return {'error': 'Cannot create sufficient windows', 'consistency_score': 0}
        
        results = []
        significant_windows = 0
        effect_sizes = []
        
        for i in range(n_steps):
            start_idx = i * step_size
            end_idx = start_idx + self.min_window_size
            
            if end_idx > total_obs:
                break
            
            window_idx = sorted_idx[start_idx:end_idx]
            window_feature = feature_data.loc[window_idx]
            window_target = target.loc[window_idx]
            
            # Clean data
            valid_mask = ~(pd.isna(window_feature) | pd.isna(window_target))
            window_feature_clean = window_feature[valid_mask]
            window_target_clean = window_target[valid_mask]
            
            if len(window_feature_clean) < self.min_window_size * 0.8:
                continue
            
            try:
                corr, p_val = pearsonr(window_feature_clean.values, window_target_clean.values)
                
                window_result = {
                    'window': i,
                    'correlation': corr,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
                
                results.append(window_result)
                
                if window_result['significant']:
                    significant_windows += 1
                
                if not pd.isna(corr):
                    effect_sizes.append(abs(corr))
                    
            except:
                continue
        
        if not results:
            return {'error': 'No valid windows', 'consistency_score': 0}
        
        consistency_score = significant_windows / len(results)
        mean_effect_size = np.mean(effect_sizes) if effect_sizes else 0
        
        return {
            'total_windows': len(results),
            'significant_windows': significant_windows,
            'consistency_score': consistency_score,
            'mean_effect_size': mean_effect_size,
            'window_results': results
        }
    
    
    def _select_final_features(self, correction_results: Dict, walk_forward_results: Dict) -> Dict:
        """Select final features based on all criteria."""
        
        significant_features = {}
        
        for feature, test_result in correction_results['corrected_results'].items():
            if not test_result['significant_after_correction']:
                continue
            
            # Check effect size
            effect_size_passed = self._check_effect_size(test_result)
            
            # Check temporal consistency
            wf_result = walk_forward_results.get(feature, {})
            temporal_consistency = wf_result.get('consistency_score', 0)
            
            if effect_size_passed and temporal_consistency >= self.min_temporal_consistency:
                significant_features[feature] = {
                    'p_value_corrected': test_result['corrected_p_value'],
                    'p_value_raw': test_result['raw_p_value'],
                    'effect_size_pearson': test_result.get('pearson_correlation', np.nan),
                    'effect_size_spearman': test_result.get('spearman_correlation', np.nan),
                    'mutual_info_score': test_result.get('mutual_info_score', np.nan),
                    'temporal_consistency': temporal_consistency,
                    'significant_windows': wf_result.get('significant_windows', 0),
                    'total_windows': wf_result.get('total_windows', 0),
                    'mean_effect_size': wf_result.get('mean_effect_size', np.nan)
                }
        
        return significant_features
    
    
    def _check_effect_size(self, test_result: Dict) -> bool:
        """Check if feature meets effect size criteria."""
        
        # Primary: Pearson correlation
        pearson_corr = test_result.get('pearson_correlation', 0)
        if not pd.isna(pearson_corr) and abs(pearson_corr) >= self.min_effect_size:
            return True
        
        # Secondary: Spearman correlation
        spearman_corr = test_result.get('spearman_correlation', 0)
        if not pd.isna(spearman_corr) and abs(spearman_corr) >= self.min_effect_size:
            return True
        
        # Tertiary: Mutual information (adjusted threshold)
        mi_score = test_result.get('mutual_info_score', 0)
        if not pd.isna(mi_score) and mi_score >= self.min_effect_size * 0.5:
            return True
        
        return False
    
    
    def _create_validation_summary(self, feature_cols: List[str], viable_features: List[str],
                                 test_results: Dict, correction_results: Dict,
                                 significant_features: Dict, processing_time: float) -> Dict:
        """Create validation summary statistics."""
        
        return {
            'total_features_tested': len(feature_cols),
            'features_passing_screening': len(viable_features),
            'features_with_sufficient_data': len(test_results),
            'features_significant_before_correction': len(correction_results['significant_before_correction']),
            'features_significant_after_correction': len(correction_results['corrected_results']),
            'features_meeting_all_criteria': len(significant_features),
            'selection_rate': len(significant_features) / len(feature_cols) if feature_cols else 0,
            'correction_method': self.correction_method,
            'significance_level': self.significance_level,
            'min_effect_size': self.min_effect_size,
            'min_temporal_consistency': self.min_temporal_consistency,
            'processing_time_seconds': processing_time
        }
    
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result dictionary."""
        
        return {
            'error': error_message,
            'significant_features': {},
            'validation_summary': {},
            'test_results': {},
            'walk_forward_results': {},
            'screening_results': {}
        }
    
    
    def _generate_report_content(self, significant_features: Dict, summary: Dict) -> str:
        """Generate formatted validation report content."""
        
        selection_rate = summary.get('selection_rate', 0)
        
        report_lines = [
            "="*90,
            "📊 STATISTICAL FEATURE VALIDATION REPORT",
            "   DAX Trend-Following Trading System",
            "="*90,
            "",
            "🎯 EXECUTIVE SUMMARY:",
            f"   Statistical validation identified {len(significant_features)} features",
            f"   out of {summary['total_features_tested']} engineered features that meet",
            f"   rigorous statistical significance and effect size criteria.",
            "",
            f"   Selection Rate: {selection_rate:.1%}",
            f"   Processing Time: {summary['processing_time_seconds']:.2f} seconds",
            "",
            "📈 VALIDATION PIPELINE RESULTS:",
            f"   ├─ Total features tested: {summary['total_features_tested']:,}",
            f"   ├─ Passed initial screening: {summary['features_passing_screening']:,}",
            f"   ├─ Had sufficient data: {summary['features_with_sufficient_data']:,}",
            f"   ├─ Significant before correction: {summary['features_significant_before_correction']:,}",
            f"   ├─ Significant after correction: {summary['features_significant_after_correction']:,}",
            f"   └─ Met all final criteria: {summary['features_meeting_all_criteria']:,}",
            "",
            "⚙️ VALIDATION PARAMETERS:",
            f"   Significance level (α): {summary['significance_level']}",
            f"   Minimum effect size: {summary['min_effect_size']}",
            f"   Multiple testing correction: {summary['correction_method']}",
            f"   Temporal consistency threshold: {summary['min_temporal_consistency']:.0%}",
            "",
        ]
        
        if significant_features:
            report_lines.extend([
                "🏆 STATISTICALLY VALIDATED FEATURES:",
                "",
                "Rank | Feature Name                    | Effect Size | P-Value    | Temporal | Status",
                "     |                                 | (Pearson)   | (Adj.)     | Consist. |",
                "─────┼─────────────────────────────────┼─────────────┼────────────┼──────────┼──────────",
            ])
            
            # Sort features by effect size
            sorted_features = sorted(
                significant_features.items(),
                key=lambda x: abs(x[1].get('effect_size_pearson', 0)) if not pd.isna(x[1].get('effect_size_pearson', 0)) else 0,
                reverse=True
            )
            
            for rank, (feature, stats) in enumerate(sorted_features, 1):
                effect_size = stats.get('effect_size_pearson', np.nan)
                p_value = stats.get('p_value_corrected', np.nan)
                consistency = stats.get('temporal_consistency', np.nan)
                
                # Format values
                effect_str = f"{effect_size:+.3f}" if not pd.isna(effect_size) else "N/A"
                p_str = f"{p_value:.2e}" if not pd.isna(p_value) else "N/A"
                consist_str = f"{consistency:.1%}" if not pd.isna(consistency) else "N/A"
                
                # Truncate long feature names
                feature_display = feature[:30] + "..." if len(feature) > 30 else feature
                
                report_lines.append(
                    f"{rank:4d} │ {feature_display:<31} │ {effect_str:>11} │ {p_str:>10} │ {consist_str:>8} │ ✓"
                )
            
            # Detailed analysis for top features
            report_lines.extend([
                "",
                "📋 TOP VALIDATED FEATURES (Detailed Analysis):",
            ])
            
            for i, (feature, stats) in enumerate(sorted_features[:5], 1):
                report_lines.extend([
                    f"",
                    f"   {i}. {feature}:",
                    f"      Effect Sizes:",
                    f"         Pearson correlation: {stats.get('effect_size_pearson', 'N/A'):.4f}",
                    f"         Spearman correlation: {stats.get('effect_size_spearman', 'N/A'):.4f}",
                    f"         Mutual information: {stats.get('mutual_info_score', 'N/A'):.4f}",
                    f"      Statistical Significance:",
                    f"         Raw p-value: {stats.get('p_value_raw', 'N/A'):.2e}",
                    f"         Corrected p-value: {stats.get('p_value_corrected', 'N/A'):.2e}",
                    f"      Temporal Validation:",
                    f"         Consistency score: {stats.get('temporal_consistency', 'N/A'):.3f}",
                    f"         Significant windows: {stats.get('significant_windows', 0)}/{stats.get('total_windows', 0)}",
                    f"         Mean effect size: {stats.get('mean_effect_size', 'N/A'):.4f}",
                ])
        else:
            report_lines.extend([
                "❌ NO FEATURES MEET ALL VALIDATION CRITERIA",
                "",
                "   Possible reasons:",
                "   • Features lack sufficient predictive power",
                "   • Validation criteria are too stringent",
                "   • Target variable definition needs refinement",
                "   • More sophisticated feature engineering required",
            ])
        
        # Recommendations
        report_lines.extend([
            "",
            "🎯 RECOMMENDATIONS FOR TRADING STRATEGY:",
        ])
        
        if len(significant_features) >= 10:
            report_lines.extend([
                "   ✅ EXCELLENT: Sufficient validated features for robust strategy",
                "   • Consider ensemble methods combining multiple features",
                "   • Implement feature importance weighting based on effect sizes",
                "   • Monitor temporal consistency in live trading",
            ])
        elif len(significant_features) >= 5:
            report_lines.extend([
                "   ✅ GOOD: Adequate validated features for strategy development",
                "   • Focus on top-performing features",
                "   • Consider feature interaction effects",
                "   • Implement careful position sizing",
            ])
        elif len(significant_features) >= 2:
            report_lines.extend([
                "   ⚠️ LIMITED: Few validated features - proceed with caution",
                "   • Develop additional feature engineering approaches",
                "   • Consider alternative target variable definitions",
                "   • Implement strict risk management",
            ])
        else:
            report_lines.extend([
                "   ❌ INSUFFICIENT: Consider alternative approaches",
                "   • Review feature engineering methodology",
                "   • Explore different timeframes or market regimes",
                "   • Consider regime-specific feature validation",
            ])
        
        report_lines.extend([
            "",
            "🔬 STATISTICAL METHODOLOGY:",
            "   This validation framework implements multiple hypothesis tests:",
            "   • Pearson correlation (linear relationships)",
            "   • Spearman correlation (monotonic relationships)", 
            "   • Independent t-test (group differences)",
            "   • Mann-Whitney U test (non-parametric differences)",
            "   • F-statistic from regression",
            "   • Mutual information (non-linear relationships)",
            "",
            f"   Multiple testing correction ({summary['correction_method']}) controls",
            "   false discovery rate to ensure statistical rigor.",
            "",
            "   Walk-forward validation tests temporal stability across",
            "   multiple time windows to ensure robustness.",
            "",
            "="*90,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "="*90
        ])
        
        return "\n".join(report_lines)


# Convenience functions for easy integration

def validate_features(features_df: pd.DataFrame,
                     target_variable: pd.Series = None,
                     **kwargs) -> Dict:
    """
    Convenience function for quick feature validation.
    
    Args:
        features_df: DataFrame with features
        target_variable: Target variable (optional)
        **kwargs: Additional parameters for StatisticalFeatureValidator
        
    Returns:
        Validation results dictionary
    """
    
    validator = StatisticalFeatureValidator(**kwargs)
    return validator.validate_features(features_df, target_variable)


def quick_validation_report(features_df: pd.DataFrame,
                          target_variable: pd.Series = None,
                          output_path: str = None) -> str:
    """
    Quick validation with automatic report generation.
    
    Args:
        features_df: DataFrame with features
        target_variable: Target variable (optional)
        output_path: Path to save report (optional)
        
    Returns:
        Validation report string
    """
    
    validator = StatisticalFeatureValidator()
    results = validator.validate_features(features_df, target_variable)
    return validator.generate_validation_report(results, output_path)


def get_validated_features_only(features_df: pd.DataFrame,
                               target_variable: pd.Series = None,
                               **kwargs) -> pd.DataFrame:
    """
    Get DataFrame with only statistically validated features.
    
    Args:
        features_df: DataFrame with features
        target_variable: Target variable (optional)
        **kwargs: Additional parameters for validation
        
    Returns:
        DataFrame with only validated features + OHLCV
    """
    
    validator = StatisticalFeatureValidator(**kwargs)
    results = validator.validate_features(features_df, target_variable)
    return validator.filter_features_to_validated(features_df, results)


# Integration helper for existing FeatureEngineer class

def add_validation_to_feature_engineer():
    """
    Add statistical validation methods to existing FeatureEngineer class.
    
    Call this function after importing FeatureEngineer to extend it with
    statistical validation capabilities.
    """
    
    try:
        from src.features.engineering import FeatureEngineer
        
        def validate_generated_features(self, features_df: pd.DataFrame,
                                       target_variable: pd.Series = None,
                                       **kwargs):
            """Add validation method to FeatureEngineer instance."""
            validator = StatisticalFeatureValidator(**kwargs)
            return validator.validate_features(features_df, target_variable)
        
        def generate_validation_report_method(self, validation_results: Dict,
                                            output_path: str = None):
            """Add report generation to FeatureEngineer instance."""
            validator = StatisticalFeatureValidator()
            return validator.generate_validation_report(validation_results, output_path)
        
        def get_validated_features_method(self, features_df: pd.DataFrame,
                                        validation_results: Dict):
            """Add feature filtering to FeatureEngineer instance."""
            validator = StatisticalFeatureValidator()
            return validator.filter_features_to_validated(features_df, validation_results)
        
        # Add methods to FeatureEngineer class
        FeatureEngineer.validate_generated_features = validate_generated_features
        FeatureEngineer.generate_validation_report = generate_validation_report_method
        FeatureEngineer.get_validated_features = get_validated_features_method
        
        logger.info("Statistical validation methods added to FeatureEngineer class")
        return True
        
    except ImportError:
        logger.error("Could not import FeatureEngineer class for extension")
        return False


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("📊 Statistical Feature Validation Module")
    print("="*60)
    print()
    print("This module provides comprehensive statistical validation for trading features.")
    print()
    print("Key capabilities:")
    print("✅ Multiple hypothesis testing (6 different tests)")
    print("✅ Multiple testing correction (FDR control)")
    print("✅ Walk-forward temporal validation")
    print("✅ Effect size requirements")
    print("✅ Comprehensive reporting")
    print()
    print("Usage examples:")
    print()
    print("# 1. Basic validation")
    print("from src.validation.statistical_validator import validate_features")
    print("results = validate_features(features_df, target_variable)")
    print()
    print("# 2. Quick validation with report")
    print("from src.validation.statistical_validator import quick_validation_report")
    print("report = quick_validation_report(features_df, output_path='validation_report.txt')")
    print()
    print("# 3. Get only validated features")
    print("from src.validation.statistical_validator import get_validated_features_only")
    print("validated_df = get_validated_features_only(features_df)")
    print()
    print("# 4. Full control with validator class")
    print("from src.validation.statistical_validator import StatisticalFeatureValidator")
    print("validator = StatisticalFeatureValidator(min_effect_size=0.3, significance_level=0.01)")
    print("results = validator.validate_features(features_df)")
    print("report = validator.generate_validation_report(results)")
    print("validated_df = validator.filter_features_to_validated(features_df, results)")
    print()
    print("For integration with existing FeatureEngineer class:")
    print("from src.validation.statistical_validator import add_validation_to_feature_engineer")
    print("add_validation_to_feature_engineer()")