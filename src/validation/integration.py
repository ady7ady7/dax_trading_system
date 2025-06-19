#!/usr/bin/env python3
"""
Statistical Validation Integration Module
Seamlessly integrates statistical validation into the DAX trading system.

Author: DAX Trading System
Created: 2025-06-19
File: src/validation/integration.py
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Import statistical validation framework
# Handle imports with error checking
try:
    from .statistical_validator import StatisticalFeatureValidator
    VALIDATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Statistical validation not available: {e}")
    StatisticalFeatureValidator = None
    VALIDATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    High-level validation pipeline that integrates with the DAX trading system.
    
    This class provides a clean interface for adding statistical validation
    to your existing feature engineering workflow without modifying core files.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize validation pipeline with configuration.
        
        Args:
            config: System configuration dictionary
        """
        
        if not VALIDATION_AVAILABLE:
            raise ImportError("Statistical validation dependencies not available")
        
        self.config = config or {}
        
        # Get validation parameters from config
        validation_config = self.config.get('validation', {})
        
        self.validator = StatisticalFeatureValidator(
            min_window_size=validation_config.get('min_window_size', 1000),
            significance_level=validation_config.get('significance_level', 0.01),
            min_effect_size=validation_config.get('min_effect_size', 0.3),
            correction_method=validation_config.get('correction_method', 'fdr_bh'),
            min_temporal_consistency=validation_config.get('min_temporal_consistency', 0.6)
        )
        
        self.reports_dir = Path("reports/validation")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ValidationPipeline initialized")
    
    
    def validate_and_filter_features(self, 
                                   features_df: pd.DataFrame,
                                   target_method: str = 'price_direction',
                                   save_report: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete validation pipeline: validate features and return filtered DataFrame.
        
        Args:
            features_df: DataFrame with engineered features
            target_method: Method for creating target variable
            save_report: Whether to save validation report
            
        Returns:
            Tuple of (validated_features_df, validation_results)
        """
        
        logger.info("="*60)
        logger.info("FEATURE VALIDATION PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Run validation
            logger.info("Running statistical validation...")
            validation_results = self.validator.validate_features(
                features_df, target_method=target_method
            )
            
            if 'error' in validation_results:
                logger.error(f"Validation failed: {validation_results['error']}")
                return features_df, validation_results
            
            # Step 2: Filter to validated features
            logger.info("Filtering to validated features...")
            validated_df = self.validator.filter_features_to_validated(
                features_df, validation_results
            )
            
            # Step 3: Generate and save report
            if save_report:
                self._save_validation_report(validation_results)
            
            # Step 4: Log summary
            self._log_validation_summary(validation_results, start_time)
            
            return validated_df, validation_results
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {e}", exc_info=True)
            error_results = {'error': str(e), 'significant_features': {}}
            return features_df, error_results
    
    
    def quick_validate(self, features_df: pd.DataFrame, **kwargs) -> Dict:
        """
        Quick validation without filtering - just get results.
        
        Args:
            features_df: DataFrame with features
            **kwargs: Additional validation parameters
            
        Returns:
            Validation results dictionary
        """
        
        # Override validator settings if provided
        if kwargs:
            validator = StatisticalFeatureValidator(**kwargs)
        else:
            validator = self.validator
        
        return validator.validate_features(features_df)
    
    
    def validate_feature_subset(self, 
                               features_df: pd.DataFrame,
                               feature_names: list,
                               target_variable: pd.Series = None) -> Dict:
        """
        Validate only a specific subset of features.
        
        Args:
            features_df: Full features DataFrame
            feature_names: List of feature names to validate
            target_variable: Optional target variable
            
        Returns:
            Validation results for subset
        """
        
        # Keep original OHLCV + specified features
        original_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_features = [f for f in feature_names if f in features_df.columns]
        subset_cols = original_cols + available_features
        
        subset_df = features_df[subset_cols]
        
        logger.info(f"Validating subset of {len(available_features)} features")
        
        return self.validator.validate_features(subset_df, target_variable)
    
    
    def compare_validation_methods(self, features_df: pd.DataFrame) -> Dict:
        """
        Compare different validation approaches for sensitivity analysis.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary with results from different validation approaches
        """
        
        logger.info("Comparing validation methods...")
        
        comparison_results = {}
        
        # Method 1: Conservative (Bonferroni correction)
        conservative_validator = StatisticalFeatureValidator(
            significance_level=0.01,
            min_effect_size=0.3,
            correction_method='bonferroni',
            min_temporal_consistency=0.7
        )
        
        comparison_results['conservative'] = conservative_validator.validate_features(features_df)
        
        # Method 2: Moderate (FDR control)
        moderate_validator = StatisticalFeatureValidator(
            significance_level=0.01,
            min_effect_size=0.25,
            correction_method='fdr_bh',
            min_temporal_consistency=0.6
        )
        
        comparison_results['moderate'] = moderate_validator.validate_features(features_df)
        
        # Method 3: Liberal (relaxed thresholds)
        liberal_validator = StatisticalFeatureValidator(
            significance_level=0.05,
            min_effect_size=0.2,
            correction_method='fdr_bh',
            min_temporal_consistency=0.5
        )
        
        comparison_results['liberal'] = liberal_validator.validate_features(features_df)
        
        # Generate comparison summary
        comparison_summary = self._create_comparison_summary(comparison_results)
        comparison_results['summary'] = comparison_summary
        
        return comparison_results
    
    
    def get_feature_rankings(self, validation_results: Dict) -> pd.DataFrame:
        """
        Get ranked list of all tested features with their statistics.
        
        Args:
            validation_results: Results from validation
            
        Returns:
            DataFrame with feature rankings and statistics
        """
        
        if 'error' in validation_results:
            return pd.DataFrame()
        
        test_results = validation_results.get('test_results', {})
        significant_features = validation_results.get('significant_features', {})
        
        rankings = []
        
        for feature, stats in test_results.items():
            ranking_entry = {
                'feature': feature,
                'pearson_correlation': stats.get('pearson_correlation', np.nan),
                'spearman_correlation': stats.get('spearman_correlation', np.nan),
                'raw_p_value': stats.get('raw_p_value', np.nan),
                'corrected_p_value': stats.get('corrected_p_value', np.nan),
                'n_observations': stats.get('n_observations', 0),
                'validated': feature in significant_features
            }
            
            # Add temporal consistency if available
            if feature in significant_features:
                sig_stats = significant_features[feature]
                ranking_entry['temporal_consistency'] = sig_stats.get('temporal_consistency', np.nan)
                ranking_entry['significant_windows'] = sig_stats.get('significant_windows', 0)
                ranking_entry['total_windows'] = sig_stats.get('total_windows', 0)
            else:
                ranking_entry['temporal_consistency'] = np.nan
                ranking_entry['significant_windows'] = 0
                ranking_entry['total_windows'] = 0
            
            rankings.append(ranking_entry)
        
        rankings_df = pd.DataFrame(rankings)
        
        # Sort by absolute correlation (effect size)
        if not rankings_df.empty:
            rankings_df['abs_correlation'] = rankings_df['pearson_correlation'].abs()
            rankings_df = rankings_df.sort_values('abs_correlation', ascending=False)
            rankings_df = rankings_df.drop('abs_correlation', axis=1)
            rankings_df.reset_index(drop=True, inplace=True)
        
        return rankings_df
    
    
    def _save_validation_report(self, validation_results: Dict) -> None:
        """Save validation report to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"validation_report_{timestamp}.txt"
        
        try:
            report = self.validator.generate_validation_report(
                validation_results, str(report_path)
            )
            
            logger.info(f"Validation report saved: {report_path}")
            
            # Also save a summary CSV for easy analysis
            rankings_df = self.get_feature_rankings(validation_results)
            if not rankings_df.empty:
                csv_path = self.reports_dir / f"feature_rankings_{timestamp}.csv"
                rankings_df.to_csv(csv_path, index=False)
                logger.info(f"Feature rankings saved: {csv_path}")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
    
    
    def _log_validation_summary(self, validation_results: Dict, start_time: datetime) -> None:
        """Log validation summary."""
        
        if 'error' in validation_results:
            logger.error(f"Validation failed: {validation_results['error']}")
            return
        
        summary = validation_results['validation_summary']
        significant_features = validation_results['significant_features']
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("="*60)
        logger.info("VALIDATION PIPELINE COMPLETED")
        logger.info("="*60)
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Features tested: {summary['total_features_tested']}")
        logger.info(f"Features validated: {len(significant_features)}")
        logger.info(f"Selection rate: {summary['selection_rate']:.1%}")
        
        if significant_features:
            logger.info("Top 3 validated features:")
            sorted_features = sorted(
                significant_features.items(),
                key=lambda x: abs(x[1].get('effect_size_pearson', 0)),
                reverse=True
            )
            
            for i, (feature, stats) in enumerate(sorted_features[:3], 1):
                effect_size = stats.get('effect_size_pearson', np.nan)
                p_value = stats.get('p_value_corrected', np.nan)
                consistency = stats.get('temporal_consistency', np.nan)
                
                logger.info(f"  {i}. {feature}")
                logger.info(f"     Effect size: {effect_size:.3f}")
                logger.info(f"     P-value: {p_value:.2e}")
                logger.info(f"     Temporal consistency: {consistency:.1%}")
    
    
    def _create_comparison_summary(self, comparison_results: Dict) -> Dict:
        """Create summary of comparison between validation methods."""
        
        summary = {}
        
        for method, results in comparison_results.items():
            if 'error' not in results:
                significant_features = results.get('significant_features', {})
                validation_summary = results.get('validation_summary', {})
                
                summary[method] = {
                    'features_validated': len(significant_features),
                    'selection_rate': validation_summary.get('selection_rate', 0),
                    'features_tested': validation_summary.get('total_features_tested', 0)
                }
                
                if significant_features:
                    effect_sizes = [
                        abs(stats.get('effect_size_pearson', 0))
                        for stats in significant_features.values()
                        if not pd.isna(stats.get('effect_size_pearson', np.nan))
                    ]
                    
                    if effect_sizes:
                        summary[method]['mean_effect_size'] = np.mean(effect_sizes)
                        summary[method]['min_effect_size'] = np.min(effect_sizes)
                        summary[method]['max_effect_size'] = np.max(effect_sizes)
        
        return summary


# Convenience functions for easy integration

def add_validation_step(features_df: pd.DataFrame, 
                       config: dict = None,
                       target_method: str = 'price_direction',
                       save_report: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Add validation step to existing feature engineering pipeline.
    
    This function can be easily inserted into your main.py workflow.
    
    Args:
        features_df: DataFrame with engineered features
        config: System configuration
        target_method: Target variable creation method
        save_report: Whether to save validation report
        
    Returns:
        Tuple of (validated_df, validation_results)
    """
    
    if not VALIDATION_AVAILABLE:
        logger.error("Statistical validation dependencies not available")
        return features_df, {'error': 'Statistical validation dependencies not available'}
    
    try:
        pipeline = ValidationPipeline(config)
        return pipeline.validate_and_filter_features(
            features_df, target_method, save_report
        )
    except Exception as e:
        logger.error(f"Validation step failed: {e}")
        return features_df, {'error': str(e)}


def quick_feature_check(features_df: pd.DataFrame,
                       min_effect_size: float = 0.2,
                       significance_level: float = 0.05) -> Dict:
    """
    Quick feature check with relaxed parameters for exploration.
    
    Args:
        features_df: DataFrame with features
        min_effect_size: Minimum effect size threshold
        significance_level: Significance level
        
    Returns:
        Validation results
    """
    
    pipeline = ValidationPipeline()
    return pipeline.quick_validate(
        features_df,
        min_effect_size=min_effect_size,
        significance_level=significance_level,
        min_temporal_consistency=0.4
    )


def get_best_features(features_df: pd.DataFrame, 
                     n_features: int = 10) -> list:
    """
    Get the top N statistically validated features.
    
    Args:
        features_df: DataFrame with features
        n_features: Number of top features to return
        
    Returns:
        List of best feature names
    """
    
    pipeline = ValidationPipeline()
    validation_results = pipeline.quick_validate(features_df)
    
    if 'error' in validation_results:
        logger.warning("Validation failed, returning empty list")
        return []
    
    significant_features = validation_results.get('significant_features', {})
    
    if not significant_features:
        logger.warning("No features passed validation")
        return []
    
    # Sort by effect size
    sorted_features = sorted(
        significant_features.items(),
        key=lambda x: abs(x[1].get('effect_size_pearson', 0)),
        reverse=True
    )
    
    return [feature for feature, _ in sorted_features[:n_features]]


# Configuration helper

def create_validation_config(conservative: bool = False) -> dict:
    """
    Create validation configuration for system config.
    
    Args:
        conservative: Whether to use conservative validation parameters
        
    Returns:
        Validation configuration dictionary
    """
    
    if conservative:
        return {
            'validation': {
                'min_window_size': 1500,
                'significance_level': 0.005,  # More stringent
                'min_effect_size': 0.35,      # Higher effect size
                'correction_method': 'bonferroni',
                'min_temporal_consistency': 0.7
            }
        }
    else:
        return {
            'validation': {
                'min_window_size': 1000,
                'significance_level': 0.01,
                'min_effect_size': 0.3,
                'correction_method': 'fdr_bh',
                'min_temporal_consistency': 0.6
            }
        }


if __name__ == "__main__":
    # Example integration test
    logging.basicConfig(level=logging.INFO)
    
    print("📊 Statistical Validation Integration Module")
    print("="*60)
    print()
    print("This module provides seamless integration of statistical validation")
    print("into your existing DAX trading system workflow.")
    print()
    print("Integration examples:")
    print()
    print("# 1. Add validation step to main.py:")
    print("from src.validation.integration import add_validation_step")
    print()
    print("# After feature engineering:")
    print("validated_df, validation_results = add_validation_step(")
    print("    features_df=features_df,")
    print("    config=config,")
    print("    target_method='price_direction'")
    print(")")
    print()
    print("# 2. Quick feature exploration:")
    print("from src.validation.integration import quick_feature_check")
    print("results = quick_feature_check(features_df, min_effect_size=0.2)")
    print()
    print("# 3. Get top validated features:")
    print("from src.validation.integration import get_best_features")
    print("top_features = get_best_features(features_df, n_features=10)")
    print()
    print("# 4. Full pipeline control:")
    print("from src.validation.integration import ValidationPipeline")
    print("pipeline = ValidationPipeline(config)")
    print("validated_df, results = pipeline.validate_and_filter_features(features_df)")
    print()
    print("Benefits:")
    print("✅ No modification to existing engineering.py")
    print("✅ Clean separation of concerns")
    print("✅ Easy to enable/disable validation")
    print("✅ Comprehensive reporting")
    print("✅ Multiple validation strategies")
    print()
    print("File structure needed:")
    print("src/")
    print("├── validation/")
    print("│   ├── __init__.py")
    print("│   ├── statistical_validator.py")
    print("│   └── integration.py")
    print("├── features/")
    print("│   └── engineering.py")
    print("└── main.py")
    print()
    print("Ready for integration into your DAX trading system!")