"""
Statistical Validation Module for DAX Trading System

This module provides comprehensive statistical validation for trading features,
implementing rigorous hypothesis testing and temporal stability validation.

Author: DAX Trading System
Created: 2025-06-19
"""

from .statistical_validator import StatisticalFeatureValidator, validate_features, quick_validation_report, get_validated_features_only
from .integration import ValidationPipeline, add_validation_step, quick_feature_check, get_best_features, create_validation_config

__version__ = "1.0.0"
__author__ = "DAX Trading System"

# Convenience imports for easy access
__all__ = [
    # Core validator
    'StatisticalFeatureValidator',
    'validate_features',
    'quick_validation_report', 
    'get_validated_features_only',
    
    # Integration helpers
    'ValidationPipeline',
    'add_validation_step',
    'quick_feature_check',
    'get_best_features',
    'create_validation_config'
]