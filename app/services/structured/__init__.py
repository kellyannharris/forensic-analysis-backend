"""
Structured Data Models Package
Description: Comprehensive crime analysis models and utilities for API consumption
Author: Kelly-Ann Harris
Date: 2024
"""

from .crime_rate_predictor import CrimeRatePredictor
from .criminal_network_analyzer import CriminalNetworkAnalyzer
from .spatial_crime_mapper import SpatialCrimeMapper
from .temporal_pattern_analyzer import TemporalPatternAnalyzer
from .data_analyzer import LAPDCrimeDataAnalyzer
from .crime_type_classifier import CrimeTypeClassifier

__version__ = "1.0.0"
__author__ = "Kelly-Ann Harris"

__all__ = [
    'CrimeRatePredictor',
    'CriminalNetworkAnalyzer', 
    'SpatialCrimeMapper',
    'TemporalPatternAnalyzer',
    'LAPDCrimeDataAnalyzer',
    'CrimeTypeClassifier'
] 