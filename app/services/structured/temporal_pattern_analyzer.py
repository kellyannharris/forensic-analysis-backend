"""
Module: temporal_pattern_analyzer
Description: Temporal pattern recognition and time series analysis for API consumption
Author: Kelly-Ann Harris
Date: 2024
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TemporalPatternAnalyzer:
    """
    Temporal pattern recognition service for API consumption.
    
    This class provides methods to analyze temporal patterns in crime data,
    including time series analysis, seasonality detection, and trend analysis.
    """
    
    def __init__(self):
        """Initialize the temporal pattern analyzer."""
        self.time_series_data = None
        self.arima_model = None
        self.decomposition = None
    
    def prepare_time_series_data(self, crime_data: pd.DataFrame, 
                                time_unit: str = "daily") -> Dict[str, Union[int, List[str]]]:
        """
        Prepare time series data for analysis.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'DATE OCC' column
            time_unit (str): Time unit for aggregation ("daily", "weekly", "monthly")
                
        Returns:
            Dict[str, Union[int, List[str]]]: Time series data statistics
                Format: {
                    "total_records": 1000,
                    "time_periods": 365,
                    "date_range": ["2023-01-01", "2023-12-31"],
                    "mean_crimes_per_period": 2.7
                }
        """
        try:
            logger.info(f"Preparing time series data with {time_unit} aggregation...")
            
            # Convert date column
            crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'])
            
            # Aggregate by time unit
            if time_unit == "daily":
                time_series = crime_data.groupby('DATE OCC').size()
            elif time_unit == "weekly":
                time_series = crime_data.groupby(crime_data['DATE OCC'].dt.to_period('W')).size()
                time_series.index = time_series.index.astype(str).map(lambda x: pd.to_datetime(x.split('/')[0]))
            elif time_unit == "monthly":
                time_series = crime_data.groupby(crime_data['DATE OCC'].dt.to_period('M')).size()
                time_series.index = time_series.index.astype(str).map(lambda x: pd.to_datetime(x))
            else:
                raise ValueError(f"Unknown time unit: {time_unit}")
            
            # Fill missing dates with 0
            time_series = time_series.reindex(
                pd.date_range(time_series.index.min(), time_series.index.max(), freq='D' if time_unit == "daily" else 'W' if time_unit == "weekly" else 'M'),
                fill_value=0
            )
            
            self.time_series_data = time_series
            
            stats = {
                "total_records": len(crime_data),
                "time_periods": len(time_series),
                "date_range": [time_series.index.min().strftime('%Y-%m-%d'), 
                              time_series.index.max().strftime('%Y-%m-%d')],
                "mean_crimes_per_period": float(time_series.mean()),
                "std_crimes_per_period": float(time_series.std()),
                "min_crimes_per_period": int(time_series.min()),
                "max_crimes_per_period": int(time_series.max())
            }
            
            logger.info(f"Time series prepared: {stats['time_periods']} periods")
            return stats
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            raise
    
    def detect_seasonality(self, period: Optional[int] = None) -> Dict[str, Union[Dict[str, float], List[float]]]:
        """
        Detect seasonality in the time series data.
        
        Args:
            period (Optional[int]): Period for seasonal decomposition (auto-detect if None)
                
        Returns:
            Dict[str, Union[Dict[str, float], List[float]]]: Seasonality analysis results
                Format: {
                    "seasonal_strength": 0.8,
                    "trend_strength": 0.6,
                    "noise_strength": 0.2,
                    "seasonal_pattern": [1.2, 0.8, 1.1, ...],
                    "is_stationary": true
                }
        """
        try:
            if self.time_series_data is None:
                raise ValueError("Time series data not prepared. Call prepare_time_series_data() first.")
            
            logger.info("Detecting seasonality patterns...")
            
            # Auto-detect period if not provided
            if period is None:
                # Try common periods
                for p in [7, 30, 365]:  # weekly, monthly, yearly
                    if len(self.time_series_data) >= 2 * p:
                        period = p
                        break
                if period is None:
                    period = len(self.time_series_data) // 4  # Default to 1/4 of data
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(self.time_series_data, period=period, extrapolate_trend='freq')
            self.decomposition = decomposition
            
            # Calculate strength measures
            seasonal_strength = np.var(decomposition.seasonal) / (np.var(decomposition.seasonal) + np.var(decomposition.resid))
            trend_strength = np.var(decomposition.trend.dropna()) / (np.var(decomposition.trend.dropna()) + np.var(decomposition.resid))
            noise_strength = 1 - seasonal_strength - trend_strength
            
            # Test for stationarity
            adf_result = adfuller(self.time_series_data.dropna())
            is_stationary = adf_result[1] < 0.05
            
            # Extract seasonal pattern
            seasonal_pattern = decomposition.seasonal[:period].tolist()
            
            results = {
                "seasonal_strength": float(seasonal_strength),
                "trend_strength": float(trend_strength),
                "noise_strength": float(noise_strength),
                "seasonal_pattern": seasonal_pattern,
                "is_stationary": bool(is_stationary),
                "adf_p_value": float(adf_result[1]),
                "period_used": int(period)
            }
            
            logger.info(f"Seasonality detected: strength={results['seasonal_strength']:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            raise
    
    def fit_arima_model(self, order: Tuple[int, int, int] = (5, 1, 0)) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Fit an ARIMA model to the time series data.
        
        Args:
            order (Tuple[int, int, int]): ARIMA order (p, d, q)
                
        Returns:
            Dict[str, Union[float, Dict[str, float]]]: ARIMA model results
                Format: {
                    "aic": 1234.56,
                    "bic": 1256.78,
                    "log_likelihood": -612.34,
                    "residuals_stats": {
                        "mean": 0.0,
                        "std": 1.2,
                        "skewness": 0.1,
                        "kurtosis": 3.2
                    }
                }
        """
        try:
            if self.time_series_data is None:
                raise ValueError("Time series data not prepared. Call prepare_time_series_data() first.")
            
            logger.info(f"Fitting ARIMA model with order {order}...")
            
            # Fit ARIMA model
            model = ARIMA(self.time_series_data, order=order)
            self.arima_model = model.fit()
            
            # Get model statistics
            residuals = self.arima_model.resid.dropna()
            
            results = {
                "aic": float(self.arima_model.aic),
                "bic": float(self.arima_model.bic),
                "log_likelihood": float(self.arima_model.llf),
                "residuals_stats": {
                    "mean": float(residuals.mean()),
                    "std": float(residuals.std()),
                    "skewness": float(residuals.skew()),
                    "kurtosis": float(residuals.kurtosis())
                },
                "model_order": order
            }
            
            logger.info(f"ARIMA model fitted: AIC={results['aic']:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def forecast_future(self, periods: int = 30, confidence_level: float = 0.95) -> Dict[str, Union[List[str], List[float]]]:
        """
        Forecast future crime rates using the fitted ARIMA model.
        
        Args:
            periods (int): Number of periods to forecast
            confidence_level (float): Confidence level for prediction intervals
                
        Returns:
            Dict[str, Union[List[str], List[float]]]: Forecast results
                Format: {
                    "dates": ["2024-01-01", "2024-01-02", ...],
                    "forecast": [100.5, 95.2, ...],
                    "lower_bound": [90.1, 85.3, ...],
                    "upper_bound": [110.9, 105.1, ...]
                }
        """
        try:
            if self.arima_model is None:
                raise ValueError("ARIMA model not fitted. Call fit_arima_model() first.")
            
            logger.info(f"Forecasting {periods} periods ahead...")
            
            # Generate forecast
            forecast_result = self.arima_model.forecast(steps=periods, alpha=1-confidence_level)
            
            # Generate future dates
            last_date = self.time_series_data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
            
            # Extract forecast values
            if hasattr(forecast_result, 'predicted_mean'):
                forecast_values = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int()
                lower_bound = conf_int.iloc[:, 0]
                upper_bound = conf_int.iloc[:, 1]
            else:
                # Handle different forecast result formats
                forecast_values = forecast_result
                lower_bound = [f * 0.9 for f in forecast_values]  # Approximate bounds
                upper_bound = [f * 1.1 for f in forecast_values]
            
            results = {
                "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
                "forecast": forecast_values.tolist() if hasattr(forecast_values, 'tolist') else list(forecast_values),
                "lower_bound": lower_bound.tolist() if hasattr(lower_bound, 'tolist') else list(lower_bound),
                "upper_bound": upper_bound.tolist() if hasattr(upper_bound, 'tolist') else list(upper_bound),
                "confidence_level": confidence_level
            }
            
            logger.info(f"Forecast generated for {periods} periods")
            return results
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise
    
    def analyze_temporal_patterns(self, crime_data: pd.DataFrame) -> Dict[str, Union[Dict[str, any], List[Dict[str, any]]]]:
        """
        Analyze various temporal patterns in crime data.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'DATE OCC' and 'TIME OCC' columns
                
        Returns:
            Dict[str, Union[Dict[str, any], List[Dict[str, any]]]]: Temporal pattern analysis
                Format: {
                    "hourly_patterns": [{"hour": 0, "crime_count": 50, "percentage": 5.0}, ...],
                    "daily_patterns": [{"day": "Monday", "crime_count": 200, "percentage": 20.0}, ...],
                    "monthly_patterns": [{"month": "January", "crime_count": 300, "percentage": 25.0}, ...],
                    "yearly_trends": [{"year": 2020, "crime_count": 1200}, ...]
                }
        """
        try:
            logger.info("Analyzing temporal patterns...")
            
            # Convert date and time columns
            crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'])
            crime_data['TIME OCC'] = pd.to_datetime(crime_data['TIME OCC'], format='%H%M', errors='coerce')
            
            # Hourly patterns
            hourly_counts = crime_data['TIME OCC'].dt.hour.value_counts().sort_index()
            total_crimes = len(crime_data)
            hourly_patterns = [
                {
                    "hour": int(hour),
                    "crime_count": int(count),
                    "percentage": float(count / total_crimes * 100)
                }
                for hour, count in hourly_counts.items()
            ]
            
            # Daily patterns
            daily_counts = crime_data['DATE OCC'].dt.day_name().value_counts()
            daily_patterns = [
                {
                    "day": day,
                    "crime_count": int(count),
                    "percentage": float(count / total_crimes * 100)
                }
                for day, count in daily_counts.items()
            ]
            
            # Monthly patterns
            monthly_counts = crime_data['DATE OCC'].dt.month_name().value_counts()
            monthly_patterns = [
                {
                    "month": month,
                    "crime_count": int(count),
                    "percentage": float(count / total_crimes * 100)
                }
                for month, count in monthly_counts.items()
            ]
            
            # Yearly trends
            yearly_counts = crime_data['DATE OCC'].dt.year.value_counts().sort_index()
            yearly_trends = [
                {
                    "year": int(year),
                    "crime_count": int(count)
                }
                for year, count in yearly_counts.items()
            ]
            
            results = {
                "hourly_patterns": hourly_patterns,
                "daily_patterns": daily_patterns,
                "monthly_patterns": monthly_patterns,
                "yearly_trends": yearly_trends,
                "total_crimes": total_crimes
            }
            
            logger.info("Temporal patterns analyzed")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            raise
    
    def get_temporal_summary(self, crime_data: pd.DataFrame) -> Dict[str, any]:
        """
        Get a comprehensive temporal analysis summary.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'DATE OCC' column
                
        Returns:
            Dict[str, any]: Temporal analysis summary
        """
        try:
            logger.info("Generating temporal analysis summary...")
            
            # Prepare time series data
            ts_stats = self.prepare_time_series_data(crime_data)
            
            # Detect seasonality
            seasonality_results = self.detect_seasonality()
            
            # Fit ARIMA model
            arima_results = self.fit_arima_model()
            
            # Generate forecast
            forecast_results = self.forecast_future(periods=30)
            
            # Analyze temporal patterns
            pattern_results = self.analyze_temporal_patterns(crime_data)
            
            summary = {
                "time_series_statistics": ts_stats,
                "seasonality_analysis": seasonality_results,
                "arima_model_results": arima_results,
                "forecast_results": forecast_results,
                "temporal_patterns": pattern_results
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating temporal summary: {e}")
            raise 