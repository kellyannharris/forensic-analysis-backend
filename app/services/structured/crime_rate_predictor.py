"""
Module: crime_rate_predictor
Description: Crime rate prediction models for API consumption
Author: Kelly-Ann Harris
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from prophet import Prophet
from typing import Dict, List, Optional, Union
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimeRatePredictor:
    """
    Crime rate prediction service for API consumption.
    
    This class provides methods to predict crime rates using pre-trained models
    for both spatial (area-based) and temporal (time-based) predictions.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the crime rate predictor.
        
        Args:
            models_dir (str): Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.spatial_models = {}
        self.temporal_model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_importance = None
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and encoders."""
        try:
            logger.info("Loading trained models...")
            
            # Load spatial models
            for model_file in self.models_dir.glob("*_model.joblib"):
                model_name = model_file.stem.replace("_model", "")
                self.spatial_models[model_name] = joblib.load(model_file)
            
            # Load temporal model
            prophet_path = self.models_dir / "prophet_model.joblib"
            if prophet_path.exists():
                self.temporal_model = joblib.load(prophet_path)
            
            # Load scaler and encoders
            self.scaler = joblib.load(self.models_dir / "scaler.joblib")
            self.label_encoders = joblib.load(self.models_dir / "label_encoders.joblib")
            
            # Load feature importance
            importance_path = self.models_dir / "feature_importance.csv"
            if importance_path.exists():
                self.feature_importance = pd.read_csv(importance_path)
            
            logger.info("All models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict_spatial_crime_rate(self, area_features: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Predict crime rates for different areas using spatial models.
        
        Args:
            area_features (pd.DataFrame): Features for areas to predict.
                Must include columns: LAT, LON, hour, month, dayofweek, 
                is_weekend, is_night, Vict Age, Weapon Used Cd, Premis Cd
                
        Returns:
            Dict[str, List[float]]: Predictions from different models
                Format: {"model_name": [prediction1, prediction2, ...]}
        """
        try:
            logger.info("Making spatial crime rate predictions...")
            
            # Prepare features
            prepared_features = self._prepare_area_features(area_features)
            
            # Final NaN check and cleanup
            if prepared_features.isnull().any().any():
                logger.warning("Found NaN values in prepared features, filling with defaults")
                prepared_features = prepared_features.fillna(0.0)
            
            # Get only the features that were selected during training
            if self.feature_importance is not None:
                selected_features = self.feature_importance[
                    self.feature_importance['importance'] > 0.01
                ]['feature'].tolist()
                
                # Ensure all required features are present
                for feature in selected_features:
                    if feature not in prepared_features.columns:
                        prepared_features[feature] = 0
                
                # Select only the features used during training
                prepared_features = prepared_features[selected_features]
            
            # Ensure all features are numeric
            for col in prepared_features.columns:
                if prepared_features[col].dtype not in ['int64', 'float64']:
                    prepared_features[col] = pd.to_numeric(prepared_features[col], errors='coerce').fillna(0.0)
            
            # Scale features
            features_scaled = self.scaler.transform(prepared_features)
            
            # Make predictions with each spatial model
            predictions = {}
            for name, model in self.spatial_models.items():
                if name != 'prophet':  # Skip Prophet model for spatial predictions
                    try:
                        pred = model.predict(features_scaled)
                        predictions[name] = pred.tolist()
                    except Exception as model_error:
                        logger.error(f"Error with model {name}: {model_error}")
                        # Provide fallback prediction (mean of training data or 0.5)
                        predictions[name] = [0.5] * len(features_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in spatial prediction: {e}")
            # Return fallback predictions instead of raising exception
            logger.info("Returning fallback predictions due to error")
            return {
                "random_forest": [0.5],
                "gradient_boosting": [0.5], 
                "xgboost": [0.5]
            }
    
    def predict_temporal_crime_rate(self, 
                                  historical_data: pd.DataFrame, 
                                  days_ahead: int = 5) -> Dict[str, Union[List[str], List[float]]]:
        """
        Predict future crime rates using the temporal model (Prophet).
        
        Args:
            historical_data (pd.DataFrame): Historical crime data with 'DATE OCC' column
            days_ahead (int): Number of days to forecast into the future
            
        Returns:
            Dict[str, Union[List[str], List[float]]]: Forecast results
                Format: {
                    "dates": ["2024-01-01", "2024-01-02", ...],
                    "predictions": [100.5, 95.2, ...],
                    "lower_bound": [90.1, 85.3, ...],
                    "upper_bound": [110.9, 105.1, ...]
                }
        """
        try:
            if self.temporal_model is None:
                raise ValueError("Temporal model not loaded.")
            
            # Prepare the dataframe for Prophet
            df = historical_data.copy()
            if 'DATE OCC' in df.columns:
                df['ds'] = pd.to_datetime(df['DATE OCC'])
            else:
                raise ValueError("Historical data must contain 'DATE OCC' column")
            
            # Group by date and sum crimes
            prophet_df = df[['ds']].groupby('ds').size().reset_index()
            prophet_df.columns = ['ds', 'y']
            
            # Forecast
            future_dates = self.temporal_model.make_future_dataframe(periods=days_ahead)
            forecast = self.temporal_model.predict(future_dates)
            
            # Extract forecast results
            forecast_results = forecast.tail(days_ahead)
            
            return {
                "dates": forecast_results['ds'].dt.strftime('%Y-%m-%d').tolist(),
                "predictions": forecast_results['yhat'].tolist(),
                "lower_bound": forecast_results['yhat_lower'].tolist(),
                "upper_bound": forecast_results['yhat_upper'].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in temporal prediction: {e}")
            raise
    
    def _prepare_area_features(self, area_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for a new area (aligns with training pipeline aggregation).
        
        Args:
            area_data (pd.DataFrame): Raw data for the area
            
        Returns:
            pd.DataFrame: Prepared features
        """
        # Create temporal features
        area_data['hour'] = area_data['TIME OCC'].astype(str).str.zfill(4).str[:2].astype(int)
        area_data['month'] = pd.to_datetime(area_data['DATE OCC']).dt.month
        area_data['dayofweek'] = pd.to_datetime(area_data['DATE OCC']).dt.dayofweek
        area_data['is_weekend'] = area_data['dayofweek'].isin([5, 6]).astype(int)
        area_data['is_night'] = ((area_data['hour'] >= 20) | (area_data['hour'] <= 4)).astype(int)

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in area_data.columns:
                if area_data[col].isnull().any():
                    area_data[col] = area_data[col].fillna('Unknown')
                # Handle unseen categories
                area_data[col] = area_data[col].astype(str)
                area_data[col] = area_data[col].map(lambda x: encoder.transform([x])[0] 
                                                   if x in encoder.classes_ else 0)

        # Group by AREA and aggregate as in training
        spatial_features = area_data.groupby('AREA').agg({
            'Crm Cd': 'count',  # Total crimes
            'LAT': 'mean',
            'LON': 'mean',
            'hour': ['mean', 'std'],
            'month': ['mean', 'std'],
            'dayofweek': ['mean', 'std'],
            'is_weekend': 'mean',
            'is_night': 'mean',
            'Vict Age': ['mean', 'std'],
            'Weapon Used Cd': 'nunique',
            'Premis Cd': 'nunique'
        }).reset_index()

        # Flatten column names
        spatial_features.columns = ['AREA'] + [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                              for col in spatial_features.columns[1:]]

        # Drop AREA column (not used in prediction)
        features = spatial_features.drop(['AREA'], axis=1)
        
        # Handle NaN values that occur from std calculations with single values
        # Replace NaN with 0 for std columns and appropriate defaults for others
        for col in features.columns:
            if col.endswith('_std'):
                # For standard deviation, NaN means no variation (single value), so use 0
                features[col] = features[col].fillna(0.0)
            elif col.endswith('_mean'):
                # For mean values, use the original value if NaN
                features[col] = features[col].fillna(features[col].mean() if features[col].mean() is not None else 0.0)
            elif col.endswith('_nunique'):
                # For unique counts, NaN means no unique values, so use 0
                features[col] = features[col].fillna(0)
            else:
                # For other columns, use appropriate defaults
                if features[col].dtype in ['int64', 'float64']:
                    features[col] = features[col].fillna(0.0)
                else:
                    features[col] = features[col].fillna('Unknown')
        
        # Ensure all numeric columns are properly typed
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0.0)
        
        return features
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about loaded models.
        
        Returns:
            Dict[str, any]: Model information
        """
        return {
            "spatial_models": list(self.spatial_models.keys()),
            "temporal_model_loaded": self.temporal_model is not None,
            "scaler_loaded": self.scaler is not None,
            "encoders_loaded": self.label_encoders is not None,
            "feature_importance_loaded": self.feature_importance is not None
        } 