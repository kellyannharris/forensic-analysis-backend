"""
Module: crime_type_classifier
Description: Crime type classification for API consumption
Author: Kelly-Ann Harris
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional
from sklearn.preprocessing import LabelEncoder
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrimeTypeClassifier:
    """
    Crime type classification service for API consumption.
    
    This class provides methods to classify crime types using pre-trained models
    and analyze crime type distributions.
    """
    
    def __init__(self, model_path: str = "crime_type_rf_model.pkl"):
        """
        Initialize the crime type classifier.
        
        Args:
            model_path (str): Path to the trained crime type classification model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained crime type classification model."""
        try:
            logger.info(f"Loading crime type classification model from {self.model_path}")
            
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info("Crime type classification model loaded successfully")
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                logger.info("Crime type classification will use mock predictions")
                self.model = None
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Crime type classification will use mock predictions")
            self.model = None
    
    def classify_crime_types(self, crime_data: pd.DataFrame) -> Dict[str, Union[List[str], Dict[str, float]]]:
        """
        Classify crime types for given crime data.
        
        Args:
            crime_data (pd.DataFrame): Crime data with features for classification
                Required columns: ['AREA', 'TIME OCC', 'Vict Age', 'Premis Cd', 'Weapon Used Cd']
                
        Returns:
            Dict[str, Union[List[str], Dict[str, float]]]: Classification results
                Format: {
                    "predictions": ["THEFT", "ASSAULT", "BURGLARY", ...],
                    "probabilities": [0.8, 0.15, 0.05, ...],
                    "confidence_scores": [0.85, 0.72, 0.91, ...]
                }
        """
        try:
            if self.model is None:
                logger.warning("Crime type model not available, using mock predictions")
                return self._mock_classification(crime_data)
            
            logger.info("Classifying crime types...")
            
            # Prepare features
            features = self._prepare_features(crime_data)
            
            # Make predictions
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            # Get confidence scores (max probability for each prediction)
            confidence_scores = np.max(probabilities, axis=1)
            
            # Convert predictions to crime type names if encoder is available
            if hasattr(self.model, 'classes_'):
                prediction_names = [str(self.model.classes_[pred]) for pred in predictions]
            else:
                prediction_names = [str(pred) for pred in predictions.tolist()]
            
            results = {
                "predictions": prediction_names,
                "probabilities": probabilities.tolist(),
                "confidence_scores": confidence_scores.tolist(),
                "num_predictions": int(len(predictions))
            }
            
            logger.info(f"Classified {len(predictions)} crime records")
            return results
            
        except Exception as e:
            logger.error(f"Error classifying crime types: {e}")
            logger.warning("Falling back to mock classification")
            return self._mock_classification(crime_data)
    
    def _mock_classification(self, crime_data: pd.DataFrame) -> Dict[str, Union[List[str], Dict[str, float]]]:
        """
        Provide mock crime type classifications when model is not available.
        
        Args:
            crime_data (pd.DataFrame): Crime data
            
        Returns:
            Dict: Mock classification results
        """
        logger.info("Using mock crime type classification")
        
        # Common crime types
        crime_types = ["THEFT", "ASSAULT", "BURGLARY", "VEHICLE THEFT", "VANDALISM"]
        
        # Generate mock predictions
        num_records = len(crime_data)
        predictions = np.random.choice(crime_types, size=num_records)
        probabilities = np.random.dirichlet(np.ones(len(crime_types)), size=num_records)
        confidence_scores = np.random.uniform(0.6, 0.9, size=num_records)
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "confidence_scores": confidence_scores.tolist(),
            "num_predictions": num_records,
            "model_status": "mock"
        }
    
    def _prepare_features(self, crime_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for crime type classification.
        
        Args:
            crime_data (pd.DataFrame): Raw crime data
            
        Returns:
            pd.DataFrame: Prepared features for classification
        """
        # Create temporal features
        crime_data['hour'] = crime_data['TIME OCC'].astype(str).str.zfill(4).str[:2].astype(int)
        
        # Handle DATE OCC if available, otherwise use default values
        if 'DATE OCC' in crime_data.columns and not crime_data['DATE OCC'].isna().all():
            crime_data['month'] = pd.to_datetime(crime_data['DATE OCC']).dt.month
            crime_data['dayofweek'] = pd.to_datetime(crime_data['DATE OCC']).dt.dayofweek
        else:
            # Use current date as default if DATE OCC is not available
            current_date = pd.Timestamp.now()
            crime_data['month'] = current_date.month
            crime_data['dayofweek'] = current_date.dayofweek
            
        crime_data['is_weekend'] = crime_data['dayofweek'].isin([5, 6]).astype(int)
        crime_data['is_night'] = ((crime_data['hour'] >= 20) | (crime_data['hour'] <= 4)).astype(int)
        
        # Select features used in training
        feature_columns = ['AREA', 'hour', 'month', 'dayofweek', 'is_weekend', 'is_night', 
                          'Vict Age', 'Premis Cd', 'Weapon Used Cd']
        
        # Ensure all required columns exist
        missing_columns = [col for col in feature_columns if col not in crime_data.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}. Filling with default values.")
            for col in missing_columns:
                if col in ['AREA', 'Premis Cd', 'Weapon Used Cd']:
                    crime_data[col] = 0
                elif col in ['hour', 'month', 'dayofweek', 'is_weekend', 'is_night']:
                    crime_data[col] = 0
                elif col == 'Vict Age':
                    crime_data[col] = crime_data['Vict Age'].median() if 'Vict Age' in crime_data.columns else 30
        
        # Handle missing values
        crime_data = crime_data.fillna({
            'AREA': 0,
            'hour': 12,
            'month': 6,
            'dayofweek': 3,
            'is_weekend': 0,
            'is_night': 0,
            'Vict Age': crime_data['Vict Age'].median() if 'Vict Age' in crime_data.columns else 30,
            'Premis Cd': 0,
            'Weapon Used Cd': 0
        })
        
        # Select features
        features = crime_data[feature_columns].copy()
        
        return features
    
    def analyze_crime_type_distribution(self, crime_data: pd.DataFrame) -> Dict[str, Union[List[Dict[str, any]], Dict[str, any]]]:
        """
        Analyze the distribution of crime types in the data.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'Crm Cd Desc' column
                
        Returns:
            Dict[str, Union[List[Dict[str, any]], Dict[str, any]]]: Distribution analysis
                Format: {
                    "crime_type_counts": [
                        {"crime_type": "THEFT", "count": 1000, "percentage": 25.0},
                        {"crime_type": "ASSAULT", "count": 800, "percentage": 20.0}
                    ],
                    "statistics": {
                        "total_crimes": 4000,
                        "unique_crime_types": 15,
                        "most_common_crime": "THEFT",
                        "least_common_crime": "ARSON"
                    }
                }
        """
        try:
            logger.info("Analyzing crime type distribution...")
            
            # Count crime types
            crime_counts = crime_data['Crm Cd Desc'].value_counts()
            total_crimes = len(crime_data)
            
            # Create detailed breakdown
            crime_type_counts = [
                {
                    "crime_type": crime_type,
                    "count": int(count),
                    "percentage": float(count / total_crimes * 100)
                }
                for crime_type, count in crime_counts.items()
            ]
            
            # Calculate statistics
            statistics = {
                "total_crimes": total_crimes,
                "unique_crime_types": len(crime_counts),
                "most_common_crime": crime_counts.index[0],
                "least_common_crime": crime_counts.index[-1],
                "most_common_count": int(crime_counts.iloc[0]),
                "least_common_count": int(crime_counts.iloc[-1])
            }
            
            results = {
                "crime_type_counts": crime_type_counts,
                "statistics": statistics
            }
            
            logger.info(f"Analyzed distribution of {statistics['unique_crime_types']} crime types")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing crime type distribution: {e}")
            raise
    
    def get_crime_type_trends(self, crime_data: pd.DataFrame) -> Dict[str, Union[List[Dict[str, any]], Dict[str, any]]]:
        """
        Analyze trends in crime types over time.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'DATE OCC' and 'Crm Cd Desc' columns
                
        Returns:
            Dict[str, Union[List[Dict[str, any]], Dict[str, any]]]: Trend analysis
                Format: {
                    "monthly_trends": [
                        {
                            "month": "2023-01",
                            "crime_types": [
                                {"crime_type": "THEFT", "count": 100},
                                {"crime_type": "ASSAULT", "count": 80}
                            ]
                        }
                    ],
                    "yearly_trends": [
                        {
                            "year": 2023,
                            "crime_types": [
                                {"crime_type": "THEFT", "count": 1200},
                                {"crime_type": "ASSAULT", "count": 960}
                            ]
                        }
                    ]
                }
        """
        try:
            logger.info("Analyzing crime type trends over time...")
            
            # Convert date column
            crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'])
            
            # Monthly trends
            monthly_data = crime_data.groupby([crime_data['DATE OCC'].dt.to_period('M'), 'Crm Cd Desc']).size().reset_index()
            monthly_data.columns = ['month', 'crime_type', 'count']
            
            monthly_trends = []
            for month in monthly_data['month'].unique():
                month_data = monthly_data[monthly_data['month'] == month]
                crime_types = [
                    {"crime_type": row['crime_type'], "count": int(row['count'])}
                    for _, row in month_data.iterrows()
                ]
                monthly_trends.append({
                    "month": str(month),
                    "crime_types": crime_types
                })
            
            # Yearly trends
            yearly_data = crime_data.groupby([crime_data['DATE OCC'].dt.year, 'Crm Cd Desc']).size().reset_index()
            yearly_data.columns = ['year', 'crime_type', 'count']
            
            yearly_trends = []
            for year in yearly_data['year'].unique():
                year_data = yearly_data[yearly_data['year'] == year]
                crime_types = [
                    {"crime_type": row['crime_type'], "count": int(row['count'])}
                    for _, row in year_data.iterrows()
                ]
                yearly_trends.append({
                    "year": int(year),
                    "crime_types": crime_types
                })
            
            results = {
                "monthly_trends": monthly_trends,
                "yearly_trends": yearly_trends
            }
            
            logger.info("Crime type trends analyzed")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing crime type trends: {e}")
            raise
    
    def get_model_performance(self) -> Dict[str, any]:
        """
        Get information about the crime type classification model.
        
        Returns:
            Dict[str, any]: Model performance information
        """
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            # Get model information
            model_info = {
                "model_type": type(self.model).__name__,
                "model_loaded": True,
                "feature_names": getattr(self.model, 'feature_names_in_', None),
                "n_classes": len(self.model.classes_) if hasattr(self.model, 'classes_') else None,
                "classes": self.model.classes_.tolist() if hasattr(self.model, 'classes_') else None
            }
            
            # Try to get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                model_info["feature_importance"] = self.model.feature_importances_.tolist()
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {"error": str(e)}
    
    def get_classification_summary(self, crime_data: pd.DataFrame) -> Dict[str, any]:
        """
        Get a comprehensive crime type classification summary.
        
        Args:
            crime_data (pd.DataFrame): Crime data with classification features
                
        Returns:
            Dict[str, any]: Classification summary
        """
        try:
            logger.info("Generating crime type classification summary...")
            
            # Classify crime types
            classification_results = self.classify_crime_types(crime_data)
            
            # Analyze distribution
            distribution_results = self.analyze_crime_type_distribution(crime_data)
            
            # Analyze trends
            trend_results = self.get_crime_type_trends(crime_data)
            
            # Get model performance
            model_performance = self.get_model_performance()
            
            summary = {
                "classification_results": classification_results,
                "distribution_analysis": distribution_results,
                "trend_analysis": trend_results,
                "model_performance": model_performance
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating classification summary: {e}")
            raise 