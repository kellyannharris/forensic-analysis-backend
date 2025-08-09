# Structured Data Models - Crime & Forensic Analysis System

## Overview

This package provides comprehensive crime analysis capabilities for API consumption, including crime rate prediction, criminal network analysis, spatial crime mapping, temporal pattern recognition, and crime type classification. All models are pre-trained and ready for production use.

**Author:** Kelly-Ann Harris  
**Version:** 1.0.0  
**Date:** 2024

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Import the package
from structured_data_models import (
    CrimeRatePredictor,
    CriminalNetworkAnalyzer,
    SpatialCrimeMapper,
    TemporalPatternAnalyzer,
    CrimeTypeClassifier,
    LAPDCrimeDataAnalyzer
)
```

### Basic Usage

```python
import pandas as pd
from structured_data_models import CrimeRatePredictor

# Initialize predictor
predictor = CrimeRatePredictor()

# Prepare sample data
crime_data = pd.DataFrame({
    'DATE OCC': ['2023-01-15', '2023-01-16'],
    'TIME OCC': ['1430', '2200'],
    'AREA': [1, 2],
    'LAT': [34.0522, 34.0622],
    'LON': [-118.2437, -118.2537],
    'Vict Age': [25, 30],
    'Premis Cd': [101, 102],
    'Weapon Used Cd': [200, 201]
})

# Make predictions
predictions = predictor.predict_spatial_crime_rate(crime_data)
print(predictions)
```

## Available Models

### 1. Crime Rate Predictor
- **Purpose**: Predict crime rates using spatial and temporal models
- **Models**: Random Forest, Gradient Boosting, XGBoost, Prophet
- **Input**: Crime data with location and temporal features
- **Output**: Predicted crime rates and confidence intervals

### 2. Criminal Network Analyzer
- **Purpose**: Analyze criminal networks using graph theory
- **Features**: Centrality analysis, community detection, key player identification
- **Input**: Crime data with location and modus operandi codes
- **Output**: Network metrics, communities, and key players

### 3. Spatial Crime Mapper
- **Purpose**: Analyze spatial patterns and detect crime hotspots
- **Features**: Clustering analysis, hotspot detection, spatial statistics
- **Input**: Crime data with latitude/longitude coordinates
- **Output**: Crime clusters, hotspots, and spatial metrics

### 4. Temporal Pattern Analyzer
- **Purpose**: Analyze temporal patterns and forecast future crime rates
- **Features**: Time series analysis, seasonality detection, ARIMA forecasting
- **Input**: Crime data with date/time information
- **Output**: Temporal patterns, forecasts, and trend analysis

### 5. Crime Type Classifier
- **Purpose**: Classify crime types using machine learning
- **Model**: Random Forest classifier
- **Input**: Crime data with relevant features
- **Output**: Crime type predictions and confidence scores

### 6. Data Analyzer
- **Purpose**: Comprehensive crime data analysis
- **Features**: Crime type analysis, temporal trends, spatial distribution, demographics
- **Input**: Crime data with various attributes
- **Output**: Statistical analysis and insights

## Data Requirements

### Standard Crime Data Schema

All models expect crime data in the following format:

```python
{
    'DATE OCC': '2023-01-15',           # Date of occurrence (YYYY-MM-DD)
    'TIME OCC': '1430',                 # Time of occurrence (HHMM format)
    'AREA': 1,                          # Area code (integer)
    'AREA NAME': 'Central',             # Area name (string)
    'Crm Cd': 510,                      # Crime code (integer)
    'Crm Cd Desc': 'VEHICLE - STOLEN',  # Crime description (string)
    'Vict Age': 25,                     # Victim age (integer)
    'Vict Sex': 'F',                    # Victim sex (string)
    'Vict Descent': 'W',                # Victim descent (string)
    'Premis Cd': 101,                   # Premise code (integer)
    'Premis Desc': 'STREET',            # Premise description (string)
    'Weapon Used Cd': 200,              # Weapon code (integer)
    'Weapon Desc': 'UNKNOWN WEAPON',    # Weapon description (string)
    'Status': 'IC',                     # Status (string)
    'Status Desc': 'Invest Cont',       # Status description (string)
    'LOCATION': '123 MAIN ST',          # Location (string)
    'LAT': 34.0522,                     # Latitude (float)
    'LON': -118.2437,                   # Longitude (float)
    'Mocodes': '1234,5678'              # Modus operandi codes (string, comma-separated)
}
```

### Required Columns by Model

| Model | Required Columns | Optional Columns |
|-------|------------------|------------------|
| Crime Rate Predictor | DATE OCC, TIME OCC, AREA, LAT, LON, Vict Age, Premis Cd, Weapon Used Cd | All others |
| Criminal Network Analyzer | LOCATION, Mocodes | All others |
| Spatial Crime Mapper | LAT, LON, Crm Cd Desc | All others |
| Temporal Pattern Analyzer | DATE OCC, TIME OCC | All others |
| Crime Type Classifier | AREA, TIME OCC, Vict Age, Premis Cd, Weapon Used Cd | All others |
| Data Analyzer | Crm Cd Desc, AREA NAME, Vict Age | All others |

## Usage Examples

### Crime Rate Prediction

```python
from structured_data_models import CrimeRatePredictor
import pandas as pd

# Initialize predictor
predictor = CrimeRatePredictor()

# Spatial prediction
area_features = pd.DataFrame({
    'DATE OCC': ['2023-01-15', '2023-01-16'],
    'TIME OCC': ['1430', '2200'],
    'AREA': [1, 2],
    'LAT': [34.0522, 34.0622],
    'LON': [-118.2437, -118.2537],
    'Vict Age': [25, 30],
    'Premis Cd': [101, 102],
    'Weapon Used Cd': [200, 201]
})

spatial_predictions = predictor.predict_spatial_crime_rate(area_features)
print("Spatial Predictions:", spatial_predictions)

# Temporal prediction
historical_data = pd.DataFrame({
    'DATE OCC': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'TIME OCC': ['1200', '1400', '1600'],
    'AREA': [1, 1, 1]
})

temporal_predictions = predictor.predict_temporal_crime_rate(historical_data, days_ahead=5)
print("Temporal Predictions:", temporal_predictions)
```

### Criminal Network Analysis

```python
from structured_data_models import CriminalNetworkAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = CriminalNetworkAnalyzer()

# Build network
crime_data = pd.DataFrame({
    'LOCATION': ['123 MAIN ST', '456 OAK AVE', '123 MAIN ST'],
    'Mocodes': ['1234,5678', '1234', '5678,9012']
})

network_stats = analyzer.build_network_from_data(crime_data)
print("Network Stats:", network_stats)

# Analyze centrality
centrality_results = analyzer.analyze_centrality(top_n=10)
print("Centrality Analysis:", centrality_results)

# Detect communities
community_results = analyzer.detect_communities(algorithm="greedy_modularity")
print("Community Detection:", community_results)

# Get comprehensive summary
summary = analyzer.get_network_summary()
print("Network Summary:", summary)
```

### Spatial Crime Mapping

```python
from structured_data_models import SpatialCrimeMapper
import pandas as pd

# Initialize mapper
mapper = SpatialCrimeMapper()

# Prepare spatial data
crime_data = pd.DataFrame({
    'LAT': [34.0522, 34.0622, 34.0722],
    'LON': [-118.2437, -118.2537, -118.2637],
    'Crm Cd Desc': ['THEFT', 'ASSAULT', 'BURGLARY']
})

data_stats = mapper.prepare_spatial_data(crime_data)
print("Data Stats:", data_stats)

# Perform clustering
clustering_results = mapper.perform_clustering(n_clusters=3, algorithm="kmeans")
print("Clustering Results:", clustering_results)

# Analyze hotspots
hotspot_results = mapper.analyze_hotspots(crime_data, radius_km=1.0)
print("Hotspot Analysis:", hotspot_results)

# Get comprehensive summary
summary = mapper.get_spatial_summary(crime_data)
print("Spatial Summary:", summary)
```

### Temporal Pattern Analysis

```python
from structured_data_models import TemporalPatternAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = TemporalPatternAnalyzer()

# Prepare time series data
crime_data = pd.DataFrame({
    'DATE OCC': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'TIME OCC': ['1200', '1400', '1600']
})

ts_stats = analyzer.prepare_time_series_data(crime_data, time_unit="daily")
print("Time Series Stats:", ts_stats)

# Detect seasonality
seasonality_results = analyzer.detect_seasonality()
print("Seasonality Analysis:", seasonality_results)

# Fit ARIMA model
arima_results = analyzer.fit_arima_model(order=(1, 1, 0))
print("ARIMA Results:", arima_results)

# Forecast future
forecast_results = analyzer.forecast_future(periods=5)
print("Forecast Results:", forecast_results)

# Get comprehensive summary
summary = analyzer.get_temporal_summary(crime_data)
print("Temporal Summary:", summary)
```

### Crime Type Classification

```python
from structured_data_models import CrimeTypeClassifier
import pandas as pd

# Initialize classifier
classifier = CrimeTypeClassifier(model_path="crime_type_rf_model.pkl")

# Classify crime types
crime_data = pd.DataFrame({
    'AREA': [1, 2],
    'TIME OCC': ['1430', '2200'],
    'Vict Age': [25, 30],
    'Premis Cd': [101, 102],
    'Weapon Used Cd': [200, 201]
})

classification_results = classifier.classify_crime_types(crime_data)
print("Classification Results:", classification_results)

# Analyze distribution
distribution_results = classifier.analyze_crime_type_distribution(crime_data)
print("Distribution Analysis:", distribution_results)

# Get comprehensive summary
summary = classifier.get_classification_summary(crime_data)
print("Classification Summary:", summary)
```

### Comprehensive Data Analysis

```python
from structured_data_models import LAPDCrimeDataAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = LAPDCrimeDataAnalyzer()

# Generate comprehensive analysis
crime_data = pd.DataFrame({
    'Crm Cd Desc': ['THEFT', 'ASSAULT', 'BURGLARY'],
    'AREA NAME': ['Central', 'West', 'Central'],
    'Vict Age': [25, 30, 35],
    'DATE OCC': ['2023-01-01', '2023-01-02', '2023-01-03']
})

report = analyzer.generate_analysis_report(crime_data)
print("Analysis Report:", report)

# Save analysis report
analyzer.save_analysis_report(report, "analysis_report.json")
```

## API Integration

### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from structured_data_models import CrimeRatePredictor

app = FastAPI(title="Crime Analysis API")
predictor = CrimeRatePredictor()

class CrimeDataRequest(BaseModel):
    crime_data: List[Dict[str, Any]]

@app.post("/predict/spatial")
async def predict_spatial_crime_rate(request: CrimeDataRequest):
    try:
        df = pd.DataFrame(request.crime_data)
        results = predictor.predict_spatial_crime_rate(df)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Flask Example

```python
from flask import Flask, request, jsonify
import pandas as pd
from structured_data_models import CrimeRatePredictor

app = Flask(__name__)
predictor = CrimeRatePredictor()

@app.route('/predict/spatial', methods=['POST'])
def predict_spatial():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['crime_data'])
        results = predictor.predict_spatial_crime_rate(df)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Performance Considerations

### Memory Usage
- **Crime Rate Predictor**: ~500MB (includes multiple ML models)
- **Criminal Network Analyzer**: ~200MB (graph analysis)
- **Spatial Crime Mapper**: ~100MB (clustering algorithms)
- **Temporal Pattern Analyzer**: ~150MB (time series models)
- **Crime Type Classifier**: ~100MB (classification model)

### Processing Time Estimates

| Operation | Small Dataset (1K records) | Medium Dataset (10K records) | Large Dataset (100K records) |
|-----------|---------------------------|------------------------------|------------------------------|
| Spatial Prediction | <1s | 2-5s | 10-30s |
| Network Analysis | 1-2s | 5-10s | 30-60s |
| Spatial Clustering | <1s | 2-5s | 10-20s |
| Temporal Analysis | 1-2s | 5-15s | 30-90s |
| Crime Classification | <1s | 2-5s | 10-25s |

### Optimization Tips
1. **Batch Processing**: Process data in batches for large datasets
2. **Caching**: Cache model results for repeated queries
3. **Parallel Processing**: Use multiple workers for independent analyses
4. **Data Sampling**: Use representative samples for exploratory analysis

## Error Handling

### Common Error Types
1. **DataFormatError**: Invalid data format or missing required columns
2. **ModelLoadError**: Failed to load pre-trained models
3. **PredictionError**: Error during prediction process
4. **ValidationError**: Input validation failed

### Error Response Format
```python
{
    "error": {
        "type": "DataFormatError",
        "message": "Missing required column: 'DATE OCC'",
        "details": {
            "required_columns": ["DATE OCC", "TIME OCC"],
            "provided_columns": ["TIME OCC", "AREA"]
        }
    },
    "status": "error",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## Model Information

### Pre-trained Models Included
- **Gradient Boosting Model**: For spatial crime rate prediction
- **Random Forest Model**: For spatial crime rate prediction
- **XGBoost Model**: For spatial crime rate prediction
- **Prophet Model**: For temporal crime rate prediction
- **Random Forest Classifier**: For crime type classification
- **Feature Scaler**: For feature normalization
- **Label Encoders**: For categorical variable encoding

### Model Performance
- **Spatial Prediction Models**: R² scores > 0.8
- **Temporal Prediction Model**: MAPE < 15%
- **Crime Type Classifier**: Accuracy > 85%

## Support and Maintenance

### Model Updates
- Models are trained on LAPD crime data and should be retrained periodically
- Feature importance and model performance should be monitored
- New crime types may require model retraining

### Performance Monitoring
- Monitor API response times
- Track memory usage
- Log prediction accuracy metrics
- Monitor error rates and types

### Troubleshooting
1. **Model Loading Issues**: Check file paths and permissions
2. **Memory Issues**: Reduce batch sizes or use data sampling
3. **Performance Issues**: Consider model optimization or hardware upgrades
4. **Data Format Issues**: Validate input data against required schema

## Documentation

For detailed API documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

## License

This project is part of the Crime & Forensic Analysis System capstone project.

## Contact

For questions and support, please contact the development team. 