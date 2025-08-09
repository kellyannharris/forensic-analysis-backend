# Crime & Forensic Analysis System - API Documentation

## Overview

This package provides comprehensive crime analysis capabilities for API consumption, including crime rate prediction, criminal network analysis, spatial crime mapping, temporal pattern recognition, and crime type classification.

**Author:** Kelly-Ann Harris  
**Version:** 1.0.0  
**Date:** 2024

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Data Format Requirements](#data-format-requirements)
3. [Crime Rate Predictor](#crime-rate-predictor)
4. [Criminal Network Analyzer](#criminal-network-analyzer)
5. [Spatial Crime Mapper](#spatial-crime-mapper)
6. [Temporal Pattern Analyzer](#temporal-pattern-analyzer)
7. [Crime Type Classifier](#crime-type-classifier)
8. [Data Analyzer](#data-analyzer)
9. [Error Handling](#error-handling)
10. [Performance Considerations](#performance-considerations)
11. [Example API Integration](#example-api-integration)

## Installation & Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Required Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
networkx>=2.8.0
statsmodels>=0.13.0
prophet>=1.1.0
xgboost>=1.6.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### Package Structure

```
structured_data_models/
├── __init__.py
├── crime_rate_predictor.py
├── criminal_network_analyzer.py
├── spatial_crime_mapper.py
├── temporal_pattern_analyzer.py
├── crime_type_classifier.py
├── data_analyzer.py
├── models/
│   ├── gradient_boosting_model.joblib
│   ├── random_forest_model.joblib
│   ├── xgboost_model.joblib
│   ├── prophet_model.joblib
│   ├── scaler.joblib
│   ├── label_encoders.joblib
│   └── feature_importance.csv
├── crime_type_rf_model.pkl
└── API_DOCUMENTATION.md
```

## Data Format Requirements

### Standard Crime Data Schema

All analysis modules expect crime data in the following format:

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

### Required Columns by Module

| Module | Required Columns | Optional Columns |
|--------|------------------|------------------|
| Crime Rate Predictor | DATE OCC, TIME OCC, AREA, LAT, LON, Vict Age, Premis Cd, Weapon Used Cd | All others |
| Criminal Network Analyzer | LOCATION, Mocodes | All others |
| Spatial Crime Mapper | LAT, LON, Crm Cd Desc | All others |
| Temporal Pattern Analyzer | DATE OCC, TIME OCC | All others |
| Crime Type Classifier | AREA, TIME OCC, Vict Age, Premis Cd, Weapon Used Cd | All others |
| Data Analyzer | Crm Cd Desc, AREA NAME, Vict Age | All others |

## Crime Rate Predictor

### Overview
Predicts crime rates using pre-trained machine learning models for both spatial (area-based) and temporal (time-based) predictions.

### Usage

```python
from structured_data_models import CrimeRatePredictor
import pandas as pd

# Initialize predictor
predictor = CrimeRatePredictor(models_dir="models")

# Spatial crime rate prediction
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
print(spatial_predictions)
# Output: {
#     "random_forest": [0.85, 0.92],
#     "gradient_boosting": [0.87, 0.89],
#     "xgboost": [0.86, 0.91]
# }

# Temporal crime rate prediction
historical_data = pd.DataFrame({
    'DATE OCC': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'TIME OCC': ['1200', '1400', '1600'],
    'AREA': [1, 1, 1]
})

temporal_predictions = predictor.predict_temporal_crime_rate(historical_data, days_ahead=5)
print(temporal_predictions)
# Output: {
#     "dates": ["2024-01-04", "2024-01-05", ...],
#     "predictions": [100.5, 95.2, ...],
#     "lower_bound": [90.1, 85.3, ...],
#     "upper_bound": [110.9, 105.1, ...]
# }
```

### API Endpoints (Example)

```python
# POST /api/predict/spatial
{
    "area_features": [
        {
            "DATE OCC": "2023-01-15",
            "TIME OCC": "1430",
            "AREA": 1,
            "LAT": 34.0522,
            "LON": -118.2437,
            "Vict Age": 25,
            "Premis Cd": 101,
            "Weapon Used Cd": 200
        }
    ]
}

# POST /api/predict/temporal
{
    "historical_data": [...],
    "days_ahead": 5
}
```

## Criminal Network Analyzer

### Overview
Analyzes criminal networks using graph theory, including centrality analysis, community detection, and network metrics.

### Usage

```python
from structured_data_models import CriminalNetworkAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = CriminalNetworkAnalyzer()

# Build network from crime data
crime_data = pd.DataFrame({
    'LOCATION': ['123 MAIN ST', '456 OAK AVE', '123 MAIN ST'],
    'Mocodes': ['1234,5678', '1234', '5678,9012']
})

network_stats = analyzer.build_network_from_data(crime_data)
print(network_stats)
# Output: {"nodes": 5, "edges": 4}

# Analyze centrality
centrality_results = analyzer.analyze_centrality(top_n=10)
print(centrality_results)
# Output: {
#     "degree_centrality": [{"node": "1234", "score": 0.8}, ...],
#     "betweenness_centrality": [{"node": "1234", "score": 0.6}, ...],
#     "eigenvector_centrality": [{"node": "1234", "score": 0.7}, ...]
# }

# Detect communities
community_results = analyzer.detect_communities(algorithm="greedy_modularity")
print(community_results)
# Output: {
#     "communities": [["node1", "node2"], ["node3", "node4"]],
#     "statistics": {
#         "num_communities": 2,
#         "largest_community_size": 3,
#         "modularity": 0.75
#     }
# }

# Get comprehensive summary
summary = analyzer.get_network_summary()
```

### API Endpoints (Example)

```python
# POST /api/network/build
{
    "crime_data": [
        {
            "LOCATION": "123 MAIN ST",
            "Mocodes": "1234,5678"
        }
    ]
}

# GET /api/network/centrality?top_n=10
# GET /api/network/communities?algorithm=greedy_modularity
# GET /api/network/summary
```

## Spatial Crime Mapper

### Overview
Analyzes spatial patterns in crime data, including hotspot detection, clustering analysis, and spatial statistics.

### Usage

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
print(data_stats)
# Output: {
#     "total_records": 3,
#     "valid_records": 3,
#     "invalid_records": 0,
#     "lat_range": [34.0522, 34.0722],
#     "lon_range": [-118.2637, -118.2437]
# }

# Perform clustering
clustering_results = mapper.perform_clustering(n_clusters=3, algorithm="kmeans")
print(clustering_results)
# Output: {
#     "cluster_centers": [[34.0522, -118.2437], [34.0622, -118.2537]],
#     "cluster_assignments": [0, 1, 0],
#     "statistics": {
#         "num_clusters": 2,
#         "cluster_sizes": [2, 1],
#         "inertia": 0.001
#     }
# }

# Analyze hotspots
hotspot_results = mapper.analyze_hotspots(crime_data, radius_km=1.0)
print(hotspot_results)
# Output: {
#     "hotspots": [
#         {
#             "center": [34.0522, -118.2437],
#             "radius_km": 1.0,
#             "crime_count": 2,
#             "crime_types": ["THEFT", "BURGLARY"],
#             "density": 0.64
#         }
#     ],
#     "statistics": {
#         "total_hotspots": 1,
#         "total_crimes_in_hotspots": 2,
#         "average_crimes_per_hotspot": 2.0
#     }
# }

# Get comprehensive summary
summary = mapper.get_spatial_summary(crime_data)
```

### API Endpoints (Example)

```python
# POST /api/spatial/prepare
{
    "crime_data": [
        {
            "LAT": 34.0522,
            "LON": -118.2437,
            "Crm Cd Desc": "THEFT"
        }
    ]
}

# POST /api/spatial/cluster
{
    "n_clusters": 3,
    "algorithm": "kmeans"
}

# POST /api/spatial/hotspots
{
    "crime_data": [...],
    "radius_km": 1.0
}

# GET /api/spatial/summary
```

## Temporal Pattern Analyzer

### Overview
Analyzes temporal patterns in crime data, including time series analysis, seasonality detection, and trend analysis.

### Usage

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
print(ts_stats)
# Output: {
#     "total_records": 3,
#     "time_periods": 3,
#     "date_range": ["2023-01-01", "2023-01-03"],
#     "mean_crimes_per_period": 1.0
# }

# Detect seasonality
seasonality_results = analyzer.detect_seasonality()
print(seasonality_results)
# Output: {
#     "seasonal_strength": 0.8,
#     "trend_strength": 0.6,
#     "noise_strength": 0.2,
#     "seasonal_pattern": [1.2, 0.8, 1.1],
#     "is_stationary": true
# }

# Fit ARIMA model
arima_results = analyzer.fit_arima_model(order=(1, 1, 0))
print(arima_results)
# Output: {
#     "aic": 1234.56,
#     "bic": 1256.78,
#     "log_likelihood": -612.34,
#     "residuals_stats": {
#         "mean": 0.0,
#         "std": 1.2,
#         "skewness": 0.1,
#         "kurtosis": 3.2
#     }
# }

# Forecast future
forecast_results = analyzer.forecast_future(periods=5)
print(forecast_results)
# Output: {
#     "dates": ["2024-01-04", "2024-01-05", ...],
#     "forecast": [100.5, 95.2, ...],
#     "lower_bound": [90.1, 85.3, ...],
#     "upper_bound": [110.9, 105.1, ...]
# }

# Get comprehensive summary
summary = analyzer.get_temporal_summary(crime_data)
```

### API Endpoints (Example)

```python
# POST /api/temporal/prepare
{
    "crime_data": [...],
    "time_unit": "daily"
}

# GET /api/temporal/seasonality
# GET /api/temporal/arima?order=1,1,0
# POST /api/temporal/forecast
{
    "periods": 5,
    "confidence_level": 0.95
}

# GET /api/temporal/summary
```

## Crime Type Classifier

### Overview
Classifies crime types using pre-trained machine learning models and analyzes crime type distributions and trends.

### Usage

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
print(classification_results)
# Output: {
#     "predictions": ["THEFT", "ASSAULT"],
#     "probabilities": [[0.8, 0.15, 0.05], [0.1, 0.8, 0.1]],
#     "confidence_scores": [0.85, 0.8],
#     "num_predictions": 2
# }

# Analyze distribution
distribution_results = classifier.analyze_crime_type_distribution(crime_data)
print(distribution_results)
# Output: {
#     "crime_type_counts": [
#         {"crime_type": "THEFT", "count": 1000, "percentage": 25.0}
#     ],
#     "statistics": {
#         "total_crimes": 4000,
#         "unique_crime_types": 15,
#         "most_common_crime": "THEFT"
#     }
# }

# Get comprehensive summary
summary = classifier.get_classification_summary(crime_data)
```

### API Endpoints (Example)

```python
# POST /api/classify/crime-types
{
    "crime_data": [
        {
            "AREA": 1,
            "TIME OCC": "1430",
            "Vict Age": 25,
            "Premis Cd": 101,
            "Weapon Used Cd": 200
        }
    ]
}

# GET /api/classify/distribution
# GET /api/classify/trends
# GET /api/classify/summary
```

## Data Analyzer

### Overview
Provides comprehensive data analysis capabilities for crime data, including crime type analysis, temporal trends, spatial distribution, and victim demographics.

### Usage

```python
from structured_data_models import LAPDCrimeDataAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = LAPDCrimeDataAnalyzer()

# Load and analyze data
crime_data = pd.DataFrame({
    'Crm Cd Desc': ['THEFT', 'ASSAULT', 'BURGLARY'],
    'AREA NAME': ['Central', 'West', 'Central'],
    'Vict Age': [25, 30, 35],
    'DATE OCC': ['2023-01-01', '2023-01-02', '2023-01-03']
})

# Generate comprehensive analysis
report = analyzer.generate_analysis_report(crime_data)
print(report)
# Output: {
#     "crime_types": {"THEFT": 1, "ASSAULT": 1, "BURGLARY": 1},
#     "temporal_trends": {2023: 3},
#     "spatial_distribution": {"Central": 2, "West": 1},
#     "victim_demographics": {
#         "min": 25,
#         "max": 35,
#         "mean": 30.0,
#         "median": 30.0
#     }
# }

# Save analysis report
analyzer.save_analysis_report(report, "analysis_report.json")
```

### API Endpoints (Example)

```python
# POST /api/analyze/report
{
    "crime_data": [...]
}

# GET /api/analyze/crime-types
# GET /api/analyze/temporal-trends
# GET /api/analyze/spatial-distribution
# GET /api/analyze/victim-demographics
```

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

### Error Handling Best Practices

```python
try:
    predictor = CrimeRatePredictor()
    results = predictor.predict_spatial_crime_rate(data)
except ValueError as e:
    # Handle validation errors
    return {"error": str(e), "status": "validation_error"}
except FileNotFoundError as e:
    # Handle missing model files
    return {"error": "Model files not found", "status": "model_error"}
except Exception as e:
    # Handle unexpected errors
    return {"error": "Internal server error", "status": "server_error"}
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

## Example API Integration

### FastAPI Integration Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from structured_data_models import (
    CrimeRatePredictor, 
    CriminalNetworkAnalyzer,
    SpatialCrimeMapper,
    TemporalPatternAnalyzer,
    CrimeTypeClassifier,
    LAPDCrimeDataAnalyzer
)

app = FastAPI(title="Crime Analysis API")

# Initialize models
predictor = CrimeRatePredictor()
network_analyzer = CriminalNetworkAnalyzer()
spatial_mapper = SpatialCrimeMapper()
temporal_analyzer = TemporalPatternAnalyzer()
classifier = CrimeTypeClassifier()
data_analyzer = LAPDCrimeDataAnalyzer()

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

@app.post("/network/analyze")
async def analyze_criminal_network(request: CrimeDataRequest):
    try:
        df = pd.DataFrame(request.crime_data)
        network_stats = network_analyzer.build_network_from_data(df)
        summary = network_analyzer.get_network_summary()
        return {"status": "success", "network_stats": network_stats, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/spatial/analyze")
async def analyze_spatial_patterns(request: CrimeDataRequest):
    try:
        df = pd.DataFrame(request.crime_data)
        summary = spatial_mapper.get_spatial_summary(df)
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/temporal/analyze")
async def analyze_temporal_patterns(request: CrimeDataRequest):
    try:
        df = pd.DataFrame(request.crime_data)
        summary = temporal_analyzer.get_temporal_summary(df)
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/classify/crime-types")
async def classify_crime_types(request: CrimeDataRequest):
    try:
        df = pd.DataFrame(request.crime_data)
        summary = classifier.get_classification_summary(df)
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/comprehensive")
async def comprehensive_analysis(request: CrimeDataRequest):
    try:
        df = pd.DataFrame(request.crime_data)
        report = data_analyzer.generate_analysis_report(df)
        return {"status": "success", "report": report}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Flask Integration Example

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

For additional support, please refer to the project documentation or contact the development team. 