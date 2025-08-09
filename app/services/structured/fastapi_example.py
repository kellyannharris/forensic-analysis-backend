"""
FastAPI Example for Crime Analysis System
This demonstrates how to integrate the structured_data_models package with FastAPI
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime

# Import our crime analysis modules
from crime_rate_predictor import CrimeRatePredictor
from criminal_network_analyzer import CriminalNetworkAnalyzer
from spatial_crime_mapper import SpatialCrimeMapper
from temporal_pattern_analyzer import TemporalPatternAnalyzer
from crime_type_classifier import CrimeTypeClassifier
from data_analyzer import LAPDCrimeDataAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crime Analysis API",
    description="Comprehensive crime analysis system with machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analysis modules
try:
    predictor = CrimeRatePredictor()
    network_analyzer = CriminalNetworkAnalyzer()
    spatial_mapper = SpatialCrimeMapper()
    temporal_analyzer = TemporalPatternAnalyzer()
    classifier = CrimeTypeClassifier()
    data_analyzer = LAPDCrimeDataAnalyzer()
    logger.info("All analysis modules initialized successfully")
except Exception as e:
    logger.error(f"Error initializing modules: {e}")
    raise

# Pydantic models for request/response validation
class CrimeDataRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="List of crime records")
    
class SpatialPredictionRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Crime data for spatial prediction")
    
class TemporalPredictionRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Historical crime data")
    days_ahead: int = Field(default=5, description="Number of days to forecast")
    
class NetworkAnalysisRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Crime data with location and modus operandi")
    top_n: int = Field(default=10, description="Number of top nodes to return")
    
class SpatialAnalysisRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Crime data with coordinates")
    n_clusters: int = Field(default=5, description="Number of clusters")
    radius_km: float = Field(default=1.0, description="Radius for hotspot analysis")
    
class ClassificationRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Crime data for classification")

# Response models
class PredictionResponse(BaseModel):
    status: str
    results: Dict[str, Any]
    timestamp: str
    processing_time: float

class AnalysisResponse(BaseModel):
    status: str
    summary: Dict[str, Any]
    timestamp: str
    processing_time: float

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "modules_loaded": {
            "crime_rate_predictor": True,
            "criminal_network_analyzer": True,
            "spatial_crime_mapper": True,
            "temporal_pattern_analyzer": True,
            "crime_type_classifier": True,
            "data_analyzer": True
        }
    }

# Crime Rate Prediction Endpoints
@app.post("/predict/spatial", response_model=PredictionResponse)
async def predict_spatial_crime_rate(request: SpatialPredictionRequest):
    """
    Predict crime rates for different areas using spatial models
    """
    start_time = datetime.now()
    try:
        df = pd.DataFrame(request.crime_data)
        results = predictor.predict_spatial_crime_rate(df)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            status="success",
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error in spatial prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/temporal", response_model=PredictionResponse)
async def predict_temporal_crime_rate(request: TemporalPredictionRequest):
    """
    Predict future crime rates using temporal models
    """
    start_time = datetime.now()
    try:
        df = pd.DataFrame(request.crime_data)
        results = predictor.predict_temporal_crime_rate(df, request.days_ahead)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            status="success",
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error in temporal prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Network Analysis Endpoints
@app.post("/network/build")
async def build_criminal_network(request: NetworkAnalysisRequest):
    """
    Build a criminal network from crime data
    """
    try:
        df = pd.DataFrame(request.crime_data)
        network_stats = network_analyzer.build_network_from_data(df)
        return {"status": "success", "network_stats": network_stats}
    except Exception as e:
        logger.error(f"Error building network: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/network/centrality")
async def analyze_network_centrality(top_n: int = 10):
    """
    Analyze network centrality measures
    """
    try:
        results = network_analyzer.analyze_centrality(top_n=top_n)
        return {"status": "success", "centrality_analysis": results}
    except Exception as e:
        logger.error(f"Error analyzing centrality: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/network/communities")
async def detect_communities(algorithm: str = "greedy_modularity"):
    """
    Detect communities in the criminal network
    """
    try:
        results = network_analyzer.detect_communities(algorithm=algorithm)
        return {"status": "success", "community_detection": results}
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/network/summary", response_model=AnalysisResponse)
async def get_network_summary():
    """
    Get comprehensive network analysis summary
    """
    start_time = datetime.now()
    try:
        summary = network_analyzer.get_network_summary()
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="success",
            summary=summary,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error getting network summary: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Spatial Analysis Endpoints
@app.post("/spatial/prepare")
async def prepare_spatial_data(request: SpatialAnalysisRequest):
    """
    Prepare spatial data for analysis
    """
    try:
        df = pd.DataFrame(request.crime_data)
        stats = spatial_mapper.prepare_spatial_data(df)
        return {"status": "success", "data_stats": stats}
    except Exception as e:
        logger.error(f"Error preparing spatial data: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/spatial/cluster")
async def perform_spatial_clustering(request: SpatialAnalysisRequest):
    """
    Perform spatial clustering on crime data
    """
    try:
        df = pd.DataFrame(request.crime_data)
        spatial_mapper.prepare_spatial_data(df)
        results = spatial_mapper.perform_clustering(n_clusters=request.n_clusters)
        return {"status": "success", "clustering_results": results}
    except Exception as e:
        logger.error(f"Error performing clustering: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/spatial/hotspots")
async def analyze_hotspots(request: SpatialAnalysisRequest):
    """
    Analyze crime hotspots
    """
    try:
        df = pd.DataFrame(request.crime_data)
        results = spatial_mapper.analyze_hotspots(df, radius_km=request.radius_km)
        return {"status": "success", "hotspot_analysis": results}
    except Exception as e:
        logger.error(f"Error analyzing hotspots: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/spatial/summary", response_model=AnalysisResponse)
async def get_spatial_summary(request: SpatialAnalysisRequest):
    """
    Get comprehensive spatial analysis summary
    """
    start_time = datetime.now()
    try:
        df = pd.DataFrame(request.crime_data)
        summary = spatial_mapper.get_spatial_summary(df)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="success",
            summary=summary,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error getting spatial summary: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Temporal Analysis Endpoints
@app.post("/temporal/prepare")
async def prepare_temporal_data(request: CrimeDataRequest, time_unit: str = "daily"):
    """
    Prepare time series data for analysis
    """
    try:
        df = pd.DataFrame(request.crime_data)
        stats = temporal_analyzer.prepare_time_series_data(df, time_unit=time_unit)
        return {"status": "success", "time_series_stats": stats}
    except Exception as e:
        logger.error(f"Error preparing temporal data: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/temporal/seasonality")
async def detect_seasonality(period: Optional[int] = None):
    """
    Detect seasonality in time series data
    """
    try:
        results = temporal_analyzer.detect_seasonality(period=period)
        return {"status": "success", "seasonality_analysis": results}
    except Exception as e:
        logger.error(f"Error detecting seasonality: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/temporal/arima")
async def fit_arima_model(order: str = "5,1,0"):
    """
    Fit ARIMA model to time series data
    """
    try:
        p, d, q = map(int, order.split(','))
        results = temporal_analyzer.fit_arima_model(order=(p, d, q))
        return {"status": "success", "arima_results": results}
    except Exception as e:
        logger.error(f"Error fitting ARIMA model: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/temporal/forecast")
async def forecast_future(request: TemporalPredictionRequest, confidence_level: float = 0.95):
    """
    Forecast future crime rates
    """
    try:
        df = pd.DataFrame(request.crime_data)
        temporal_analyzer.prepare_time_series_data(df)
        temporal_analyzer.fit_arima_model()
        results = temporal_analyzer.forecast_future(request.days_ahead, confidence_level)
        return {"status": "success", "forecast_results": results}
    except Exception as e:
        logger.error(f"Error forecasting: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/temporal/summary", response_model=AnalysisResponse)
async def get_temporal_summary(request: CrimeDataRequest):
    """
    Get comprehensive temporal analysis summary
    """
    start_time = datetime.now()
    try:
        df = pd.DataFrame(request.crime_data)
        summary = temporal_analyzer.get_temporal_summary(df)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="success",
            summary=summary,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error getting temporal summary: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Crime Classification Endpoints
@app.post("/classify/crime-types", response_model=PredictionResponse)
async def classify_crime_types(request: ClassificationRequest):
    """
    Classify crime types using machine learning
    """
    start_time = datetime.now()
    try:
        df = pd.DataFrame(request.crime_data)
        results = classifier.classify_crime_types(df)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            status="success",
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error classifying crime types: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/classify/distribution")
async def analyze_crime_distribution(request: ClassificationRequest):
    """
    Analyze crime type distribution
    """
    try:
        df = pd.DataFrame(request.crime_data)
        results = classifier.analyze_crime_type_distribution(df)
        return {"status": "success", "distribution_analysis": results}
    except Exception as e:
        logger.error(f"Error analyzing distribution: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/classify/trends")
async def analyze_crime_trends(request: ClassificationRequest):
    """
    Analyze crime type trends over time
    """
    try:
        df = pd.DataFrame(request.crime_data)
        results = classifier.get_crime_type_trends(df)
        return {"status": "success", "trend_analysis": results}
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/classify/summary", response_model=AnalysisResponse)
async def get_classification_summary(request: ClassificationRequest):
    """
    Get comprehensive classification summary
    """
    start_time = datetime.now()
    try:
        df = pd.DataFrame(request.crime_data)
        summary = classifier.get_classification_summary(df)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="success",
            summary=summary,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error getting classification summary: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Data Analysis Endpoints
@app.post("/analyze/report", response_model=AnalysisResponse)
async def generate_analysis_report(request: CrimeDataRequest):
    """
    Generate comprehensive crime analysis report
    """
    start_time = datetime.now()
    try:
        df = pd.DataFrame(request.crime_data)
        report = data_analyzer.generate_analysis_report(df)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            status="success",
            summary=report,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analyze/crime-types")
async def analyze_crime_types():
    """
    Analyze crime types (requires data to be loaded)
    """
    try:
        # This would need to be implemented with loaded data
        return {"status": "success", "message": "Crime type analysis endpoint"}
    except Exception as e:
        logger.error(f"Error analyzing crime types: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Model Information Endpoints
@app.get("/models/info")
async def get_model_info():
    """
    Get information about loaded models
    """
    try:
        predictor_info = predictor.get_model_info()
        classifier_info = classifier.get_model_performance()
        
        return {
            "status": "success",
            "models": {
                "crime_rate_predictor": predictor_info,
                "crime_type_classifier": classifier_info
            }
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 