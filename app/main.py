"""
Forensic Backend API
Kelly-Ann Harris - Capstone Project
Main FastAPI app for crime and forensic analysis
"""

import sys
import os
from pathlib import Path

# Add services to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "services" / "structured"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from datetime import datetime
import time
import threading

# Import analysis modules
from app.services.structured.crime_rate_predictor import CrimeRatePredictor
from app.services.structured.criminal_network_analyzer import CriminalNetworkAnalyzer
from app.services.structured.spatial_crime_mapper import SpatialCrimeMapper
from app.services.structured.temporal_pattern_analyzer import TemporalPatternAnalyzer
from app.services.structured.crime_type_classifier import CrimeTypeClassifier
from app.services.structured.data_analyzer import LAPDCrimeDataAnalyzer

# Import forensic endpoints
from app.api.endpoints import unstructured

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forensic_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="Forensic Analysis API",
    description="Crime analysis and forensic system for capstone project",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
predictor = None
network_analyzer = None
spatial_mapper = None
temporal_analyzer = None
classifier = None
data_analyzer = None

@app.on_event("startup")
async def startup_event():
    """Load all the models when app starts"""
    global predictor, network_analyzer, spatial_mapper, temporal_analyzer, classifier, data_analyzer
    
    def load_models():
        try:
            logger.info("Loading models...")
            
            models_dir = current_dir / "services" / "structured"
            
            global predictor, network_analyzer, spatial_mapper, temporal_analyzer, classifier, data_analyzer
            predictor = CrimeRatePredictor(models_dir=str(models_dir))
            network_analyzer = CriminalNetworkAnalyzer()
            spatial_mapper = SpatialCrimeMapper()
            temporal_analyzer = TemporalPatternAnalyzer()
            classifier = CrimeTypeClassifier(model_path=str(models_dir / "crime_type_rf_model.pkl"))
            data_analyzer = LAPDCrimeDataAnalyzer()
            
            logger.info("All models loaded!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    # Load in background
    model_thread = threading.Thread(target=load_models)
    model_thread.daemon = True
    model_thread.start()

# Request models
class CrimeDataRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Crime records")
    
class SpatialPredictionRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Crime data for prediction")
    
class TemporalPredictionRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Historical crime data")
    days_ahead: int = Field(default=5, ge=1, le=365, description="Days to forecast")
    
class NetworkAnalysisRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Crime data for network analysis")
    top_n: int = Field(default=10, ge=1, le=100, description="Top nodes to return")
    
class SpatialAnalysisRequest(BaseModel):
    crime_data: List[Dict[str, Any]] = Field(..., description="Crime data with coordinates")
    n_clusters: int = Field(default=5, ge=2, le=20, description="Number of clusters")
    radius_km: float = Field(default=1.0, ge=0.1, le=50.0, description="Radius for hotspots")
    
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

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Check if models are ready
def check_models_loaded():
    """Make sure models are loaded"""
    if not predictor or not network_analyzer or not spatial_mapper or not temporal_analyzer or not classifier or not data_analyzer:
        raise HTTPException(
            status_code=503, 
            detail="Models are still loading. Please try again in a few seconds."
        )

@app.get("/")
async def welcome():
    """Welcome endpoint"""
    return {
        "message": "Forensic Analysis API",
        "author": "Kelly-Ann Harris",
        "project": "Crime & Forensic Analysis Capstone",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models": "/models/info"
        }
    }

@app.get("/health")
async def health_check():
    """Check if everything is working"""
    model_status = {
        "predictor": predictor is not None,
        "network_analyzer": network_analyzer is not None,
        "spatial_mapper": spatial_mapper is not None,
        "temporal_analyzer": temporal_analyzer is not None,
        "classifier": classifier is not None,
        "data_analyzer": data_analyzer is not None
    }
    
    all_loaded = all(model_status.values())
    
    return {
        "status": "healthy" if all_loaded else "loading",
        "models_loaded": model_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/info")
async def get_models_info():
    """Get info about loaded models"""
    check_models_loaded()
    
    return {
        "models": {
            "crime_rate_predictor": "Spatial and temporal crime rate prediction",
            "network_analyzer": "Criminal network analysis",
            "spatial_mapper": "Crime hotspot mapping",
            "temporal_analyzer": "Time series analysis",
            "classifier": "Crime type classification",
            "data_analyzer": "General crime data analysis"
        },
        "status": "all_loaded",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/spatial", response_model=PredictionResponse)
async def predict_spatial_crime_rate(request: SpatialPredictionRequest):
    """Predict crime rates by location"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        # Convert to DataFrame
        df = pd.DataFrame(request.crime_data)
        
        # Get predictions
        results = predictor.predict_spatial_crime_rate(df)
        
        return PredictionResponse(
            status="success",
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Spatial prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/temporal", response_model=PredictionResponse)
async def predict_temporal_crime_rate(request: TemporalPredictionRequest):
    """Predict future crime rates"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        df = pd.DataFrame(request.crime_data)
        results = predictor.predict_temporal_crime_rate(df, days_ahead=request.days_ahead)
        
        return PredictionResponse(
            status="success",
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Temporal prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/crime-rate", response_model=PredictionResponse)
async def predict_crime_rate(request: CrimeDataRequest):
    """Predict crime rates (general endpoint)"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        df = pd.DataFrame(request.crime_data)
        
        # Use spatial prediction as default
        results = predictor.predict_spatial_crime_rate(df)
        
        return PredictionResponse(
            status="success",
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Crime rate prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/network/build", response_model=AnalysisResponse)
async def build_criminal_network(request: NetworkAnalysisRequest):
    """Build criminal network from crime data"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        df = pd.DataFrame(request.crime_data)
        results = network_analyzer.build_network(df, top_n=request.top_n)
        
        return AnalysisResponse(
            status="success",
            summary=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Network analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/network/summary", response_model=AnalysisResponse)
async def get_network_summary():
    """Get network analysis summary"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        results = network_analyzer.get_network_summary()
        
        return AnalysisResponse(
            status="success",
            summary=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Network summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/spatial/cluster", response_model=AnalysisResponse)
async def perform_spatial_clustering(request: SpatialAnalysisRequest):
    """Cluster crime locations"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        df = pd.DataFrame(request.crime_data)
        results = spatial_mapper.perform_clustering(df, n_clusters=request.n_clusters)
        
        return AnalysisResponse(
            status="success",
            summary=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Spatial clustering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/spatial/hotspots", response_model=AnalysisResponse)
async def analyze_crime_hotspots(request: SpatialAnalysisRequest):
    """Find crime hotspots"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        df = pd.DataFrame(request.crime_data)
        results = spatial_mapper.analyze_hotspots(df, radius_km=request.radius_km)
        
        return AnalysisResponse(
            status="success",
            summary=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Hotspot analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/temporal/forecast", response_model=PredictionResponse)
async def forecast_crime_trends(request: TemporalPredictionRequest):
    """Forecast crime trends"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        df = pd.DataFrame(request.crime_data)
        results = temporal_analyzer.forecast_trends(df, periods=request.days_ahead)
        
        return PredictionResponse(
            status="success",
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Temporal forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/crime-types", response_model=PredictionResponse)
async def classify_crime_types(request: ClassificationRequest):
    """Classify crime types"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        df = pd.DataFrame(request.crime_data)
        results = classifier.classify_crimes(df)
        
        return PredictionResponse(
            status="success",
            results=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/report", response_model=AnalysisResponse)
async def generate_analysis_report(request: CrimeDataRequest):
    """Generate comprehensive crime analysis report"""
    start_time = time.time()
    
    try:
        check_models_loaded()
        
        df = pd.DataFrame(request.crime_data)
        results = data_analyzer.generate_comprehensive_report(df)
        
        return AnalysisResponse(
            status="success",
            summary=results,
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Analysis report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include forensic endpoints
app.include_router(unstructured.router, prefix="/api/unstructured", tags=["Forensic Analysis"])

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 