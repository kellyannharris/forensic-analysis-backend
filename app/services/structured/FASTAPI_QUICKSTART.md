# FastAPI Quick Start Guide

## 🚀 **FastAPI is the BEST choice for your Crime Analysis API!**

### Why FastAPI?
- ✅ **Automatic API Documentation** - Your API will be self-documenting
- ✅ **Type Safety** - Built-in validation prevents errors
- ✅ **Performance** - One of the fastest Python web frameworks
- ✅ **Modern** - Built for modern Python with async support
- ✅ **Perfect for ML/AI** - Designed with data science in mind

## Quick Setup

### 1. Install Dependencies
```bash
cd structured_data_models
pip install -r fastapi_requirements.txt
```

### 2. Run the API
```bash
# Development mode (with auto-reload)
python fastapi_example.py

# Or using uvicorn directly
uvicorn fastapi_example:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access Your API
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🎯 **Key Features You Get**

### Automatic API Documentation
Your API will have beautiful, interactive documentation at `/docs` that includes:
- All endpoints with descriptions
- Request/response schemas
- Try-it-out functionality
- Example requests and responses

### Type Safety & Validation
```python
# FastAPI automatically validates this data
class CrimeDataRequest(BaseModel):
    crime_data: List[Dict[str, Any]]
    days_ahead: int = Field(default=5, ge=1, le=365)
```

### Performance Monitoring
```python
# Built-in timing for all endpoints
@app.post("/predict/spatial")
async def predict_spatial_crime_rate(request: SpatialPredictionRequest):
    start_time = datetime.now()
    # ... your analysis code ...
    processing_time = (datetime.now() - start_time).total_seconds()
    return {"processing_time": processing_time, "results": results}
```

## 📊 **Available Endpoints**

### Crime Rate Prediction
- `POST /predict/spatial` - Spatial crime rate prediction
- `POST /predict/temporal` - Temporal crime rate forecasting

### Network Analysis
- `POST /network/build` - Build criminal network
- `GET /network/centrality` - Analyze network centrality
- `GET /network/communities` - Detect communities
- `GET /network/summary` - Get comprehensive network summary

### Spatial Analysis
- `POST /spatial/prepare` - Prepare spatial data
- `POST /spatial/cluster` - Perform spatial clustering
- `POST /spatial/hotspots` - Analyze crime hotspots
- `POST /spatial/summary` - Get spatial analysis summary

### Temporal Analysis
- `POST /temporal/prepare` - Prepare time series data
- `GET /temporal/seasonality` - Detect seasonality
- `GET /temporal/arima` - Fit ARIMA model
- `POST /temporal/forecast` - Generate forecasts
- `POST /temporal/summary` - Get temporal analysis summary

### Crime Classification
- `POST /classify/crime-types` - Classify crime types
- `POST /classify/distribution` - Analyze crime type distribution
- `POST /classify/trends` - Analyze crime type trends
- `POST /classify/summary` - Get classification summary

### Data Analysis
- `POST /analyze/report` - Generate comprehensive analysis report
- `GET /models/info` - Get model information

## 🔧 **Example Usage**

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Spatial prediction
curl -X POST "http://localhost:8000/predict/spatial" \
     -H "Content-Type: application/json" \
     -d '{
       "crime_data": [
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
     }'
```

### Using Python requests
```python
import requests
import json

# API base URL
base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# Spatial prediction
crime_data = [{
    "DATE OCC": "2023-01-15",
    "TIME OCC": "1430",
    "AREA": 1,
    "LAT": 34.0522,
    "LON": -118.2437,
    "Vict Age": 25,
    "Premis Cd": 101,
    "Weapon Used Cd": 200
}]

response = requests.post(
    f"{base_url}/predict/spatial",
    json={"crime_data": crime_data}
)
print(response.json())
```

### Using JavaScript/fetch
```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Spatial prediction
const crimeData = [{
  "DATE OCC": "2023-01-15",
  "TIME OCC": "1430",
  "AREA": 1,
  "LAT": 34.0522,
  "LON": -118.2437,
  "Vict Age": 25,
  "Premis Cd": 101,
  "Weapon Used Cd": 200
}];

fetch('http://localhost:8000/predict/spatial', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ crime_data: crimeData })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🚀 **Production Deployment**

### Using Gunicorn
```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn fastapi_example:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r fastapi_requirements.txt

EXPOSE 8000

CMD ["uvicorn", "fastapi_example:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
# Set environment variables for production
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export WORKERS=4
export HOST=0.0.0.0
export PORT=8000
```

## 📈 **Performance Tips**

### 1. Async Operations
FastAPI supports async operations for better performance:
```python
@app.post("/predict/spatial")
async def predict_spatial_crime_rate(request: SpatialPredictionRequest):
    # This runs asynchronously
    results = await process_crime_data(request.crime_data)
    return results
```

### 2. Background Tasks
For long-running operations:
```python
@app.post("/analyze/comprehensive")
async def comprehensive_analysis(
    request: CrimeDataRequest, 
    background_tasks: BackgroundTasks
):
    # Start analysis in background
    background_tasks.add_task(run_comprehensive_analysis, request.crime_data)
    return {"status": "analysis_started", "task_id": "task_123"}
```

### 3. Caching
Implement caching for repeated requests:
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8")
    FastAPICache.init(RedisBackend(redis), prefix="crime-analysis")
```

## 🔍 **Monitoring & Logging**

### Built-in Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict/spatial")
async def predict_spatial_crime_rate(request: SpatialPredictionRequest):
    logger.info(f"Processing spatial prediction for {len(request.crime_data)} records")
    # ... your code ...
    logger.info("Spatial prediction completed successfully")
```

### Health Monitoring
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models_loaded": check_model_status()
    }
```

## 🎉 **What You Get**

With this FastAPI implementation, you get:

✅ **Professional API** with automatic documentation  
✅ **Type-safe** request/response handling  
✅ **High performance** async operations  
✅ **Production-ready** error handling  
✅ **Easy testing** with built-in test client  
✅ **Scalable** architecture for growth  
✅ **Modern** Python best practices  

## 🚀 **Next Steps**

1. **Start the API**: `python fastapi_example.py`
2. **Visit the docs**: http://localhost:8000/docs
3. **Test endpoints**: Use the interactive documentation
4. **Deploy**: Use the production deployment options above
5. **Monitor**: Set up logging and health checks

Your crime analysis API is now ready for production use! 🎯 