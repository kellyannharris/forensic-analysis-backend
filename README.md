# Forensic Analysis Backend API

## 🔬 Crime & Forensic Analysis System

**Author:** Kelly-Ann Harris  
**Project:** Capstone Project - Crime & Forensic Analysis System  
**Version:** 1.0.0  
**Date:** January 2025  

A comprehensive FastAPI-based backend system for crime analysis and forensic evidence processing, combining structured crime data analytics with advanced unstructured forensic analysis capabilities.

---

## 🚀 Features

### 📊 Structured Crime Data Analysis
- **Crime Rate Prediction**: Spatial and temporal crime rate forecasting using ensemble ML models
- **Criminal Network Analysis**: Graph-based analysis for identifying criminal networks and key players
- **Spatial Crime Mapping**: Hotspot detection and clustering analysis of crime locations
- **Temporal Pattern Analysis**: Time series analysis and crime trend forecasting
- **Crime Type Classification**: Automated classification of crime types using machine learning
- **Comprehensive Reporting**: Statistical analysis and crime data insights

### 🧬 Unstructured Forensic Analysis
- **Blood Splatter Analysis**: CNN-based pattern classification and impact angle estimation
- **Handwriting Analysis**: Writer identification and verification using advanced ML models
- **Cartridge Case Analysis**: 3D surface analysis and firearm identification from X3P files
- **Cross-Modal Analysis**: Integrated analysis combining multiple evidence types

### 🔧 System Features
- **FastAPI Framework**: High-performance asynchronous API with automatic documentation
- **Model Pre-loading**: Background model initialization for optimal response times
- **Comprehensive Validation**: Input validation and error handling
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Monitoring**: System health and model status endpoints
- **Batch Processing**: Support for analyzing multiple files simultaneously

---

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended for large datasets)
- **Storage**: 5GB free space for models and temporary files
- **OS**: Windows, macOS, or Linux

### Dependencies
The project uses modern Python libraries for machine learning, data analysis, and web services. See `requirements.txt` for complete dependency list.

**Key Dependencies:**
- FastAPI >= 0.104.0
- scikit-learn >= 1.1.0
- pandas >= 1.5.0
- networkx >= 2.8.0
- XGBoost >= 1.6.0
- Prophet >= 1.1.0

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd forensic_backend
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv forensic_env

# Activate virtual environment
# On Windows:
forensic_env\Scripts\activate
# On macOS/Linux:
source forensic_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Run health check
python -c "import fastapi, pandas, sklearn, networkx; print('All dependencies installed successfully!')"
```

### 5. Start the Server
```bash
# Option 1: Using the run script
python run_server.py

# Option 2: Direct uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Using the main module
python app/main.py
```

### 6. Verify Setup
Once the server is running, verify the installation:

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Model Status**: http://localhost:8000/models/info

---

## 📖 API Documentation

### 🌐 Base Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message and API information |
| `/health` | GET | System health and model status |
| `/models/info` | GET | Information about loaded models |
| `/docs` | GET | Interactive API documentation |

### 📊 Structured Data Analysis Endpoints

#### Crime Rate Prediction
```http
POST /predict/spatial
POST /predict/temporal
```

#### Network Analysis
```http
POST /network/build
GET /network/summary
```

#### Spatial Analysis
```http
POST /spatial/cluster
POST /spatial/hotspots
```

#### Temporal Analysis
```http
POST /temporal/forecast
```

#### Classification & Reporting
```http
POST /classify/crime-types
POST /analyze/report
```

### 🧬 Unstructured Forensic Analysis Endpoints

#### Blood Splatter Analysis
```http
POST /api/unstructured/bloodsplatter/analyze
POST /api/unstructured/bloodsplatter/analyze/batch
POST /api/unstructured/bloodsplatter/compare
```

#### Handwriting Analysis
```http
POST /api/unstructured/handwriting/identify
POST /api/unstructured/handwriting/verify
POST /api/unstructured/handwriting/analyze/batch
```

#### Cartridge Case Analysis
```http
POST /api/unstructured/cartridge-case/analyze
POST /api/unstructured/cartridge-case/compare
POST /api/unstructured/cartridge-case/identify
```

#### Cross-Modal Analysis
```http
POST /api/unstructured/cross-modal/analyze
```

---

## 💡 Usage Examples

### Structured Crime Data Analysis

#### Crime Rate Prediction
```python
import requests
import json

# Prepare crime data
crime_data = {
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
}

# Make spatial prediction request
response = requests.post(
    "http://localhost:8000/predict/spatial",
    json=crime_data
)

print(json.dumps(response.json(), indent=2))
```

#### Criminal Network Analysis
```python
network_data = {
    "crime_data": [
        {
            "LOCATION": "123 MAIN ST",
            "Mocodes": "1234,5678"
        },
        {
            "LOCATION": "456 OAK AVE", 
            "Mocodes": "1234"
        }
    ],
    "top_n": 10
}

response = requests.post(
    "http://localhost:8000/network/build",
    json=network_data
)
```

### Unstructured Forensic Analysis

#### Blood Splatter Analysis
```python
# Analyze blood splatter image
files = {"image": open("bloodsplatter.jpg", "rb")}
response = requests.post(
    "http://localhost:8000/api/unstructured/bloodsplatter/analyze",
    files=files
)
```

#### Handwriting Analysis
```python
# Identify writer from handwriting sample
files = {"image": open("handwriting_sample.png", "rb")}
params = {"top_k": 5}
response = requests.post(
    "http://localhost:8000/api/unstructured/handwriting/identify",
    files=files,
    params=params
)
```

#### Cross-Modal Analysis
```python
# Analyze multiple evidence types
files = {
    "bloodsplatter_image": open("bloodsplatter.jpg", "rb"),
    "handwriting_image": open("handwriting.png", "rb"),
    "cartridge_case": open("cartridge.x3p", "rb")
}
response = requests.post(
    "http://localhost:8000/api/unstructured/cross-modal/analyze",
    files=files
)
```

---

## 📊 Data Formats

### Structured Crime Data Schema
```json
{
    "DATE OCC": "2023-01-15",
    "TIME OCC": "1430",
    "AREA": 1,
    "AREA NAME": "Central",
    "Crm Cd": 510,
    "Crm Cd Desc": "VEHICLE - STOLEN",
    "Vict Age": 25,
    "Vict Sex": "F",
    "Vict Descent": "W",
    "Premis Cd": 101,
    "Premis Desc": "STREET",
    "Weapon Used Cd": 200,
    "Weapon Desc": "UNKNOWN WEAPON",
    "Status": "IC",
    "Status Desc": "Invest Cont",
    "LOCATION": "123 MAIN ST",
    "LAT": 34.0522,
    "LON": -118.2437,
    "Mocodes": "1234,5678"
}
```

### Unstructured Data Formats
- **Blood Splatter**: JPG, PNG, TIFF (max 45MB)
- **Handwriting**: PNG images
- **Cartridge Cases**: X3P format files

---

## 🧪 Testing

### Run Comprehensive Tests
```bash
# Run all API tests
python test_api.py

# Run specific endpoint tests
python test_actual_endpoints.py

# Run HTTP tests
python run_http_tests.py
```

### Manual Testing
Use the interactive API documentation at http://localhost:8000/docs to test endpoints manually.

---

## 🏗️ Project Structure

```
forensic_backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # Main FastAPI application
│   ├── api/
│   │   └── endpoints/
│   │       └── unstructured.py      # Forensic analysis endpoints
│   ├── core/
│   │   └── config.py               # Application configuration
│   ├── models/                     # Data models
│   │   ├── forensic/
│   │   └── ml/
│   ├── services/
│   │   ├── structured/             # Crime data analysis services
│   │   │   ├── crime_rate_predictor.py
│   │   │   ├── criminal_network_analyzer.py
│   │   │   ├── spatial_crime_mapper.py
│   │   │   ├── temporal_pattern_analyzer.py
│   │   │   ├── crime_type_classifier.py
│   │   │   └── data_analyzer.py
│   │   └── unstructured/           # Forensic analysis services
│   │       ├── bloodsplatter_service.py
│   │       ├── handwriting_service.py
│   │       └── cartridge_case_service.py
│   └── utils/                      # Utility modules
├── config/                         # Configuration files
├── tests/                          # Test files
├── requirements.txt                # Python dependencies
├── run_server.py                   # Server startup script
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## ⚙️ Configuration

### Environment Variables
Create a `.env` file or set environment variables:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Model Configuration
MODELS_DIR=./app/services/structured
LOG_LEVEL=INFO

# CORS Configuration
ALLOWED_ORIGINS=*
```

### Configuration Template
Copy `config.template` to `config.py` and modify as needed.

---

## 🔧 Model Information

### Pre-trained Models Included

#### Structured Data Models
- **Gradient Boosting Model**: Spatial crime rate prediction (R² > 0.8)
- **Random Forest Model**: Spatial crime rate prediction (R² > 0.8)
- **XGBoost Model**: Enhanced spatial prediction
- **Prophet Model**: Temporal forecasting (MAPE < 15%)
- **Crime Type Classifier**: Random Forest classifier (Accuracy > 85%)

#### Unstructured Data Models
- **Blood Splatter CNN**: Deep learning model for pattern classification
- **Handwriting Analysis Model**: Writer identification and verification
- **Cartridge Case Analysis Model**: 3D surface analysis and firearm identification

### Model Performance
- All models are trained from scratch without using pretrained models
- Regular retraining recommended for optimal performance
- Performance monitoring endpoints available

---

## 🚨 Error Handling

### Common Error Responses
```json
{
    "error": "DataFormatError",
    "detail": "Missing required column: 'DATE OCC'",
    "timestamp": "2025-01-15T10:30:00Z"
}
```

### Status Codes
- `200`: Success
- `400`: Bad Request (invalid data format)
- `422`: Validation Error
- `500`: Internal Server Error
- `503`: Service Unavailable (models loading)

---

## 📈 Performance Considerations

### Processing Time Estimates

| Operation | Small (1K records) | Medium (10K records) | Large (100K records) |
|-----------|-------------------|---------------------|---------------------|
| Spatial Prediction | <1s | 2-5s | 10-30s |
| Network Analysis | 1-2s | 5-10s | 30-60s |
| Spatial Clustering | <1s | 2-5s | 10-20s |
| Temporal Analysis | 1-2s | 5-15s | 30-90s |
| Crime Classification | <1s | 2-5s | 10-25s |

### Optimization Tips
1. **Batch Processing**: Use batch endpoints for multiple files
2. **Data Sampling**: Use representative samples for large datasets
3. **Model Caching**: Models are pre-loaded for optimal response times
4. **Parallel Processing**: System supports concurrent requests

---

## 🛡️ Security Considerations

- Input validation on all endpoints
- File type verification for uploads
- Size limits on uploaded files
- CORS configuration for web applications
- No sensitive data logged

---

## 🔄 Development

### Adding New Models
1. Create model class in appropriate service directory
2. Add model loading to startup event in `main.py`
3. Create API endpoints in relevant router
4. Add tests for new functionality

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints throughout
- Comprehensive error handling
- Proper logging practices

---

## 📝 Logging

Logs are written to:
- Console output (development)
- `forensic_backend.log` file

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

---

## 🤝 Contributing

This is a capstone project by Kelly-Ann Harris. For questions or contributions:

1. Review the code structure and documentation
2. Follow existing patterns for new features
3. Include tests for new functionality
4. Update documentation as needed

---

## 📄 License

This project is part of the Crime & Forensic Analysis System capstone project.

---

## 🆘 Troubleshooting

### Common Issues

#### Models Not Loading
```bash
# Check model files exist
ls app/services/structured/*.pkl
ls app/services/structured/*.joblib

# Check logs
tail -f forensic_backend.log
```

#### Memory Issues
- Reduce batch sizes for large datasets
- Consider increasing system memory
- Use data sampling for exploration

#### Connection Issues
- Verify server is running on correct port
- Check firewall settings
- Confirm CORS configuration

### Getting Help
1. Check the logs: `forensic_backend.log`
2. Visit the health endpoint: `/health`
3. Review API documentation: `/docs`
4. Verify model status: `/models/info`

---

## 🎯 Next Steps

1. **Deployment**: Consider containerization with Docker
2. **Scaling**: Implement load balancing for production
3. **Monitoring**: Add application performance monitoring
4. **Security**: Implement authentication and authorization
5. **Documentation**: Generate API client libraries

---

**🚀 Ready to analyze crime data and forensic evidence!**

For complete API documentation, visit http://localhost:8000/docs after starting the server. 