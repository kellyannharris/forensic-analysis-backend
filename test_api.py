#!/usr/bin/env python3
"""
Test script for Forensic Backend API
Author: Kelly-Ann Harris
Date: 2024
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None, description=""):
    """Test an API endpoint and print results"""
    print(f"\n{'='*50}")
    print(f"Testing: {description}")
    print(f"Endpoint: {method} {endpoint}")
    print(f"{'='*50}")
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")

def main():
    """Run all API tests"""
    print("Forensic Backend API Test Suite")
    print("=" * 60)
    
    # Sample crime data for testing
    sample_crime_data = [
        {
            "DATE OCC": "2024-01-01",
            "TIME OCC": "1200",
            "AREA": 1,
            "LAT": 34.0522,
            "LON": -118.2437,
            "Crm Cd": 510,
            "Vict Age": 25,
            "Premis Cd": 101,
            "Weapon Used Cd": 200,
            "AREA NAME": "Central",
            "Vict Sex": "M",
            "Vict Descent": "W",
            "Mocodes": "0416 0344"
        }
    ]
    
    # Test system endpoints
    test_endpoint("/health", "GET", description="Health Check")
    test_endpoint("/models/info", "GET", description="Model Information")
    
    # Test prediction endpoints
    test_endpoint("/predict/spatial", "POST", 
                 {"crime_data": sample_crime_data}, 
                 "Spatial Crime Prediction")
    
    test_endpoint("/predict/temporal", "POST", 
                 {"crime_data": sample_crime_data, "days_ahead": 7}, 
                 "Temporal Crime Prediction")
    
    # Test analysis endpoints
    test_endpoint("/network/analyze", "POST", 
                 {"crime_data": sample_crime_data * 5, "top_n": 3}, 
                 "Criminal Network Analysis")
    
    test_endpoint("/spatial/analyze", "POST", 
                 {"crime_data": sample_crime_data * 3, "n_clusters": 2, "radius_km": 1.0}, 
                 "Spatial Pattern Analysis")
    
    test_endpoint("/temporal/analyze", "POST", 
                 {"crime_data": sample_crime_data * 5}, 
                 "Temporal Pattern Analysis")
    
    test_endpoint("/classify/crime-types", "POST", 
                 {"crime_data": sample_crime_data}, 
                 "Crime Type Classification")
    
    test_endpoint("/analyze/comprehensive", "POST", 
                 {"crime_data": sample_crime_data * 10}, 
                 "Comprehensive Analysis")
    
    print(f"\n{'='*60}")
    print("API Test Suite Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 