#!/usr/bin/env python3
"""
Automated HTTP Test Runner for Forensic Backend API
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

# Sample crime data for testing
SAMPLE_CRIME_DATA = [
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
        "Mocodes": "0416 0344",
        "LOCATION": "123 Main St",
        "Crm Cd Desc": "Vehicle - Stolen",
        "Premis Desc": "Street",
        "Weapon Desc": "Unknown",
        "Status": "IC"
    },
    {
        "DATE OCC": "2024-01-02",
        "TIME OCC": "1800",
        "AREA": 2,
        "LAT": 34.0622,
        "LON": -118.2537,
        "Crm Cd": 440,
        "Vict Age": 30,
        "Premis Cd": 102,
        "Weapon Used Cd": 205,
        "AREA NAME": "Rampart",
        "Vict Sex": "F",
        "Vict Descent": "H",
        "Mocodes": "0420 0348",
        "LOCATION": "456 Oak Ave",
        "Crm Cd Desc": "Theft - From Vehicle",
        "Premis Desc": "Parking Lot",
        "Weapon Desc": "None",
        "Status": "AA"
    },
    {
        "DATE OCC": "2024-01-03",
        "TIME OCC": "0900",
        "AREA": 3,
        "LAT": 34.0722,
        "LON": -118.2637,
        "Crm Cd": 330,
        "Vict Age": 35,
        "Premis Cd": 103,
        "Weapon Used Cd": 210,
        "AREA NAME": "Southwest",
        "Vict Sex": "M",
        "Vict Descent": "B",
        "Mocodes": "0425 0352",
        "LOCATION": "789 Pine St",
        "Crm Cd Desc": "Burglary - Residential",
        "Premis Desc": "Single Family Dwelling",
        "Weapon Desc": "None",
        "Status": "IC"
    }
]

def run_test(name, method, endpoint, data=None, expected_status=200):
    """Run a single HTTP test"""
    print(f"\n🧪 {name}")
    print(f"   {method} {endpoint}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=30)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == expected_status:
            print(f"✅ SUCCESS - {response.status_code} ({response_time:.2f}s)")
            
            # Show key response info
            try:
                result = response.json()
                if "results" in result:
                    if isinstance(result["results"], dict):
                        print(f"   Results: {list(result['results'].keys())}")
                    elif isinstance(result["results"], list):
                        print(f"   Results: {len(result['results'])} items")
                
                if "summary" in result:
                    print(f"   Summary: {list(result['summary'].keys())}")
                
                if "status" in result:
                    print(f"   Status: {result['status']}")
                    
            except:
                print(f"   Response length: {len(response.text)} chars")
                
            return True
        else:
            print(f"❌ FAILED - {response.status_code} (expected {expected_status})")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        return False

def main():
    """Run all HTTP tests"""
    print("🚀 FORENSIC BACKEND API - HTTP TEST SUITE")
    print("=" * 60)
    print(f"Testing API at: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test cases
    test_cases = [
        # System tests
        ("Health Check", "GET", "/health"),
        ("Models Info", "GET", "/models/info"),
        
        # Prediction tests
        ("Spatial Prediction", "POST", "/predict/spatial", {"crime_data": SAMPLE_CRIME_DATA[:2]}),
        ("Temporal Prediction", "POST", "/predict/temporal", {"crime_data": SAMPLE_CRIME_DATA, "days_ahead": 7}),
        
        # Network analysis tests
        ("Network Build", "POST", "/network/build", {"crime_data": SAMPLE_CRIME_DATA, "top_n": 5}),
        ("Network Summary", "GET", "/network/summary"),
        ("Network Centrality", "GET", "/network/centrality?top_n=5"),
        
        # Spatial analysis tests
        ("Spatial Prepare", "POST", "/spatial/prepare", {"crime_data": SAMPLE_CRIME_DATA[:2], "n_clusters": 2, "radius_km": 1.0}),
        ("Spatial Hotspots", "POST", "/spatial/hotspots", {"crime_data": SAMPLE_CRIME_DATA[:2], "n_clusters": 2, "radius_km": 1.0}),
        ("Spatial Summary", "POST", "/spatial/summary", {"crime_data": SAMPLE_CRIME_DATA[:2], "n_clusters": 2, "radius_km": 1.0}),
        
        # Temporal analysis tests
        ("Temporal Prepare", "POST", "/temporal/prepare", {"crime_data": SAMPLE_CRIME_DATA}),
        ("Temporal Seasonality", "GET", "/temporal/seasonality"),
        ("Temporal Summary", "POST", "/temporal/summary", {"crime_data": SAMPLE_CRIME_DATA}),
        
        # Classification tests
        ("Crime Classification", "POST", "/classify/crime-types", {"crime_data": SAMPLE_CRIME_DATA[:2]}),
        ("Crime Distribution", "POST", "/classify/distribution", {"crime_data": SAMPLE_CRIME_DATA[:2]}),
        ("Crime Trends", "POST", "/classify/trends", {"crime_data": SAMPLE_CRIME_DATA[:2]}),
        
        # Analysis tests
        ("Analysis Report", "POST", "/analyze/report", {"crime_data": SAMPLE_CRIME_DATA}),
        ("Analyze Crime Types", "GET", "/analyze/crime-types"),
    ]
    
    # Run tests
    results = []
    total_start_time = time.time()
    
    for test_case in test_cases:
        if len(test_case) == 3:
            name, method, endpoint = test_case
            data = None
        else:
            name, method, endpoint, data = test_case
            
        success = run_test(name, method, endpoint, data)
        results.append((name, success))
        
        # Small delay between tests
        time.sleep(0.5)
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Total Time: {total_time:.2f} seconds")
    
    print(f"\n📋 DETAILED RESULTS:")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name:<30} {status}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ Forensic Backend API is fully operational!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("Check the logs above for details")
    
    print(f"\n🔗 API Documentation: {BASE_URL}/docs")
    print(f"📈 Health Check: {BASE_URL}/health")

if __name__ == "__main__":
    main() 