#!/usr/bin/env python3
"""
Comprehensive API endpoint testing with actual data
"""
import requests
import json
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

# Create realistic test data
def create_test_data():
    """Create realistic crime data for testing"""
    base_date = datetime(2024, 1, 1)
    test_data = []
    
    # Create multiple crime records with different patterns
    crime_codes = [510, 440, 330, 624, 310, 230, 210, 121, 110, 420]
    areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for i in range(20):
        date_offset = timedelta(days=i)
        current_date = base_date + date_offset
        
        # Create varied crime data
        crime_record = {
            "DATE OCC": current_date.strftime("%Y-%m-%d"),
            "TIME OCC": f"{(i % 24):02d}00",
            "AREA": areas[i % len(areas)],
            "LAT": 34.0522 + (i % 10) * 0.01,
            "LON": -118.2437 + (i % 10) * 0.01,
            "Crm Cd": crime_codes[i % len(crime_codes)],
            "Vict Age": 20 + (i % 50),
            "Premis Cd": 100 + (i % 50),
            "Weapon Used Cd": 200 + (i % 10),
            "AREA NAME": f"Area {areas[i % len(areas)]}",
            "Vict Sex": "M" if i % 2 == 0 else "F",
            "Vict Descent": ["W", "H", "B", "A", "O"][i % 5],
            "Mocodes": f"04{i:02d} 03{i:02d}",
            "LOCATION": f"Street {i}",
            "Crm Cd Desc": f"Crime Type {i % 5}",
            "Premis Desc": f"Premise {i % 5}",
            "Weapon Desc": f"Weapon {i % 5}",
            "Status": "IC"
        }
        test_data.append(crime_record)
    
    return test_data

def test_endpoint(endpoint, method="GET", data=None, description=""):
    """Test an endpoint and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Endpoint: {method} {endpoint}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=30)
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=30)
        
        end_time = time.time()
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Response Keys: {list(result.keys())}")
            
            # Print key information based on endpoint
            if "results" in result:
                if isinstance(result["results"], dict):
                    print(f"Results: {list(result['results'].keys())}")
                    for key, value in result["results"].items():
                        if isinstance(value, list) and len(value) > 0:
                            print(f"  {key}: {value[0]:.4f} (first prediction)")
                        else:
                            print(f"  {key}: {value}")
                elif isinstance(result["results"], list):
                    print(f"Results count: {len(result['results'])}")
            
            if "summary" in result:
                print(f"Summary keys: {list(result['summary'].keys())}")
            
            if "analysis" in result:
                print(f"Analysis keys: {list(result['analysis'].keys())}")
            
            if "clusters" in result:
                print(f"Clusters: {result['clusters']}")
            
            if "hotspots" in result:
                print(f"Hotspots: {result['hotspots']}")
            
            if "seasonality" in result:
                print(f"Seasonality: {result['seasonality']}")
                
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Error detail: {response.text}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
    
    return response.status_code == 200

def main():
    """Run comprehensive tests"""
    print("🧪 COMPREHENSIVE API ENDPOINT TESTING")
    print("="*60)
    
    # Create test data
    test_data = create_test_data()
    print(f"Created {len(test_data)} test crime records")
    
    results = {}
    
    # Test 1: System Health Check
    results["health"] = test_endpoint("/health", "GET", description="System Health Check")
    
    # Test 2: Models Info
    results["models_info"] = test_endpoint("/models/info", "GET", description="Model Information")
    
    # Test 3: Spatial Crime Rate Prediction
    results["spatial_prediction"] = test_endpoint(
        "/predict/spatial", 
        "POST", 
        {"crime_data": test_data[:5]}, 
        "Spatial Crime Rate Prediction"
    )
    
    # Test 4: Temporal Crime Rate Prediction
    results["temporal_prediction"] = test_endpoint(
        "/predict/temporal", 
        "POST", 
        {"crime_data": test_data[:10], "days_ahead": 14}, 
        "Temporal Crime Rate Prediction"
    )
    
    # Test 5: Criminal Network Analysis
    results["network_analysis"] = test_endpoint(
        "/network/analyze", 
        "POST", 
        {"crime_data": test_data, "top_n": 5}, 
        "Criminal Network Analysis"
    )
    
    # Test 6: Spatial Pattern Analysis
    results["spatial_analysis"] = test_endpoint(
        "/spatial/analyze", 
        "POST", 
        {"crime_data": test_data[:10], "n_clusters": 3, "radius_km": 2.0}, 
        "Spatial Pattern Analysis"
    )
    
    # Test 7: Temporal Pattern Analysis
    results["temporal_analysis"] = test_endpoint(
        "/temporal/analyze", 
        "POST", 
        {"crime_data": test_data}, 
        "Temporal Pattern Analysis"
    )
    
    # Test 8: Crime Type Classification
    results["crime_classification"] = test_endpoint(
        "/classify/crime-types", 
        "POST", 
        {"crime_data": test_data[:5]}, 
        "Crime Type Classification"
    )
    
    # Test 9: Comprehensive Analysis
    results["comprehensive_analysis"] = test_endpoint(
        "/analyze/comprehensive", 
        "POST", 
        {"crime_data": test_data}, 
        "Comprehensive Data Analysis"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! API is fully functional!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Check the logs above.")

if __name__ == "__main__":
    main()
