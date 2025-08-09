#!/usr/bin/env python3
"""
Test actual API endpoints with real data
"""
import requests
import json
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def create_test_data():
    """Create realistic crime data for testing"""
    base_date = datetime(2024, 1, 1)
    test_data = []
    
    crime_codes = [510, 440, 330, 624, 310, 230, 210, 121, 110, 420]
    areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for i in range(20):
        date_offset = timedelta(days=i)
        current_date = base_date + date_offset
        
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
    """Test an endpoint"""
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
            
            # Print sample of results
            if "results" in result:
                if isinstance(result["results"], dict):
                    print(f"Results keys: {list(result['results'].keys())}")
                elif isinstance(result["results"], list):
                    print(f"Results count: {len(result['results'])}")
            
            if "summary" in result:
                print(f"Summary keys: {list(result['summary'].keys())}")
            
            if "analysis" in result:
                print(f"Analysis keys: {list(result['analysis'].keys())}")
                
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Error detail: {response.text}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
    
    return response.status_code == 200

def main():
    """Test all actual endpoints"""
    print("🧪 TESTING ACTUAL API ENDPOINTS")
    print("="*60)
    
    test_data = create_test_data()
    print(f"Created {len(test_data)} test crime records")
    
    results = {}
    
    # System endpoints
    results["health"] = test_endpoint("/health", "GET", description="System Health Check")
    results["models_info"] = test_endpoint("/models/info", "GET", description="Model Information")
    
    # Prediction endpoints
    results["spatial_prediction"] = test_endpoint(
        "/predict/spatial", 
        "POST", 
        {"crime_data": test_data[:5]}, 
        "Spatial Crime Rate Prediction"
    )
    
    results["temporal_prediction"] = test_endpoint(
        "/predict/temporal", 
        "POST", 
        {"crime_data": test_data[:10], "days_ahead": 14}, 
        "Temporal Crime Rate Prediction"
    )
    
    # Network analysis endpoints
    results["network_build"] = test_endpoint(
        "/network/build", 
        "POST", 
        {"crime_data": test_data, "top_n": 5}, 
        "Build Criminal Network"
    )
    
    results["network_centrality"] = test_endpoint(
        "/network/centrality", 
        "GET", 
        description="Network Centrality Analysis"
    )
    
    results["network_communities"] = test_endpoint(
        "/network/communities", 
        "GET", 
        description="Network Community Detection"
    )
    
    results["network_summary"] = test_endpoint(
        "/network/summary", 
        "GET", 
        description="Network Summary"
    )
    
    # Spatial analysis endpoints
    results["spatial_prepare"] = test_endpoint(
        "/spatial/prepare", 
        "POST", 
        {"crime_data": test_data[:10], "n_clusters": 3, "radius_km": 2.0}, 
        "Prepare Spatial Data"
    )
    
    results["spatial_cluster"] = test_endpoint(
        "/spatial/cluster", 
        "POST", 
        {"crime_data": test_data[:10], "n_clusters": 3, "radius_km": 2.0}, 
        "Spatial Clustering"
    )
    
    results["spatial_hotspots"] = test_endpoint(
        "/spatial/hotspots", 
        "POST", 
        {"crime_data": test_data[:10], "n_clusters": 3, "radius_km": 2.0}, 
        "Hotspot Analysis"
    )
    
    results["spatial_summary"] = test_endpoint(
        "/spatial/summary", 
        "POST", 
        {"crime_data": test_data[:10], "n_clusters": 3, "radius_km": 2.0}, 
        "Spatial Summary"
    )
    
    # Temporal analysis endpoints
    results["temporal_prepare"] = test_endpoint(
        "/temporal/prepare", 
        "POST", 
        {"crime_data": test_data}, 
        "Prepare Temporal Data"
    )
    
    results["temporal_seasonality"] = test_endpoint(
        "/temporal/seasonality", 
        "GET", 
        description="Seasonality Detection"
    )
    
    results["temporal_arima"] = test_endpoint(
        "/temporal/arima", 
        "GET", 
        description="ARIMA Model"
    )
    
    results["temporal_forecast"] = test_endpoint(
        "/temporal/forecast", 
        "POST", 
        {"crime_data": test_data, "days_ahead": 7}, 
        "Temporal Forecast"
    )
    
    results["temporal_summary"] = test_endpoint(
        "/temporal/summary", 
        "POST", 
        {"crime_data": test_data}, 
        "Temporal Summary"
    )
    
    # Classification endpoints
    results["crime_classification"] = test_endpoint(
        "/classify/crime-types", 
        "POST", 
        {"crime_data": test_data[:5]}, 
        "Crime Type Classification"
    )
    
    results["crime_distribution"] = test_endpoint(
        "/classify/distribution", 
        "POST", 
        {"crime_data": test_data[:5]}, 
        "Crime Distribution Analysis"
    )
    
    results["crime_trends"] = test_endpoint(
        "/classify/trends", 
        "POST", 
        {"crime_data": test_data[:5]}, 
        "Crime Trends Analysis"
    )
    
    results["crime_summary"] = test_endpoint(
        "/classify/summary", 
        "POST", 
        {"crime_data": test_data[:5]}, 
        "Crime Classification Summary"
    )
    
    # Data analysis endpoints
    results["analysis_report"] = test_endpoint(
        "/analyze/report", 
        "POST", 
        {"crime_data": test_data}, 
        "Generate Analysis Report"
    )
    
    results["analyze_crime_types"] = test_endpoint(
        "/analyze/crime-types", 
        "GET", 
        description="Analyze Crime Types"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    print("SYSTEM ENDPOINTS:")
    print(f"  Health Check: {'✅' if results['health'] else '❌'}")
    print(f"  Models Info: {'✅' if results['models_info'] else '❌'}")
    
    print("\nPREDICTION ENDPOINTS:")
    print(f"  Spatial Prediction: {'✅' if results['spatial_prediction'] else '❌'}")
    print(f"  Temporal Prediction: {'✅' if results['temporal_prediction'] else '❌'}")
    
    print("\nNETWORK ANALYSIS ENDPOINTS:")
    print(f"  Network Build: {'✅' if results['network_build'] else '❌'}")
    print(f"  Network Centrality: {'✅' if results['network_centrality'] else '❌'}")
    print(f"  Network Communities: {'✅' if results['network_communities'] else '❌'}")
    print(f"  Network Summary: {'✅' if results['network_summary'] else '❌'}")
    
    print("\nSPATIAL ANALYSIS ENDPOINTS:")
    print(f"  Spatial Prepare: {'✅' if results['spatial_prepare'] else '❌'}")
    print(f"  Spatial Cluster: {'✅' if results['spatial_cluster'] else '❌'}")
    print(f"  Spatial Hotspots: {'✅' if results['spatial_hotspots'] else '❌'}")
    print(f"  Spatial Summary: {'✅' if results['spatial_summary'] else '❌'}")
    
    print("\nTEMPORAL ANALYSIS ENDPOINTS:")
    print(f"  Temporal Prepare: {'✅' if results['temporal_prepare'] else '❌'}")
    print(f"  Temporal Seasonality: {'✅' if results['temporal_seasonality'] else '❌'}")
    print(f"  Temporal ARIMA: {'✅' if results['temporal_arima'] else '❌'}")
    print(f"  Temporal Forecast: {'✅' if results['temporal_forecast'] else '❌'}")
    print(f"  Temporal Summary: {'✅' if results['temporal_summary'] else '❌'}")
    
    print("\nCLASSIFICATION ENDPOINTS:")
    print(f"  Crime Classification: {'✅' if results['crime_classification'] else '❌'}")
    print(f"  Crime Distribution: {'✅' if results['crime_distribution'] else '❌'}")
    print(f"  Crime Trends: {'✅' if results['crime_trends'] else '❌'}")
    print(f"  Crime Summary: {'✅' if results['crime_summary'] else '❌'}")
    
    print("\nDATA ANALYSIS ENDPOINTS:")
    print(f"  Analysis Report: {'✅' if results['analysis_report'] else '❌'}")
    print(f"  Analyze Crime Types: {'✅' if results['analyze_crime_types'] else '❌'}")
    
    print(f"\n{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! API is fully functional!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed.")

if __name__ == "__main__":
    main()
