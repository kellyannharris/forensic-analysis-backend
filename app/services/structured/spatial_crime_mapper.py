"""
Module: spatial_crime_mapper
Description: Spatial crime mapping and hotspot analysis for API consumption
Author: Kelly-Ann Harris
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialCrimeMapper:
    """
    Spatial crime mapping service for API consumption.
    
    This class provides methods to analyze spatial patterns in crime data,
    including hotspot detection, clustering analysis, and spatial statistics.
    """
    
    def __init__(self):
        """Initialize the spatial crime mapper."""
        self.clustering_model = None
        self.spatial_data = None
        self.cluster_centers = None
    
    def prepare_spatial_data(self, crime_data: pd.DataFrame) -> Dict[str, Union[int, List[float]]]:
        """
        Prepare spatial data for analysis, filtering out invalid coordinates.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'LAT' and 'LON' columns
                
        Returns:
            Dict[str, Union[int, List[float]]]: Spatial data statistics
                Format: {
                    "total_records": 1000,
                    "valid_records": 950,
                    "invalid_records": 50,
                    "lat_range": [33.0, 35.0],
                    "lon_range": [-119.0, -117.0]
                }
        """
        try:
            logger.info("Preparing spatial data for analysis...")
            
            # Filter for valid LA coordinates
            valid_mask = (
                crime_data['LAT'].between(33, 35) &
                crime_data['LON'].between(-119, -117)
            )
            
            valid_data = crime_data.loc[valid_mask]
            invalid_count = len(crime_data) - len(valid_data)
            
            # Extract spatial coordinates
            self.spatial_data = valid_data[['LAT', 'LON']].dropna().values
            
            stats = {
                "total_records": len(crime_data),
                "valid_records": len(valid_data),
                "invalid_records": invalid_count,
                "lat_range": [float(valid_data['LAT'].min()), float(valid_data['LAT'].max())],
                "lon_range": [float(valid_data['LON'].min()), float(valid_data['LON'].max())]
            }
            
            logger.info(f"Prepared spatial data: {stats['valid_records']} valid records")
            return stats
            
        except Exception as e:
            logger.error(f"Error preparing spatial data: {e}")
            raise
    
    def perform_clustering(self, n_clusters: int = 5, algorithm: str = "kmeans") -> Dict[str, Union[List[List[float]], Dict[str, any]]]:
        """
        Perform spatial clustering on crime data.
        
        Args:
            n_clusters (int): Number of clusters to create
            algorithm (str): Clustering algorithm ("kmeans", "dbscan", "hierarchical")
                
        Returns:
            Dict[str, Union[List[List[float]], Dict[str, any]]]: Clustering results
                Format: {
                    "cluster_centers": [[lat1, lon1], [lat2, lon2], ...],
                    "cluster_assignments": [0, 1, 2, 0, 1, ...],
                    "statistics": {
                        "num_clusters": 5,
                        "cluster_sizes": [100, 150, 200, 120, 130],
                        "inertia": 1234.56
                    }
                }
        """
        try:
            if self.spatial_data is None:
                raise ValueError("Spatial data not prepared. Call prepare_spatial_data() first.")
            
            logger.info(f"Performing {algorithm} clustering with {n_clusters} clusters...")
            
            if algorithm == "kmeans":
                self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_assignments = self.clustering_model.fit_predict(self.spatial_data)
                self.cluster_centers = self.clustering_model.cluster_centers_
                inertia = self.clustering_model.inertia_
            elif algorithm == "dbscan":
                from sklearn.cluster import DBSCAN
                self.clustering_model = DBSCAN(eps=0.01, min_samples=5)
                cluster_assignments = self.clustering_model.fit_predict(self.spatial_data)
                self.cluster_centers = []
                for cluster_id in set(cluster_assignments):
                    if cluster_id != -1:  # Skip noise points
                        cluster_points = self.spatial_data[cluster_assignments == cluster_id]
                        self.cluster_centers.append(cluster_points.mean(axis=0))
                inertia = None
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Calculate cluster statistics
            unique_clusters, cluster_sizes = np.unique(cluster_assignments, return_counts=True)
            cluster_stats = {
                "num_clusters": len(unique_clusters),
                "cluster_sizes": cluster_sizes.tolist(),
                "inertia": float(inertia) if inertia is not None else None
            }
            
            results = {
                "cluster_centers": self.cluster_centers.tolist() if hasattr(self.cluster_centers, 'tolist') else self.cluster_centers,
                "cluster_assignments": cluster_assignments.tolist(),
                "statistics": cluster_stats
            }
            
            logger.info(f"Clustering completed: {cluster_stats['num_clusters']} clusters found")
            return results
            
        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
            raise
    
    def analyze_hotspots(self, crime_data: pd.DataFrame, radius_km: float = 1.0) -> Dict[str, Union[List[Dict[str, any]], Dict[str, any]]]:
        """
        Analyze crime hotspots using density-based analysis.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'LAT' and 'LON' columns
            radius_km (float): Radius in kilometers for hotspot analysis
                
        Returns:
            Dict[str, Union[List[Dict[str, any]], Dict[str, any]]]: Hotspot analysis results
                Format: {
                    "hotspots": [
                        {
                            "center": [lat, lon],
                            "radius_km": 1.0,
                            "crime_count": 150,
                            "crime_types": ["THEFT", "ASSAULT", ...],
                            "density": 47.7
                        }
                    ],
                    "statistics": {
                        "total_hotspots": 5,
                        "total_crimes_in_hotspots": 750,
                        "average_crimes_per_hotspot": 150
                    }
                }
        """
        try:
            logger.info(f"Analyzing crime hotspots with {radius_km}km radius...")
            
            # Convert radius to degrees (approximate)
            radius_deg = radius_km / 111.0  # 1 degree ≈ 111 km
            
            # Prepare spatial data
            valid_mask = (
                crime_data['LAT'].between(33, 35) &
                crime_data['LON'].between(-119, -117)
            )
            valid_data = crime_data.loc[valid_mask].copy()
            
            # Use clustering centers as hotspot centers if available
            if self.cluster_centers is not None:
                hotspot_centers = self.cluster_centers
            else:
                # Create grid-based hotspots
                lat_range = np.linspace(valid_data['LAT'].min(), valid_data['LAT'].max(), 5)
                lon_range = np.linspace(valid_data['LON'].min(), valid_data['LON'].max(), 5)
                hotspot_centers = []
                for lat in lat_range:
                    for lon in lon_range:
                        hotspot_centers.append([lat, lon])
            
            hotspots = []
            total_crimes_in_hotspots = 0
            
            for center in hotspot_centers:
                # Find crimes within radius
                distances = np.sqrt(
                    (valid_data['LAT'] - center[0])**2 + 
                    (valid_data['LON'] - center[1])**2
                )
                crimes_in_radius = valid_data[distances <= radius_deg]
                
                if len(crimes_in_radius) > 0:
                    # Calculate density (crimes per square km)
                    area_km2 = np.pi * radius_km**2
                    density = len(crimes_in_radius) / area_km2
                    
                    # Get crime types
                    crime_types = crimes_in_radius['Crm Cd Desc'].value_counts().head(5).index.tolist()
                    
                    hotspot = {
                        "center": [float(center[0]), float(center[1])],
                        "radius_km": radius_km,
                        "crime_count": len(crimes_in_radius),
                        "crime_types": crime_types,
                        "density": float(density)
                    }
                    hotspots.append(hotspot)
                    total_crimes_in_hotspots += len(crimes_in_radius)
            
            # Sort hotspots by crime count
            hotspots.sort(key=lambda x: x["crime_count"], reverse=True)
            
            statistics = {
                "total_hotspots": len(hotspots),
                "total_crimes_in_hotspots": total_crimes_in_hotspots,
                "average_crimes_per_hotspot": total_crimes_in_hotspots / len(hotspots) if hotspots else 0
            }
            
            return {
                "hotspots": hotspots,
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing hotspots: {e}")
            raise
    
    def calculate_spatial_statistics(self, crime_data: pd.DataFrame) -> Dict[str, Union[float, List[float]]]:
        """
        Calculate spatial statistics for crime data.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'LAT' and 'LON' columns
                
        Returns:
            Dict[str, Union[float, List[float]]]: Spatial statistics
                Format: {
                    "mean_center": [lat, lon],
                    "standard_distance": 0.5,
                    "spatial_autocorrelation": 0.75,
                    "nearest_neighbor_ratio": 0.8
                }
        """
        try:
            logger.info("Calculating spatial statistics...")
            
            # Prepare valid data
            valid_mask = (
                crime_data['LAT'].between(33, 35) &
                crime_data['LON'].between(-119, -117)
            )
            valid_data = crime_data.loc[valid_mask]
            
            # Mean center
            mean_center = [float(valid_data['LAT'].mean()), float(valid_data['LON'].mean())]
            
            # Standard distance
            lat_var = valid_data['LAT'].var()
            lon_var = valid_data['LON'].var()
            standard_distance = np.sqrt(lat_var + lon_var)
            
            # Nearest neighbor ratio (simplified)
            from scipy.spatial.distance import pdist, squareform
            coords = valid_data[['LAT', 'LON']].values
            distances = pdist(coords)
            mean_nearest_neighbor = np.mean(np.min(squareform(distances), axis=1))
            
            # Expected nearest neighbor distance for random distribution
            area = (valid_data['LAT'].max() - valid_data['LAT'].min()) * (valid_data['LON'].max() - valid_data['LON'].min())
            density = len(valid_data) / area
            expected_distance = 1 / (2 * np.sqrt(density))
            
            nearest_neighbor_ratio = mean_nearest_neighbor / expected_distance
            
            statistics = {
                "mean_center": mean_center,
                "standard_distance": float(standard_distance),
                "nearest_neighbor_ratio": float(nearest_neighbor_ratio),
                "total_crimes": len(valid_data),
                "area_covered": float(area)
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating spatial statistics: {e}")
            raise
    
    def get_spatial_summary(self, crime_data: pd.DataFrame) -> Dict[str, any]:
        """
        Get a comprehensive spatial analysis summary.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'LAT' and 'LON' columns
                
        Returns:
            Dict[str, any]: Spatial analysis summary
        """
        try:
            logger.info("Generating spatial analysis summary...")
            
            # Prepare data
            data_stats = self.prepare_spatial_data(crime_data)
            
            # Perform clustering
            clustering_results = self.perform_clustering(n_clusters=5)
            
            # Analyze hotspots
            hotspot_results = self.analyze_hotspots(crime_data)
            
            # Calculate spatial statistics
            spatial_stats = self.calculate_spatial_statistics(crime_data)
            
            summary = {
                "data_preparation": data_stats,
                "clustering_analysis": clustering_results,
                "hotspot_analysis": hotspot_results,
                "spatial_statistics": spatial_stats
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating spatial summary: {e}")
            raise 