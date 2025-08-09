"""
Module: criminal_network_analyzer
Description: Criminal network analysis for API consumption
Author: Kelly-Ann Harris
Date: 2024
"""

import pandas as pd
import networkx as nx
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CriminalNetworkAnalyzer:
    """
    Criminal network analysis service for API consumption.
    
    This class provides methods to analyze criminal networks using graph theory,
    including centrality analysis, community detection, and network metrics.
    """
    
    def __init__(self):
        """Initialize the criminal network analyzer."""
        self.network = None
        self.centrality_scores = None
        self.communities = None
    
    def build_network_from_data(self, crime_data: pd.DataFrame) -> Dict[str, int]:
        """
        Build a criminal network from crime data.
        
        Args:
            crime_data (pd.DataFrame): Crime data with 'LOCATION' and 'Mocodes' columns
                
        Returns:
            Dict[str, int]: Network statistics
                Format: {"nodes": int, "edges": int}
        """
        try:
            logger.info("Building criminal network from data...")
            
            # Clean data
            df = crime_data.dropna(subset=['LOCATION', 'Mocodes'])
            
            # Build edges
            edges = []
            for idx, row in df.iterrows():
                location = row['LOCATION']
                mocodes = [m.strip() for m in str(row['Mocodes']).split(',') if m.strip()]
                for m in mocodes:
                    edges.append((location, m))
            
            # Create network
            self.network = nx.Graph()
            self.network.add_edges_from(edges)
            
            stats = {
                "nodes": self.network.number_of_nodes(),
                "edges": self.network.number_of_edges()
            }
            
            logger.info(f"Network built: {stats['nodes']} nodes, {stats['edges']} edges.")
            return stats
            
        except Exception as e:
            logger.error(f"Error building network: {e}")
            raise
    
    def analyze_centrality(self, top_n: int = 100) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Analyze network centrality measures.
        
        Args:
            top_n (int): Number of top nodes to return
            
        Returns:
            Dict[str, List[Dict[str, Union[str, float]]]]: Centrality analysis results
                Format: {
                    "degree_centrality": [{"node": "node_name", "score": 0.5}, ...],
                    "betweenness_centrality": [{"node": "node_name", "score": 0.3}, ...],
                    "eigenvector_centrality": [{"node": "node_name", "score": 0.7}, ...]
                }
        """
        try:
            if self.network is None:
                raise ValueError("Network not built. Call build_network_from_data() first.")
            
            logger.info("Analyzing network centrality...")
            
            # Calculate different centrality measures
            degree_cent = nx.degree_centrality(self.network)
            betweenness_cent = nx.betweenness_centrality(self.network)
            eigenvector_cent = nx.eigenvector_centrality(self.network, max_iter=1000)
            
            # Store centrality scores
            self.centrality_scores = {
                "degree": degree_cent,
                "betweenness": betweenness_cent,
                "eigenvector": eigenvector_cent
            }
            
            # Get top nodes for each centrality measure
            results = {}
            for cent_type, cent_scores in self.centrality_scores.items():
                top_nodes = sorted(cent_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                results[f"{cent_type}_centrality"] = [
                    {"node": node, "score": float(score)} for node, score in top_nodes
                ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing centrality: {e}")
            raise
    
    def detect_communities(self, algorithm: str = "greedy_modularity") -> Dict[str, Union[List[List[str]], Dict[str, any]]]:
        """
        Detect communities in the criminal network.
        
        Args:
            algorithm (str): Community detection algorithm
                Options: "greedy_modularity", "louvain", "label_propagation"
                
        Returns:
            Dict[str, Union[List[List[str]], Dict[str, any]]]: Community detection results
                Format: {
                    "communities": [["node1", "node2", ...], ["node3", "node4", ...], ...],
                    "statistics": {
                        "num_communities": 5,
                        "largest_community_size": 100,
                        "modularity": 0.75
                    }
                }
        """
        try:
            if self.network is None:
                raise ValueError("Network not built. Call build_network_from_data() first.")
            
            logger.info(f"Detecting communities using {algorithm} algorithm...")
            
            if algorithm == "greedy_modularity":
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(self.network))
            elif algorithm == "louvain":
                from community import best_partition
                partition = best_partition(self.network)
                communities = []
                for com in set(partition.values()):
                    communities.append([nodes for nodes in partition.keys() if partition[nodes] == com])
            elif algorithm == "label_propagation":
                from networkx.algorithms.community import label_propagation_communities
                communities = list(label_propagation_communities(self.network))
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Store communities
            self.communities = communities
            
            # Calculate statistics
            community_sizes = [len(comm) for comm in communities]
            modularity = nx.community.modularity(self.network, communities)
            
            statistics = {
                "num_communities": len(communities),
                "largest_community_size": max(community_sizes) if community_sizes else 0,
                "smallest_community_size": min(community_sizes) if community_sizes else 0,
                "average_community_size": np.mean(community_sizes) if community_sizes else 0,
                "modularity": float(modularity)
            }
            
            return {
                "communities": [list(comm) for comm in communities],
                "statistics": statistics
            }
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            raise
    
    def calculate_network_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Calculate comprehensive network metrics.
        
        Returns:
            Dict[str, Union[int, float]]: Network metrics
                Format: {
                    "density": 0.1,
                    "average_clustering": 0.5,
                    "average_shortest_path": 3.2,
                    "diameter": 8,
                    "connected_components": 3,
                    "largest_component_size": 1000
                }
        """
        try:
            if self.network is None:
                raise ValueError("Network not built. Call build_network_from_data() first.")
            
            logger.info("Calculating network metrics...")
            
            # Basic metrics
            density = nx.density(self.network)
            average_clustering = nx.average_clustering(self.network)
            
            # Connected components
            connected_components = list(nx.connected_components(self.network))
            largest_component = max(connected_components, key=len)
            largest_component_graph = self.network.subgraph(largest_component)
            
            # Path metrics (for largest component only)
            try:
                average_shortest_path = nx.average_shortest_path_length(largest_component_graph)
                diameter = nx.diameter(largest_component_graph)
            except:
                average_shortest_path = float('inf')
                diameter = 0
            
            metrics = {
                "density": float(density),
                "average_clustering": float(average_clustering),
                "average_shortest_path": float(average_shortest_path) if average_shortest_path != float('inf') else None,
                "diameter": int(diameter),
                "connected_components": len(connected_components),
                "largest_component_size": len(largest_component)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
            raise
    
    def find_key_players(self, top_n: int = 10) -> Dict[str, List[Dict[str, Union[str, float, List[str]]]]]:
        """
        Find key players in the criminal network using multiple criteria.
        
        Args:
            top_n (int): Number of top key players to return
            
        Returns:
            Dict[str, List[Dict[str, Union[str, float, List[str]]]]]: Key players analysis
                Format: {
                    "key_players": [
                        {
                            "node": "node_name",
                            "degree_centrality": 0.8,
                            "betweenness_centrality": 0.6,
                            "eigenvector_centrality": 0.7,
                            "clustering_coefficient": 0.5,
                            "connections": ["node1", "node2", ...]
                        }
                    ]
                }
        """
        try:
            if self.network is None:
                raise ValueError("Network not built. Call build_network_from_data() first.")
            
            if self.centrality_scores is None:
                self.analyze_centrality()
            
            logger.info("Finding key players in the network...")
            
            # Calculate additional metrics
            clustering_coeff = nx.clustering(self.network)
            
            # Combine all metrics for each node
            node_scores = {}
            for node in self.network.nodes():
                node_scores[node] = {
                    "node": node,
                    "degree_centrality": self.centrality_scores["degree"].get(node, 0),
                    "betweenness_centrality": self.centrality_scores["betweenness"].get(node, 0),
                    "eigenvector_centrality": self.centrality_scores["eigenvector"].get(node, 0),
                    "clustering_coefficient": clustering_coeff.get(node, 0),
                    "connections": list(self.network.neighbors(node))
                }
            
            # Calculate composite score (weighted average)
            for node_data in node_scores.values():
                composite_score = (
                    0.3 * node_data["degree_centrality"] +
                    0.3 * node_data["betweenness_centrality"] +
                    0.3 * node_data["eigenvector_centrality"] +
                    0.1 * node_data["clustering_coefficient"]
                )
                node_data["composite_score"] = composite_score
            
            # Sort by composite score and get top players
            key_players = sorted(node_scores.values(), 
                               key=lambda x: x["composite_score"], 
                               reverse=True)[:top_n]
            
            return {"key_players": key_players}
            
        except Exception as e:
            logger.error(f"Error finding key players: {e}")
            raise
    
    def get_network_summary(self) -> Dict[str, any]:
        """
        Get a comprehensive summary of the network analysis.
        
        Returns:
            Dict[str, any]: Network summary
        """
        try:
            if self.network is None:
                raise ValueError("Network not built. Call build_network_from_data() first.")
            
            # Get all analyses
            centrality = self.analyze_centrality(top_n=10)
            communities = self.detect_communities()
            metrics = self.calculate_network_metrics()
            key_players = self.find_key_players(top_n=10)
            
            summary = {
                "network_statistics": {
                    "nodes": self.network.number_of_nodes(),
                    "edges": self.network.number_of_edges()
                },
                "centrality_analysis": centrality,
                "community_detection": communities,
                "network_metrics": metrics,
                "key_players": key_players
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating network summary: {e}")
            raise 