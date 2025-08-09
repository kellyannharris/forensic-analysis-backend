import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import json
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LAPDCrimeDataAnalyzer:
    """Class to analyze LAPD crime data."""

    def __init__(self, data_path: str = "/Users/harrisk/Documents/forensic-application-capstone/data/structured/crime_data.csv"):
        """
        Initialize the data analyzer.
        
        Args:
            data_path (str): Path to the preprocessed LAPD crime data CSV file
        """
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        """Load the preprocessed LAPD crime data into a pandas DataFrame."""
        if not self.data_path.exists():
            logger.error(f"Preprocessed LAPD data not found at {self.data_path}")
            raise FileNotFoundError(f"Preprocessed LAPD data not found at {self.data_path}")
        logger.info(f"Loading preprocessed LAPD data from {self.data_path}")
        df = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df

    def analyze_crime_types(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of crime types.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing crime type analysis results
        """
        crime_type_counts = df['Crm Cd Desc'].value_counts()
        top_crimes = crime_type_counts.head(5).to_dict()
        logger.info(f"Top 5 crime types: {top_crimes}")
        return top_crimes

    def analyze_temporal_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze temporal trends in crime occurrences.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing temporal trend analysis results
        """
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
        crimes_by_year = df['DATE OCC'].dt.year.value_counts().to_dict()
        logger.info(f"Crimes by year: {crimes_by_year}")
        return crimes_by_year

    def analyze_spatial_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the spatial distribution of crimes.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing spatial distribution analysis results
        """
        crimes_by_area = df['AREA NAME'].value_counts().to_dict()
        logger.info(f"Crimes by area: {crimes_by_area}")
        return crimes_by_area

    def analyze_victim_demographics(self, df: pd.DataFrame) -> Dict:
        """
        Analyze victim demographics.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing victim demographic analysis results
        """
        victim_age_stats = {
            'min': df['Vict Age'].min(),
            'max': df['Vict Age'].max(),
            'mean': df['Vict Age'].mean(),
            'median': df['Vict Age'].median()
        }
        logger.info(f"Victim age statistics: {victim_age_stats}")
        return victim_age_stats

    def generate_analysis_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive analysis report.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing the complete analysis report
        """
        report = {
            'crime_types': self.analyze_crime_types(df),
            'temporal_trends': self.analyze_temporal_trends(df),
            'spatial_distribution': self.analyze_spatial_distribution(df),
            'victim_demographics': self.analyze_victim_demographics(df)
        }
        return report

    def save_analysis_report(self, report: Dict, output_path: str = "data/analysis/analysis_report.json"):
        """
        Save the analysis report to a JSON file.
        
        Args:
            report (Dict): Analysis report dictionary
            output_path (str): Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logger.info(f"Analysis report saved to {output_path}")

if __name__ == "__main__":
    # Create analyzer instance
    analyzer = LAPDCrimeDataAnalyzer()
    
    # Load preprocessed data
    df = analyzer.load_data()
    
    # Generate and save analysis report
    report = analyzer.generate_analysis_report(df)
    analyzer.save_analysis_report(report)
    
    # Print key findings
    print("\nKey Analysis Findings:")
    print("Top 5 Crime Types:")
    for crime, count in report['crime_types'].items():
        print(f"{crime}: {count}")
    
    print("\nCrimes by Year:")
    for year, count in report['temporal_trends'].items():
        print(f"{year}: {count}")
    
    print("\nCrimes by Area:")
    for area, count in report['spatial_distribution'].items():
        print(f"{area}: {count}")
    
    print("\nVictim Age Statistics:")
    for stat, value in report['victim_demographics'].items():
        print(f"{stat}: {value}") 