import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LAPDCrimeDataExplorer:
    """Class to explore and analyze LAPD crime data."""
    
    def __init__(self, data_path: str = "/Users/harrisk/Documents/forensic-application-capstone/data/structured/crime_data.csv"):
        """
        Initialize the data explorer.
        
        Args:
            data_path (str): Path to the LAPD crime data CSV file
        """
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        """Load the real LAPD crime data into a pandas DataFrame."""
        if not self.data_path.exists():
            logger.error(f"Real LAPD data not found at {self.data_path}")
            raise FileNotFoundError(f"Real LAPD data not found at {self.data_path}")
        logger.info(f"Loading real LAPD data from {self.data_path}")
        df = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df

    def analyze_data_structure(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the basic structure of the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing data structure information
        """
        structure_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # in MB
        }
        
        return structure_info
    
    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze temporal patterns in the data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing temporal analysis results
        """
        # Convert date columns to datetime
        df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
        
        temporal_info = {
            'date_range': {
                'start': df['DATE OCC'].min(),
                'end': df['DATE OCC'].max()
            },
            'crimes_by_year': df['DATE OCC'].dt.year.value_counts().to_dict(),
            'crimes_by_month': df['DATE OCC'].dt.month.value_counts().to_dict(),
            'crimes_by_day': df['DATE OCC'].dt.dayofweek.value_counts().to_dict(),
            'crimes_by_hour': df['TIME OCC'].value_counts().sort_index().to_dict()
        }
        
        return temporal_info
    
    def analyze_spatial_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze spatial patterns in the data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing spatial analysis results
        """
        spatial_info = {
            'crimes_by_area': df['AREA NAME'].value_counts().to_dict(),
            'coordinate_stats': {
                'lat': {
                    'min': df['LAT'].min(),
                    'max': df['LAT'].max(),
                    'mean': df['LAT'].mean()
                },
                'lon': {
                    'min': df['LON'].min(),
                    'max': df['LON'].max(),
                    'mean': df['LON'].mean()
                }
            }
        }
        
        return spatial_info
    
    def analyze_crime_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze crime patterns and categories.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing crime pattern analysis results
        """
        crime_info = {
            'crimes_by_type': df['Crm Cd Desc'].value_counts().to_dict(),
            'crimes_by_status': df['Status Desc'].value_counts().to_dict(),
            'crimes_by_weapon': df['Weapon Desc'].value_counts().to_dict(),
            'crimes_by_premise': df['Premis Desc'].value_counts().to_dict()
        }
        
        return crime_info
    
    def analyze_victim_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze victim-related patterns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing victim pattern analysis results
        """
        victim_info = {
            'victim_age_stats': {
                'min': df['Vict Age'].min(),
                'max': df['Vict Age'].max(),
                'mean': df['Vict Age'].mean(),
                'median': df['Vict Age'].median()
            },
            'victim_sex_distribution': df['Vict Sex'].value_counts().to_dict(),
            'victim_descent_distribution': df['Vict Descent'].value_counts().to_dict()
        }
        
        return victim_info
    
    def generate_exploration_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive exploration report.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Dict: Dictionary containing the complete exploration report
        """
        report = {
            'data_structure': self.analyze_data_structure(df),
            'temporal_patterns': self.analyze_temporal_patterns(df),
            'spatial_patterns': self.analyze_spatial_patterns(df),
            'crime_patterns': self.analyze_crime_patterns(df),
            'victim_patterns': self.analyze_victim_patterns(df)
        }
        
        return report
    
    def save_exploration_report(self, report: Dict, output_path: str = "data/analysis/exploration_report.json"):
        """
        Save the exploration report to a JSON file.
        
        Args:
            report (Dict): Exploration report dictionary
            output_path (str): Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logger.info(f"Exploration report saved to {output_path}")

if __name__ == "__main__":
    # Create explorer instance
    explorer = LAPDCrimeDataExplorer()
    
    # Load real data
    df = explorer.load_data()
    
    # Generate and save exploration report
    report = explorer.generate_exploration_report(df)
    explorer.save_exploration_report(report)
    
    # Print key findings
    print("\nKey Data Structure Information:")
    print(f"Number of records: {report['data_structure']['shape'][0]}")
    print(f"Number of columns: {report['data_structure']['shape'][1]}")
    print(f"Memory usage: {report['data_structure']['memory_usage']:.2f} MB")
    
    print("\nDate Range:")
    print(f"Start: {report['temporal_patterns']['date_range']['start']}")
    print(f"End: {report['temporal_patterns']['date_range']['end']}")
    
    print("\nTop 5 Crime Types:")
    top_crimes = dict(sorted(report['crime_patterns']['crimes_by_type'].items(), 
                           key=lambda x: x[1], reverse=True)[:5])
    for crime, count in top_crimes.items():
        print(f"{crime}: {count}") 