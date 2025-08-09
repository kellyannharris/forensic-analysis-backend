import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LAPDCrimeDataPreprocessor:
    """Class to preprocess LAPD crime data."""

    def __init__(self, data_path: str = "/Users/harrisk/Documents/forensic-application-capstone/data/structured/crime_data.csv"):
        """
        Initialize the data preprocessor.
        
        Args:
            data_path (str): Path to the LAPD crime data CSV file
        """
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        """Load the LAPD crime data into a pandas DataFrame."""
        if not self.data_path.exists():
            logger.error(f"LAPD data not found at {self.data_path}")
            raise FileNotFoundError(f"LAPD data not found at {self.data_path}")
        logger.info(f"Loading LAPD data from {self.data_path}")
        df = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        # Fill missing values in categorical columns with 'Unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        # Fill missing values in numerical columns with the median
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())
        
        logger.info("Handled missing values in the DataFrame.")
        return df

    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types to appropriate formats.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with converted data types
        """
        # Convert date columns to datetime
        df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
        
        # Convert categorical columns to category type
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].astype('category')
        
        logger.info("Converted data types in the DataFrame.")
        return df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the LAPD crime data.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        df = self.load_data()
        df = self.handle_missing_values(df)
        df = self.convert_data_types(df)
        logger.info("Data preprocessing completed.")
        return df

if __name__ == "__main__":
    # Create preprocessor instance
    preprocessor = LAPDCrimeDataPreprocessor()
    
    # Preprocess the data
    preprocessed_df = preprocessor.preprocess_data()
    
    # Display the first few rows of the preprocessed DataFrame
    print(preprocessed_df.head()) 