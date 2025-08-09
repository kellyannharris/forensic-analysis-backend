import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_lapd_data(data_path: str = "/Users/harrisk/Documents/forensic-application-capstone/data/structured/crime_data.csv") -> pd.DataFrame:
    """
    Load LAPD crime data from the specified CSV file.
    Returns a pandas DataFrame.
    """
    path = Path(data_path)
    if path.exists():
        logger.info(f"Loading real LAPD data from {path}")
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df
    else:
        logger.error(f"Real LAPD data not found at {path}")
        raise FileNotFoundError(f"Real LAPD data not found at {path}")

if __name__ == "__main__":
    df = load_lapd_data()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataFrame info:")
    print(df.info()) 