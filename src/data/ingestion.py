# src/data/ingestion.py
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

class DataIngestion:
    """
    Handles loading and initial validation of raw data.
    Industry pattern: encapsulate all data loading logic in one class.
    """
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.raw_path = self.config['data']['raw_path']
        self.encoding = self.config['data']['encoding']
    
    def load(self) -> pd.DataFrame:
        """Load raw CSV and return DataFrame."""
        path = Path(self.raw_path)
        if not path.exists():
            raise FileNotFoundError(
                f'Dataset not found at: {path}\n'
                f'Download from Kaggle and place in data/raw/'
            )
        logger.info(f'Loading dataset from {path}')
        df = pd.read_csv(path, encoding=self.encoding)
        logger.info(f'Loaded {len(df):,} rows x {df.shape[1]} columns')
        return df
    
    def get_summary(self, df: pd.DataFrame) -> dict:
        """Return basic dataset summary as dict."""
        return {
            'rows':          len(df),
            'columns':       df.shape[1],
            'missing_cells': int(df.isnull().sum().sum()),
            'memory_mb':     round(df.memory_usage(deep=True).sum()/1e6, 2),
            'dtypes':        df.dtypes.value_counts().to_dict()
        }