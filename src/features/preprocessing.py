# src/features/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.features.engineer import FraudFeatureEngineer, RAW_FEATURES
from src.utils.logger import get_logger
import joblib, os

logger = get_logger(__name__)

def build_preprocessing_pipeline() -> Pipeline:
    """
    Build the full preprocessing pipeline.
    
    Steps:
    1. Feature engineering (creates 8 new features)
    2. Imputation (fills missing values with median)
    3. Scaling (StandardScaler → mean=0, std=1)
    
    Returns sklearn Pipeline ready to fit_transform.
    """
    pipeline = Pipeline(steps=[
        ('engineer', FraudFeatureEngineer()),
        ('imputer',  SimpleImputer(strategy='median')),
        ('scaler',   StandardScaler())
    ])
    logger.info('Preprocessing pipeline built: engineer → imputer → scaler')
    return pipeline


def fit_and_save_pipeline(df: pd.DataFrame,
                          save_path: str = 'models/artifacts/pipeline.pkl'
                          ) -> tuple:
    """
    Fit pipeline on training data, save it, return (pipeline, df_processed).
    """
    # Select raw features
    X_raw = df[RAW_FEATURES].copy()
    
    # Build and fit pipeline
    pipeline = build_preprocessing_pipeline()
    X_processed = pipeline.fit_transform(X_raw)
    
    # Get feature names after engineering
    engineer = pipeline.named_steps['engineer']
    feature_names = engineer.all_features
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    
    # Save pipeline
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f'Pipeline saved to {save_path}')
    logger.info(f'Output shape: {df_processed.shape}')
    
    return pipeline, df_processed