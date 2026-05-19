import sys
import os

# Ensure Python looks inside the root directory to find the 'src' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.ingestion import DataIngestion
from src.features.preprocessing import fit_and_save_pipeline

def main():
    print("Initializing Data Ingestion...")
    # 1. Load data
    ingestion = DataIngestion()
    df = ingestion.load()

    print("Running Preprocessing and Feature Engineering Pipeline...")
    # 2. Process data
    pipeline, df_processed = fit_and_save_pipeline(df)

    # 3. Print clean summary metrics
    print("\n==============================================")
    print("--- PIPELINE RUN SUCCESSFUL ---")
    print("==============================================")
    print(f"Input shape : {df.shape[0]:,} rows x {df.shape[1]} raw features")
    print(f"Output shape: {df_processed.shape[0]:,} rows x {df_processed.shape[1]} features")

    print(f'\nFeature names after engineering:')
    for i, col in enumerate(df_processed.columns, 1):
        tag = '(engineered)' if i > 7 else '(raw)'
        print(f'  {i:2}. {col:<30} {tag}')

    print('\nSample stats (mean should be ~0, std should be ~1):')
    print(df_processed.describe().T[['mean','std']].round(3).head(15))

if __name__ == "__main__":
    main()
