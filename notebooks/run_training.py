import sys
import os

# Add project root directory to path so Python finds 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.ingestion import DataIngestion
from src.features.preprocessing import fit_and_save_pipeline
from src.models.train import FraudModelTrainer

def main():
    print("Ingesting and processing dataset...")
    ingestion = DataIngestion()
    df = ingestion.load()
    _, df_processed = fit_and_save_pipeline(df)
    
    print("\nInitializing Fraud Model Trainer...")
    trainer = FraudModelTrainer()
    
    # 1. Train Isolation Forest
    print("\nTraining Model 1: Isolation Forest...")
    r1 = trainer.train_isolation_forest(df_processed)
    print(f"-> Isolation Forest Complete!")
    
    # 2. Train Local Outlier Factor (Smart Check for Naming)
    print("\nTraining Model 2: Local Outlier Factor...")
    if hasattr(trainer, 'train_lof'):
        r2 = trainer.train_lof(df_processed)
        print("-> Local Outlier Factor Complete!")
    elif hasattr(trainer, 'train_local_outlier_factor'):
        r2 = trainer.train_local_outlier_factor(df_processed)
        print("-> Local Outlier Factor Complete!")
    else:
        print("⚠️ Warning: Could not find a LOF training method in train.py. Skipping.")
        # Let's see what methods actually exist to help you see
        methods = [m for m in dir(trainer) if not m.startswith('_')]
        print(f"Available methods in your train.py: {methods}")
    
    print("\nEvaluating and saving best model architecture...")
    best_path = trainer.save_best_model()
    print(f"Success! Model exported securely to: {best_path}")

if __name__ == "__main__":
    main()
