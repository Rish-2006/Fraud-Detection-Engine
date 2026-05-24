# notebooks/run_training.py
import sys
import os

# Add project root directory to path so Python finds 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.ingestion import DataIngestion
from src.features.preprocessing import fit_and_save_pipeline
from src.models.train import FraudModelTrainer

# Day 5 Custom Modules
from src.models.evaluator import ModelEvaluator
from src.models.explainer import FraudExplainer

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
        methods = [m for m in dir(trainer) if not m.startswith('_')]
        print(f"Available methods in your train.py: {methods}")
    
    print("\nEvaluating and saving best model architecture...")
    best_path = trainer.save_best_model()
    print(f"Success! Model exported securely to: {best_path}")

    # =========================================================================
    # 🆕 DAY 5 RUN LOOPS
    # =========================================================================
    print("\n=== Running Day 5: Model Evaluation & SHAP Interpretability ===")
    
    # Remove label column if present to isolate feature vectors
    features_df = df_processed.drop(columns=['is_fraud'], errors='ignore')
    feature_list = list(features_df.columns)

    # 1. Run Evaluator Module (Generates metrics file and score graphs)
    print("Generating distribution metrics and plots...")
    evaluator = ModelEvaluator(r1, feature_names=feature_list)
    evaluator.evaluate(features_df, save_dir="outputs")
    print("-> Evaluation reports written to 'outputs/' directory.")

    # 🛡️ Defensive Check: Extract raw model estimator for SHAP if wrapped in a dict
    base_model = r1
    if isinstance(base_model, dict):
        if 'model' in base_model:
            base_model = base_model['model']
        elif 'isolation_forest' in base_model:
            base_model = base_model['isolation_forest']
        else:
            base_model = list(base_model.values())[0]

    # 2. Run Explainer Module (Fits SHAP and processes batch explanations)
    print("Fitting SHAP Explainer (sampling for compute optimization)...")
    explainer = FraudExplainer(base_model, feature_names=feature_list)
    explainer.fit(features_df, n_samples=30)
    
    importance_df = explainer.explain_batch(features_df, n_samples=10, save_dir="outputs")
    
    print("\n=== SUCCESS: DAY 5 PIPELINE COMPLETE ===")
    print("Top Feature Anomaly Triggers:")
    print(importance_df.head(3))

if __name__ == "__main__":
    main()
