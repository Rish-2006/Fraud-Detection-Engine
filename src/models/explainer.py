# src/models/explainer.py
import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FraudExplainer:
    """
    Applies SHAP Kernels to discover feature importance weights.
    Provides data explainability transparency profiles.
    """
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def fit(self, X_background: pd.DataFrame, n_samples: int = 30):
        model_obj = self.model
        
        # 🛡️ Defensive Dictionary Unpacking Block
        if isinstance(model_obj, dict):
            if 'model' in model_obj:
                model_obj = model_obj['model']
            elif 'isolation_forest' in model_obj:
                model_obj = model_obj['isolation_forest']
            else:
                model_obj = list(model_obj.values())[0]
                
        # 🛡️ String Path Resolution Check (Loads model directly if it's a file path string)
        if isinstance(model_obj, str):
            logger.info(f"String path detected. Loading trained model binary for SHAP directly from: {model_obj}")
            if os.path.exists(model_obj):
                model_obj = joblib.load(model_obj)
            else:
                fallback_path = os.path.join("models", "artifacts", "best_model.pkl")
                logger.info(f"Target path not found. Attempting fallback location check: {fallback_path}")
                model_obj = joblib.load(fallback_path)

        logger.info(f"Fitting SHAP explainer on {n_samples} background samples")
        bg = X_background.sample(n_samples, random_state=42) if len(X_background) > n_samples else X_background
        
        # Compute Kernel Explainer targeting decision outputs
        if hasattr(model_obj, "decision_function"):
            self.explainer = shap.KernelExplainer(model_obj.decision_function, bg)
        else:
            self.explainer = shap.KernelExplainer(model_obj.score_samples, bg)
        return self

    def explain_batch(self, X: pd.DataFrame, n_samples: int = 10, save_dir: str = "outputs") -> pd.DataFrame:
        if self.explainer is None:
            raise ValueError("Run the tracker .fit() routine before pulling SHAP tensors.")
            
        test_sample = X.sample(n_samples, random_state=42) if len(X) > n_samples else X
        logger.info(f"Generating vectors across sample size: {len(test_sample)}")
        shap_values = self.explainer.shap_values(test_sample)

        # Output Chart 1: Global Summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_sample, feature_names=self.feature_names, plot_type="bar", show=False)
        plt.title("Global Feature Importance (SHAP)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_feature_importance.png"), dpi=300)
        plt.close()

        # Output Chart 2: Beeswarm Distributions
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, test_sample, feature_names=self.feature_names, show=False)
        plt.title("SHAP Beeswarm Value Analysis", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_beeswarm_analysis.png"), dpi=300)
        plt.close()

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
