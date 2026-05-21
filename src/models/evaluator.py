# src/models/evaluator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from src.utils.logger import get_logger
import json, os, joblib

logger = get_logger(__name__)

class ModelEvaluator:
    """
    Evaluates anomaly detection model and generates reports.
    For unsupervised models, we evaluate score distributions,
    threshold sensitivity, and feature impact.
    """
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
    
    def evaluate(self, X: pd.DataFrame, save_dir: str = 'outputs') -> dict:
        os.makedirs(save_dir, exist_ok=True)
        
        # 🛡️ DEFENSIVE DICTIONARY UNPACKING BLOCK
        model_obj = self.model
        if isinstance(model_obj, dict):
            logger.info("Dictionary model wrapper detected. Extracting core estimator...")
            if 'model' in model_obj:
                model_obj = model_obj['model']
            elif 'isolation_forest' in model_obj:
                model_obj = model_obj['isolation_forest']
            else:
                # Fallback to the first trained item in your dict
                model_obj = list(model_obj.values())[0]
        
        # 🛡️ NEW: STR PATH CHECK (Loads actual model binary from disk if it's a file path string)
        if isinstance(model_obj, str):
            logger.info(f"String path detected. Loading trained model binary directly from: {model_obj}")
            if os.path.exists(model_obj):
                model_obj = joblib.load(model_obj)
            else:
                fallback_path = os.path.join("models", "artifacts", "best_model.pkl")
                logger.info(f"Target path not found. Attempting fallback location check: {fallback_path}")
                model_obj = joblib.load(fallback_path)
        
        # Compute inferences against the clean model estimator object
        preds  = model_obj.predict(X)
        scores = model_obj.decision_function(X)
        
        fraud_mask  = preds == -1
        normal_mask = preds == 1
        
        metrics = {
            'total_transactions':  len(X),
            'fraud_flagged':       int(fraud_mask.sum()),
            'normal_transactions': int(normal_mask.sum()),
            'fraud_rate_pct':      round(fraud_mask.mean() * 100, 3),
            'score_mean_all':      round(float(scores.mean()), 4),
            'score_mean_fraud':    round(float(scores[fraud_mask].mean()), 4) if fraud_mask.sum() > 0 else 0.0,
            'score_mean_normal':   round(float(scores[normal_mask].mean()), 4) if normal_mask.sum() > 0 else 0.0,
            'score_separation':    round(
                float(scores[normal_mask].mean()) - float(scores[fraud_mask].mean()), 4
            ) if (normal_mask.sum() > 0 and fraud_mask.sum() > 0) else 0.0,
            'high_risk_count':     int((scores < -0.15).sum()),
            'medium_risk_count':   int(((scores >= -0.15) & (scores < -0.05)).sum())
        }
        
        logger.info(f'Evaluation complete: {metrics["fraud_rate_pct"]}% fraud rate')
        logger.info(f'Score separation (higher=better): {metrics["score_separation"]:.4f}')
        
        # Save metrics
        with open(f'{save_dir}/evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self._plot_evaluation(scores, preds, save_dir)
        return metrics
    
    def _plot_evaluation(self, scores, preds, save_dir):
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig)
        
        fraud_scores  = scores[preds == -1]
        normal_scores = scores[preds == 1]
        
        # Plot 1: Score distribution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(normal_scores, bins=80, color='steelblue',
                 alpha=0.6, label='Normal', density=True)
        if len(fraud_scores) > 0:
            ax1.hist(fraud_scores,  bins=80, color='crimson',
                     alpha=0.7, label='FRAUD',  density=True)
        ax1.axvline(-0.15, color='red',    ls='--', lw=1.5, label='High risk threshold')
        ax1.axvline(-0.05, color='orange', ls='--', lw=1.5, label='Medium risk threshold')
        ax1.set_title('Anomaly Score Distribution', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Anomaly Score (lower = more suspicious)')
        ax1.legend()
        
        # Plot 2: Pie chart
        ax2 = fig.add_subplot(gs[0, 2])
        n_fraud  = (preds == -1).sum()
        n_normal = (preds == 1).sum()
        ax2.pie([n_normal, n_fraud],
                labels=['Normal', 'FRAUD'],
                colors=['steelblue', 'crimson'],
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Transaction Breakdown', fontsize=12)
        
        # Plot 3: Risk level bar
        ax3 = fig.add_subplot(gs[1, 0])
        high   = (scores < -0.15).sum()
        medium = ((scores >= -0.15) & (scores < -0.05)).sum()
        low    = (scores >= -0.05).sum()
        ax3.bar(['Low Risk', 'Medium Risk', 'High Risk'],
                [low, medium, high],
                color=['#2ecc71', '#f39c12', '#e74c3c'])
        ax3.set_title('Risk Level Distribution', fontsize=12)
        ax3.set_ylabel('Count')
        
        # Plot 4: Score boxplot by group
        ax4 = fig.add_subplot(gs[1, 1:])
        data_to_box = [normal_scores]
        box_labels = ['Normal']
        if len(fraud_scores) > 0:
            data_to_box.append(fraud_scores)
            box_labels.append('FRAUD')
            
        ax4.boxplot(data_to_box,
                    labels=box_labels,
                    patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7),
                    medianprops=dict(color='white', lw=2))
        ax4.set_title('Score Distribution by Group', fontsize=12)
        ax4.set_ylabel('Anomaly Score')
        
        plt.suptitle('FraudGuard Pro — Model Evaluation Report',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/evaluation_report.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f'Evaluation charts saved to {save_dir}/')
