# src/models/trainer.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
import joblib, os, time

logger = get_logger(__name__)

class FraudModelTrainer:
    """
    Trains multiple anomaly detection models and tracks
    experiments with MLflow. Selects best model automatically.
    """
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.models = {}
        self.results = {}
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        mlflow.set_tracking_uri(
            self.config['mlflow']['tracking_uri']
        )
        mlflow.set_experiment(
            self.config['mlflow']['experiment_name']
        )
        logger.info('MLflow tracking configured')
    
    def train_isolation_forest(self, X: pd.DataFrame) -> dict:
        """Train Isolation Forest and log to MLflow."""
        params = self.config['model']['isolation_forest']
        
        with mlflow.start_run(run_name='isolation_forest'):
            # Log hyperparameters
            mlflow.log_params(params)
            
            start = time.time()
            model = IsolationForest(**params)
            model.fit(X)
            train_time = time.time() - start
            
            # Get predictions
            preds  = model.predict(X)
            scores = model.decision_function(X)
            
            # Calculate metrics
            fraud_rate = (preds == -1).mean()
            score_mean = scores.mean()
            score_std  = scores.std()
            
            # Log metrics
            mlflow.log_metrics({
                'fraud_rate':   round(float(fraud_rate), 4),
                'score_mean':   round(float(score_mean), 4),
                'score_std':    round(float(score_std),  4),
                'training_time_sec': round(train_time, 2),
                'n_fraud_flagged': int((preds == -1).sum())
            })
            
            # Save model artifact to MLflow
            mlflow.sklearn.log_model(model, 'model')
            run_id = mlflow.active_run().info.run_id
        
        self.models['isolation_forest'] = model
        result = {
            'model_name':    'IsolationForest',
            'fraud_rate':    fraud_rate,
            'training_time': train_time,
            'run_id':        run_id,
            'n_flagged':     int((preds == -1).sum())
        }
        self.results['isolation_forest'] = result
        logger.info(f'IsolationForest: {fraud_rate*100:.1f}% fraud rate, '
                   f'{train_time:.1f}s training')
        return result
    
    def train_lof(self, X: pd.DataFrame) -> dict:
        """Train Local Outlier Factor and log to MLflow."""
        params = self.config['model']['lof']
        
        with mlflow.start_run(run_name='local_outlier_factor'):
            mlflow.log_params(params)
            start = time.time()
            # LOF with novelty=True allows predict() on new data
            model = LocalOutlierFactor(**params, novelty=True)
            model.fit(X)
            train_time = time.time() - start
            preds      = model.predict(X)
            scores     = model.decision_function(X)
            fraud_rate = (preds == -1).mean()
            mlflow.log_metrics({
                'fraud_rate':      round(float(fraud_rate), 4),
                'training_time_sec': round(train_time, 2),
                'n_fraud_flagged': int((preds == -1).sum())
            })
            mlflow.sklearn.log_model(model, 'model')
            run_id = mlflow.active_run().info.run_id
        
        self.models['lof'] = model
        result = {
            'model_name':    'LocalOutlierFactor',
            'fraud_rate':    fraud_rate,
            'training_time': train_time,
            'run_id':        run_id,
            'n_flagged':     int((preds == -1).sum())
        }
        self.results['lof'] = result
        logger.info(f'LOF: {fraud_rate*100:.1f}% fraud rate')
        return result
    
    def get_best_model(self) -> tuple:
        """
        Select best model: closest fraud_rate to target contamination.
        Returns (model_name, model).
        """
        target = self.config['model']['isolation_forest']['contamination']
        best   = min(self.results,
                     key=lambda k: abs(self.results[k]['fraud_rate'] - target))
        logger.info(f'Best model selected: {best}')
        return best, self.models[best]
    
    def save_best_model(self, save_dir: str = 'models/artifacts'):
        """Save the best model and its metadata."""
        os.makedirs(save_dir, exist_ok=True)
        name, model = self.get_best_model()
        
        path = f'{save_dir}/best_model.pkl'
        joblib.dump(model, path)
        
        # Save metadata
        import json
        meta = {
            'model_name': name,
            'result':     {k: str(v) for k, v in self.results[name].items()}
        }
        with open(f'{save_dir}/model_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f'Best model ({name}) saved to {path}')
        return path