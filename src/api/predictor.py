# src/api/predictor.py
import pandas as pd
import numpy as np
import joblib
from src.utils.logger import get_logger
from src.features.engineer import FraudFeatureEngineer, RAW_FEATURES
from src.api.schemas import RiskLevel

logger = get_logger(__name__)

COL_MAP = {
    'order_item_quantity':     'Order Item Quantity',
    'sales':                   'Sales',
    'order_item_discount':     'Order Item Discount',
    'order_item_profit_ratio': 'Order Item Profit Ratio',
    'order_item_total':        'Order Item Total',
    'late_delivery_risk':      'Late_delivery_risk',
    'order_item_discount_rate':'Order Item Discount Rate'
}

class FraudPredictor:
    def __init__(self):
        self.model    = joblib.load('models/artifacts/best_model.pkl')
        self.pipeline = joblib.load('models/artifacts/pipeline.pkl')
        self.engineer = FraudFeatureEngineer()
        logger.info('FraudPredictor initialized')
    
    def predict_one(self, raw_input: dict) -> dict:
        # Rename API keys to dataset column names
        renamed = {COLMAP[k]: v for k, v in raw_input.items() if k in COL_MAP}
        df = pd.DataFrame([renamed])
        
        # Apply full pipeline
        X = self.pipeline.transform(df)
        engineer = self.pipeline.named_steps['engineer']
        feat_names = engineer.all_features
        df_feat = pd.DataFrame(X, columns=feat_names)
        
        pred  = self.model.predict(df_feat)[0]
        score = float(self.model.decision_function(df_feat)[0])
        label = 'FRAUD' if pred == -1 else 'Normal'
        
        # Risk level from score
        if pred == 1:
            risk = RiskLevel.LOW
        elif score > -0.1:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.HIGH
        
        confidence = min(100, max(0, (0.5 - score) * 100))
        explanation = self._explain(raw_input, label, score)
        
        return {
            'label':           label,
            'fraud_score':     round(score, 4),
            'risk_level':      risk,
            'confidence_pct':  round(confidence, 1),
            'explanation':     explanation,
            'top_risk_factors': []
        }
    
    def _explain(self, t: dict, label: str, score: float) -> str:
        if label == 'Normal':
            return f'Transaction is within normal parameters (score: {score:.3f}).'
        flags = []
        if t.get('order_item_quantity', 0) > 100: flags.append('abnormal quantity')
        if t.get('order_item_profit_ratio', 0) < 0: flags.append('negative profit')
        if t.get('order_item_discount_rate', 0) > 0.5: flags.append('extreme discount')
        if t.get('late_delivery_risk', 0): flags.append('delivery risk flag')
        return f'FRAUD detected (score: {score:.3f}). Signals: {", ".join(flags) or "statistical anomaly"}.'

COLMAP = COL_MAP  # alias for predictor