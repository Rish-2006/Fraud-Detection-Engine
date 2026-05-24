# src/features/engineer.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_FEATURES = [
    'Order Item Quantity', 'Sales', 'Order Item Discount',
    'Order Item Profit Ratio', 'Order Item Total',
    'Late_delivery_risk', 'Order Item Discount Rate'
]

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that creates fraud-signal features.
    BaseEstimator + TransformerMixin = compatible with sklearn Pipeline.
    """
    def fit(self, X, y=None):
        # No fitting needed for rule-based features
        # But we store column stats for later validation
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self  # always return self in fit()
    
    def transform(self, X, y=None) -> pd.DataFrame:
        """Create 8 new fraud-signal features."""
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        logger.info('Engineering fraud features...')
        
        # FEATURE 1: Discount-to-Sales ratio
        # Flags: discount > sales amount (impossible in honest business)
        df['discount_to_sales_ratio'] = (
            df['Order Item Discount'] /
            (df['Sales'] + 1e-6)  # +tiny to avoid division by zero
        ).clip(0, 10)  # cap at 10x to handle extreme outliers
        
        # FEATURE 2: Revenue efficiency
        # Low or negative = the order is losing money
        df['revenue_efficiency'] = (
            df['Order Item Profit Ratio'] *
            df['Sales']
        )
        
        # FEATURE 3: Abnormal quantity flag
        # Orders > 100 units are statistically unusual
        df['is_large_order'] = (
            df['Order Item Quantity'] > 100
        ).astype(int)
        
        # FEATURE 4: Total vs Sales mismatch
        # In honest data: total should roughly equal sales - discount
        # Large mismatch = inflated billing
        df['total_sales_mismatch'] = abs(
            df['Order Item Total'] -
            (df['Sales'] - df['Order Item Discount'])
        )
        
        # FEATURE 5: Combined risk score (rule-based)
        # Combines 3 risk signals into one composite score
        df['combined_risk'] = (
            (df['Order Item Profit Ratio'] < 0).astype(int) +
            (df['Order Item Discount Rate'] > 0.3).astype(int) +
            df['Late_delivery_risk']
        )
        
        # FEATURE 6: Extreme discount flag
        # Discount rate > 50% is extremely unusual in B2B
        df['extreme_discount'] = (
            df['Order Item Discount Rate'] > 0.5
        ).astype(int)
        
        # FEATURE 7: Loss magnitude
        # How much money is being lost per order
        df['loss_magnitude'] = (
            df['Order Item Profit Ratio'].clip(upper=0).abs() *
            df['Sales']
        )
        
        # FEATURE 8: Suspicion composite
        # Weighted combination of strongest fraud signals
        df['suspicion_score'] = (
            df['discount_to_sales_ratio'] * 0.3 +
            df['loss_magnitude'].clip(0, 100) / 100 * 0.3 +
            df['combined_risk'] / 3 * 0.4
        )
        
        n_new = 8
        logger.info(f'Created {n_new} new features. '
                   f'Total features: {df.shape[1]}')
        return df
    
    @property
    def engineered_features(self):
        return [
            'discount_to_sales_ratio', 'revenue_efficiency',
            'is_large_order', 'total_sales_mismatch',
            'combined_risk', 'extreme_discount',
            'loss_magnitude', 'suspicion_score'
        ]
    
    @property
    def all_features(self):
        return RAW_FEATURES + self.engineered_features