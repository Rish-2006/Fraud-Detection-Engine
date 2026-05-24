# tests/test_features.py
import pytest
import pandas as pd
from src.features.engineer import FraudFeatureEngineer, RAW_FEATURES

@pytest.fixture
def engineer():
    return FraudFeatureEngineer()

@pytest.fixture
def sample_df():
    return pd.DataFrame([{
        'Order Item Quantity': 5,
        'Sales': 200.0,
        'Order Item Discount': 10.0,
        'Order Item Profit Ratio': 0.25,
        'Order Item Total': 190.0,
        'Late_delivery_risk': 0,
        'Order Item Discount Rate': 0.05
    }])

def test_engineer_adds_8_features(engineer, sample_df):
    result = engineer.fit_transform(sample_df)
    assert result.shape[1] == 7 + 8  # 7 raw + 8 engineered

def test_discount_ratio_computed(engineer, sample_df):
    result = engineer.fit_transform(sample_df)
    expected = 10.0 / (200.0 + 1e-6)
    assert abs(result['discount_to_sales_ratio'].iloc[0] - expected) < 0.001

def test_large_order_flag(engineer):
    df = pd.DataFrame([{
        'Order Item Quantity': 999,
        'Sales': 9999.0, 'Order Item Discount': 999.0,
        'Order Item Profit Ratio': -2.0, 'Order Item Total': 9000.0,
        'Late_delivery_risk': 1, 'Order Item Discount Rate': 0.99
    }])
    result = engineer.fit_transform(df)
    assert result['is_large_order'].iloc[0] == 1

def test_normal_order_not_flagged(engineer, sample_df):
    result = engineer.fit_transform(sample_df)
    assert result['is_large_order'].iloc[0] == 0
    assert result['extreme_discount'].iloc[0] == 0