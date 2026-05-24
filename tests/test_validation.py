# tests/test_validation.py
import pytest
import pandas as pd
from src.data.validation import DataValidator, ValidationResult

@pytest.fixture
def validator():
    return DataValidator()

@pytest.fixture
def valid_df():
    """Minimal valid DataFrame for testing (replicated to pass 1000 row check)."""
    base_df = pd.DataFrame({
        'Order Item Quantity':     [5, 10, 3],
        'Sales':                   [199.95, 500.0, 75.0],
        'Order Item Discount':     [10.0, 25.0, 5.0],
        'Order Item Profit Ratio': [0.25, 0.3, 0.15],
        'Order Item Total':        [189.95, 475.0, 70.0],
        'Late_delivery_risk':      [0, 0, 1],
        'Order Item Discount Rate':[0.05, 0.05, 0.07]
    })
    return pd.concat([base_df] * 400, ignore_index=True)

def test_valid_dataframe_passes(validator, valid_df):
    result = validator.validate(valid_df)
    assert result.passed is True
    assert len(result.errors) == 0

def test_missing_columns_fails(validator):
    # Replicated to bypass row check, keeping only 'Sales' to trigger missing column check
    base_df = pd.DataFrame({'Sales': [100, 200]})
    df = pd.concat([base_df] * 600, ignore_index=True)
    result = validator.validate(df)
    assert result.passed is False
    assert any('Missing required columns' in e for e in result.errors)

def test_too_few_rows_fails(validator):
    df = pd.DataFrame({'Sales': [1]})
    result = validator.validate(df)
    assert result.passed is False

def test_negative_sales_is_warning(validator, valid_df):
    valid_df.loc[0, 'Sales'] = -100  # introduce negative sale
    result = validator.validate(valid_df)
    assert any('negative' in w.lower() for w in result.warnings)

def test_validation_result_has_stats(validator, valid_df):
    result = validator.validate(valid_df)
    assert 'total_checks' in result.stats or 'total_rows' in result.stats
