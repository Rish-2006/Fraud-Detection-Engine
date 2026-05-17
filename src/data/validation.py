# src/data/validation.py
import pandas as pd
from dataclasses import dataclass, field
from typing import List
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """Stores the outcome of all validation checks."""
    passed:   bool = True
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats:    dict = field(default_factory=dict)


class DataValidator:
    """
    Runs schema and quality checks on incoming data.
    Fails loudly if critical checks fail.
    """
    REQUIRED_COLUMNS = [
        'Order Item Quantity', 'Sales', 'Order Item Discount',
        'Order Item Profit Ratio', 'Order Item Total',
        'Late_delivery_risk', 'Order Item Discount Rate'
    ]
    MIN_ROWS = 1000  # reject tiny datasets
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult()
        
        # CHECK 1: Minimum row count
        if len(df) < self.MIN_ROWS:
            result.errors.append(
                f'Too few rows: {len(df)} < {self.MIN_ROWS} minimum'
            )
        
        # CHECK 2: Required columns present
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            result.errors.append(
                f'Missing required columns: {missing_cols}'
            )
        
        # CHECK 3: Missing value rate per column
        if not missing_cols:  # only if columns exist
            for col in self.REQUIRED_COLUMNS:
                miss_pct = df[col].isnull().mean() * 100
                if miss_pct > 30:
                    result.errors.append(
                        f'{col}: {miss_pct:.1f}% missing (>30% threshold)'
                    )
                elif miss_pct > 5:
                    result.warnings.append(
                        f'{col}: {miss_pct:.1f}% missing (watch this)'
                    )
        
        # CHECK 4: Negative sales (impossible in real data)
        if 'Sales' in df.columns:
            neg_sales = (df['Sales'] < 0).sum()
            if neg_sales > 0:
                result.warnings.append(
                    f'{neg_sales:,} rows have negative Sales values'
                )
        
        # CHECK 5: Discount rate > 1 (impossible — 100% = 1.0)
        if 'Order Item Discount Rate' in df.columns:
            invalid_rate = (df['Order Item Discount Rate'] > 1).sum()
            if invalid_rate > 0:
                result.warnings.append(
                    f'{invalid_rate:,} rows have Discount Rate > 1.0'
                )
        
        result.passed = len(result.errors) == 0
        result.stats = {
            'total_checks': 5,
            'errors':       len(result.errors),
            'warnings':     len(result.warnings)
        }
        
        # Log results
        if result.passed:
            logger.info('Data validation PASSED')
        else:
            logger.error(f'Data validation FAILED: {result.errors}')
        
        for w in result.warnings:
            logger.warning(w)
        
        return result