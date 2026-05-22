# src/api/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum

class RiskLevel(str, Enum):
    LOW    = 'Low'
    MEDIUM = 'Medium'
    HIGH   = 'High'

class TransactionInput(BaseModel):
    """Input schema — what the API expects."""
    order_item_quantity:     float = Field(..., ge=0, description='Order quantity (must be >= 0)')
    sales:                   float = Field(..., ge=0, description='Sales amount in Rs')
    order_item_discount:     float = Field(..., ge=0, description='Discount amount in Rs')
    order_item_profit_ratio: float = Field(..., description='Profit ratio (can be negative)')
    order_item_total:        float = Field(..., ge=0, description='Total order amount')
    late_delivery_risk:      int   = Field(..., ge=0, le=1, description='0 or 1')
    order_item_discount_rate:float = Field(..., ge=0, le=1, description='Discount rate 0-1')
    
    @field_validator('order_item_discount_rate')
    @classmethod
    def validate_discount_rate(cls, v):
        if v > 1.0:
            raise ValueError('Discount rate cannot exceed 1.0 (100%)')
        return v
    
    model_config = {
        'json_schema_extra': {
            'example': {
                'order_item_quantity': 5,
                'sales': 199.95,
                'order_item_discount': 10.0,
                'order_item_profit_ratio': 0.25,
                'order_item_total': 189.95,
                'late_delivery_risk': 0,
                'order_item_discount_rate': 0.05
            }
        }
    }

class PredictionOutput(BaseModel):
    """Output schema — what the API returns."""
    transaction_id:  str
    label:           str
    fraud_score:     float
    risk_level:      RiskLevel
    confidence_pct:  float
    explanation:     str
    top_risk_factors:List[dict]
    model_version:   str = '1.0.0'

class BatchInput(BaseModel):
    transactions: List[TransactionInput]
    
class BatchOutput(BaseModel):
    total:        int
    fraud_count:  int
    normal_count: int
    fraud_rate:   float
    predictions:  List[PredictionOutput]

class HealthResponse(BaseModel):
    status:    str = 'ok'
    model:     str
    version:   str
    uptime_sec:float