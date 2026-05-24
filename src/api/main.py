# src/api/main.py
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.schemas import (TransactionInput, PredictionOutput,
                              BatchInput, BatchOutput, HealthResponse,
                              RiskLevel)
from src.api.predictor import FraudPredictor
from src.utils.logger import get_logger
import uuid, time, os

logger = get_logger(__name__)

# ── APP INITIALIZATION ────────────────────────────────────
app = FastAPI(
    title='FraudGuard Pro API',
    description='''
    ## Supply Chain Fraud Detection API
    
    Detects fraudulent supply chain transactions using an
    Isolation Forest ML model trained on 180,519 real orders.
    
    ### Features
    - Single transaction analysis
    - Batch CSV analysis (up to 1,000 transactions)
    - Risk scoring (Low / Medium / High)
    - SHAP-based explanations
    - MLflow experiment tracking
    ''',
    version='1.0.0',
    docs_url='/docs',
    redoc_url='/redoc'
)

# CORS — allows frontend apps to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

# ── LOAD PREDICTOR ON STARTUP ──────────────────────────────
predictor = None
start_time = time.time()

@app.on_event('startup')
async def startup():
    global predictor
    logger.info('API starting up...')
    predictor = FraudPredictor()
    logger.info('Predictor loaded. API ready.')

# ── ENDPOINTS ─────────────────────────────────────────────
@app.get('/health', response_model=HealthResponse,
         summary='Health check')
async def health():
    """Returns API health status. Used by load balancers."""
    return HealthResponse(
        status='ok',
        model='IsolationForest',
        version='1.0.0',
        uptime_sec=round(time.time() - start_time, 1)
    )

@app.post('/predict', response_model=PredictionOutput,
          summary='Analyze single transaction')
async def predict(transaction: TransactionInput):
    """
    Analyze one transaction and return fraud verdict.
    Returns: label (FRAUD/Normal), fraud_score, risk_level, explanation.
    """
    if predictor is None:
        raise HTTPException(503, 'Model not loaded')
    
    try:
        result = predictor.predict_one(transaction.model_dump())
        return PredictionOutput(
            transaction_id   = str(uuid.uuid4())[:8],
            label            = result['label'],
            fraud_score      = result['fraud_score'],
            risk_level       = result['risk_level'],
            confidence_pct   = result['confidence_pct'],
            explanation      = result['explanation'],
            top_risk_factors = result['top_risk_factors']
        )
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        raise HTTPException(500, f'Prediction failed: {str(e)}')

@app.post('/predict/batch', response_model=BatchOutput,
          summary='Analyze multiple transactions')
async def predict_batch(batch: BatchInput):
    """Analyze up to 1,000 transactions at once."""
    if len(batch.transactions) > 1000:
        raise HTTPException(400, 'Max 1000 transactions per batch')
    
    results = []
    for t in batch.transactions:
        r = predictor.predict_one(t.model_dump())
        results.append(PredictionOutput(
            transaction_id   = str(uuid.uuid4())[:8],
            label            = r['label'],
            fraud_score      = r['fraud_score'],
            risk_level       = r['risk_level'],
            confidence_pct   = r['confidence_pct'],
            explanation      = r['explanation'],
            top_risk_factors = r['top_risk_factors']
        ))
    
    fraud_count = sum(1 for r in results if r.label == 'FRAUD')
    return BatchOutput(
        total        = len(results),
        fraud_count  = fraud_count,
        normal_count = len(results) - fraud_count,
        fraud_rate   = round(fraud_count / len(results) * 100, 2),
        predictions  = results
    )

# Run: uvicorn src.api.main:app --reload --port 8000