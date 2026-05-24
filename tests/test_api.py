# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    """Fixture that initializes the client using a context manager.
    This guarantees that the FastAPI @app.on_event('startup') code triggers!
    """
    with TestClient(app) as c:
        yield c

NORMAL_TXN = {
    'order_item_quantity': 5.0,
    'sales': 199.95,
    'order_item_discount': 10.0,
    'order_item_profit_ratio': 0.25,
    'order_item_total': 189.95,
    'late_delivery_risk': 0,
    'order_item_discount_rate': 0.05
}

FRAUD_TXN = {
    'order_item_quantity': 999.0,
    'sales': 99999.0,
    'order_item_discount': 9999.0,
    'order_item_profit_ratio': -5.0,
    'order_item_total': 999999.0,
    'late_delivery_risk': 1,
    'order_item_discount_rate': 0.99
}

def test_health_endpoint_returns_ok(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_predict_normal_transaction(client):
    r = client.post('/predict', json=NORMAL_TXN)
    assert r.status_code == 200
    data = r.json()
    assert 'label'       in data
    assert 'fraud_score' in data
    assert 'risk_level'  in data
    assert data['label'] in ['FRAUD', 'Normal']

def test_predict_fraud_transaction(client):
    r = client.post('/predict', json=FRAUD_TXN)
    assert r.status_code == 200
    data = r.json()
    assert data['label'] == 'FRAUD'

def test_invalid_discount_rate_rejected(client):
    bad = {**NORMAL_TXN, 'order_item_discount_rate': 5.0}
    r = client.post('/predict', json=bad)
    assert r.status_code == 422

def test_batch_predict(client):
    r = client.post('/predict/batch',
                    json={'transactions': [NORMAL_TXN, FRAUD_TXN]})
    assert r.status_code == 200
    data = r.json()
    assert data['total'] == 2
    assert data['fraud_count'] >= 1

def test_batch_limit_enforced(client):
    big_batch = {'transactions': [NORMAL_TXN] * 1001}
    r = client.post('/predict/batch', json=big_batch)
    assert r.status_code in [400, 422]
