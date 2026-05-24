# src/dashboard/Home.py
import streamlit as st
import requests
import os

st.set_page_config(
    page_title='FraudGuard Pro',
    page_icon='🛡️',
    layout='wide'
)

st.title('🛡️ FraudGuard Pro')
st.subheader('Industry-Grade Supply Chain Fraud Detection System')
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric('Transactions Trained On', '180,519')
col2.metric('Fraud Detection Rate',   '5.0%')
col3.metric('Features Engineered',    '15')
col4.metric('Models Compared',        '2')

st.divider()
st.subheader('System Architecture')
st.markdown('''
| Layer | Technology | Purpose |
|-------|-----------|--------|
| **Ingestion** | Python · Pandas | Validated data loading |
| **Feature Engineering** | Scikit-learn Pipeline | 15 features (7 raw + 8 engineered) |
| **Model Training** | Isolation Forest + LOF | Multi-model comparison |
| **Experiment Tracking** | MLflow | Hyperparameter + metric logging |
| **Explainability** | SHAP | Feature importance per transaction |
| **API** | FastAPI | REST endpoints with OpenAPI docs |
| **Dashboard** | Streamlit | This interface |
| **Testing** | Pytest | Unit + integration tests |
| **CI/CD** | GitHub Actions | Auto-test on every push |
| **Container** | Docker | Reproducible deployment |
''')

# API health check — works both locally and inside Docker
API_URL = os.getenv('API_URL', 'http://localhost:8000')

try:
    r = requests.get(f'{API_URL}/health', timeout=2)
    if r.status_code == 200:
        data = r.json()
        st.success(f'✅ API is online and ready — Model: {data.get("model", "IsolationForest")} | Version: {data.get("version", "1.0.0")} | Uptime: {data.get("uptime_sec", 0):.0f}s')
    else:
        st.warning('⚠️ API responded with non-200 status')
except:
    st.error('❌ API offline — run: docker-compose up')