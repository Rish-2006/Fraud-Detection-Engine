# pages/1_Analyze.py
import streamlit as st
import requests

st.title('🔍 Transaction Analyzer')
st.caption('Enter transaction details for instant fraud analysis')

col1, col2 = st.columns(2)
with col1:
    qty  = st.number_input('Order Quantity',     min_value=0, value=5)
    sale = st.number_input('Sales Amount (Rs)',  min_value=0.0, value=199.95)
    disc = st.number_input('Discount Amount',    min_value=0.0, value=10.0)
    prof = st.number_input('Profit Ratio',       value=0.25, step=0.01)
with col2:
    tot  = st.number_input('Order Total',        min_value=0.0, value=189.95)
    late = st.selectbox('Late Delivery Risk',    [0,1])
    rate = st.number_input('Discount Rate 0-1',  min_value=0.0, max_value=1.0, value=0.05, step=0.01)

if st.button('Analyze Transaction', type='primary', use_container_width=True):
    payload = {
        'order_item_quantity': qty, 'sales': sale,
        'order_item_discount': disc, 'order_item_profit_ratio': prof,
        'order_item_total': tot, 'late_delivery_risk': late,
        'order_item_discount_rate': rate
    }
    with st.spinner('Calling FraudGuard API...'):
        try:
            r = requests.post('http://localhost:8000/predict', json=payload, timeout=10)
            if r.status_code == 200:
                res = r.json()
                st.divider()
                if res['label'] == 'FRAUD':
                    st.error(f'🚨 FRAUD DETECTED — Risk: {res["risk_level"]}')
                else:
                    st.success(f'✅ NORMAL — Risk: {res["risk_level"]}')
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('Verdict',     res['label'])
                c2.metric('Fraud Score', f'{res["fraud_score"]:.4f}')
                c3.metric('Risk Level',  res['risk_level'])
                c4.metric('Confidence',  f'{res["confidence_pct"]}%')
                st.info(f'📋 {res["explanation"]}')
                st.caption(f'Transaction ID: {res["transaction_id"]} | Model v{res["model_version"]}')
            else:
                st.error(f'API Error: {r.json()}')
        except Exception as e:
            st.error(f'Could not reach API: {e}')