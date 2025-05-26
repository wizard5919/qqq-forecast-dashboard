import streamlit as st
st.set_page_config(page_title="QQQ Forecast Simulator", layout="wide")

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import plotly.graph_objects as go

@st.cache_resource
def load_model_and_data():
    qqq = yf.download("QQQ", start="2018-01-01")
    vix = yf.download("^VIX", start="2018-01-01")['Close']
    treasury10 = yf.download("^TNX", start="2018-01-01")['Close']

    qqq = qqq.dropna().copy()
    vix = vix.reindex(qqq.index, method='ffill')
    treasury10 = treasury10.reindex(qqq.index, method='ffill')

    qqq['Date_Ordinal'] = qqq.index.map(datetime.toordinal)
    qqq['FedFunds'] = 5.25
    qqq['Unemployment'] = 3.9
    qqq['CPI'] = 3.5
    qqq['GDP'] = 21000
    qqq['VIX'] = vix
    qqq['10Y_Yield'] = treasury10
    qqq['Sentiment'] = 70  # placeholder constant

    features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX', '10Y_Yield', 'Sentiment']
    X = qqq[features]
    y = qqq['Close']

    model_xgb = xgb.XGBRegressor(n_estimators=100)
    model_lr = LinearRegression()
    model_ensemble = VotingRegressor(estimators=[('xgb', model_xgb), ('lr', model_lr)])
    model_ensemble.fit(X, y)

    return model_ensemble, features, qqq

model, features, qqq_data = load_model_and_data()

st.title("📈 QQQ Forecast Simulator")

with st.expander("Adjust Macro Variables"):
    fed_rate = st.slider("Fed Funds Rate (%)", 0.0, 10.0, 5.25)
    cpi = st.slider("CPI (%)", 1.0, 10.0, 3.5)
    unemployment = st.slider("Unemployment (%)", 2.0, 10.0, 3.9)
    gdp = st.slider("GDP ($B)", 10000, 30000, 21000)
    vix_val = st.slider("VIX Index", 10.0, 80.0, 20.0)
    treasury_yield = st.slider("10-Year Treasury Yield (%)", 0.5, 6.0, 4.0)
    sentiment = st.slider("Consumer Sentiment (1-100)", 20, 100, 70)

future_dates = pd.date_range(start=datetime.today(), periods=30)
date_ordinals = future_dates.map(datetime.toordinal)

future_df = pd.DataFrame({
    'Date_Ordinal': date_ordinals,
    'FedFunds': fed_rate,
    'Unemployment': unemployment,
    'CPI': cpi,
    'GDP': gdp,
    'VIX': vix_val,
    '10Y_Yield': treasury_yield,
    'Sentiment': sentiment
})

forecast = model.predict(future_df)

st.metric("Forecasted 1-Month QQQ", f"${forecast[-1]:.2f}", delta=f"{(forecast[-1] - forecast[0]):.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Forecast'))
fig.update_layout(title="QQQ Forecast (Next 30 Days)", xaxis_title="Date", yaxis_title="Price", template="plotly_white")

st.plotly_chart(fig, use_container_width=True)

if st.checkbox("Show Feature Importance (XGBoost)"):
    import matplotlib.pyplot as plt
    booster = model.named_estimators_['xgb']
    fig_imp, ax = plt.subplots()
    xgb.plot_importance(booster, ax=ax)
    st.pyplot(fig_imp)
