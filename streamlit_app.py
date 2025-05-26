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
import requests
import json
import io
import matplotlib.pyplot as plt

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
    qqq['Sentiment'] = 70

    qqq['MA_20'] = qqq['Close'].rolling(window=20).mean()
    qqq['MA_50'] = qqq['Close'].rolling(window=50).mean()
    qqq['Volatility'] = qqq['Close'].rolling(window=20).std()
    qqq = qqq.dropna()

    features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX', '10Y_Yield', 'Sentiment']
    X = qqq[features].copy()
    if isinstance(X.columns, pd.MultiIndex):
        X.columns = [" ".join(col).strip() for col in X.columns.values]
    else:
        X.columns = X.columns.str.strip()
    y = qqq['Close']

    model_xgb = xgb.XGBRegressor(n_estimators=100)
    model_lr = LinearRegression()
    model_ensemble = VotingRegressor(estimators=[('xgb', model_xgb), ('lr', model_lr)])
    model_ensemble.fit(X, y)

    return model_ensemble, features, qqq, qqq['Close'].iloc[-1]

model, features, qqq_data, latest_close = load_model_and_data()

st.title("📈 QQQ Forecast Simulator")

use_live_price = st.checkbox("📡 Use Live QQQ Close ($%.2f) as Starting Point" % latest_close, value=True)

horizon = st.selectbox("📆 Forecast Horizon (days)", [30, 60, 90], index=0)
show_tech = st.checkbox("📊 Show Technical Indicators (MA 20/50, Volatility)")

scenarios = {
    "Recession": dict(fed=5.5, cpi=6.5, unemp=6.0, gdp=19000, vix=45.0, yield_=2.5, sent=35),
    "Rate Cut": dict(fed=3.0, cpi=2.0, unemp=3.8, gdp=25000, vix=15.0, yield_=3.0, sent=85),
    "Soft Landing": dict(fed=4.0, cpi=2.8, unemp=4.2, gdp=24000, vix=18.0, yield_=3.8, sent=70)
}

scenario = st.radio("Select a Scenario", ["Custom"] + list(scenarios.keys()))

if scenario == "Custom":
    with st.expander("Adjust Macro Variables"):
        fed_rate = st.slider("Fed Funds Rate (%)", 0.0, 10.0, 5.25)
        cpi = st.slider("CPI (%)", 1.0, 10.0, 3.5)
        unemployment = st.slider("Unemployment (%)", 2.0, 10.0, 3.9)
        gdp = st.slider("GDP ($B)", 10000, 30000, 21000)
        vix_val = st.slider("VIX Index", 10.0, 80.0, 20.0)
        treasury_yield = st.slider("10-Year Treasury Yield (%)", 0.5, 6.0, 4.0)
        sentiment = st.slider("Consumer Sentiment", 20, 100, 70)
        custom_scenario = dict(fed=fed_rate, cpi=cpi, unemp=unemployment, gdp=gdp, vix=vix_val, yield_=treasury_yield, sent=sentiment)
        if st.button("Save Scenario"):
            st.download_button("Download Scenario", json.dumps(custom_scenario), file_name="scenario.json")
        uploaded = st.file_uploader("Load Scenario", type="json")
        if uploaded:
            try:
                loaded = json.load(uploaded)
                fed_rate, cpi, unemployment, gdp = loaded['fed'], loaded['cpi'], loaded['unemp'], loaded['gdp']
                vix_val, treasury_yield, sentiment = loaded['vix'], loaded['yield_'], loaded['sent']
                st.success("Scenario loaded.")
            except:
                st.error("Invalid scenario file.")
else:
    s = scenarios[scenario]
    fed_rate, cpi, unemployment, gdp, vix_val, treasury_yield, sentiment = s['fed'], s['cpi'], s['unemp'], s['gdp'], s['vix'], s['yield_'], s['sent']

compare = st.checkbox("Compare All Scenarios")
history_mode = st.checkbox("Backtest from Past Date")
technical_download = st.checkbox("📥 Download Technical Indicators")
threshold = st.number_input("🔔 Set Alert Threshold for QQQ", value=500.0, step=1.0)

future_dates = pd.date_range(start=datetime.today(), periods=horizon)
date_ordinals = future_dates.map(datetime.toordinal)

fig = go.Figure()
forecast_df = pd.DataFrame()

if compare:
    all_scenarios = scenarios.copy()
    all_scenarios["Custom"] = dict(fed=fed_rate, cpi=cpi, unemp=unemployment, gdp=gdp, vix=vix_val, yield_=treasury_yield, sent=sentiment)
    for name, s in all_scenarios.items():
        future_df = pd.DataFrame({
            'Date_Ordinal': date_ordinals,
            'FedFunds': s['fed'],
            'Unemployment': s['unemp'],
            'CPI': s['cpi'],
            'GDP': s['gdp'],
            'VIX': s['vix'],
            '10Y_Yield': s['yield_'],
            'Sentiment': s['sent']
        })
        forecast = model.predict(future_df)
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name=name))
        forecast_df[name] = forecast
else:
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
    st.metric("Forecasted QQQ", f"${forecast[-1]:.2f}", delta=f"{(forecast[-1] - forecast[0]):.2f}")
    if forecast[-1] >= threshold:
        st.warning(f"🚨 Alert: Forecasted QQQ (${forecast[-1]:.2f}) exceeds threshold of ${threshold:.2f}!")
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Forecast'))
    forecast_df['Forecast'] = forecast

fig.update_layout(title=f"QQQ Forecast (Next {horizon} Days)", xaxis_title="Date", yaxis_title="Price", template="plotly_white")

if show_tech:
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['MA_20'], name="MA 20", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['MA_50'], name="MA 50", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Volatility'], name="Volatility", yaxis="y2", line=dict(color='orange')))
    fig.update_layout(yaxis2=dict(title="Volatility", overlaying='y', side='right'))

st.plotly_chart(fig, use_container_width=True)

forecast_df['Date'] = future_dates
forecast_df = forecast_df.set_index('Date')
st.download_button("📥 Download Forecast CSV", forecast_df.to_csv().encode(), file_name="forecast.csv")

if technical_download:
    tech_df = qqq_data[['Close', 'MA_20', 'MA_50', 'Volatility']].dropna()
    st.download_button("📥 Download Technical Data CSV", tech_df.to_csv().encode(), file_name="technical_indicators.csv")

buf = io.BytesIO()
fig.write_image(buf, format="png")
st.download_button("📸 Download Chart as PNG", buf.getvalue(), file_name="forecast_chart.png")

if history_mode:
    backtest_date = st.date_input("Select Backtest Start Date", value=datetime(2023, 1, 1))
    back_q = qqq_data.loc[qqq_data.index >= pd.to_datetime(backtest_date)].copy()
    if not back_q.empty:
        back_q['Prediction'] = model.predict(back_q[features])
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=back_q.index, y=back_q['Close'], mode='lines', name='Actual'))
        fig_hist.add_trace(go.Scatter(x=back_q.index, y=back_q['Prediction'], mode='lines', name='Predicted'))
        fig_hist.update_layout(title="Backtest: QQQ Price vs Prediction", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_hist, use_container_width=True)

if st.checkbox("Show Feature Importance (XGBoost)"):
    booster = model.named_estimators_['xgb']
    fig_imp, ax = plt.subplots()
    xgb.plot_importance(booster, ax=ax)
    st.pyplot(fig_imp)
