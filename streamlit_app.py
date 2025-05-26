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
import json
import io
import matplotlib.pyplot as plt

@st.cache_resource
def load_model_and_data():
    qqq = yf.download("QQQ", start="2018-01-01")
    vix = yf.download("^VIX", start="2018-01-01")["Close"].squeeze()
    treasury10 = yf.download("^TNX", start="2018-01-01")["Close"].squeeze()
    treasury2 = yf.download("^IRX", start="2018-01-01")["Close"].squeeze()

    qqq = qqq.dropna().copy()
    vix = vix.reindex(qqq.index, method='ffill')
    treasury10 = treasury10.reindex(qqq.index, method='ffill')
    treasury2 = treasury2.reindex(qqq.index, method='ffill')

    qqq['Date_Ordinal'] = qqq.index.map(datetime.toordinal)
    qqq['FedFunds'] = 5.25
    qqq['Unemployment'] = 3.9
    qqq['CPI'] = 3.5
    qqq['GDP'] = 21000
    qqq['VIX'] = vix
    qqq['10Y_Yield'] = treasury10
    qqq['2Y_Yield'] = treasury2
    qqq['Yield_Spread'] = (treasury10 - treasury2).reindex(qqq.index, method='ffill')
    qqq['EPS_Growth'] = np.linspace(5, 15, len(qqq))
    qqq['Sentiment'] = 70

    qqq['MA_20'] = qqq['Close'].rolling(window=20).mean()
    qqq['MA_50'] = qqq['Close'].rolling(window=50).mean()
    qqq['Volatility'] = qqq['Close'].rolling(window=20).std()
    qqq = qqq.dropna()

    features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX',
                '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'EPS_Growth', 'Sentiment']
    X = qqq[features].copy()
    
    y = qqq['Close']

    model_xgb = xgb.XGBRegressor(n_estimators=100)
    model_lr = LinearRegression()
    model_xgb.fit(X, y)
    model_lr.fit(X[['Date_Ordinal']], y)
    model_ensemble = VotingRegressor(estimators=[('xgb', model_xgb), ('lr', model_lr)])
    model_ensemble.fit(X, y)

    return model_xgb, features, qqq, qqq['Close'].iloc[-1]

model, features, qqq_data, latest_close = load_model_and_data()

st.title("📈 QQQ Forecast Simulator")

use_live_price = st.checkbox("📡 Use Live QQQ Close ($%.2f) as Starting Point" % latest_close, value=True)

horizon = st.selectbox("📆 Forecast Horizon (days)", [30, 60, 90], index=0)
show_tech = st.checkbox("📊 Show Technical Indicators (MA 20/50, Volatility)")
macro_bias = st.slider("🧠 Macro News Sentiment Overlay (-10% to +10%)", -0.10, 0.10, 0.0, step=0.01)

scenarios = {
    "Recession": dict(fed=5.5, cpi=6.5, unemp=6.0, gdp=19000, vix=45.0, yield_10=2.5, yield_2=4.0, eps=5.0, sent=35),
    "Rate Cut": dict(fed=3.0, cpi=2.0, unemp=3.8, gdp=25000, vix=15.0, yield_10=3.0, yield_2=2.5, eps=12.0, sent=85),
    "Soft Landing": dict(fed=4.0, cpi=2.8, unemp=4.2, gdp=24000, vix=18.0, yield_10=3.8, yield_2=3.5, eps=10.0, sent=70)
}

scenario = st.radio("Select Scenario", ["Custom"] + list(scenarios.keys()))

if scenario == "Custom":
    fed = st.number_input("Fed Funds Rate", value=5.25)
    cpi = st.number_input("CPI", value=3.5)
    unemp = st.number_input("Unemployment", value=3.9)
    gdp = st.number_input("GDP", value=21000)
    vix = st.number_input("VIX", value=20.0)
    yield_10 = st.number_input("10Y Yield", value=4.0)
    yield_2 = st.number_input("2Y Yield", value=3.5)
    eps = st.number_input("EPS Growth", value=10.0)
    sent = st.slider("Sentiment", 0, 100, 70)
else:
    s = scenarios[scenario]
    fed, cpi, unemp, gdp = s['fed'], s['cpi'], s['unemp'], s['gdp']
    vix, yield_10, yield_2, eps, sent = s['vix'], s['yield_10'], s['yield_2'], s['eps'], s['sent']

future_dates = pd.date_range(start=datetime.today(), periods=horizon)
date_ordinals = future_dates.map(datetime.toordinal)

future_df = pd.DataFrame({
    'Date_Ordinal': date_ordinals,
    'FedFunds': fed,
    'Unemployment': unemp,
    'CPI': cpi,
    'GDP': gdp,
    'VIX': vix,
    '10Y_Yield': yield_10,
    '2Y_Yield': yield_2,
    'Yield_Spread': yield_10 - yield_2,
    'EPS_Growth': eps,
    'Sentiment': sent
})

forecast = model.predict(future_df)
forecast *= (1 + macro_bias)

if use_live_price:
    shift = latest_close - forecast[0]
    forecast += shift

confidence_std = 0.03 * forecast
forecast_upper = forecast + confidence_std
forecast_lower = forecast - confidence_std

fig = go.Figure()
fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Close'], name="Historical QQQ"))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_upper, name="Upper Bound", line=dict(color='lightblue'), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_lower, name="Lower Bound", fill='tonexty', line=dict(color='lightblue'), fillcolor='rgba(173, 216, 230, 0.3)', showlegend=False))

if show_tech:
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['MA_20'], name="MA 20", line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['MA_50'], name="MA 50", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Volatility'], name="Volatility", yaxis="y2", line=dict(color='orange')))
    fig.update_layout(yaxis2=dict(title="Volatility", overlaying='y', side='right'))

st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 CPI Sensitivity Analysis")
cpi_range = np.linspace(cpi - 1, cpi + 1, 5)
sens_fig = go.Figure()

for val in cpi_range:
    temp_df = future_df.copy()
    temp_df['CPI'] = val
    temp_pred = model.predict(temp_df)
    if use_live_price:
        temp_pred += (latest_close - temp_pred[0])
    temp_pred *= (1 + macro_bias)
    sens_fig.add_trace(go.Scatter(x=future_dates, y=temp_pred, name=f"CPI={val:.1f}"))

sens_fig.update_layout(title="QQQ Forecast Sensitivity to CPI", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(sens_fig, use_container_width=True)

st.subheader("📊 Multi-Scenario Forecast Comparison")
compare = st.checkbox("Compare Scenarios")
if compare:
    comp_fig = go.Figure()
    for name, s in scenarios.items():
        comp_df = pd.DataFrame({
            'Date_Ordinal': date_ordinals,
            'FedFunds': s['fed'],
            'Unemployment': s['unemp'],
            'CPI': s['cpi'],
            'GDP': s['gdp'],
            'VIX': s['vix'],
            '10Y_Yield': s['yield_10'],
            '2Y_Yield': s['yield_2'],
            'Yield_Spread': s['yield_10'] - s['yield_2'],
            'EPS_Growth': s['eps'],
            'Sentiment': s['sent']
        })
        yhat = model.predict(comp_df)
        if use_live_price:
            yhat += (latest_close - yhat[0])
        comp_fig.add_trace(go.Scatter(x=future_dates, y=yhat, name=name))
    comp_fig.update_layout(title="Scenario Forecast Comparison", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(comp_fig, use_container_width=True)

st.subheader("🔁 Backtest Model Accuracy")
backtest_mode = st.checkbox("Enable Backtest")
if backtest_mode:
    backtest_date = st.date_input("Start Backtest From", value=datetime(2023, 1, 1))
    back_df = qqq_data.loc[qqq_data.index >= pd.to_datetime(backtest_date)].copy()
    if not back_df.empty:
        back_df['Prediction'] = model.predict(back_df[features])
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=back_df.index, y=back_df['Close'], name="Actual"))
        fig_bt.add_trace(go.Scatter(x=back_df.index, y=back_df['Prediction'], name="Predicted"))
        fig_bt.update_layout(title="Backtest: QQQ Actual vs Predicted", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_bt, use_container_width=True)

st.subheader("📌 Feature Importance (XGBoost)")
if st.checkbox("Show Feature Importance"):
    booster = model.named_estimators_['xgb']
    fig_imp, ax = plt.subplots()
    xgb.plot_importance(booster, ax=ax)
    st.pyplot(fig_imp)

st.subheader("📥 Download Forecast Data")
forecast_df = future_df.copy()
forecast_df['Forecast'] = forecast
forecast_df['Date'] = future_dates
forecast_df.set_index('Date', inplace=True)
st.download_button("Download Forecast CSV", forecast_df.to_csv().encode(), file_name="qqq_forecast.csv")

if show_tech:
    tech_df = qqq_data[['Close', 'MA_20', 'MA_50', 'Volatility']].dropna()
    st.download_button("Download Technical Indicators CSV", tech_df.to_csv().encode(), file_name="technical_indicators.csv")

buf = io.BytesIO()
fig.write_image(buf, format="png")
st.download_button("Download Forecast Chart as PNG", buf.getvalue(), file_name="forecast_chart.png")
