import streamlit as st
st.set_page_config(page_title="📈 QQQ Forecast Simulator", layout="wide")

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

@st.cache_resource
def load_data_and_models():
    qqq = yf.download("QQQ", start="2018-01-01")
    qqq = qqq.dropna()
    qqq['Date'] = qqq.index
    qqq['Date_Ordinal'] = qqq['Date'].map(datetime.toordinal)
    qqq['FedFunds'] = 5.25
    qqq['Unemployment'] = 3.9
    qqq['CPI'] = 3.5
    qqq['GDP'] = 21000

    features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP']
    X = qqq[features].copy()
    y = qqq['Close'].copy()

    xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_model.fit(X, y)

    linear_model = LinearRegression()
    linear_model.fit(X[['Date_Ordinal']], y)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X[['Date_Ordinal']])
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    return qqq, xgb_model, linear_model, poly_model, poly

qqq, xgb_model, linear_model, poly_model, poly = load_data_and_models()

st.sidebar.header("Macro Inputs")
fed_rate = st.sidebar.slider("Fed Funds Rate (%)", 0.0, 7.0, 5.25, 0.25)
cpi = st.sidebar.slider("CPI (%)", 1.0, 10.0, 3.5, 0.1)
unemp = st.sidebar.slider("Unemployment (%)", 2.0, 10.0, 3.9, 0.1)
model_choice = st.sidebar.radio("Model Type", ["XGBoost", "Linear Regression", "Polynomial Regression"])

future_dates = pd.date_range(start=qqq.index.max() + pd.Timedelta(days=1), end="2025-12-31")
future_df = pd.DataFrame({
    'Date': future_dates,
    'Date_Ordinal': future_dates.map(datetime.toordinal),
    'FedFunds': fed_rate,
    'Unemployment': unemp,
    'CPI': cpi,
    'GDP': 21000
})

if model_choice == "XGBoost":
    forecast = xgb_model.predict(future_df[['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP']])
elif model_choice == "Linear Regression":
    forecast = linear_model.predict(future_df[['Date_Ordinal']])
else:
    future_poly = poly.transform(future_df[['Date_Ordinal']])
    forecast = poly_model.predict(future_poly)

fig = go.Figure()
fig.add_trace(go.Scatter(x=qqq['Date'], y=qqq['Close'], name="Historical QQQ", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_df['Date'], y=forecast, name=f"Forecast ({model_choice})", line=dict(color='orange')))
fig.add_trace(go.Scatter(x=[qqq.index.min(), future_df['Date'].max()], y=[500, 500], name='Breakout $500', line=dict(color='red', dash='dot')))
above_500 = forecast > 500
fig.add_trace(go.Scatter(x=future_df['Date'][above_500], y=forecast[above_500], mode='markers', name='> $500', marker=dict(color='green')))
fig.update_layout(title="QQQ Forecast with Macro Variables", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

y_true = qqq['Close']
y_pred = xgb_model.predict(qqq[['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP']])
col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{r2_score(y_true, y_pred):.4f}")
col2.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.2f}")
col3.metric("RMSE", f"{mean_squared_error(y_true, y_pred, squared=False):.2f}")

forecast_df = future_df.copy()
forecast_df['Forecast'] = forecast
st.download_button("Download Forecast as CSV", forecast_df.to_csv(index=False), file_name="qqq_forecast.csv")