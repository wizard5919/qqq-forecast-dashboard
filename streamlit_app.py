import streamlit as st
st.set_page_config(page_title="QQQ Forecast Simulator", layout="wide")

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import xgboost as xgb
import plotly.graph_objects as go

@st.cache_resource
def load_model_and_data():
    # --- Data Loading ---
    qqq = yf.download("QQQ", start="2018-01-01")
    qqq = qqq.dropna().copy()

    # --- Critical Fix: Force flat column index ---
    if isinstance(qqq.columns, pd.MultiIndex):
        qqq.columns = pd.Index(['_'.join(col).strip() for col in qqq.columns.values])
    else:
        qqq.columns = pd.Index([str(col).strip() for col in qqq.columns])  # Force flat index

    # --- Column Validation ---
    if 'Close' not in qqq.columns:
        if 'Close_QQQ' in qqq.columns:
            qqq.rename(columns={'Close_QQQ': 'Close'}, inplace=True)
        else:
            raise ValueError("Missing 'Close' column. Available columns: " + str(qqq.columns.tolist()))

    # --- Feature Engineering ---
    qqq['Date_Ordinal'] = qqq.index.map(datetime.toordinal)
    qqq['FedFunds'] = 5.25  # Default training value
    qqq['Unemployment'] = 3.9
    qqq['CPI'] = 3.5
    qqq['GDP'] = 21000

    # --- Model Training ---
    features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP']
    X = qqq[features].copy()
    X.columns = pd.Index([str(col).strip() for col in X.columns])  # Explicit flat index
    y = qqq['Close']

    model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    return model, qqq

# --- UI Section ---
model, qqq = load_model_and_data()

st.title("📈 QQQ Forecast Simulator")
col1, col2, col3 = st.columns(3)
fed_rate = col1.slider("Fed Funds Rate (%)", 0.0, 7.0, 5.25, 0.25)
cpi = col2.slider("CPI (%)", 1.0, 10.0, 3.5, 0.1)
unemp = col3.slider("Unemployment (%)", 2.0, 10.0, 3.9, 0.1)

# --- Forecasting ---
future_dates = pd.date_range(start=qqq.index.max() + pd.Timedelta(days=1), end="2025-12-31")
future_df = pd.DataFrame({
    'Date_Ordinal': future_dates.map(datetime.toordinal),
    'FedFunds': fed_rate,
    'Unemployment': unemp,
    'CPI': cpi,
    'GDP': 21000
})

# Critical Fix: Future dataframe columns
future_df.columns = pd.Index([str(col).strip() for col in future_df.columns])
forecast = model.predict(future_df[['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP']])

# --- Rest of your visualization code remains unchanged ---
future_df.columns = future_df.columns.str.strip()
forecast = model.predict(future_df)

# 📊 Build the chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=qqq['Date'], y=qqq['Close'], name="Historical QQQ", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast (XGBoost)", line=dict(color='orange')))
fig.add_trace(go.Scatter(x=[qqq.index.min(), future_dates.max()], y=[500, 500], name='Breakout Level ($500)', line=dict(color='red', dash='dot')))

# Highlight breakout points
above_500 = forecast > 500
if above_500.any(): # Only add trace if there are points above 500
    fig.add_trace(go.Scatter(
        x=future_dates[above_500],
        y=forecast[above_500],
        mode='markers',
        name='Above $500',
        marker=dict(color='green', size=7),
        hovertemplate='Price: %{y:.2f}<br>Date: %{x|%Y-%m-%d}<extra>Breakout</extra>'
    ))

fig.update_layout(
    title="QQQ Forecast (2024–2025)",
    xaxis_title='Date',
    yaxis_title='QQQ Price',
    template='plotly_dark',
    height=600,
    hovermode='x unified',
    legend=dict(orientation="h", y=1.05, x=0.5, xanchor='center')
)

st.plotly_chart(fig, use_container_width=True)

# 🔢 Summary of breakout duration
days_above_500 = int((forecast > 500).sum())
st.info(f"📈 Forecasted days above $500: **{days_above_500}**")

# Footer disclaimer
st.markdown("---")
st.caption("📊 Forecast is for illustrative purposes only and does not constitute financial advice.")
