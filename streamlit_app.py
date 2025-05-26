import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="QQQ Forecast Simulator", layout="wide")

@st.cache_resource
def load_model_and_data():
    try:
        # Download data
        start_date = "2018-01-01"
        qqq = yf.download("QQQ", start=start_date, progress=False)
        vix = yf.download("^VIX", start=start_date, progress=False)["Close"].squeeze()
        treasury10 = yf.download("^TNX", start=start_date, progress=False)["Close"].squeeze()
        treasury2 = yf.download("^IRX", start=start_date, progress=False)["Close"].squeeze()

        # Ensure data alignment
        qqq = qqq.dropna().copy()
        vix = vix.reindex(qqq.index, method='ffill').fillna(method='ffill')
        treasury10 = treasury10.reindex(qqq.index, method='ffill').fillna(method='ffill')
        treasury2 = treasury2.reindex(qqq.index, method='ffill').fillna(method='ffill')

        # Add features
        qqq['Date_Ordinal'] = qqq.index.map(datetime.toordinal)
        qqq['FedFunds'] = 5.25  # Placeholder; ideally fetch real data
        qqq['Unemployment'] = 3.9
        qqq['CPI'] = 3.5
        qqq['GDP'] = 21000
        qqq['VIX'] = vix
        qqq['10Y_Yield'] = treasury10
        qqq['2Y_Yield'] = treasury2
        qqq['Yield_Spread'] = (treasury10 - treasury2)
        qqq['EPS_Growth'] = np.linspace(5, 15, len(qqq))
        qqq['Sentiment'] = 70

        # Technical indicators
        qqq['MA_20'] = qqq['Close'].rolling(window=20).mean()
        qqq['MA_50'] = qqq['Close'].rolling(window=50).mean()
        qqq['Volatility'] = qqq['Close'].rolling(window=20).std()
        qqq = qqq.dropna()

        # Features and target
        features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX',
                    '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'EPS_Growth', 'Sentiment']
        X = qqq[features].copy()
        y = qqq['Close']

        # Ensure feature names are strings
        X.columns = X.columns.astype(str).str.strip()

        # Train XGBoost model
        model_xgb = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model_xgb.fit(X, y)

        return model_xgb, features, qqq, qqq['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error loading data or training model: {e}")
        return None, None, None, None

# Load model and data
model, features, qqq_data, latest_close = load_model_and_data()

if model is None or qqq_data is None:
    st.stop()

st.title("📈 QQQ Forecast Simulator")

# User inputs
use_live_price = st.checkbox("📡 Use Live QQQ Close ($%.2f) as Starting Point" % latest_close, value=True)
horizon = st.selectbox("📆 Forecast Horizon (days)", [30, 60, 90], index=0)
show_tech = st.checkbox("📊 Show Technical Indicators (MA 20/50, Volatility)")
macro_bias = st.slider("🧠 Macro News Sentiment Overlay (-10% to +10%)", -0.10, 0.10, 0.0, step=0.01)

# Scenarios
scenarios = {
    "Recession": dict(fed=5.5, cpi=6.5, unemp=6.0, gdp=19000, vix=45.0, yield_10=2.5, yield_2=4.0, eps=5.0, sent=35),
    "Rate Cut": dict(fed=3.0, cpi=2.0, unemp=3.8, gdp=25000, vix=15.0, yield_10=3.0, yield_2=2.5, eps=12.0, sent=85),
    "Soft Landing": dict(fed=4.0, cpi=2.8, unemp=4.2, gdp=24000, vix=18.0, yield_10=3.8, yield_2=3.5, eps=10.0, sent=70)
}

scenario = st.radio("Select Scenario", ["Custom"] + list(scenarios.keys()))

# Scenario inputs
if scenario == "Custom":
    fed = st.number_input("Fed Funds Rate", value=5.25, step=0.1)
    cpi = st.number_input("CPI", value=3.5, step=0.1)
    unemp = st.number_input("Unemployment", value=3.9, step=0.1)
    gdp = st.number_input("GDP", value=21000, step=1000)
    vix = st.number_input("VIX", value=20.0, step=1.0)
    yield_10 = st.number_input("10Y Yield", value=4.0, step=0.1)
    yield_2 = st.number_input("2Y Yield", value=3.5, step=0.1)
    eps = st.number_input("EPS Growth", value=10.0, step=0.5)
    sent = st.slider("Sentiment", 0, 100, 70)
else:
    s = scenarios[scenario]
    fed, cpi, unemp, gdp = s['fed'], s['cpi'], s['unemp'], s['gdp']
    vix, yield_10, yield_2, eps, sent = s['vix'], s['yield_10'], s['yield_2'], s['eps'], s['sent']

# Create future dataframe
future_dates = pd.date_range(start=datetime.today(), periods=horizon, freq='D')
date_ordinals = [d.toordinal() for d in future_dates]

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
}, index=future_dates)

# Ensure feature order matches training data
future_df = future_df[features]

# Make predictions
forecast = model.predict(future_df)
forecast *= (1 + macro_bias)

# Adjust forecast to match live price if selected
if use_live_price:
    shift = latest_close - forecast[0]
    forecast += shift

# Confidence intervals
confidence_std = 0.03 * forecast
forecast_upper = forecast + confidence_std
forecast_lower = forecast - confidence_std

# Main forecast plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Close'], name="Historical QQQ", line=dict(color='black')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_upper, name="Upper Bound", line=dict(color='lightblue'), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_lower, name="Lower Bound", fill='tonexty', line=dict(color='lightblue'), fillcolor='rgba(173, 216, 230, 0.3)', showlegend=False))

# Add technical indicators
if show_tech:
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['MA_20'], name="MA 20", line=dict(dash='dot', color='green')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['MA_50'], name="MA 50", line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Volatility'], name="Volatility", yaxis="y2", line=dict(color='orange')))
    fig.update_layout(yaxis2=dict(title="Volatility", overlaying='y', side='right'))

fig.update_layout(title="QQQ Price Forecast", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# CPI Sensitivity Analysis
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

sens_fig.update_layout(title="QQQ Forecast Sensitivity to CPI", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
st.plotly_chart(sens_fig, use_container_width=True)

# Multi-Scenario Forecast Comparison
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
        }, index=future_dates)
        comp_df = comp_df[features]
        yhat = model.predict(comp_df)
        if use_live_price:
            yhat += (latest_close - yhat[0])
        yhat *= (1 + macro_bias)
        comp_fig.add_trace(go.Scatter(x=future_dates, y=yhat, name=name))
    comp_fig.update_layout(title="Scenario Forecast Comparison", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(comp_fig, use_container_width=True)

# Backtest Model Accuracy
st.subheader("🔁 Backtest Model Accuracy")
backtest_mode = st.checkbox("Enable Backtest")
if backtest_mode:
    backtest_date = st.date_input("Start Backtest From", value=datetime(2023, 1, 1))
    back_df = qqq_data.loc[qqq_data.index >= pd.to_datetime(backtest_date)].copy()
    if not back_df.empty:
        back_df['Prediction'] = model.predict(back_df[features])
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=back_df.index, y=back_df['Close'], name="Actual", line=dict(color='black')))
        fig_bt.add_trace(go.Scatter(x=back_df.index, y=back_df['Prediction'], name="Predicted", line=dict(color='blue')))
        fig_bt.update_layout(title="Backtest: QQQ Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.warning("No data available for the selected backtest period.")

# Feature Importance
st.subheader("📌 Feature Importance (XGBoost)")
if st.checkbox("Show Feature Importance"):
    fig_imp, ax = plt.subplots(figsize=(10, 6))
    xgb.plot_importance(model.get_booster(), ax=ax, importance_type='weight')
    plt.tight_layout()
    st.pyplot(fig_imp)

# Download Forecast Data
st.subheader("📥 Download Forecast Data")
forecast_df = future_df.copy()
forecast_df['Forecast'] = forecast
forecast_df['Date'] = future_dates
forecast_df.set_index('Date', inplace=True)
st.download_button("Download Forecast CSV", forecast_df.to_csv().encode(), file_name="qqq_forecast.csv")

# Download Technical Indicators
if show_tech:
    tech_df = qqq_data[['Close', 'MA_20', 'MA_50', 'Volatility']].dropna()
    st.download_button("Download Technical Indicators CSV", tech_df.to_csv().encode(), file_name="technical_indicators.csv")

# Download Forecast Chart
buf = io.BytesIO()
fig.write_image(buf, format="png")
st.download_button("Download Forecast Chart as PNG", buf.getvalue(), file_name="forecast_chart.png")
