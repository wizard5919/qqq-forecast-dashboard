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
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page configuration
st.set_page_config(page_title="QQQ Forecast Simulator", layout="wide")

# Initialize session state
if 'fed' not in st.session_state:
    st.session_state.fed = 5.25
    st.session_state.cpi = 3.5
    st.session_state.unemp = 3.9
    st.session_state.gdp = 21000
    st.session_state.vix = 20.0
    st.session_state.yield_10 = 4.0
    st.session_state.yield_2 = 3.5
    st.session_state.eps = 10.0
    st.session_state.sent = 70
    st.session_state.macro_bias = 0.0
    st.session_state.horizon = 30

def fetch_data(ticker, start_date):
    """Fetch data for a given ticker using yfinance."""
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def add_technical_indicators(df):
    """
    Add EMA (9, 20, 50, 200), VWAP, and Keltner Channels to the DataFrame.
    Assumes df has 'Close', 'High', 'Low', 'Volume' columns from yfinance.
    """
    required_cols = ['Close', 'High', 'Low', 'Volume']
    if not all(col in df.columns for col in required_cols):
        df = yf.download("QQQ", start=df.index[0], end=df.index[-1], progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Price_x_Volume'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_Price_Volume'] = df['Price_x_Volume'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_Price_Volume'] / df['Cumulative_Volume']
    df['KC_Middle'] = df['Close'].ewm(span=20, adjust=False).mean()
    tr = pd.DataFrame()
    tr['h_l'] = df['High'] - df['Low']
    tr['h_pc'] = abs(df['High'] - df['Close'].shift())
    tr['l_pc'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = tr[['h_l', 'h_pc', 'l_pc']].max(axis=1)
    df['ATR'] = df['TR'].ewm(span=20, adjust=False).mean()
    k_mult = 2
    df['KC_Upper'] = df['KC_Middle'] + k_mult * df['ATR']
    df['KC_Lower'] = df['KC_Middle'] - k_mult * df['ATR']
    df = df.drop(['Typical_Price', 'Price_x_Volume', 'Cumulative_Price_Volume', 'Cumulative_Volume', 'TR', 'ATR'], axis=1, errors='ignore')
    return df

def train_model(X, y):
    """Train an XGBoost model."""
    try:
        model_xgb = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model_xgb.fit(X, y)
        return model_xgb
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

@st.cache_resource
def load_data_and_models():
    try:
        start_date = "2018-01-01"
        qqq = yf.download("QQQ", start=start_date, progress=False)
        if isinstance(qqq.columns, pd.MultiIndex):
            qqq.columns = [col[0] for col in qqq.columns]

        vix = fetch_data("^VIX", start_date)
        treasury10 = fetch_data("^TNX", start_date)
        treasury2 = fetch_data("^IRX", start_date)

        if any(df is None for df in [qqq, vix, treasury10, treasury2]):
            return None, None, None, None

        vix = vix['Close'].squeeze().reindex(qqq.index, method='ffill').ffill()
        treasury10 = treasury10['Close'].squeeze().reindex(qqq.index, method='ffill').ffill()
        treasury2 = treasury2['Close'].squeeze().reindex(qqq.index, method='ffill').ffill()

        qqq = add_technical_indicators(qqq)
        qqq['Date_Ordinal'] = qqq.index.map(datetime.toordinal)
        qqq['FedFunds'] = 5.25
        qqq['Unemployment'] = 3.9
        qqq['CPI'] = 3.5
        qqq['GDP'] = 21000
        qqq['VIX'] = vix
        qqq['10Y_Yield'] = treasury10
        qqq['2Y_Yield'] = treasury2
        qqq['Yield_Spread'] = (treasury10 - treasury2)
        qqq['EPS_Growth'] = np.linspace(5, 15, len(qqq))
        qqq['Sentiment'] = 70
        qqq['MA_20'] = qqq['EMA_20']
        qqq['MA_50'] = qqq['EMA_50']
        qqq['Volatility'] = qqq['Close'].rolling(window=20).std()

        features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX',
                    '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'EPS_Growth', 'Sentiment',
                    'EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle']

        X = qqq[features].copy()
        y = qqq['Close']
        X = X.dropna()
        y = y.loc[X.index]

        # Fix MultiIndex issue
        X.columns = ['_'.join(map(str, col)).strip() for col in X.columns] if isinstance(X.columns, pd.MultiIndex) else X.columns.astype(str).str.strip()

        model_xgb = train_model(X, y)
        if model_xgb is None:
            return None, None, None, None

        model_linear = LinearRegression().fit(X, y)
        poly_features = np.column_stack([X.values, X.values ** 2])
        model_poly = LinearRegression().fit(poly_features, y)

        # Combine models into an ensemble
        model = VotingRegressor(estimators=[
            ('xgb', model_xgb),
            ('linear', model_linear),
            ('poly', model_poly)
        ])
        model.fit(X, y)

        return model, features, qqq, qqq['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error in load_data_and_models: {e}")
        return None, None, None, None

# Load model and data
model, features, qqq_data, latest_close = load_data_and_models()

if model is None or qqq_data is None:
    st.error("Failed to load model or data. Please try again later.")
    st.stop()

st.title("📈 QQQ Forecast Simulator")

# User inputs
use_live_price = st.checkbox("📡 Use Live QQQ Close ($%.2f) as Starting Point" % latest_close, value=True)
horizon = st.selectbox("📆 Forecast Horizon (days)", [30, 60, 90], index=[30, 60, 90].index(st.session_state.horizon))
show_tech = st.checkbox("📊 Show Technical Indicators")
macro_bias = st.slider("🧠 Macro News Sentiment Overlay (-10% to +10%)", -0.10, 0.10, st.session_state.macro_bias, step=0.01)
st.session_state.macro_bias = macro_bias
st.session_state.horizon = horizon

# Scenarios
scenarios = {
    "Recession": dict(fed=5.5, cpi=6.5, unemp=6.0, gdp=19000, vix=45.0, yield_10=2.5, yield_2=4.0, eps=5.0, sent=35),
    "Rate Cut": dict(fed=3.0, cpi=2.0, unemp=3.8, gdp=25000, vix=15.0, yield_10=3.0, yield_2=2.5, eps=12.0, sent=85),
    "Soft Landing": dict(fed=4.0, cpi=2.8, unemp=4.2, gdp=24000, vix=18.0, yield_10=3.8, yield_2=3.5, eps=10.0, sent=70)
}

scenario = st.radio("Select Scenario", ["Custom"] + list(scenarios.keys()))

# Scenario inputs
if scenario == "Custom":
    fed = st.number_input("Fed Funds Rate", value=st.session_state.fed, step=0.1)
    cpi = st.number_input("CPI", value=st.session_state.cpi, step=0.1)
    unemp = st.number_input("Unemployment", value=st.session_state.unemp, step=0.1)
    gdp = st.number_input("GDP", value=st.session_state.gdp, step=1000)
    vix = st.number_input("VIX", value=st.session_state.vix, step=1.0)
    yield_10 = st.number_input("10Y Yield", value=st.session_state.yield_10, step=0.1)
    yield_2 = st.number_input("2Y Yield", value=st.session_state.yield_2, step=0.1)
    eps = st.number_input("EPS Growth", value=st.session_state.eps, step=0.5)
    sent = st.slider("Sentiment", 0, 100, st.session_state.sent)
    st.session_state.update({'fed': fed, 'cpi': cpi, 'unemp': unemp, 'gdp': gdp, 'vix': vix,
                            'yield_10': yield_10, 'yield_2': yield_2, 'eps': eps, 'sent': sent})
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
    'Sentiment': sent,
    'EMA_9': qqq_data['EMA_9'].iloc[-1],
    'EMA_20': qqq_data['EMA_20'].iloc[-1],
    'EMA_50': qqq_data['EMA_50'].iloc[-1],
    'EMA_200': qqq_data['EMA_200'].iloc[-1],
    'VWAP': qqq_data['VWAP'].iloc[-1],
    'KC_Upper': qqq_data['KC_Upper'].iloc[-1],
    'KC_Lower': qqq_data['KC_Lower'].iloc[-1],
    'KC_Middle': qqq_data['KC_Middle'].iloc[-1]
}, index=future_dates)
future_df = future_df[features]

# Make predictions
try:
    forecast = model.predict(future_df)
    forecast *= (1 + macro_bias)
    if use_live_price:
        shift = latest_close - forecast[0]
        forecast += shift
    confidence_std = 0.03 * forecast
    forecast_upper = forecast + confidence_std
    forecast_lower = forecast - confidence_std
except Exception as e:
    st.error(f"Error making predictions: {e}")
    st.stop()

# Main forecast plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Close'], name="Historical QQQ", line=dict(color='black')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_upper, name="Upper Bound", line=dict(color='lightblue'), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_lower, name="Lower Bound", fill='tonexty', line=dict(color='lightblue'), fillcolor='rgba(173, 216, 230, 0.3)', showlegend=False))

if show_tech:
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['EMA_9'], name="EMA 9", line=dict(dash='dot', color='purple')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['EMA_20'], name="EMA 20", line=dict(dash='dot', color='green')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['EMA_50'], name="EMA 50", line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['EMA_200'], name="EMA 200", line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['VWAP'], name="VWAP", line=dict(dash='solid', color='orange')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['KC_Upper'], name="KC Upper", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['KC_Lower'], name="KC Lower", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['KC_Middle'], name="KC Middle", line=dict(dash='dot', color='gray')))
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
    try:
        temp_pred = model.predict(temp_df)
        if use_live_price:
            temp_pred += (latest_close - temp_pred[0])
        temp_pred *= (1 + macro_bias)
        sens_fig.add_trace(go.Scatter(x=future_dates, y=temp_pred, name=f"CPI={val:.1f}"))
    except Exception as e:
        st.warning(f"Error in CPI sensitivity analysis for CPI={val:.1f}: {e}")
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
            'Sentiment': s['sent'],
            'EMA_9': qqq_data['EMA_9'].iloc[-1],
            'EMA_20': qqq_data['EMA_20'].iloc[-1],
            'EMA_50': qqq_data['EMA_50'].iloc[-1],
            'EMA_200': qqq_data['EMA_200'].iloc[-1],
            'VWAP': qqq_data['VWAP'].iloc[-1],
            'KC_Upper': qqq_data['KC_Upper'].iloc[-1],
            'KC_Lower': qqq_data['KC_Lower'].iloc[-1],
            'KC_Middle': qqq_data['KC_Middle'].iloc[-1]
        }, index=future_dates)
        comp_df = comp_df[features]
        try:
            yhat = model.predict(comp_df)
            if use_live_price:
                yhat += (latest_close - yhat[0])
            yhat *= (1 + macro_bias)
            comp_fig.add_trace(go.Scatter(x=future_dates, y=yhat, name=name))
        except Exception as e:
            st.warning(f"Error in scenario {name}: {e}")
    comp_fig.update_layout(title="Scenario Forecast Comparison", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(comp_fig, use_container_width=True)

# Backtest Model Accuracy
st.subheader("🔁 Backtest Model Accuracy")
backtest_mode = st.checkbox("Enable Backtest")
if backtest_mode:
    backtest_date = st.date_input("Start Backtest From", value=datetime(2023, 1, 1))
    back_df = qqq_data.loc[qqq_data.index >= pd.to_datetime(backtest_date)].copy()
    if not back_df.empty:
        try:
            back_df['Prediction'] = model.predict(back_df[features])
            mae = mean_absolute_error(back_df['Close'], back_df['Prediction'])
            rmse = np.sqrt(mean_squared_error(back_df['Close'], back_df['Prediction']))
            st.write(f"**Backtest Metrics:** MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=back_df.index, y=back_df['Close'], name="Actual", line=dict(color='black')))
            fig_bt.add_trace(go.Scatter(x=back_df.index, y=back_df['Prediction'], name="Predicted", line=dict(color='blue')))
            fig_bt.update_layout(title="Backtest: QQQ Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
            st.plotly_chart(fig_bt, use_container_width=True)
        except Exception as e:
            st.warning(f"Error in backtest: {e}")
    else:
        st.warning("No data available for the selected backtest period.")

# Feature Importance
st.subheader("📌 Feature Importance (XGBoost)")
if st.checkbox("Show Feature Importance"):
    try:
        fig_imp, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(model.estimators_[0].get_booster(), ax=ax, importance_type='weight')
        plt.tight_layout()
        st.pyplot(fig_imp)
    except Exception as e:
        st.warning(f"Error displaying feature importance: {e}")

# Download Forecast Data
st.subheader("📥 Download Forecast Data")
forecast_df = future_df.copy()
forecast_df['Forecast'] = forecast
forecast_df['Date'] = future_dates
forecast_df.set_index('Date', inplace=True)
st.download_button("Download Forecast CSV", forecast_df.to_csv().encode(), file_name="qqq_forecast.csv")

# Download Technical Indicators
if show_tech:
    tech_df = qqq_data[['Close', 'EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle', 'Volatility']].dropna()
    st.download_button("Download Technical Indicators CSV", tech_df.to_csv().encode(), file_name="technical_indicators.csv")

# Download Forecast Chart
buf = io.BytesIO()
try:
    fig.write_image(buf, format="png")
    st.download_button("Download Forecast Chart as PNG", buf.getvalue(), file_name="forecast_chart.png")
except Exception as e:
    st.warning(f"Error generating chart download: {e}")
