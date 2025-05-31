
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import plotly.graph_objects as go
import io
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests

st.set_page_config(page_title="QQQ Forecast Simulator", layout="wide")

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

qqq_data = None
xgb_model = None
linear_model = None
poly_model = None
poly = None
available_features = None
latest_close = 400.0

def fetch_data(ticker, start_date):
    """Fetch data with retries and fallbacks"""
    max_retries = 5
    retry_delay = 5
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    for i in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, progress=False, timeout=10)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                else:
                    df.columns = [str(col).strip().title() for col in df.columns]
                
                df.columns = [str(col).strip().title() for col in df.columns]  # Normalize again after flattening
                missing_required = [col for col in expected_columns if col not in df.columns]
                if not missing_required:
                    if 'Adj Close' not in df.columns:
                        df['Adj Close'] = df['Close']
                        st.write(f"Added synthetic Adj Close for {ticker}")
                    st.info(f"Successfully fetched data for {ticker}")
                    return df
                else:
                    st.warning(f"Missing required columns {missing_required} for {ticker}, attempt {i+1}/{max_retries}")
            else:
                st.warning(f"Empty data for {ticker}, attempt {i+1}/{max_retries}")
        except requests.exceptions.RequestException as e:
            st.warning(f"Network error fetching {ticker}: {e}, attempt {i+1}/{max_retries}")
        except Exception as e:
            st.warning(f"Unexpected error fetching {ticker}: {e}, attempt {i+1}/{max_retries}")
        time.sleep(retry_delay)

    st.error(f"Failed to fetch {ticker} after {max_retries} attempts. Using synthetic data.")
    dates = pd.date_range(start=start_date, end=datetime.today())
    df = pd.DataFrame({
        'Open': np.linspace(100, 500, len(dates)),
        'High': np.linspace(105, 505, len(dates)),
        'Low': np.linspace(95, 495, len(dates)),
        'Close': np.linspace(100, 500, len(dates)),
        'Volume': np.linspace(1000000, 5000000, len(dates)),
        'Adj Close': np.linspace(100, 500, len(dates))
    }, index=dates)
    df.columns = [str(col).strip().title() for col in df.columns]  # Normalize fallback column names
    return df

# Be sure to apply the same column flattening logic to all data pulls (e.g., VIX, TNX, IRX)

# Also in load_data_and_models, replace any hard-coded title-casing of columns with:
# qqq.columns = [str(col).strip().title() for col in qqq.columns]
# vix.columns = [str(col).strip().title() for col in vix.columns]
# treasury10.columns = [str(col).strip().title() for col in treasury10.columns]
# treasury2.columns = [str(col).strip().title() for col in treasury2.columns]

def add_technical_indicators(df):
    df = df.copy()
    required_cols = ['Close', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.warning(f"⚠️ Column '{col}' not found. Using default value 100.")
            df[col] = 100.0

    try:
        for span in [9, 20, 50, 200]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['Typical_Price'] = typical_price
        df['Price_x_Volume'] = typical_price * df['Volume']
        df['Cumulative_Price_Volume'] = df['Price_x_Volume'].cumsum()
        df['Cumulative_Volume'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cumulative_Price_Volume'] / df['Cumulative_Volume']
        df['KC_Middle'] = df['Close'].ewm(span=20, adjust=False).mean()
        tr = pd.DataFrame()
        tr['h_l'] = df['High'] - df['Low']
        tr['h_pc'] = abs(df['High'] - df['Close'].shift())
        tr['l_pc'] = abs(df['Low'] - df['Close'].shift())
        df['TR'] = tr.max(axis=1)
        df['ATR'] = df['TR'].ewm(span=20, adjust=False).mean()
        k_mult = 2
        df['KC_Upper'] = df['KC_Middle'] + k_mult * df['ATR']
        df['KC_Lower'] = df['KC_Middle'] - k_mult * df['ATR']
        cols_to_drop = ['Typical_Price', 'Price_x_Volume', 'Cumulative_Price_Volume', 
                       'Cumulative_Volume', 'TR', 'ATR']
        for col in cols_to_drop:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df
    except Exception as e:
        st.error(f"Error adding technical indicators: {e}")
        import traceback
        st.error(traceback.format_exc())
        return df

def train_model(X, y):
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
        st.info("📡 Loading data and models... This may take a minute")
        start_date = "2018-01-01"
        
        qqq = fetch_data("QQQ", start_date)
        st.write(f"QQQ Data: Shape={qqq.shape}, Columns={list(qqq.columns)}")
        qqq.columns = [col.title() for col in qqq.columns]
        
        if qqq is None or qqq.empty or 'Close' not in qqq.columns:
            st.error("❌ Failed to load valid QQQ data. Using fallback data.")
            dates = pd.date_range(start=start_date, end=datetime.today())
            qqq = pd.DataFrame({
                'Open': np.linspace(100, 500, len(dates)),
                'High': np.linspace(105, 505, len(dates)),
                'Low': np.linspace(95, 495, len(dates)),
                'Close': np.linspace(100, 500, len(dates)),
                'Volume': np.linspace(1000000, 5000000, len(dates)),
                'Adj Close': np.linspace(100, 500, len(dates))
            }, index=dates)
            qqq.columns = [col.title() for col in qqq.columns]
        
        def get_data_with_fallback(ticker, fallback_value, name):
            data = fetch_data(ticker, start_date)
            st.write(f"{name} Data: Shape={data.shape}, Columns={list(data.columns)}")
            data.columns = [col.title() for col in data.columns]
            if data is None or data.empty or 'Close' not in data.columns:
                st.warning(f"⚠️ Using fallback value {fallback_value} for {name}")
                return pd.Series(fallback_value, index=qqq.index, name='Close')
            return data['Close'].squeeze().reindex(qqq.index, method='ffill').ffill().bfill()
        
        vix = get_data_with_fallback("^VIX", 20.0, "VIX")
        treasury10 = get_data_with_fallback("^TNX", 4.0, "10Y Yield")
        treasury2 = get_data_with_fallback("^IRX", 3.5, "2Y Yield")

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
        
        if 'EMA_20' in qqq.columns:
            qqq['MA_20'] = qqq['EMA_20']
        else:
            qqq['MA_20'] = qqq['Close'].rolling(window=20).mean()
            
        if 'EMA_50' in qqq.columns:
            qqq['MA_50'] = qqq['EMA_50']
        else:
            qqq['MA_50'] = qqq['Close'].rolling(window=50).mean()
            
        qqq['Volatility'] = qqq['Close'].rolling(window=20).std()

        essential_features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX',
                              '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'EPS_Growth', 'Sentiment']
        
        available_features = essential_features.copy()
        for col in ['EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle']:
            if col in qqq.columns:
                available_features.append(col)

        missing_features = [col for col in available_features if col not in qqq.columns]
        if missing_features:
            st.warning(f"Missing features {missing_features} in qqq DataFrame. Removing from available_features.")
            available_features = [col for col in available_features if col in qqq.columns]
        
        X = qqq[available_features].copy()
        y = qqq['Close']
        X = X.dropna()
        y = y.loc[X.index]

        model_xgb = train_model(X, y)
        if model_xgb is None:
            st.warning("XGBoost model training failed. Using linear models only.")
            model_xgb = LinearRegression()
            model_xgb.fit(X, y)

        model_linear = LinearRegression().fit(X, y)
        
        if len(available_features) < 20:
            poly_transformer = PolynomialFeatures(degree=2)
            poly_features = poly_transformer.fit_transform(X)
            model_poly = LinearRegression().fit(poly_features, y)
        else:
            st.warning("Too many features for polynomial model. Using linear model instead.")
            poly_transformer = None
            model_poly = model_linear

        return qqq, model_xgb, linear_model, model_poly, poly_transformer, available_features
    except Exception as e:
        st.error(f"❌ Critical error in load_data_and_models: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None, None

with st.spinner("Loading data and training models. This may take up to 30 seconds..."):
    try:
        result = load_data_and_models()
        qqq_data, xgb_model, linear_model, poly_model, poly, available_features = result
    except Exception as e:
        st.error(f"Error loading data and models: {e}")
        result = None

if result is None or any(x is None for x in [qqq_data, xgb_model, linear_model, poly_model, available_features]):
    st.error("""
    ❌ Failed to load QQQ data. Possible reasons:
    1. Yahoo Finance API is unavailable
    2. Network connection issues
    3. Data format changes
    Using synthetic data for demonstration purposes.
    """)
    
    dates = pd.date_range(start="2018-01-01", end=datetime.today())
    qqq_data = pd.DataFrame({
        'Open': np.linspace(100, 500, len(dates)),
        'High': np.linspace(105, 505, len(dates)),
        'Low': np.linspace(95, 495, len(dates)),
        'Close': np.linspace(100, 500, len(dates)),
        'Volume': np.linspace(1000000, 5000000, len(dates)),
        'Adj Close': np.linspace(100, 500, len(dates))
    }, index=dates)
    
    qqq_data['EMA_9'] = qqq_data['Close']
    qqq_data['EMA_20'] = qqq_data['Close']
    qqq_data['EMA_50'] = qqq_data['Close']
    qqq_data['EMA_200'] = qqq_data['Close']
    qqq_data['VWAP'] = qqq_data['Close']
    qqq_data['KC_Upper'] = qqq_data['Close'] * 1.05
    qqq_data['KC_Lower'] = qqq_data['Close'] * 0.95
    qqq_data['KC_Middle'] = qqq_data['Close']
    qqq_data['Date_Ordinal'] = qqq_data.index.map(datetime.toordinal)
    qqq_data['FedFunds'] = 5.25
    qqq_data['Unemployment'] = 3.9
    qqq_data['CPI'] = 3.5
    qqq_data['GDP'] = 21000
    qqq_data['VIX'] = 20.0
    qqq_data['10Y_Yield'] = 4.0
    qqq_data['2Y_Yield'] = 3.5
    qqq_data['Yield_Spread'] = qqq_data['10Y_Yield'] - qqq_data['2Y_Yield']
    qqq_data['EPS_Growth'] = np.linspace(5, 15, len(qqq_data))
    qqq_data['Sentiment'] = 70
    qqq_data['Volatility'] = qqq_data['Close'].rolling(window=20).std().fillna(0)
    
    qqq_data.columns = [col.title() for col in qqq_data.columns]
    
    available_features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX',
                         '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'EPS_Growth', 'Sentiment',
                         'EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle']
    
    X = qqq_data[available_features].copy()
    y = qqq_data['Close']
    xgb_model = LinearRegression().fit(X, y)
    linear_model = LinearRegression().fit(X, y)
    poly_model = LinearRegression().fit(X, y)
    poly = None

st.title("📈 QQQ Forecast Simulator")

try:
    if 'Close' in qqq_data.columns and not qqq_data['Close'].empty:
        latest_close = qqq_data['Close'].iloc[-1]
        if isinstance(latest_close, pd.Series):
            latest_close = latest_close.values[0]
        elif isinstance(latest_close, pd.DataFrame):
            latest_close = latest_close.iloc[0, 0]
    else:
        st.error("Close price not found in data. Using default value.")
        latest_close = 400.0
except:
    st.error("Error getting latest close price. Using default value.")
    latest_close = 400.0

try:
    checkbox_label = f"📡 Use Live QQQ Close (${latest_close:.2f}) as Starting Point"
except:
    checkbox_label = "📡 Use Live QQQ Close as Starting Point"
    latest_close = 400.0

use_live_price = st.checkbox(checkbox_label, value=True)
horizon = st.selectbox("📆 Forecast Horizon (days)", [30, 60, 90], index=[30, 60, 90].index(st.session_state.horizon))
show_tech = st.checkbox("📊 Show Technical Indicators")
macro_bias = st.slider("🧠 Macro News Sentiment Overlay (-10% to +10%)", -0.10, 0.10, st.session_state.macro_bias, step=0.01)
st.session_state.macro_bias = macro_bias
st.session_state.horizon = horizon

scenarios = {
    "Recession": dict(fed=5.5, cpi=6.5, unemp=6.0, gdp=19000, vix=45.0, yield_10=2.5, yield_2=4.0, eps=5.0, sent=35),
    "Rate Cut": dict(fed=3.0, cpi=2.0, unemp=3.8, gdp=25000, vix=15.0, yield_10=3.0, yield_2=2.5, eps=12.0, sent=85),
    "Soft Landing": dict(fed=4.0, cpi=2.8, unemp=4.2, gdp=24000, vix=18.0, yield_10=3.8, yield_2=3.5, eps=10.0, sent=70)
}

scenario = st.radio("Select Scenario", ["Custom"] + list(scenarios.keys()))

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

future_dates = pd.date_range(start=datetime.today(), periods=horizon, freq='D')
date_ordinals = [d.toordinal() for d in future_dates]

forecast = None
forecast_upper = None
forecast_lower = None

try:
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
    
    for col in ['EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle']:
        if col in available_features:
            if col in qqq_data.columns:
                future_df[col] = qqq_data[col].iloc[-1]
            else:
                future_df[col] = latest_close

    future_df = future_df[available_features]
    
    forecast = xgb_model.predict(future_df)
    if poly is not None:
        future_poly = poly.transform(future_df)
        forecast_poly = poly_model.predict(future_poly)
        forecast = (forecast + forecast_poly) / 2
    else:
        forecast_linear = linear_model.predict(future_df)
        forecast = (forecast + forecast_linear) / 2
        
    forecast = forecast.ravel()
    forecast *= (1 + macro_bias)
    if use_live_price:
        shift = latest_close - forecast[0]
        forecast += shift
        
    confidence_std = 0.03 * forecast
    forecast_upper = forecast + confidence_std
    forecast_lower = forecast - confidence_std
except Exception as e:
    st.error(f"Error making predictions: {e}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()

if forecast is None or forecast_upper is None or forecast_lower is None:
    st.error("Forecasting failed. Please try again.")
    st.stop()

fig = go.Figure()
fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Close'], name="Historical QQQ", line=dict(color='black')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_upper, name="Upper Bound", line=dict(color='lightblue'), showlegend=False))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_lower, name="Lower Bound", fill='tonexty', line=dict(color='lightblue'), fillcolor='rgba(173, 216, 230, 0.3)', showlegend=False))

if show_tech:
    tech_cols = ['EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle', 'Volatility']
    colors = ['purple', 'green', 'red', 'blue', 'orange', 'gray', 'gray', 'gray', 'orange']
    styles = ['dot', 'dot', 'dash', 'dash', 'solid', 'solid', 'solid', 'solid', 'solid']
    
    for col, color, style in zip(tech_cols, colors, styles):
        if col in qqq_data.columns:
            if col == 'Volatility':
                fig.add_trace(go.Scatter(
                    x=qqq_data.index, 
                    y=qqq_data[col], 
                    name=col, 
                    yaxis="y2", 
                    line=dict(color=color, dash=style)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=qqq_data.index, 
                    y=qqq_data[col], 
                    name=col, 
                    line=dict(color=color, dash=style)
                ))
    
    if 'Volatility' in qqq_data.columns:
        fig.update_layout(yaxis2=dict(title="Volatility", overlaying='y', side='right'))
    
fig.update_layout(title="QQQ Price Forecast", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📈 CPI Sensitivity Analysis")
cpi_range = np.linspace(cpi - 1, cpi + 1, 5)
sens_fig = go.Figure()
for val in cpi_range:
    temp_df = future_df.copy()
    temp_df['CPI'] = val
    
    try:
        temp_pred = xgb_model.predict(temp_df)
        if poly is not None:
            temp_poly = poly.transform(temp_df)
            temp_pred_poly = poly_model.predict(temp_poly)
            temp_pred = (temp_pred + temp_pred_poly) / 2
        else:
            temp_pred_linear = linear_model.predict(temp_df)
            temp_pred = (temp_pred + temp_pred_linear) / 2
            
        temp_pred = temp_pred.ravel()
        temp_pred *= (1 + macro_bias)
        if use_live_price:
            temp_pred += (latest_close - temp_pred[0])
        sens_fig.add_trace(go.Scatter(x=future_dates, y=temp_pred, name=f"CPI={val:.1f}"))
    except Exception as e:
        st.warning(f"Error in CPI sensitivity analysis for CPI={val:.1f}: {e}")
sens_fig.update_layout(title="QQQ Forecast Sensitivity to CPI", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
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
        }, index=future_dates)
        
        for col in ['EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle']:
            if col in available_features:
                if col in qqq_data.columns:
                    comp_df[col] = qqq_data[col].iloc[-1]
                else:
                    comp_df[col] = latest_close
        
        comp_df = comp_df[available_features]
        
        try:
            yhat = xgb_model.predict(comp_df)
            if poly is not None:
                yhat_poly = poly_model.predict(poly.transform(comp_df))
                yhat = (yhat + yhat_poly) / 2
            else:
                yhat_linear = linear_model.predict(comp_df)
                yhat = (yhat + yhat_linear) / 2
                
            yhat = yhat.ravel()
            yhat *= (1 + macro_bias)
            if use_live_price:
                yhat += (latest_close - yhat[0])
            comp_fig.add_trace(go.Scatter(x=future_dates, y=yhat, name=name))
        except Exception as e:
            st.warning(f"Error in scenario {name}: {e}")
    comp_fig.update_layout(title="Scenario Forecast Comparison", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(comp_fig, use_container_width=True)

st.subheader("🔁 Backtest Model Accuracy")
backtest_mode = st.checkbox("Enable Backtest")
if backtest_mode:
    backtest_date = st.date_input("Start Backtest From", value=datetime(2023, 1, 1))
    back_df = qqq_data.loc[qqq_data.index >= pd.to_datetime(backtest_date)].copy()
    if not back_df.empty:
        try:
            backtest_features = [col for col in available_features if col in back_df.columns]
            back_df = back_df[backtest_features]
            
            back_df['Prediction'] = xgb_model.predict(back_df)
            if poly is not None:
                back_df['Prediction_Poly'] = poly_model.predict(poly.transform(back_df))
            else:
                back_df['Prediction_Poly'] = linear_model.predict(back_df)
                
            back_df['Prediction'] = (back_df['Prediction'] + back_df['Prediction_Poly']) / 2
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

st.subheader("📌 Feature Importance (XGBoost)")
if st.checkbox("Show Feature Importance"):
    try:
        fig_imp, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(xgb_model.get_booster(), ax=ax, importance_type='weight')
        plt.tight_layout()
        st.pyplot(fig_imp)
    except Exception as e:
        st.warning(f"Error displaying feature importance: {e}")

st.subheader("📥 Download Forecast Data")
forecast_df = future_df.copy()
forecast_df['Forecast'] = forecast
forecast_df['Date'] = future_dates
forecast_df.set_index('Date', inplace=True)
st.download_button("Download Forecast CSV", forecast_df.to_csv().encode(), file_name="qqq_forecast.csv")

if show_tech:
    tech_cols = ['Close', 'EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle', 'Volatility']
    tech_cols = [col for col in tech_cols if col in qqq_data.columns]
    if tech_cols:
        tech_df = qqq_data[tech_cols].dropna()
        st.download_button("Download Technical Indicators CSV", tech_df.to_csv().encode(), file_name="technical_indicators.csv")

buf = io.BytesIO()
try:
    fig.write_image(buf, format="png")
    st.download_button("Download Forecast Chart as PNG", buf.getvalue(), file_name="forecast_chart.png")
except Exception as e:
    st.warning(f"Error generating chart download: {e}")
