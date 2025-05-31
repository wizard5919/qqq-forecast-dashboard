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
        # Handle MultiIndex columns by flattening to single level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def add_technical_indicators(df):
    """
    Add EMA (9, 20, 50, 200), VWAP, and Keltner Channels to the DataFrame.
    Assumes df has 'Close', 'High', 'Low', 'Volume' columns from yfinance.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    try:
        # Calculate indicators
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
        
        # Cleanup temporary columns
        cols_to_drop = ['Typical_Price', 'Price_x_Volume', 'Cumulative_Price_Volume', 
                       'Cumulative_Volume', 'TR', 'ATR']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        return df
    except KeyError as e:
        st.error(f"Missing required column in data: {e}")
        # Attempt to add missing columns with dummy data
        for col in ['Close', 'High', 'Low', 'Volume']:
            if col not in df.columns:
                df[col] = 100.0
        return add_technical_indicators(df)  # Retry with dummy data

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
        st.info("📡 Loading data and models... This may take a minute")
        start_date = "2018-01-01"
        
        # Fetch QQQ data with retries
        qqq = None
        for i in range(3):  # Retry up to 3 times
            qqq = fetch_data("QQQ", start_date)
            if qqq is not None and not qqq.empty and 'Close' in qqq.columns:
                break
            st.warning(f"Attempt {i+1}/3 failed, retrying...")
            time.sleep(2)  # Wait before retrying
            
        if qqq is None or qqq.empty:
            st.error("❌ Failed to load QQQ data after multiple attempts. Using fallback data.")
            # Create fallback data
            dates = pd.date_range(start=start_date, end=datetime.today())
            qqq = pd.DataFrame({
                'Open': np.linspace(100, 500, len(dates)),
                'High': np.linspace(105, 505, len(dates)),
                'Low': np.linspace(95, 495, len(dates)),
                'Close': np.linspace(100, 500, len(dates)),
                'Volume': np.linspace(1000000, 5000000, len(dates))
            }, index=dates)
        
        # Fetch other data with fallbacks
        def get_data_with_fallback(ticker, fallback_value, name):
            data = fetch_data(ticker, start_date)
            if data is None or data.empty or 'Close' not in data.columns:
                st.warning(f"⚠️ Using fallback value for {name}")
                return pd.Series(fallback_value, index=qqq.index, name='Close')
            return data['Close'].squeeze().reindex(qqq.index, method='ffill').ffill().bfill()
        
        vix = get_data_with_fallback("^VIX", 20.0, "VIX")
        treasury10 = get_data_with_fallback("^TNX", 4.0, "10Y Yield")
        treasury2 = get_data_with_fallback("^IRX", 3.5, "2Y Yield")

        # Add technical indicators
        qqq = add_technical_indicators(qqq)
        
        # Add macroeconomic data
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
        
        # Add moving averages if available
        if 'EMA_20' in qqq.columns:
            qqq['MA_20'] = qqq['EMA_20']
        else:
            qqq['MA_20'] = qqq['Close'].rolling(window=20).mean()
            
        if 'EMA_50' in qqq.columns:
            qqq['MA_50'] = qqq['EMA_50']
        else:
            qqq['MA_50'] = qqq['Close'].rolling(window=50).mean()
            
        qqq['Volatility'] = qqq['Close'].rolling(window=20).std()

        # Prepare features
        features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX',
                    '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'EPS_Growth', 'Sentiment']
        
        # Add technical indicators if available
        for col in ['EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle']:
            if col in qqq.columns:
                features.append(col)

        X = qqq[features].copy()
        y = qqq['Close']
        X = X.dropna()
        y = y.loc[X.index]

        # Handle column names safely
        X.columns = [str(col).strip() for col in X.columns]

        # Train models
        model_xgb = train_model(X, y)
        if model_xgb is None:
            st.warning("XGBoost model training failed. Using linear models only.")
            # Create a dummy XGBoost model
            model_xgb = LinearRegression()
            model_xgb.fit(X, y)

        model_linear = LinearRegression().fit(X, y)
        poly = PolynomialFeatures(degree=2)
        poly_features = poly.fit_transform(X)
        model_poly = LinearRegression().fit(poly_features, y)

        return qqq, model_xgb, model_linear, model_poly, poly
    except Exception as e:
        st.error(f"❌ Critical error in load_data_and_models: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None

# Load model and data with spinner
with st.spinner("Loading data and training models. This may take up to 30 seconds..."):
    qqq_data, xgb_model, linear_model, poly_model, poly = load_data_and_models()

# Final safety check - create synthetic data if everything else failed
if qqq_data is None:
    st.error("""
    ❌ Failed to load QQQ data. Possible reasons:
    1. Yahoo Finance API is unavailable
    2. Network connection issues
    3. Data format changes
    
    Using synthetic data for demonstration purposes.
    """)
    
    # Create synthetic data as a last resort
    dates = pd.date_range(start="2018-01-01", end=datetime.today())
    qqq_data = pd.DataFrame({
        'Open': np.linspace(100, 500, len(dates)),
        'High': np.linspace(105, 505, len(dates)),
        'Low': np.linspace(95, 495, len(dates)),
        'Close': np.linspace(100, 500, len(dates)),
        'Volume': np.linspace(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Add required columns
    qqq_data['EMA_9'] = qqq_data['Close']
    qqq_data['EMA_20'] = qqq_data['Close']
    qqq_data['EMA_50'] = qqq_data['Close']
    qqq_data['EMA_200'] = qqq_data['Close']
    qqq_data['VWAP'] = qqq_data['Close']
    qqq_data['KC_Upper'] = qqq_data['Close'] * 1.05
    qqq_data['KC_Lower'] = qqq_data['Close'] * 0.95
    qqq_data['KC_Middle'] = qqq_data['Close']
    
    # Create simple linear models as fallback
    X = np.array(range(len(qqq_data))).reshape(-1, 1)
    y = qqq_data['Close']
    xgb_model = LinearRegression().fit(X, y)
    linear_model = LinearRegression().fit(X, y)
    poly_model = LinearRegression().fit(X, y)
    poly = None

# Now we can safely proceed
st.title("📈 QQQ Forecast Simulator")

# User inputs
if 'Close' not in qqq_data.columns:
    # If we're missing Close column in our synthetic data, add it
    qqq_data['Close'] = np.linspace(100, 500, len(qqq_data))
    
latest_close = qqq_data['Close'].iloc[-1] if 'Close' in qqq_data.columns else None
if latest_close is None:
    st.error("Close price not found in data")
    st.stop()

use_live_price = st.checkbox(f"📡 Use Live QQQ Close (${latest_close:.2f}) as Starting Point", value=True)
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

# Initialize forecast variables with None
forecast = None
forecast_upper = None
forecast_lower = None

try:
    # Create future dataframe
    features = ['Date_Ordinal', 'FedFunds', 'Unemployment', 'CPI', 'GDP', 'VIX',
                '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'EPS_Growth', 'Sentiment',
                'EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle']
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
    # Use XGBoost for main forecast
    forecast = xgb_model.predict(future_df)
    # Adjust with polynomial features for poly_model
    future_poly = poly.transform(future_df)
    forecast_poly = poly_model.predict(future_poly)
    # Average the predictions
    forecast = (forecast + forecast_poly) / 2
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

# Only proceed if we have a valid forecast
if forecast is None or forecast_upper is None or forecast_lower is None:
    st.error("Forecasting failed. Please try again.")
    st.stop()

# Main forecast plot
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
                ))  # Added closing parenthesis here
            else:
                fig.add_trace (go.Scatter(
                    x=qqq_data.index, 
                    y=qqq_data[col], 
                    name=col, 
                    line=dict(color=color, dash=style)
                ))
    
    if 'Volatility' in qqq_data.columns:
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
        temp_pred = xgb_model.predict(temp_df)
        temp_poly = poly.transform(temp_df)
        temp_pred_poly = poly_model.predict(temp_poly)
        temp_pred = (temp_pred + temp_pred_poly) / 2
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
            yhat = xgb_model.predict(comp_df)
            yhat_poly = poly_model.predict(poly.transform(comp_df))
            yhat = (yhat + yhat_poly) / 2
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
            features = [col for col in features if col in back_df.columns]
            back_df['Prediction'] = xgb_model.predict(back_df[features])
            back_df['Prediction_Poly'] = poly_model.predict(poly.transform(back_df[features]))
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

# Feature Importance
st.subheader("📌 Feature Importance (XGBoost)")
if st.checkbox("Show Feature Importance"):
    try:
        fig_imp, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(xgb_model.get_booster(), ax=ax, importance_type='weight')
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
    tech_cols = ['Close', 'EMA_9', 'EMA_20', 'EMA_50', 'EMA_200', 'VWAP', 'KC_Upper', 'KC_Lower', 'KC_Middle', 'Volatility']
    tech_cols = [col for col in tech_cols if col in qqq_data.columns]
    if tech_cols:
        tech_df = qqq_data[tech_cols].dropna()
        st.download_button("Download Technical Indicators CSV", tech_df.to_csv().encode(), file_name="technical_indicators.csv")

# Download Forecast Chart
buf = io.BytesIO()
try:
    fig.write_image(buf, format="png")
    st.download_button("Download Forecast Chart as PNG", buf.getvalue(), file_name="forecast_chart.png")
except Exception as e:
    st.warning(f"Error generating chart download: {e}")
