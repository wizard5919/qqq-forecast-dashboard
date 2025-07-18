import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time  # Import time here
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import plotly.graph_objects as go
import io
import time as ttime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import shap
from prophet import Prophet
from streamlit_option_menu import option_menu
from fredapi import Fred
import pytz
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FRED API setup (configure as environment variable in Streamlit Cloud)
FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY != "YOUR_FRED_API_KEY" else None

# Set page config as the first Streamlit command
st.set_page_config(page_title="QQQ Forecast Simulator", layout="wide", initial_sidebar_state="expanded")

def fetch_economic_indicator(series_id, fallback_value):
    """Fetch latest economic indicator value from FRED"""
    try:
        if fred is None:
            return fallback_value
        series = fred.get_series(series_id, observation_start='2025-01-01')
        if not series.empty:
            return float(series.iloc[-1])
        return fallback_value
    except Exception:
        return fallback_value

def fetch_yfinance_indicator(ticker, fallback_value):
    """Fetch latest indicator value from yfinance with retry"""
    max_retries = 3
    retry_delay = 5
    for _ in range(max_retries):
        try:
            data = yf.download(ticker, period="1d", progress=False)
            if not data.empty and 'Close' in data.columns:
                return float(data['Close'].iloc[-1])
            return fallback_value
        except Exception:
            ttime.sleep(retry_delay)
    return fallback_value

# Initialize session state with latest economic indicators
def initialize_session_state():
    """Initialize session state variables with latest economic indicators"""
    if 'fed' not in st.session_state:
        st.session_state.fed = fetch_economic_indicator('FEDFUNDS', 5.25)
        st.session_state.cpi = fetch_economic_indicator('CPIAUCSL', 3.5)
        st.session_state.unemp = fetch_economic_indicator('UNRATE', 3.9)
        st.session_state.gdp = fetch_economic_indicator('GDP', 21000)
        st.session_state.vix = fetch_yfinance_indicator('^VIX', 20.0)
        st.session_state.yield_10 = fetch_yfinance_indicator('^TNX', 4.0)
        st.session_state.yield_2 = fetch_yfinance_indicator('^IRX', 3.5)
        st.session_state.eps = 10.0
        st.session_state.sent = 70
        st.session_state.macro_bias = 0.0
        st.session_state.horizon = 30
        st.session_state.saved_scenarios = {}
        st.session_state.active_tab = "Forecast"
        st.session_state.forecast = None
        st.session_state.latest_close = 561.26  # Default to July 18, 2025, 4 PM EST close

initialize_session_state()

# Global variables
qqq_data = None
xgb_model = None
linear_model = None
poly_model = None
poly = None
available_features = None
latest_close = st.session_state.latest_close
prophet_model = None
shap_explainer = None

def get_market_adjusted_price(base_price):
    """Adjust price based on market session"""
    try:
        eastern = pytz.timezone('America/New_York')
        now = datetime.now(eastern)
        current_time = now.time()
    except (pytz.exceptions.UnknownTimeZoneError, Exception) as e:
        logger.error(f"Time zone or datetime error: {e}. Falling back to local time.")
        now = datetime.now()
        current_time = now.time()
    logger.debug(f"Current time: {current_time}, Type: {type(current_time)}")

    # Ensure current_time is a datetime.time object and handle comparison
    if not isinstance(current_time, time):
        logger.error(f"Invalid time object: {current_time}. Using default adjustment.")
        return base_price  # Default to no adjustment if type is invalid

    # Use separate conditions for clarity
    if current_time >= time(4, 0) and current_time < time(9, 30):  # Pre-market
        return base_price * 1.01
    elif current_time >= time(9, 30) and current_time < time(16, 0):  # Market open
        return base_price
    elif current_time >= time(16, 0) and current_time < time(20, 0):  # Post-market
        return base_price * 0.99
    else:  # Outside trading hours
        return base_price

def fetch_data(ticker, start_date):
    """Fetch data with retries and fallbacks"""
    max_retries = 5
    retry_delay = 15
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    cache_file = f"cache/{ticker}_cache.csv"
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty and all(col in df.columns for col in expected_columns):
                return df
        except Exception:
            pass

    for i in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, progress=False, timeout=20)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(1)
                df.columns = [str(col).strip().title() for col in df.columns]
                missing_required = [col for col in expected_columns if col not in df.columns]
                if not missing_required:
                    if 'Adj Close' not in df.columns:
                        df['Adj Close'] = df['Close']
                    try:
                        df.to_csv(cache_file)
                    except Exception:
                        pass
                    return df
            ttime.sleep(retry_delay)
        except Exception:
            ttime.sleep(retry_delay)

    # Fallback to synthetic data with updated latest close
    dates = pd.date_range(start=start_date, end=datetime.today())
    df = pd.DataFrame({
        'Open': np.linspace(100, latest_close, len(dates)),
        'High': np.linspace(105, latest_close * 1.01, len(dates)),
        'Low': np.linspace(95, latest_close * 0.99, len(dates)),
        'Close': np.linspace(100, latest_close, len(dates)),
        'Volume': np.linspace(1000000, 5000000, len(dates)),
        'Adj Close': np.linspace(100, latest_close, len(dates))
    }, index=dates)
    return df

def normalize_columns(df):
    """Normalize column names to title case"""
    if df is not None and hasattr(df, 'columns'):
        df.columns = [str(col).strip().title() for col in df.columns]
    return df

def safe_feature_subset(df, feature_list):
    """Safely select features from DataFrame"""
    if df is None or feature_list is None:
        return pd.DataFrame()
    if not hasattr(df, 'columns'):
        return pd.DataFrame()
    existing = [f for f in feature_list if f in df.columns]
    if not existing:
        return pd.DataFrame()
    return df[existing].copy()

def add_technical_indicators(df):
    """Add technical indicators to DataFrame"""
    df = df.copy()
    required_cols = ['Close', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = latest_close

    try:
        for span in [9, 20, 50, 200]:
            df[f'Ema_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['Typical_Price'] = typical_price
        df['Price_X_Volume'] = typical_price * df['Volume']
        df['Cumulative_Price_Volume'] = df['Price_X_Volume'].cumsum()
        df['Cumulative_Volume'] = df['Volume'].cumsum()
        df['Vwap'] = df['Cumulative_Price_Volume'] / df['Cumulative_Volume']
        
        df['Kc_Middle'] = df['Close'].ewm(span=20, adjust=False).mean()
        tr = pd.DataFrame()
        tr['h_l'] = df['High'] - df['Low']
        tr['h_pc'] = abs(df['High'] - df['Close'].shift())
        tr['l_pc'] = abs(df['Low'] - df['Close'].shift())
        df['Tr'] = tr.max(axis=1)
        df['Atr'] = df['Tr'].ewm(span=20, adjust=False).mean()
        k_mult = 2
        df['Kc_Upper'] = df['Kc_Middle'] + k_mult * df['Atr']
        df['Kc_Lower'] = df['Kc_Middle'] - k_mult * df['Atr']
        
        cols_to_drop = ['Typical_Price', 'Price_X_Volume', 'Cumulative_Price_Volume', 
                        'Cumulative_Volume', 'Tr', 'Atr']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        return df
    except Exception:
        return df

def train_model(X, y):
    """Train XGBoost model"""
    try:
        model_xgb = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model_xgb.fit(X, y)
        return model_xgb
    except Exception:
        return None

def get_economic_calendar():
    """Fetch upcoming economic events"""
    try:
        if fred is None:
            return pd.DataFrame({
                'Date': [datetime.today().strftime('%Y-%m-%d')],
                'Event': ['FRED API not configured'],
                'Importance': ['Low']
            })
            
        releases = fred.releases()
        upcoming = pd.DataFrame()
        today = datetime.today()
        future_date = today + timedelta(days=30)
        important_series = ['GDP', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'PPIACO']
        events = []
        
        for series_id in important_series:
            try:
                info = fred.series_info(series_id)
                next_release = fred.series_next_release(series_id)
                if next_release and today <= next_release <= future_date:
                    events.append({
                        'Date': next_release.strftime('%Y-%m-%d'),
                        'Event': info['title'],
                        'Importance': 'High'
                    })
            except:
                continue
        
        return pd.DataFrame(events)
    except Exception as e:
        st.error(f"Economic calendar failed: {str(e)}")
        return pd.DataFrame({
            'Date': [datetime.today().strftime('%Y-%m-%d')],
            'Event': ['No economic data available'],
            'Importance': ['Low']
        })

def portfolio_impact(forecast_prices, portfolio):
    """Calculate portfolio impact based on forecast"""
    try:
        if not portfolio or forecast_prices is None or len(forecast_prices) == 0:
            return None
            
        results = []
        current_total = 0.0
        forecast_total = 0.0
        
        for ticker, qty in portfolio.items():
            try:
                current_data = yf.download(ticker, period="1d", progress=False)
                current_price = float(current_data['Close'].iloc[-1]) if not current_data.empty and 'Close' in current_data.columns else 0.0
                qqq_current = latest_close
                qqq_forecast = float(forecast_prices[-1]) if isinstance(forecast_prices, (list, np.ndarray)) else forecast_prices
                price_change_pct = (qqq_forecast - qqq_current) / qqq_current if qqq_current != 0 else 0.0
                forecast_price = current_price * (1 + price_change_pct) if current_price != 0 else 0.0
                current_val = current_price * qty
                forecast_val = forecast_price * qty
                
                results.append({
                    'Ticker': ticker,
                    'Quantity': qty,
                    'Current Price': current_price,
                    'Current Value': current_val,
                    'Forecast Price': forecast_price,
                    'Forecast Value': forecast_val,
                    'Change': forecast_val - current_val,
                    'Change %': ((forecast_val - current_val) / current_val * 100) if current_val != 0 else 0.0
                })
                
                current_total += current_val
                forecast_total += forecast_val
            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")
                results.append({
                    'Ticker': ticker,
                    'Quantity': qty,
                    'Current Price': 0.0,
                    'Current Value': 0.0,
                    'Forecast Price': 0.0,
                    'Forecast Value': 0.0,
                    'Change': 0.0,
                    'Change %': 0.0
                })
        
        if not results:
            return None
        return {
            'holdings': pd.DataFrame(results),
            'current_total': current_total,
            'forecast_total': forecast_total,
            'total_change': forecast_total - current_total,
            'total_change_pct': ((forecast_total - current_total) / current_total * 100) if current_total != 0 else 0.0
        }
    except Exception as e:
        st.error(f"Portfolio calculation failed: {str(e)}")
        return None

def generate_shap_plot(model, X):
    """Generate SHAP explainer plot"""
    try:
        if shap_explainer is None:
            return None
        shap_values = shap_explainer.shap_values(X)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"SHAP plot failed: {str(e)}")
        return None

@st.cache_resource
def load_data_and_models():
    """Load data and train models"""
    try:
        start_date = "2018-01-01"
        qqq = fetch_data("QQQ", start_date)
        qqq = normalize_columns(qqq)
        
        def get_data_with_fallback(ticker, fallback_value):
            data = fetch_data(ticker, start_date)
            data = normalize_columns(data)
            if data is None or data.empty or 'Close' not in data.columns:
                return pd.Series(fallback_value, index=qqq.index, name='Close')
            return data['Close'].squeeze().reindex(qqq.index, method='ffill').ffill().bfill()
        
        vix = get_data_with_fallback("^VIX", st.session_state.vix)
        treasury10 = get_data_with_fallback("^TNX", st.session_state.yield_10)
        treasury2 = get_data_with_fallback("^IRX", st.session_state.yield_2)
        qqq = add_technical_indicators(qqq)
        
        qqq['Date_Ordinal'] = qqq.index.map(datetime.toordinal)
        qqq['Fedfunds'] = st.session_state.fed
        qqq['Unemployment'] = st.session_state.unemp
        qqq['Cpi'] = st.session_state.cpi
        qqq['Gdp'] = st.session_state.gdp
        qqq['Vix'] = vix
        qqq['10Y_Yield'] = treasury10
        qqq['2Y_Yield'] = treasury2
        qqq['Yield_Spread'] = treasury10 - treasury2
        qqq['Eps_Growth'] = st.session_state.eps
        qqq['Sentiment'] = st.session_state.sent
        qqq['Ma_20'] = qqq['Close'].rolling(window=20).mean()
        qqq['Ma_50'] = qqq['Close'].rolling(window=50).mean()
        qqq['Volatility'] = qqq['Close'].rolling(window=20).std()

        essential_features = ['Date_Ordinal', 'Fedfunds', 'Unemployment', 'Cpi', 'Gdp', 'Vix',
                              '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'Eps_Growth', 'Sentiment',
                              'Volatility', 'Ma_20', 'Ma_50']
        technical_features = ['Ema_9', 'Ema_20', 'Ema_50', 'Ema_200', 'Vwap', 'Kc_Upper', 'Kc_Lower', 'Kc_Middle']
        available_features = essential_features + [col for col in technical_features if col in qqq.columns]
        available_features = [col for col in available_features if col in qqq.columns]

        X = safe_feature_subset(qqq, available_features)
        if X.empty:
            X = qqq[['Close']].copy()
            available_features = ['Close']
        
        y = qqq['Close'].loc[X.index]
        X = X.dropna()
        y = y.loc[X.index]

        model_xgb = train_model(X, y)
        model_linear = LinearRegression().fit(X, y)
        poly_transformer = None
        model_poly = model_linear
        if len(available_features) < 20:
            poly_transformer = PolynomialFeatures(degree=2)
            poly_features = poly_transformer.fit_transform(X)
            model_poly = LinearRegression().fit(poly_features, y)

        # Prophet model training
        prophet_model = None
        try:
            prophet_df = qqq.reset_index()[['index', 'Close']].rename(columns={'index': 'ds', 'Close': 'y'})
            prophet_model = Prophet(daily_seasonality=False)
            prophet_model.fit(prophet_df)
        except Exception as e:
            st.error(f"Prophet model failed: {str(e)}")

        shap_explainer = None
        try:
            if model_xgb is not None:
                shap_explainer = shap.TreeExplainer(model_xgb)
        except:
            shap_explainer = None

        return qqq, model_xgb, model_linear, model_poly, poly_transformer, available_features, prophet_model, shap_explainer
    except Exception:
        return None, None, None, None, None, None, None, None

# Load data and models
with st.spinner("Loading data and training models. This may take up to 1 minute..."):
    qqq_data, xgb_model, linear_model, poly_model, poly, available_features, prophet_model, shap_explainer = load_data_and_models()

# Handle failed data loading
if any(x is None for x in [qqq_data, xgb_model, linear_model, poly_model, available_features]):
    dates = pd.date_range(start="2018-01-01", end=datetime.today())
    qqq_data = pd.DataFrame({
        'Open': np.linspace(100, latest_close, len(dates)),
        'High': np.linspace(105, latest_close * 1.01, len(dates)),
        'Low': np.linspace(95, latest_close * 0.99, len(dates)),
        'Close': np.linspace(100, latest_close, len(dates)),
        'Volume': np.linspace(1000000, 5000000, len(dates)),
        'Adj Close': np.linspace(100, latest_close, len(dates))
    }, index=dates)
    
    qqq_data = add_technical_indicators(qqq_data)
    qqq_data['Date_Ordinal'] = qqq_data.index.map(datetime.toordinal)
    qqq_data['Fedfunds'] = st.session_state.fed
    qqq_data['Unemployment'] = st.session_state.unemp
    qqq_data['Cpi'] = st.session_state.cpi
    qqq_data['Gdp'] = st.session_state.gdp
    qqq_data['Vix'] = st.session_state.vix
    qqq_data['10Y_Yield'] = st.session_state.yield_10
    qqq_data['2Y_Yield'] = st.session_state.yield_2
    qqq_data['Yield_Spread'] = qqq_data['10Y_Yield'] - qqq_data['2Y_Yield']
    qqq_data['Eps_Growth'] = st.session_state.eps
    qqq_data['Sentiment'] = st.session_state.sent
    qqq_data['Volatility'] = qqq_data['Close'].rolling(window=20).std().fillna(0)
    qqq_data['Ma_20'] = qqq_data['Close'].rolling(window=20).mean()
    qqq_data['Ma_50'] = qqq_data['Close'].rolling(window=50).mean()
    
    available_features = ['Date_Ordinal', 'Fedfunds', 'Unemployment', 'Cpi', 'Gdp', 'Vix',
                         '10Y_Yield', '2Y_Yield', 'Yield_Spread', 'Eps_Growth', 'Sentiment',
                         'Volatility', 'Ma_20', 'Ma_50']
    available_features = [col for col in available_features if col in qqq_data.columns]
    
    X = safe_feature_subset(qqq_data, available_features)
    y = qqq_data['Close']
    xgb_model = LinearRegression().fit(X, y)
    linear_model = LinearRegression().fit(X, y)
    poly_model = LinearRegression().fit(X, y)
    poly = None
    prophet_model = None
    shap_explainer = None

# Get latest close price with improved accuracy
try:
    if qqq_data is not None and 'Close' in qqq_data.columns and not qqq_data['Close'].empty:
        latest_close = float(qqq_data['Close'].iloc[-1])
        if np.isnan(latest_close):
            latest_close = fetch_yfinance_indicator("QQQ", 561.26)
    else:
        latest_close = fetch_yfinance_indicator("QQQ", 561.26)
except Exception:
    latest_close = 561.26
latest_close = get_market_adjusted_price(latest_close)  # Adjust for current session
st.session_state.latest_close = latest_close  # Store for UI display

# UI
st.title("📈 QQQ Forecast Simulator")

# Tab-based navigation
tabs = ["Forecast", "Portfolio Impact", "Economic Calendar", "Model Explainability"]
with st.sidebar:
    st.session_state.active_tab = option_menu(
        "Navigation", 
        tabs,
        icons=['graph-up', 'wallet', 'calendar', 'lightbulb'],
        default_index=tabs.index(st.session_state.active_tab)
    )

# Save/Load scenarios
with st.sidebar.expander("💾 Scenario Manager"):
    scenario_name = st.text_input("Save as:", key="scenario_name")
    if st.button("Save Current Scenario"):
        st.session_state.saved_scenarios[scenario_name] = {
            'fed': st.session_state.fed,
            'cpi': st.session_state.cpi,
            'unemp': st.session_state.unemp,
            'gdp': st.session_state.gdp,
            'vix': st.session_state.vix,
            'yield_10': st.session_state.yield_10,
            'yield_2': st.session_state.yield_2,
            'eps': st.session_state.eps,
            'sent': st.session_state.sent
        }
        st.success(f"Scenario '{scenario_name}' saved!")
    
    saved_names = list(st.session_state.saved_scenarios.keys())
    selected_scenario = st.selectbox("Load Scenario:", [""] + saved_names)
    if selected_scenario:
        scenario_data = st.session_state.saved_scenarios[selected_scenario]
        st.session_state.fed = scenario_data['fed']
        st.session_state.cpi = scenario_data['cpi']
        st.session_state.unemp = scenario_data['unemp']
        st.session_state.gdp = scenario_data['gdp']
        st.session_state.vix = scenario_data['vix']
        st.session_state.yield_10 = scenario_data['yield_10']
        st.session_state.yield_2 = scenario_data['yield_2']
        st.session_state.eps = scenario_data['eps']
        st.session_state.sent = scenario_data['sent']
        st.success(f"Scenario '{selected_scenario}' loaded!")

# Model selection
with st.sidebar:
    st.subheader("Model Selection")
    model_options = ["XGBoost (Default)", "Prophet", "Ensemble Average"]
    selected_model = st.radio("Forecasting Model", model_options, index=0)

# Main forecasting UI
if st.session_state.active_tab == "Forecast":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        checkbox_label = f"📡 Use Live QQQ Close (${latest_close:.2f}) as Starting Point"
        use_live_price = st.checkbox(checkbox_label, value=True)
        horizon = st.selectbox("📆 Forecast Horizon (days)", [30, 60, 90], index=0)
        show_tech = st.checkbox("📊 Show Technical Indicators")
    
    with col2:
        macro_bias = st.slider("🧠 Macro News Sentiment Overlay (-10% to +10%)", -0.10, 0.10, 0.0, step=0.01)
        st.caption("Adjusts forecast based on positive/negative news sentiment")

    scenarios = {
        "Recession": dict(fed=5.5, cpi=6.5, unemp=6.0, gdp=19000, vix=45.0, yield_10=2.5, yield_2=4.0, eps=5.0, sent=35),
        "Rate Cut": dict(fed=3.0, cpi=2.0, unemp=3.8, gdp=25000, vix=15.0, yield_10=3.0, yield_2=2.5, eps=12.0, sent=85),
        "Soft Landing": dict(fed=4.0, cpi=2.8, unemp=4.2, gdp=24000, vix=18.0, yield_10=3.8, yield_2=3.5, eps=10.0, sent=70)
    }

    scenario = st.radio("Select Scenario", ["Custom"] + list(scenarios.keys()), horizontal=True)
    
    sc_col1, sc_col2 = st.columns(2)
    
    with sc_col1:
        if scenario == "Custom":
            fed = st.number_input("Fed Funds Rate (%)", value=st.session_state.fed, step=0.1)
            cpi = st.number_input("CPI (%)", value=st.session_state.cpi, step=0.1)
            unemp = st.number_input("Unemployment (%)", value=st.session_state.unemp, step=0.1)
            gdp = st.number_input("GDP (Billions)", value=st.session_state.gdp, step=1000)
        else:
            s = scenarios[scenario]
            fed, cpi, unemp, gdp = s['fed'], s['cpi'], s['unemp'], s['gdp']
            st.info(f"Using {scenario} scenario parameters")
    
    with sc_col2:
        if scenario == "Custom":
            vix = st.number_input("VIX", value=st.session_state.vix, step=1.0)
            yield_10 = st.number_input("10Y Yield (%)", value=st.session_state.yield_10, step=0.1)
            yield_2 = st.number_input("2Y Yield (%)", value=st.session_state.yield_2, step=0.1)
            eps = st.number_input("EPS Growth (%)", value=st.session_state.eps, step=0.5)
            sent = st.slider("Sentiment (0-100)", 0, 100, int(st.session_state.sent))
        else:
            s = scenarios[scenario]
            vix, yield_10, yield_2, eps, sent = s['vix'], s['yield_10'], s['yield_2'], s['eps'], s['sent']

    if scenario == "Custom":
        st.session_state.update({'fed': fed, 'cpi': cpi, 'unemp': unemp, 'gdp': gdp, 'vix': vix,
                                'yield_10': yield_10, 'yield_2': yield_2, 'eps': eps, 'sent': sent})

    future_dates = pd.date_range(start=datetime.today(), periods=horizon, freq='D')
    date_ordinals = [d.toordinal() for d in future_dates]

    future_df = pd.DataFrame({
        'Date_Ordinal': date_ordinals,
        'Fedfunds': fed,
        'Unemployment': unemp,
        'Cpi': cpi,
        'Gdp': gdp,
        'Vix': vix,
        '10Y_Yield': yield_10,
        '2Y_Yield': yield_2,
        'Yield_Spread': yield_10 - yield_2,
        'Eps_Growth': eps,
        'Sentiment': sent
    }, index=future_dates)

    for col in ['Ema_9', 'Ema_20', 'Ema_50', 'Ema_200', 'Vwap', 'Kc_Upper', 'Kc_Lower', 'Kc_Middle', 'Volatility', 'Ma_20', 'Ma_50']:
        if col in qqq_data.columns:
            future_df[col] = qqq_data[col].iloc[-1] if not qqq_data.empty else latest_close
        else:
            future_df[col] = latest_close

    future_df = safe_feature_subset(future_df, available_features)

    forecasts = {}
    if selected_model in ["XGBoost (Default)", "Ensemble Average"]:
        xgb_forecast = xgb_model.predict(future_df)
        if poly is not None:
            future_poly = poly.transform(future_df)
            forecast_poly = poly_model.predict(future_poly)
            xgb_forecast = (xgb_forecast + forecast_poly) / 2
        else:
            forecast_linear = linear_model.predict(future_df)
            xgb_forecast = (xgb_forecast + forecast_linear) / 2
        forecasts["XGBoost"] = xgb_forecast.ravel()
    
    if selected_model in ["Prophet", "Ensemble Average"] and prophet_model is not None:
        try:
            prophet_future = prophet_model.make_future_dataframe(periods=horizon)
            prophet_forecast = prophet_model.predict(prophet_future)
            prophet_forecast = prophet_forecast.set_index('ds')['yhat'][-horizon:]
            forecasts["Prophet"] = prophet_forecast.values
        except Exception as e:
            st.error(f"Prophet forecast failed: {str(e)}")
    
    if selected_model == "Ensemble Average" and len(forecasts) > 1:
        forecast = np.mean(list(forecasts.values()), axis=0)
    elif selected_model == "Prophet" and "Prophet" in forecasts:
        forecast = forecasts["Prophet"]
    else:
        forecast = forecasts.get("XGBoost", np.full(horizon, latest_close))
    
    forecast = forecast * (1 + macro_bias)
    if use_live_price:
        shift = latest_close - forecast[0]
        forecast += shift
    
    # Store forecast for portfolio impact
    st.session_state.forecast = forecast

    confidence_std = 0.03 * forecast
    forecast_upper = forecast + confidence_std
    forecast_lower = forecast - confidence_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qqq_data.index, y=qqq_data['Close'], name="Historical QQQ", line=dict(color='black')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_upper, name="Upper Bound", line=dict(color='lightblue'), showlegend=False))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_lower, name="Lower Bound", fill='tonexty', line=dict(color='lightblue'), fillcolor='rgba(173, 216, 230, 0.3)', showlegend=False))

    if selected_model == "Ensemble Average":
        for model_name, values in forecasts.items():
            if use_live_price:
                shift = latest_close - values[0]
                values += shift
            fig.add_trace(go.Scatter(x=future_dates, y=values, name=model_name, line=dict(dash='dash')))
    
    if show_tech:
        tech_cols = ['Ema_9', 'Ema_20', 'Ema_50', 'Ema_200', 'Vwap', 'Kc_Upper', 'Kc_Lower', 'Kc_Middle', 'Volatility']
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
    
    if selected_model == "Ensemble Average" and len(forecasts) > 1:
        st.subheader("Model Comparison")
        model_fig = go.Figure()
        for model_name, values in forecasts.items():
            if use_live_price:
                shift = latest_close - values[0]
                values += shift
            model_fig.add_trace(go.Scatter(x=future_dates, y=values, name=model_name))
        model_fig.update_layout(title="Model Forecast Comparison", template="plotly_white")
        st.plotly_chart(model_fig, use_container_width=True)

    st.subheader("📈 CPI Sensitivity Analysis")
    cpi_range = np.linspace(cpi - 1, cpi + 1, 5)
    sens_fig = go.Figure()
    for val in cpi_range:
        temp_df = future_df.copy()
        temp_df['Cpi'] = val
        temp_df = safe_feature_subset(temp_df, available_features)
        
        temp_pred = xgb_model.predict(temp_df)
        if poly is not None:
            temp_poly = poly.transform(temp_df)
            temp_pred_poly = poly_model.predict(temp_poly)
            temp_pred = (temp_pred + temp_pred_poly) / 2
        else:
            temp_pred_linear = linear_model.predict(temp_df)
            temp_pred = (temp_pred + temp_pred_linear) / 2
            
        temp_pred = temp_pred.ravel() * (1 + macro_bias)
        if use_live_price:
            temp_pred += (latest_close - temp_pred[0])
        sens_fig.add_trace(go.Scatter(x=future_dates, y=temp_pred, name=f"CPI={val:.1f}"))
    sens_fig.update_layout(title="QQQ Forecast Sensitivity to CPI", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
    st.plotly_chart(sens_fig, use_container_width=True)

    st.subheader("📊 Multi-Scenario Forecast Comparison")
    if st.checkbox("Compare Scenarios"):
        comp_fig = go.Figure()
        for name, s in scenarios.items():
            comp_df = pd.DataFrame({
                'Date_Ordinal': date_ordinals,
                'Fedfunds': s['fed'],
                'Unemployment': s['unemp'],
                'Cpi': s['cpi'],
                'Gdp': s['gdp'],
                'Vix': s['vix'],
                '10Y_Yield': s['yield_10'],
                '2Y_Yield': s['yield_2'],
                'Yield_Spread': s['yield_10'] - s['yield_2'],
                'Eps_Growth': s['eps'],
                'Sentiment': s['sent']
            }, index=future_dates)
            
            for col in ['Ema_9', 'Ema_20', 'Ema_50', 'Ema_200', 'Vwap', 'Kc_Upper', 'Kc_Lower', 'Kc_Middle', 'Volatility', 'Ma_20', 'Ma_50']:
                if col in qqq_data.columns:
                    comp_df[col] = qqq_data[col].iloc[-1] if not qqq_data.empty else latest_close
                else:
                    comp_df[col] = latest_close
            
            comp_df = safe_feature_subset(comp_df, available_features)
            yhat = xgb_model.predict(comp_df)
            if poly is not None:
                yhat_poly = poly_model.predict(poly.transform(comp_df))
                yhat = (yhat + yhat_poly) / 2
            else:
                yhat_linear = linear_model.predict(comp_df)
                yhat = (yhat + yhat_linear) / 2
            yhat = yhat.ravel() * (1 + macro_bias)
            if use_live_price:
                yhat += (latest_close - yhat[0])
            comp_fig.add_trace(go.Scatter(x=future_dates, y=yhat, name=name))
        comp_fig.update_layout(title="Scenario Forecast Comparison", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
        st.plotly_chart(comp_fig, use_container_width=True)

    st.subheader("🔁 Backtest Model Accuracy")
    if st.checkbox("Enable Backtest"):
        backtest_date = st.date_input("Start Backtest From", value=datetime(2023, 1, 1))
        back_df = qqq_data.loc[qqq_data.index >= pd.to_datetime(backtest_date)].copy()
        if not back_df.empty:
            back_df = safe_feature_subset(back_df, available_features)
            back_df['Prediction'] = xgb_model.predict(back_df)
            if poly is not None:
                back_df['Prediction_Poly'] = poly_model.predict(poly.transform(back_df[available_features]))
            else:
                back_df['Prediction_Poly'] = linear_model.predict(back_df[available_features])
            back_df['Prediction'] = (back_df['Prediction'] + back_df['Prediction_Poly']) / 2
            mae = mean_absolute_error(qqq_data.loc[back_df.index, 'Close'], back_df['Prediction'])
            rmse = np.sqrt(mean_squared_error(qqq_data.loc[back_df.index, 'Close'], back_df['Prediction']))
            st.write(f"**Backtest Metrics:** MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=back_df.index, y=qqq_data.loc[back_df.index, 'Close'], name="Actual", line=dict(color='black')))
            fig_bt.add_trace(go.Scatter(x=back_df.index, y=back_df['Prediction'], name="Predicted", line=dict(color='blue')))
            fig_bt.update_layout(title="Backtest: QQQ Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", template="plotly_white")
            st.plotly_chart(fig_bt, use_container_width=True)

    st.subheader("📌 Feature Importance (XGBoost)")
    if st.checkbox("Show Feature Importance") and hasattr(xgb_model, 'feature_importances_'):
        try:
            fig_imp, ax = plt.subplots(figsize=(10, 6))
            xgb.plot_importance(xgb_model, ax=ax, importance_type='weight')
            plt.tight_layout()
            st.pyplot(fig_imp)
        except Exception:
            pass

    st.subheader("📥 Download Forecast Data")
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast,
        'Upper_Bound': forecast_upper,
        'Lower_Bound': forecast_lower
    })
    csv = forecast_df.to_csv(index=False).encode()
    st.download_button("Download Forecast CSV", csv, file_name="qqq_forecast.csv")

    try:
        buf = io.BytesIO()
        fig.write_image(buf, format="png", engine="kaleido")
        st.download_button("Download Forecast Chart as PNG", buf.getvalue(), file_name="forecast_chart.png")
    except Exception:
        pass

elif st.session_state.active_tab == "Portfolio Impact":
    st.header("📊 Portfolio Impact Simulator")
    st.info("Enter your portfolio holdings to see the forecasted impact")
    
    portfolio_input = st.text_area(
        "Enter portfolio (Ticker:Quantity, one per line)", 
        "AAPL:10\nMSFT:5\nGOOG:5\nAMZN:3\nMETA:8"
    )
    
    portfolio = {}
    for line in portfolio_input.split('\n'):
        if ':' in line:
            parts = line.split(':')
            if len(parts) == 2:
                try:
                    ticker = parts[0].strip().upper()
                    quantity = float(parts[1].strip())
                    portfolio[ticker] = quantity
                except:
                    continue
    
    if not portfolio:
        st.warning("Please enter a valid portfolio")
    elif st.session_state.forecast is None:
        st.warning("Please generate a forecast first on the Forecast tab")
    else:
        result = portfolio_impact(st.session_state.forecast, portfolio)
        if result and not result['holdings'].empty:
            st.subheader("Portfolio Impact Summary")
            col1, col2, col3 = st.columns(3)
            current_total = float(result['current_total'])
            forecast_total = float(result['forecast_total'])
            total_change = float(result['total_change'])
            total_change_pct = float(result['total_change_pct'])
            
            col1.metric("Current Value", f"${current_total:,.2f}")
            col2.metric("Forecast Value", f"${forecast_total:,.2f}", 
                       delta=f"{total_change_pct:.2f}%")
            col3.metric("Projected Change", f"${total_change:,.2f}",
                       delta_color="inverse" if total_change < 0 else "normal")
            
            st.subheader("Holding Details")
            st.dataframe(result['holdings'].style.format({
                'Current Price': '${:,.2f}',
                'Current Value': '${:,.2f}',
                'Forecast Price': '${:,.2f}',
                'Forecast Value': '${:,.2f}',
                'Change': '${:,.2f}',
                'Change %': '{:.2f}%'
            }))
        else:
            st.error("Failed to calculate portfolio impact. Check portfolio input or forecast data.")

elif st.session_state.active_tab == "Economic Calendar":
    st.header("📅 Economic Calendar")
    st.info("Upcoming economic events that could impact QQQ")
    
    calendar = get_economic_calendar()
    if not calendar.empty:
        st.dataframe(calendar)
    else:
        st.warning("No upcoming economic events found")

elif st.session_state.active_tab == "Model Explainability":
    st.header("🔍 Model Explainability")
    st.info("Understand how different factors impact the forecast")
    
    if shap_explainer is None:
        st.warning("SHAP explainer not available")
    else:
        try:
            X_explain = safe_feature_subset(qqq_data, available_features).iloc[[-1]]
            if not X_explain.empty:
                fig = generate_shap_plot(xgb_model, X_explain)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("Failed to generate SHAP plot")
            else:
                st.warning("No data available for explanation")
        except Exception as e:
            st.error(f"Explainability failed: {str(e)}")
    
    st.subheader("Model Interpretation")
    st.markdown("""
    - **Fed Funds Rate**: Higher rates typically pressure tech stocks
    - **CPI**: High inflation may lead to Fed tightening
    - **VIX**: Fear index, higher values indicate market stress
    - **Yield Spread**: Inverted curve often signals recession
    - **Sentiment**: Bullish sentiment supports tech stocks
    - **Technical Indicators**: EMAs, VWAP show support/resistance levels
    """)
