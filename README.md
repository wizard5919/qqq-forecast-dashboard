# QQQ Forecast Dashboard

This Streamlit app forecasts QQQ ETF prices using macroeconomic assumptions and technical indicators with an XGBoost model.

## Features
- Interactive macro variable sliders (Fed Funds, CPI, Unemployment, etc.)
- Technical indicators: EMA (9, 20, 50, 200), VWAP, Keltner Channels, Volatility
- Model performance metrics (R², MAE, RMSE)
- Downloadable forecast and technical indicators CSV
- Breakout detection (>$500)
- Scenario-based forecasting (Recession, Rate Cut, Soft Landing)
- Feature importance visualization
- Backtesting with performance metrics

## Usage
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run `streamlit run streamlit_app.py`
4. Or deploy on [Streamlit Cloud](https://share.streamlit.io)

## Requirements
See `requirements.txt` for dependencies, including `streamlit`, `pandas`, `yfinance`, `xgboost`, `scikit-learn`, and `plotly`.
