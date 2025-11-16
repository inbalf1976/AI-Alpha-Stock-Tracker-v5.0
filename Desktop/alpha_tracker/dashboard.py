import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests
import plotly.graph_objs as go
import time
import threading
import os
import warnings
import joblib
import json
from pathlib import Path
# New Imports needed for the requested indicators and patterns
import ta
from scipy.signal import argrelextrema
# End New Imports
# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
# ================================
# 1. CONFIG & KEYS (SECURED)
# ================================
try:
    BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY") or os.getenv("ALPHA_VANTAGE_KEY")
except:
    BOT_TOKEN = None
    CHAT_ID = None
    ALPHA_VANTAGE_KEY = None
# ================================
# 2. ASSETS
# ================================
ASSET_CATEGORIES = {
    "Commodities": {
        "Crude Oil": "CL=F", "Brent Oil": "BZ=F", "Natural Gas": "NG=F",
        "Gold": "GC=F", "Silver": "SI=F", "Copper": "HG=F",
        "Corn": "ZC=F",
        "Wheat": "KE=F",
        "Soybeans": "ZS=F",
        "Coffee": "KC=F"
    },
    "Indices": {
        "S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "Russell 2000": "^RUT"
    },
    "Currencies": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X", "AUD/USD": "AUDUSD=X"
    },
    "Popular": {
        "Tesla": "TSLA", "NVIDIA": "NVDA", "Apple": "AAPL", "Microsoft": "MSFT",
        "Alphabet": "GOOGL", "Coinbase": "COIN", "Palantir": "PLTR"
    }
}
# ================================
# 3. DIRECTORIES & PATHS
# ================================
MODEL_DIR = Path("models")
SCALER_DIR = Path("scalers")
ACCURACY_DIR = Path("accuracy_logs")
METADATA_DIR = Path("metadata")
PREDICTIONS_DIR = Path("predictions")
CONFIG_DIR = Path("config")
for dir_path in [MODEL_DIR, SCALER_DIR, ACCURACY_DIR, METADATA_DIR, PREDICTIONS_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)
# ================================
# 4. PERSISTENT CONFIG PATHS
# ================================
DAEMON_CONFIG_PATH = CONFIG_DIR / "daemon_config.json"
MONITORING_CONFIG_PATH = CONFIG_DIR / "monitoring_config.json"
ALERT_HISTORY_PATH = CONFIG_DIR / "alert_history.json"
# ================================
# 5. SELF-LEARNING CONFIG
# ================================
LEARNING_CONFIG = {
    "accuracy_threshold": 0.08,
    "min_predictions_for_eval": 10,
    "retrain_interval_days": 30,
    "volatility_change_threshold": 0.5,
    "fine_tune_epochs": 5,
    "full_retrain_epochs": 25,
    "lookback_window": 60
}
# ================================
# 5.5 ALERT DEDUPLICATION CONFIG
# ================================
ALERT_CONFIG = {
    "cooldown_hours": 24,
    "magnitude_threshold": 3.0
}
# ================================
# 6. THREAD-SAFE LOCKS
# ================================
model_cache_lock = threading.Lock()
accuracy_lock = threading.Lock()
config_lock = threading.Lock()
alert_history_lock = threading.Lock()
# ================================
# 7. PERSISTENT CONFIG MANAGEMENT
# ================================
def load_daemon_config():
    if DAEMON_CONFIG_PATH.exists():
        try:
            with config_lock:
                with open(DAEMON_CONFIG_PATH, 'r') as f:
                    return json.load(f)
        except:
            pass
    return {"enabled": False, "last_started": None}

def save_daemon_config(enabled):
    config = {
        "enabled": enabled,
        "last_started": datetime.now().isoformat() if enabled else None
    }
    with config_lock:
        with open(DAEMON_CONFIG_PATH, 'w') as f:
            json.dump(config, f)

def load_monitoring_config():
    if MONITORING_CONFIG_PATH.exists():
        try:
            with config_lock:
                with open(MONITORING_CONFIG_PATH, 'r') as f:
                    return json.load(f)
        except:
            pass
    return {"enabled": False, "last_started": None}

def save_monitoring_config(enabled):
    config = {
        "enabled": enabled,
        "last_started": datetime.now().isoformat() if enabled else None
    }
    with config_lock:
        with open(MONITORING_CONFIG_PATH, 'w') as f:
            json.dump(config, f)
# ================================
# 7.5 ALERT HISTORY MANAGEMENT
# ================================
def load_alert_history():
    if ALERT_HISTORY_PATH.exists():
        try:
            with alert_history_lock:
                with open(ALERT_HISTORY_PATH, 'r') as f:
                    return json.load(f)
        except:
            pass
    return {}

def save_alert_history(history):
    with alert_history_lock:
        with open(ALERT_HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

def should_send_alert(asset, direction, confidence, history):
    if asset not in history:
        return True
    last_alert = history[asset]
    last_time = datetime.fromisoformat(last_alert["timestamp"])
    time_diff = (datetime.now() - last_time).total_seconds() / 3600
    if last_alert["direction"] != direction:
        return True
    last_confidence = last_alert.get("confidence", 0)
    magnitude_change = abs(confidence - last_confidence)
    if magnitude_change >= ALERT_CONFIG["magnitude_threshold"]:
        return True
    if time_diff >= ALERT_CONFIG["cooldown_hours"]:
        return True
    return False
# ================================
# 8. HELPER FUNCTIONS
# ================================
def get_safe_ticker_name(ticker):
    return ticker.replace('=', '_').replace('^', '').replace('/', '_')

def get_model_path(ticker):
    return MODEL_DIR / f"{get_safe_ticker_name(ticker)}_lstm.h5"

def get_scaler_path(ticker):
    return SCALER_DIR / f"{get_safe_ticker_name(ticker)}_scaler.pkl"

def get_accuracy_path(ticker):
    return ACCURACY_DIR / f"{get_safe_ticker_name(ticker)}_accuracy.json"

def get_metadata_path(ticker):
    return METADATA_DIR / f"{get_safe_ticker_name(ticker)}_meta.json"

def get_prediction_path(ticker, date):
    return PREDICTIONS_DIR / f"{get_safe_ticker_name(ticker)}_{date}.json"
# ================================
# 9. PRICE FETCHING
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if data.empty or len(data) == 0:
            return None
        price = float(data['Close'].iloc[-1])
        return round(price, 4) if ticker.endswith(("=F", "=X")) else round(price, 2)
    except:
        return None
# ================================
# 10. ACCURACY TRACKING SYSTEM
# ================================
def load_accuracy_log(ticker):
    path = get_accuracy_path(ticker)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "predictions": [],
        "errors": [],
        "dates": [],
        "avg_error": 0.0,
        "total_predictions": 0
    }

def save_accuracy_log(ticker, log_data):
    path = get_accuracy_path(ticker)
    with accuracy_lock:
        with open(path, 'w') as f:
            json.dump(log_data, f)

def record_prediction(ticker, predicted_price, prediction_date):
    pred_path = get_prediction_path(ticker, prediction_date)
    pred_data = {
        "ticker": ticker,
        "predicted_price": float(predicted_price),
        "prediction_date": prediction_date,
        "timestamp": datetime.now().isoformat()
    }
    with open(pred_path, 'w') as f:
        json.dump(pred_data, f)

def validate_predictions(ticker):
    accuracy_log = load_accuracy_log(ticker)
    updated = False
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    pred_path = get_prediction_path(ticker, yesterday)
    if pred_path.exists():
        try:
            with open(pred_path, 'r') as f:
                pred_data = json.load(f)
            actual_price = get_latest_price(ticker)
            if actual_price:
                predicted_price = pred_data["predicted_price"]
                error = abs(predicted_price - actual_price) / actual_price
                accuracy_log["predictions"].append(predicted_price)
                accuracy_log["errors"].append(error)
                accuracy_log["dates"].append(yesterday)
                accuracy_log["total_predictions"] += 1
                if len(accuracy_log["errors"]) > 50:
                    accuracy_log["predictions"] = accuracy_log["predictions"][-50:]
                    accuracy_log["errors"] = accuracy_log["errors"][-50:]
                    accuracy_log["dates"] = accuracy_log["dates"][-50:]
                accuracy_log["avg_error"] = np.mean(accuracy_log["errors"][-30:])
                save_accuracy_log(ticker, accuracy_log)
                updated = True
                pred_path.unlink()
        except Exception as e:
            pass
    return updated, accuracy_log
# ================================
# 11. MODEL METADATA SYSTEM
# ================================
def load_metadata(ticker):
    path = get_metadata_path(ticker)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "trained_date": None,
        "training_samples": 0,
        "training_volatility": 0.0,
        "version": 1,
        "retrain_count": 0,
        "last_accuracy": 0.0
    }

def save_metadata(ticker, metadata):
    path = get_metadata_path(ticker)
    with open(path, 'w') as f:
        json.dump(metadata, f)
# ================================
# 12. RETRAINING DECISION ENGINE
# ================================
def should_retrain(ticker, accuracy_log, metadata):
    reasons = []
    if not get_model_path(ticker).exists():
        reasons.append("No model exists")
        return True, reasons
    if len(accuracy_log["errors"]) >= LEARNING_CONFIG["min_predictions_for_eval"]:
        avg_error = accuracy_log["avg_error"]
        if avg_error > LEARNING_CONFIG["accuracy_threshold"]:
            reasons.append(f"Accuracy below threshold ({avg_error:.2%} error)")
            return True, reasons
    if metadata["trained_date"]:
        try:
            last_trained = datetime.fromisoformat(metadata["trained_date"])
            days_since = (datetime.now() - last_trained).days
            if days_since >= LEARNING_CONFIG["retrain_interval_days"]:
                reasons.append(f"Model is {days_since} days old")
                return True, reasons
        except:
            pass
    try:
        df = yf.download(ticker, period="30d", progress=False)
        if len(df) > 5:
            current_vol = df['Close'].pct_change().std()
            training_vol = metadata.get("training_volatility", 0)
            if training_vol > 0:
                vol_change = abs(current_vol - training_vol) / training_vol
                if vol_change > LEARNING_CONFIG["volatility_change_threshold"]:
                    reasons.append(f"Volatility changed {vol_change:.1%}")
                    return True, reasons
    except:
        pass
    return False, reasons
# ================================
# 13. LSTM MODEL BUILDER
# ================================
def build_lstm_model(n_features):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LEARNING_CONFIG["lookback_window"], n_features)),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
# ================================
# 13.5. CUP AND HANDLE DETECTION
# ================================
def detect_cup_and_handle(df, cup_depth_min=0.20, cup_depth_max=0.60, handle_retracement=0.30):
    df_recent = df['Close'].iloc[-200:].copy()
    prices = df_recent.values
    minima_idx = argrelextrema(prices, np.less)[0]
    maxima_idx = argrelextrema(prices, np.greater)[0]
    if len(maxima_idx) < 2 or len(minima_idx) < 1:
        return 0.0
    for i in range(len(maxima_idx) - 1):
        p1_idx = maxima_idx[i]
        p2_idx = maxima_idx[i+1]
        if p1_idx >= p2_idx:
            continue
        trough_indices = [idx for idx in minima_idx if p1_idx < idx < p2_idx]
        if not trough_indices:
            continue
        trough_idx = trough_indices[np.argmin(prices[trough_indices])]
        p1_price = prices[p1_idx]
        p2_price = prices[p2_idx]
        trough_price = prices[trough_idx]
        if abs(p1_price - p2_price) / ((p1_price + p2_price) / 2) > 0.10:
            continue
        peak_avg = (p1_price + p2_price) / 2
        cup_depth = (peak_avg - trough_price) / peak_avg
        if not (cup_depth_min <= cup_depth <= cup_depth_max):
            continue
        handle_start_idx = p2_idx
        handle_data = df_recent.iloc[handle_start_idx:handle_start_idx + 20]['Close']
        if len(handle_data) < 5:
            continue
        handle_peak = handle_data.max()
        handle_trough = handle_data.min()
        handle_retracement_check = (handle_peak - handle_trough) / peak_avg
        if (handle_retracement_check < handle_retracement) and (handle_trough > trough_price):
            return 1.0
    return 0.0
# ================================
# 13.6. OPTIMAL FEATURE ENGINEERING (FIXED)
# ================================
def engineer_optimal_features(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.ffill().bfill()
    close_series = df['Close'].copy()

    if close_series.dtype != np.float64:
        close_series = close_series.astype(np.float64)

    # === 1. TREND FEATURES ===
    sma_result = ta.trend.sma_indicator(close_series, window=50)
    df['SMA_50'] = sma_result.iloc[:, 0] if isinstance(sma_result, pd.DataFrame) else sma_result
    df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']

    # === 2. MOMENTUM FEATURES ===
    rsi_result = ta.momentum.rsi(close_series, window=14)
    df['RSI'] = rsi_result.iloc[:, 0] if isinstance(rsi_result, pd.DataFrame) else rsi_result

    macd_result = ta.trend.macd_diff(close_series, window_fast=12, window_slow=26, window_sign=9)
    df['MACD_Signal'] = macd_result.iloc[:, 0] if isinstance(macd_result, pd.DataFrame) else macd_result

    # === 3. VOLATILITY FEATURES ===
    atr_result = ta.volatility.average_true_range(
        df['High'].astype(np.float64),
        df['Low'].astype(np.float64),
        close_series, window=14
    )
    df['ATR'] = atr_result.iloc[:, 0] if isinstance(atr_result, pd.DataFrame) else atr_result
    df['ATR_Normalized'] = df['ATR'] / df['Close']

    bbl_result = ta.volatility.bollinger_lband(close_series, window=20, window_dev=2)
    df['BBL'] = bbl_result.iloc[:, 0] if isinstance(bbl_result, pd.DataFrame) else bbl_result

    bbu_result = ta.volatility.bollinger_hband(close_series, window=20, window_dev=2)
    df['BBU'] = bbu_result.iloc[:, 0] if isinstance(bbu_result, pd.DataFrame) else bbu_result

    df['Bollinger_Position'] = (df['Close'] - df['BBL']) / (df['BBU'] - df['BBL'])

    # === 4. VOLUME FEATURES ===
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    obv_result = ta.volume.on_balance_volume(close_series, df['Volume'].astype(np.float64))
    df['OBV'] = obv_result.iloc[:, 0] if isinstance(obv_result, pd.DataFrame) else obv_result
    df['OBV_Trend'] = df['OBV'].pct_change(periods=10)

    # === 5. PATTERN FEATURES ===
    df['CupHandle_Signal'] = df.apply(lambda x: detect_cup_and_handle(df), axis=1)

    # Define optimal feature set
    OPTIMAL_FEATURES = [
        'Close',
        'Price_vs_SMA50',
        'RSI',
        'MACD_Signal',
        'ATR_Normalized',
        'Bollinger_Position',
        'Volume_Ratio',
        'OBV_Trend',
        'CupHandle_Signal'
    ]

    df = df.drop(columns=['SMA_50', 'BBL', 'BBU', 'ATR', 'OBV', 'Volume_MA'], errors='ignore')
    df = df.dropna()
    df_features = df[OPTIMAL_FEATURES].copy()

    return df_features, len(OPTIMAL_FEATURES)
# ================================
# 14. SELF-LEARNING TRAINING SYSTEM
# ================================
def train_self_learning_model(ticker, days=5, force_retrain=False):
    model_path = get_model_path(ticker)
    scaler_path = get_scaler_path(ticker)
    updated, accuracy_log = validate_predictions(ticker)
    if updated:
        st.session_state.setdefault('learning_log', []).append(f"Validated prediction for {ticker}")
    metadata = load_metadata(ticker)
    needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
    if not needs_retrain and not force_retrain:
        training_type = "fine-tune"
    else:
        training_type = "full-retrain"
        if reasons:
            st.session_state.setdefault('learning_log', []).append(f"Retraining {ticker}: {', '.join(reasons)}")
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if len(df) < 200:
            return None, None, None
        df = df.reset_index(drop=True)
    except:
        return None, None, None

    df_features, N_FEATURES = engineer_optimal_features(df)
    if len(df_features) < LEARNING_CONFIG["lookback_window"]:
         return None, None, None

    if training_type == "full-retrain" or not scaler_path.exists():
        scaler = MinMaxScaler()
        scaler.fit(df_features)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)

    scaled = scaler.transform(df_features)
    X, y = [], []
    lookback = LEARNING_CONFIG["lookback_window"]
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        return None, None, None

    with model_cache_lock:
        if training_type == "full-retrain":
            model = build_lstm_model(N_FEATURES)
            epochs = LEARNING_CONFIG["full_retrain_epochs"]
            model.fit(X, y, epochs=epochs, batch_size=32, verbose=0,
                      validation_split=0.1,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
            metadata["retrain_count"] += 1
            st.session_state.setdefault('learning_log', []).append(
                f"Full retrain #{metadata['retrain_count']} for {ticker} completed on {N_FEATURES} features"
            )
        else:
            try:
                model = tf.keras.models.load_model(str(model_path))
                if model.input_shape[-1] != N_FEATURES:
                    raise ValueError("Model input shape mismatch")
                epochs = LEARNING_CONFIG["fine_tune_epochs"]
                recent_size = int(len(X) * 0.3)
                model.fit(X[-recent_size:], y[-recent_size:], epochs=epochs, batch_size=32, verbose=0)
                st.session_state.setdefault('learning_log', []).append(
                    f"Fine-tuned {ticker} on recent data (using {N_FEATURES} features)"
                )
            except:
                model = build_lstm_model(N_FEATURES)
                metadata["retrain_count"] += 1
                model.fit(X, y, epochs=LEARNING_CONFIG["full_retrain_epochs"],
                          batch_size=32, verbose=0, validation_split=0.1)
                st.session_state.setdefault('learning_log', []).append(
                    f"Model error. Full retrain #{metadata['retrain_count']} completed."
                )
        try:
            model.save(str(model_path))
        except:
            st.session_state.setdefault('errors', []).append(f"Model save failed {ticker}")
        metadata["trained_date"] = datetime.now().isoformat()
        metadata["training_samples"] = len(X)
        metadata["training_volatility"] = float(df['Close'].pct_change().std())
        metadata["version"] += 1
        metadata["last_accuracy"] = accuracy_log["avg_error"]
        save_metadata(ticker, metadata)

    last_scaled_features = scaled[-lookback:].reshape(1, lookback, N_FEATURES)
    preds = []
    for _ in range(days):
        pred = model.predict(last_scaled_features, verbose=0)
        preds.append(pred[0, 0])
        new_feature_vector = np.zeros((1, 1, N_FEATURES))
        new_feature_vector[0, 0, 0] = pred[0, 0]
        new_feature_vector[0, 0, 1:] = last_scaled_features[0, -1, 1:]
        last_scaled_features = np.append(last_scaled_features[:, 1:, :], new_feature_vector, axis=1)

    dummy_pred_array = np.zeros((len(preds), N_FEATURES))
    dummy_pred_array[:, 0] = np.array(preds).flatten()
    forecast = scaler.inverse_transform(dummy_pred_array)[:, 0]

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    record_prediction(ticker, forecast[0], tomorrow)

    dates = []
    i = 1
    while len(dates) < days:
        next_date = datetime.now().date() + timedelta(days=i)
        if next_date.weekday() < 5:
            dates.append(next_date)
        i += 1

    tf.keras.backend.clear_session()
    return forecast, dates, N_FEATURES
# ================================
# 15. DAILY RECOMMENDATION
# ================================
def daily_recommendation(ticker, asset):
    price = get_latest_price(ticker)
    if not price:
        return "<span style='color:orange'>Market closed or no data</span>"
    forecast, _, N_FEATURES = train_self_learning_model(ticker, 1)
    if forecast is None or len(forecast) == 0:
        return "<span style='color:orange'>Unable to generate forecast</span>"
    pred_price = round(forecast[0], 2)
    change = (pred_price - price) / price * 100
    action = "BUY" if change >= 1.5 else "SELL" if change <= -1.5 else "HOLD"
    color = "#00C853" if action == "BUY" else "#D50000" if action == "SELL" else "#FFA726"
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    learning_status = ""
    if accuracy_log["total_predictions"] > 0:
        learning_status = f"<p><small>Model Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | Features: {N_FEATURES} | Version: {metadata['version']}</small></p>"
    return f"""
    <div style="background:#1a1a1a;padding:20px;border-radius:12px;border-left:6px solid {color};color:#fff;margin:15px 0;">
    <h3 style="margin:0;color:{color};">{asset.upper()} — DAILY RECOMMENDATION</h3>
    <p><strong>Live:</strong> ${price:.2f} → <strong>AI Predicts:</strong> ${pred_price:.2f} ({change:+.2f}%)</p>
    <p><strong>Action:</strong> <span style="font-size:1.3em;color:{color};">{action}</span></p>
    {learning_status}
    </div>
    """
# ================================
# 16. 5-DAY FORECAST
# ================================
def show_5day_forecast(ticker, asset_name):
    forecast, dates, N_FEATURES = train_self_learning_model(ticker, days=5)
    if forecast is None:
        st.error("Failed to generate forecast.")
        return
    current_price = get_latest_price(ticker)
    if not current_price:
        current_price = forecast[0] * 0.99
    fig = go.Figure()
    try:
        hist = yf.download(ticker, period="30d", progress=False)['Close']
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', name='Historical', line=dict(color='#888')))
    except:
        pass
    fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines+markers', name='AI Forecast', line=dict(color='#00C853', width=3, dash='dot'), marker=dict(size=10)))
    fig.add_hline(y=current_price, line_dash="dash", line_color="#FFA726", annotation_text=f"Live: ${current_price:.2f}")
    fig.update_layout(title=f"{asset_name.upper()} — 5-Day Forecast", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        day1_change = (forecast[0] - current_price) / current_price * 100
        st.metric("Day 1", f"${forecast[0]:.2f}", f"{day1_change:+.2f}%")
    with col2:
        day3_change = (forecast[2] - current_price) / current_price * 100
        st.metric("Day 3", f"${forecast[2]:.2f}", f"{day3_change:+.2f}%")
    with col3:
        day5_change = (forecast[4] - current_price) / current_price * 100
        st.metric("Day 5", f"${forecast[4]:.2f}", f"{day5_change:+.2f}%")
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    if accuracy_log["total_predictions"] > 0:
        st.info(f"Model learns from {accuracy_log['total_predictions']} predictions | Accuracy: {(1 - accuracy_log['avg_error'])*100:.1f}% | Features: {N_FEATURES} | Version: {metadata['version']} | Retrains: {metadata['retrain_count']}")
# ================================
# 17. BACKGROUND LEARNING DAEMON
# ================================
def continuous_learning_daemon():
    while True:
        daemon_config = load_daemon_config()
        if not daemon_config.get("enabled", False):
            break
        try:
            all_tickers = [ticker for cat in ASSET_CATEGORIES.values() for _, ticker in cat.items()]
            for ticker in all_tickers:
                daemon_config = load_daemon_config()
                if not daemon_config.get("enabled", False):
                    break
                updated, accuracy_log = validate_predictions(ticker)
                if updated:
                    metadata = load_metadata(ticker)
                    needs_retrain, reasons = should_retrain(ticker, accuracy_log, metadata)
                    if needs_retrain:
                        st.session_state.setdefault('learning_log', []).append(f"Auto-retraining {ticker}: {', '.join(reasons)}")
                        train_self_learning_model(ticker, days=1, force_retrain=True)
            time.sleep(3600)
        except Exception as e:
            st.session_state.setdefault('errors', []).append(f"Daemon error: {str(e)[:50]}")
            time.sleep(600)
# ================================
# 18. 6%+ DETECTION
# ================================
@st.cache_data(ttl=60, show_spinner=False)
def detect_pre_move_6percent(ticker, name):
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False)
        if len(data) < 60:
            return None
        close = data['Close'].values
        volume = data['Volume'].values
        recent = close[-15:]
        momentum = (recent[-1] - recent[0]) / recent[0]
        vol_spike = volume[-1] / np.mean(volume[-15:-1]) if np.mean(volume[-15:-1]) > 0 else 1
        if (abs(momentum) > 0.015 and vol_spike > 2.5):
            direction = "UP" if momentum > 0 else "DOWN"
            confidence = min(98, int(60 + vol_spike * 8))
            return {"asset": name, "direction": direction, "confidence": confidence}
    except:
        pass
    return None

def send_telegram_alert(text):
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        response = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": text}, timeout=5)
        return response.status_code == 200
    except:
        return False

def monitor_6percent_pre_move():
    all_assets = {name: ticker for cat in ASSET_CATEGORIES.values() for name, ticker in cat.items()}
    while True:
        monitoring_config = load_monitoring_config()
        if not monitoring_config.get("enabled", False):
            break
        for name, ticker in all_assets.items():
            monitoring_config = load_monitoring_config()
            if not monitoring_config.get("enabled", False):
                break
            alert = detect_pre_move_6percent(ticker, name)
            if alert:
                history = load_alert_history()
                if should_send_alert(alert["asset"], alert["direction"], alert["confidence"], history):
                    text = f"6%+ MOVE INCOMING\n{alert['asset'].upper()} {alert['direction']}\nCONFIDENCE: {alert['confidence']}%"
                    if send_telegram_alert(text):
                        history[alert["asset"]] = {
                            "direction": alert["direction"],
                            "confidence": alert["confidence"],
                            "timestamp": datetime.now().isoformat()
                        }
                        save_alert_history(history)
                        st.session_state.setdefault('learning_log', []).append(f"Alert sent: {alert['asset']} {alert['direction']} ({alert['confidence']}%)")
                        time.sleep(2)
        time.sleep(60)
# ================================
# 19. AUTO-RESTART THREADS
# ================================
def initialize_background_threads():
    if "threads_initialized" not in st.session_state:
        st.session_state.threads_initialized = True
        daemon_config = load_daemon_config()
        if daemon_config.get("enabled", False):
            threading.Thread(target=continuous_learning_daemon, daemon=True).start()
            st.session_state.setdefault('learning_log', []).append("Learning Daemon auto-started")
        monitoring_config = load_monitoring_config()
        if monitoring_config.get("enabled", False):
            threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
            st.session_state.setdefault('learning_log', []).append("6%+ Monitoring auto-started")
# ================================
# 20. BRANDING
# ================================
def add_header():
    st.markdown("""
    <div style='text-align:center;padding:15px;background:#1a1a1a;color:#00C853;margin-bottom:20px;border-radius:8px;'>
        <h2 style='margin:0;'>AI - ALPHA STOCK TRACKER v4.2 (9 Optimal Features)</h2>
        <p style='margin:5px 0;'>Enhanced Accuracy • 9-Feature LSTM • Smart Alerts • Persistent 24/7</p>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <div style='text-align:center;padding:20px;background:#1a1a1a;color:#666;margin-top:40px;border-radius:8px;'>
        <p style='margin:0;'>© 2025 AI - Alpha Stock Tracker v4.2 | 9 Optimal Features</p>
    </div>
    """, unsafe_allow_html=True)
# ================================
# 21. MAIN APP
# ================================
st.set_page_config(page_title="AI - Alpha Stock Tracker v4.2", layout="wide")
if 'alert_history' not in st.session_state:
    st.session_state['alert_history'] = {}
initialize_background_threads()
add_header()
for key in ["learning_log", "errors"]:
    if key not in st.session_state:
        st.session_state[key] = []

with st.sidebar:
    st.header("Asset Selection")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]
    st.markdown("---")
    st.subheader("Self-Learning Status")
    accuracy_log = load_accuracy_log(ticker)
    metadata = load_metadata(ticker)
    if metadata["trained_date"]:
        trained = datetime.fromisoformat(metadata["trained_date"])
        st.metric("Last Trained", trained.strftime("%Y-%m-%d"))
        st.metric("Model Version", f"v{metadata['version']}")
        st.metric("Retrains", metadata["retrain_count"])
        try:
            scaler = joblib.load(get_scaler_path(ticker))
            N_FEATURES_USED = scaler.n_features_in_
        except:
            N_FEATURES_USED = "N/A"
        st.metric("Features Used", N_FEATURES_USED)
        if accuracy_log["total_predictions"] > 0:
            acc_pct = (1 - accuracy_log["avg_error"]) * 100
            st.metric("Accuracy", f"{acc_pct:.1f}%")
            st.metric("Predictions", accuracy_log["total_predictions"])
    else:
        st.info("No model trained yet")
    if st.button("Force Retrain", use_container_width=True):
        with st.spinner("Retraining..."):
            train_self_learning_model(ticker, days=1, force_retrain=True)
        st.success("Retrained!")
        st.rerun()
    st.markdown("---")
    st.subheader("Learning Daemon")
    daemon_config = load_daemon_config()
    daemon_status = "RUNNING" if daemon_config.get("enabled", False) else "STOPPED"
    st.markdown(f"**Status:** {daemon_status}")
    if daemon_config.get("last_started"):
        try:
            started = datetime.fromisoformat(daemon_config["last_started"])
            st.caption(f"Started: {started.strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", key="start_daemon", use_container_width=True):
            save_daemon_config(True)
            threading.Thread(target=continuous_learning_daemon, daemon=True).start()
            st.success("Started!")
            time.sleep(0.5)
            st.rerun()
    with col2:
        if st.button("Stop", key="stop_daemon", use_container_width=True):
            save_daemon_config(False)
            st.success("Stopped!")
            time.sleep(0.5)
            st.rerun()
    st.markdown("---")
    st.subheader("Smart Alert System")
    monitoring_config = load_monitoring_config()
    monitoring_status = "RUNNING" if monitoring_config.get("enabled", False) else "STOPPED"
    st.markdown(f"**Status:** {monitoring_status}")
    st.caption(f"Cooldown: {ALERT_CONFIG['cooldown_hours']}h")
    st.caption(f"Magnitude: {ALERT_CONFIG['magnitude_threshold']}%")
    if monitoring_config.get("last_started"):
        try:
            started = datetime.fromisoformat(monitoring_config["last_started"])
            st.caption(f"Started: {started.strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    alert_history = load_alert_history()
    if alert_history:
        st.markdown("**Recent Alerts:**")
        for asset, data in list(alert_history.items())[-5:]:
            try:
                alert_time = datetime.fromisoformat(data["timestamp"])
                st.caption(f"• {asset}: {data['direction']} ({data['confidence']}%) - {alert_time.strftime('%H:%M')}")
            except:
                pass
    if st.button("Test Telegram", key="test_telegram", use_container_width=True):
        success = send_telegram_alert("TEST ALERT\nAI - Alpha Stock Tracker v4.2\nSystem Active")
        if success:
            st.success("Sent!")
        else:
            st.error("Check keys")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Alerts", key="start_alerts", use_container_width=True):
            save_monitoring_config(True)
            threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
            st.success("Started!")
            time.sleep(0.5)
            st.rerun()
    with col2:
        if st.button("Stop Alerts", key="stop_alerts", use_container_width=True):
            save_monitoring_config(False)
            st.success("Stopped!")
            time.sleep(0.5)
            st.rerun()

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    price = get_latest_price(ticker)
    if price:
        st.markdown(f"<h2 style='text-align:center;'>LIVE: <code style='font-size:1.5em;background:#333;padding:8px 16px;border-radius:8px;'>${price:.2f}</code></h2>", unsafe_allow_html=True)
    else:
        st.warning("Market closed or no data")
    if st.button("Daily Recommendation", use_container_width=True):
        with st.spinner("Analyzing..."):
            st.markdown(daily_recommendation(ticker, asset), unsafe_allow_html=True)
    if st.button("5-Day Self-Learning Forecast", use_container_width=True):
        with st.spinner("Adapting model..."):
            show_5day_forecast(ticker, asset)
st.markdown("---")
st.subheader("Self-Learning Activity Log")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Recent Events:**")
    if st.session_state.learning_log:
        for log_entry in st.session_state.learning_log[-10:]:
            st.text(log_entry)
    else:
        st.info("No activity yet.")
with col2:
    st.markdown("**Errors:**")
    if st.session_state.errors:
        for error_entry in st.session_state.errors[-10:]:
            st.error(error_entry)
    else:
        st.info("No errors.")
add_footer()
