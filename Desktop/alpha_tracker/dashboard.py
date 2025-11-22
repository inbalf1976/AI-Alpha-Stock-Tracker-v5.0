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
import logging
from logging.handlers import RotatingFileHandler
from enum import Enum
from typing import Tuple, List
import socket

# ================================
# SUPPRESS WARNINGS
# ================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

try:
    from streamlit.runtime.scriptrunner import script_run_context
    if hasattr(script_run_context, '_LOGGER'):
        script_run_context._LOGGER.setLevel('ERROR')
except:
    pass

# ================================
# LOGGING SETUP
# ================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
ERROR_LOG_PATH = LOG_DIR / "error_tracking.json"

class ErrorSeverity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def reset_all_logs_on_startup():
    try:
        for f in ['app.log', 'errors.log']:
            p = LOG_DIR / f
            if p.exists():
                try:
                    p.unlink()
                except PermissionError:
                    pass
        if ERROR_LOG_PATH.exists():
            try:
                ERROR_LOG_PATH.unlink()
            except PermissionError:
                pass
        with open(ERROR_LOG_PATH, 'w') as f:
            json.dump([], f)
    except Exception as e:
        print(f"Could not fully reset logs: {e}")

reset_all_logs_on_startup()

def setup_logging():
    logger = logging.getLogger('stock_tracker')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    error_handler = RotatingFileHandler(LOG_DIR / 'errors.log', maxBytes=5*1024*1024, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.info("=== NEW SESSION STARTED - ALL LOGS RESET ===")
    return logger

logger = setup_logging()

def log_error(severity, function_name, error, ticker=None, user_message="An error occurred", show_to_user=True):
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity.value,
        "ticker": ticker or "N/A",
        "function": function_name,
        "error": str(error),
        "user_message": user_message
    }
    msg = f"{ticker or 'N/A'} | {user_message}: {error}"
    if severity == ErrorSeverity.ERROR:
        logger.error(msg, exc_info=True)
    elif severity == ErrorSeverity.CRITICAL:
        logger.critical(msg, exc_info=True)
    elif severity == ErrorSeverity.WARNING:
        logger.warning(msg)
    else:
        logger.info(msg)

    try:
        with open(ERROR_LOG_PATH, 'r') as f:
            history = json.load(f)
    except:
        history = []
    history.append(error_data)
    history = history[-500:]
    try:
        with open(ERROR_LOG_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass

    if show_to_user and 'st' in globals():
        if severity == ErrorSeverity.CRITICAL:
            st.error(f"{user_message}")
        elif severity == ErrorSeverity.ERROR:
            st.error(f"{user_message}")
        elif severity == ErrorSeverity.WARNING:
            st.warning(f"{user_message}")

# ================================
# ACCESS CONTROL
# ================================
def is_local_user():
    try:
        if os.getenv('STREAMLIT_SHARING_MODE') or os.getenv('STREAMLIT_SERVER_HEADLESS'):
            return False
        cloud = ['HEROKU', 'AWS', 'AZURE', 'GOOGLE_CLOUD', 'STREAMLIT_CLOUD', 'RENDER', 'RAILWAY']
        if any(os.getenv(x) for x in cloud):
            return False
        hn = socket.gethostname()
        return hn in ['localhost', '127.0.0.1'] or hn.startswith(('DESKTOP-', 'LAPTOP-'))
    except:
        return False

def require_local_access(func_name):
    if not is_local_user():
        st.error(f"Access Restricted: '{func_name}' is local-only.")
        st.info("Online deployment disabled for security.")
        logger.warning(f"Blocked online access to {func_name}")
        return False
    return True

# ================================
# CONFIG & KEYS
# ================================
try:
    BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    logger.info("Configuration loaded")
except:
    BOT_TOKEN = CHAT_ID = None

# ================================
# TELEGRAM ALERT (defined early!)
# ================================
def send_telegram_alert(text):
    if not BOT_TOKEN or not CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log_error(ErrorSeverity.WARNING, "send_telegram_alert", e, user_message="Telegram failed")
        return False

# ================================
# DIRECTORIES & ASSETS
# ================================
MODEL_DIR = Path("models")
SCALER_DIR = Path("scalers")
ACCURACY_DIR = Path("accuracy_logs")
METADATA_DIR = Path("metadata")
PREDICTIONS_DIR = Path("predictions")
CONFIG_DIR = Path("config")

for d in [MODEL_DIR, SCALER_DIR, ACCURACY_DIR, METADATA_DIR, PREDICTIONS_DIR, CONFIG_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

DAEMON_CONFIG_PATH = CONFIG_DIR / "daemon_config.json"
MONITORING_CONFIG_PATH = CONFIG_DIR / "monitoring_config.json"

ASSET_CATEGORIES = {
    "Tech Stocks": {"Apple": "AAPL", "Tesla": "TSLA", "NVIDIA": "NVDA", "Microsoft": "MSFT", "Alphabet": "GOOGL"},
    "High Growth": {"Palantir": "PLTR", "MicroStrategy": "MSTR", "Coinbase": "COIN"},
    "Commodities": {"Corn Futures": "ZC=F", "Gold Futures": "GC=F", "Crude Oil": "CL=F", "Wheat": "ZW=F"},
    "ETFs": {"S&P 500 ETF": "SPY", "WHEAT": "WEAT"}
}

# ================================
# PRICE FETCHING
# ================================
PRICE_RANGES = { ... }  # (same as before)

def validate_price(ticker, price):
    if price is None or price <= 0: return False
    if ticker in PRICE_RANGES:
        mn, mx = PRICE_RANGES[ticker]
        if not (mn <= price <= mx):
            return False
    return True

@st.cache_data(ttl=60, show_spinner=False)
def get_latest_price(ticker):
    time.sleep(0.3)
    for interval, period in [("1m", "1d"), ("5m", "1d"), ("1d", "5d")]:
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                if validate_price(ticker, price):
                    return round(price, 2)
        except:
            continue
    try:
        info = yf.Ticker(ticker).info
        for k in ["regularMarketPrice", "currentPrice", "previousClose"]:
            if k in info and info[k]:
                p = float(info[k])
                if validate_price(ticker, p):
                    return round(p, 2)
    except:
        pass
    return None

# ================================
# FILE HELPERS
# ================================
def get_safe_ticker_name(t): return t.replace('=', '_').replace('^', '').replace('/', '_')
def get_model_path(t): return MODEL_DIR / f"{get_safe_ticker_name(t)}_lstm.h5"
def get_scaler_path(t): return SCALER_DIR / f"{get_safe_ticker_name(t)}_scaler.pkl"
def get_accuracy_path(t): return ACCURACY_DIR / f"{get_safe_ticker_name(t)}_accuracy.json"
def get_metadata_path(t): return METADATA_DIR / f"{get_safe_ticker_name(t)}_meta.json"
def get_prediction_path(t, d): return PREDICTIONS_DIR / f"{get_safe_ticker_name(t)}_{d}.json"

# ================================
# ACCURACY & METADATA
# ================================
def load_accuracy_log(ticker):
    p = get_accuracy_path(ticker)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except:
            pass
    return {"predictions": [], "errors": [], "dates": [], "avg_error": 0.0, "total_predictions": 0}

def save_accuracy_log(ticker, data):
    try:
        with open(get_accuracy_path(ticker), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass

def load_metadata(ticker):
    p = get_metadata_path(ticker)
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except:
            pass
    return {"trained_date": None, "training_samples": 0, "training_volatility": 0.0, "version": 1, "retrain_count": 0}

def save_metadata(ticker, meta):
    try:
        with open(get_metadata_path(ticker), 'w') as f:
            json.dump(meta, f, indent=2)
    except:
        pass

# ================================
# MODEL TRAINING (FULL FUNCTION)
# ================================
def build_lstm_model():
    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(30),
        Dropout(0.2),
        Dense(15),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_self_learning_model(ticker, days=5, force_retrain=False):
    # ... [Full original function — unchanged, just moved up]
    # (All the training logic you had — exactly the same)
    # Returns forecast, dates, model
    pass  # ← Replace with your full original function when pasting

# ================================
# CONFIDENCE CHECKS
# ================================
def high_confidence_checklist(ticker: str, forecast: list, current_price: float) -> tuple:
    # ... your full function
    pass

def ultra_confidence_shield(ticker: str, forecast: List[float], current_price: float) -> Tuple[bool, List[str]]:
    # ... your full function
    pass

# ================================
# RECOMMENDATIONS & FORECAST
# ================================
def daily_recommendation(ticker, asset):
    # ... your full function
    pass

def show_5day_forecast(ticker, asset_name):
    # ... your full function
    pass

# ================================
# 6%+ PREDICTIVE DETECTION
# ================================
def detect_pre_move_6percent(ticker, name):
    # ... your full function
    pass

def monitor_6percent_pre_move():
    # ... your full function
    pass

# ================================
# DAEMON & WATCHDOG
# ================================
def continuous_learning_daemon():
    # ... your full function
    pass

def thread_watchdog():
    # ... your full function
    pass

def initialize_background_threads():
    if "threads_initialized" not in st.session_state:
        st.session_state.threads_initialized = True
        st.session_state.setdefault('learning_log', [])
        st.session_state.setdefault('alert_history', {})
        if load_daemon_config().get("enabled", False):
            threading.Thread(target=continuous_learning_daemon, daemon=True).start()
        if load_monitoring_config().get("enabled", False):
            threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
        if any(load_daemon_config().get("enabled", False), load_monitoring_config().get("enabled", False)):
            threading.Thread(target=thread_watchdog, daemon=True).start()

# ================================
# PERSISTENT CONFIG
# ================================
def load_daemon_config():
    try:
        if DAEMON_CONFIG_PATH.exists():
            with open(DAEMON_CONFIG_PATH) as f:
                return json.load(f)
    except:
        pass
    return {"enabled": False}

def save_daemon_config(enabled):
    try:
        data = {"enabled": enabled, "last_started": datetime.now().isoformat() if enabled else None}
        with open(DAEMON_CONFIG_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass

def load_monitoring_config():
    try:
        if MONITORING_CONFIG_PATH.exists():
            with open(MONITORING_CONFIG_PATH) as f:
                return json.load(f)
    except:
        pass
    return {"enabled": False}

def save_monitoring_config(enabled):
    try:
        data = {"enabled": enabled, "last_started": datetime.now().isoformat() if enabled else None}
        with open(MONITORING_CONFIG_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass

# ================================
# UI
# ================================
st.set_page_config(page_title="AI - Alpha Stock Tracker v4.1", layout="wide")

if 'alert_history' not in st.session_state:
    st.session_state.alert_history = {}
for k in ["learning_log", "error_logs"]:
    st.session_state.setdefault(k, [])

initialize_background_threads()

st.markdown("""
<div style='text-align:center;padding:20px;background:#00C853;color:black;border-radius:12px;'>
<h1>AI - ALPHA STOCK TRACKER v4.1</h1>
<p>Self-Learning • Ultra-Confidence • 6%+ Predictive Alerts</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Asset Selection")
    category = st.selectbox("Category", list(ASSET_CATEGORIES.keys()))
    asset = st.selectbox("Asset", list(ASSET_CATEGORIES[category].keys()))
    ticker = ASSET_CATEGORIES[category][asset]

    st.markdown("---")
    st.subheader("Model Status")
    meta = load_metadata(ticker)
    acc = load_accuracy_acc = load_accuracy_log(ticker)
    if meta.get("trained_date"):
        st.metric("Last Trained", meta["trained_date"][:10])
        st.metric("Accuracy", f"{(1-acc.get('avg_error',1))*100:.1f}%" if acc["total_predictions"]>0 else "N/A")

    if st.button("Force Retrain", key="force_retrain_btn"):
        if require_local_access("Force Retrain"):
            with st.spinner("Retraining..."):
                train_self_learning_model(ticker, days=1, force_retrain=True)
                st.success("Done!")
                st.rerun()

    st.markdown("---")
    st.subheader("Learning Daemon")
    dcfg = load_daemon_config()
    st.write(f"Status: {'RUNNING' if dcfg.get('enabled') else 'STOPPED'}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start", key="daemon_start"):
            if require_local_access("Daemon"):
                save_daemon_config(True)
                threading.Thread(target=continuous_learning_daemon, daemon=True).start()
                st.rerun()
    with c2:
        if st.button("Stop", key="daemon_stop"):
            save_daemon_config(False)
            st.rerun()

    st.markdown("---")
    st.subheader("6%+ Alerts")
    mcfg = load_monitoring_config()
    st.write(f"Status: {'RUNNING' if mcfg.get('enabled') else 'STOPPED'}")

    if st.button("Test Telegram", key="test_telegram_btn"):
        if require_local_access("Telegram Test"):
            ok = send_telegram_alert("TEST ALERT\nAI Alpha Tracker v4.1")
            st.toast("Sent!" if ok else "Failed")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Alerts", key="start_alerts"):
            if require_local_access("Alerts"):
                save_monitoring_config(True)
                threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
                st.rerun()
    with c2:
        if st.button("Stop Alerts", key="stop_alerts"):
            save_monitoring_config(False)
            st.rerun()

# Main area
price = get_latest_price(ticker)
if price:
    st.markdown(f"<h2 style='text-align:center;'>LIVE: ${price:.2f}</h2>", unsafe_allow_html=True)
else:
    st.warning("Market closed or no data")

c1, c2 = st.columns(2)
with c1:
    if st.button("Daily Recommendation", use_container_width=True):
        with st.spinner("Analyzing..."):
            st.markdown(daily_recommendation(ticker, asset), unsafe_allow_html=True)
with c2:
    if st.button("5-Day Forecast", use_container_width=True):
        with st.spinner("Forecasting..."):
            show_5day_forecast(ticker, asset)

st.markdown("---")
tab1, tab2 = st.tabs(["Learning Log", "Error Dashboard"])
with tab1:
    for entry in st.session_state.learning_log[-20:]:
        st.write(entry)
with tab2:
    stats = get_error_statistics()
    st.json(stats, expanded=False)

st.markdown("<br><hr><p style='text-align:center;color:#666'>© 2025 AI - Alpha Stock Tracker v4.1</p>", unsafe_allow_html=True)
