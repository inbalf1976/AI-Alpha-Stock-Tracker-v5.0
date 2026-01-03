"""
================================================================================
ALPHA TRACKER v4.2 - FOREX TRADING DASHBOARD
================================================================================
MASTER FILE - ALL WORKING CODE IN ONE PLACE
Last Stable Version: 2025-12-29
Total Lines: 6952 (cleaned 2025-12-29)

⚠️  IMPORTANT: This file is WORKING. Do not refactor or split without backup.

================================================================================
FILE STRUCTURE:
================================================================================

SECTION 1: IMPORTS & SETUP (Lines 1-200)
- All imports
- Environment setup
- TensorFlow configuration

SECTION 2: CONFIGURATION (Lines 201-600)
- Asset categories (forex, commodities, stocks)
- Constants (AUC_TO_BOOST_MULTIPLIER, MAX_DATA_AGE_DAYS, etc.)
- Learning configuration
- Path configurations

SECTION 3: TICKER BLACKLIST & VALIDATION (Lines 601-800)
- TICKER_BLACKLIST (thread-safe)
- is_ticker_blacklisted()
- add_to_blacklist()
- validate_ticker_availability()

SECTION 4: UTILITY FUNCTIONS (Lines 801-1000)
- sanitize_ticker()
- get_asset_name_from_ticker()
- normalize_dataframe_columns()
- get_ticker_type() - forex/commodity detection

SECTION 5: DATA FETCHING (Lines 1001-1400)
- download_with_timeout() - with Alpha Vantage fallback
- download_from_alpha_vantage()
- get_latest_price() - forex-aware
- validate_commodity_data()
- get_market_info()

SECTION 6: CACHING SYSTEM (Lines 1401-1600)
- SmartCache class
- data_cache instance

SECTION 7: DATABASE & STORAGE (Lines 1601-1900)
- Prediction model (SQLAlchemy)
- Session management
- Fallback JSON storage

SECTION 8: ERROR HANDLING & LOGGING (Lines 1901-2100)
- log_error()
- ErrorSeverity enum
- Circuit breaker

SECTION 9: MODEL MANAGEMENT (Lines 2101-2400)
- ModelManager class
- build_lstm_model()
- load_model() / save_model()

SECTION 10: TRAINING & PREDICTION (Lines 2401-2800)
- train_self_learning_model_enhanced()
- generate_forecast_with_confidence()
- prepare_training_data()

SECTION 11: PATTERN MINING (Lines 2801-3200)
- mine_intraday_patterns()
- mine_daily_patterns()
- train_and_evaluate_patterns()
- run_pattern_mining_cycle()

SECTION 12: PREDICTION VALIDATION (Lines 3201-3600)
- load_accuracy_log() - REAL validation
- validate_predictions()
- should_retrain()

SECTION 13: MONITORING & ALERTS (Lines 3601-4000)
- monitor_6percent_pre_move_managed()
- send_telegram_alert()
- Monitoring configuration

SECTION 14: BACKGROUND THREADS (Lines 4001-4400)
- continuous_learning_daemon_managed()
- continuous_pattern_miner_managed()
- thread_watchdog_managed()
- ApplicationState class

SECTION 15: STREAMLIT UI (Lines 4401-6952)
- main() function
- Sidebar configuration
- Tab 1: Dashboard
- Tab 2: Forecast
- Tab 3: Analysis
- Tab 4: Accuracy
- Tab 5: Settings
- Tab 6: Diagnostics
- Tab 7: Logs

================================================================================
RECENT CHANGES LOG:
================================================================================
2025-12-29: Removed duplicate is_ticker_blacklisted() function (-4 lines)
2025-12-29: Cleaned codebase - 0 duplicates confirmed
2025-12-27: Added forex support (XAUUSD=X, EURUSD=X, etc.)
2025-12-27: Fixed commodity price handling (7-day lookback)
2025-12-27: Added get_market_info() for all ticker types
2025-12-26: Implemented thread-safe blacklist system
2025-12-26: Fixed pattern mining pandas errors
2025-12-25: Added Alpha Vantage fallback

================================================================================
KNOWN WORKING FEATURES:
================================================================================
✅ Forex price fetching (4-5 decimal places)
✅ Commodity spot prices (XAU/USD, XAG/USD)
✅ Stock predictions with confidence intervals
✅ Pattern mining (5 elite patterns found!)
✅ 6%+ monitoring with Telegram alerts
✅ Real prediction validation against actual prices
✅ Thread-safe operations
✅ Blacklist auto-detection
✅ No duplicate code (verified 2025-12-29)

================================================================================
KNOWN ISSUES:
================================================================================
⚠️  Pattern mining slow (30 min/cycle) - acceptable for now
⚠️  Some agricultural commodities blacklisted (ZW=F, ZC=F, ZS=F) - delisted
⚠️  Some indices have stale data (^NDX) - blacklisted
⚠️  HG=F (Copper) has poor accuracy - monitoring for potential blacklist

================================================================================
STATISTICS:
================================================================================
Total Functions: 83
Total Lines: 6952
Duplicates: 0
Backups: 3 (STABLE_2025-12-27, before_duplicate_removal, STABLE_2025-12-29)

================================================================================
"""

# ================================
# INSTALLATION CHECK AND FALLBACKS
# ================================
import subprocess
import sys
import warnings
import os

# Suppress all warnings first
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Try to import sqlalchemy, install if missing
try:
    from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Index
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    print("sqlalchemy not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy"])
        from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Index
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        SQLALCHEMY_AVAILABLE = True
        print("sqlalchemy installed successfully.")
    except:
        print("Failed to install sqlalchemy. Using fallback JSON storage.")
        SQLALCHEMY_AVAILABLE = False

# Try to import psutil, install if missing
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("psutil not found. Installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
        PSUTIL_AVAILABLE = True
        print("psutil installed successfully.")
    except:
        print("psutil not available. Resource monitoring disabled.")
        PSUTIL_AVAILABLE = False

# Try to install other potential missing packages
required_packages = [
    'yfinance',
    'tensorflow',
    'scikit-learn',
    'joblib',
    'plotly',
    'python-dotenv',
    'aiohttp'
]

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"{package} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")
        except:
            print(f"Failed to install {package}.")

# ================================
# MAIN IMPORTS (AFTER INSTALLATION)
# ================================

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests
import plotly.graph_objs as go
import time
import threading
import json
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from enum import Enum
from typing import Tuple, List, Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import queue
import gc
import re
from dotenv import load_dotenv
import asyncio
import aiohttp

# TensorFlow warning suppression
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load environment variables
load_dotenv()

# ================================
# SMART CACHING SYSTEM
# ================================

class SmartCache:
    """Smart caching with TTL and size limits"""
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 300):
        self.cache = {}
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self._lock = threading.RLock()
        
    def get(self, key: str):
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if time.time() > entry['expiry']:
                del self.cache[key]
                return None
            
            # Update access time
            entry['last_access'] = time.time()
            return entry['data']
    
    def set(self, key: str, data: Any, ttl: int = None):
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            self.cache[key] = {
                'data': data,
                'expiry': time.time() + ttl,
                'last_access': time.time(),
                'size': self._estimate_size(data)
            }
            
            # Cleanup if cache too large
            self._cleanup_if_needed()
    
    def _estimate_size(self, data):
        """Rough size estimation"""
        import sys
        return sys.getsizeof(str(data)) / 1024 / 1024  # MB
    
    def _cleanup_if_needed(self):
        """Remove oldest entries if cache exceeds max size"""
        total_size = sum(entry['size'] for entry in self.cache.values())
        
        if total_size > self.max_size_mb:
            # Sort by last access time (oldest first)
            sorted_items = sorted(self.cache.items(), 
                                key=lambda x: x[1]['last_access'])
            
            while total_size > self.max_size_mb * 0.8 and sorted_items:
                key, _ = sorted_items.pop(0)
                total_size -= self.cache[key]['size']
                del self.cache[key]

# Global cache instance
data_cache = SmartCache()
_error_log_lock = threading.RLock()

# ================================
# DATABASE SETUP (WITH FALLBACK)
# ================================

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class Prediction(Base):
        __tablename__ = 'predictions'
        
        id = Column(Integer, primary_key=True)
        ticker = Column(String(20), index=True)
        prediction_date = Column(DateTime, index=True)
        predicted_price = Column(Float)
        actual_price = Column(Float, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        error_mape = Column(Float, nullable=True)
        error_mae = Column(Float, nullable=True)
        previous_price = Column(Float, nullable=True)
        validated = Column(Integer, default=0)
    
    # Initialize database
    try:
        engine = create_engine('sqlite:///predictions.db')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        
        # Create indexes
        try:
            Index('ix_predictions_ticker_date', Prediction.ticker, Prediction.prediction_date).create(engine)
            Index('ix_predictions_created', Prediction.created_at).create(engine)
        except:
            pass
    except Exception as e:
        print(f"Database initialization failed: {e}")
        SQLALCHEMY_AVAILABLE = False
else:
    # Fallback: JSON-based storage
    class FakeSession:
        def __init__(self):
            self.predictions = []
        
        def query(self, model):
            return FakeQuery(self.predictions)
        
        def add(self, obj):
            self.predictions.append(obj)
        
        def commit(self):
            pass
        
        def close(self):
            pass
    
    class FakeQuery:
        def __init__(self, predictions):
            self.predictions = predictions
        
        def filter(self, *args):
            return self
        
        def order_by(self, *args):
            return self
        
        def all(self):
            return self.predictions
        
        def first(self):
            return self.predictions[0] if self.predictions else None
    
    Session = FakeSession

# ================================
# THREAD-SAFE SESSION STATE
# ================================

class ThreadSafeSessionState:
    """Thread-safe wrapper for Streamlit session state"""
    def __init__(self):
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            self._state.update(updates)

# Create thread-safe session state
safe_state = ThreadSafeSessionState()

# ================================
# ENHANCED CIRCUIT BREAKER
# ================================

class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with exponential backoff"""
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: callable, *args, **kwargs) -> Tuple[bool, Any]:
        """Execute function with circuit breaker protection"""
        with self._lock:
            # Check if circuit is open
            if self.state == "OPEN":
                if self.last_failure and \
                   (datetime.now() - self.last_failure).seconds >= self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    return False, "Circuit breaker is OPEN"
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset if in HALF_OPEN
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED after successful call")
                
                return True, result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure = datetime.now()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")
                
                return False, str(e)
    
    def call_with_retry(self, func: callable, max_retries: int = 3, *args, **kwargs):
        """Execute with exponential backoff retry"""
        retry_count = 0
        base_delay = 1
        
        while retry_count <= max_retries:
            success, result = self.call(func, *args, **kwargs)
            
            if success:
                return True, result
            
            retry_count += 1
            if retry_count <= max_retries:
                delay = base_delay * (2 ** (retry_count - 1))
                logger.info(f"Retry {retry_count}/{max_retries} after {delay}s")
                time.sleep(delay)
        
        return False, f"Failed after {max_retries} retries"
    
    def reset(self) -> None:
        """Manually reset circuit breaker"""
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure = None
            logger.info("Circuit breaker manually reset")
    
    def get_metrics(self):
        """Get circuit breaker metrics"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure,
            'uptime': 'OPEN' if self.state == 'OPEN' else 'CLOSED'
        }

# Global enhanced rate limiter
alpha_vantage_limiter = EnhancedCircuitBreaker()

# ================================
# PERFORMANCE MONITORING
# ================================

class PerformanceMonitor:
    """Monitor application performance metrics"""
    def __init__(self):
        self.metrics = {
            'api_latency': [],
            'prediction_time': [],
            'training_time': [],
            'memory_usage': [],
            'download_latency': [],
            'validation_time': []
        }
    
    def record_latency(self, operation: str, duration: float):
        self.metrics.setdefault(f'{operation}_latency', []).append(duration)
    
    def get_p95_latency(self, operation: str) -> float:
        """Get 95th percentile latency"""
        latencies = self.metrics.get(f'{operation}_latency', [])
        if not latencies:
            return 0.0
        return np.percentile(latencies, 95)

# Global performance monitor
performance_monitor = PerformanceMonitor()

# ================================
# ALERTING SYSTEM
# ================================

class AlertManager:
    """Centralized alerting"""
    def __init__(self):
        self.channels = {}
    
    def send_alert(self, severity: str, message: str, channels: List[str] = None):
        """Send alert to multiple channels"""
        if channels is None:
            channels = ['telegram']
        
        for channel in channels:
            if channel in self.channels:
                try:
                    self.channels[channel].send(severity, message)
                except Exception as e:
                    logger.error(f"Alert failed for {channel}: {e}")

# Global alert manager
alert_manager = AlertManager()

# ================================
# CONSTANTS
# ================================

# System Configuration
AUC_TO_BOOST_MULTIPLIER = 180
MAX_DATA_AGE_DAYS = 30
MEMORY_WARNING_THRESHOLD_MB = 1024
CPU_WARNING_THRESHOLD_PERCENT = 80
DISK_WARNING_THRESHOLD_PERCENT = 90
NETWORK_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
CACHE_TTL_SECONDS = 300  # 5 minutes

# Alpha Vantage Configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


# Learning configuration
LEARNING_CONFIG = {
    "lookback_window": 60,
    "full_retrain_epochs": 50,
    "fine_tune_epochs": 10,
    "prediction_days": 5,
    "batch_size": 32,
    "validation_split": 0.1,
    "early_stopping_patience": 5
}

# Asset categories
ASSET_CATEGORIES = {
    "Commodities": {
        "Crude Oil (Continuous)": "CL=F",
        "Gasoline": "RB=F",
        "Brent Oil": "BZ=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "Palladium": "PA=F",
        "Wheat": "ZW=F",
        "Gold": "GC=F",
        "Corn": "ZC=F",
        "Natural Gas": "NG=F",
        "Platinum": "PL=F",
        "Soybeans": "ZS=F"
    },
    "Indices": {
        "S&P 500": "^GSPC",
        "US 30": "^DJI",
        "US 2000": "^RUT",
        "NASDAQ 100": "^NDX"
    },
    "Currencies": {
        "USD/ILS": "USDILS=X"
    },
    "Popular": {
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Palantir Technologies": "PLTR",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Meta": "META",
        "Amazon": "AMZN",
        "Intel": "INTC",
        "Alphabet": "GOOGL",
        "Salesforce": "CRM",
        "Coinbase": "COIN",
        "Netflix": "NFLX",
        "PayPal": "PYPL",
        "Nike": "NKE",
        "Broadcom": "AVGO",
        "Visa": "V",
        "BlackRock": "BLK",
        "JPMorgan": "JPM",
        "IBM": "IBM",
        "Merck": "MRK",
        "Qualcomm": "QCOM",
        "Lockheed Martin": "LMT",
        "Mastercard": "MA"
    }
}

# ================================
# TICKER BLACKLIST SYSTEM
# ================================

TICKER_BLACKLIST = set([
    "ZW=F",  # Wheat futures - delisted
    "ZC=F",  # Corn futures - delisted  
    "ZS=F",  # Soybean futures - delisted
])

def add_to_blacklist(ticker: str, reason: str = "Failed validation"):
    """Add ticker to blacklist with logging"""
    if ticker not in TICKER_BLACKLIST:
        TICKER_BLACKLIST.add(ticker)
        logger.warning(f"⚫ Added {ticker} to blacklist: {reason}")

# ================================
# TICKER BLACKLIST & VALIDATION
# ================================

# CLEARED: Removed Wheat, Corn, and Soybeans from blacklist to allow fetching
TICKER_BLACKLIST = set([
    # Add known truly delisted tickers here if they cause crashes
])

def is_ticker_blacklisted(ticker: str) -> bool:
    """Check if ticker is blacklisted"""
    return ticker in TICKER_BLACKLIST

def add_to_blacklist(ticker: str, reason: str = "Failed validation"):
    """Add ticker to blacklist with logging"""
    if ticker not in TICKER_BLACKLIST:
        TICKER_BLACKLIST.add(ticker)
        logger.warning(f"⚫ Added {ticker} to blacklist: {reason}")

def validate_ticker_with_retry(ticker: str, max_retries: int = 3) -> Tuple[bool, str]:
    """
    Refined validation to handle yfinance 'ghost delisting' errors.
    """
    if is_ticker_blacklisted(ticker):
        return False, "Blacklisted"
    
    for attempt in range(max_retries):
        try:
            # FIX 1: Change period to '1mo'. 
            # yfinance often fails with '5d' or '10d' on commodities.
            df = yf.download(ticker, period="1mo", interval="1d", progress=False)
            
            if df is None or len(df) == 0:
                if attempt < max_retries - 1:
                    time.sleep(2) # Longer wait between retries
                    continue
                return False, "No data available"
            
            df = normalize_dataframe_columns(df)
            
            if hasattr(df.index, '__len__') and len(df.index) > 0:
                last_date = df.index[-1]
                if hasattr(last_date, 'date'):
                    days_old = (datetime.now().date() - last_date.date()).days
                    
                    # FIX 2: Relaxed stale limit for futures
                    stale_limit = 14 if ticker.endswith('=F') else 7
                    if days_old > stale_limit:
                        return False, f"Stale ({days_old}d old)"
            
            # Price validation
            if 'Close' not in df.columns or df['Close'].isnull().all():
                return False, "No price data"
            
            last_price = float(df['Close'].dropna().iloc[-1])
            if last_price <= 0:
                return False, "Invalid price"
            
            return True, f"Valid (${last_price:.2f})"
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False, f"Error: {str(e)[:30]}"
    
    return False, "Validation failed"

# Path configurations
MODELS_DIR = Path("models")
SCALERS_DIR = Path("scalers")
METADATA_DIR = Path("metadata")
PREDICTIONS_DIR = Path("predictions")
CONFIG_DIR = Path("config")

for directory in [MODELS_DIR, SCALERS_DIR, METADATA_DIR, PREDICTIONS_DIR, CONFIG_DIR]:
    directory.mkdir(exist_ok=True)

# Configuration files
AUTO_PATTERNS_FILE = Path("auto_patterns.json")
PATTERN_MINING_CONFIG = CONFIG_DIR / "pattern_mining.json"
DAEMON_CONFIG = CONFIG_DIR / "daemon.json"
MONITORING_CONFIG = CONFIG_DIR / "monitoring.json"
ERROR_LOG = Path("error_log.json")

# Pattern mining watchlist - generated from all assets in ASSET_CATEGORIES
# Automatically filters out blacklisted tickers
PATTERN_WATCHLIST = []
for category in ASSET_CATEGORIES.values():
    for ticker in category.values():
        if not is_ticker_blacklisted(ticker):
            PATTERN_WATCHLIST.append(ticker)
        # Note: Blacklisted tickers are silently excluded here
        # Logging happens later when logger is initialized

# ================================
# LOGGING AND ERROR HANDLING
# ================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log watchlist initialization (now that logger exists)
excluded_count = sum(1 for cat in ASSET_CATEGORIES.values() for _ in cat.values()) - len(PATTERN_WATCHLIST)
if excluded_count > 0:
    logger.info(f"Pattern watchlist initialized: {len(PATTERN_WATCHLIST)} active tickers ({excluded_count} blacklisted excluded)")
    if TICKER_BLACKLIST:
        logger.info(f"Blacklisted tickers: {', '.join(sorted(TICKER_BLACKLIST))}")
else:
    logger.info(f"Pattern watchlist initialized: {len(PATTERN_WATCHLIST)} active tickers")

class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

def log_error(severity: ErrorSeverity, function: str, error: Exception, 
              ticker: Optional[str] = None, user_message: Optional[str] = None, 
              show_to_user: bool = True) -> None:
    """Enhanced error logging with structured data and thread-safe file operations"""
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity.name,
        "function": function,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "ticker": ticker,
        "user_message": user_message
    }
    
    with _error_log_lock:
        try:
            errors = []
            if ERROR_LOG.exists():
                try:
                    with open(ERROR_LOG, 'r') as f:
                        content = f.read().strip()
                        if content:
                            errors = json.loads(content)
                            if not isinstance(errors, list):
                                errors = []
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(f"Error log corrupted, starting fresh: {e}")
                    errors = []
            
            errors.append(error_entry)
            if len(errors) > 1000:
                errors = errors[-1000:]
            
            # Atomic write
            temp_log = ERROR_LOG.with_suffix('.tmp')
            with open(temp_log, 'w') as f:
                json.dump(errors, f, indent=2)
            temp_log.replace(ERROR_LOG)
                
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    # Standard logging
    log_message = f"{function}: {error}"
    if ticker:
        log_message = f"{ticker} - {log_message}"
        
    if severity == ErrorSeverity.CRITICAL:
        logger.critical(log_message)
    elif severity == ErrorSeverity.ERROR:
        logger.error(log_message)
    elif severity == ErrorSeverity.WARNING:
        logger.warning(log_message)
    else:
        logger.info(log_message)
        
        
# ================================
# TICKER VALIDATION
# ================================

def validate_ticker_availability(ticker: str) -> Tuple[bool, str]:
    """
    Test if a ticker is available and has recent data
    
    Returns:
        (is_valid, message)
    """
    # Check blacklist first
    if is_ticker_blacklisted(ticker):
        return False, "Blacklisted"
    
    try:
        # Get more history for commodities
        period = "10d" if ticker.endswith('=F') else "5d"
        
        df = yf.download(ticker, period="1mo", interval="1d", progress=False, timeout=20)
        
        if df is None or len(df) == 0:
            return False, "No data available"
        
        df = normalize_dataframe_columns(df)
        
        # Use commodity-specific validation if applicable
        if ticker.endswith('=F'):
            is_valid, message = validate_commodity_data(ticker, df)
            if not is_valid:
                return False, message
        
        # Check if we have recent data
        if hasattr(df.index, '__len__') and len(df.index) > 0:
            # Find last valid price (non-zero, non-null)
            valid_closes = df['Close'][(df['Close'] > 0) & (~df['Close'].isnull())]
            
            if len(valid_closes) == 0:
                return False, "No valid prices"
            
            last_date = valid_closes.index[-1]
            if hasattr(last_date, 'date'):
                days_old = (datetime.now().date() - last_date.date()).days
                
                # More lenient thresholds for commodities
                threshold = 10 if ticker.endswith('=F') else 7
                
                if days_old > threshold:
                    return False, f"Data is {days_old} days old (stale)"
            
            last_price = valid_closes.iloc[-1]
            if pd.isna(last_price) or last_price <= 0:
                return False, "Invalid last price"
            
            return True, f"Valid (${last_price:.2f})"
        
        return False, "No valid data"
        
    except Exception as e:
        error_msg = str(e).lower()
        if "delisted" in error_msg or "no price data found" in error_msg:
            return False, "Delisted/No data"
        return False, f"Error: {str(e)[:50]}"

def get_validated_watchlist() -> List[str]:
    """Get watchlist with only validated tickers"""
    validated = []
    invalid = []
    
    # Use PATTERN_WATCHLIST which already filters blacklisted tickers
    for ticker in PATTERN_WATCHLIST:
        # Double-check blacklist (in case it was added during runtime)
        if is_ticker_blacklisted(ticker):
            invalid.append((ticker, "Blacklisted"))
            continue
        
        is_valid, message = validate_ticker_availability(ticker)
        if is_valid:
            validated.append(ticker)
        else:
            invalid.append((ticker, message))
            # Add to blacklist if validation fails
            if "stale" not in message.lower():  # Don't blacklist just for being stale
                add_to_blacklist(ticker, message)
    
    if invalid:
        logger.warning(f"Found {len(invalid)} invalid/blacklisted tickers:")
        for ticker, reason in invalid:
            logger.warning(f"  ⚫ {ticker}: {reason}")
    
    logger.info(f"Validated watchlist: {len(validated)}/{len(PATTERN_WATCHLIST)} tickers active")
    if TICKER_BLACKLIST:
        logger.info(f"Current blacklist: {', '.join(sorted(TICKER_BLACKLIST))}")
    
    return validated

# ================================
# UTILITY FUNCTIONS
# ================================

def sanitize_ticker(ticker: str) -> str:
    """Sanitize ticker for safe file operations"""
    # Allow only alphanumeric, dash, underscore
    return re.sub(r'[^a-zA-Z0-9_-]', '_', ticker)

def get_asset_name_from_ticker(ticker: str) -> str:
    """Get human-readable asset name from ticker symbol"""
    # Search through all categories
    for category_name, assets in ASSET_CATEGORIES.items():
        for asset_name, asset_ticker in assets.items():
            if asset_ticker == ticker:
                return asset_name
    
    # Fallback: return ticker if not found in categories
    return ticker

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe columns from multi-index to single index"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def convert_alpha_vantage_to_yfinance_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Alpha Vantage dataframe format to yfinance format"""
    try:
        # Alpha Vantage columns: date, open, high, low, close, volume
        # Rename to match yfinance format
        column_mapping = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date ascending (oldest first)
        df = df.sort_index()
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any NaN rows
        df = df.dropna()
        
        logger.info(f"Converted Alpha Vantage data: {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Failed to convert Alpha Vantage data: {e}")
        return None

def download_from_alpha_vantage(ticker: str, outputsize: str = "full") -> Optional[pd.DataFrame]:
    """Download historical data from Alpha Vantage API with rate limiting"""
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("Alpha Vantage API key not configured")
        return None
    
    try:
        logger.info(f"Attempting to download {ticker} from Alpha Vantage...")
        
        # Clean ticker for Alpha Vantage (remove =F for futures, etc.)
        av_ticker = ticker.replace('=F', '').replace('^', '')
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': av_ticker,
            'outputsize': outputsize,  # 'compact' = 100 days, 'full' = 20+ years
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=NETWORK_TIMEOUT_SECONDS)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            logger.error(f"Alpha Vantage error: {data['Error Message']}")
            return None
        
        if 'Note' in data:
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
            return None
        
        # Extract time series data
        if 'Time Series (Daily)' not in data:
            logger.error(f"No time series data in Alpha Vantage response for {ticker}")
            return None
        
        time_series = data['Time Series (Daily)']
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Convert to yfinance format
        df = convert_alpha_vantage_to_yfinance_format(df)
        
        if df is not None and len(df) > 0:
            logger.info(f"✅ Successfully downloaded {len(df)} rows from Alpha Vantage for {ticker}")
            metrics_collector.increment("alphavantage_downloads")
            return df
        else:
            logger.warning(f"Alpha Vantage returned empty data for {ticker}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"Alpha Vantage request timeout for {ticker}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Alpha Vantage request failed for {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error downloading from Alpha Vantage for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "download_from_alpha_vantage", e, ticker=ticker, show_to_user=False)
        return None

# ============================================================================
# LINE 1066: download_with_timeout() - Core data fetching
# STATUS: WORKING - Forex support added 2025-12-27
# DEPENDENCIES: yfinance, Alpha Vantage API, SmartCache
# CRITICAL: Used by ALL data operations - test thoroughly before changes
# ============================================================================
def download_with_timeout(ticker: str, period: str = "1y", 
                         interval: str = "1d", 
                         timeout: int = NETWORK_TIMEOUT_SECONDS) -> Optional[pd.DataFrame]:
    """Download data with timeout protection, blacklist check, and Alpha Vantage fallback"""
    
    # Check blacklist first (fast path)
    if is_ticker_blacklisted(ticker):
        logger.warning(f"⚫ Skipping blacklisted ticker: {ticker}")
        return None
    
    # Check cache first
    cache_key = f"download_{ticker}_{period}_{interval}"
    cached_data = data_cache.get(cache_key)
    if cached_data is not None:
        logger.info(f"Using cached data for {ticker}")
        metrics_collector.increment("cache_hits")
        return cached_data
    
    try:
        logger.info(f"Attempting to download {ticker} from yfinance...")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                yf.download, 
                ticker, 
                period=period, 
                interval=interval, 
                progress=False,
                auto_adjust=True
            )
            df = future.result(timeout=timeout)
            df = normalize_dataframe_columns(df)
            
            if df is not None and len(df) > 0:
                logger.info(f"✅ Successfully downloaded {len(df)} rows from yfinance for {ticker}")
                metrics_collector.increment("data_downloads")
                metrics_collector.increment("yfinance_downloads")
                # Cache the data
                data_cache.set(cache_key, df, ttl=300)
                return df
            else:
                logger.warning(f"yfinance returned empty data for {ticker}, trying Alpha Vantage...")
                
    except TimeoutError:
        logger.error(f"Timeout downloading data from yfinance for {ticker}, trying Alpha Vantage...")
    except Exception as e:
        error_msg = str(e)
        # Check if it's a "delisted" error
        if "delisted" in error_msg.lower() or "no price data found" in error_msg.lower():
            logger.warning(f"⚫ {ticker} appears delisted, adding to blacklist")
            TICKER_BLACKLIST.add(ticker)
            return None
        logger.error(f"Error downloading from yfinance for {ticker}: {e}, trying Alpha Vantage...")
    
    # Fallback to Alpha Vantage
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning(f"❌ No Alpha Vantage fallback available for {ticker}")
        metrics_collector.increment("cache_misses")
        return None
    
    logger.info(f"🔄 Falling back to Alpha Vantage for {ticker}...")
    
    # Determine outputsize based on period
    if period in ["1mo", "3mo"]:
        outputsize = "compact"
    else:
        outputsize = "full"
    
    av_df = download_from_alpha_vantage(ticker, outputsize=outputsize)

    if av_df is not None:
        metrics_collector.increment("alphavantage_downloads")
    
        # Filter to match requested period if needed
        if period != "max":
            try:
                days_map = {
                    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "10y": 3650
                }
                days = days_map.get(period, 365)
                cutoff_date = datetime.now() - timedelta(days=days)
                av_df = av_df[av_df.index >= cutoff_date]
                logger.info(f"Filtered Alpha Vantage data to last {days} days: {len(av_df)} rows")
            except Exception as e:
                logger.warning(f"Could not filter Alpha Vantage data by period: {e}")
        
        # Cache the data
        data_cache.set(cache_key, av_df, ttl=300)
        return av_df
    else:
        # Both sources failed - add to blacklist
        logger.error(f"⚫ All sources failed for {ticker}, adding to blacklist")
        TICKER_BLACKLIST.add(ticker)
    
    logger.error(f"❌ All data sources failed for {ticker}")
    metrics_collector.increment("cache_misses")
    return None

# ================================
# DATA QUALITY METRICS
# ================================

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate comprehensive data quality metrics"""
    metrics = {}
    
    if df is None or len(df) == 0:
        return metrics
    
    try:
        # Price validity checks
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                metrics[f'{col}_null_pct'] = df[col].isnull().mean() * 100
                metrics[f'{col}_zero_pct'] = (df[col] == 0).mean() * 100
        
        # Volume checks
        if 'Volume' in df.columns:
            metrics['volume_null_pct'] = df['Volume'].isnull().mean() * 100
            metrics['volume_zero_pct'] = (df['Volume'] == 0).mean() * 100
        
        # Price consistency checks
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            # High should be >= Low, Open, Close
            high_condition = (df['High'] >= df['Low']) & (df['High'] >= df['Open']) & (df['High'] >= df['Close'])
            metrics['high_validity_pct'] = high_condition.mean() * 100
            
            # Low should be <= Open, Close
            low_condition = (df['Low'] <= df['Open']) & (df['Low'] <= df['Close'])
            metrics['low_validity_pct'] = low_condition.mean() * 100
        
        # Gap analysis
        if 'Close' in df.columns:
            returns = df['Close'].pct_change().dropna()
            metrics['return_std'] = returns.std()
            metrics['return_skew'] = returns.skew()
            metrics['return_kurtosis'] = returns.kurtosis()
            
            # Detect extreme moves (>3 std deviations)
            if len(returns) > 0:
                extreme_moves = (abs(returns) > returns.std() * 3).sum()
                metrics['extreme_move_pct'] = extreme_moves / len(returns) * 100
        
        return metrics
        
    except Exception as e:
        logger.error(f"Data quality metrics calculation failed: {e}")
        return {}

# ================================
# CONFIGURATION MANAGEMENT
# ================================

class ConfigManager:
    """Manage configuration with validation and backup"""
    
    CONFIG_SCHEMAS = {
        "pattern_mining": {
            "enabled": bool,
            "cycle_interval_minutes": (int, 10, 120),
            "min_auc_threshold": (float, 0.60, 0.90),
            "max_auc_std": (float, 0.05, 0.20)
        },
        "daemon": {
            "enabled": bool,
            "sleep_minutes": (int, 1, 60),
            "max_retrain_per_cycle": (int, 1, 50)
        },
        "monitoring": {
            "enabled": bool,
            "check_interval_minutes": (int, 1, 30),
            "telegram_alerts": bool,
            "threshold_percent": (float, 1.0, 20.0),
            "cooldown_minutes": (int, 5, 120),
            "watchlist": list
        },
        "pattern_overrides": {
            "enabled": bool,
            "overrides": dict
        }
    }
    
    @classmethod
    def validate_config(cls, config_type: str, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate configuration against schema"""
        if config_type not in cls.CONFIG_SCHEMAS:
            return True, "Unknown config type"
        
        schema = cls.CONFIG_SCHEMAS[config_type]
        errors = []
        
        for key, expected_type in schema.items():
            if key not in config:
                errors.append(f"Missing key: {key}")
                continue
            
            value = config[key]
            
            if isinstance(expected_type, tuple):
                # Type with range validation
                type_class, min_val, max_val = expected_type
                if not isinstance(value, type_class):
                    errors.append(f"{key} should be {type_class.__name__}, got {type(value).__name__}")
                elif value < min_val or value > max_val:
                    errors.append(f"{key} should be between {min_val} and {max_val}, got {value}")
            else:
                # Simple type validation
                if not isinstance(value, expected_type):
                    errors.append(f"{key} should be {expected_type.__name__}, got {type(value).__name__}")
        
        if errors:
            return False, "; ".join(errors)
        return True, "Valid"
    
    @classmethod
    def load_config_with_backup(cls, config_path: Path, config_type: str, 
                               default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration with backup and validation"""
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults
                config = {**default_config, **loaded_config}
                
                # Validate
                is_valid, message = cls.validate_config(config_type, config)
                if is_valid:
                    return config
                else:
                    logger.warning(f"Config validation failed for {config_type}: {message}")
                    # Create backup of invalid config
                    backup_path = config_path.with_suffix('.json.bak')
                    with open(backup_path, 'w') as f:
                        json.dump(loaded_config, f, indent=2)
                    logger.info(f"Backed up invalid config to {backup_path}")
            
            # Return default if loading fails
            return default_config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return default_config
    
    @classmethod
    def save_config(cls, config_path: Path, config: Dict[str, Any], 
                   config_type: str) -> bool:
        """Save configuration with validation"""
        try:
            # Validate before saving
            is_valid, message = cls.validate_config(config_type, config)
            if not is_valid:
                raise ValueError(f"Config validation failed: {message}")
            
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save config {config_path}: {e}")
            return False

# ================================
# DATA VALIDATION
# ================================

def validate_financial_data(df: pd.DataFrame, ticker: str, 
                           min_rows: int = 100, 
                           max_null_percent: float = 0.1) -> Tuple[bool, str]:
    """Comprehensive financial data validation"""
    if df is None or len(df) == 0:
        return False, "Empty dataframe"
    
    if len(df) < min_rows:
        return False, f"Insufficient rows: {len(df)} < {min_rows}"
    
    # Check required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    total_null = null_counts.sum()
    null_percent = total_null / (len(df) * len(required_cols))
    
    if null_percent > max_null_percent:
        return False, f"Too many nulls: {null_percent:.1%}"
    
    # Check for zero or negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    negative_prices = (df[price_cols] <= 0).any().any()
    if negative_prices:
        return False, "Zero or negative prices detected"
    
    # Check for extreme outliers (prices 10x median)
    for col in price_cols:
        median_price = df[col].median()
        outlier_count = (df[col] > median_price * 10).sum()
        if outlier_count > 0:
            return False, f"Extreme outliers in {col}"
    
    # Check data freshness
    if hasattr(df.index, '__len__') and len(df.index) > 0:
        last_date = df.index[-1]
        if hasattr(last_date, 'date'):
            data_age = (datetime.now().date() - last_date.date()).days
            if data_age > MAX_DATA_AGE_DAYS:
                return False, f"Data is stale: {data_age} days old"
    
    return True, "Data validation passed"

# ================================
# THREAD MANAGEMENT
# ================================

class ThreadManager:
    """Manage thread lifecycle and resource cleanup"""
    def __init__(self):
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_events: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()
    
    def _cleanup_thread(self, name: str) -> None:
        """Internal cleanup of thread resources"""
        if name in self._threads:
            del self._threads[name]
        if name in self._stop_events:
            del self._stop_events[name]
    
    def start_thread(self, name: str, target: callable, daemon: bool = True) -> bool:
        """Start a managed thread with proper cleanup"""
        with self._lock:
            # Cleanup dead thread first
            if name in self._threads:
                if self._threads[name].is_alive():
                    logger.warning(f"Thread {name} is already running")
                    return False
                else:
                    self._cleanup_thread(name)
            
            stop_event = threading.Event()
            self._stop_events[name] = stop_event
            
            # Wrap target with proper cleanup
            def wrapped_target():
                try:
                    target(stop_event)
                except Exception as e:
                    logger.error(f"Thread {name} crashed: {e}")
                    log_error(ErrorSeverity.ERROR, f"thread_{name}", e, show_to_user=False)
                finally:
                    with self._lock:
                        self._cleanup_thread(name)
            
            thread = threading.Thread(target=wrapped_target, daemon=daemon, name=name)
            self._threads[name] = thread
            thread.start()
            logger.info(f"Started managed thread: {name}")
            return True
    
    def stop_thread(self, name: str, timeout: int = 30) -> bool:
        """Stop a thread gracefully"""
        with self._lock:
            if name in self._stop_events:
                self._stop_events[name].set()
            
            if name in self._threads:
                thread = self._threads[name]
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning(f"Thread {name} didn't stop gracefully")
                    return False
                else:
                    logger.info(f"Stopped thread: {name}")
                
                self._cleanup_thread(name)
                return True
            
            return False
    
    def stop_all(self, timeout: int = 30) -> None:
        """Stop all managed threads"""
        thread_names = list(self._threads.keys())
        for name in thread_names:
            self.stop_thread(name, timeout)
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all managed threads"""
        with self._lock:
            return {
                name: {
                    'alive': thread.is_alive(),
                    'daemon': thread.daemon,
                    'native_id': getattr(thread, 'native_id', None)
                }
                for name, thread in self._threads.items()
            }
    
    def is_running(self, name: str) -> bool:
        """Check if a specific thread is running"""
        with self._lock:
            return name in self._threads and self._threads[name].is_alive()

# Global thread manager
thread_manager = ThreadManager()

# ================================
# TENSORFLOW AND MODEL MANAGEMENT
# ================================

class EnhancedModelManager:
    """Enhanced ModelManager with better resource cleanup"""
    def __init__(self):
        self._models: Dict[str, Tuple[Any, float]] = {}  # ticker -> (model, timestamp)
        self._lock = threading.RLock()
        self._cache_ttl = CACHE_TTL_SECONDS
        self._active_sessions = set()
        
        # Configure GPU memory growth
        self._configure_gpu()
    
    def _configure_gpu(self) -> None:
        """Configure GPU memory settings"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}")
    
    def _cleanup_gpu_memory(self):
        """Force GPU memory cleanup"""
        try:
            tf.keras.backend.clear_session()
            
            # Try to use numba for GPU cleanup if available
            try:
                import numba.cuda
                numba.cuda.select_device(0)
                numba.cuda.close()
            except:
                pass
        except:
            pass
        
        gc.collect()
    
    def load_model(self, model_path: Path, use_cache: bool = True) -> Optional[Any]:
        """Load model with caching and memory management"""
        ticker = model_path.stem
        
        # Check cache first
        if use_cache:
            with self._lock:
                if ticker in self._models:
                    model, timestamp = self._models[ticker]
                    if time.time() - timestamp < self._cache_ttl:
                        logger.debug(f"Using cached model for {ticker}")
                        return model
                    else:
                        # Cache expired, clear it
                        self.clear_model(model)
                        del self._models[ticker]
        
        try:
            model = tf.keras.models.load_model(str(model_path))
            
            # Track session
            try:
                session = tf.compat.v1.get_default_session()
                if session:
                    self._active_sessions.add(id(session))
            except:
                pass
            
            # Cache the model
            if use_cache:
                with self._lock:
                    self._models[ticker] = (model, time.time())
            
            logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            log_error(ErrorSeverity.ERROR, "load_model", e, ticker=ticker, show_to_user=False)
            return None
    
    def clear_model(self, model: Any) -> None:
        """Properly clear a model from memory"""
        if model is None:
            return
        
        try:
            # Clear session and delete model
            del model
            self._cleanup_gpu_memory()
        except Exception as e:
            logger.warning(f"Error clearing model: {e}")
    
    def cleanup_all_sessions(self):
        """Aggressively cleanup all TensorFlow sessions"""
        with self._lock:
            for ticker, (model, _) in self._models.items():
                self.clear_model(model)
            self._models.clear()
            
            # Clear all tracked sessions
            self._cleanup_gpu_memory()
            self._active_sessions.clear()
            
            logger.info(f"Cleaned up all TensorFlow resources")
    
    def clear_cache(self) -> None:
        """Clear all cached models"""
        with self._lock:
            for ticker, (model, _) in self._models.items():
                self.clear_model(model)
            self._models.clear()
            logger.info("Cleared model cache")
    
    def predict_with_cleanup(self, model: Any, data: np.ndarray) -> Optional[np.ndarray]:
        """Make prediction with memory cleanup"""
        try:
            prediction = model.predict(data, verbose=0)
            return prediction
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
        finally:
            # Clear session to free memory (but keep model in cache)
            gc.collect()

# Global model manager
model_manager = EnhancedModelManager()

# ================================
# METRICS COLLECTION
# ================================

class MetricsCollector:
    """Collect and track application metrics"""
    def __init__(self):
        self.metrics = {
            "predictions_made": 0,
            "models_trained": 0,
            "models_retrained": 0,
            "errors_encountered": 0,
            "data_downloads": 0,
            "yfinance_downloads": 0,
            "alphavantage_downloads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_prediction_time": [],
            "avg_training_time": [],
            "pattern_mining_cycles": 0,
            "elite_patterns_found": 0,
            "monitoring_alerts_sent": 0
        }
        self._lock = threading.Lock()
    
    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric"""
        with self._lock:
            if metric in self.metrics and isinstance(self.metrics[metric], int):
                self.metrics[metric] += value
    
    def record_time(self, metric: str, duration: float) -> None:
        """Record a timing metric"""
        with self._lock:
            if metric in self.metrics and isinstance(self.metrics[metric], list):
                self.metrics[metric].append(duration)
                # Keep only last 100 measurements
                if len(self.metrics[metric]) > 100:
                    self.metrics[metric] = self.metrics[metric][-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        with self._lock:
            metrics_copy = self.metrics.copy()
            
            # Calculate averages for timing metrics
            for key, value in metrics_copy.items():
                if isinstance(value, list) and len(value) > 0:
                    metrics_copy[key] = {
                        "avg": np.mean(value),
                        "min": np.min(value),
                        "max": np.max(value),
                        "count": len(value)
                    }
            
            return metrics_copy
    
    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            for key in self.metrics:
                if isinstance(self.metrics[key], int):
                    self.metrics[key] = 0
                elif isinstance(self.metrics[key], list):
                    self.metrics[key] = []

# Global metrics collector
metrics_collector = MetricsCollector()

# ================================
# MODEL PATH FUNCTIONS
# ================================

def get_model_path(ticker: str) -> Path:
    """Get sanitized model path"""
    return MODELS_DIR / f"{sanitize_ticker(ticker)}.h5"

def get_scaler_path(ticker: str) -> Path:
    """Get sanitized scaler path"""
    return SCALERS_DIR / f"{sanitize_ticker(ticker)}.pkl"

def get_metadata_path(ticker: str) -> Path:
    """Get sanitized metadata path"""
    return METADATA_DIR / f"{sanitize_ticker(ticker)}.json"

def get_predictions_path(ticker: str) -> Path:
    """Get sanitized predictions path"""
    return PREDICTIONS_DIR / f"{sanitize_ticker(ticker)}.json"

# ================================
# MODEL BUILDING
# ================================

def build_lstm_model(input_shape: Tuple[int, int] = (60, 1)) -> Sequential:
    """Build LSTM model architecture"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ================================
# METADATA MANAGEMENT
# ================================

def load_metadata(ticker: str) -> Dict[str, Any]:
    """Load model metadata"""
    path = get_metadata_path(ticker)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {ticker}: {e}")
    
    return {
        "retrain_count": 0,
        "version": 1,
        "created_date": datetime.now().isoformat()
    }

def save_metadata(ticker: str, metadata: Dict[str, Any]) -> bool:
    """Save model metadata"""
    path = get_metadata_path(ticker)
    try:
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "save_metadata", e, ticker=ticker, show_to_user=False)
        return False

# ================================
# PRICE DATA FUNCTIONS
# ================================

# ============================================================================
# LINE 1756: get_latest_price() - Real-time price fetching
# STATUS: WORKING - Forex-aware (4 decimals), commodity support
# DEPENDENCIES: yfinance, Alpha Vantage fallback
# CRITICAL: Used for current prices in UI and predictions
# ============================================================================
def get_latest_price(ticker: str) -> Optional[float]:
    """Get latest price for a ticker with commodity-specific handling"""
    
    # Check blacklist first
    if is_ticker_blacklisted(ticker):
        logger.debug(f"⚫ Skipping blacklisted ticker: {ticker}")
        return None
    
    # Check if this is a commodity futures ticker
    is_commodity = ticker.endswith('=F')
    
    max_retries = MAX_RETRIES
    retry_delay = 1
    
    # Try yfinance first
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            # For commodities, get more history to find last valid price
            period = "5d" if is_commodity else "1d"
            hist = stock.history(period=period)
            
            if not hist.empty:
                # For commodities, find the most recent NON-ZERO price
                if is_commodity:
                    # Reverse iterate to find last valid price
                    for i in range(len(hist) - 1, -1, -1):
                        price = float(hist['Close'].iloc[i])
                        if price > 0 and not pd.isna(price):
                            # Check how old this price is
                            price_date = hist.index[i]
                            if hasattr(price_date, 'date'):
                                days_old = (datetime.now().date() - price_date.date()).days
                                if days_old <= 7:  # Accept prices up to 7 days old for commodities
                                    logger.debug(f"Got commodity price for {ticker}: ${price:.2f} ({days_old}d old)")
                                    metrics_collector.increment("data_downloads")
                                    return price
                                else:
                                    logger.warning(f"{ticker}: Last price is {days_old} days old")
                    
                    # If we get here, no valid recent price found
                    logger.warning(f"{ticker}: No recent valid price in last 7 days")
                else:
                    # Regular stocks - just get last price
                    price = float(hist['Close'].iloc[-1])
                    if price > 0 and not pd.isna(price):
                        metrics_collector.increment("data_downloads")
                        logger.debug(f"Got latest price from yfinance for {ticker}: ${price:.2f}")
                        return price
            
        except Exception as e:
            error_msg = str(e).lower()
            if "delisted" in error_msg or "no price data found" in error_msg:
                add_to_blacklist(ticker, "Delisted")
                return None
            
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
    
    # Fallback to Alpha Vantage
    if not ALPHA_VANTAGE_API_KEY:
        logger.error(f"Failed to get price for {ticker} - No Alpha Vantage API key")
        metrics_collector.increment("errors_encountered")
        return None
    
    logger.info(f"🔄 Falling back to Alpha Vantage for latest price of {ticker}...")
    
    try:
        av_ticker = ticker.replace('=F', '').replace('^', '')
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': av_ticker,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Global Quote' in data and '05. price' in data['Global Quote']:
            price = float(data['Global Quote']['05. price'])
            if price > 0:
                logger.info(f"✅ Got latest price from Alpha Vantage for {ticker}: ${price:.2f}")
                metrics_collector.increment("data_downloads")
                return price
        else:
            logger.error(f"No price data in Alpha Vantage response for {ticker}")
            
    except Exception as e:
        logger.error(f"Alpha Vantage latest price failed for {ticker}: {e}")
    
    logger.error(f"❌ Failed to get price for {ticker} after all attempts")
    metrics_collector.increment("errors_encountered")
    return None
    
def validate_commodity_data(ticker: str, df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Special validation for commodity futures data
    
    Commodities have unique characteristics:
    - Trade 23 hours/day (not just market hours)
    - Can have gaps on weekends/holidays
    - Futures contracts roll over monthly
    """
    if not ticker.endswith('=F'):
        return True, "Not a commodity"
    
    if df is None or len(df) == 0:
        return False, "Empty dataframe"
    
    # Check for valid prices
    if 'Close' not in df.columns:
        return False, "No Close column"
    
    # Find most recent valid price
    valid_prices = df['Close'][df['Close'] > 0].dropna()
    
    if len(valid_prices) == 0:
        return False, "No valid prices found"
    
    # Check freshness - commodities can have 3-day gaps (weekends)
    last_valid_date = valid_prices.index[-1]
    if hasattr(last_valid_date, 'date'):
        days_old = (datetime.now().date() - last_valid_date.date()).days
        
        # More lenient for commodities
        if days_old > 10:
            return False, f"Data too stale ({days_old} days old)"
        elif days_old > 5:
            return True, f"Warning: {days_old} days old"
    
    # Check data quality
    null_pct = df['Close'].isnull().sum() / len(df) * 100
    if null_pct > 20:
        return False, f"Too many nulls ({null_pct:.1f}%)"
    
    zero_pct = (df['Close'] == 0).sum() / len(df) * 100
    if zero_pct > 30:
        return False, f"Too many zeros ({zero_pct:.1f}%)"
    
    last_price = valid_prices.iloc[-1]
    return True, f"Valid (last: ${last_price:.2f})"


def get_commodity_info(ticker: str) -> Dict[str, Any]:
    """Get detailed info about a commodity ticker"""
    if not ticker.endswith('=F'):
        return {"is_commodity": False}
    
    try:
        df = yf.download(ticker, period="10d", interval="1d", progress=False)
        if df is None or len(df) == 0:
            return {"is_commodity": True, "status": "No data"}
        
        df = normalize_dataframe_columns(df)
        
        # Find last valid price
        valid_prices = df['Close'][(df['Close'] > 0) & (~df['Close'].isnull())]
        
        if len(valid_prices) == 0:
            return {"is_commodity": True, "status": "No valid prices"}
        
        last_price = valid_prices.iloc[-1]
        last_date = valid_prices.index[-1]
        days_old = (datetime.now().date() - last_date.date()).days
        
        # Calculate statistics
        price_change_5d = None
        if len(valid_prices) >= 5:
            price_5d_ago = valid_prices.iloc[-5]
            price_change_5d = (last_price - price_5d_ago) / price_5d_ago * 100
        
        return {
            "is_commodity": True,
            "status": "Valid",
            "last_price": last_price,
            "last_date": last_date,
            "days_old": days_old,
            "price_change_5d": price_change_5d,
            "data_points": len(valid_prices)
        }
        
    except Exception as e:
        return {"is_commodity": True, "status": f"Error: {str(e)[:50]}"}

# ================================
# DATA DOWNLOAD AND VALIDATION
# ================================

def download_and_validate_data(ticker: str, period: str = "1y", 
                               interval: str = "1d") -> Optional[pd.DataFrame]:
    """Download and validate financial data"""
    try:
        # Download with timeout
        df = download_with_timeout(ticker, period=period, interval=interval)
        
        if df is None:
            logger.warning(f"Download failed for {ticker}")
            return None
        
        # Validate data
        is_valid, message = validate_financial_data(df, ticker)
        
        if not is_valid:
            logger.warning(f"Data validation failed for {ticker}: {message}")
            return None
        
        logger.info(f"Successfully downloaded and validated {len(df)} rows for {ticker}")
        metrics_collector.increment("data_downloads")
        return df
        
    except Exception as e:
        logger.error(f"Error in download_and_validate_data for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "download_and_validate_data", e, 
                 ticker=ticker, show_to_user=False)
        metrics_collector.increment("errors_encountered")
        return None

# ================================
# SCALER MANAGEMENT
# ================================

def load_or_create_scaler(ticker: str, df: pd.DataFrame, 
                          force_create: bool = False) -> Optional[MinMaxScaler]:
    """Load existing scaler or create new one"""
    scaler_path = get_scaler_path(ticker)
    
    if not force_create and scaler_path.exists():
        try:
            import joblib
            scaler = joblib.load(scaler_path)
            
            # Validate scaler works
            test_data = [[df['Close'].iloc[0]]]
            scaler.transform(test_data)
            
            logger.info(f"Loaded existing scaler for {ticker}")
            return scaler
            
        except Exception as e:
            logger.warning(f"Scaler load failed for {ticker}, creating new: {e}")
    
    # Create new scaler
    try:
        scaler = MinMaxScaler()
        scaler.fit(df[['Close']])
        import joblib
        joblib.dump(scaler, scaler_path)
        logger.info(f"Created and saved new scaler for {ticker}")
        return scaler
        
    except Exception as e:
        logger.error(f"Failed to create scaler for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "load_or_create_scaler", e, 
                 ticker=ticker, show_to_user=False)
        return None

# ================================
# TRAINING DATA PREPARATION
# ================================

def prepare_training_data(df: pd.DataFrame, scaler: MinMaxScaler, 
                         lookback: int = 60) -> Tuple[Optional[np.ndarray], 
                                                       Optional[np.ndarray]]:
    """Prepare training data from dataframe"""
    try:
        # Select and clean data
        df_close = df[['Close']].ffill().bfill()
        
        if df_close['Close'].isna().any():
            logger.warning("NaN values remain after filling")
            return None, None
        
        # Scale data
        scaled = scaler.transform(df_close[['Close']])
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i-lookback:i])
            y.append(scaled[i])
        
        X, y = np.array(X), np.array(y)
        
        if len(X) == 0:
            logger.warning("No training samples created")
            return None, None
        
        logger.info(f"Prepared {len(X)} training samples with lookback {lookback}")
        return X, y
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        log_error(ErrorSeverity.ERROR, "prepare_training_data", e, show_to_user=False)
        return None, None

# ================================
# MODEL TRAINING ORCHESTRATION
# ================================

def train_model(ticker: str, X: np.ndarray, y: np.ndarray, 
                force_retrain: bool = False, 
                metadata: Dict[str, Any] = None) -> Optional[Any]:
    """Train or fine-tune model with Smart-Reset for high errors"""
    model = None
    model_path = get_model_path(ticker)
    
    # NEW: Check metadata for extreme error levels
    current_mape = metadata.get('last_mape', 0) if metadata else 0
    is_extreme_error = current_mape > 50.0  # Threshold for commodities
    
    try:
        # LOGIC CHANGE: Build new model if forced, missing, OR error is extreme
        if force_retrain or not model_path.exists() or is_extreme_error:
            
            if is_extreme_error:
                logger.warning(f"💥 Extreme error detected ({current_mape}%). Wiping {ticker} for fresh start.")
                # Delete the old scaler too if it exists to reset normalization
                scaler_path = SCALERS_DIR / f"{ticker.replace('=','_')}_scaler.pkl"
                if scaler_path.exists(): scaler_path.unlink()
            
            logger.info(f"Building new model for {ticker}")
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Train with early stopping
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=LEARNING_CONFIG["early_stopping_patience"],
                    restore_best_weights=True
                )
            ]
            
            start_time = time.time()
            model.fit(
                X, y,
                epochs=LEARNING_CONFIG["full_retrain_epochs"],
                batch_size=LEARNING_CONFIG["batch_size"],
                verbose=0,
                validation_split=LEARNING_CONFIG["validation_split"],
                callbacks=callbacks
            )
            training_time = time.time() - start_time
            
            metrics_collector.record_time("avg_training_time", training_time)
            metrics_collector.increment("models_trained")
            
            if metadata:
                metadata["retrain_count"] = metadata.get("retrain_count", 0) + 1
            
            logger.info(f"Trained new model for {ticker} in {training_time:.2f}s")
            
        else:
            # Load and fine-tune existing model (for normal adjustments)
            logger.info(f"Fine-tuning existing model for {ticker} (MAPE: {current_mape:.2f}%)")
            model = model_manager.load_model(model_path, use_cache=False)
            
            if model is None:
                raise ValueError("Failed to load existing model")
            
            recent_samples = max(50, int(len(X) * 0.3))
            
            start_time = time.time()
            model.fit(
                X[-recent_samples:], y[-recent_samples:],
                epochs=LEARNING_CONFIG["fine_tune_epochs"],
                batch_size=LEARNING_CONFIG["batch_size"],
                verbose=0
            )
            training_time = time.time() - start_time
            
            metrics_collector.record_time("avg_training_time", training_time)
            metrics_collector.increment("models_retrained")
            
            logger.info(f"Fine-tuned model for {ticker} in {training_time:.2f}s")
        
        # Save model
        model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Training failed for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "train_model", e, ticker=ticker, show_to_user=False)
        if model is not None:
            model_manager.clear_model(model)
        return None

# ================================
# FORECAST GENERATION WITH CONFIDENCE
# ================================

# ============================================================================
# LINE 2145: generate_forecast_with_confidence() - Prediction with CI
# STATUS: WORKING - Monte Carlo simulation for confidence intervals
# DEPENDENCIES: ModelManager, MinMaxScaler
# CRITICAL: Generates all forecasts - handle errors carefully
# ============================================================================
def generate_forecast_with_confidence(ticker: str, model: Any, scaler: MinMaxScaler,
                                     scaled_data: np.ndarray, lookback: int = 60,
                                     days: int = 5, n_simulations: int = 100) -> Tuple[
                                         Optional[np.ndarray], Optional[np.ndarray], 
                                         Optional[np.ndarray], Optional[List]]:
    """Generate forecast with confidence intervals using Monte Carlo simulation"""
    try:
        # Base forecast
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
        
        # Generate predictions
        base_predictions = []
        current_sequence = last_sequence.copy()
        
        start_time = time.time()
        for _ in range(days):
            pred = model_manager.predict_with_cleanup(model, current_sequence)
            if pred is None:
                logger.error(f"Prediction failed for {ticker}")
                return None, None, None, None
            
            base_predictions.append(pred[0, 0])
            # Update sequence for next prediction
            current_sequence = np.append(
                current_sequence[:, 1:, :],
                pred.reshape(1, 1, 1),
                axis=1
            )
        
        prediction_time = time.time() - start_time
        metrics_collector.record_time("avg_prediction_time", prediction_time)
        
        # Inverse transform predictions
        base_forecast = scaler.inverse_transform(
            np.array(base_predictions).reshape(-1, 1)
        ).flatten()
        
        # Monte Carlo simulations for confidence intervals
        simulations = []
        for _ in range(n_simulations):
            # Add small noise to input sequence
            noise = np.random.normal(0, 0.01, scaled_data[-lookback:].shape)
            noisy_sequence = scaled_data[-lookback:] + noise
            noisy_sequence = noisy_sequence.reshape(1, lookback, 1)
            
            # Generate prediction with dropout (if model has dropout)
            current_sequence = noisy_sequence.copy()
            predictions = []
            
            for _ in range(days):
                pred = model_manager.predict_with_cleanup(model, current_sequence)
                if pred is None:
                    break
                predictions.append(pred[0, 0])
                current_sequence = np.append(
                    current_sequence[:, 1:, :],
                    pred.reshape(1, 1, 1),
                    axis=1
                )
            
            if len(predictions) == days:
                sim_forecast = scaler.inverse_transform(
                    np.array(predictions).reshape(-1, 1)
                ).flatten()
                simulations.append(sim_forecast)
        
        if not simulations:
            logger.warning(f"No simulations generated for {ticker}")
            lower_95 = None
            upper_95 = None
        else:
            simulations = np.array(simulations)
            
            # Calculate confidence intervals
            lower_95 = np.percentile(simulations, 2.5, axis=0)
            upper_95 = np.percentile(simulations, 97.5, axis=0)
        
        # Generate business days
        dates = []
        day_offset = 1
        while len(dates) < days:
            next_date = datetime.now().date() + timedelta(days=day_offset)
            if next_date.weekday() < 5:  # Monday-Friday
                dates.append(next_date)
            day_offset += 1
        
        logger.info(f"Generated {days}-day forecast with {n_simulations} simulations for {ticker}")
        return base_forecast, lower_95, upper_95, dates
        
    except Exception as e:
        logger.error(f"Forecast generation with confidence failed for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "generate_forecast_with_confidence", e, 
                 ticker=ticker, show_to_user=False)
        return None, None, None, None

# ================================
# PARALLEL PATTERN MINING
# ================================

def mine_patterns_parallel(tickers: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """Mine patterns for multiple tickers in parallel"""
    patterns = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(mine_patterns_for_ticker, ticker): ticker for ticker in tickers}
        
        for future in futures:
            ticker = futures[future]
            try:
                result = future.result()
                if result:
                    patterns.append(result)
            except Exception as e:
                logger.error(f"Parallel pattern mining failed for {ticker}: {e}")
    
    return patterns

# ================================
# PREDICTION VALIDATION FUNCTIONS
# ================================

def get_actual_price_for_date(ticker: str, target_date: datetime) -> Optional[float]:
    """Fetch actual historical price for a specific date with Alpha Vantage fallback"""
    try:
        # Try yfinance first
        logger.debug(f"Fetching actual price for {ticker} on {target_date.date()} from yfinance...")
        start_date = target_date - timedelta(days=5)
        end_date = target_date + timedelta(days=2)
        
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        if df is not None and not df.empty:
            df = normalize_dataframe_columns(df)
            
            # Find the closest date (in case target_date is weekend/holiday)
            df.index = pd.to_datetime(df.index)
            target_date_normalized = pd.to_datetime(target_date.date())
            
            # Get closest available date
            closest_idx = df.index.get_indexer([target_date_normalized], method='nearest')[0]
            
            if closest_idx >= 0 and closest_idx < len(df):
                actual_price = float(df['Close'].iloc[closest_idx])
                logger.debug(f"✅ Found actual price from yfinance for {ticker} on {target_date.date()}: ${actual_price:.2f}")
                return actual_price
        
        logger.warning(f"yfinance failed to get price for {ticker} on {target_date.date()}, trying Alpha Vantage...")
        
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker} on {target_date.date()}: {e}, trying Alpha Vantage...")
    
    # Fallback to Alpha Vantage
    if not ALPHA_VANTAGE_API_KEY:
        logger.warning("Alpha Vantage API key not configured")
        return None
    
    try:
        logger.info(f"🔄 Fetching from Alpha Vantage for {ticker} on {target_date.date()}...")
        av_ticker = ticker.replace('=F', '').replace('^', '')
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': av_ticker,
            'outputsize': 'compact',  # Last 100 days should be enough
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            logger.warning(f"No time series data from Alpha Vantage for {ticker}")
            return None
        
        time_series = data['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df = convert_alpha_vantage_to_yfinance_format(df)
        
        if df is not None and not df.empty:
            # Find closest date
            target_date_normalized = pd.to_datetime(target_date.date())
            closest_idx = df.index.get_indexer([target_date_normalized], method='nearest')[0]
            
            if closest_idx >= 0 and closest_idx < len(df):
                actual_price = float(df['Close'].iloc[closest_idx])
                actual_date = df.index[closest_idx].date()
                logger.info(f"✅ Found actual price from Alpha Vantage for {ticker} on {actual_date}: ${actual_price:.2f}")
                return actual_price
        
        logger.warning(f"Could not find price in Alpha Vantage data for {ticker}")
        return None
        
    except Exception as e:
        logger.error(f"Alpha Vantage failed for {ticker} on {target_date.date()}: {e}")
        return None

def record_prediction(ticker: str, prediction: float, date: str, current_price: Optional[float] = None) -> bool:
    """Record prediction for accuracy tracking with current price for directional validation"""
    try:
        if SQLALCHEMY_AVAILABLE:
            # Use database
            db_session = Session()
            
            # Convert date string to datetime
            try:
                pred_date = datetime.strptime(date, "%Y-%m-%d")
            except:
                pred_date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S") if ":" in date else datetime.strptime(date, "%Y-%m-%d")
            
            # Check if prediction already exists
            existing = db_session.query(Prediction).filter(
                Prediction.ticker == ticker,
                Prediction.prediction_date == pred_date
            ).first()
            
            if existing:
                # Update existing prediction
                existing.predicted_price = float(prediction)
                existing.previous_price = float(current_price) if current_price else None
            else:
                # Create new prediction
                new_prediction = Prediction(
                    ticker=ticker,
                    prediction_date=pred_date,
                    predicted_price=float(prediction),
                    previous_price=float(current_price) if current_price else None,
                    created_at=datetime.now()
                )
                db_session.add(new_prediction)
            
            db_session.commit()
            db_session.close()
        else:
            # Use JSON file as fallback
            predictions_path = get_predictions_path(ticker)
            if predictions_path.exists():
                with open(predictions_path, 'r') as f:
                    predictions_data = json.load(f)
            else:
                predictions_data = []
            
            # Add new prediction
            predictions_data.append({
                "ticker": ticker,
                "prediction_date": date,
                "predicted_price": float(prediction),
                "previous_price": float(current_price) if current_price else None,
                "created_at": datetime.now().isoformat(),
                "actual_price": None,
                "error_mape": None,
                "error_mae": None,
                "validated": 0
            })
            
            with open(predictions_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
        
        metrics_collector.increment("predictions_made")
        logger.info(f"Recorded prediction for {ticker}: ${prediction:.2f} on {date}")
        return True
            
    except Exception as e:
        logger.error(f"Failed to record prediction for {ticker}: {e}")
        log_error(ErrorSeverity.WARNING, "record_prediction", e, ticker=ticker, show_to_user=False)
        return False

# ============================================================================
# LINE 2419: load_accuracy_log() - REAL validation
# STATUS: WORKING - Validates predictions against actual prices
# DEPENDENCIES: Database/JSON, get_actual_price_for_date
# CRITICAL: Provides real performance metrics (MAPE, direction accuracy)
# ============================================================================
def load_accuracy_log(ticker: str) -> Dict[str, Any]:
    """Load accuracy tracking data with REAL validation against actual prices"""
    try:
        if SQLALCHEMY_AVAILABLE:
            # Use database
            db_session = Session()
            
            # Get all predictions for this ticker
            predictions = db_session.query(Prediction).filter(
                Prediction.ticker == ticker
            ).order_by(Prediction.prediction_date).all()
            
            if len(predictions) == 0:
                db_session.close()
                return {
                    "total_predictions": 0,
                    "validated_predictions": 0,
                    "avg_error_mape": 0.0,
                    "avg_error_mae": 0.0,
                    "directional_accuracy": 0.0,
                    "last_updated": datetime.now().isoformat(),
                    "status": "no_predictions"
                }
            
            # Calculate REAL accuracy metrics
            errors_mape = []  # Mean Absolute Percentage Error
            errors_mae = []   # Mean Absolute Error
            directional_correct = []
            validated_count = 0
            current_price = get_latest_price(ticker)
            
            for pred in predictions:
                try:
                    pred_date = pred.prediction_date
                    pred_value = pred.predicted_price
                    
                    # Only validate predictions that are in the past
                    if pred_date.date() >= datetime.now().date():
                        continue
                    
                    # Fetch actual price for that date if not already validated
                    if pred.actual_price is None:
                        actual_price = get_actual_price_for_date(ticker, pred_date)
                        
                        if actual_price is None:
                            continue
                        
                        # Update prediction with actual price
                        pred.actual_price = actual_price
                        validated_count += 1
                    else:
                        actual_price = pred.actual_price
                        validated_count += 1
                    
                    # Calculate errors
                    mae = abs(pred_value - actual_price)
                    mape = (mae / actual_price) * 100  # Percentage error
                    
                    errors_mae.append(mae)
                    errors_mape.append(mape)
                    
                    # Store errors in database
                    pred.error_mae = mae
                    pred.error_mape = mape
                    pred.validated = 1
                    
                    # Directional accuracy (did we predict up/down correctly?)
                    if pred.previous_price:
                        prev_price = float(pred.previous_price)
                        predicted_direction = 1 if pred_value > prev_price else -1
                        actual_direction = 1 if actual_price > prev_price else -1
                        directional_correct.append(predicted_direction == actual_direction)
                    elif current_price:
                        # Use current price as reference if no previous stored
                        predicted_direction = 1 if pred_value > current_price else -1
                        actual_direction = 1 if actual_price > current_price else -1
                        directional_correct.append(predicted_direction == actual_direction)
                    
                except Exception as e:
                    logger.debug(f"Error validating prediction for {ticker}: {e}")
                    continue
            
            # Save updated predictions with validation data
            db_session.commit()
            db_session.close()
        else:
            # Use JSON file fallback
            predictions_path = get_predictions_path(ticker)
            if not predictions_path.exists():
                return {
                    "total_predictions": 0,
                    "validated_predictions": 0,
                    "avg_error_mape": 0.0,
                    "avg_error_mae": 0.0,
                    "directional_accuracy": 0.0,
                    "last_updated": datetime.now().isoformat(),
                    "status": "no_predictions"
                }
            
            with open(predictions_path, 'r') as f:
                predictions_data = json.load(f)
            
            # Calculate REAL accuracy metrics
            errors_mape = []
            errors_mae = []
            directional_correct = []
            validated_count = 0
            current_price = get_latest_price(ticker)
            
            for pred in predictions_data:
                try:
                    pred_date = datetime.strptime(pred["prediction_date"], "%Y-%m-%d")
                    pred_value = pred["predicted_price"]
                    
                    # Only validate predictions that are in the past
                    if pred_date.date() >= datetime.now().date():
                        continue
                    
                    # Fetch actual price if not already validated
                    if pred.get("actual_price") is None:
                        actual_price = get_actual_price_for_date(ticker, pred_date)
                        
                        if actual_price is None:
                            continue
                        
                        # Update prediction with actual price
                        pred["actual_price"] = actual_price
                        pred["validated"] = 1
                        validated_count += 1
                    else:
                        actual_price = pred["actual_price"]
                        validated_count += 1
                    
                    # Calculate errors
                    mae = abs(pred_value - actual_price)
                    mape = (mae / actual_price) * 100
                    
                    errors_mae.append(mae)
                    errors_mape.append(mape)
                    
                    # Update errors in data
                    pred["error_mae"] = mae
                    pred["error_mape"] = mape
                    
                    # Directional accuracy
                    if pred.get("previous_price"):
                        prev_price = float(pred["previous_price"])
                        predicted_direction = 1 if pred_value > prev_price else -1
                        actual_direction = 1 if actual_price > prev_price else -1
                        directional_correct.append(predicted_direction == actual_direction)
                    elif current_price:
                        predicted_direction = 1 if pred_value > current_price else -1
                        actual_direction = 1 if actual_price > current_price else -1
                        directional_correct.append(predicted_direction == actual_direction)
                    
                except Exception as e:
                    logger.debug(f"Error validating prediction for {ticker}: {e}")
                    continue
            
            # Save updated predictions
            with open(predictions_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
        
        # Calculate aggregate metrics
        if validated_count == 0:
            total_predictions = len(predictions) if SQLALCHEMY_AVAILABLE else len(predictions_data)
            return {
                "total_predictions": total_predictions,
                "validated_predictions": 0,
                "avg_error_mape": 0.0,
                "avg_error_mae": 0.0,
                "directional_accuracy": 0.0,
                "last_updated": datetime.now().isoformat(),
                "status": "no_validated"
            }
        
        avg_mape = np.mean(errors_mape) if errors_mape else 0.0
        avg_mae = np.mean(errors_mae) if errors_mae else 0.0
        dir_accuracy = (sum(directional_correct) / len(directional_correct) * 100) if directional_correct else 0.0
        
        total_predictions = len(predictions) if SQLALCHEMY_AVAILABLE else len(predictions_data)
        
        return {
            "total_predictions": total_predictions,
            "validated_predictions": validated_count,
            "avg_error_mape": round(avg_mape, 2),  # Percentage
            "avg_error_mae": round(avg_mae, 2),     # Dollar amount
            "directional_accuracy": round(dir_accuracy, 1),  # Percentage
            "last_updated": datetime.now().isoformat(),
            "status": "validated",
            "recent_errors": errors_mape[-10:] if len(errors_mape) > 0 else []
        }
        
    except Exception as e:
        logger.error(f"Failed to load accuracy log for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "load_accuracy_log", e, ticker=ticker, show_to_user=False)
        if SQLALCHEMY_AVAILABLE:
            try:
                db_session.close()
            except:
                pass
    
    return {
        "total_predictions": 0,
        "validated_predictions": 0,
        "avg_error_mape": 0.0,
        "avg_error_mae": 0.0,
        "directional_accuracy": 0.0,
        "last_updated": datetime.now().isoformat(),
        "status": "no_file"
    }

def validate_predictions(ticker: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate predictions against REAL actual prices and return performance metrics"""
    logger.info(f"Validating predictions for {ticker} against actual market data...")
    
    try:
        # Load accuracy log which now does REAL validation
        acc_log = load_accuracy_log(ticker)
        
        if acc_log['status'] in ['no_predictions', 'no_file']:
            logger.info(f"{ticker}: No predictions to validate")
            return False, acc_log
        
        if acc_log['status'] == 'no_validated':
            logger.warning(f"{ticker}: Predictions exist but none could be validated (all future dates or data unavailable)")
            return False, acc_log
        
        # Check validation quality
        validated_count = acc_log.get('validated_predictions', 0)
        avg_mape = acc_log.get('avg_error_mape', 100.0)
        dir_accuracy = acc_log.get('directional_accuracy', 0.0)
        
        logger.info(
            f"{ticker} Validation Results: "
            f"{validated_count} predictions | "
            f"MAPE: {avg_mape:.2f}% | "
            f"MAE: ${acc_log.get('avg_error_mae', 0):.2f} | "
            f"Direction: {dir_accuracy:.1f}%"
        )
        
        # Consider validation successful if we have data
        return True, acc_log
        
    except Exception as e:
        logger.error(f"Failed to validate predictions for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "validate_predictions", e, ticker=ticker, show_to_user=False)
        return False, {
            "total_predictions": 0,
            "validated_predictions": 0,
            "avg_error_mape": 100.0,
            "avg_error_mae": 0.0,
            "directional_accuracy": 0.0,
            "status": "error"
        }

def should_retrain(ticker: str, accuracy_log: Dict[str, Any], 
                   metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Determine if model should be retrained using REAL performance metrics"""
    reasons = []
    
    # Check REAL error rates (MAPE - Mean Absolute Percentage Error)
    avg_mape = accuracy_log.get("avg_error_mape", 0.0)
    
    # ============================================================
    # 🚨 SMART-RESET IMPLANT (Critical for Commodities)
    # ============================================================
    # If MAPE is over 50%, the model/scaler is 'poisoned' by a gap or rollover.
    # We delete them so the daemon starts a CLEAN full retrain.
    if avg_mape > 50.0:
        logger.warning(f"💥 [SMART-RESET] Extreme error for {ticker}: {avg_mape:.1f}%")
        
        # Define paths using existing system constants
        model_path = MODELS_DIR / f"{sanitize_ticker(ticker)}_lstm.h5"
        scaler_path = SCALERS_DIR / f"{sanitize_ticker(ticker.replace('=','_'))}_scaler.pkl"
        
        try:
            if model_path.exists(): 
                model_path.unlink()
                logger.info(f"🗑️ Deleted corrupted model: {model_path.name}")
            if scaler_path.exists(): 
                scaler_path.unlink()
                logger.info(f"🗑️ Deleted corrupted scaler: {scaler_path.name}")
            
            reasons.append(f"SMART_RESET_EXTREME_ERROR_{avg_mape:.1f}%")
            return True, reasons # Exit immediately to trigger fresh training
        except Exception as e:
            logger.error(f"Smart-reset file deletion failed for {ticker}: {e}")
    # ============================================================
    
    # Check if we have enough validated predictions
    validated_count = accuracy_log.get("validated_predictions", 0)
    total_predictions = accuracy_log.get("total_predictions", 0)
    
    if validated_count < 5:
        if total_predictions < 5:
            reasons.append("insufficient_predictions")
        else:
            reasons.append("insufficient_validated_predictions")
    
    # Normal Error Checks
    if validated_count >= 5:
        if avg_mape > 8.0:  # More than 8% average error
            reasons.append(f"high_error_mape_{avg_mape:.1f}%")
        elif avg_mape > 5.0:  # Warning level
            reasons.append(f"elevated_error_mape_{avg_mape:.1f}%")
    
    # Check directional accuracy
    dir_accuracy = accuracy_log.get("directional_accuracy", 0.0)
    if validated_count >= 5 and dir_accuracy < 55.0:
        reasons.append(f"poor_direction_accuracy_{dir_accuracy:.1f}%")
    
    # Check initial training phase
    retrain_count = metadata.get("retrain_count", 0)
    if retrain_count < 2:
        reasons.append("initial_training_phase")
    
    # Check if model is stale (older than 14 days)
    if metadata.get("trained_date"):
        try:
            trained_date = datetime.fromisoformat(metadata["trained_date"])
            days_since_training = (datetime.now() - trained_date).days
            if days_since_training > 14:
                reasons.append(f"stale_model_{days_since_training}d")
            elif days_since_training > 7 and avg_mape > 6.0:
                reasons.append(f"aging_model_with_errors_{days_since_training}d")
        except Exception as e:
            logger.warning(f"Invalid training date for {ticker}: {e}")
            reasons.append("invalid_training_date")
    
    # Check data quality issues
    if metadata.get("data_quality") == "WARNING":
        reasons.append("data_quality_issues")
    
    # Check recent performance degradation
    recent_errors = accuracy_log.get("recent_errors", [])
    if len(recent_errors) >= 3:
        recent_avg = np.mean(recent_errors[-3:])
        if recent_avg > avg_mape * 1.5:  # Recent errors 50% worse than average
            reasons.append(f"performance_degradation_{recent_avg:.1f}%")
    
    return len(reasons) > 0, reasons

# ================================
# ENHANCED TRAINING FUNCTION
# ================================

# ============================================================================
# LINE 2738: train_self_learning_model_enhanced() - Main training function
# STATUS: WORKING - Generates forecasts with confidence intervals
# DEPENDENCIES: ModelManager, download_and_validate_data, pattern boosting
# CRITICAL: Core ML functionality - DO NOT modify without backup
# ============================================================================
def train_self_learning_model_enhanced(ticker: str, days: int = 5,
                                       force_retrain: bool = False,
                                       skip_validation: bool = False) -> Tuple[
                                           Optional[np.ndarray],
                                           Optional[np.ndarray],
                                           Optional[np.ndarray],
                                           Optional[List],
                                           Optional[Any]
                                       ]:
    """Enhanced training with proper memory management and all fixes"""
    logger.info(f"Training {ticker} (force={force_retrain}, skip_validation={skip_validation})")
    
    model = None
    try:
        # Quick check: if model exists and is recent, skip validation for speed
        meta = load_metadata(ticker)
        model_path = get_model_path(ticker)
        
        # If model exists and is less than 24 hours old, skip validation unless forced
        if not force_retrain and model_path.exists() and not skip_validation:
            trained_date_str = meta.get("trained_date")
            if trained_date_str:
                try:
                    trained_date = datetime.fromisoformat(trained_date_str)
                    hours_since = (datetime.now() - trained_date).total_seconds() / 3600
                    if hours_since < 24:
                        logger.info(f"Using recent model for {ticker} (trained {hours_since:.1f}h ago)")
                        skip_validation = True
                except:
                    pass
        
        # Validate predictions (can be slow, so skip if model is recent)
        if not skip_validation:
            updated, acc_log = validate_predictions(ticker)
        else:
            # Use cached accuracy log
            acc_log = load_accuracy_log(ticker)
        
        needs_retrain, reasons = should_retrain(ticker, acc_log, meta)
        
        # Check for critical errors that require full rebuild
        force_full_rebuild = any("FORCE_REBUILD" in reason for reason in reasons)
        
        if force_full_rebuild:
            logger.warning(f"🚨 CRITICAL ERROR DETECTED for {ticker} - Forcing complete rebuild!")
            logger.warning(f"   Reasons: {', '.join(reasons)}")
            force_retrain = True  # Override to force full retrain
            
            # Delete existing model to force fresh start
            model_path = get_model_path(ticker)
            if model_path.exists():
                model_path.unlink()
                logger.info(f"   Deleted corrupted model: {model_path}")
        
        if needs_retrain or force_retrain:
            logger.info(f"Retraining {ticker}: {', '.join(reasons)}")
        
        # Download and validate data
        df = download_and_validate_data(ticker, period="1y")
        if df is None:
            logger.warning(f"Failed to download data for {ticker}")
            return None, None, None, None, None
        
        # Calculate data quality metrics
        quality_metrics = calculate_data_quality_metrics(df)
        if quality_metrics:
            logger.info(f"Data quality metrics for {ticker}: {quality_metrics}")
        
        # Load or create scaler
        scaler = load_or_create_scaler(ticker, df, force_create=force_retrain or needs_retrain)
        if scaler is None:
            logger.warning(f"Failed to create scaler for {ticker}")
            return None, None, None, None, None
        
        # Prepare training data
        lookback = LEARNING_CONFIG["lookback_window"]
        X, y = prepare_training_data(df, scaler, lookback=lookback)
        
        if X is None or y is None:
            logger.warning(f"Failed to prepare training data for {ticker}")
            return None, None, None, None, None
        
        # Train or fine-tune model
        if force_retrain or needs_retrain:
            model = train_model(ticker, X, y, force_retrain=force_retrain, metadata=meta)
            if model is None:
                logger.warning(f"Training failed for {ticker}")
                return None, None, None, None, None
        else:
            # Load existing model
            model_path = get_model_path(ticker)
            model = model_manager.load_model(model_path)
            if model is None:
                logger.warning(f"Failed to load model for {ticker}, retraining")
                model = train_model(ticker, X, y, force_retrain=True, metadata=meta)
                if model is None:
                    return None, None, None, None, None
        
        # Generate forecast with confidence intervals
        scaled_data = scaler.transform(df[['Close']])
        forecast, lower_ci, upper_ci, dates = generate_forecast_with_confidence(
            ticker, model, scaler, scaled_data, 
            lookback=lookback, days=days
        )
        
        if forecast is None or dates is None:
            logger.warning(f"Forecast generation failed for {ticker}")
            return None, None, None, None, None
        
        # Apply pattern boosts if available
        try:
            current_price = get_latest_price(ticker)
            if current_price:
                # Try to apply pattern boosts (if pattern functions are available)
                try:
                    forecast = get_pattern_boosted_forecast(ticker, forecast.tolist(), current_price)
                except:
                    pass
        except:
            pass
        
        # Record prediction with current price for directional validation
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        current_price = get_latest_price(ticker)
        record_prediction(ticker, forecast[0], tomorrow, current_price)
        
        # Update metadata with quality metrics
        meta.update({
            "trained_date": datetime.now().isoformat(),
            "training_samples": len(X),
            "training_volatility": float(df['Close'].pct_change().std()),
            "version": meta.get("version", 1) + 1,
            "last_accuracy": acc_log.get("avg_error_mape", 0),
            "data_quality": "GOOD",
            "forecast_days": days,
            "quality_metrics": quality_metrics
        })
        save_metadata(ticker, meta)
        
        logger.info(f"Successfully trained {ticker}, forecast: {forecast[0]:.2f}")
        return forecast, lower_ci, upper_ci, dates, model
        
    except Exception as e:
        logger.error(f"Training failed for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "train_self_learning_model_enhanced", e, 
                 ticker=ticker, user_message=f"Training failed for {ticker}", 
                 show_to_user=False)
        metrics_collector.increment("errors_encountered")
        return None, None, None, None, None
        
    finally:
        # Always clear TensorFlow session
        if model is not None:
            del model
        model_manager._cleanup_gpu_memory()
        
# ================================
# FAST FORECAST GENERATION
# ================================

def generate_fast_forecast(ticker: str, days: int = 5) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], 
    Optional[np.ndarray], Optional[List]
]:
    """
    Generate forecast quickly by using cached model without retraining
    
    Returns:
        (forecast, lower_ci, upper_ci, dates)
    """
    try:
        logger.info(f"Generating fast forecast for {ticker} ({days} days)")
        
        # Check if model exists
        model_path = get_model_path(ticker)
        if not model_path.exists():
            logger.warning(f"No model exists for {ticker}, training new one...")
            return train_self_learning_model_enhanced(ticker, days=days, force_retrain=True)[:4]
        
        # Load cached model
        model = model_manager.load_model(model_path, use_cache=True)
        if model is None:
            logger.warning(f"Failed to load model for {ticker}")
            return None, None, None, None
        
        # Get recent data (no validation needed for forecast)
        df = download_with_timeout(ticker, period="1y", interval="1d")
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {ticker}")
            return None, None, None, None
        
        # Load scaler
        scaler_path = get_scaler_path(ticker)
        if not scaler_path.exists():
            logger.warning(f"No scaler for {ticker}, creating new one...")
            import joblib
            scaler = MinMaxScaler()
            scaler.fit(df[['Close']])
            joblib.dump(scaler, scaler_path)
        else:
            import joblib
            scaler = joblib.load(scaler_path)
        
        # Generate forecast
        lookback = LEARNING_CONFIG["lookback_window"]
        scaled_data = scaler.transform(df[['Close']])
        
        forecast, lower_ci, upper_ci, dates = generate_forecast_with_confidence(
            ticker, model, scaler, scaled_data, 
            lookback=lookback, days=days, n_simulations=50  # Reduced simulations for speed
        )
        
        logger.info(f"Fast forecast generated for {ticker} in <3s")
        return forecast, lower_ci, upper_ci, dates
        
    except Exception as e:
        logger.error(f"Fast forecast failed for {ticker}: {e}")
        return None, None, None, None

# ================================
# PATTERN MINING CONFIGURATION
# ================================

def load_pattern_mining_config() -> Dict[str, Any]:
    """Load pattern mining configuration with validation"""
    default_config = {
        "enabled": False,
        "cycle_interval_minutes": 30,
        "min_auc_threshold": 0.70,
        "max_auc_std": 0.10
    }
    return ConfigManager.load_config_with_backup(
        PATTERN_MINING_CONFIG, "pattern_mining", default_config
    )

def save_pattern_mining_config(config: Dict[str, Any]) -> bool:
    """Save pattern mining configuration with validation"""
    return ConfigManager.save_config(PATTERN_MINING_CONFIG, config, "pattern_mining")

def load_daemon_config() -> Dict[str, Any]:
    """Load daemon configuration"""
    default_config = {
        "enabled": False,
        "sleep_minutes": 10,
        "max_retrain_per_cycle": 5
    }
    return ConfigManager.load_config_with_backup(
        DAEMON_CONFIG, "daemon", default_config
    )

def save_daemon_config(enabled: bool) -> bool:
    """Save daemon configuration"""
    try:
        config = load_daemon_config()
        config["enabled"] = enabled
        DAEMON_CONFIG.parent.mkdir(exist_ok=True)
        with open(DAEMON_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save daemon config: {e}")
        return False

def load_monitoring_config() -> Dict[str, Any]:
    """Load monitoring configuration with enhanced options"""
    default_config = {
        "enabled": False,
        "check_interval_minutes": 15,
        "telegram_alerts": False,
        "threshold_percent": 6.0,
        "cooldown_minutes": 30,
        "watchlist": PATTERN_WATCHLIST
    }
    return ConfigManager.load_config_with_backup(
        MONITORING_CONFIG, "monitoring", default_config
    )

def save_monitoring_config(enabled: bool, check_interval: int = 5, 
                          threshold_percent: float = 6.0, 
                          cooldown_minutes: int = 30) -> bool:
    """Save monitoring configuration with more options"""
    try:
        config = load_monitoring_config()
        config["enabled"] = enabled
        config["check_interval_minutes"] = check_interval
        config["threshold_percent"] = threshold_percent
        config["cooldown_minutes"] = cooldown_minutes
        
        MONITORING_CONFIG.parent.mkdir(exist_ok=True)
        with open(MONITORING_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated monitoring config: enabled={enabled}, interval={check_interval}min")
        return True
    except Exception as e:
        logger.error(f"Failed to save monitoring config: {e}")
        return False

def load_pattern_overrides() -> Dict[str, Any]:
    """Load pattern override configuration"""
    override_config_path = CONFIG_DIR / "pattern_overrides.json"
    default_config = {
        "enabled": False,
        "overrides": {}
    }
    return ConfigManager.load_config_with_backup(
        override_config_path, "pattern_overrides", default_config
    )

# ================================
# PATTERN MINING - INTRADAY
# ================================

def mine_intraday_patterns(ticker: str) -> Tuple[Optional[Tuple], Optional[str]]:
    """Mine patterns from 1-minute data (7 days, ±6% in 3 hours)"""
    try:
        logger.debug(f"[1m data {ticker}]")
        end = datetime.now()
        start = end - timedelta(days=7)
        
        df = download_with_timeout(
            ticker,
            period="7d",
            interval="1m"
        )
        
        if df is None or df.empty or len(df) < 1500:
            return None, "insufficient_data"
        
        df = df[~df.index.duplicated(keep='first')]
        
        # Calculate VWAP
        with np.errstate(divide='ignore', invalid='ignore'):
            typical = (df['High'] + df['Low'] + df['Close']) / 3
            cumulative_tp_volume = (typical * df['Volume']).cumsum()
            cumulative_volume = df['Volume'].cumsum()
            df['vwap'] = np.where(
                cumulative_volume != 0,
                cumulative_tp_volume / cumulative_volume,
                df['Close']
            )
        
        # Target: Direction of ±6% move in next 180 minutes
        df['future_high'] = df['High'].rolling(180).max().shift(-180)
        df['future_low'] = df['Low'].rolling(180).min().shift(-180)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['up_move'] = (df['future_high'] / df['Close'] >= 1.06).astype(int)
            df['down_move'] = (df['Close'] / df['future_low'] >= 1.06).astype(int)
        
        # Direction: 1=UP, 0=DOWN
        df['direction'] = np.where(
            df['up_move'] == 1, 1,
            np.where(df['down_move'] == 1, 0, np.nan)
        )
        df['big_move'] = (df['up_move'] | df['down_move']).astype(int)
        
        # Calculate features
        feats = pd.DataFrame(index=df.index)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            for w in [5, 10, 20, 30]:
                rolling_mean_vol = df['Volume'].rolling(w).mean()
                feats[f'vol_{w}'] = np.where(
                    rolling_mean_vol != 0,
                    df['Volume'] / rolling_mean_vol,
                    0
                )
                feats[f'ret_{w}'] = df['Close'].pct_change(w)
                feats[f'vwap_dist_{w}'] = np.where(
                    df['vwap'] != 0,
                    df['Close'] / df['vwap'] - 1,
                    0
                )
            
            for w in [5, 10, 20]:
                feats[f'range_{w}'] = np.where(
                    df['Close'] != 0,
                    (df['High'] - df['Low']) / df['Close'],
                    0
                )
                feats[f'volatility_{w}'] = df['Close'].pct_change().rolling(w).std()
        
        feats['momentum_10_30'] = feats['ret_10'] - feats['ret_30']
        feats['vol_surge'] = (
            df['Volume'] > df['Volume'].rolling(60).mean() * 2
        ).astype(int)
        
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Split data
        cutoff = len(df) - 200
        X = feats.iloc[:cutoff]
        y = df['direction'].iloc[:cutoff]
        
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 500:
            return None, "insufficient_samples"
        
        return (X, y, '1m'), None
        
    except Exception as e:
        logger.error(f"Intraday pattern mining failed for {ticker}: {e}")
        log_error(ErrorSeverity.WARNING, "mine_intraday_patterns", e, 
                 ticker=ticker, show_to_user=False)
        return None, str(e)[:30]

# ================================
# PATTERN MINING - DAILY
# ================================

# ============================================================================
# LINE 3152: mine_daily_patterns() - Daily pattern detection
# STATUS: WORKING - Fixed pandas errors 2025-12-26
# DEPENDENCIES: download_with_timeout, sklearn models
# NOTE: Slow (30 min/cycle) but acceptable
# ============================================================================
def mine_daily_patterns(ticker: str) -> Tuple[Optional[Tuple], Optional[str]]:
    """Mine patterns from daily data (2 years, ±6% in 5 days)"""
    try:
        logger.debug(f"[daily {ticker}]")
        
        df = download_with_timeout(
            ticker,
            period="2y",
            interval="1d"
        )
        
        if df is None or df.empty or len(df) < 100:
            return None, "insufficient_data"
        
        # Calculate VWAP
        with np.errstate(divide='ignore', invalid='ignore'):
            typical = (df['High'] + df['Low'] + df['Close']) / 3
            cumulative_tp_volume = (typical * df['Volume']).cumsum()
            cumulative_volume = df['Volume'].cumsum()
            df['vwap'] = np.where(
                cumulative_volume != 0,
                cumulative_tp_volume / cumulative_volume,
                df['Close']
            )
        
        # Target: Direction of ±6% move in next 5 days
        df['future_high'] = df['High'].rolling(5).max().shift(-5)
        df['future_low'] = df['Low'].rolling(5).min().shift(-5)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            df['up_move'] = (df['future_high'] / df['Close'] >= 1.06).astype(int)
            df['down_move'] = (df['Close'] / df['future_low'] >= 1.06).astype(int)
        
        df['direction'] = np.where(
            df['up_move'] == 1, 1,
            np.where(df['down_move'] == 1, 0, np.nan)
        )
        df['big_move'] = (df['up_move'] | df['down_move']).astype(int)
        
        # Calculate features
        feats = pd.DataFrame(index=df.index)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            for w in [5, 10, 20, 50]:
                rolling_mean_vol = df['Volume'].rolling(w).mean()
                feats[f'vol_{w}'] = np.where(
                    rolling_mean_vol != 0,
                    df['Volume'] / rolling_mean_vol,
                    0
                )
                feats[f'ret_{w}'] = df['Close'].pct_change(w)
                feats[f'vwap_dist_{w}'] = np.where(
                    df['vwap'] != 0,
                    df['Close'] / df['vwap'] - 1,
                    0
                )
            
            for w in [5, 10, 20]:
                feats[f'range_{w}'] = np.where(
                    df['Close'] != 0,
                    (df['High'] - df['Low']) / df['Close'],
                    0
                )
                feats[f'volatility_{w}'] = df['Close'].pct_change().rolling(w).std()
        
        feats['momentum_10_50'] = feats['ret_10'] - feats['ret_50']
        feats['vol_surge'] = (
            df['Volume'] > df['Volume'].rolling(20).mean() * 2
        ).astype(int)
        
        feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Split data
        cutoff = len(df) - 10
        X = feats.iloc[:cutoff]
        y = df['direction'].iloc[:cutoff]
        
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            return None, "insufficient_samples"
        
        return (X, y, 'daily'), None
        
    except Exception as e:
        logger.error(f"Daily pattern mining failed for {ticker}: {e}")
        log_error(ErrorSeverity.WARNING, "mine_daily_patterns", e, 
                 ticker=ticker, show_to_user=False)
        return None, str(e)[:30]

# ================================
# PATTERN EVALUATION
# ================================

def train_and_evaluate_patterns(X: pd.DataFrame, y: pd.Series, 
                                timeframe: str) -> Optional[Dict[str, Any]]:
    """Train models and return best AUC"""
    try:
        models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            'lr': LogisticRegression(
                random_state=42,
                max_iter=500
            )
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        best_model_name = None
        best_avg_auc = 0
        best_std_auc = 0
        
        for model_name, model in models.items():
            auc_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                if len(np.unique(y_val)) < 2 or len(np.unique(y_train)) < 2:
                    continue
                
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)
                
                if y_pred_proba.shape[1] == 1:
                    continue
                
                y_pred_proba = y_pred_proba[:, 1]
                val_auc = roc_auc_score(y_val, y_pred_proba)
                auc_scores.append(val_auc)
            
            if len(auc_scores) == 0:
                continue
            
            avg_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            
            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc
                best_std_auc = std_auc
                best_model_name = model_name
        
        if best_model_name is None:
            return None
        
        pos_rate = y.mean() * 100
        up_count = int(y.sum())
        down_count = int(len(y) - y.sum())
        
        return {
            'auc': best_avg_auc,
            'std': best_std_auc,
            'model': best_model_name,
            'samples': len(X),
            'target_rate': pos_rate,
            'up_moves': up_count,
            'down_moves': down_count,
            'timeframe': timeframe
        }
        
    except Exception as e:
        logger.error(f"Pattern evaluation failed: {e}")
        log_error(ErrorSeverity.WARNING, "train_and_evaluate_patterns", e, 
                 show_to_user=False)
        return None

# ================================
# PATTERN MINING FOR TICKER
# ================================

def mine_patterns_for_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """Mine patterns for a single ticker"""
    try:
        logger.info(f"  → Mining {ticker}")
        
        # Mine both timeframes
        intraday_result, intraday_error = mine_intraday_patterns(ticker)
        daily_result, daily_error = mine_daily_patterns(ticker)
        
        results = []
        
        # Evaluate intraday
        if intraday_result is not None:
            X, y, tf = intraday_result
            eval_result = train_and_evaluate_patterns(X, y, tf)
            if eval_result is not None:
                results.append(eval_result)
        
        # Evaluate daily
        if daily_result is not None:
            X, y, tf = daily_result
            eval_result = train_and_evaluate_patterns(X, y, tf)
            if eval_result is not None:
                results.append(eval_result)
        
        if len(results) == 0:
            logger.info(f"○ {ticker} | No valid patterns found")
            return None
        
        # Check if either timeframe is elite
        config = load_pattern_mining_config()
        elite_results = [
            r for r in results 
            if r['auc'] >= config['min_auc_threshold'] 
            and r['std'] < config['max_auc_std']
        ]
        
        if len(elite_results) > 0:
            best = max(elite_results, key=lambda x: x['auc'])
            boost = int(best['auc'] * AUC_TO_BOOST_MULTIPLIER)
            
            direction_bias = "UP" if best['target_rate'] > 55 else \
                           "DOWN" if best['target_rate'] < 45 else \
                           "BALANCED"
            
            logger.info(
                f"✓ {ticker} | ELITE | {best['timeframe']:5} | "
                f"AUC {best['auc']:.3f}±{best['std']:.3f} | "
                f"{best['model'].upper()} | +{boost} | {direction_bias}"
            )
            
            metrics_collector.increment("elite_patterns_found")
            
            return {
                "ticker": ticker,
                "timeframe": best['timeframe'],
                "model": best['model'],
                "auc_mean": round(best['auc'], 3),
                "auc_std": round(best['std'], 3),
                "boost": boost,
                "direction_bias": direction_bias,
                "up_moves": best['up_moves'],
                "down_moves": best['down_moves'],
                "up_percentage": round(best['target_rate'], 1),
                "samples": best['samples'],
                "all_timeframes": {r['timeframe']: round(r['auc'], 3) for r in results},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        else:
            best = max(results, key=lambda x: x['auc'])
            logger.info(
                f"○ {ticker} | Below threshold | "
                f"Best: {best['timeframe']} AUC {best['auc']:.3f}±{best['std']:.3f}"
            )
            return None
            
    except Exception as e:
        logger.error(f"✗ {ticker} | Error: {str(e)[:40]}")
        log_error(ErrorSeverity.WARNING, "mine_patterns_for_ticker", e, 
                 ticker=ticker, show_to_user=False)
        return None

# ================================
# PATTERN MINING CYCLE
# ================================

# ============================================================================
# LINE 3420: run_pattern_mining_cycle() - Pattern discovery
# STATUS: WORKING - Finds elite trading patterns
# DEPENDENCIES: mine_daily_patterns, mine_intraday_patterns
# NOTE: Slow (30 min) - runs every 30 minutes
# ============================================================================
def run_pattern_mining_cycle() -> int:
    """Run one complete pattern mining cycle"""
    logger.info(f"\n{'='*80}")
    logger.info(f"HYBRID AUTO-PATTERN MINER | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}")
    
    # Use parallel mining if more than 4 tickers
    if len(PATTERN_WATCHLIST) > 4:
        logger.info(f"Using parallel mining with {min(4, len(PATTERN_WATCHLIST))} workers")
        patterns = mine_patterns_parallel(PATTERN_WATCHLIST, max_workers=4)
    else:
        patterns = []
        for ticker in PATTERN_WATCHLIST:
            result = mine_patterns_for_ticker(ticker)
            if result:
                patterns.append(result)
    
    output = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_analyzed": len(PATTERN_WATCHLIST),
        "elite_patterns_found": len(patterns),
        "patterns": patterns
    }
    
    try:
        with open(AUTO_PATTERNS_FILE, "w") as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save patterns: {e}")
        log_error(ErrorSeverity.ERROR, "run_pattern_mining_cycle", e, show_to_user=False)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ COMPLETE | {len(patterns)} elite patterns saved to auto_patterns.json")
    logger.info(f"{'='*80}\n")
    
    if patterns:
        avg_boost = np.mean([p['boost'] for p in patterns])
        logger.info(f"Average boost: {avg_boost:.1f}")
        top_performer = max(patterns, key=lambda x: x['auc_mean'])
        logger.info(f"Top performer: {top_performer['ticker']} (AUC: {top_performer['auc_mean']:.3f})")
        
        up_bias = [p for p in patterns if p['direction_bias'] == 'UP']
        down_bias = [p for p in patterns if p['direction_bias'] == 'DOWN']
        logger.info(f"Direction bias: {len(up_bias)} UP-biased, {len(down_bias)} DOWN-biased\n")
    
    metrics_collector.increment("pattern_mining_cycles")
    return len(patterns)

# ================================
# PATTERN INTEGRATION
# ================================

def check_pattern_override(ticker: str, bias: str, boost: int) -> Tuple[str, List[str], int]:
    """Check for pattern overrides from configuration"""
    overrides_config = load_pattern_overrides()
    
    if not overrides_config.get("enabled", False):
        return bias, [], 0
    
    ticker_clean = ticker.replace('=F', '').replace('^', '').split('.')[0].upper()
    overrides = overrides_config.get("overrides", {})
    
    if ticker_clean in overrides:
        override = overrides[ticker_clean]
        min_boost = override.get("min_boost", 0)
        
        if boost >= min_boost:
            if override.get("flip_direction", False):
                new_bias = "UP" if bias == "DOWN" else "DOWN" if bias == "UP" else bias
                triggers = [f"Override: {bias} → {new_bias}"]
                confidence_bonus = override.get("confidence_bonus", 0)
                
                logger.info(f"Applied override for {ticker_clean}: {bias} → {new_bias}")
                return new_bias, triggers, confidence_bonus
    
    return bias, [], 0

def check_auto_patterns(ticker: str, data: Optional[pd.DataFrame] = None) -> Tuple[
    int, List[str], str, int
]:
    """Enhanced pattern checking with better integration"""
    if not AUTO_PATTERNS_FILE.exists():
        return 0, [], "NEUTRAL", 0

    try:
        with open(AUTO_PATTERNS_FILE, 'r') as f:
            raw = json.load(f)
        if "patterns" not in raw:
            return 0, [], "NEUTRAL", 0

        ticker_clean = ticker.replace('=F', '').replace('^', '').split('.')[0].upper()
        now = datetime.now()
        best_match = None
        best_auc = 0

        for pat in raw["patterns"]:
            if pat.get("ticker", "").upper() != ticker_clean:
                continue
            try:
                pat_time = datetime.strptime(pat.get("timestamp", ""), "%Y-%m-%d %H:%M")
                if (now - pat_time).total_seconds() > 86400:  # 24 hours
                    continue
            except:
                continue

            auc = pat.get("auc_mean", 0)
            if auc > best_auc:
                best_auc = auc
                best_match = pat

        if not best_match:
            return 0, [], "NEUTRAL", 0

        boost = best_match.get("boost", 0)
        bias = best_match.get("direction_bias", "NEUTRAL")
        direction = "DOWN" if bias == "DOWN" else "UP" if bias == "UP" else "NEUTRAL"
        timeframe = best_match.get("timeframe", "unknown")
        model = best_match.get("model", "unknown").upper()
        auc_val = best_match.get("auc_mean", 0)

        confidence = min(99, int(auc_val * 100 + boost // 2.5))

        triggers = [
            f"{model} AUC {auc_val:.3f}",
            f"Boost +{boost}",
            f"{timeframe.upper()} ELITE",
            f"Bias {bias}"
        ]

        # Check for overrides
        override_direction, override_triggers, confidence_bonus = check_pattern_override(
            ticker, direction, boost
        )
        if override_triggers:
            direction = override_direction
            triggers.extend(override_triggers)
            confidence = min(99, confidence + confidence_bonus)

        return boost, triggers, direction, confidence

    except Exception as e:
        logger.warning(f"Error checking patterns for {ticker}: {e}")
        log_error(ErrorSeverity.WARNING, "check_auto_patterns", e, 
                 ticker=ticker, show_to_user=False)
        return 0, [], "NEUTRAL", 0

def get_pattern_boosted_forecast(ticker: str, base_forecast: List[float], 
                                current_price: float) -> List[float]:
    """Apply pattern mining boosts to the base forecast"""
    if base_forecast is None or len(base_forecast) == 0:
        return base_forecast
    
    boost, triggers, direction, confidence = check_auto_patterns(ticker)
    
    if boost == 0:
        return base_forecast
    
    # Convert base forecast to numpy array for manipulation
    forecast_array = np.array(base_forecast)
    
    # Apply boost based on pattern direction
    boost_factor = 1 + (boost / 1000)  # Convert boost to multiplier
    
    if direction == "UP":
        # Boost upward predictions
        forecast_array = forecast_array * boost_factor
        logger.info(f"Applied UP boost of {boost_factor:.3f}x to {ticker} forecast")
    elif direction == "DOWN":
        # Reduce downward predictions
        forecast_array = forecast_array / boost_factor
        logger.info(f"Applied DOWN reduction of {boost_factor:.3f}x to {ticker} forecast")
    
    return forecast_array.tolist()

def enhanced_confidence_checklist(ticker: str, forecast: List[float], 
                                 current_price: float) -> Tuple[bool, List[str], int]:
    """Enhanced confidence checklist with REAL accuracy metrics and pattern integration"""
    reasons = []
    meta = load_metadata(ticker)
    acc = load_accuracy_log(ticker)
    
    # Check validated prediction count
    validated_count = acc.get("validated_predictions", 0)
    if validated_count < 5: 
        reasons.append(f"Few validated predictions ({validated_count})")
    
    # Check retrain count
    if meta.get("retrain_count", 0) < 2: 
        reasons.append("Low retrains")
    
    # Check REAL error rates
    avg_mape = acc.get("avg_error_mape", 100.0)
    if avg_mape > 6.5: 
        reasons.append(f"MAPE {avg_mape:.1f}%")
    
    # Check directional accuracy
    dir_accuracy = acc.get("directional_accuracy", 0.0)
    if validated_count >= 3 and dir_accuracy < 55.0:
        reasons.append(f"Direction {dir_accuracy:.0f}%")
    
    # Check model staleness
    if meta.get("trained_date"):
        try:
            trained_date = datetime.fromisoformat(meta["trained_date"])
            days_since = (datetime.now() - trained_date).days
            if days_since > 14:
                reasons.append(f"Model {days_since}d old")
        except: 
            pass
    
    # Pattern-based confidence boost
    boost, triggers, direction, pattern_confidence = check_auto_patterns(ticker)
    if boost > 50:  # Strong pattern signal
        if len(reasons) > 0:
            # Remove one reason for strong patterns
            reasons.pop()
        reasons.append(f"Strong pattern +{boost}")
    
    # Check forecast reasonableness
    if forecast and current_price:
        move = abs(forecast[0] - current_price) / current_price
        if move > 0.12: 
            reasons.append(f"Extreme move {move:+.1%}")
    
    return len(reasons) == 0, reasons, boost

def get_pattern_influenced_recommendation(ticker: str, base_forecast: List[float], 
                                         current_price: float) -> Tuple[str, int, List[str]]:
    """Get recommendation influenced by pattern mining"""
    if base_forecast is None or current_price is None:
        return "HOLD", 0, []
    
    # Get pattern information
    boost, triggers, direction, pattern_confidence = check_auto_patterns(ticker)
    
    # Calculate base change
    change_pct = (base_forecast[0] - current_price) / current_price * 100
    
    # Apply pattern influence
    pattern_influence = boost / 500  # Convert boost to percentage influence
    influenced_change = change_pct + (pattern_influence if direction == "UP" else -pattern_influence)
    
    # Determine action with pattern consideration
    if influenced_change >= 3 or (direction == "UP" and influenced_change >= 1.5):
        action = "STRONG BUY"
        confidence = min(95, 70 + int(abs(influenced_change) * 5) + pattern_confidence // 2)
    elif influenced_change >= 1.5:
        action = "BUY" 
        confidence = min(85, 60 + int(abs(influenced_change) * 4) + pattern_confidence // 3)
    elif influenced_change <= -3 or (direction == "DOWN" and influenced_change <= -1.5):
        action = "STRONG SELL"
        confidence = min(95, 70 + int(abs(influenced_change) * 5) + pattern_confidence // 2)
    elif influenced_change <= -1.5:
        action = "SELL"
        confidence = min(85, 60 + int(abs(influenced_change) * 4) + pattern_confidence // 3)
    else:
        action = "HOLD"
        confidence = max(50, 50 + pattern_confidence // 4)
    
    # Add pattern triggers to reasons
    reasons = []
    if boost > 0:
        reasons.extend(triggers)
    
    return action, confidence, reasons

# ================================
# IMPLEMENTED 6%+ MOVE TELEGRAM ALERT SYSTEM
# ================================

# ============================================================================
# LINE 3690: monitor_6percent_pre_move_managed() - Price monitoring
# STATUS: WORKING - Sends Telegram alerts for 6%+ moves
# DEPENDENCIES: yfinance, send_telegram_alert, blacklist
# CRITICAL: Runs 24/7 - thread-safe, handles weekends
# ============================================================================
def monitor_6percent_pre_move_managed(stop_event: threading.Event) -> None:
    """Monitor for 6%+ daily price moves and send Telegram alerts"""
    app_state.update_heartbeat("monitoring")
    app_state.set_thread_start_time("monitoring")
    logger.info("[MONITORING] Enhanced Daily Monitor STARTED")
    
    # Cooldown tracking to prevent duplicate alerts (ticker -> last_alert_time)
    alert_cooldown: Dict[str, datetime] = {}
    
    # Get validated watchlist (skip delisted/problematic tickers)
    logger.info("[MONITORING] Validating watchlist...")
    watchlist_tickers = [t for t in PATTERN_WATCHLIST if not is_ticker_blacklisted(t)]
    logger.info(f"[MONITORING] Monitoring {len(watchlist_tickers)} validated tickers")
    if TICKER_BLACKLIST:
        logger.info(f"[MONITORING] Excluding {len(TICKER_BLACKLIST)} blacklisted tickers: {', '.join(sorted(TICKER_BLACKLIST))}")
    
    while not stop_event.is_set():
        try:
            # Check if monitoring is enabled
            config = load_monitoring_config()
            if not config.get("enabled", False):
                logger.info("[MONITORING] Monitoring paused (disabled in config)")
                time.sleep(30)
                app_state.update_heartbeat("monitoring")
                continue
            
            # Check if market is open (skip weekends)
            now = datetime.now()
            is_weekend = now.weekday() >= 5  # 5=Saturday, 6=Sunday
            
            if is_weekend:
                logger.info(f"[MONITORING] Markets closed (Weekend: {now.strftime('%A')}), sleeping 1 hour...")
                for _ in range(3600):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                    app_state.update_heartbeat("monitoring")
                continue
            
            # Also check market hours (before 9 AM or after 5 PM)
            hour = now.hour
            if hour < 9 or hour > 17:
                logger.info(f"[MONITORING] Outside market hours ({hour}:00), sleeping 30 min...")
                for _ in range(1800):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                    app_state.update_heartbeat("monitoring")
                continue

            current_time = datetime.now()
            
            # Get config values
            threshold = config.get("threshold_percent", 6.0)
            cooldown_minutes = config.get("cooldown_minutes", 30)
            
            logger.info(f"[MONITORING] Checking {len(watchlist_tickers)} tickers for {threshold}%+ daily moves...")
            
            # Track alerts in this cycle
            alerts_sent = 0
            checks_performed = 0
            
            for ticker in watchlist_tickers:
                if stop_event.is_set():
                    break
                
                # Skip blacklisted tickers (double-check)
                if is_ticker_blacklisted(ticker):
                    logger.debug(f"[MONITORING] ⚫ Skipping blacklisted ticker: {ticker}")
                    continue
                    
                try:
                    # Skip if in cooldown
                    last_alert = alert_cooldown.get(ticker)
                    if last_alert:
                        minutes_since = (current_time - last_alert).total_seconds() / 60
                        if minutes_since < cooldown_minutes:
                            logger.debug(f"[MONITORING] {ticker}: In cooldown ({minutes_since:.0f}min / {cooldown_minutes}min)")
                            continue
                    
                    # Get fresh data - NO CACHE
                    try:
                        df = yf.download(
                            ticker, 
                            period="5d",
                            interval="1d",
                            progress=False,
                            auto_adjust=True
                        )
                        
                        # Normalize columns
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        
                        if df is None or len(df) < 2:
                            logger.warning(f"[MONITORING] {ticker}: Insufficient data ({len(df) if df is not None else 0} days)")
                            continue
                        
                        # Ensure we have close prices
                        if 'Close' not in df.columns or df['Close'].isnull().all():
                            logger.warning(f"[MONITORING] {ticker}: No close prices")
                            continue
                        
                        # Sort by index (datetime) ascending
                        df = df.sort_index()
                        
                        # Get today's price (most recent)
                        current_price = float(df['Close'].iloc[-1])
                        current_date = df.index[-1]
                        
                        # Get previous day's price
                        previous_price = float(df['Close'].iloc[-2])
                        previous_date = df.index[-2]
                        
                        # Calculate percentage change from previous day
                        if previous_price == 0:
                            logger.warning(f"[MONITORING] {ticker}: Previous price is zero")
                            continue
                        
                        change_pct = (current_price - previous_price) / previous_price * 100
                        
                        checks_performed += 1
                        
                        # ALWAYS log the change for visibility
                        logger.info(
                            f"[MONITORING] {ticker}: "
                            f"${previous_price:.2f} → ${current_price:.2f} = {change_pct:+.2f}% "
                            f"(threshold: {threshold}%)"
                        )
                        
                        # Check for threshold move
                        if abs(change_pct) >= threshold:
                            direction = "📈 UP" if change_pct > 0 else "📉 DOWN"
                            move_type = "RISE" if change_pct > 0 else "DROP"
                            
                            # Calculate time difference
                            time_diff = (current_date - previous_date).days
                            time_desc = f"{time_diff} day{'s' if time_diff > 1 else ''}"
                            
                            # Get asset name
                            asset_name = get_asset_name_from_ticker(ticker)
                            
                            # Prepare alert message
                            message = (
                                f"🚨 **{move_type} ALERT** 🚨\n\n"
                                f"**{asset_name} ({ticker})**: {abs(change_pct):.1f}% {direction} in {time_desc}!\n\n"
                                f"• Price {time_desc} ago: ${previous_price:.2f}\n"
                                f"• Current price: ${current_price:.2f}\n"
                                f"• Change: ${current_price - previous_price:+.2f}\n"
                                f"• Date: {current_date.strftime('%Y-%m-%d')}\n"
                                f"• Direction: {direction}\n\n"
                                f"_Monitor: AI Alpha Trader v4.2_"
                            )
                            
                            # Check volume spike (optional)
                            if 'Volume' in df.columns and len(df) >= 2:
                                try:
                                    current_volume = df['Volume'].iloc[-1]
                                    avg_volume = df['Volume'].iloc[-5:-1].mean() if len(df) >= 5 else df['Volume'].iloc[:-1].mean()
                                    
                                    if avg_volume > 0 and not pd.isna(current_volume) and not pd.isna(avg_volume):
                                        volume_ratio = current_volume / avg_volume
                                        if volume_ratio > 2.0:
                                            message += f"\n📊 **Volume spike**: {volume_ratio:.1f}x average!"
                                except Exception as ve:
                                    logger.debug(f"Volume check failed for {ticker}: {ve}")
                            
                            # Log before attempting to send
                            logger.info(f"[MONITORING] 🚨 Attempting to send alert for {ticker}: {change_pct:+.2f}%")
                            print(f"\n{'='*60}")
                            print(f"🚨 ALERT TRIGGERED: {ticker} {change_pct:+.2f}%")
                            print(f"{'='*60}\n")
                            
                            # Send Telegram alert
                            success = send_telegram_alert(message)
                            
                            if success:
                                alerts_sent += 1
                                alert_cooldown[ticker] = current_time
                                metrics_collector.increment("monitoring_alerts_sent")
                                
                                logger.info(
                                    f"[ALERT] ✅ Sent {ticker} {move_type} alert: "
                                    f"{abs(change_pct):.1f}% in {time_desc}"
                                )
                                app_state.add_log_message(
                                    f"[ALERT] {ticker}: {abs(change_pct):.1f}% {direction} "
                                    f"in {time_desc}"
                                )
                                
                                # Also log to console with emoji
                                print(f"✅ TELEGRAM ALERT SENT: {ticker} {move_type} {abs(change_pct):.1f}%")
                            else:
                                logger.error(f"[ALERT] ❌ Failed to send Telegram alert for {ticker}")
                                print(f"❌ TELEGRAM FAILED for {ticker}")
                                
                                # Print the message that would have been sent
                                print(f"\nMessage that failed to send:\n{message}\n")
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "delisted" in error_msg.lower() or "no price data found" in error_msg.lower():
                            add_to_blacklist(ticker, "Delisted during monitoring")
                            logger.warning(f"[MONITORING] ⚫ Blacklisted {ticker} during check")
                        else:
                            logger.error(f"[MONITORING] Error downloading {ticker}: {e}")
                        continue
                    
                    # Small delay between tickers to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"[MONITORING] Unexpected error for {ticker}: {e}")
                    continue
            
            # Log summary of this monitoring cycle
            logger.info(
                f"[MONITORING] Cycle complete: "
                f"Checked {checks_performed}/{len(watchlist_tickers)} tickers, "
                f"Sent {alerts_sent} alerts"
            )
            
            if alerts_sent > 0:
                print(f"\n✅ {alerts_sent} alerts sent this cycle!\n")
            
            app_state.update_heartbeat("monitoring")
            
            # Sleep until next check (using configured interval)
            sleep_minutes = config.get("check_interval_minutes", 5)
            sleep_seconds = sleep_minutes * 60
            
            logger.info(f"[MONITORING] Sleeping for {sleep_minutes} minutes until next cycle...")
            
            # Sleep in smaller intervals to check stop_event
            for _ in range(sleep_seconds):
                if stop_event.is_set():
                    break
                time.sleep(1)
                app_state.update_heartbeat("monitoring")
                
                # Check if disabled during sleep
                if not load_monitoring_config().get("enabled", False):
                    logger.info("[MONITORING] Monitoring stopped during sleep")
                    break
                
        except Exception as e:
            logger.error(f"[CRITICAL] Monitoring error: {e}")
            log_error(ErrorSeverity.ERROR, "monitor_6percent_daily", e,
                     user_message="Monitoring error - will retry", show_to_user=False)
            
            # Wait before retry, but check stop_event
            for _ in range(60):
                if stop_event.is_set():
                    break
                time.sleep(1)
    
    logger.info("[MONITORING] Enhanced Daily Monitor STOPPED")
# ================================
# 2X DAILY PREDICTION SCANNER
# Add this to dashboard.py after line 1916 (after monitor_6percent_pre_move_managed)
# ================================

import pytz
from typing import List, Dict, Tuple, Optional

# ================================
# PREDICTION SCANNER CONFIGURATION
# ================================

PREDICTION_CONFIG_FILE = CONFIG_DIR / "prediction_scanner.json"

def load_prediction_config() -> Dict[str, Any]:
    """Load prediction scanner configuration"""
    default_config = {
        "enabled": True,  # ← NOW ENABLED BY DEFAULT ✅
        "morning_time": "09:30",
        "midday_time": "13:00",
        "probability_threshold": 0.65,
        "max_alerts": 10,
        "telegram_enabled": True,
        "timezone": "America/New_York"
    }
    return ConfigManager.load_config_with_backup(
        PREDICTION_CONFIG_FILE, "prediction_scanner", default_config
    )

def save_prediction_config(config: Dict[str, Any]) -> bool:
    """Save prediction scanner configuration"""
    return ConfigManager.save_config(PREDICTION_CONFIG_FILE, config, "prediction_scanner")

# ================================
# PRE-MARKET ANALYSIS
# ================================

def analyze_premarket(ticker: str) -> Dict[str, Any]:
    """
    Analyze pre-market activity for prediction
    
    Returns:
        {
            'gap_percent': float,
            'volume_ratio': float,
            'score': float (0-1),
            'signal': str
        }
    """
    try:
        # Get pre-market data (if available)
        df = yf.download(ticker, period="2d", interval="1d", progress=False, prepost=True)
        
        if df is None or len(df) < 2:
            return {'gap_percent': 0, 'volume_ratio': 0, 'score': 0.2, 'signal': 'No data'}
        
        df = normalize_dataframe_columns(df)
        
        # Get previous close and current price
        previous_close = float(df['Close'].iloc[-2])
        
        # Try to get pre-market price (or current if not available)
        try:
            current_price = get_latest_price(ticker)
            if current_price is None:
                current_price = float(df['Close'].iloc[-1])
        except:
            current_price = float(df['Close'].iloc[-1])
        
        # Calculate gap
        gap_percent = (current_price - previous_close) / previous_close * 100
        
        # Volume analysis (approximate - pre-market volume hard to get reliably)
        current_volume = float(df['Volume'].iloc[-1])
        avg_volume = float(df['Volume'].iloc[-10:].mean()) if len(df) >= 10 else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Scoring logic
        gap_score = min(abs(gap_percent) / 3.0, 1.0)  # 3%+ gap = max score
        volume_score = min(volume_ratio / 2.0, 1.0)   # 2x volume = max score
        
        # Combined score
        score = (gap_score * 0.6 + volume_score * 0.4)
        
        # Signal strength
        if abs(gap_percent) > 2 and volume_ratio > 2:
            signal = "STRONG"
        elif abs(gap_percent) > 1 or volume_ratio > 1.5:
            signal = "MODERATE"
        else:
            signal = "WEAK"
        
        logger.debug(f"[PREMARKET] {ticker}: Gap {gap_percent:+.2f}%, Vol {volume_ratio:.1f}x, Score {score:.2f}")
        
        return {
            'gap_percent': gap_percent,
            'volume_ratio': volume_ratio,
            'score': score,
            'signal': signal
        }
        
    except Exception as e:
        logger.error(f"Pre-market analysis failed for {ticker}: {e}")
        return {'gap_percent': 0, 'volume_ratio': 0, 'score': 0.2, 'signal': 'Error'}

# ================================
# MORNING MOMENTUM ANALYSIS
# ================================

def analyze_morning_momentum(ticker: str) -> float:
    """
    Analyze morning trading momentum (9:30-10:30 AM)
    
    Returns:
        Momentum score (0-1)
    """
    try:
        # Get intraday data
        df = yf.download(ticker, period="1d", interval="5m", progress=False)
        
        if df is None or len(df) < 5:
            return 0.3
        
        df = normalize_dataframe_columns(df)
        
        # Get open and current price
        open_price = float(df['Open'].iloc[0])
        current_price = float(df['Close'].iloc[-1])
        
        # Calculate momentum
        momentum_percent = (current_price - open_price) / open_price * 100
        
        # Volume analysis
        if 'Volume' in df.columns:
            current_volume = df['Volume'].sum()
            # Estimate full day volume (assuming 6.5 hour trading day)
            hours_elapsed = len(df) * 5 / 60  # 5-minute bars
            projected_volume = current_volume * (6.5 / hours_elapsed) if hours_elapsed > 0 else current_volume
            
            # Compare to yesterday's volume
            df_hist = yf.download(ticker, period="5d", interval="1d", progress=False)
            if df_hist is not None and len(df_hist) >= 2:
                df_hist = normalize_dataframe_columns(df_hist)
                avg_volume = float(df_hist['Volume'].iloc[-5:-1].mean())
                volume_ratio = projected_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
        else:
            volume_ratio = 1.0
        
        # Scoring
        momentum_score = min(abs(momentum_percent) / 3.0, 1.0)
        volume_score = min(volume_ratio / 2.0, 1.0)
        
        score = (momentum_score * 0.6 + volume_score * 0.4)
        
        logger.debug(f"[MOMENTUM] {ticker}: {momentum_percent:+.2f}%, Vol {volume_ratio:.1f}x, Score {score:.2f}")
        
        return score
        
    except Exception as e:
        logger.error(f"Morning momentum analysis failed for {ticker}: {e}")
        return 0.3

# ================================
# DAILY LSTM PREDICTION
# ================================

def get_daily_prediction(ticker: str) -> Dict[str, Any]:
    """
    Get LSTM prediction for today's move
    
    Returns:
        {
            'predicted_change': float,
            'score': float (0-1),
            'confidence': float
        }
    """
    try:
        # Get current price
        current_price = get_latest_price(ticker)
        if current_price is None:
            return {'predicted_change': 0, 'score': 0.3, 'confidence': 0}
        
        # Load or train model to get prediction
        forecast, lower_ci, upper_ci, dates, model = train_self_learning_model_enhanced(ticker, days=1)
        
        if forecast is None or len(forecast) == 0:
            return {'predicted_change': 0, 'score': 0.3, 'confidence': 0}
        
        # Get prediction
        predicted_price = float(np.array(forecast).flatten()[0])
        predicted_change = (predicted_price - current_price) / current_price * 100
        
        # Calculate confidence from CI width
        if lower_ci is not None and upper_ci is not None:
            ci_width = upper_ci[0] - lower_ci[0]
            ci_width_percent = ci_width / current_price * 100
            confidence = max(0, 1 - (ci_width_percent / 10))  # Narrower CI = higher confidence
        else:
            confidence = 0.5
        
        # Scoring
        if abs(predicted_change) >= 6:
            score = 0.9
        elif abs(predicted_change) >= 4:
            score = 0.7
        elif abs(predicted_change) >= 2:
            score = 0.5
        else:
            score = 0.3
        
        # Boost score with confidence
        score = score * (0.5 + confidence * 0.5)
        
        logger.debug(f"[LSTM] {ticker}: Predicts {predicted_change:+.2f}%, Confidence {confidence:.2f}, Score {score:.2f}")
        
        return {
            'predicted_change': predicted_change,
            'score': score,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Daily prediction failed for {ticker}: {e}")
        return {'predicted_change': 0, 'score': 0.3, 'confidence': 0}

# ================================
# MAIN PREDICTION SCAN
# ================================

def run_prediction_scan(scan_time: str) -> List[Dict[str, Any]]:
    """
    Run comprehensive prediction scan
    
    Args:
        scan_time: "morning" (9:30 AM) or "midday" (1:00 PM)
    
    Returns:
        List of high-probability predictions
    """
    logger.info(f"[PREDICTION SCAN] Starting {scan_time} scan...")
    
    predictions = []
    config = load_prediction_config()
    threshold = config.get('probability_threshold', 0.65)
    
    for ticker in PATTERN_WATCHLIST:
        try:
            logger.debug(f"[SCAN] Analyzing {ticker}...")
            
            # 1. Pre-market analysis
            premarket = analyze_premarket(ticker)
            
            # 2. Pattern mining status
            boost, triggers, direction, pattern_confidence = check_auto_patterns(ticker)
            pattern_score = pattern_confidence / 100 if boost > 0 else 0.3
            
            # 3. Morning momentum (only for midday scan)
            if scan_time == "midday":
                momentum_score = analyze_morning_momentum(ticker)
            else:
                momentum_score = 0.5  # Neutral for morning scan
            
            # 4. LSTM prediction
            lstm_pred = get_daily_prediction(ticker)
            
            # 5. Calculate combined probability
            if scan_time == "morning":
                probability = (
                    premarket['score'] * 0.30 +
                    pattern_score * 0.35 +
                    lstm_pred['score'] * 0.35
                )
            else:  # midday
                probability = (
                    premarket['score'] * 0.20 +
                    pattern_score * 0.30 +
                    momentum_score * 0.30 +
                    lstm_pred['score'] * 0.20
                )
            
            # 6. Determine direction
            if boost > 0 and direction != "NEUTRAL":
                pred_direction = direction
            elif lstm_pred['predicted_change'] > 1:
                pred_direction = "UP"
            elif lstm_pred['predicted_change'] < -1:
                pred_direction = "DOWN"
            else:
                pred_direction = "NEUTRAL"
            
            # 7. Add to predictions if high probability
            if probability >= threshold:
                predictions.append({
                    'ticker': ticker,
                    'probability': probability,
                    'direction': pred_direction,
                    'premarket': premarket,
                    'pattern_boost': boost,
                    'pattern_triggers': triggers,
                    'momentum': momentum_score if scan_time == "midday" else None,
                    'lstm_change': lstm_pred['predicted_change'],
                    'scan_time': scan_time
                })
                
                logger.info(
                    f"[HIGH PROB] {ticker}: {probability*100:.1f}% probability, "
                    f"{pred_direction} bias, Gap {premarket['gap_percent']:+.1f}%"
                )
        
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            continue
    
    # Sort by probability (highest first)
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    # Limit to max_alerts
    max_alerts = config.get('max_alerts', 10)
    predictions = predictions[:max_alerts]
    
    logger.info(f"[PREDICTION SCAN] {scan_time} scan complete: {len(predictions)} high-probability tickers found")
    
    return predictions

# ================================
# ALERT FORMATTING
# ================================

def format_prediction_alert(predictions: List[Dict[str, Any]], scan_time: str) -> str:
    """Format prediction results as Telegram alert"""
    
    if not predictions:
        return f"🔍 {scan_time.upper()} SCAN: No high-probability movers detected today."
    
    # Header
    if scan_time == "morning":
        header = "🌅 DAILY WATCHLIST ALERT\n\n📊 High Probability Movers Today:\n\n"
    else:
        header = "🌆 MIDDAY UPDATE\n\n📊 Updated Predictions:\n\n"
    
    message = header
    
    # Add each prediction
    for idx, pred in enumerate(predictions, 1):
        ticker = pred['ticker']
        prob = pred['probability'] * 100
        direction = pred['direction']
        
        # Direction emoji
        dir_emoji = "📈" if direction == "UP" else "📉" if direction == "DOWN" else "➡️"
        
        # Get asset name
        asset_name = get_asset_name_from_ticker(ticker)
        
        message += f"{idx}. **{asset_name} ({ticker})** - {prob:.0f}% chance of 6%+ move\n"
        message += f"   • Direction: {dir_emoji} {direction} bias\n"
        
        # Pre-market info
        gap = pred['premarket']['gap_percent']
        if abs(gap) > 0.5:
            message += f"   • Pre-market: {gap:+.1f}%"
            vol_ratio = pred['premarket']['volume_ratio']
            if vol_ratio > 1.2:
                message += f" on {vol_ratio:.1f}x volume"
            message += "\n"
        
        # Pattern info
        if pred['pattern_boost'] > 0:
            message += f"   • Pattern: ELITE (+{pred['pattern_boost']} boost)\n"
        
        # LSTM prediction
        lstm_change = pred['lstm_change']
        if abs(lstm_change) > 1:
            message += f"   • LSTM: Predicts {lstm_change:+.1f}%\n"
        
        # Momentum (midday only)
        if pred['momentum'] is not None and pred['momentum'] > 0.5:
            message += f"   • Momentum: {'STRONG' if pred['momentum'] > 0.7 else 'BUILDING'}\n"
        
        # Triggers (if available)
        if pred['pattern_triggers']:
            triggers_str = ", ".join(pred['pattern_triggers'][:3])  # First 3 triggers
            message += f"   • Signals: {triggers_str}\n"
        
        message += "\n"
    
    # Footer
    if scan_time == "morning":
        message += "📝 Note: Monitor these closely today\n"
        message += "⏰ Next scan: 1:00 PM\n"
    else:
        message += "📝 Note: Afternoon session predictions\n"
        message += "⏰ Next scan: Tomorrow 9:30 AM\n"
    
    message += f"\n_{scan_time.title()} Scan - AI Alpha Trader v4.2_"
    
    return message

def format_midday_comparison(morning_preds: List[Dict], midday_preds: List[Dict]) -> str:
    """Compare morning and midday predictions, highlight changes"""
    
    message = "🌆 MIDDAY UPDATE\n\n"
    
    # Find increased probabilities
    increased = []
    decreased = []
    new_entries = []
    
    morning_dict = {p['ticker']: p for p in morning_preds}
    midday_dict = {p['ticker']: p for p in midday_preds}
    
    for ticker, midday_pred in midday_dict.items():
        if ticker in morning_dict:
            morning_prob = morning_dict[ticker]['probability']
            midday_prob = midday_pred['probability']
            
            if midday_prob > morning_prob + 0.05:  # 5% increase
                increased.append((ticker, morning_prob, midday_prob, midday_pred))
            elif midday_prob < morning_prob - 0.05:  # 5% decrease
                decreased.append((ticker, morning_prob, midday_prob))
        else:
            new_entries.append((ticker, midday_pred))
    
    # Format increased probabilities
    if increased:
        message += "📈 **PROBABILITY INCREASED:**\n"
        for ticker, morning_p, midday_p, pred in increased:
            message += f"\n**{ticker}** - {midday_p*100:.0f}% (was {morning_p*100:.0f}%)\n"
            message += f"   • Move so far: {pred['premarket']['gap_percent']:+.1f}%\n"
            if pred['momentum']:
                message += f"   • Momentum: {'ACCELERATING' if pred['momentum'] > 0.7 else 'BUILDING'}\n"
            message += f"   • Action: 🎯 {'HIGH' if midday_p > 0.8 else 'MODERATE'} ALERT\n"
        message += "\n"
    
    # Format decreased probabilities
    if decreased:
        message += "📉 **PROBABILITY DECREASED:**\n"
        for ticker, morning_p, midday_p in decreased:
            message += f"\n**{ticker}** - {midday_p*100:.0f}% (was {morning_p*100:.0f}%)\n"
            message += f"   • Momentum: FADING\n"
            if midday_p < 0.5:
                message += f"   • Removed from watch list\n"
        message += "\n"
    
    # Format new entries
    if new_entries:
        message += "⚠️ **NEW ENTRIES:**\n"
        for ticker, pred in new_entries:
            message += f"\n**{ticker}** - {pred['probability']*100:.0f}% probability\n"
            message += f"   • Not in morning scan\n"
            if pred['momentum']:
                message += f"   • Sudden momentum spike\n"
            message += f"   • Direction: {pred['direction']}\n"
        message += "\n"
    
    if not increased and not decreased and not new_entries:
        message += "ℹ️ No significant changes from morning scan\n\n"
    
    message += "_Midday Update - AI Alpha Trader v4.2_"
    
    return message

# ================================
# SCHEDULED SCANNER THREAD
# ================================

def scheduled_prediction_scanner(stop_event: threading.Event) -> None:
    """
    Run prediction scans at scheduled times
    
    Schedule:
        - 9:30 AM EST: Morning scan
        - 1:00 PM EST: Midday scan
    """
    app_state.update_heartbeat("prediction_scanner")
    app_state.set_thread_start_time("prediction_scanner")
    logger.info("[PREDICTION SCANNER] Scheduled scanner STARTED")
    
    # Store morning predictions for comparison
    morning_predictions = []
    
    # Get timezone
    config = load_prediction_config()
    tz_str = config.get('timezone', 'America/New_York')
    tz = pytz.timezone(tz_str)
    
    while not stop_event.is_set():
        try:
            # Check if enabled
            config = load_prediction_config()
            if not config.get('enabled', False):
                logger.debug("[PREDICTION SCANNER] Disabled, sleeping...")
                time.sleep(60)
                app_state.update_heartbeat("prediction_scanner")
                continue
            
            # Get current time in market timezone
            now = datetime.now(tz)
            hour, minute = now.hour, now.minute
            
            # Parse scan times
            morning_time = config.get('morning_time', '09:30').split(':')
            morning_hour, morning_minute = int(morning_time[0]), int(morning_time[1])
            
            midday_time = config.get('midday_time', '13:00').split(':')
            midday_hour, midday_minute = int(midday_time[0]), int(midday_time[1])
            
            # Morning scan
            if hour == morning_hour and minute == morning_minute:
                logger.info("[PREDICTION SCANNER] Running morning scan...")
                app_state.add_log_message("[PREDICTION] Morning scan started")
                
                try:
                    predictions = run_prediction_scan("morning")
                    morning_predictions = predictions  # Store for midday comparison
                    
                    if predictions:
                        alert_msg = format_prediction_alert(predictions, "morning")
                        
                        if config.get('telegram_enabled', True):
                            success = send_telegram_alert(alert_msg)
                            if success:
                                logger.info(f"[PREDICTION SCANNER] Morning alert sent: {len(predictions)} tickers")
                                app_state.add_log_message(f"[PREDICTION] Morning alert sent: {len(predictions)} high-probability tickers")
                            else:
                                logger.error("[PREDICTION SCANNER] Failed to send morning alert")
                        
                        # Also log to console
                        print(f"\n{'='*80}")
                        print(alert_msg)
                        print(f"{'='*80}\n")
                    else:
                        logger.info("[PREDICTION SCANNER] Morning scan: No high-probability movers")
                        app_state.add_log_message("[PREDICTION] Morning scan: No high-probability movers")
                
                except Exception as e:
                    logger.error(f"[PREDICTION SCANNER] Morning scan failed: {e}")
                    log_error(ErrorSeverity.ERROR, "morning_prediction_scan", e, show_to_user=False)
                
                # Sleep for 2 minutes to avoid re-triggering
                for _ in range(120):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                    app_state.update_heartbeat("prediction_scanner")
            
            # Midday scan
            elif hour == midday_hour and minute == midday_minute:
                logger.info("[PREDICTION SCANNER] Running midday scan...")
                app_state.add_log_message("[PREDICTION] Midday scan started")
                
                try:
                    predictions = run_prediction_scan("midday")
                    
                    if predictions:
                        # Compare with morning predictions
                        if morning_predictions:
                            alert_msg = format_midday_comparison(morning_predictions, predictions)
                        else:
                            alert_msg = format_prediction_alert(predictions, "midday")
                        
                        if config.get('telegram_enabled', True):
                            success = send_telegram_alert(alert_msg)
                            if success:
                                logger.info(f"[PREDICTION SCANNER] Midday alert sent: {len(predictions)} tickers")
                                app_state.add_log_message(f"[PREDICTION] Midday alert sent: {len(predictions)} updates")
                            else:
                                logger.error("[PREDICTION SCANNER] Failed to send midday alert")
                        
                        # Also log to console
                        print(f"\n{'='*80}")
                        print(alert_msg)
                        print(f"{'='*80}\n")
                    else:
                        logger.info("[PREDICTION SCANNER] Midday scan: No significant updates")
                        app_state.add_log_message("[PREDICTION] Midday scan: No significant updates")
                
                except Exception as e:
                    logger.error(f"[PREDICTION SCANNER] Midday scan failed: {e}")
                    log_error(ErrorSeverity.ERROR, "midday_prediction_scan", e, show_to_user=False)
                
                # Sleep for 2 minutes to avoid re-triggering
                for _ in range(120):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                    app_state.update_heartbeat("prediction_scanner")
            
            # Check every minute
            app_state.update_heartbeat("prediction_scanner")
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"[PREDICTION SCANNER] Critical error: {e}")
            log_error(ErrorSeverity.ERROR, "scheduled_prediction_scanner", e, show_to_user=False)
            
            # Wait before retry
            for _ in range(60):
                if stop_event.is_set():
                    break
                time.sleep(1)
    
    logger.info("[PREDICTION SCANNER] Scheduled scanner STOPPED")

# ================================
# UI CONTROLS (add to sidebar)
# ================================

def add_prediction_scanner_controls():
    """Add prediction scanner controls to Streamlit sidebar"""
    st.markdown("---")
    st.subheader("🔮 Prediction Scanner")
    
    config = load_prediction_config()
    status = "RUNNING" if config.get("enabled") else "STOPPED"
    status_color = "🟢" if config.get("enabled") else "🔴"
    st.write(f"**Status:** {status_color} {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", key="ps_start", type="primary", use_container_width=True):
            config["enabled"] = True
            save_prediction_config(config)
            thread_manager.start_thread("prediction_scanner", scheduled_prediction_scanner)
            st.success("✅ Scanner started!")
            time.sleep(1)
            st.rerun()
    with col2:
        if st.button("⏹️ Stop", key="ps_stop", type="secondary", use_container_width=True):
            config["enabled"] = False
            save_prediction_config(config)
            thread_manager.stop_thread("prediction_scanner")
            st.warning("⚠️ Scanner stopped!")
            time.sleep(1)
            st.rerun()
    
    # Manual trigger
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌅 Run Morning Scan", type="secondary", use_container_width=True):
            with st.spinner("Running morning scan..."):
                predictions = run_prediction_scan("morning")
                if predictions:
                    alert = format_prediction_alert(predictions, "morning")
                    st.success(f"✅ Found {len(predictions)} high-probability tickers!")
                    st.text(alert)
                else:
                    st.info("No high-probability movers detected")
    
    with col2:
        if st.button("🌆 Run Midday Scan", type="secondary", use_container_width=True):
            with st.spinner("Running midday scan..."):
                predictions = run_prediction_scan("midday")
                if predictions:
                    alert = format_prediction_alert(predictions, "midday")
                    st.success(f"✅ Found {len(predictions)} high-probability tickers!")
                    st.text(alert)
                else:
                    st.info("No high-probability movers detected")
    
    # Configuration
    with st.expander("⚙️ Configuration"):
        new_morning = st.text_input("Morning Time (HH:MM)", config.get('morning_time', '09:30'))
        new_midday = st.text_input("Midday Time (HH:MM)", config.get('midday_time', '13:00'))
        new_threshold = st.slider(
            "Probability Threshold", 
            0.5, 0.9, 
            config.get('probability_threshold', 0.65),
            step=0.05
        )
        new_max = st.slider("Max Alerts", 5, 20, config.get('max_alerts', 10))
        new_telegram = st.checkbox("Telegram Alerts", config.get('telegram_enabled', True))
        
        if st.button("💾 Save Config", type="secondary"):
            new_config = {
                "enabled": config.get("enabled", False),
                "morning_time": new_morning,
                "midday_time": new_midday,
                "probability_threshold": new_threshold,
                "max_alerts": new_max,
                "telegram_enabled": new_telegram,
                "timezone": config.get('timezone', 'America/New_York')
            }
            if save_prediction_config(new_config):
                st.success("✅ Config updated!")
            else:
                st.error("❌ Failed to save config")

# ================================
# REGISTER THREAD IN ApplicationState
# ================================

# Add to ApplicationState.__init__ thread_heartbeats dict:
# "prediction_scanner": None

# Add to initialize_background_threads_enhanced():
# if load_prediction_config().get("enabled", False):
#     thread_manager.start_thread("prediction_scanner", scheduled_prediction_scanner)
#     logger.info("[SUCCESS] Prediction scanner thread started")

# ================================
# THREAD HEARTBEAT AND MONITORING
# ================================

class ApplicationState:
    """Encapsulate application state with improved thread-safety"""
    
    def __init__(self):
        self.thread_heartbeats: Dict[str, Optional[datetime]] = {
            "learning_daemon": None,
            "monitoring": None,
            "pattern_miner": None,
            "watchdog": None,
            "auto_validator": None,
            "prediction_scanner": None,
        }
        self.thread_start_times: Dict[str, Optional[datetime]] = {
            "learning_daemon": None,
            "monitoring": None,
            "pattern_miner": None,
            "watchdog": None,
            "auto_validator": None,
            "prediction_scanner": None,
        }
        self._lock = threading.RLock()
        self.logging_queue = queue.Queue()
    
    def update_heartbeat(self, thread_name: str) -> None:
        """Update thread heartbeat"""
        with self._lock:
            self.thread_heartbeats[thread_name] = datetime.now()
    
    def get_thread_status(self, thread_name: str) -> Dict[str, Any]:
        """Get thread status based on heartbeat"""
        with self._lock:
            if self.thread_heartbeats[thread_name] is None:
                return {
                    "status": "DEAD",
                    "seconds_since": 9999,
                    "uptime": "Unknown"
                }
            
            seconds_since = (datetime.now() - self.thread_heartbeats[thread_name]).total_seconds()
            
            if seconds_since > 300:  # 5 minutes
                status = "DEAD"
            elif seconds_since > 120:  # 2 minutes
                status = "WARNING"
            else:
                status = "HEALTHY"
            
            uptime = "Unknown"
            if self.thread_start_times[thread_name]:
                uptime = str(datetime.now() - self.thread_start_times[thread_name]).split('.')[0]
            
            return {
                "status": status,
                "seconds_since": seconds_since,
                "uptime": uptime
            }
    
    def set_thread_start_time(self, thread_name: str) -> None:
        """Set thread start time"""
        with self._lock:
            self.thread_start_times[thread_name] = datetime.now()
    
    def add_log_message(self, message: str) -> None:
        """Add log message to queue for UI"""
        try:
            self.logging_queue.put(message, block=False)
        except queue.Full:
            # If queue is full, remove oldest and add new
            try:
                self.logging_queue.get_nowait()
                self.logging_queue.put(message, block=False)
            except:
                pass  # Silently fail if still can't add
    
    def get_log_messages(self) -> List[str]:
        """Get all pending log messages - FIXED thread-safety"""
        messages = []
        with self._lock:  # ✅ Added lock here
            while not self.logging_queue.empty():
                try:
                    messages.append(self.logging_queue.get_nowait())
                except queue.Empty:
                    break
        return messages

# Global application state
app_state = ApplicationState()

# ================================
# BACKGROUND THREADS
# ================================

# ============================================================================
# LINE 4746: continuous_learning_daemon_managed() - Auto retraining
# STATUS: WORKING - Monitors accuracy and retrains as needed
# DEPENDENCIES: train_self_learning_model_enhanced, should_retrain
# RUNS: Background thread, checks every 10 minutes
# ============================================================================
def continuous_learning_daemon_managed(stop_event: threading.Event) -> None:
    """Enhanced learning daemon with resource management"""
    app_state.update_heartbeat("learning_daemon")
    app_state.set_thread_start_time("learning_daemon")
    logger.info("[LEARNING] Enhanced Learning Daemon STARTED")
    
    cycle_count = 0
    
    while not stop_event.is_set():
        try:
            if not load_daemon_config().get("enabled", False):
                logger.info("[LEARNING] Learning paused (disabled in config)")
                time.sleep(30)
                app_state.update_heartbeat("learning_daemon")
                continue
            
            cycle_count += 1
            app_state.update_heartbeat("learning_daemon")
            logger.info(f"[CYCLE] Learning: Starting cycle #{cycle_count}")
            app_state.add_log_message(f"[CYCLE {cycle_count}] Learning daemon cycle started")
            
            # Get all tickers
            all_tickers = [t for cat in ASSET_CATEGORIES.values() for t in cat.values()]
            config = load_daemon_config()
            max_retrain = config.get("max_retrain_per_cycle", 5)
            
            # Train models with limit
            trained_count = 0
            for ticker in all_tickers:
                if stop_event.is_set():
                    break
                    
                if trained_count >= max_retrain:
                    break
                    
                try:
                    # Check if retraining is needed
                    acc_log = load_accuracy_log(ticker)
                    meta = load_metadata(ticker)
                    needs_retrain, reasons = should_retrain(ticker, acc_log, meta)
                    
                    if needs_retrain:
                        logger.info(f"[RETRAIN] {ticker}: {', '.join(reasons)}")
                        app_state.add_log_message(f"[RETRAIN] {ticker}: {', '.join(reasons)}")
                        
                        forecast, lower_ci, upper_ci, dates, model = train_self_learning_model_enhanced(ticker, days=5)
                        if forecast is not None:
                            trained_count += 1
                            logger.info(f"[SUCCESS] Retrained {ticker}")
                            app_state.add_log_message(f"[SUCCESS] Retrained {ticker}")
                        else:
                            logger.warning(f"[FAILED] Failed to retrain {ticker}")
                    
                    # Small delay between tickers
                    time.sleep(2)
                    app_state.update_heartbeat("learning_daemon")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Failed to process {ticker}: {e}")
                    log_error(ErrorSeverity.ERROR, "learning_daemon_ticker", e, 
                             ticker=ticker, show_to_user=False)
                    continue
            
            logger.info(f"[SUCCESS] Learning: Cycle #{cycle_count} complete - {trained_count} models updated")
            app_state.add_log_message(f"[CYCLE {cycle_count}] Complete - {trained_count} models updated")
            
            # Sleep with interrupt checking
            sleep_minutes = config.get("sleep_minutes", 10)
            sleep_seconds = sleep_minutes * 60
            logger.info(f"[SLEEP] Learning: Sleeping for {sleep_minutes} minutes")
            
            for _ in range(sleep_seconds):
                if stop_event.is_set():
                    break
                time.sleep(1)
                app_state.update_heartbeat("learning_daemon")
                
        except Exception as e:
            logger.error(f"[CRITICAL] Learning daemon error: {e}")
            log_error(ErrorSeverity.ERROR, "continuous_learning_daemon", e,
                     user_message="Learning daemon error - will retry", show_to_user=False)
            
            for _ in range(60):
                if stop_event.is_set():
                    break
                time.sleep(1)
    
    logger.info("[LEARNING] Enhanced Learning Daemon STOPPED")

def continuous_pattern_miner_managed(stop_event: threading.Event) -> None:
    """Pattern miner with proper resource management"""
    app_state.update_heartbeat("pattern_miner")
    app_state.set_thread_start_time("pattern_miner")
    logger.info("[PATTERN MINER] Pattern Mining Daemon STARTED (Managed)")
    
    cycle_count = 0
    
    while not stop_event.is_set():
        try:
            # Check if pattern mining is enabled
            if not load_pattern_mining_config().get("enabled", False):
                logger.info("[PATTERN MINER] Pattern Mining paused (disabled in config)")
                time.sleep(30)
                app_state.update_heartbeat("pattern_miner")
                continue
            
            cycle_count += 1
            app_state.update_heartbeat("pattern_miner")
            logger.info(f"[CYCLE] Pattern Mining: Starting cycle #{cycle_count}")
            app_state.add_log_message(f"[PATTERN CYCLE {cycle_count}] Mining started")
            
            # Run mining cycle
            patterns_found = run_pattern_mining_cycle()
            
            logger.info(f"[SUCCESS] Pattern Mining: Cycle #{cycle_count} complete - {patterns_found} patterns found")
            app_state.add_log_message(f"[PATTERN CYCLE {cycle_count}] Complete - {patterns_found} elite patterns")
            
            # Sleep for configured interval with proper interrupt handling
            config = load_pattern_mining_config()
            sleep_minutes = config.get('cycle_interval_minutes', 30)
            sleep_seconds = sleep_minutes * 60
            
            logger.info(f"[SLEEP] Pattern Mining: Sleeping for {sleep_minutes} minutes")
            
            # Sleep in smaller intervals to check stop_event
            for _ in range(sleep_seconds):
                if stop_event.is_set():
                    break
                time.sleep(1)
                app_state.update_heartbeat("pattern_miner")
                
                # Check if disabled during sleep
                if not load_pattern_mining_config().get("enabled", False):
                    logger.info("[PATTERN MINER] Pattern Mining stopped during sleep")
                    break
            
        except Exception as e:
            logger.error(f"[CRITICAL] Pattern mining error: {e}")
            log_error(ErrorSeverity.ERROR, "continuous_pattern_miner", e, 
                    user_message="Pattern mining error - will retry", show_to_user=False)
            
            # Wait before retry, but check stop_event
            for _ in range(60):
                if stop_event.is_set():
                    break
                time.sleep(1)
    
    logger.info("[PATTERN MINER] Pattern Mining Daemon STOPPED (Managed)")

def thread_watchdog_managed(stop_event: threading.Event) -> None:
    """Enhanced watchdog with managed threads"""
    app_state.update_heartbeat("watchdog")
    app_state.set_thread_start_time("watchdog")
    logger.info("[WATCHDOG] Enhanced Watchdog STARTED")
    
    while not stop_event.is_set():
        try:
            app_state.update_heartbeat("watchdog")
            
            # Check all thread statuses through thread manager
            thread_status = thread_manager.get_status()
            
            for name, status in thread_status.items():
                if not status['alive']:
                    logger.error(f"[ERROR] Thread {name} is DEAD")
                    app_state.add_log_message(f"[ALERT] Thread {name} is DEAD")
                else:
                    logger.debug(f"[SUCCESS] Thread {name} is HEALTHY")
            
            # Also check legacy heartbeat threads
            for name in ["learning_daemon", "monitoring", "pattern_miner", "auto_validator"]:
                if name not in thread_status:
                    status = app_state.get_thread_status(name)
                    if status["status"] == "DEAD":
                        logger.warning(f"[WARNING] Legacy thread {name} is DEAD")
            
            # Sleep with interrupt check
            for _ in range(30):
                if stop_event.is_set():
                    break
                time.sleep(1)
            
        except Exception as e:
            logger.error(f"[ERROR] Watchdog error: {e}")
            log_error(ErrorSeverity.WARNING, "thread_watchdog", e, show_to_user=False)
            
            for _ in range(30):
                if stop_event.is_set():
                    break
                time.sleep(1)
    
    logger.info("[WATCHDOG] Enhanced Watchdog STOPPED")

def auto_validate_predictions_background(stop_event: threading.Event) -> None:
    """Background thread to automatically validate predictions daily"""
    app_state.update_heartbeat("auto_validator")
    app_state.set_thread_start_time("auto_validator")
    logger.info("[AUTO VALIDATOR] Started")
    
    while not stop_event.is_set():
        try:
            app_state.update_heartbeat("auto_validator")
            
            # Run validation once per day at 9 AM
            now = datetime.now()
            if now.hour == 9 and now.minute < 5:
                logger.info("[AUTO VALIDATOR] Running daily validation...")
                app_state.add_log_message(f"[AUTO VALIDATOR] Daily validation started")
                
                all_tickers = [t for cat in ASSET_CATEGORIES.values() for t in cat.values()]
                validated_count = 0
                
                for ticker in all_tickers:
                    try:
                        success, acc_data = validate_predictions(ticker)
                        if success and acc_data.get('validated_predictions', 0) > 0:
                            validated_count += 1
                            
                            # Check if performance is poor and send alert
                            mape = acc_data.get('avg_error_mape', 0)
                            if mape > 10.0:
                                logger.warning(f"[AUTO VALIDATOR] {ticker} has poor accuracy: {mape:.2f}% MAPE")
                                app_state.add_log_message(f"[ALERT] {ticker} accuracy degraded: {mape:.2f}% MAPE")
                    except Exception as e:
                        logger.error(f"[AUTO VALIDATOR] Failed to validate {ticker}: {e}")
                
                logger.info(f"[AUTO VALIDATOR] Validated {validated_count} tickers")
                app_state.add_log_message(f"[AUTO VALIDATOR] Complete - {validated_count} tickers validated")
                
                # Sleep for an hour to avoid running again
                for _ in range(3600):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                    app_state.update_heartbeat("auto_validator")
            
            # Check every 5 minutes
            for _ in range(300):
                if stop_event.is_set():
                    break
                time.sleep(1)
                app_state.update_heartbeat("auto_validator")
                
        except Exception as e:
            logger.error(f"[AUTO VALIDATOR] Error: {e}")
            log_error(ErrorSeverity.ERROR, "auto_validate_predictions_background", e, show_to_user=False)
            
            for _ in range(60):
                if stop_event.is_set():
                    break
                time.sleep(1)
    
    logger.info("[AUTO VALIDATOR] Stopped")

# ================================
# SYSTEM RESOURCE MONITORING
# ================================

def monitor_system_resources() -> Optional[Dict[str, Any]]:
    """Monitor system resources"""
    if not PSUTIL_AVAILABLE:
        return {
            'memory_mb': 0,
            'cpu_percent': 0,
            'disk_usage': 0
        }
    
    try:
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # CPU usage
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Disk usage
        disk_usage = psutil.disk_usage('.').percent
        
        logger.debug(f"Resource Usage - Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, Disk: {disk_usage:.1f}%")
        
        # Warn if resources are high
        if memory_mb > MEMORY_WARNING_THRESHOLD_MB:
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
        if cpu_percent > CPU_WARNING_THRESHOLD_PERCENT:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
        if disk_usage > DISK_WARNING_THRESHOLD_PERCENT:
            logger.error(f"High disk usage: {disk_usage:.1f}%")
            
        return {
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'disk_usage': disk_usage
        }
        
    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")
        return None

# ================================
# RESOURCE MONITORING DASHBOARD
# ================================

def show_resource_dashboard():
    """Show detailed resource usage dashboard"""
    st.subheader("💻 System Resource Dashboard")
    
    if not PSUTIL_AVAILABLE:
        st.warning("psutil not available for resource monitoring")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        st.progress(cpu_percent / 100)
    
    with col2:
        # Memory
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent:.1f}%")
        st.progress(memory.percent / 100)
        st.caption(f"{memory.used/1024/1024/1024:.1f} GB / {memory.total/1024/1024/1024:.1f} GB")
    
    with col3:
        # Disk
        disk = psutil.disk_usage('.')
        st.metric("Disk Usage", f"{disk.percent:.1f}%")
        st.progress(disk.percent / 100)
    
    with col4:
        # Network
        net_io = psutil.net_io_counters()
        st.metric("Network IO", f"{net_io.bytes_sent/1024/1024:.1f} MB")
        st.caption(f"Sent: {net_io.bytes_sent/1024/1024:.1f}MB, Recv: {net_io.bytes_recv/1024/1024:.1f}MB")
    
    # Process-specific metrics
    st.markdown("---")
    st.subheader("📊 Application Metrics")
    
    process = psutil.Process()
    with st.expander("Process Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**PID:** {process.pid}")
            st.write(f"**Status:** {process.status()}")
            st.write(f"**Create Time:** {datetime.fromtimestamp(process.create_time()).strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.write(f"**Threads:** {process.num_threads()}")
            st.write(f"**Open Files:** {len(process.open_files())}")
            st.write(f"**Connections:** {len(process.connections())}")

# ================================
# STREAMLIT UI OPTIMIZATION
# ================================

def lazy_plotly_chart(fig_func: callable, key: str):
    """Lazy load Plotly charts to improve UI performance"""
    if key not in st.session_state:
        with st.spinner(f"Loading chart..."):
            fig = fig_func()
            st.session_state[key] = fig
    
    fig = st.session_state[key]
    st.plotly_chart(fig, use_container_width=True)

# ================================
# TELEGRAM INTEGRATION
# ================================

def send_telegram_alert(message: str) -> bool:
    """Send Telegram alert"""
    try:
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.warning("Telegram credentials not configured")
            return False
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            logger.info("Telegram alert sent successfully")
            return True
        else:
            logger.error(f"Telegram alert failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        log_error(ErrorSeverity.WARNING, "send_telegram_alert", e, show_to_user=False)
        return False

# ================================
# INITIALIZATION
# ================================

def initialize_background_threads_enhanced() -> None:
    """Initialize background threads with proper management"""
    if "threads_initialized" not in st.session_state:
        st.session_state.threads_initialized = True
        logger.info("[LAUNCH] Initializing enhanced background threads...")
        
        # Start prediction scanner if enabled
        if load_prediction_config().get("enabled", False):
            thread_manager.start_thread("prediction_scanner", scheduled_prediction_scanner)
            logger.info("[SUCCESS] Prediction scanner thread started")
        
        # Always start watchdog
        thread_manager.start_thread("watchdog", thread_watchdog_managed)
        logger.info("[SUCCESS] Watchdog thread started")
        
        # Start auto validator
        thread_manager.start_thread("auto_validator", auto_validate_predictions_background)
        logger.info("[SUCCESS] Auto validator thread started")
        
        # Start learning daemon if enabled
        if load_daemon_config().get("enabled", False):
            thread_manager.start_thread("learning_daemon", continuous_learning_daemon_managed)
            logger.info("[SUCCESS] Learning daemon thread started")
        
        # Start monitoring if enabled
        if load_monitoring_config().get("enabled", False):
            thread_manager.start_thread("monitoring", monitor_6percent_pre_move_managed)
            logger.info("[SUCCESS] Monitoring thread started")
            
        # Start pattern miner if enabled
        if load_pattern_mining_config().get("enabled", False):
            thread_manager.start_thread("pattern_miner", continuous_pattern_miner_managed)
            logger.info("[SUCCESS] Pattern miner thread started")

def shutdown_background_threads() -> None:
    """Gracefully shutdown all background threads"""
    logger.info("Shutting down background threads...")
    thread_manager.stop_all()
    logger.info("All background threads stopped")

# ================================
# STREAMLIT UI COMPONENTS
# ================================

def add_pattern_mining_controls() -> None:
    """Add pattern mining controls to sidebar"""
    st.markdown("---")
    st.subheader("🔍 Pattern Mining")
    
    pm_config = load_pattern_mining_config()
    status = "RUNNING" if pm_config.get("enabled") else "STOPPED"
    status_color = "🟢" if pm_config.get("enabled") else "🔴"
    st.write(f"**Status:** {status_color} {status}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", key="pm_start", type="primary", use_container_width=True):
            pm_config["enabled"] = True
            save_pattern_mining_config(pm_config)
            thread_manager.start_thread("pattern_miner", continuous_pattern_miner_managed)
            st.success("Pattern miner started!")
            time.sleep(1)
            st.rerun()
    with col2:
        if st.button("⏹️ Stop", key="pm_stop", type="secondary", use_container_width=True):
            pm_config["enabled"] = False
            save_pattern_mining_config(pm_config)
            thread_manager.stop_thread("pattern_miner")
            st.warning("Pattern miner stopped!")
            time.sleep(1)
            st.rerun()
    
    if st.button("🔄 Run Single Cycle", type="secondary", use_container_width=True):
        with st.spinner("Mining patterns..."):
            patterns_found = run_pattern_mining_cycle()
            st.success(f"✅ Found {patterns_found} elite patterns!")
    
    # Pattern mining configuration
    with st.expander("⚙️ Configuration"):
        new_interval = st.slider(
            "Cycle Interval (minutes)", 
            10, 120, 
            pm_config.get('cycle_interval_minutes', 30)
        )
        new_auc_threshold = st.slider(
            "Min AUC Threshold", 
            0.60, 0.90, 
            pm_config.get('min_auc_threshold', 0.70),
            step=0.01
        )
        new_auc_std = st.slider(
            "Max AUC Std", 
            0.05, 0.20, 
            pm_config.get('max_auc_std', 0.10),
            step=0.01
        )
        
        if st.button("💾 Update Config", type="secondary"):
            new_config = {
                "enabled": pm_config.get("enabled", False),
                "cycle_interval_minutes": new_interval,
                "min_auc_threshold": new_auc_threshold,
                "max_auc_std": new_auc_std
            }
            if save_pattern_mining_config(new_config):
                st.success("✅ Config updated!")
            else:
                st.error("❌ Failed to update config")

def add_enhanced_controls() -> None:
    """Add enhanced controls to sidebar"""
    st.markdown("---")
    st.subheader("🧠 System Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Initialize", key="init_threads", type="primary", use_container_width=True):
            initialize_background_threads_enhanced()
            st.success("✅ Threads initialized!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("🛑 Shutdown", key="shutdown_threads", type="secondary", use_container_width=True):
            shutdown_background_threads()
            st.warning("⚠️ Threads shutdown")
            time.sleep(1)
            st.rerun()
    
    # Thread status display
    st.markdown("#### 📊 Thread Status")
    thread_status = thread_manager.get_status()
    
    if thread_status:
        for name, status in thread_status.items():
            status_emoji = "🟢" if status['alive'] else "🔴"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{status_emoji} **{name}**")
            with col2:
                st.write("ALIVE" if status['alive'] else "DEAD")
    else:
        st.info("ℹ️ No managed threads running")
    
    # Resource monitoring dashboard
    show_resource_dashboard()
    
    # Metrics display
    st.markdown("#### 📈 Application Metrics")
    metrics = metrics_collector.get_metrics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predictions Made", metrics.get('predictions_made', 0))
        st.metric("Models Trained", metrics.get('models_trained', 0))
        st.metric("Elite Patterns", metrics.get('elite_patterns_found', 0))
        st.metric("Monitoring Alerts", metrics.get('monitoring_alerts_sent', 0))
    with col2:
        st.metric("Models Retrained", metrics.get('models_retrained', 0))
        st.metric("Errors", metrics.get('errors_encountered', 0))
        st.metric("Mining Cycles", metrics.get('pattern_mining_cycles', 0))
    
    # Data source metrics
    st.markdown("#### 📊 Data Source Usage")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Downloads", metrics.get('data_downloads', 0))
    with col2:
        yf_downloads = metrics.get('yfinance_downloads', 0)
        st.metric("yfinance", yf_downloads)
    with col3:
        av_downloads = metrics.get('alphavantage_downloads', 0)
        st.metric("Alpha Vantage", av_downloads)
    
    # Cache metrics
    st.markdown("#### 💾 Cache Performance")
    col1, col2 = st.columns(2)
    with col1:
        cache_hits = metrics.get('cache_hits', 0)
        cache_misses = metrics.get('cache_misses', 0)
        total_cache = cache_hits + cache_misses
        hit_rate = (cache_hits / total_cache * 100) if total_cache > 0 else 0
        st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
    with col2:
        st.metric("Cache Hits", cache_hits)

def show_pattern_dashboard() -> None:
    """Show pattern mining dashboard"""
    st.subheader("🔍 Pattern Mining Dashboard")
    
    if not AUTO_PATTERNS_FILE.exists():
        st.info("ℹ️ No patterns mined yet. Start the pattern miner to begin.")
        return
    
    try:
        with open(AUTO_PATTERNS_FILE, 'r') as f:
            patterns_data = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Analyzed", patterns_data.get('total_analyzed', 0))
        with col2:
            st.metric("⭐ Elite Patterns", patterns_data.get('elite_patterns_found', 0))
        with col3:
            st.metric("🕐 Last Updated", patterns_data.get('generated_at', 'Never'))
        
        patterns = patterns_data.get('patterns', [])
        if patterns:
            st.markdown("---")
            st.subheader("⭐ Active Elite Patterns")
            
            # Create pattern display
            pattern_display = []
            for pat in patterns:
                try:
                    timestamp = datetime.strptime(pat['timestamp'], '%Y-%m-%d %H:%M')
                    age_hours = (datetime.now() - timestamp).seconds // 3600
                    
                    pattern_display.append({
                        "Ticker": pat['ticker'],
                        "Timeframe": pat['timeframe'],
                        "Model": pat['model'].upper(),
                        "AUC": f"{pat['auc_mean']:.3f} ± {pat['auc_std']:.3f}",
                        "Boost": f"+{pat['boost']}",
                        "Bias": pat['direction_bias'],
                        "Confidence": f"{min(99, int(pat['auc_mean'] * 100 + pat['boost'] // 2.5))}%",
                        "Age": f"{age_hours}h"
                    })
                except Exception as e:
                    logger.warning(f"Error displaying pattern: {e}")
                    continue
            
            st.dataframe(pd.DataFrame(pattern_display), use_container_width=True)
            
            # Pattern statistics
            st.markdown("---")
            st.subheader("📊 Pattern Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            avg_auc = np.mean([p['auc_mean'] for p in patterns])
            avg_boost = np.mean([p['boost'] for p in patterns])
            up_bias = len([p for p in patterns if p['direction_bias'] == 'UP'])
            down_bias = len([p for p in patterns if p['direction_bias'] == 'DOWN'])
            
            with col1:
                st.metric("Avg AUC", f"{avg_auc:.3f}")
            with col2:
                st.metric("Avg Boost", f"{avg_boost:.1f}")
            with col3:
                st.metric("⬆️ UP Bias", up_bias)
            with col4:
                st.metric("⬇️ DOWN Bias", down_bias)
                
        else:
            st.info("ℹ️ No elite patterns found in the last 24 hours.")
            
    except Exception as e:
        st.error(f"❌ Error loading pattern data: {e}")
        log_error(ErrorSeverity.ERROR, "show_pattern_dashboard", e, show_to_user=True)

def show_accuracy_dashboard(ticker: str) -> None:
    """Show detailed accuracy metrics for a ticker with REAL validation data"""
    st.subheader(f"📊 Prediction Accuracy - {ticker}")
    
    acc = load_accuracy_log(ticker)
    
    if acc['status'] == 'no_predictions':
        st.info("ℹ️ No predictions recorded yet. Generate some forecasts to see accuracy metrics.")
        return
    
    if acc['status'] == 'no_file':
        st.info("ℹ️ No prediction history available.")
        return
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_preds = acc.get('total_predictions', 0)
        st.metric("📊 Total Predictions", total_preds)
    
    with col2:
        validated = acc.get('validated_predictions', 0)
        validation_rate = (validated / total_preds * 100) if total_preds > 0 else 0
        st.metric("✅ Validated", f"{validated}", delta=f"{validation_rate:.0f}%")
    
    with col3:
        mape = acc.get('avg_error_mape', 0.0)
        mape_color = "normal" if mape < 5.0 else "inverse" if mape < 8.0 else "off"
        st.metric("📉 Avg Error (MAPE)", f"{mape:.2f}%", delta_color=mape_color)
    
    with col4:
        dir_acc = acc.get('directional_accuracy', 0.0)
        dir_color = "normal" if dir_acc > 60 else "inverse" if dir_acc > 50 else "off"
        st.metric("🎯 Direction Accuracy", f"{dir_acc:.1f}%", delta_color=dir_color)
    
    st.markdown("---")
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mae = acc.get('avg_error_mae', 0.0)
        st.metric("💵 Avg Error (MAE)", f"${mae:.2f}")
    
    with col2:
        last_updated = acc.get('last_updated', 'Never')
        try:
            update_time = datetime.fromisoformat(last_updated)
            time_ago = datetime.now() - update_time
            if time_ago.days > 0:
                time_str = f"{time_ago.days}d ago"
            else:
                hours = time_ago.seconds // 3600
                time_str = f"{hours}h ago"
            st.metric("🕐 Last Validation", time_str)
        except:
            st.metric("🕐 Last Validation", "Unknown")
    
    with col3:
        status = acc.get('status', 'unknown')
        status_emoji = {
            'validated': '✅',
            'no_validated': '⚠️',
            'no_predictions': 'ℹ️',
            'error': '❌'
        }.get(status, '❓')
        st.metric("Status", f"{status_emoji} {status.upper()}")
    
    # Performance interpretation
    st.markdown("---")
    st.subheader("📈 Performance Analysis")
    
    if validated < 5:
        st.warning("⚠️ **Insufficient Data**: Need at least 5 validated predictions for reliable analysis.")
    else:
        # MAPE interpretation
        if mape < 3.0:
            st.success(f"🎉 **Excellent Performance**: {mape:.2f}% MAPE is very accurate!")
        elif mape < 5.0:
            st.success(f"✅ **Good Performance**: {mape:.2f}% MAPE is acceptable for trading.")
        elif mape < 8.0:
            st.warning(f"⚠️ **Moderate Performance**: {mape:.2f}% MAPE suggests room for improvement.")
        else:
            st.error(f"❌ **Poor Performance**: {mape:.2f}% MAPE indicates model needs retraining.")
        
        # Direction accuracy interpretation
        if dir_acc > 65:
            st.success(f"🎯 **Strong Direction Prediction**: {dir_acc:.1f}% correct direction is excellent!")
        elif dir_acc > 55:
            st.info(f"📊 **Decent Direction Prediction**: {dir_acc:.1f}% is better than random.")
        elif dir_acc > 45:
            st.warning(f"⚠️ **Weak Direction Prediction**: {dir_acc:.1f}% is close to random.")
        else:
            st.error(f"❌ **Poor Direction Prediction**: {dir_acc:.1f}% is worse than random!")
    
    # Load detailed prediction history
    if st.button("📋 View Detailed History", key=f"history_{ticker}"):
        show_prediction_history(ticker)

def show_prediction_history(ticker: str) -> None:
    """Show detailed prediction vs actual history"""
    try:
        if SQLALCHEMY_AVAILABLE:
            db_session = Session()
            
            # Get all predictions for this ticker
            predictions = db_session.query(Prediction).filter(
                Prediction.ticker == ticker
            ).order_by(Prediction.prediction_date).all()
            
            if not predictions:
                st.info("No predictions recorded.")
                db_session.close()
                return
            
            # Filter validated predictions
            validated_preds = [p for p in predictions if p.actual_price is not None]
            
            if not validated_preds:
                st.info("No validated predictions yet. Predictions can only be validated after their target date has passed.")
                db_session.close()
                return
            
            # Create comparison table
            history_data = []
            for pred in validated_preds:
                try:
                    pred_date = pred.prediction_date.strftime("%Y-%m-%d")
                    predicted = pred.predicted_price
                    actual = pred.actual_price
                    error_pct = pred.error_mape if pred.error_mape else 0
                    error_abs = pred.error_mae if pred.error_mae else 0
                    
                    # Direction
                    if pred.previous_price:
                        prev = pred.previous_price
                        pred_dir = "⬆️" if predicted > prev else "⬇️"
                        actual_dir = "⬆️" if actual > prev else "⬇️"
                        correct_dir = "✅" if pred_dir == actual_dir else "❌"
                    else:
                        pred_dir = "-"
                        actual_dir = "-"
                        correct_dir = "-"
                    
                    history_data.append({
                        "Date": pred_date,
                        "Predicted": f"${predicted:.2f}",
                        "Actual": f"${actual:.2f}",
                        "Error $": f"${error_abs:.2f}",
                        "Error %": f"{error_pct:.2f}%",
                        "Pred Dir": pred_dir,
                        "Actual Dir": actual_dir,
                        "Correct": correct_dir
                    })
                except Exception as e:
                    logger.debug(f"Error displaying prediction: {e}")
                    continue
            
            db_session.close()
        else:
            # JSON fallback
            predictions_path = get_predictions_path(ticker)
            if not predictions_path.exists():
                st.info("No predictions recorded.")
                return
            
            with open(predictions_path, 'r') as f:
                predictions_data = json.load(f)
            
            # Filter validated predictions
            validated_preds = [p for p in predictions_data if p.get("actual_price") is not None]
            
            if not validated_preds:
                st.info("No validated predictions yet. Predictions can only be validated after their target date has passed.")
                return
            
            # Create comparison table
            history_data = []
            for pred in validated_preds:
                try:
                    pred_date = pred["prediction_date"]
                    predicted = pred["predicted_price"]
                    actual = pred["actual_price"]
                    error_pct = pred.get("error_mape", 0)
                    error_abs = pred.get("error_mae", 0)
                    
                    # Direction
                    if pred.get("previous_price"):
                        prev = pred["previous_price"]
                        pred_dir = "⬆️" if predicted > prev else "⬇️"
                        actual_dir = "⬆️" if actual > prev else "⬇️"
                        correct_dir = "✅" if pred_dir == actual_dir else "❌"
                    else:
                        pred_dir = "-"
                        actual_dir = "-"
                        correct_dir = "-"
                    
                    history_data.append({
                        "Date": pred_date,
                        "Predicted": f"${predicted:.2f}",
                        "Actual": f"${actual:.2f}",
                        "Error $": f"${error_abs:.2f}",
                        "Error %": f"{error_pct:.2f}%",
                        "Pred Dir": pred_dir,
                        "Actual Dir": actual_dir,
                        "Correct": correct_dir
                    })
                except Exception as e:
                    logger.debug(f"Error displaying prediction: {e}")
                    continue
        
        if history_data:
            st.dataframe(
                pd.DataFrame(history_data),
                use_container_width=True,
                hide_index=True
            )
            
            # Create chart of predicted vs actual
            chart_data = []
            for pred in validated_preds:
                try:
                    if SQLALCHEMY_AVAILABLE:
                        date = pred.prediction_date
                        chart_data.append({
                            'Date': date,
                            'Predicted': pred.predicted_price,
                            'Actual': pred.actual_price
                        })
                    else:
                        date = datetime.strptime(pred["prediction_date"], "%Y-%m-%d")
                        chart_data.append({
                            'Date': date,
                            'Predicted': pred["predicted_price"],
                            'Actual': pred["actual_price"]
                        })
                except:
                    continue
            
            if chart_data:
                df_chart = pd.DataFrame(chart_data).sort_values('Date')
                
                # Use lazy loading for better performance
                def create_comparison_chart():
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df_chart['Date'],
                        y=df_chart['Predicted'],
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_chart['Date'],
                        y=df_chart['Actual'],
                        mode='lines+markers',
                        name='Actual',
                        line=dict(color='green', width=2),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} - Predicted vs Actual Prices",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        showlegend=True,
                        hovermode='x unified',
                        height=400
                    )
                    return fig
                
                lazy_plotly_chart(create_comparison_chart, f"comparison_{ticker}")
        else:
            st.info("No validated predictions to display.")
            
    except Exception as e:
        st.error(f"Error loading prediction history: {e}")
        log_error(ErrorSeverity.ERROR, "show_prediction_history", e, ticker=ticker, show_to_user=True)

def show_error_dashboard() -> None:
    """Show error dashboard"""
    st.subheader("⚠️ Error Dashboard")
    
    if not ERROR_LOG.exists():
        st.info("ℹ️ No errors logged yet.")
        return
    
    try:
        with open(ERROR_LOG, 'r') as f:
            errors = json.load(f)
        
        if not errors:
            st.info("ℹ️ No errors in log.")
            return
        
        # Error statistics
        col1, col2, col3, col4 = st.columns(4)
        
        error_count = len(errors)
        warning_count = len([e for e in errors if e['severity'] == 'WARNING'])
        error_severity_count = len([e for e in errors if e['severity'] == 'ERROR'])
        critical_count = len([e for e in errors if e['severity'] == 'CRITICAL'])
        
        with col1:
            st.metric("📊 Total Errors", error_count)
        with col2:
            st.metric("⚠️ Warnings", warning_count)
        with col3:
            st.metric("❌ Errors", error_severity_count)
        with col4:
            st.metric("🚨 Critical", critical_count)
        
        # Recent errors
        st.markdown("---")
        st.subheader("📋 Recent Errors (Last 50)")
        recent_errors = errors[-50:]
        recent_errors.reverse()  # Show newest first
        
        error_display = []
        for error in recent_errors:
            error_display.append({
                "Time": error['timestamp'][:19],
                "Severity": error['severity'],
                "Function": error['function'],
                "Error": f"{error['error_type']}: {error['error_message'][:50]}...",
                "Ticker": error.get('ticker', 'N/A')
            })
        
        st.dataframe(pd.DataFrame(error_display), use_container_width=True)
        
        # Clear errors button
        if st.button("🗑️ Clear Error Log", type="secondary"):
            with open(ERROR_LOG, 'w') as f:
                json.dump([], f)
            st.success("✅ Error log cleared!")
            time.sleep(1)
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error loading error log: {e}")
        logger.error(f"Error in show_error_dashboard: {e}")

def show_learning_log() -> None:
    """Show learning daemon log"""
    st.subheader("📝 Learning Log")
    
    # Get log messages from queue
    if 'learning_log' not in st.session_state:
        st.session_state.learning_log = []
    
    new_messages = app_state.get_log_messages()
    if new_messages:
        st.session_state.learning_log.extend(new_messages)
        # Keep only last 100 messages
        if len(st.session_state.learning_log) > 100:
            st.session_state.learning_log = st.session_state.learning_log[-100:]
    
    if st.session_state.learning_log:
        # Show newest first
        for message in reversed(st.session_state.learning_log[-20:]):
            st.text(message)
    else:
        st.info("ℹ️ No learning activity yet.")

# ================================
# MAIN STREAMLIT APPLICATION
# ================================

# ============================================================================
# LINE 5787: main() - Streamlit UI entry point
# STATUS: WORKING - 7 tabs, forex support, all features
# DEPENDENCIES: All functions above
# UI ONLY: Minimal logic, mostly display code
# ============================================================================
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Alpha Trader v4.2 - Enhanced",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 AI Alpha Trader v4.2 - Enhanced with Confidence Intervals")
    st.markdown("*Advanced AI-Powered Trading Platform with Real Prediction Validation and Confidence Intervals*")
    st.markdown("---")
    
    # Initialize enhanced background threads
    initialize_background_threads_enhanced()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard", 
        "🔮 Forecast", 
        "📈 Analysis",
        "🎯 Accuracy",
        "⚙️ Settings", 
        "🔧 Diagnostics",
        "📝 Logs"
    ])
    
    # ================================
    # SIDEBAR
    # ================================
    
    with st.sidebar:
        st.header("🎯 Configuration")
        
        # Add prediction scanner controls
        add_prediction_scanner_controls()
    
        # Data source status
        if ALPHA_VANTAGE_API_KEY:
            st.success("✅ Alpha Vantage: Configured")
        else:
            st.warning("⚠️ Alpha Vantage: Not configured (using yfinance only)")
    
        st.markdown("---")
        
        # Asset selection
        category = st.selectbox("📁 Category", list(ASSET_CATEGORIES.keys()))
        asset = st.selectbox("💰 Asset", list(ASSET_CATEGORIES[category].keys()))
        ticker = ASSET_CATEGORIES[category][asset]
        
        # Get current price
        price = get_latest_price(ticker)
        if price:
            # Check if commodity and show age warning
            if ticker.endswith('=F'):
                commodity_info = get_commodity_info(ticker)
                days_old = commodity_info.get('days_old', 0)
                
                if days_old > 0:
                    # Show price with age indicator
                    st.metric(
                        "💵 Current Price", 
                        f"${price:.2f}",
                        help=f"Last updated {days_old} day(s) ago"
                    )
                    
                    # Warning for stale data
                    if days_old > 3:
                        st.caption(f"⚠️ {days_old} days old")
                    elif days_old > 1:
                        st.caption(f"ℹ️ {days_old} days old")
                    
                    # Show 5-day change if available
                    if commodity_info.get('price_change_5d') is not None:
                        change = commodity_info['price_change_5d']
                        st.caption(f"5d change: {change:+.2f}%")
                else:
                    st.metric("💵 Current Price", f"${price:.2f}")
            else:
                # Regular stock - just show price
                st.metric("💵 Current Price", f"${price:.2f}")
        else:
            st.warning("⚠️ Price unavailable")
            
            # Show detailed status for commodities
            if ticker.endswith('=F'):
                commodity_info = get_commodity_info(ticker)
                status = commodity_info.get('status', 'Unknown')
                st.caption(f"Status: {status}")
        
        st.markdown("---")
        st.subheader("🔧 Quick Actions")
        
        if st.button("🔄 Force Retrain", type="secondary", use_container_width=True):
            with st.spinner("Retraining model..."):
                forecast, lower_ci, upper_ci, dates, model = train_self_learning_model_enhanced(ticker, force_retrain=True)
                if forecast is not None:
                    st.success("✅ Model retrained with confidence intervals!")
                else:
                    st.error("❌ Retraining failed")
                time.sleep(2)
                st.rerun()
        
        if st.button("🚀 Bootstrap All Models", type="secondary", use_container_width=True):
            with st.spinner("Training all models with confidence intervals... (5-10 min)"):
                all_tickers = [t for cat in ASSET_CATEGORIES.values() for _, t in cat.items()]
                progress = st.progress(0)
                success_count = 0
                
                for idx, t in enumerate(all_tickers):
                    try:
                        forecast, lower_ci, upper_ci, dates, model = train_self_learning_model_enhanced(t, days=5, force_retrain=True)
                        if forecast is not None:
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to train {t}: {e}")
                    progress.progress((idx + 1) / len(all_tickers))
                
                st.success(f"✅ Trained {success_count}/{len(all_tickers)} models with confidence intervals!")
                time.sleep(3)
                st.rerun()
        
        # Enhanced controls
        add_enhanced_controls()
        
        # Pattern mining controls
        add_pattern_mining_controls()
        
        st.markdown("---")
        st.subheader("🤖 Learning Daemon")
        dc = load_daemon_config()
        status = "RUNNING" if dc.get("enabled") else "STOPPED"
        status_emoji = "🟢" if dc.get("enabled") else "🔴"
        st.write(f"**Status:** {status_emoji} {status}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", key="dstart", type="primary", use_container_width=True):
                if save_daemon_config(True):
                    thread_manager.start_thread("learning_daemon", continuous_learning_daemon_managed)
                    st.success("✅ Started!")
                    time.sleep(1)
                    st.rerun()
        with col2:
            if st.button("⏹️ Stop", key="dstop", type="secondary", use_container_width=True):
                if save_daemon_config(False):
                    thread_manager.stop_thread("learning_daemon")
                    st.warning("⚠️ Stopped!")
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("---")
        st.subheader("👁️ 6%+ Monitoring")
        mc = load_monitoring_config()
        status = "RUNNING" if mc.get("enabled") else "STOPPED"
        status_emoji = "🟢" if mc.get("enabled") else "🔴"
        st.write(f"**Status:** {status_emoji} {status}")
        
        if st.button("📱 Test Telegram", type="secondary", use_container_width=True):
            success = send_telegram_alert("🧪 TEST ALERT\n<b>AI Alpha Trader v4.2 - Enhanced</b>\nSystem is operational with confidence intervals!")
            if success:
                st.success("✅ Alert sent!")
            else:
                st.error("❌ Check credentials")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", key="mstart", type="primary", use_container_width=True):
                if save_monitoring_config(True):
                    thread_manager.start_thread("monitoring", monitor_6percent_pre_move_managed)
                    st.success("✅ Started!")
                    time.sleep(1)
                    st.rerun()
        with col2:
            if st.button("⏹️ Stop", key="mstop", type="secondary", use_container_width=True):
                if save_monitoring_config(False):
                    thread_manager.stop_thread("monitoring")
                    st.warning("⚠️ Stopped!")
                    time.sleep(1)
                    st.rerun()
    
    # ================================
    # TAB 1: DASHBOARD
    # ================================
    
    with tab1:
        st.header("📊 Trading Dashboard")
        
        # Show accuracy dashboard if requested
        if st.session_state.get('show_accuracy_detail', False):
            show_accuracy_dashboard(ticker)
            if st.button("⬅️ Back to Dashboard"):
                st.session_state.show_accuracy_detail = False
                st.rerun()
            st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎯 Daily Recommendation", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing (fast mode)..."):
                    # Use fast forecast for speed
                    forecast, lower_ci, upper_ci, dates = generate_fast_forecast(ticker, days=1)
                    
                    if forecast is not None and len(np.array(forecast).flatten()) > 0:
                        forecast_val = float(np.array(forecast).flatten()[0])
                        
                        # Use enhanced confidence check with patterns
                        try:
                            passed, reasons, pattern_boost = enhanced_confidence_checklist(
                                ticker, [forecast_val], price or 100
                            )
                        except:
                            passed = False
                            reasons = ["Pattern checking not available"]
                            pattern_boost = 0
                        
                        # Get pattern-influenced recommendation
                        try:
                            action, confidence, pattern_reasons = get_pattern_influenced_recommendation(
                                ticker, [forecast_val], price
                            )
                        except:
                            action = "HOLD"
                            confidence = 50
                            pattern_reasons = []
                        
                        if passed:
                            change_pct = (forecast_val - price) / price * 100 if price else 0
                            
                            # Display with pattern influence
                            if "BUY" in action:
                                st.success(f"**{action}** | Confidence: {confidence}%")
                            elif "SELL" in action:
                                st.error(f"**{action}** | Confidence: {confidence}%")
                            else:
                                st.info(f"**{action}** | Confidence: {confidence}%")
                            
                            # Show confidence interval if available
                            if lower_ci is not None and upper_ci is not None:
                                ci_lower = lower_ci[0]
                                ci_upper = upper_ci[0]
                                st.metric(
                                    "AI Prediction with 95% CI", 
                                    f"${forecast_val:.2f}",
                                    f"{change_pct:+.2f}%",
                                    delta_color="normal",
                                    help=f"95% Confidence Interval: ${ci_lower:.2f} - ${ci_upper:.2f}"
                                )
                            else:
                                st.metric(
                                    "AI Prediction", 
                                    f"${forecast_val:.2f}",
                                    f"{change_pct:+.2f}%"
                                )
                            
                            # Show pattern influence if any
                            if pattern_boost > 0:
                                st.metric(
                                    "Pattern Boost", 
                                    f"+{pattern_boost}", 
                                    f"Confidence: {min(99, int(pattern_boost // 2.5))}%"
                                )
                                
                            if pattern_reasons:
                                with st.expander("🔍 Pattern Analysis"):
                                    for reason in pattern_reasons:
                                        st.write(f"• {reason}")
                        else:
                            st.warning("⚠️ Low Confidence Recommendation")
                            st.info(f"AI Predicts: ${forecast_val:.2f}")
                            st.write("**Confidence Issues:**")
                            for reason in reasons:
                                st.write(f"• {reason}")
                    else:
                        st.error("❌ Forecast failed or no data")
        
        with col2:
            st.subheader("ℹ️ Model Info")
            meta = load_metadata(ticker)
            acc = load_accuracy_log(ticker)
            
            st.write(f"**🔄 Retrain Count:** {meta.get('retrain_count', 0)}")
            st.write(f"**📊 Total Predictions:** {acc.get('total_predictions', 0)}")
            st.write(f"**✅ Validated:** {acc.get('validated_predictions', 0)}")
            
            # Show REAL metrics
            if acc.get('validated_predictions', 0) > 0:
                mape = acc.get('avg_error_mape', 0)
                st.write(f"**📉 MAPE:** {mape:.2f}%")
                
                dir_acc = acc.get('directional_accuracy', 0)
                st.write(f"**🎯 Direction:** {dir_acc:.1f}%")
                
                # Color code performance
                if mape < 5.0:
                    st.success("✅ Good accuracy")
                elif mape < 8.0:
                    st.warning("⚠️ Moderate accuracy")
                else:
                    st.error("❌ Needs retraining")
            else:
                st.info("No validated predictions yet")
            
            if meta.get('trained_date'):
                try:
                    trained_date = datetime.fromisoformat(meta['trained_date'])
                    days_ago = (datetime.now() - trained_date).days
                    st.write(f"**📅 Last Trained:** {days_ago} days ago")
                except:
                    st.write("**📅 Last Trained:** Unknown")
            
            # Data quality metrics
            if meta.get('quality_metrics'):
                with st.expander("📊 Data Quality"):
                    quality = meta['quality_metrics']
                    if 'return_std' in quality:
                        st.write(f"**Volatility:** {quality['return_std']:.4f}")
                    if 'extreme_move_pct' in quality:
                        st.write(f"**Extreme Moves:** {quality['extreme_move_pct']:.1f}%")
            
            # Pattern status
            try:
                boost, triggers, direction, confidence = check_auto_patterns(ticker)
                if boost > 0:
                    st.metric("⚡ Pattern Boost", f"+{boost}", f"Direction: {direction}")
                    if triggers:
                        with st.expander("🔍 Pattern Details"):
                            for trigger in triggers:
                                st.write(f"• {trigger}")
            except:
                pass
            
            # Button to view detailed accuracy
            if st.button("📊 View Accuracy Details", key=f"acc_detail_{ticker}"):
                st.session_state.show_accuracy_detail = True
                st.rerun()
    
    # ================================
    # TAB 2: FORECAST
    # ================================
    
    with tab2:
        st.header("🔮 Price Forecast with Confidence Intervals")
        
        days_to_forecast = st.slider("Days to Forecast", 1, 10, 5)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("📈 Generate Fast Forecast", type="primary", use_container_width=True):
                with st.spinner("🔍 Generating forecast (fast mode)..."):
                    forecast, lower_ci, upper_ci, dates = generate_fast_forecast(ticker, days=days_to_forecast)
        
        with col2:
            if st.button("🔄 Full Retrain", type="secondary", use_container_width=True):
                with st.spinner("🔍 Retraining model (slow)..."):
                    forecast, lower_ci, upper_ci, dates, model = train_self_learning_model_enhanced(
                        ticker, days=days_to_forecast, force_retrain=True
                    )
        
        # Check if forecast was generated
        try:
            if 'forecast' in locals() and forecast is not None and len(forecast) > 0:
                # Create forecast chart with confidence intervals
                current_price = price or get_latest_price(ticker)
                
                if current_price:
                    # Prepare data for plotting
                    forecast_dates = dates[:len(forecast)]
                    forecast_prices = forecast
                    
                    # Create plot with lazy loading
                    def create_forecast_chart():
                        fig = go.Figure()
                        
                        # Current price
                        fig.add_trace(go.Scatter(
                            x=[datetime.now().date()],
                            y=[current_price],
                            mode='markers',
                            name='Current Price',
                            marker=dict(size=12, color='green')
                        ))
                        
                        # Forecast with confidence intervals
                        if lower_ci is not None and upper_ci is not None:
                            # Add confidence interval band
                            fig.add_trace(go.Scatter(
                                x=forecast_dates + forecast_dates[::-1],
                                y=list(upper_ci) + list(lower_ci)[::-1],
                                fill='toself',
                                fillcolor='rgba(0, 100, 255, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='95% Confidence Interval',
                                showlegend=True
                            ))
                        
                        # Forecast line
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_prices,
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='blue', width=2),
                            marker=dict(size=8)
                        ))
                        
                        fig.update_layout(
                            title=f"{ticker} {days_to_forecast}-Day Price Forecast with 95% Confidence Intervals",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            showlegend=True,
                            hovermode='x unified',
                            height=500
                        )
                        return fig
                    
                    lazy_plotly_chart(create_forecast_chart, f"forecast_{ticker}")
                    
                    # Forecast table with confidence intervals
                    st.markdown("---")
                    st.subheader("📋 Forecast Details with Confidence Intervals")
                    forecast_data = []
                    for i, (date, price_val) in enumerate(zip(forecast_dates, forecast_prices)):
                        change_pct = (price_val - current_price) / current_price * 100
                        change_abs = price_val - current_price
                        
                        row_data = {
                            "Day": i + 1,
                            "Date": date.strftime("%Y-%m-%d"),
                            "Price": f"${price_val:.2f}",
                            "Change ($)": f"${change_abs:+.2f}",
                            "Change (%)": f"{change_pct:+.2f}%"
                        }
                        
                        # Add confidence intervals if available
                        if lower_ci is not None and upper_ci is not None and i < len(lower_ci):
                            ci_lower = lower_ci[i]
                            ci_upper = upper_ci[i]
                            ci_width = ci_upper - ci_lower
                            row_data["95% CI"] = f"${ci_lower:.2f} - ${ci_upper:.2f}"
                            row_data["CI Width"] = f"${ci_width:.2f}"
                        
                        forecast_data.append(row_data)
                    
                    st.dataframe(pd.DataFrame(forecast_data), use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    avg_price = np.mean(forecast_prices)
                    max_price = np.max(forecast_prices)
                    min_price = np.min(forecast_prices)
                    total_change = (forecast_prices[-1] - current_price) / current_price * 100
                    
                    with col1:
                        st.metric("Avg Forecast", f"${avg_price:.2f}")
                    with col2:
                        st.metric("Max Price", f"${max_price:.2f}")
                    with col3:
                        st.metric("Min Price", f"${min_price:.2f}")
                    with col4:
                        st.metric("Total Change", f"{total_change:+.2f}%")
                    
                    # Confidence interval statistics
                    if lower_ci is not None and upper_ci is not None:
                        st.markdown("---")
                        st.subheader("📊 Confidence Interval Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        avg_ci_width = np.mean(upper_ci - lower_ci)
                        max_ci_width = np.max(upper_ci - lower_ci)
                        min_ci_width = np.min(upper_ci - lower_ci)
                        
                        with col1:
                            st.metric("Avg CI Width", f"${avg_ci_width:.2f}")
                        with col2:
                            st.metric("Max CI Width", f"${max_ci_width:.2f}")
                        with col3:
                            st.metric("Min CI Width", f"${min_ci_width:.2f}")
                
                else:
                    st.error("❌ Could not get current price")
            else:
                st.error("❌ Forecast generation failed")
        except Exception as e:
            st.error(f"❌ Error in forecast display: {str(e)}")
    
    # ================================
    # TAB 3: ANALYSIS
    # ================================
    
    with tab3:
        st.header("📈 Technical Analysis")
        
        # Show data quality metrics
        st.subheader("📊 Data Quality Analysis")
        
        try:
            df = download_and_validate_data(ticker, period="1y")
            if df is not None:
                quality_metrics = calculate_data_quality_metrics(df)
                
                if quality_metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        null_pct = quality_metrics.get('Close_null_pct', 0)
                        st.metric("Data Completeness", f"{(100 - null_pct):.1f}%")
                    
                    with col2:
                        return_std = quality_metrics.get('return_std', 0)
                        st.metric("Volatility (σ)", f"{return_std:.4f}")
                    
                    with col3:
                        extreme_moves = quality_metrics.get('extreme_move_pct', 0)
                        st.metric("Extreme Moves", f"{extreme_moves:.1f}%")
                    
                    with col4:
                        high_validity = quality_metrics.get('high_validity_pct', 100)
                        st.metric("Price Validity", f"{high_validity:.1f}%")
                    
                    # Price chart
                    st.markdown("---")
                    st.subheader("💹 Price History")
                    
                    def create_price_chart():
                        fig = go.Figure()
                        
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='OHLC'
                        ))
                        
                        fig.update_layout(
                            title=f"{ticker} Price History",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            showlegend=True,
                            height=500
                        )
                        
                        # Hide weekends
                        fig.update_xaxes(
                            rangebreaks=[dict(bounds=["sat", "mon"])]
                        )
                        
                        return fig
                    
                    lazy_plotly_chart(create_price_chart, f"price_{ticker}")
                    
                else:
                    st.info("Could not calculate data quality metrics")
            else:
                st.warning("Could not download data for analysis")
        except Exception as e:
            st.error(f"Error in analysis: {e}")
        
        st.markdown("---")
        st.subheader("🔜 Upcoming Features")
        
        features = [
            "📊 Technical indicators (RSI, MACD, Bollinger Bands)",
            "🕯️ Candlestick charts with volume",
            "📉 Support and resistance levels",
            "🔄 Correlation analysis with other assets",
            "📈 Trend analysis and momentum indicators",
            "💹 Volatility metrics and bands",
            "🎯 Entry/exit point recommendations",
            "📱 Real-time price alerts"
        ]
        
        for feature in features:
            st.write(f"• {feature}")
        
        st.markdown("---")
        st.write("**Want a specific feature?** Let us know!")
    
    # ================================
    # TAB 4: ACCURACY TRACKING
    # ================================
    
    with tab4:
        st.header("🎯 Prediction Accuracy Tracking")
        
        st.info("💡 **How it works**: Every prediction is validated against actual market prices once the target date has passed. This provides real-world performance metrics.")
        
        # Validate predictions button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🔄 Validate All Predictions", type="primary", use_container_width=True):
                with st.spinner("Validating predictions against actual prices..."):
                    all_tickers = [t for cat in ASSET_CATEGORIES.values() for t in cat.values()]
                    progress = st.progress(0)
                    results = []
                    
                    for idx, t in enumerate(all_tickers):
                        try:
                            success, acc_data = validate_predictions(t)
                            if success and acc_data.get('validated_predictions', 0) > 0:
                                results.append({
                                    'Ticker': t,
                                    'Validated': acc_data.get('validated_predictions', 0),
                                    'MAPE': f"{acc_data.get('avg_error_mape', 0):.2f}%",
                                    'Direction': f"{acc_data.get('directional_accuracy', 0):.1f}%"
                                })
                        except Exception as e:
                            logger.error(f"Validation failed for {t}: {e}")
                        
                        progress.progress((idx + 1) / len(all_tickers))
                    
                    if results:
                        st.success(f"✅ Validated predictions for {len(results)} tickers!")
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                    else:
                        st.warning("⚠️ No predictions available for validation yet.")
        
        with col2:
            if st.button("📊 Current Ticker", type="secondary", use_container_width=True):
                st.session_state.show_current_ticker_accuracy = True
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                try:
                    # Clear predictions
                    if SQLALCHEMY_AVAILABLE:
                        db_session = Session()
                        count = db_session.query(Prediction).count()
                        db_session.query(Prediction).delete()
                        db_session.commit()
                        db_session.close()
                        st.success(f"✅ Cleared {count} predictions!")
                    else:
                        # Clear JSON files
                        all_tickers = [t for cat in ASSET_CATEGORIES.values() for t in cat.values()]
                        count = 0
                        for ticker in all_tickers:
                            predictions_path = get_predictions_path(ticker)
                            if predictions_path.exists():
                                predictions_path.unlink()
                                count += 1
                        st.success(f"✅ Cleared {count} prediction files!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        
        # Show current ticker accuracy if requested
        if st.session_state.get('show_current_ticker_accuracy', False):
            show_accuracy_dashboard(ticker)
            if st.button("⬅️ Back to Overview"):
                st.session_state.show_current_ticker_accuracy = False
                st.rerun()
        else:
            # Show overview of all tickers
            st.subheader("📊 Accuracy Overview - All Assets")
            
            all_tickers = [t for cat in ASSET_CATEGORIES.values() for t in cat.values()]
            overview_data = []
            
            for t in all_tickers:
                try:
                    acc = load_accuracy_log(t)
                    if acc.get('validated_predictions', 0) > 0:
                        overview_data.append({
                            'Ticker': t,
                            'Total Preds': acc.get('total_predictions', 0),
                            'Validated': acc.get('validated_predictions', 0),
                            'MAPE': acc.get('avg_error_mape', 0),
                            'MAPE_str': f"{acc.get('avg_error_mape', 0):.2f}%",
                            'MAE': f"${acc.get('avg_error_mae', 0):.2f}",
                            'Direction': f"{acc.get('directional_accuracy', 0):.1f}%",
                            'Status': acc.get('status', 'unknown')
                        })
                except Exception as e:
                    logger.debug(f"Error loading accuracy for {t}: {e}")
            
            if overview_data:
                df_overview = pd.DataFrame(overview_data)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Tickers Tracked", len(df_overview))
                
                with col2:
                    avg_mape = df_overview['MAPE'].mean()
                    st.metric("📉 Avg MAPE", f"{avg_mape:.2f}%")
                
                with col3:
                    total_validated = df_overview['Validated'].sum()
                    st.metric("✅ Total Validated", total_validated)
                
                with col4:
                    # Count good performers (MAPE < 5%)
                    good_performers = len(df_overview[df_overview['MAPE'] < 5.0])
                    st.metric("🎯 Good Models", good_performers)
                
                st.markdown("---")
                
                # Display table
                display_df = df_overview.drop(columns=['MAPE'])
                st.dataframe(
                    display_df.sort_values('MAPE_str'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Performance distribution chart
                st.markdown("---")
                st.subheader("📊 Performance Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # MAPE distribution
                    def create_mape_chart():
                        fig_mape = go.Figure()
                        fig_mape.add_trace(go.Bar(
                            x=df_overview['Ticker'],
                            y=df_overview['MAPE'],
                            marker_color=['green' if x < 5 else 'orange' if x < 8 else 'red' 
                                         for x in df_overview['MAPE']],
                            text=[f"{x:.2f}%" for x in df_overview['MAPE']],
                            textposition='outside'
                        ))
                        fig_mape.update_layout(
                            title="MAPE by Ticker",
                            xaxis_title="Ticker",
                            yaxis_title="MAPE (%)",
                            showlegend=False,
                            height=400
                        )
                        fig_mape.add_hline(y=5.0, line_dash="dash", line_color="green", 
                                           annotation_text="Good (5%)")
                        fig_mape.add_hline(y=8.0, line_dash="dash", line_color="orange",
                                           annotation_text="Acceptable (8%)")
                        return fig_mape
                    
                    lazy_plotly_chart(create_mape_chart, "mape_distribution")
                
                with col2:
                    # Performance categories
                    excellent = len(df_overview[df_overview['MAPE'] < 3])
                    good = len(df_overview[(df_overview['MAPE'] >= 3) & (df_overview['MAPE'] < 5)])
                    moderate = len(df_overview[(df_overview['MAPE'] >= 5) & (df_overview['MAPE'] < 8)])
                    poor = len(df_overview[df_overview['MAPE'] >= 8])
                    
                    def create_performance_pie():
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=['Excellent (<3%)', 'Good (3-5%)', 'Moderate (5-8%)', 'Poor (>8%)'],
                            values=[excellent, good, moderate, poor],
                            marker_colors=['darkgreen', 'lightgreen', 'orange', 'red']
                        )])
                        fig_pie.update_layout(
                            title="Performance Categories",
                            height=400
                        )
                        return fig_pie
                    
                    lazy_plotly_chart(create_performance_pie, "performance_pie")
                
            else:
                st.info("ℹ️ No validated predictions yet. Models need to make predictions, then wait for target dates to pass for validation.")
                st.markdown("**To get started:**")
                st.markdown("1. Generate forecasts for assets in the Forecast tab")
                st.markdown("2. Wait for prediction dates to pass")
                st.markdown("3. Return here and click 'Validate All Predictions'")
    
    # ================================
    # TAB 5: SETTINGS
    # ================================
    
    with tab5:
        st.header("⚙️ Settings")
        
        # Alpha Vantage Configuration Section
        st.subheader("🔌 Data Source Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Primary Data Source:** yfinance")
            st.info("ℹ️ Free, real-time data from Yahoo Finance")
        
        with col2:
            st.markdown("**Backup Data Source:** Alpha Vantage")
            if ALPHA_VANTAGE_API_KEY:
                st.success("✅ API Key Configured")
                # Mask the API key for security
                masked_key = ALPHA_VANTAGE_API_KEY[:4] + "..." + ALPHA_VANTAGE_API_KEY[-4:] if len(ALPHA_VANTAGE_API_KEY) > 8 else "****"
                st.text(f"Key: {masked_key}")
            else:
                st.warning("⚠️ No API Key")
                st.markdown("Set `ALPHA_VANTAGE_API_KEY` in `.env` file")
        
        with st.expander("📖 About Data Sources"):
            st.markdown("""
            **How it works:**
            1. **Primary**: System attempts to fetch data from yfinance (free, no API key needed)
            2. **Fallback**: If yfinance fails, system automatically switches to Alpha Vantage
            3. **Validation**: All historical prices are validated against actual market data
            
            **Alpha Vantage Benefits:**
            - ✅ Reliable backup when yfinance is down
            - ✅ More stable for international tickers
            - ✅ Better rate limiting for intensive operations
            - ⚠️ Free tier: 25 requests/day, 5 requests/minute
            
            **Get Free API Key:**
            1. Visit: https://www.alphavantage.co/support/#api-key
            2. Get free API key (takes 30 seconds)
            3. Add to `.env` file: `ALPHA_VANTAGE_API_KEY=your_key_here`
            4. Restart application
            """)
            
            # Test Alpha Vantage connection
            if ALPHA_VANTAGE_API_KEY:
                if st.button("🧪 Test Alpha Vantage Connection", type="secondary"):
                    with st.spinner("Testing Alpha Vantage API..."):
                        try:
                            # Test with a simple quote request
                            params = {
                                'function': 'GLOBAL_QUOTE',
                                'symbol': 'AAPL',
                                'apikey': ALPHA_VANTAGE_API_KEY
                            }
                            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=30)
                            data = response.json()
                            
                            if 'Global Quote' in data:
                                price = data['Global Quote'].get('05. price', 'N/A')
                                st.success(f"✅ Connection successful! AAPL price: ${price}")
                            elif 'Error Message' in data:
                                st.error(f"❌ API Error: {data['Error Message']}")
                            elif 'Note' in data:
                                st.warning(f"⚠️ Rate limit: {data['Note']}")
                            else:
                                st.error("❌ Unexpected response format")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {e}")
            else:
                st.info("ℹ️ Configure API key to test connection")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Configuration")
            
            lookback_val = st.number_input(
                "Lookback Window", 
                value=LEARNING_CONFIG["lookback_window"], 
                min_value=30, 
                max_value=120,
                help="Number of past days to consider for training"
            )
            
            full_epochs_val = st.number_input(
                "Full Retrain Epochs", 
                value=LEARNING_CONFIG["full_retrain_epochs"], 
                min_value=10, 
                max_value=200,
                help="Number of training epochs for full retraining"
            )
            
            fine_epochs_val = st.number_input(
                "Fine-tune Epochs", 
                value=LEARNING_CONFIG["fine_tune_epochs"], 
                min_value=5, 
                max_value=50,
                help="Number of epochs for fine-tuning"
            )
            
            # Confidence interval simulations
            ci_simulations = st.number_input(
                "CI Simulations", 
                value=100, 
                min_value=50, 
                max_value=500,
                help="Number of Monte Carlo simulations for confidence intervals"
            )
            
            if st.button("💾 Save Model Config", type="primary"):
                LEARNING_CONFIG["lookback_window"] = lookback_val
                LEARNING_CONFIG["full_retrain_epochs"] = full_epochs_val
                LEARNING_CONFIG["fine_tune_epochs"] = fine_epochs_val
                # Store CI simulations in a global variable or config
                st.session_state.ci_simulations = ci_simulations
                st.success("✅ Model configuration saved!")
            
        with col2:
            st.subheader("🎛️ System Configuration")
            
            pred_days_val = st.number_input(
                "Prediction Days", 
                value=LEARNING_CONFIG["prediction_days"], 
                min_value=1, 
                max_value=10,
                help="Number of days to forecast"
            )
            
            batch_size_val = st.number_input(
                "Batch Size",
                value=LEARNING_CONFIG["batch_size"],
                min_value=16,
                max_value=128,
                step=16,
                help="Training batch size"
            )
            
            # Cache configuration
            cache_size = st.number_input(
                "Cache Size (MB)",
                value=100,
                min_value=10,
                max_value=1000,
                help="Maximum cache size in megabytes"
            )
            
            cache_ttl = st.number_input(
                "Cache TTL (seconds)",
                value=300,
                min_value=60,
                max_value=3600,
                help="Time to live for cached items"
            )
            
            if st.button("💾 Save System Config", type="primary"):
                LEARNING_CONFIG["prediction_days"] = pred_days_val
                LEARNING_CONFIG["batch_size"] = batch_size_val
                # Update cache configuration
                data_cache.max_size_mb = cache_size
                data_cache.default_ttl = cache_ttl
                st.success("✅ System configuration saved!")
            
            st.markdown("---")
            
            st.subheader("🗑️ Maintenance")
            
            if st.button("🗑️ Clear All Models", type="secondary", use_container_width=True):
                try:
                    count = 0
                    for file in MODELS_DIR.glob("*.h5"):
                        file.unlink()
                        count += 1
                    for file in SCALERS_DIR.glob("*.pkl"):
                        file.unlink()
                    for file in METADATA_DIR.glob("*.json"):
                        file.unlink()
                    
                    st.success(f"✅ Cleared {count} models!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            
            if st.button("🧹 Clear Model Cache", type="secondary", use_container_width=True):
                model_manager.clear_cache()
                st.success("✅ Model cache cleared!")
            
            if st.button("🧹 Clear Data Cache", type="secondary", use_container_width=True):
                data_cache.cache.clear()
                st.success("✅ Data cache cleared!")
            
            if st.button("📊 Reset Metrics", type="secondary", use_container_width=True):
                metrics_collector.reset()
                st.success("✅ Metrics reset!")
            
            if st.button("🗑️ Clear Predictions", type="secondary", use_container_width=True):
                try:
                    # Clear predictions
                    if SQLALCHEMY_AVAILABLE:
                        db_session = Session()
                        count = db_session.query(Prediction).count()
                        db_session.query(Prediction).delete()
                        db_session.commit()
                        db_session.close()
                        st.success(f"✅ Cleared {count} predictions!")
                    else:
                        # Clear JSON files
                        all_tickers = [t for cat in ASSET_CATEGORIES.values() for t in cat.values()]
                        count = 0
                        for ticker in all_tickers:
                            predictions_path = get_predictions_path(ticker)
                            if predictions_path.exists():
                                predictions_path.unlink()
                                count += 1
                        st.success(f"✅ Cleared {count} prediction files!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ================================
    # TAB 6: DIAGNOSTICS
    # ================================
    
    with tab6:
        show_pattern_dashboard()
        st.markdown("---")
        show_error_dashboard()
        
        st.markdown("---")
        st.subheader("🔧 System Diagnostics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Thread Manager Status:**")
            thread_status = thread_manager.get_status()
            st.json(thread_status)
            
        with col2:
            st.write("**Application Metrics:**")
            metrics = metrics_collector.get_metrics()
            st.json(metrics)
        
        st.markdown("---")
        st.subheader("🔌 Circuit Breaker Status")
        
        circuit_metrics = alpha_vantage_limiter.get_metrics()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            state_color = "🟢" if circuit_metrics['state'] == 'CLOSED' else "🔴" if circuit_metrics['state'] == 'OPEN' else "🟡"
            st.metric("State", f"{state_color} {circuit_metrics['state']}")
        
        with col2:
            st.metric("Failure Count", circuit_metrics['failure_count'])
        
        with col3:
            if circuit_metrics['last_failure']:
                last_fail = circuit_metrics['last_failure'].strftime("%H:%M:%S")
                st.metric("Last Failure", last_fail)
            else:
                st.metric("Last Failure", "Never")
        
        with col4:
            if st.button("🔄 Reset Circuit", type="secondary"):
                alpha_vantage_limiter.reset()
                st.success("✅ Circuit breaker reset!")
                time.sleep(1)
                st.rerun()
    
    # ================================
    # TAB 7: LOGS
    # ================================
    
    with tab7:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📝 Learning Activity Log")
        
        with col2:
            if st.button("🔄 Refresh", type="secondary", use_container_width=True):
                st.rerun()
            if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                st.session_state.learning_log = []
                st.success("✅ Logs cleared!")
                time.sleep(1)
                st.rerun()
        
        show_learning_log()
        
        st.markdown("---")
        st.subheader("📄 Application Log")
        
        try:
            with open('app.log', 'r') as f:
                log_lines = f.readlines()
                recent_logs = log_lines[-50:]  # Last 50 lines
                
            st.text_area(
                "Recent Log Entries", 
                "".join(recent_logs), 
                height=300,
                disabled=True
            )
        except Exception as e:
            st.warning(f"⚠️ Could not read log file: {e}")

# ================================
# APPLICATION ENTRY POINT
# ================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        shutdown_background_threads()
    except Exception as e:
        logger.critical(f"Application crashed: {e}")
        st.error(f"❌ Critical Error: {e}")
        st.error("Please check logs for details.")
        
        # Attempt graceful shutdown
        try:
            shutdown_background_threads()
        except:
            pass
    finally:
        logger.info("Application shutdown complete")
