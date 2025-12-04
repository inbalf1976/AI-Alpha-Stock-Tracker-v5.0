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
import os
import warnings
import joblib
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

# Load environment variables
load_dotenv()

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# Suppress Streamlit thread warnings
try:
    from streamlit.runtime.scriptrunner import script_run_context
    if hasattr(script_run_context, '_LOGGER'):
        script_run_context._LOGGER.setLevel('ERROR')
except:
    pass

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
    "Stocks": {
        "Apple": "AAPL",
        "Tesla": "TSLA", 
        "Nvidia": "NVDA",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Palantir": "PLTR",
        "MicroStrategy": "MSTR",
        "Coinbase": "COIN"
    },
    "ETFs": {
        "SPDR S&P 500": "SPY",
        "3x Long QQQ": "TQQQ",
        "3x Short QQQ": "SQQQ",
        "3x Long Semis": "SOXL"
    },
    "Commodities": {
        "Crude Oil": "CL=F",
        "Gold": "GC=F",
        "Corn": "ZC=F",
        "Wheat": "ZW=F"
    }
}

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

# Pattern mining watchlist
PATTERN_WATCHLIST = [
    "AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "PLTR", "MSTR", "COIN", 
    "SPY", "CL=F", "GC=F", "ZC=F", "ZW=F", "WEAT",
    "META", "AMZN", "NFLX", "AMD", "SOXL", "TQQQ", "SQQQ"
]

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

class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

def log_error(severity: ErrorSeverity, function: str, error: Exception, 
              ticker: Optional[str] = None, user_message: Optional[str] = None, 
              show_to_user: bool = True) -> None:
    """Enhanced error logging with structured data"""
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity.name,
        "function": function,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "ticker": ticker,
        "user_message": user_message
    }
    
    try:
        if ERROR_LOG.exists():
            with open(ERROR_LOG, 'r') as f:
                errors = json.load(f)
        else:
            errors = []
        
        errors.append(error_entry)
        
        # Keep only last 1000 errors
        if len(errors) > 1000:
            errors = errors[-1000:]
        
        with open(ERROR_LOG, 'w') as f:
            json.dump(errors, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to log error: {e}")
    
    # Also log to standard logger
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
# UTILITY FUNCTIONS
# ================================

def sanitize_ticker(ticker: str) -> str:
    """Sanitize ticker for safe file operations"""
    # Allow only alphanumeric, dash, underscore
    return re.sub(r'[^a-zA-Z0-9_-]', '_', ticker)

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
    """Download historical data from Alpha Vantage API"""
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
           logger.info(f"✅ Successfully downloaded {len(df)} rows from yfinance for {ticker}")
           metrics_collector.increment("data_downloads")
           metrics_collector.increment("yfinance_downloads")
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

def download_with_timeout(ticker: str, period: str = "1y", 
                         interval: str = "1d", 
                         timeout: int = NETWORK_TIMEOUT_SECONDS) -> Optional[pd.DataFrame]:
    """Download data with timeout protection and Alpha Vantage fallback"""
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
                return df
            else:
                logger.warning(f"yfinance returned empty data for {ticker}, trying Alpha Vantage...")
                
    except TimeoutError:
        logger.error(f"Timeout downloading data from yfinance for {ticker}, trying Alpha Vantage...")
    except Exception as e:
        logger.error(f"Error downloading from yfinance for {ticker}: {e}, trying Alpha Vantage...")
    
    # Fallback to Alpha Vantage
    logger.info(f"🔄 Falling back to Alpha Vantage for {ticker}...")
    
    # Determine outputsize based on period
    if period in ["1mo", "3mo"]:
        outputsize = "compact"  # Last 100 days
    else:
        outputsize = "full"  # 20+ years
    
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
        
        return av_df
    
    logger.error(f"❌ All data sources failed for {ticker}")
    return None

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
            "telegram_alerts": bool
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

class ModelManager:
    """Manage TensorFlow models and memory with caching"""
    def __init__(self):
        self._models: Dict[str, Tuple[Any, float]] = {}  # ticker -> (model, timestamp)
        self._lock = threading.RLock()
        self._cache_ttl = CACHE_TTL_SECONDS
        
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
            tf.keras.backend.clear_session()
            # Force garbage collection
            gc.collect()
        except Exception as e:
            logger.warning(f"Error clearing model: {e}")
    
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
model_manager = ModelManager()

# ================================
# CIRCUIT BREAKER PATTERN
# ================================

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
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
    
    def reset(self) -> None:
        """Manually reset circuit breaker"""
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure = None
            logger.info("Circuit breaker manually reset")

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
            "elite_patterns_found": 0
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
# REAL PREDICTION VALIDATION
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
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=15)
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
    path = get_predictions_path(ticker)
    try:
        if path.exists():
            with open(path, 'r') as f:
                predictions = json.load(f)
        else:
            predictions = []
        
        prediction_record = {
            "date": date,
            "prediction": float(prediction),
            "timestamp": datetime.now().isoformat(),
            "previous_price": float(current_price) if current_price else None,
            "validated": False
        }
        
        predictions.append(prediction_record)
        
        # Keep only last 100 predictions
        if len(predictions) > 100:
            predictions = predictions[-100:]
        
        with open(path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        metrics_collector.increment("predictions_made")
        logger.info(f"Recorded prediction for {ticker}: ${prediction:.2f} on {date}")
        return True
            
    except Exception as e:
        logger.error(f"Failed to record prediction for {ticker}: {e}")
        log_error(ErrorSeverity.WARNING, "record_prediction", e, ticker=ticker, show_to_user=False)
        return False

def load_accuracy_log(ticker: str) -> Dict[str, Any]:
    """Load accuracy tracking data with REAL validation against actual prices"""
    path = get_predictions_path(ticker)
    if path.exists():
        try:
            with open(path, 'r') as f:
                predictions = json.load(f)
            
            if len(predictions) == 0:
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
                    pred_date = datetime.fromisoformat(pred['date'])
                    pred_value = float(pred['prediction'])
                    
                    # Only validate predictions that are in the past
                    if pred_date.date() >= datetime.now().date():
                        continue
                    
                    # Fetch actual price for that date
                    actual_price = get_actual_price_for_date(ticker, pred_date)
                    
                    if actual_price is None:
                        continue
                    
                    validated_count += 1
                    
                    # Calculate errors
                    mae = abs(pred_value - actual_price)
                    mape = (mae / actual_price) * 100  # Percentage error
                    
                    errors_mae.append(mae)
                    errors_mape.append(mape)
                    
                    # Directional accuracy (did we predict up/down correctly?)
                    if 'previous_price' in pred:
                        prev_price = float(pred['previous_price'])
                        predicted_direction = 1 if pred_value > prev_price else -1
                        actual_direction = 1 if actual_price > prev_price else -1
                        directional_correct.append(predicted_direction == actual_direction)
                    elif current_price:
                        # Use current price as reference if no previous stored
                        predicted_direction = 1 if pred_value > current_price else -1
                        actual_direction = 1 if actual_price > current_price else -1
                        directional_correct.append(predicted_direction == actual_direction)
                    
                    # Store validation in prediction record
                    pred['actual_price'] = actual_price
                    pred['error_mae'] = mae
                    pred['error_mape'] = mape
                    pred['validated_at'] = datetime.now().isoformat()
                    
                except Exception as e:
                    logger.debug(f"Error validating prediction for {ticker}: {e}")
                    continue
            
            # Save updated predictions with validation data
            with open(path, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            # Calculate aggregate metrics
            if validated_count == 0:
                return {
                    "total_predictions": len(predictions),
                    "validated_predictions": 0,
                    "avg_error_mape": 0.0,
                    "avg_error_mae": 0.0,
                    "directional_accuracy": 0.0,
                    "last_updated": datetime.now().isoformat(),
                    "status": "no_validated"
                }
            
            avg_mape = np.mean(errors_mape)
            avg_mae = np.mean(errors_mae)
            dir_accuracy = (sum(directional_correct) / len(directional_correct) * 100) if directional_correct else 0.0
            
            return {
                "total_predictions": len(predictions),
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
    
    # Check if we have validated predictions
    validated_count = accuracy_log.get("validated_predictions", 0)
    total_predictions = accuracy_log.get("total_predictions", 0)
    
    if validated_count < 5:
        if total_predictions < 5:
            reasons.append("insufficient_predictions")
        else:
            reasons.append("insufficient_validated_predictions")
    
    # Check REAL error rates (MAPE - Mean Absolute Percentage Error)
    avg_mape = accuracy_log.get("avg_error_mape", 0.0)
    if validated_count >= 5:  # Only check error if we have enough validated predictions
        if avg_mape > 8.0:  # More than 8% average error
            reasons.append(f"high_error_mape_{avg_mape:.1f}%")
        elif avg_mape > 5.0:  # Warning level
            reasons.append(f"elevated_error_mape_{avg_mape:.1f}%")
    
    # Check directional accuracy
    dir_accuracy = accuracy_log.get("directional_accuracy", 0.0)
    if validated_count >= 5 and dir_accuracy < 55.0:  # Less than 55% correct direction
        reasons.append(f"poor_direction_accuracy_{dir_accuracy:.1f}%")
    
    # Check initial training
    retrain_count = metadata.get("retrain_count", 0)
    if retrain_count < 2:
        reasons.append("initial_training_phase")
    
    # Check if model is stale
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
# PRICE DATA FUNCTIONS
# ================================

def get_latest_price(ticker: str) -> Optional[float]:
    """Get latest price for a ticker with retry logic and Alpha Vantage fallback"""
    max_retries = MAX_RETRIES
    retry_delay = 1
    
    # Try yfinance first
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            
            if not hist.empty:
                metrics_collector.increment("data_downloads")
                price = float(hist['Close'].iloc[-1])
                logger.debug(f"Got latest price from yfinance for {ticker}: ${price:.2f}")
                return price
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    # Fallback to Alpha Vantage
    logger.info(f"🔄 Falling back to Alpha Vantage for latest price of {ticker}...")
    
    if not ALPHA_VANTAGE_API_KEY:
        logger.error(f"Failed to get price for {ticker} - No Alpha Vantage API key")
        metrics_collector.increment("errors_encountered")
        return None
    
    try:
        av_ticker = ticker.replace('=F', '').replace('^', '')
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': av_ticker,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Global Quote' in data and '05. price' in data['Global Quote']:
            price = float(data['Global Quote']['05. price'])
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
    """Train or fine-tune model"""
    model = None
    model_path = get_model_path(ticker)
    
    try:
        if force_retrain or not model_path.exists():
            # Build new model
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
            # Load and fine-tune existing model
            logger.info(f"Fine-tuning existing model for {ticker}")
            model = model_manager.load_model(model_path, use_cache=False)
            
            if model is None:
                raise ValueError("Failed to load existing model")
            
            # Fine-tune with recent data
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
# FORECAST GENERATION
# ================================

def generate_forecast(ticker: str, model: Any, scaler: MinMaxScaler,
                     scaled_data: np.ndarray, lookback: int = 60,
                     days: int = 5) -> Tuple[Optional[np.ndarray], Optional[List]]:
    """Generate price forecast"""
    try:
        # Prepare last sequence
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
        
        # Generate predictions
        predictions = []
        current_sequence = last_sequence.copy()
        
        start_time = time.time()
        for _ in range(days):
            pred = model_manager.predict_with_cleanup(model, current_sequence)
            if pred is None:
                logger.error(f"Prediction failed for {ticker}")
                return None, None
            
            predictions.append(pred[0, 0])
            # Update sequence for next prediction
            current_sequence = np.append(
                current_sequence[:, 1:, :],
                pred.reshape(1, 1, 1),
                axis=1
            )
        
        prediction_time = time.time() - start_time
        metrics_collector.record_time("avg_prediction_time", prediction_time)
        
        # Inverse transform predictions
        forecast = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        # Generate business days
        dates = []
        day_offset = 1
        while len(dates) < days:
            next_date = datetime.now().date() + timedelta(days=day_offset)
            if next_date.weekday() < 5:  # Monday-Friday
                dates.append(next_date)
            day_offset += 1
        
        logger.info(f"Generated {days}-day forecast for {ticker}")
        return forecast, dates
        
    except Exception as e:
        logger.error(f"Forecast generation failed for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "generate_forecast", e, 
                 ticker=ticker, show_to_user=False)
        return None, None

# Continue to Part 3...

# ================================
# ENHANCED TRAINING FUNCTION
# ================================

def train_self_learning_model_enhanced(ticker: str, days: int = 5, 
                                       force_retrain: bool = False) -> Tuple[
                                           Optional[np.ndarray],
                                           Optional[List],
                                           Optional[Any]
                                       ]:
    """Enhanced training with proper memory management and all fixes"""
    logger.info(f"Training {ticker} (force={force_retrain})")
    
    model = None
    try:
        # Validate predictions first
        updated, acc_log = validate_predictions(ticker)
        meta = load_metadata(ticker)
        needs_retrain, reasons = should_retrain(ticker, acc_log, meta)
        
        if needs_retrain or force_retrain:
            logger.info(f"Retraining {ticker}: {', '.join(reasons)}")
        
        # Download and validate data
        df = download_and_validate_data(ticker, period="1y")
        if df is None:
            logger.warning(f"Failed to download data for {ticker}")
            return None, None, None
        
        # Load or create scaler
        scaler = load_or_create_scaler(ticker, df, force_create=force_retrain or needs_retrain)
        if scaler is None:
            logger.warning(f"Failed to create scaler for {ticker}")
            return None, None, None
        
        # Prepare training data
        lookback = LEARNING_CONFIG["lookback_window"]
        X, y = prepare_training_data(df, scaler, lookback=lookback)
        
        if X is None or y is None:
            logger.warning(f"Failed to prepare training data for {ticker}")
            return None, None, None
        
        # Train or fine-tune model
        if force_retrain or needs_retrain:
            model = train_model(ticker, X, y, force_retrain=force_retrain, metadata=meta)
            if model is None:
                logger.warning(f"Training failed for {ticker}")
                return None, None, None
        else:
            # Load existing model
            model_path = get_model_path(ticker)
            model = model_manager.load_model(model_path)
            if model is None:
                logger.warning(f"Failed to load model for {ticker}, retraining")
                model = train_model(ticker, X, y, force_retrain=True, metadata=meta)
                if model is None:
                    return None, None, None
        
        # Generate forecast
        scaled_data = scaler.transform(df[['Close']])
        forecast, dates = generate_forecast(ticker, model, scaler, scaled_data, 
                                           lookback=lookback, days=days)
        
        if forecast is None or dates is None:
            logger.warning(f"Forecast generation failed for {ticker}")
            return None, None, None
        
        # Apply pattern boosts
        current_price = get_latest_price(ticker)
        if current_price:
            forecast = get_pattern_boosted_forecast(ticker, forecast.tolist(), current_price)
        
        # Record prediction with current price for directional validation
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        record_prediction(ticker, forecast[0], tomorrow, current_price)
        
        # Update metadata
        meta.update({
            "trained_date": datetime.now().isoformat(),
            "training_samples": len(X),
            "training_volatility": float(df['Close'].pct_change().std()),
            "version": meta.get("version", 1) + 1,
            "last_accuracy": acc_log.get("avg_error_mape", 0),
            "data_quality": "GOOD",
            "forecast_days": days
        })
        save_metadata(ticker, meta)
        
        logger.info(f"Successfully trained {ticker}, forecast: {forecast[0]:.2f}")
        return forecast, dates, model
        
    except Exception as e:
        logger.error(f"Training failed for {ticker}: {e}")
        log_error(ErrorSeverity.ERROR, "train_self_learning_model_enhanced", e, 
                 ticker=ticker, user_message=f"Training failed for {ticker}", 
                 show_to_user=False)
        metrics_collector.increment("errors_encountered")
        return None, None, None
        
    finally:
        # Always clear TensorFlow session
        if model is not None:
            del model
        tf.keras.backend.clear_session()
        gc.collect()

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
    """Load monitoring configuration"""
    default_config = {
        "enabled": False,
        "check_interval_minutes": 5,
        "telegram_alerts": False
    }
    return ConfigManager.load_config_with_backup(
        MONITORING_CONFIG, "monitoring", default_config
    )

def save_monitoring_config(enabled: bool) -> bool:
    """Save monitoring configuration"""
    try:
        config = load_monitoring_config()
        config["enabled"] = enabled
        MONITORING_CONFIG.parent.mkdir(exist_ok=True)
        with open(MONITORING_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
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

# Continuing with more functions...

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

def run_pattern_mining_cycle() -> int:
    """Run one complete pattern mining cycle"""
    logger.info(f"\n{'='*80}")
    logger.info(f"HYBRID AUTO-PATTERN MINER | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*80}")
    
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
        raw = json.loads(AUTO_PATTERNS_FILE.read_text(encoding="utf-8"))
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
# THREAD HEARTBEAT AND MONITORING
# ================================

class ApplicationState:
    """Encapsulate application state"""
    def __init__(self):
        self.thread_heartbeats: Dict[str, Optional[datetime]] = {
            "learning_daemon": None,
            "monitoring": None,
            "pattern_miner": None,
            "watchdog": None,
            "auto_validator": None
        }
        self.thread_start_times: Dict[str, Optional[datetime]] = {
            "learning_daemon": None,
            "monitoring": None,
            "pattern_miner": None,
            "watchdog": None,
            "auto_validator": None
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
        self.logging_queue.put(message)
    
    def get_log_messages(self) -> List[str]:
        """Get all pending log messages"""
        messages = []
        while not self.logging_queue.empty():
            try:
                messages.append(self.logging_queue.get_nowait())
            except queue.Empty:
                break
        return messages

# Global application state
app_state = ApplicationState()

# Continue to Part 5 with background threads...

# ================================
# BACKGROUND THREADS
# ================================

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
                        
                        forecast, dates, model = train_self_learning_model_enhanced(ticker, days=5)
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

def monitor_6percent_pre_move_managed(stop_event: threading.Event) -> None:
    """Enhanced monitoring with resource management"""
    app_state.update_heartbeat("monitoring")
    app_state.set_thread_start_time("monitoring")
    logger.info("[MONITORING] Enhanced 6%+ Monitor STARTED")
    
    while not stop_event.is_set():
        try:
            if not load_monitoring_config().get("enabled", False):
                logger.info("[MONITORING] Monitoring paused (disabled in config)")
                time.sleep(30)
                app_state.update_heartbeat("monitoring")
                continue
            
            app_state.update_heartbeat("monitoring")
            
            # Monitoring logic placeholder
            # In a real implementation, this would check for significant price movements
            # and send alerts via Telegram or other channels
            
            # Sleep with interrupt checking
            config = load_monitoring_config()
            sleep_minutes = config.get("check_interval_minutes", 5)
            sleep_seconds = sleep_minutes * 60
            
            for _ in range(sleep_seconds):
                if stop_event.is_set():
                    break
                time.sleep(1)
                app_state.update_heartbeat("monitoring")
                
        except Exception as e:
            logger.error(f"[CRITICAL] Monitoring error: {e}")
            log_error(ErrorSeverity.ERROR, "monitor_6percent_pre_move", e,
                     user_message="Monitoring error - will retry", show_to_user=False)
            
            for _ in range(60):
                if stop_event.is_set():
                    break
                time.sleep(1)
    
    logger.info("[MONITORING] Enhanced 6%+ Monitor STOPPED")

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
    try:
        import psutil
        
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
        
    except ImportError:
        logger.warning("psutil not available for resource monitoring")
        return None
    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")
        return None

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
        
        response = requests.post(url, json=payload, timeout=10)
        
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

# Continue to Part 6 with UI components...

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
    
    # Resource monitoring
    st.markdown("#### 💻 System Resources")
    resources = monitor_system_resources()
    if resources:
        col1, col2, col3 = st.columns(3)
        with col1:
            mem_color = "normal" if resources['memory_mb'] < MEMORY_WARNING_THRESHOLD_MB else "off"
            st.metric("Memory", f"{resources['memory_mb']:.1f}MB", delta_color=mem_color)
        with col2:
            cpu_color = "normal" if resources['cpu_percent'] < CPU_WARNING_THRESHOLD_PERCENT else "off"
            st.metric("CPU", f"{resources['cpu_percent']:.1f}%", delta_color=cpu_color)
        with col3:
            disk_color = "normal" if resources['disk_usage'] < DISK_WARNING_THRESHOLD_PERCENT else "off"
            st.metric("Disk", f"{resources['disk_usage']:.1f}%", delta_color=disk_color)
    
# Metrics display
st.markdown("#### 📈 Application Metrics")
metrics = metrics_collector.get_metrics()

col1, col2 = st.columns(2)
with col1:
    st.metric("Predictions Made", metrics.get('predictions_made', 0))
    st.metric("Models Trained", metrics.get('models_trained', 0))
    st.metric("Elite Patterns", metrics.get('elite_patterns_found', 0))
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
    path = get_predictions_path(ticker)
    
    if not path.exists():
        st.info("No prediction history available.")
        return
    
    try:
        with open(path, 'r') as f:
            predictions = json.load(f)
        
        if not predictions:
            st.info("No predictions recorded.")
            return
        
        # Filter validated predictions
        validated_preds = [p for p in predictions if p.get('actual_price') is not None]
        
        if not validated_preds:
            st.info("No validated predictions yet. Predictions can only be validated after their target date has passed.")
            return
        
        # Create comparison table
        history_data = []
        for pred in validated_preds:
            try:
                pred_date = datetime.fromisoformat(pred['date']).strftime("%Y-%m-%d")
                predicted = pred['prediction']
                actual = pred['actual_price']
                error_pct = pred.get('error_mape', 0)
                error_abs = pred.get('error_mae', 0)
                
                # Direction
                if pred.get('previous_price'):
                    prev = pred['previous_price']
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
                    date = datetime.fromisoformat(pred['date'])
                    chart_data.append({
                        'Date': date,
                        'Predicted': pred['prediction'],
                        'Actual': pred['actual_price']
                    })
                except:
                    continue
            
            if chart_data:
                df_chart = pd.DataFrame(chart_data).sort_values('Date')
                
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
                
                st.plotly_chart(fig, use_container_width=True)
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
        log_error(ErrorSeverity.ERROR, "show_error_dashboard", e, show_to_user=True)

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

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Alpha Trader v4.2 - 100% Real",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 AI Alpha Trader v4.2 - 100% Real Validation")
    st.markdown("*Advanced AI-Powered Trading Platform with Real Prediction Validation*")
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
            st.metric("💵 Current Price", f"${price:.2f}")
        else:
            st.warning("⚠️ Price unavailable")
        
        st.markdown("---")
        st.subheader("🔧 Quick Actions")
        
        if st.button("🔄 Force Retrain", type="secondary", use_container_width=True):
            with st.spinner("Retraining model..."):
                forecast, dates, model = train_self_learning_model_enhanced(ticker, force_retrain=True)
                if forecast is not None:
                    st.success("✅ Model retrained!")
                else:
                    st.error("❌ Retraining failed")
                time.sleep(2)
                st.rerun()
        
        if st.button("🚀 Bootstrap All Models", type="secondary", use_container_width=True):
            with st.spinner("Training all models... (5-10 min)"):
                all_tickers = [t for cat in ASSET_CATEGORIES.values() for _, t in cat.items()]
                progress = st.progress(0)
                success_count = 0
                
                for idx, t in enumerate(all_tickers):
                    try:
                        forecast, dates, model = train_self_learning_model_enhanced(t, days=5, force_retrain=True)
                        if forecast is not None:
                            success_count += 1
                    except Exception as e:
                        logger.error(f"Failed to train {t}: {e}")
                    progress.progress((idx + 1) / len(all_tickers))
                
                st.success(f"✅ Trained {success_count}/{len(all_tickers)} models!")
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
            success = send_telegram_alert("🧪 TEST ALERT\n<b>AI Alpha Trader v4.2</b>\nSystem is operational!")
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
                with st.spinner("🔍 Analyzing..."):
                    forecast, _, _ = train_self_learning_model_enhanced(ticker, days=1)
                    
                    if forecast is not None and len(np.array(forecast).flatten()) > 0:
                        forecast_val = float(np.array(forecast).flatten()[0])
                        
                        # Use enhanced confidence check with patterns
                        passed, reasons, pattern_boost = enhanced_confidence_checklist(
                            ticker, [forecast_val], price or 100
                        )
                        
                        # Get pattern-influenced recommendation
                        action, confidence, pattern_reasons = get_pattern_influenced_recommendation(
                            ticker, [forecast_val], price
                        )
                        
                        if passed:
                            change_pct = (forecast_val - price) / price * 100 if price else 0
                            
                            # Display with pattern influence
                            if "BUY" in action:
                                st.success(f"**{action}** | Confidence: {confidence}%")
                            elif "SELL" in action:
                                st.error(f"**{action}** | Confidence: {confidence}%")
                            else:
                                st.info(f"**{action}** | Confidence: {confidence}%")
                            
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
            
            # Pattern status
            boost, triggers, direction, confidence = check_auto_patterns(ticker)
            if boost > 0:
                st.metric("⚡ Pattern Boost", f"+{boost}", f"Direction: {direction}")
                if triggers:
                    with st.expander("🔍 Pattern Details"):
                        for trigger in triggers:
                            st.write(f"• {trigger}")
            
            # Button to view detailed accuracy
            if st.button("📊 View Accuracy Details", key=f"acc_detail_{ticker}"):
                st.session_state.show_accuracy_detail = True
                st.rerun()
    
    # ================================
    # TAB 2: FORECAST
    # ================================
    
    with tab2:
        st.header("🔮 Price Forecast")
        
        days_to_forecast = st.slider("Days to Forecast", 1, 10, 5)
        
        if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("🔍 Generating forecast..."):
                forecast, dates, model = train_self_learning_model_enhanced(ticker, days=days_to_forecast)
                
                if forecast is not None and len(forecast) > 0:
                    # Create forecast chart
                    current_price = price or get_latest_price(ticker)
                    
                    if current_price:
                        # Prepare data for plotting
                        forecast_dates = dates[:len(forecast)]
                        forecast_prices = forecast
                        
                        # Create plot
                        fig = go.Figure()
                        
                        # Current price
                        fig.add_trace(go.Scatter(
                            x=[datetime.now().date()],
                            y=[current_price],
                            mode='markers',
                            name='Current Price',
                            marker=dict(size=12, color='green')
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_prices,
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='blue', width=2),
                            marker=dict(size=8)
                        ))
                        
                        fig.update_layout(
                            title=f"{ticker} {days_to_forecast}-Day Price Forecast",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            showlegend=True,
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast table
                        st.markdown("---")
                        st.subheader("📋 Forecast Details")
                        forecast_data = []
                        for i, (date, price_val) in enumerate(zip(forecast_dates, forecast_prices)):
                            change_pct = (price_val - current_price) / current_price * 100
                            change_abs = price_val - current_price
                            
                            forecast_data.append({
                                "Day": i + 1,
                                "Date": date.strftime("%Y-%m-%d"),
                                "Price": f"${price_val:.2f}",
                                "Change ($)": f"${change_abs:+.2f}",
                                "Change (%)": f"{change_pct:+.2f}%"
                            })
                        
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
                        
                    else:
                        st.error("❌ Could not get current price")
                else:
                    st.error("❌ Forecast generation failed")
    
    # ================================
    # TAB 3: ANALYSIS
    # ================================
    
    with tab3:
        st.header("📈 Technical Analysis")
        
        st.info("ℹ️ Advanced analysis features coming soon...")
        
        # Placeholder for future features
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
                    count = 0
                    for file in PREDICTIONS_DIR.glob("*.json"):
                        file.unlink()
                        count += 1
                    st.success(f"✅ Cleared {count} files!")
                    time.sleep(1)
                    st.rerun()
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
                    st.plotly_chart(fig_mape, use_container_width=True)
                
                with col2:
                    # Performance categories
                    excellent = len(df_overview[df_overview['MAPE'] < 3])
                    good = len(df_overview[(df_overview['MAPE'] >= 3) & (df_overview['MAPE'] < 5)])
                    moderate = len(df_overview[(df_overview['MAPE'] >= 5) & (df_overview['MAPE'] < 8)])
                    poor = len(df_overview[df_overview['MAPE'] >= 8])
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Excellent (<3%)', 'Good (3-5%)', 'Moderate (5-8%)', 'Poor (>8%)'],
                        values=[excellent, good, moderate, poor],
                        marker_colors=['darkgreen', 'lightgreen', 'orange', 'red']
                    )])
                    fig_pie.update_layout(
                        title="Performance Categories",
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
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
                            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
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
            
            if st.button("💾 Save Model Config", type="primary"):
                LEARNING_CONFIG["lookback_window"] = lookback_val
                LEARNING_CONFIG["full_retrain_epochs"] = full_epochs_val
                LEARNING_CONFIG["fine_tune_epochs"] = fine_epochs_val
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
            
            if st.button("💾 Save System Config", type="primary"):
                LEARNING_CONFIG["prediction_days"] = pred_days_val
                LEARNING_CONFIG["batch_size"] = batch_size_val
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
            
            if st.button("📊 Reset Metrics", type="secondary", use_container_width=True):
                metrics_collector.reset()
                st.success("✅ Metrics reset!")
            
            if st.button("🗑️ Clear Predictions", type="secondary", use_container_width=True):
                try:
                    count = 0
                    for file in PREDICTIONS_DIR.glob("*.json"):
                        file.unlink()
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
        log_error(ErrorSeverity.CRITICAL, "main", e, show_to_user=True)
        st.error(f"❌ Critical Error: {e}")
        st.error("Please check logs for details.")
        
        # Attempt graceful shutdown
        try:
            shutdown_background_threads()
        except:
            pass
    finally:
        logger.info("Application shutdown complete")
