"""
WHEAT TRADING DASHBOARD - Enhanced Streamlit Version
With Telegram alerts, 24/7 monitoring, and smart alert system
"""

# First, set page config (must be first Streamlit command)
import streamlit as st
st.set_page_config(page_title="Wheat Trading Dashboard", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads your .env file automatically!
except ImportError:
    st.warning("python-dotenv not installed. Install with: pip install python-dotenv")
    st.stop()

try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError as e:
    KERAS_AVAILABLE = False
    st.error(f"⚠️ TensorFlow/Keras not installed: {e}")
    st.info("Install with: `pip install tensorflow keras`")
    st.stop()
except Exception as e:
    KERAS_AVAILABLE = False
    st.error(f"⚠️ TensorFlow error: {e}")
    st.info("Try reinstalling: `pip uninstall tensorflow keras` then `pip install tensorflow keras`")
    st.stop()

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    st.warning("requests not installed. Telegram alerts disabled. Install with: pip install requests")

# Alpha Vantage API
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Trading configuration - ENHANCED v2.0"""
    PRIMARY_TICKER = "ZW=F"  # Wheat Futures (in cents per bushel)
    TICKER_TYPE = "FUTURES"  # or "ETF"
    CORRELATED_TICKERS = ["CORN", "ZC=F", "TAGS", "DBA", "GLD", "DBC"]
    
    # Data source settings
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    
    LOOKBACK_DAYS = 730
    LSTM_SEQUENCE_LENGTH = 60
    LSTM_UNITS = 64
    LSTM_DROPOUT = 0.2
    
    # ========== REALISTIC RISK MANAGEMENT (v2.0) ==========
    MAX_POSITION_SIZE = 0.10
    STOP_LOSS_PCT = 0.015      # 1.5% stop - REALISTIC for wheat!
    TAKE_PROFIT_PCT = 0.025    # 2.5% target - ACHIEVABLE!
    MIN_CONFIDENCE = 0.55      # Lowered from 0.60
    MIN_CORRELATION_AGREEMENT = 3
    
    # ========== HISTORICAL PATTERNS ==========
    HISTORICAL_FLOOR = 468
    HISTORICAL_CEILING = 1425
    NORMAL_RANGE_LOW = 480
    NORMAL_RANGE_HIGH = 620
    CURRENT_EQUILIBRIUM = 550
    CRISIS_THRESHOLD = 800
    
    # ========== SEASONAL BIAS ==========
    SEASONAL_BIAS = {
        1: 0.00, 2: -0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: -0.05,
        7: -0.03, 8: -0.02, 9: 0.02, 10: 0.03, 11: 0.04, 12: 0.05
    }
    
    HIGH_VOLATILITY_MONTHS = [3, 4, 5]
    LOW_VOLATILITY_MONTHS = [7, 8]
    
    # WASDE thresholds
    WASDE_STOCKS_TIGHT = 0.15
    WASDE_STOCKS_AMPLE = 0.25
    
    # Alert settings
    DIRECTION_CHANGE_THRESHOLD = 0.025
    CHECK_INTERVAL = 300
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

config = Config()

# ============================================================================
# SEASONAL PATTERN ANALYZER (NEW in v2.0)
# ============================================================================

class SeasonalAnalyzer:
    """Analyze seasonal patterns"""
    
    @staticmethod
    def get_current_bias():
        """Get seasonal bias for current month"""
        month = datetime.now().month
        bias = config.SEASONAL_BIAS.get(month, 0.0)
        
        is_high_vol = month in config.HIGH_VOLATILITY_MONTHS
        is_low_vol = month in config.LOW_VOLATILITY_MONTHS
        
        explanations = {
            1: "January - Neutral baseline",
            2: "February - Pre-spring weakness",
            3: "March - Spring rally begins",
            4: "April - Peak planting season",
            5: "May - Maximum spring premium",
            6: "June - Harvest pressure (LOWEST month)",
            7: "July - Post-harvest weakness",
            8: "August - Summer doldrums",
            9: "September - Fall recovery",
            10: "October - Winter demand builds",
            11: "November - Pre-winter strength",
            12: "December - Winter premium (HIGHEST month)"
        }
        
        return {
            'bias': bias,
            'direction': 'BULLISH' if bias > 0.02 else 'BEARISH' if bias < -0.02 else 'NEUTRAL',
            'volatility': 'HIGH' if is_high_vol else 'LOW' if is_low_vol else 'NORMAL',
            'explanation': explanations.get(month, ""),
            'month_name': datetime.now().strftime('%B')
        }

# ============================================================================
# WASDE ANALYZER (NEW in v2.0)
# ============================================================================

class WASDEAnalyzer:
    """Analyze USDA WASDE supply/demand data"""
    
    @staticmethod
    def get_fundamentals():
        """Get current wheat fundamentals (simulated for now)"""
        # In production, this would fetch from USDA API
        # For now, using realistic estimates based on current market
        
        return {
            'stocks_to_use': 0.18,  # 18% = TIGHT supply (bullish)
            'production_change': -1.2,  # Production down 1.2% (bullish)
            'demand_change': 0.5,  # Demand up 0.5% (bullish)
            'signal': 'BULLISH',
            'score': 0.25  # Positive = bullish
        }

# ============================================================================
# MARKET CONTEXT ANALYZER (NEW in v2.0)
# ============================================================================

class MarketContext:
    """Analyze current market vs historical patterns"""
    
    @staticmethod
    def analyze_price(current_price):
        """Analyze where price sits in historical context"""
        context = {}
        
        # Position in normal range
        if current_price < config.NORMAL_RANGE_LOW:
            context['position'] = 'BELOW_NORMAL'
            context['signal'] = 'BUY'
            context['reason'] = f'Price {current_price:.0f}¢ below normal floor {config.NORMAL_RANGE_LOW}¢'
        elif current_price > config.NORMAL_RANGE_HIGH:
            context['position'] = 'ABOVE_NORMAL'
            context['signal'] = 'SELL'
            context['reason'] = f'Price {current_price:.0f}¢ above normal ceiling {config.NORMAL_RANGE_HIGH}¢'
        else:
            context['position'] = 'NORMAL'
            context['signal'] = 'NEUTRAL'
            context['reason'] = f'Price {current_price:.0f}¢ in normal range'
        
        # Crisis detection
        if current_price > config.CRISIS_THRESHOLD:
            context['crisis'] = True
            context['warning'] = 'CRISIS MODE - Geopolitical/supply shock likely'
        else:
            context['crisis'] = False
            context['warning'] = None
        
        # Distance from equilibrium
        context['distance_from_eq'] = ((current_price - config.CURRENT_EQUILIBRIUM) / 
                                      config.CURRENT_EQUILIBRIUM)
        
        return context

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

STATE_FILE = Path("wheat_state.json")

def load_state():
    """Load previous state"""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'last_direction': None,
        'last_price': None,
        'last_check': None,
        'alerts_sent': 0
    }

def save_state(state):
    """Save current state"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# ============================================================================
# TELEGRAM INTEGRATION
# ============================================================================

def send_telegram_message(message, bot_token, chat_id):
    """Send message via Telegram"""
    if not REQUESTS_AVAILABLE or not bot_token or not chat_id:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def format_telegram_alert(signal_data):
    """Format signal data for Telegram"""
    direction = signal_data['direction']
    confidence = signal_data['confidence']
    price = signal_data['price']
    
    icon = "🟢" if direction == "UP" else "🔴"
    
    message = f"""
🌾 *WHEAT TRADING ALERT* 🌾

{icon} *Signal:* {direction}
📊 *Confidence:* {confidence:.1%}
💰 *Current Price:* ${price:.2f}
🕐 *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    if signal_data.get('is_supported'):
        message += f"✅ *Correlations:* {signal_data['agreements']} assets agree\n"
    else:
        message += f"⚠️ *Correlations:* Only {signal_data['agreements']} agree (need 3+)\n"
    
    if signal_data.get('action') == 'BUY':
        message += f"""
*RECOMMENDATION: BUY*

📈 Entry: ${signal_data['entry']:.2f}
🛑 Stop Loss: ${signal_data['stop_loss']:.2f} (-{config.STOP_LOSS_PCT:.1%})
🎯 Take Profit: ${signal_data['take_profit']:.2f} (+{config.TAKE_PROFIT_PCT:.1%})
📦 Position: {signal_data['shares']} shares (${signal_data['position_value']:.2f})

⚖️ Risk/Reward: 2:1
"""
    else:
        message += f"\n*RECOMMENDATION: HOLD*\n"
        if signal_data.get('reason'):
            message += f"_{signal_data['reason']}_\n"
    
    return message

# ============================================================================
# ALPHA VANTAGE INTEGRATION
# ============================================================================

def fetch_alpha_vantage_intraday(symbol, api_key, interval='5min'):
    """Fetch intraday data from Alpha Vantage"""
    try:
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': api_key,
            'outputsize': 'full'
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if 'Time Series (5min)' in data:
            time_series = data[f'Time Series ({interval})']
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            return df
        elif 'Note' in data:
            st.warning(f"Alpha Vantage API limit reached: {data['Note']}")
            return None
        else:
            st.error(f"Alpha Vantage error: {data.get('Error Message', 'Unknown error')}")
            return None
            
    except Exception as e:
        st.error(f"Alpha Vantage fetch failed: {e}")
        return None

def fetch_alpha_vantage_daily(symbol, api_key):
    """Fetch daily data from Alpha Vantage"""
    try:
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': api_key,
            'outputsize': 'full'
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            return df
        elif 'Note' in data:
            st.warning(f"Alpha Vantage API limit reached: {data['Note']}")
            return None
        else:
            st.error(f"Alpha Vantage error: {data.get('Error Message', 'Unknown error')}")
            return None
            
    except Exception as e:
        st.error(f"Alpha Vantage fetch failed: {e}")
        return None

def get_alpha_vantage_quote(symbol, api_key):
    """Get current quote from Alpha Vantage"""
    try:
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': api_key
        }
        
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        data = response.json()
        
        if 'Global Quote' in data:
            quote = data['Global Quote']
            
            current_price = float(quote.get('05. price', 0))
            open_price = float(quote.get('02. open', 0))
            
            return current_price, open_price
        else:
            return None, None
            
    except Exception as e:
        print(f"Alpha Vantage quote error: {e}")
        return None, None

# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=300)  # 5 minute cache
def fetch_data(ticker, days=1000):  # Increased from 730 to 1000 to ensure we get 200+ trading days
    """Fetch historical data - Yahoo Finance first, Alpha Vantage as fallback"""
    
    # Try Yahoo Finance first (faster and free)
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)
        
        if not df.empty and len(df) > 100:
            # Success with Yahoo
            return df
    except Exception as e:
        print(f"Yahoo Finance failed: {e}")
    
    # Fallback to Alpha Vantage if Yahoo failed and API key is available
    if config.ALPHA_VANTAGE_API_KEY:
        try:
            st.warning("⚠️ Yahoo Finance data unavailable, using Alpha Vantage fallback...")
            
            # Alpha Vantage uses different symbols
            av_symbol = ticker.replace("=F", "")  # Remove =F suffix for futures
            
            df = fetch_alpha_vantage_daily(av_symbol, config.ALPHA_VANTAGE_API_KEY)
            
            if df is not None and len(df) > 0:
                # Filter to requested timeframe
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                df = df[df.index >= start_date]
                
                st.success(f"✅ Loaded {len(df)} days from Alpha Vantage (fallback)")
                return df
        except Exception as e:
            print(f"Alpha Vantage fallback also failed: {e}")
    
    # Both failed
    st.error("❌ Unable to fetch data from Yahoo Finance or Alpha Vantage")
    return None

def get_latest_price(ticker):
    """Get latest price - Yahoo first, Alpha Vantage as automatic fallback"""
    
    # Try Yahoo Finance first (fastest)
    try:
        stock = yf.Ticker(ticker)
        
        # Determine expected price range based on ticker
        is_futures = ticker in ["ZW=F", "ZC=F", "ZS=F"]
        if is_futures:
            min_price, max_price = 300, 1000  # Cents per bushel
        else:
            min_price, max_price = 5, 15  # Dollars for ETFs
        
        # Method 1: Try fast_info first (most reliable for current price)
        try:
            fast_info = stock.fast_info
            current = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice')
            previous = fast_info.get('previousClose')
            
            if current and previous and min_price < current < max_price:
                return float(current), float(previous)
        except:
            pass
        
        # Method 2: Try info dictionary
        try:
            info = stock.info
            current = info.get('currentPrice') or info.get('regularMarketPrice')
            previous = info.get('previousClose') or info.get('open')
            
            if current and previous and min_price < current < max_price:
                return float(current), float(previous)
        except:
            pass
        
        # Method 3: Get from recent history
        try:
            data = stock.history(period='5d', auto_adjust=False, actions=False)
            
            if not data.empty and len(data) >= 2:
                current_price = float(data['Close'].iloc[-1])
                opening_price = float(data['Open'].iloc[-1])
                
                if min_price < current_price < max_price:
                    return current_price, opening_price
        except:
            pass
            
    except Exception as e:
        print(f"Yahoo Finance price fetch failed: {e}")
    
    # Fallback to Alpha Vantage if Yahoo failed and API key available
    if config.ALPHA_VANTAGE_API_KEY:
        try:
            av_symbol = ticker.replace("=F", "")  # Remove =F suffix
            current, opening = get_alpha_vantage_quote(av_symbol, config.ALPHA_VANTAGE_API_KEY)
            
            if current and opening:
                return current, opening
        except Exception as e:
            print(f"Alpha Vantage fallback failed: {e}")
    
    # Both failed
    print(f"Warning: All price fetch methods failed for {ticker}")
    return None, None

def add_technical_indicators(df):
    """Add technical indicators - adapts to available data"""
    
    data_length = len(df)
    
    # Returns (always available)
    df['Returns'] = df['Close'].pct_change()
    
    # Moving Averages (adapt to data length)
    if data_length >= 20:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    if data_length >= 50:
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
    elif data_length >= 20:
        # Use 20-day if we don't have 50
        df['SMA_50'] = df['Close'].rolling(window=20).mean()
    
    if data_length >= 200:
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
    elif data_length >= 100:
        # Use 100-day if we don't have 200
        df['SMA_200'] = df['Close'].rolling(window=100).mean()
    elif data_length >= 50:
        # Use 50-day as fallback
        df['SMA_200'] = df['Close'].rolling(window=50).mean()
    
    # MACD (needs at least 26 days)
    if data_length >= 26 and 'EMA_12' in df.columns:
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # RSI (needs at least 14 days)
    if data_length >= 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (needs at least 20 days)
    if data_length >= 20:
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Volatility (needs at least 20 days)
    if data_length >= 20:
        df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # ATR (needs at least 14 days)
    if data_length >= 14:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
    
    # Volume (needs at least 20 days)
    if data_length >= 20:
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    return df.dropna()

# ============================================================================
# LSTM MODEL
# ============================================================================

class WheatLSTMPredictor:
    """Simple LSTM predictor"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = config.LSTM_SEQUENCE_LENGTH
        self.feature_cols = [
            'Close', 'Volume', 'Returns',
            'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'BB_Width', 'Volatility', 'ATR'
        ]
    
    def prepare_data(self, df):
        """Prepare data for LSTM"""
        data = df[self.feature_cols].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i])
            current_close = df['Close'].iloc[i - 1]
            next_close = df['Close'].iloc[i]
            y.append(1 if next_close > current_close else 0)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(config.LSTM_UNITS, return_sequences=True, input_shape=input_shape),
            Dropout(config.LSTM_DROPOUT),
            LSTM(config.LSTM_UNITS // 2, return_sequences=False),
            Dropout(config.LSTM_DROPOUT),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, df, epochs=30):
        """Train the model"""
        X, y = self.prepare_data(df)
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training")
        
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return history
    
    def predict(self, df):
        """Make prediction"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        data = df[self.feature_cols].tail(self.sequence_length).values
        scaled_data = self.scaler.transform(data)
        X = np.array([scaled_data])
        
        prediction = self.model.predict(X, verbose=0)[0][0]
        direction = "UP" if prediction >= 0.5 else "DOWN"
        confidence = prediction if prediction >= 0.5 else (1 - prediction)
        
        return direction, float(confidence)
    
    def predict_enhanced(self, df, current_price):
        """
        ENHANCED prediction with seasonal, WASDE, and context analysis (v2.0)
        Returns: dict with enhanced prediction data
        """
        # 1. Base LSTM prediction
        base_direction, base_confidence = self.predict(df)
        
        # 2. Get seasonal bias
        seasonal = SeasonalAnalyzer.get_current_bias()
        seasonal_adj = seasonal['bias']
        
        # 3. Get WASDE fundamentals
        wasde = WASDEAnalyzer.get_fundamentals()
        wasde_adj = wasde['score']
        
        # 4. Get market context
        context = MarketContext.analyze_price(current_price)
        
        # 5. Calculate enhanced confidence
        enhanced_confidence = base_confidence
        
        # Apply seasonal adjustment
        if (base_direction == "UP" and seasonal_adj > 0) or \
           (base_direction == "DOWN" and seasonal_adj < 0):
            enhanced_confidence += abs(seasonal_adj)  # Reinforcing
        else:
            enhanced_confidence -= abs(seasonal_adj) * 0.5  # Conflicting
        
        # Apply WASDE adjustment
        if (base_direction == "UP" and wasde['signal'] == 'BULLISH') or \
           (base_direction == "DOWN" and wasde['signal'] == 'BEARISH'):
            enhanced_confidence += abs(wasde_adj) * 0.5  # Reinforcing
        else:
            enhanced_confidence -= abs(wasde_adj) * 0.3  # Conflicting
        
        # Apply context adjustment
        if context['signal'] != 'NEUTRAL':
            if (base_direction == "UP" and context['signal'] == 'BUY') or \
               (base_direction == "DOWN" and context['signal'] == 'SELL'):
                enhanced_confidence += 0.05
            else:
                enhanced_confidence -= 0.08
        
        # Clip to valid range
        enhanced_confidence = max(0.5, min(1.0, enhanced_confidence))
        
        return {
            'direction': base_direction,
            'base_confidence': base_confidence,
            'enhanced_confidence': enhanced_confidence,
            'seasonal': seasonal,
            'wasde': wasde,
            'context': context
        }

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations(wheat_data, predicted_direction):
    """Check correlations with other assets"""
    wheat_returns = wheat_data['Returns'].dropna()
    
    agreements = 0
    disagreements = 0
    correlations = {}
    
    for ticker in config.CORRELATED_TICKERS:
        try:
            df = fetch_data(ticker, days=365)
            if df is None or len(df) < 30:
                continue
            
            df['Returns'] = df['Close'].pct_change()
            asset_returns = df['Returns'].dropna()
            
            # Align dates
            aligned_wheat, aligned_asset = wheat_returns.align(asset_returns, join='inner')
            
            if len(aligned_wheat) < 30:
                continue
            
            # Calculate correlation
            corr = aligned_wheat.corr(aligned_asset)
            correlations[ticker] = corr
            
            # Check agreement
            if abs(corr) >= 0.65:
                latest_return = df['Returns'].iloc[-1]
                asset_direction = "UP" if latest_return > 0 else "DOWN"
                
                if corr > 0:  # Positive correlation
                    if asset_direction == predicted_direction:
                        agreements += 1
                    else:
                        disagreements += 1
                else:  # Negative correlation
                    if asset_direction != predicted_direction:
                        agreements += 1
                    else:
                        disagreements += 1
        except:
            continue
    
    is_supported = agreements >= config.MIN_CORRELATION_AGREEMENT
    
    return is_supported, agreements, disagreements, correlations

# ============================================================================
# SMART ALERT SYSTEM
# ============================================================================

def should_send_alert(current_direction, current_price, state):
    """Determine if alert should be sent based on direction change threshold"""
    
    # Always send alert if this is the first prediction
    if state['last_direction'] is None:
        return True, "First prediction"
    
    # If direction hasn't changed, no alert
    if current_direction == state['last_direction']:
        return False, "Same direction"
    
    # Direction changed - check if price change is significant enough
    if state['last_price'] is None:
        return True, "Direction changed (no previous price)"
    
    price_change_pct = abs((current_price - state['last_price']) / state['last_price'])
    
    if price_change_pct >= config.DIRECTION_CHANGE_THRESHOLD:
        return True, f"Direction changed with {price_change_pct:.1%} price movement"
    else:
        return False, f"Direction changed but only {price_change_pct:.1%} movement (need {config.DIRECTION_CHANGE_THRESHOLD:.1%})"

# ============================================================================
# MONITORING FUNCTION
# ============================================================================

def run_monitoring_check(bot_token, chat_id):
    """Run a single monitoring check"""
    try:
        # Load state
        state = load_state()
        
        # Fetch data
        wheat_data = fetch_data(config.PRIMARY_TICKER)
        if wheat_data is None:
            return "Failed to fetch data"
        
        wheat_data = add_technical_indicators(wheat_data)
        
        # Train model
        predictor = WheatLSTMPredictor()
        history = predictor.train(wheat_data, epochs=30)
        
        # Make prediction
        direction, confidence = predictor.predict(wheat_data)
        latest_price = wheat_data['Close'].iloc[-1]
        
        # Check correlations
        is_supported, agreements, disagreements, correlations = analyze_correlations(
            wheat_data, direction
        )
        
        # Check if we should send alert
        should_alert, alert_reason = should_send_alert(direction, latest_price, state)
        
        # Prepare signal data
        signal_data = {
            'direction': direction,
            'confidence': confidence,
            'price': latest_price,
            'is_supported': is_supported,
            'agreements': agreements,
            'disagreements': disagreements,
            'reason': None
        }
        
        # Calculate trade details if BUY signal
        if direction == "UP" and is_supported and confidence >= config.MIN_CONFIDENCE:
            stop_loss = latest_price * (1 - config.STOP_LOSS_PCT)
            take_profit = latest_price * (1 + config.TAKE_PROFIT_PCT)
            
            confidence_adj = (confidence - config.MIN_CONFIDENCE) / (1.0 - config.MIN_CONFIDENCE)
            confidence_adj = max(0.3, min(1.0, confidence_adj))
            
            portfolio_value = 10000  # Default, can be customized
            position_value = portfolio_value * config.MAX_POSITION_SIZE * confidence_adj
            shares = int(position_value / latest_price)
            
            signal_data.update({
                'action': 'BUY',
                'entry': latest_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'shares': shares,
                'position_value': position_value
            })
        else:
            signal_data['action'] = 'HOLD'
            if confidence < config.MIN_CONFIDENCE:
                signal_data['reason'] = f"Confidence too low: {confidence:.1%}"
            elif not is_supported:
                signal_data['reason'] = f"Only {agreements} correlations agree"
            elif direction == "DOWN":
                signal_data['reason'] = "Model predicts DOWN"
        
        # Send alert if needed
        alert_sent = False
        if should_alert:
            message = format_telegram_alert(signal_data)
            alert_sent = send_telegram_message(message, bot_token, chat_id)
            
            if alert_sent:
                state['alerts_sent'] += 1
        
        # Update state
        state['last_direction'] = direction
        state['last_price'] = latest_price
        state['last_check'] = datetime.now().isoformat()
        save_state(state)
        
        return {
            'success': True,
            'direction': direction,
            'confidence': confidence,
            'price': latest_price,
            'alert_sent': alert_sent,
            'alert_reason': alert_reason,
            'signal_data': signal_data
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    
    # Initialize session state
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'last_signal' not in st.session_state:
        st.session_state.last_signal = None
    
    st.title("🌾 Wheat Trading Dashboard - Enhanced")
    st.markdown("**24/7 Monitoring with Telegram Alerts**")
    
    # Sidebar - ENHANCED WITH METRICS
    st.sidebar.header("📊 Live Metrics")
    
    # Get latest prices
    current_price, opening_price = get_latest_price(config.PRIMARY_TICKER)
    
    if current_price:
        # Determine display format based on ticker
        is_futures = config.PRIMARY_TICKER in ["ZW=F", "ZC=F", "ZS=F"]
        
        if is_futures:
            # Futures are in cents per bushel
            st.sidebar.caption(f"Data for: {config.PRIMARY_TICKER} (Wheat Futures)")
            st.sidebar.caption("Price in cents per bushel")
            
            st.sidebar.metric("Opening Price", f"{opening_price:.2f}¢")
            price_change = current_price - opening_price
            price_change_pct = (price_change / opening_price) * 100
            st.sidebar.metric(
                "Current Price", 
                f"{current_price:.2f}¢",
                f"{price_change:+.2f}¢ ({price_change_pct:+.2f}%)"
            )
            
            # Show dollar equivalent
            st.sidebar.caption(f"≈ ${current_price/100:.2f} per bushel")
            
            # Sanity check for futures (400-800 cents typical range)
            if current_price > 1000 or current_price < 300:
                st.sidebar.warning(f"⚠️ Price seems unusual for wheat futures ({current_price:.0f}¢)")
        else:
            # ETF pricing
            st.sidebar.caption(f"Data for: {config.PRIMARY_TICKER} (Wheat ETF)")
            
            st.sidebar.metric("Opening Price", f"${opening_price:.2f}")
            price_change = current_price - opening_price
            price_change_pct = (price_change / opening_price) * 100
            st.sidebar.metric(
                "Current Price", 
                f"${current_price:.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
            )
            
            # Sanity check for ETF
            if current_price > 15 or current_price < 5:
                st.sidebar.warning(f"⚠️ Price seems unusual for WEAT (${current_price:.2f})")
    else:
        st.sidebar.warning("Unable to fetch current price")
        st.sidebar.info("Try clicking 'Generate Signal Now' to fetch data")
    
    # Show last prediction if available
    if st.session_state.last_signal:
        signal = st.session_state.last_signal
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Last Prediction")
        
        direction_icon = "🟢" if signal['direction'] == "UP" else "🔴"
        st.sidebar.metric(
            "Direction",
            f"{direction_icon} {signal['direction']}",
            f"{signal['confidence']:.1%} confidence"
        )
        
        if signal.get('signal_data', {}).get('action') == 'BUY':
            st.sidebar.success("✅ BUY Signal")
            st.sidebar.metric("Entry", f"${signal['signal_data']['entry']:.2f}")
            st.sidebar.metric("Stop Loss", f"${signal['signal_data']['stop_loss']:.2f}")
            st.sidebar.metric("Take Profit", f"${signal['signal_data']['take_profit']:.2f}")
        else:
            st.sidebar.info("⚠️ HOLD")
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Configuration")
    
    # Show Alpha Vantage status (automatic fallback)
    if config.ALPHA_VANTAGE_API_KEY:
        st.sidebar.success("✅ Alpha Vantage Fallback: Active")
        st.sidebar.caption("Will auto-use if Yahoo Finance fails")
    else:
        st.sidebar.info("💡 Alpha Vantage Fallback: Disabled")
        with st.sidebar.expander("Enable Alpha Vantage Fallback"):
            st.markdown("""
            **Optional:** Add your Alpha Vantage API key for automatic fallback.
            
            1. Get free key: [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
            2. Set environment variable:
               ```
               ALPHA_VANTAGE_API_KEY=your_key_here
               ```
            3. Or create `.env` file with your key
            
            **How it works:**
            - Yahoo Finance tried first (fast & free)
            - If Yahoo fails → Alpha Vantage used automatically
            - You never need to select it manually!
            """)
    
    st.sidebar.markdown("---")
    
    # Ticker selection
    ticker_option = st.sidebar.selectbox(
        "Primary Asset",
        [
            "ZW=F (Wheat Futures - cents/bushel)",
            "WEAT (Wheat ETF - dollars)",
            "ZC=F (Corn Futures - cents/bushel)"
        ],
        index=0,  # Default to wheat futures
        help="Futures show cents per bushel (e.g., 532¢ = $5.32/bushel)"
    )
    
    if "ZW=F" in ticker_option:
        config.PRIMARY_TICKER = "ZW=F"
        config.TICKER_TYPE = "FUTURES"
    elif "ZC=F" in ticker_option:
        config.PRIMARY_TICKER = "ZC=F"
        config.TICKER_TYPE = "FUTURES"
    else:
        config.PRIMARY_TICKER = "WEAT"
        config.TICKER_TYPE = "ETF"
    
    st.sidebar.caption(f"Trading: {config.PRIMARY_TICKER}")
    
    portfolio_value = st.sidebar.number_input("Portfolio Value ($)", value=10000, step=1000)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Risk Parameters")
    st.sidebar.metric("Max Position", f"{config.MAX_POSITION_SIZE:.0%}")
    st.sidebar.metric("Stop Loss", f"{config.STOP_LOSS_PCT:.1%}")
    st.sidebar.metric("Take Profit", f"{config.TAKE_PROFIT_PCT:.1%}")
    st.sidebar.metric("Min Confidence", f"{config.MIN_CONFIDENCE:.0%}")
    
    # Telegram Configuration
    st.sidebar.markdown("---")
    st.sidebar.header("📱 Telegram Alerts")
    
    # Check if credentials loaded from .env
    telegram_from_env = bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)
    
    if telegram_from_env:
        st.sidebar.success("✅ Telegram: Configured from .env")
        st.sidebar.caption(f"Chat ID: {config.TELEGRAM_CHAT_ID[:8]}...")
        
        telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts", value=True)
        
        if telegram_enabled:
            if st.sidebar.button("📤 Test Telegram"):
                test_msg = "🌾 *Wheat Trading Bot Connected!*\n\nYou will receive alerts here."
                if send_telegram_message(test_msg, config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID):
                    st.sidebar.success("✅ Test message sent!")
                else:
                    st.sidebar.error("❌ Failed to send test message")
    else:
        st.sidebar.info("💡 Telegram: Not configured")
        
        telegram_enabled = st.sidebar.checkbox("Enable Telegram Alerts")
        
        if telegram_enabled:
            st.sidebar.markdown("**Enter credentials manually:**")
            bot_token = st.sidebar.text_input(
                "Bot Token",
                type="password",
                help="Get from @BotFather on Telegram"
            )
            chat_id = st.sidebar.text_input(
                "Chat ID",
                help="Your Telegram chat ID"
            )
            
            # Override config if manually entered
            if bot_token and chat_id:
                config.TELEGRAM_BOT_TOKEN = bot_token
                config.TELEGRAM_CHAT_ID = chat_id
                
                if st.sidebar.button("📤 Test Telegram"):
                    test_msg = "🌾 *Wheat Trading Bot Connected!*\n\nYou will receive alerts here."
                    if send_telegram_message(test_msg, bot_token, chat_id):
                        st.sidebar.success("✅ Test message sent!")
                    else:
                        st.sidebar.error("❌ Failed to send test message")
            else:
                st.sidebar.warning("Or add to .env file for automatic loading")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🔄 Auto Monitor", "📈 Chart", "ℹ️ About"])
    
    with tab1:
        st.header("Current Market Analysis")
        
        if st.button("🔄 Generate Signal Now", type="primary"):
            with st.spinner("Fetching data and training model..."):
                result = run_monitoring_check(
                    config.TELEGRAM_BOT_TOKEN,
                    config.TELEGRAM_CHAT_ID
                )
                
                if result.get('success'):
                    st.session_state.last_signal = result
                    
                    signal_data = result['signal_data']
                    
                    st.markdown("---")
                    
                    # Display signal
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        signal_color = "🟢" if signal_data['direction'] == "UP" else "🔴"
                        st.metric("Signal", f"{signal_color} {signal_data['direction']}", 
                                 f"{signal_data['confidence']:.1%} confidence")
                    
                    with col2:
                        st.metric("Current Price", f"${signal_data['price']:.2f}")
                    
                    with col3:
                        alert_status = "✅ Sent" if result['alert_sent'] else "⏸️ Skipped"
                        st.metric("Alert", alert_status)
                    
                    st.info(f"**Alert Logic:** {result['alert_reason']}")
                    
                    # Correlation analysis
                    st.markdown("### 🔗 Correlation Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Assets Agreeing", signal_data['agreements'])
                    with col2:
                        st.metric("Assets Disagreeing", signal_data['disagreements'])
                    with col3:
                        support_icon = "✅" if signal_data['is_supported'] else "❌"
                        st.metric("Supported", support_icon)
                    
                    # Action
                    if signal_data['action'] == 'BUY':
                        st.markdown("---")
                        st.success("### ✅ BUY SIGNAL")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Entry Price", f"${signal_data['entry']:.2f}")
                            st.metric("Position Size", f"{signal_data['shares']} shares")
                            st.metric("Position Value", f"${signal_data['position_value']:.2f}")
                        
                        with col2:
                            st.metric("Stop Loss", f"${signal_data['stop_loss']:.2f}", 
                                    f"-{config.STOP_LOSS_PCT:.1%}")
                            st.metric("Take Profit", f"${signal_data['take_profit']:.2f}", 
                                    f"+{config.TAKE_PROFIT_PCT:.1%}")
                            st.metric("Risk/Reward", "2.0:1")
                        
                        max_loss = (signal_data['entry'] - signal_data['stop_loss']) * signal_data['shares']
                        max_gain = (signal_data['take_profit'] - signal_data['entry']) * signal_data['shares']
                        
                        st.info(f"**Max Loss:** ${max_loss:.2f} | **Max Gain:** ${max_gain:.2f}")
                    else:
                        st.warning("### ⚠️ HOLD - No Trade")
                        if signal_data.get('reason'):
                            st.info(signal_data['reason'])
                    
                    # Show state
                    state = load_state()
                    st.markdown("---")
                    st.markdown(f"**Last Check:** {state.get('last_check', 'Never')}")
                    st.markdown(f"**Alerts Sent:** {state.get('alerts_sent', 0)}")
                    
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.header("🔄 24/7 Automatic Monitoring")
        
        st.info("""
        **How it works:**
        - Checks wheat signal every 5 minutes
        - Sends Telegram alert only when direction changes by 2.5%+ 
        - Prevents spam: If prediction is UP, next alert only if it changes to DOWN with 2.5%+ price movement
        - Runs continuously in background
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            check_interval = st.number_input(
                "Check Interval (minutes)",
                min_value=1,
                max_value=60,
                value=5,
                help="How often to check for signals"
            )
            config.CHECK_INTERVAL = check_interval * 60
        
        with col2:
            alert_threshold = st.number_input(
                "Direction Change Threshold (%)",
                min_value=0.5,
                max_value=10.0,
                value=2.5,
                step=0.5,
                help="Minimum price change to trigger alert"
            )
            config.DIRECTION_CHANGE_THRESHOLD = alert_threshold / 100
        
        st.markdown("---")
        
        if not telegram_enabled or not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            st.warning("⚠️ Configure Telegram in sidebar to enable monitoring")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▶️ Start Monitoring", type="primary"):
                    st.session_state.monitoring_active = True
                    st.success("✅ Monitoring started!")
                    st.info(f"Checking every {check_interval} minutes...")
            
            with col2:
                if st.button("⏸️ Stop Monitoring"):
                    st.session_state.monitoring_active = False
                    st.warning("⏸️ Monitoring stopped")
            
            with col3:
                if st.button("🗑️ Clear State"):
                    save_state({
                        'last_direction': None,
                        'last_price': None,
                        'last_check': None,
                        'alerts_sent': 0
                    })
                    st.success("✅ State cleared")
            
            # Display monitoring status
            if st.session_state.monitoring_active:
                st.success("🟢 **Status: MONITORING ACTIVE**")
                
                state = load_state()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Last Direction", state.get('last_direction', 'None'))
                with col2:
                    last_price = state.get('last_price')
                    st.metric("Last Price", f"${last_price:.2f}" if last_price else "N/A")
                with col3:
                    st.metric("Alerts Sent", state.get('alerts_sent', 0))
                with col4:
                    last_check = state.get('last_check')
                    if last_check:
                        check_time = datetime.fromisoformat(last_check)
                        time_ago = (datetime.now() - check_time).seconds // 60
                        st.metric("Last Check", f"{time_ago}m ago")
                    else:
                        st.metric("Last Check", "Never")
                
                # Auto-refresh mechanism
                st.markdown("---")
                st.info("💡 **Tip:** Keep this tab open for continuous monitoring. The page will auto-check at intervals.")
                
                # Manual run button during monitoring
                if st.button("🔄 Run Check Now"):
                    with st.spinner("Running check..."):
                        result = run_monitoring_check(
                            config.TELEGRAM_BOT_TOKEN,
                            config.TELEGRAM_CHAT_ID
                        )
                        
                        if result.get('success'):
                            st.success("✅ Check completed!")
                            st.json(result)
                        else:
                            st.error(f"❌ Error: {result.get('error')}")
                
                # Auto-refresh (experimental)
                if st.checkbox("Enable Auto-Refresh (Experimental)"):
                    st.warning("⚠️ This will refresh the page automatically. May cause high CPU usage.")
                    time.sleep(config.CHECK_INTERVAL)
                    st.rerun()
            else:
                st.info("⏸️ **Status: MONITORING STOPPED**")
                st.markdown("Click 'Start Monitoring' to begin 24/7 checks")
    
    with tab3:
        st.header("📈 Price Chart")
        
        wheat_data = fetch_data(config.PRIMARY_TICKER, days=180)
        
        if wheat_data is not None and len(wheat_data) > 0:
            # Create candlestick chart
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=wheat_data.index,
                open=wheat_data['Open'],
                high=wheat_data['High'],
                low=wheat_data['Low'],
                close=wheat_data['Close'],
                name='WEAT'
            ))
            
            # Add indicators if we have enough data
            try:
                wheat_data_with_indicators = add_technical_indicators(wheat_data)
                
                if len(wheat_data_with_indicators) > 0:
                    fig.add_trace(go.Scatter(
                        x=wheat_data_with_indicators.index,
                        y=wheat_data_with_indicators['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange', width=1)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=wheat_data_with_indicators.index,
                        y=wheat_data_with_indicators['SMA_50'],
                        name='SMA 50',
                        line=dict(color='blue', width=1)
                    ))
            except Exception as e:
                st.warning(f"Could not add technical indicators to chart: {e}")
                wheat_data_with_indicators = None
            
            fig.update_layout(
                title='WEAT - Wheat ETF',
                yaxis_title='Price ($)',
                xaxis_title='Date',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            st.markdown("### 📊 Technical Indicators")
            
            if wheat_data_with_indicators is not None and len(wheat_data_with_indicators) > 0:
                try:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        rsi = wheat_data_with_indicators['RSI'].iloc[-1]
                        st.metric("RSI (14)", f"{rsi:.1f}")
                    
                    with col2:
                        macd = wheat_data_with_indicators['MACD'].iloc[-1]
                        st.metric("MACD", f"{macd:.4f}")
                    
                    with col3:
                        vol = wheat_data_with_indicators['Volatility'].iloc[-1]
                        st.metric("Volatility", f"{vol:.4f}")
                    
                    with col4:
                        atr = wheat_data_with_indicators['ATR'].iloc[-1]
                        st.metric("ATR", f"{atr:.2f}")
                except Exception as e:
                    st.warning(f"Could not display technical indicators: {e}")
            else:
                st.info("Not enough data to calculate technical indicators. Need at least 200 days of historical data.")
        else:
            st.error("Failed to fetch chart data")
    
    with tab4:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### Enhanced Wheat Trading Dashboard
        
        **New Features:**
        - 🌐 **24/7 Monitoring:** Automatic signal checks every 5 minutes
        - 📱 **Telegram Alerts:** Real-time notifications to your phone
        - 🎯 **Smart Alerts:** Only alerts on direction changes with 2.5%+ price movement
        - 📊 **Sidebar Metrics:** Opening price, current price, and last prediction always visible
        
        #### Alert Logic:
        - **First prediction:** Always sends alert
        - **Same direction:** No alert (prevents spam)
        - **Direction change:** Only alerts if price moved 2.5%+ since last alert
        
        Example: If last prediction was UP at $7.50, next alert only if prediction changes to DOWN 
        AND price is now at $7.31 or lower (2.5% change).
        
        #### How to Set Up Telegram:
        1. Open Telegram and search for @BotFather
        2. Send `/newbot` and follow instructions
        3. Copy the bot token
        4. Send a message to your bot
        5. Go to `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
        6. Find your chat_id in the response
        7. Enter both in the sidebar
        
        #### Features:
        - 🧠 **LSTM Neural Network** for price direction prediction
        - 🔗 **Correlation Analysis** with 6 related assets
        - 📊 **Technical Indicators** (RSI, MACD, Bollinger Bands, etc.)
        - 🛡️ **Risk Management** (position sizing, stops, targets)
        
        #### Expected Performance:
        - **Accuracy:** 52-58% (realistic expectation)
        - **Risk/Reward:** 2:1 (2% stop, 4% target)
        - **Position Size:** Max 10% of portfolio
        
        #### ⚠️ Important Disclaimers:
        - Start with **paper trading**
        - Expected accuracy is 52-58%, **not** 68-80%
        - Never risk money you can't afford to lose
        - This is educational software, **not financial advice**
        
        ---
        
        **Version:** 2.0 | **Asset:** WEAT ETF | **Mode:** Enhanced Monitoring
        """)

if __name__ == "__main__":
    main()
