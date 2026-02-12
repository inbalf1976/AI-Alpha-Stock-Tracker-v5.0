"""
Wheat Trading Monitor - GitHub Actions Version
Runs once per execution (GitHub Actions calls it every 5 minutes)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import json
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# ============================================================================
# CONFIGURATION
# ============================================================================

PRIMARY_TICKER = "ZW=F"  # Wheat Futures
DIRECTION_CHANGE_THRESHOLD = 0.025  # 2.5%
MIN_CONFIDENCE = 0.60

# Get from GitHub Secrets
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

STATE_FILE = Path("wheat_monitor_state.json")

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def load_state():
    """Load monitoring state"""
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
        'alerts_sent': 0,
        'last_alert_time': None
    }

def save_state(state):
    """Save monitoring state"""
    state['last_check'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"✓ State saved: {state}")

# ============================================================================
# TELEGRAM
# ============================================================================

def send_telegram(message):
    """Send Telegram message"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=data, timeout=10)
        success = response.status_code == 200
        print(f"✓ Telegram: {'Sent' if success else 'Failed'}")
        return success
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_data(ticker, days=730):
    """Fetch historical data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)
        
        if df.empty:
            print(f"❌ No data for {ticker}")
            return None
        
        print(f"✓ Fetched {len(df)} days of data")
        return df
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return None

def add_technical_indicators(df):
    """Add technical indicators"""
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df.dropna()

# ============================================================================
# LSTM MODEL
# ============================================================================

class WheatPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60
        self.feature_cols = [
            'Close', 'Volume', 'Returns',
            'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'BB_Width', 'Volatility', 'ATR'
        ]
    
    def prepare_data(self, df):
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
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, df):
        X, y = self.prepare_data(df)
        if len(X) < 100:
            raise ValueError("Insufficient data")
        
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)
        
        self.model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
        print("✓ Model trained")
    
    def predict(self, df):
        if self.model is None:
            raise ValueError("Model not trained")
        
        data = df[self.feature_cols].tail(self.sequence_length).values
        scaled_data = self.scaler.transform(data)
        X = np.array([scaled_data])
        
        prediction = self.model.predict(X, verbose=0)[0][0]
        direction = "UP" if prediction >= 0.5 else "DOWN"
        confidence = prediction if prediction >= 0.5 else (1 - prediction)
        
        return direction, float(confidence)

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def analyze_correlations(wheat_data, predicted_direction):
    """Quick correlation check"""
    correlated_tickers = ["CORN", "ZC=F", "TAGS", "DBA"]
    wheat_returns = wheat_data['Returns'].dropna()
    
    agreements = 0
    
    for ticker in correlated_tickers:
        try:
            df = fetch_data(ticker, days=365)
            if df is None or len(df) < 30:
                continue
            
            df['Returns'] = df['Close'].pct_change()
            asset_returns = df['Returns'].dropna()
            
            aligned_wheat, aligned_asset = wheat_returns.align(asset_returns, join='inner')
            if len(aligned_wheat) < 30:
                continue
            
            corr = aligned_wheat.corr(aligned_asset)
            
            if abs(corr) >= 0.65:
                latest_return = df['Returns'].iloc[-1]
                asset_direction = "UP" if latest_return > 0 else "DOWN"
                
                if (corr > 0 and asset_direction == predicted_direction) or \
                   (corr < 0 and asset_direction != predicted_direction):
                    agreements += 1
        except:
            continue
    
    is_supported = agreements >= 3
    print(f"✓ Correlations: {agreements} agree, Supported: {is_supported}")
    return is_supported, agreements

# ============================================================================
# ALERT LOGIC
# ============================================================================

def should_send_alert(current_direction, current_price, state):
    """Determine if alert should be sent"""
    
    if state['last_direction'] is None:
        return True, "First prediction"
    
    if current_direction == state['last_direction']:
        return False, "Same direction"
    
    if state['last_price'] is None:
        return True, "Direction changed"
    
    price_change_pct = abs((current_price - state['last_price']) / state['last_price'])
    
    if price_change_pct >= DIRECTION_CHANGE_THRESHOLD:
        return True, f"Direction changed with {price_change_pct:.1%} movement"
    else:
        return False, f"Direction changed but only {price_change_pct:.1%} (need {DIRECTION_CHANGE_THRESHOLD:.1%})"

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run single monitoring check"""
    print(f"\n{'='*70}")
    print(f"🌾 WHEAT MONITOR - GitHub Actions")
    print(f"Check at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*70}\n")
    
    state = load_state()
    
    try:
        # Fetch data
        print(f"📊 Fetching {PRIMARY_TICKER} data...")
        df = fetch_data(PRIMARY_TICKER)
        if df is None:
            print("❌ Cannot proceed without data")
            return
        
        df = add_technical_indicators(df)
        latest_price = df['Close'].iloc[-1]
        print(f"✓ Current price: {latest_price:.2f}¢")
        
        # Train model
        print("🧠 Training LSTM model...")
        predictor = WheatPredictor()
        predictor.train(df)
        
        # Predict
        direction, confidence = predictor.predict(df)
        print(f"✓ Prediction: {direction} ({confidence:.1%})")
        
        # Check correlations
        is_supported, agreements = analyze_correlations(df, direction)
        
        # Determine if alert needed
        should_alert, reason = should_send_alert(direction, latest_price, state)
        print(f"📢 Alert decision: {reason}")
        
        # Send alert if needed
        if should_alert and confidence >= MIN_CONFIDENCE:
            
            message = f"""
🌾 *WHEAT ALERT* 🌾

{'🟢' if direction == 'UP' else '🔴'} *Signal:* {direction}
📊 *Confidence:* {confidence:.1%}
💰 *Price:* {latest_price:.2f}¢ (${latest_price/100:.2f}/bushel)
🕐 *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

{'✅' if is_supported else '⚠️'} *Correlations:* {agreements} assets agree

_{reason}_

_Monitored by GitHub Actions 🤖_
"""
            
            if send_telegram(message):
                state['alerts_sent'] += 1
                state['last_alert_time'] = datetime.now().isoformat()
            
        # Update state
        state['last_direction'] = direction
        state['last_price'] = latest_price
        save_state(state)
        
        print(f"\n📊 Total alerts sent: {state['alerts_sent']}")
        print(f"{'='*70}\n")
        print("✅ Monitoring check complete")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Send error notification
        if TELEGRAM_BOT_TOKEN:
            error_msg = f"⚠️ *Wheat Monitor Error*\n\n`{str(e)}`\n\n_GitHub Actions monitoring_"
            send_telegram(error_msg)

if __name__ == "__main__":
    main()
