"""
Wheat Monitor for GitHub Actions - v2.0 ENHANCED
Runs every 15 minutes, sends Telegram alerts automatically
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

# CONFIG v2.0
PRIMARY_TICKER = "ZW=F"
STOP_LOSS_PCT = 0.015       # 1.5%
TAKE_PROFIT_PCT = 0.025     # 2.5%
MIN_CONFIDENCE = 0.55
DIRECTION_CHANGE_THRESHOLD = 0.025

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

NORMAL_RANGE_LOW = 480
NORMAL_RANGE_HIGH = 620

SEASONAL_BIAS = {
    1: 0.00, 2: -0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: -0.05,
    7: -0.03, 8: -0.02, 9: 0.02, 10: 0.03, 11: 0.04, 12: 0.05
}

STATE_FILE = Path("wheat_monitor_state.json")

# ANALYZERS
def get_seasonal_bias():
    month = datetime.now().month
    bias = SEASONAL_BIAS.get(month, 0.0)
    explanations = {
        1: "Neutral", 2: "Pre-spring", 3: "Spring rally",
        4: "Peak planting", 5: "Max premium", 6: "Harvest (LOW)",
        7: "Post-harvest", 8: "Summer lull", 9: "Fall recovery",
        10: "Winter demand", 11: "Pre-winter", 12: "Winter (HIGH)"
    }
    direction = 'BULLISH' if bias > 0.02 else 'BEARISH' if bias < -0.02 else 'NEUTRAL'
    return {'bias': bias, 'direction': direction, 'explanation': explanations.get(month, "")}

def get_wasde_signal():
    return {'stocks_to_use': 0.18, 'signal': 'BULLISH', 'score': 0.25}

def get_market_context(price):
    if price < NORMAL_RANGE_LOW:
        return {'position': 'BELOW_NORMAL', 'signal': 'BUY'}
    elif price > NORMAL_RANGE_HIGH:
        return {'position': 'ABOVE_NORMAL', 'signal': 'SELL'}
    else:
        return {'position': 'NORMAL', 'signal': 'NEUTRAL'}

# STATE
def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'last_direction': None, 'last_price': None, 'alerts_sent': 0}

def save_state(state):
    state['last_check'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# TELEGRAM
def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram not configured")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

# DATA
def fetch_data(ticker, days=730):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)
        return None if df.empty else df
    except Exception as e:
        print(f"Data fetch error: {e}")
        return None

def add_indicators(df):
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
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
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    return df.dropna()

# LSTM
class WheatPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.seq_len = 60
        self.features = ['Close', 'Volume', 'Returns', 'SMA_20', 'SMA_50', 
                        'RSI', 'MACD', 'BB_Width', 'Volatility', 'ATR']
    
    def train(self, df):
        data = df[self.features].values
        scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i-self.seq_len:i])
            y.append(1 if df['Close'].iloc[i] > df['Close'].iloc[i-1] else 0)
        X, y = np.array(X), np.array(y)
        if len(X) < 100:
            raise ValueError("Not enough data")
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
    
    def predict_enhanced(self, df, current_price):
        data = df[self.features].tail(self.seq_len).values
        scaled = self.scaler.transform(data)
        X = np.array([scaled])
        pred = self.model.predict(X, verbose=0)[0][0]
        direction = "UP" if pred >= 0.5 else "DOWN"
        base_conf = pred if pred >= 0.5 else (1 - pred)
        
        seasonal = get_seasonal_bias()
        wasde = get_wasde_signal()
        context = get_market_context(current_price)
        
        enhanced_conf = base_conf
        if (direction == "UP" and seasonal['bias'] > 0) or (direction == "DOWN" and seasonal['bias'] < 0):
            enhanced_conf += abs(seasonal['bias'])
        else:
            enhanced_conf -= abs(seasonal['bias']) * 0.5
        
        if (direction == "UP" and wasde['signal'] == 'BULLISH') or (direction == "DOWN" and wasde['signal'] == 'BEARISH'):
            enhanced_conf += abs(wasde['score']) * 0.5
        else:
            enhanced_conf -= abs(wasde['score']) * 0.3
        
        if context['signal'] != 'NEUTRAL':
            if (direction == "UP" and context['signal'] == 'BUY') or (direction == "DOWN" and context['signal'] == 'SELL'):
                enhanced_conf += 0.05
            else:
                enhanced_conf -= 0.08
        
        enhanced_conf = max(0.5, min(1.0, enhanced_conf))
        return {'direction': direction, 'confidence': enhanced_conf, 'seasonal': seasonal, 'wasde': wasde, 'context': context}

# ALERT LOGIC
def should_alert(direction, price, state):
    if state['last_direction'] is None:
        return True, "First prediction"
    if direction == state['last_direction']:
        return False, "Same direction"
    if state['last_price'] is None:
        return True, "Direction changed"
    change_pct = abs((price - state['last_price']) / state['last_price'])
    if change_pct >= DIRECTION_CHANGE_THRESHOLD:
        return True, f"Direction changed with {change_pct:.1%} move"
    else:
        return False, f"Only {change_pct:.1%} move"

# MAIN
def main():
    print(f"\n{'='*70}")
    print(f"🌾 WHEAT MONITOR v2.0 - GitHub Actions")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*70}\n")
    
    state = load_state()
    
    try:
        print(f"📊 Fetching {PRIMARY_TICKER}...")
        df = fetch_data(PRIMARY_TICKER)
        if df is None:
            print("❌ No data")
            return
        
        df = add_indicators(df)
        price = df['Close'].iloc[-1]
        print(f"✓ Price: {price:.2f}¢")
        
        print("🧠 Training model...")
        predictor = WheatPredictor()
        predictor.train(df)
        print("✓ Ready")
        
        result = predictor.predict_enhanced(df, price)
        direction = result['direction']
        confidence = result['confidence']
        seasonal = result['seasonal']
        wasde = result['wasde']
        context = result['context']
        
        print(f"\n🎯 {direction} ({confidence:.1%})")
        print(f"   Seasonal: {seasonal['direction']}")
        print(f"   WASDE: {wasde['signal']}")
        print(f"   Context: {context['position']}")
        
        send_alert, reason = should_alert(direction, price, state)
        print(f"\n📢 {reason}")
        
        if send_alert and confidence >= MIN_CONFIDENCE:
            stop = price * (1 - STOP_LOSS_PCT) if direction == "UP" else price * (1 + STOP_LOSS_PCT)
            target = price * (1 + TAKE_PROFIT_PCT) if direction == "UP" else price * (1 - TAKE_PROFIT_PCT)
            
            message = f"""
🌾 *WHEAT ALERT v2.0* 🌾

{'🟢' if direction == 'UP' else '🔴'} *{direction}* ({confidence:.1%})
💰 *{price:.2f}¢* (${price/100:.2f}/bu)

📅 {seasonal['direction']} - {seasonal['explanation']}
📊 WASDE: {wasde['signal']}
🎯 {context['position']}

💼 Entry: {price:.2f}¢
🛑 Stop: {stop:.2f}¢ ({STOP_LOSS_PCT:.1%})
🎯 Target: {target:.2f}¢ ({TAKE_PROFIT_PCT:.1%})

_{reason}_ 🤖
"""
            
            if send_telegram(message):
                print("✅ Alert sent!")
                state['alerts_sent'] += 1
            else:
                print("❌ Alert failed")
        
        state['last_direction'] = direction
        state['last_price'] = price
        save_state(state)
        
        print(f"\n📊 Total alerts: {state['alerts_sent']}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    main()
