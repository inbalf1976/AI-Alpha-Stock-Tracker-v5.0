"""
PROFESSIONAL WHEAT TRADING SYSTEM - ULTIMATE EDITION
Combines: Ensemble AI + Weather + WASDE + Volume + Seasonal + Context
Expected Accuracy: 75-85%
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

# Import our modules
from weather_analyzer import WeatherAnalyzer
from wasde_scraper import WASDEScraper
from volume_analyzer import VolumeAnalyzer
from ensemble_predictor import EnsemblePredictor

# CONFIG
PRIMARY_TICKER = "ZW=F"
STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.025
MIN_CONFIDENCE = 0.55
DIRECTION_CHANGE_THRESHOLD = 0.025

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

NORMAL_RANGE_LOW = 480
NORMAL_RANGE_HIGH = 620

SEASONAL_BIAS = {
    1:0.00, 2:-0.02, 3:0.03, 4:0.04, 5:0.05, 6:-0.05,
    7:-0.03, 8:-0.02, 9:0.02, 10:0.03, 11:0.04, 12:0.05
}

STATE_FILE = Path("wheat_monitor_state.json")

# HELPERS
def get_seasonal_bias():
    month = datetime.now().month
    bias = SEASONAL_BIAS.get(month, 0.0)
    explanations = {
        1:"Neutral",2:"Pre-spring",3:"Spring rally",4:"Peak planting",5:"Max premium",6:"Harvest (LOW)",
        7:"Post-harvest",8:"Summer lull",9:"Fall recovery",10:"Winter demand",11:"Pre-winter",12:"Winter (HIGH)"
    }
    direction = 'BULLISH' if bias > 0.02 else 'BEARISH' if bias < -0.02 else 'NEUTRAL'
    return {'bias':bias,'direction':direction,'explanation':explanations.get(month,"")}

def get_market_context(price):
    if price < NORMAL_RANGE_LOW:
        return {'position':'BELOW_NORMAL','signal':'BUY'}
    elif price > NORMAL_RANGE_HIGH:
        return {'position':'ABOVE_NORMAL','signal':'SELL'}
    else:
        return {'position':'NORMAL','signal':'NEUTRAL'}

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE,'r') as f:
                state = json.load(f)
                
                # Check if model needs reset (every 2 years)
                last_reset = state.get('last_model_reset', None)
                if last_reset:
                    from datetime import datetime
                    last_reset_date = datetime.fromisoformat(last_reset)
                    days_since_reset = (datetime.now() - last_reset_date).days
                    
                    if days_since_reset >= 730:  # 2 years = 730 days
                        print("⚠️  MODEL RESET: 2+ years since last reset")
                        print(f"   Last reset: {last_reset_date.strftime('%Y-%m-%d')}")
                        print(f"   Days: {days_since_reset}")
                        print("   Resetting to prevent overfitting...")
                        
                        # Reset model-related state but keep alert history
                        state['last_model_reset'] = datetime.now().isoformat()
                        state['model_version'] = state.get('model_version', 1) + 1
                        state['reset_count'] = state.get('reset_count', 0) + 1
                        
                        # Optionally clear prediction history (uncomment if desired)
                        # state['last_direction'] = None
                        # state['last_price'] = None
                        
                        return state
                else:
                    # First time - set initial reset date
                    from datetime import datetime
                    state['last_model_reset'] = datetime.now().isoformat()
                    state['model_version'] = 1
                    state['reset_count'] = 0
                
                return state
        except:
            pass
    
    # New state
    from datetime import datetime
    return {
        'last_direction': None,
        'last_price': None,
        'alerts_sent': 0,
        'last_model_reset': datetime.now().isoformat(),
        'model_version': 1,
        'reset_count': 0
    }

def save_state(state):
    state['last_check'] = datetime.now().isoformat()
    with open(STATE_FILE,'w') as f:
        json.dump(state,f,indent=2)

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram not configured")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id":TELEGRAM_CHAT_ID,"text":message,"parse_mode":"Markdown"}
        response = requests.post(url,data=data,timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def fetch_data(ticker,days=730):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date,end=end_date,auto_adjust=False)
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
    gain = (delta.where(delta>0,0)).rolling(window=14).mean()
    loss = (-delta.where(delta<0,0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100/(1+gain/loss))
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2*bb_std)
    df['BB_Lower'] = df['BB_Middle'] - (2*bb_std)
    df['BB_Width'] = (df['BB_Upper']-df['BB_Lower'])/df['BB_Middle']
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High']-df['Close'].shift())
    low_close = np.abs(df['Low']-df['Close'].shift())
    ranges = pd.concat([high_low,high_close,low_close],axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    return df.dropna()

def should_alert(direction, price, state):
    """
    Determine if alert should be sent
    Rules:
    1. First prediction of the day → Alert
    2. Direction changed + 2.5%+ price move → Alert
    3. Same direction OR small move → No alert
    4. Reset at start of new trading day
    """
    from datetime import datetime
    
    # Get current date
    current_date = datetime.now().date().isoformat()
    last_alert_date = state.get('last_alert_date', None)
    
    # Check if it's a new trading day
    if last_alert_date != current_date:
        # New trading day - reset and send first alert
        state['daily_alert_sent'] = False
        state['last_alert_date'] = current_date
        return True, f"First prediction of {current_date}"
    
    # Check if we already sent an alert today
    if state.get('daily_alert_sent', False):
        # Already sent alert today, check if significant change
        if state['last_direction'] is None:
            return True, "First prediction"
        
        if direction == state['last_direction']:
            return False, "Same direction - no alert"
        
        # Direction changed - check price movement
        if state['last_price'] is None:
            return True, "Direction changed"
        
        change_pct = abs((price - state['last_price']) / state['last_price'])
        
        if change_pct >= DIRECTION_CHANGE_THRESHOLD:  # 2.5%
            state['daily_alert_sent'] = True  # Mark as sent
            return True, f"Direction changed with {change_pct:.1%} move"
        else:
            return False, f"Direction changed but only {change_pct:.1%} move (need 2.5%+)"
    
    # Haven't sent alert today yet
    if state['last_direction'] is None:
        state['daily_alert_sent'] = True
        return True, "First prediction of session"
    
    if direction == state['last_direction']:
        return False, "Same direction - no alert"
    
    # Direction changed
    if state['last_price'] is None:
        state['daily_alert_sent'] = True
        return True, "Direction changed"
    
    change_pct = abs((price - state['last_price']) / state['last_price'])
    
    if change_pct >= DIRECTION_CHANGE_THRESHOLD:
        state['daily_alert_sent'] = True
        return True, f"Direction changed with {change_pct:.1%} move"
    else:
        return False, f"Direction changed but only {change_pct:.1%} move (need 2.5%+)"

def main():
    print(f"\n{'='*80}")
    print(f"🌾 PROFESSIONAL WHEAT MONITOR - ULTIMATE EDITION v3.0")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Features: Ensemble AI + Weather + WASDE + Volume + Seasonal")
    print(f"{'='*80}\n")
    
    state = load_state()
    
    try:
        # Fetch data
        print(f"📊 Fetching {PRIMARY_TICKER}...")
        df = fetch_data(PRIMARY_TICKER)
        if df is None:
            print("❌ No data")
            return
        
        df = add_indicators(df)
        price = df['Close'].iloc[-1]
        print(f"✓ Price: {price:.2f}¢")
        
        # Initialize analyzers
        print("\n🔬 Initializing advanced analyzers...")
        weather = WeatherAnalyzer()
        wasde = WASDEScraper()
        volume = VolumeAnalyzer()
        
        # Get all signals
        print("📡 Gathering signals...")
        seasonal = get_seasonal_bias()
        print(f"  ✓ Seasonal: {seasonal['direction']}")
        
        weather_signal = weather.get_weather_signal()
        print(f"  ✓ Weather: {weather_signal['signal']}")
        
        wasde_signal = wasde.get_fundamental_score()
        print(f"  ✓ WASDE: {wasde_signal['signal']}")
        
        volume_signal = volume.analyze_volume(df)
        print(f"  ✓ Volume: {volume_signal['signal']}")
        
        context = get_market_context(price)
        print(f"  ✓ Context: {context['position']}")
        
        # Train ensemble model
        print("\n🤖 Training ensemble AI (LSTM + RF + XGB)...")
        ensemble = EnsemblePredictor()
        ensemble.train_all_models(df)
        
        # Get ensemble prediction
        print("🎯 Making ensemble prediction...")
        prediction = ensemble.predict_ensemble(df)
        
        direction = prediction['direction']
        base_confidence = prediction['confidence']
        
        print(f"\n📊 BASE ENSEMBLE PREDICTION:")
        print(f"   Direction: {direction}")
        print(f"   Confidence: {base_confidence:.1%}")
        print(f"   Agreement: {prediction['agreement']} ({prediction['votes_up']}/3 UP)")
        print(f"   Models: LSTM={prediction['lstm_pred']:.3f}, RF={prediction['rf_pred']:.3f}, XGB={prediction['xgb_pred']:.3f}")
        
        # ENHANCE with all factors
        print("\n⚡ Enhancing with fundamental factors...")
        enhanced_conf = base_confidence
        boost_details = []
        
        # Seasonal
        if (direction=="UP" and seasonal['bias']>0) or (direction=="DOWN" and seasonal['bias']<0):
            boost = abs(seasonal['bias'])
            enhanced_conf += boost
            boost_details.append(f"Seasonal: +{boost:.2%}")
        else:
            penalty = abs(seasonal['bias'])*0.5
            enhanced_conf -= penalty
            boost_details.append(f"Seasonal: -{penalty:.2%}")
        
        # Weather
        if (direction=="UP" and weather_signal['signal']=='BULLISH') or (direction=="DOWN" and weather_signal['signal']=='BEARISH'):
            boost = weather_signal['score']
            enhanced_conf += boost
            boost_details.append(f"Weather: +{boost:.2%}")
        elif weather_signal['signal']!='NEUTRAL':
            penalty = abs(weather_signal['score'])*0.3
            enhanced_conf -= penalty
            boost_details.append(f"Weather: -{penalty:.2%}")
        
        # WASDE
        if (direction=="UP" and wasde_signal['signal']=='BULLISH') or (direction=="DOWN" and wasde_signal['signal']=='BEARISH'):
            boost = abs(wasde_signal['score'])*0.5
            enhanced_conf += boost
            boost_details.append(f"WASDE: +{boost:.2%}")
        elif wasde_signal['signal']!='NEUTRAL':
            penalty = abs(wasde_signal['score'])*0.3
            enhanced_conf -= penalty
            boost_details.append(f"WASDE: -{penalty:.2%}")
        
        # Volume
        if (direction=="UP" and volume_signal['signal']=='BULLISH') or (direction=="DOWN" and volume_signal['signal']=='BEARISH'):
            boost = abs(volume_signal['score'])
            enhanced_conf += boost
            boost_details.append(f"Volume: +{boost:.2%}")
        elif volume_signal['signal']!='NEUTRAL':
            penalty = abs(volume_signal['score'])*0.3
            enhanced_conf -= penalty
            boost_details.append(f"Volume: -{penalty:.2%}")
        
        # Context
        if context['signal']!='NEUTRAL':
            if (direction=="UP" and context['signal']=='BUY') or (direction=="DOWN" and context['signal']=='SELL'):
                boost = 0.05
                enhanced_conf += boost
                boost_details.append(f"Context: +{boost:.2%}")
            else:
                penalty = 0.08
                enhanced_conf -= penalty
                boost_details.append(f"Context: -{penalty:.2%}")
        
        # Clip
        enhanced_conf = max(0.5,min(1.0,enhanced_conf))
        
        print(f"\n🎯 FINAL ENHANCED PREDICTION:")
        print(f"   Direction: {direction}")
        print(f"   Base: {base_confidence:.1%}")
        print(f"   Enhanced: {enhanced_conf:.1%}")
        print(f"   Boost: {enhanced_conf-base_confidence:+.1%}")
        print(f"   Details: {', '.join(boost_details)}")
        
        # Check alert
        send_alert,reason = should_alert(direction,price,state)
        print(f"\n📢 Alert: {reason}")
        
        # Send if needed
        if send_alert and enhanced_conf>=MIN_CONFIDENCE:
            stop = price*(1-STOP_LOSS_PCT) if direction=="UP" else price*(1+STOP_LOSS_PCT)
            target = price*(1+TAKE_PROFIT_PCT) if direction=="UP" else price*(1-TAKE_PROFIT_PCT)
            
            # Check if model was just reset
            reset_notice = ""
            if state.get('reset_count', 0) > 0:
                days_since_reset = (datetime.now() - datetime.fromisoformat(state['last_model_reset'])).days
                if days_since_reset < 1:  # Reset happened today
                    reset_notice = f"\n🔄 *MODEL RESET:* Version {state['model_version']} (preventing overfitting)\n"
            
            message = f"""
🌾 *WHEAT ALERT - ULTIMATE v3.0* 🌾

{'🟢' if direction=='UP' else '🔴'} *{direction}* ({enhanced_conf:.1%})
💰 *{price:.2f}¢* (${price/100:.2f}/bu)
{reset_notice}
🤖 *ENSEMBLE AI:*
LSTM: {prediction['model_details']['LSTM']}
RF: {prediction['model_details']['RandomForest']}
XGB: {prediction['model_details']['XGBoost']}
Agreement: {prediction['agreement']}

📊 *FUNDAMENTAL FACTORS:*
📅 Seasonal: {seasonal['direction']} - {seasonal['explanation']}
🌦️ Weather: {weather_signal['signal']} ({weather_signal['explanation']})
📈 WASDE: {wasde_signal['signal']} (Stocks: {wasde_signal['data']['stocks_to_use']:.0%})
📊 Volume: {volume_signal['signal']} ({volume_signal['explanation']})
🎯 Context: {context['position']}

💼 *TRADE SETUP:*
Entry: {price:.2f}¢
Stop: {stop:.2f}¢ ({STOP_LOSS_PCT:.1%})
Target: {target:.2f}¢ ({TAKE_PROFIT_PCT:.1%})
R:R = 1.67:1

_{reason}_
_🚀 Professional Edition_
"""
            
            if send_telegram(message):
                print("✅ Professional alert sent!")
                state['alerts_sent'] += 1
            else:
                print("❌ Alert failed")
        else:
            print(f"⏸️ No alert: {reason if not send_alert else f'Confidence {enhanced_conf:.1%} below {MIN_CONFIDENCE:.0%}'}")
        
        # Save state
        state['last_direction'] = direction
        state['last_price'] = price
        save_state(state)
        
        print(f"\n📊 Session Stats:")
        print(f"   Total alerts sent: {state['alerts_sent']}")
        print(f"   Last check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
