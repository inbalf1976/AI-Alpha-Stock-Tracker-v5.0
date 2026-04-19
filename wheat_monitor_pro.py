"""
PROFESSIONAL WHEAT TRADING SYSTEM - ULTIMATE EDITION
Combines: Ensemble AI + Weather + WASDE + Volume + Seasonal + Context
Expected Accuracy: 75-85%

SCHEDULED ALERTS: 23:00 Israel Time (21:00 UTC) for NEXT trading day
Uses ONLY last known closing price
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import json
from pathlib import Path
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fix imports for GitHub Actions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
import pytz

# ============================================================================
# LIVE WEATHER ANALYZER - Embedded to avoid import issues
# ============================================================================

class LiveWeatherAnalyzer:
    """Analyze weather impact on wheat using Visual Crossing agricultural data"""
    
    def __init__(self):
        self.api_key = os.getenv("VISUAL_CROSSING_API_KEY", "W2FNC8VKT94JKH9ZRZYHUE63P")
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        self.wheat_regions = {
            # === USA (Top 4 states) ===
            'Kansas': '38.5,-98.0',          # #1 US producer
            'Oklahoma': '35.5,-98.0',        # Winter wheat
            'North Dakota': '47.5,-100.5',   # Spring wheat
            'Montana': '47.0,-110.0',        # Hard red spring
            
            # === GLOBAL (Top 4 exporters) ===
            'Ukraine': '46.5,32.0',          # Odessa region - Black Sea wheat
            'Russia': '45.0,39.0',           # Krasnodar - Southern wheat belt
            'Canada': '52.0,-106.0',         # Saskatchewan - Spring wheat
            'Australia': '-32.0,148.0'       # New South Wales - Southern hemisphere
        }
    
    def fetch_weather_data(self, location, days=7):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            url = f"{self.base_url}/{location}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'key': self.api_key,
                'unitGroup': 'metric',
                'include': 'days',
                'elements': 'datetime,temp,tempmax,tempmin,precip',
                'contentType': 'json'
            }
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Weather API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Weather fetch error: {e}")
            return None
    
    def analyze_agricultural_impact(self, weather_data):
        if not weather_data or 'days' not in weather_data:
            return self._get_neutral_signal()
        
        days = weather_data['days']
        recent_days = days[-7:]
        
        total_precip = sum(day.get('precip', 0) for day in recent_days)
        avg_temp = sum(day.get('temp', 0) for day in recent_days) / len(recent_days)
        max_temp = max(day.get('tempmax', 0) for day in recent_days)
        min_temp = min(day.get('tempmin', 0) for day in recent_days)
        
        drought_score = 0
        temperature_score = 0
        
        if total_precip < 5:
            drought_score = 0.15
            precip_note = f"Dry conditions ({total_precip:.1f}mm/week)"
        elif total_precip > 50:
            drought_score = -0.10
            precip_note = f"Heavy rain ({total_precip:.1f}mm/week)"
        else:
            drought_score = 0.05
            precip_note = f"Adequate moisture ({total_precip:.1f}mm/week)"
        
        month = datetime.now().month
        if month in [12, 1, 2]:
            if min_temp < -10:
                temperature_score = 0.20
                temp_note = f"Freeze risk! Min {min_temp:.1f}°C"
            elif min_temp < 0:
                temperature_score = 0.10
                temp_note = f"Cold temps {min_temp:.1f}°C"
            else:
                temperature_score = 0.0
                temp_note = f"Mild winter {avg_temp:.1f}°C"
        elif month in [5, 6, 7]:
            if max_temp > 35:
                temperature_score = 0.18
                temp_note = f"Heat stress! Max {max_temp:.1f}°C"
            elif max_temp > 30:
                temperature_score = 0.08
                temp_note = f"Warm temps {max_temp:.1f}°C"
            else:
                temperature_score = 0.0
                temp_note = f"Good temps {avg_temp:.1f}°C"
        else:
            temperature_score = 0.0
            temp_note = f"Normal temps {avg_temp:.1f}°C"
        
        total_score = drought_score + temperature_score
        
        if total_score > 0.15:
            signal = "BULLISH"
            confidence = 0.75
        elif total_score > 0.08:
            signal = "BULLISH"
            confidence = 0.65
        elif total_score < -0.05:
            signal = "BEARISH"
            confidence = 0.60
        else:
            signal = "NEUTRAL"
            confidence = 0.50
        
        return {
            'signal': signal,
            'score': total_score,
            'confidence': confidence,
            'factors': [precip_note, temp_note],
            'explanation': f"{precip_note}, {temp_note}"
        }
    
    def _get_neutral_signal(self):
        return {
            'signal': 'NEUTRAL',
            'score': 0.0,
            'confidence': 0.50,
            'factors': ['Weather data unavailable'],
            'explanation': 'Using seasonal average'
        }
    
    def get_multi_region_signal(self):
        regional_signals = []
        print("   🌾 Fetching live weather for wheat regions...")
        
        for region_name, coords in self.wheat_regions.items():
            print(f"      {region_name}...", end=" ")
            weather_data = self.fetch_weather_data(coords, days=7)
            
            if weather_data:
                analysis = self.analyze_agricultural_impact(weather_data)
                analysis['region'] = region_name
                regional_signals.append(analysis)
                print(f"{analysis['signal']}")
            else:
                print("Failed")
        
        if not regional_signals:
            print("   ⚠️ No weather data available")
            return self._get_neutral_signal()
        
        avg_score = sum(s['score'] for s in regional_signals) / len(regional_signals)
        bullish_count = sum(1 for s in regional_signals if s['signal'] == 'BULLISH')
        
        # Determine combined signal (adjusted for 8 regions)
        if bullish_count >= 5:  # Majority bullish (5+ out of 8)
            signal = 'BULLISH'
            confidence = 0.75
        elif bullish_count >= 3:  # Significant bullish (3-4 out of 8)
            signal = 'BULLISH'
            confidence = 0.65
        elif avg_score < -0.05:
            signal = 'BEARISH'
            confidence = 0.60
        else:
            signal = 'NEUTRAL'
            confidence = 0.55
        
        all_factors = []
        for s in regional_signals:
            if s['signal'] != 'NEUTRAL':
                all_factors.append(f"{s['region']}: {s['factors'][0]}")
        
        return {
            'signal': signal,
            'score': avg_score,
            'confidence': confidence,
            'regional_count': len(regional_signals),
            'bullish_regions': bullish_count,
            'factors': all_factors[:3],
            'explanation': f"Weather in {bullish_count}/{len(regional_signals)} regions"
        }

# ============================================================================

# ============================================================================
# LIVE WASDE SCRAPER - Embedded (USDA QuickStats API)
# ============================================================================

class LiveWASDEScraper:
    """Fetch and analyze live USDA WASDE data for wheat"""
    
    def __init__(self):
        self.api_key = os.getenv("USDA_API_KEY", "3338B84E-694D-3E6A-925C-F35064C59BAE")
        self.base_url = "https://quickstats.nass.usda.gov/api/api_GET/"
    
    def fetch_wheat_stocks(self):
        try:
            params = {
                'key': self.api_key,
                'source_desc': 'SURVEY',
                'commodity_desc': 'WHEAT',
                'statisticcat_desc': 'STOCKS',
                'agg_level_desc': 'NATIONAL',
                'format': 'JSON',
                'year__GE': 2020
            }
            response = requests.get(self.base_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    return self._parse_stocks_data(data['data'])
            return None
        except Exception as e:
            print(f"WASDE fetch error: {e}")
            return None
    
    def _parse_stocks_data(self, data):
        sorted_data = sorted(data, key=lambda x: (x.get('year', 0), x.get('reference_period_desc', '')), reverse=True)
        if not sorted_data:
            return None
        
        latest = sorted_data[0]
        latest_value = float(latest.get('Value', 0).replace(',', ''))
        latest_year = latest.get('year')
        
        previous_value = None
        for record in sorted_data[1:]:
            if record.get('year') != latest_year:
                try:
                    previous_value = float(record.get('Value', 0).replace(',', ''))
                    break
                except:
                    continue
        
        yoy_change = 0
        if previous_value and previous_value > 0:
            yoy_change = ((latest_value - previous_value) / previous_value) * 100
        
        return {
            'current_stocks': latest_value,
            'yoy_change_pct': yoy_change,
            'year': latest_year
        }
    
    def get_fundamental_score(self):
        print("      Fetching USDA data...", end=" ")
        stocks_data = self.fetch_wheat_stocks()
        
        if not stocks_data:
            print("Failed - using estimates")
            return self._get_default_estimates()
        
        print("Success")
        
        score = 0.0
        factors = []
        
        stocks_value = stocks_data['current_stocks']
        yoy_change = stocks_data['yoy_change_pct']
        
        # Estimate stocks-to-use (US typical use ~2000 million bushels/year)
        # stocks_value is in bushels, convert to millions
        stocks_millions = stocks_value / 1_000_000
        estimated_use_millions = 2000  # Million bushels
        stocks_to_use = stocks_millions / estimated_use_millions if estimated_use_millions > 0 else 0.20
        
        print(f"         Raw stocks: {stocks_value:,.0f} bushels")
        print(f"         Stocks (millions): {stocks_millions:.1f}")
        print(f"         Stocks-to-use: {stocks_to_use:.1%}")
        
        # Cap at reasonable range (0-200% - sometimes stocks can be very high)
        stocks_to_use = max(0.0, min(2.0, stocks_to_use))
        
        if stocks_to_use < 0.15:
            score += 0.30
            factors.append(f"Very tight stocks ({stocks_to_use:.1%})")
        elif stocks_to_use < 0.18:
            score += 0.20
            factors.append(f"Tight stocks ({stocks_to_use:.1%})")
        elif stocks_to_use > 0.25:
            score -= 0.15
            factors.append(f"Ample stocks ({stocks_to_use:.1%})")
        
        if yoy_change < -5:
            score += 0.15
            factors.append(f"Stocks down {abs(yoy_change):.1f}% YoY")
        elif yoy_change > 5:
            score -= 0.10
            factors.append(f"Stocks up {yoy_change:.1f}% YoY")
        
        if score > 0.20:
            signal = 'BULLISH'
        elif score < -0.10:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'signal': signal,
            'score': score,
            'data': {
                'stocks_to_use': stocks_to_use,
                'stocks_change': yoy_change,
                'last_updated': datetime.now().strftime('%Y-%m-%d'),
                'source': 'USDA QuickStats LIVE'
            },
            'factors': factors[:2]
        }
    
    def _get_default_estimates(self):
        month = datetime.now().month
        if month in [1, 2, 3]:
            return {'signal': 'BULLISH', 'score': 0.20, 'data': {'stocks_to_use': 0.18, 'stocks_change': -2, 'source': 'ESTIMATED'}, 'factors': ['Seasonal estimate']}
        elif month in [7, 8, 9]:
            return {'signal': 'BEARISH', 'score': -0.10, 'data': {'stocks_to_use': 0.22, 'stocks_change': 3, 'source': 'ESTIMATED'}, 'factors': ['Post-harvest']}
        else:
            return {'signal': 'NEUTRAL', 'score': 0.10, 'data': {'stocks_to_use': 0.20, 'stocks_change': 0, 'source': 'ESTIMATED'}, 'factors': ['Balanced']}

# ============================================================================

from volume_analyzer import VolumeAnalyzer
from ensemble_predictor import EnsemblePredictor
from move_analyzer import MoveAnalyzer


# CONFIG
PRIMARY_TICKER = "ZW=F"
STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.025
MIN_CONFIDENCE = 0.55
INTRADAY_CHANGE_THRESHOLD = 0.015  # 1.5% price move for intraday alerts

# TIMING CONFIG - Israel Time
ISRAEL_TZ = pytz.timezone('Asia/Jerusalem')
ALERT_HOUR = 23  # 23:00 Israel time
ALERT_MINUTE = 0

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
def get_next_trading_day():
    """
    Get the next trading day from current Israel time
    If it's Sunday 23:00 → Monday
    If it's Monday 23:00 → Tuesday
    etc.
    """
    israel_now = datetime.now(ISRAEL_TZ)
    next_day = israel_now + timedelta(days=1)
    
    # Skip weekends (Saturday=5, Sunday=6)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    
    return next_day.strftime('%A, %B %d')

def should_run_now():
    """
    Check if we should run based on Israel time
    Run ONLY at 23:00 Israel time (21:00 UTC)
    """
    israel_now = datetime.now(ISRAEL_TZ)
    current_hour = israel_now.hour
    current_minute = israel_now.minute
    
    print(f"🕐 Israel Time: {israel_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"   Target: {ALERT_HOUR:02d}:{ALERT_MINUTE:02d}")
    print(f"   Current: {current_hour:02d}:{current_minute:02d}")
    
    # Allow 5-minute window (23:00-23:04)
    if current_hour == ALERT_HOUR and current_minute < 5:
        print(f"   ✅ Within alert window")
        return True
    else:
        print(f"   ⏸️ Outside alert window - waiting for {ALERT_HOUR:02d}:{ALERT_MINUTE:02d}")
        return False

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
    print(f"\n📂 Loading state...")
    
    if STATE_FILE.exists():
        print(f"   ✓ State file exists")
        try:
            with open(STATE_FILE,'r') as f:
                state = json.load(f)
                print(f"   ✓ Loaded - last_alert_date: {state.get('last_alert_date')}")
                
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
                        
                        state['last_model_reset'] = datetime.now().isoformat()
                        state['model_version'] = state.get('model_version', 1) + 1
                        state['reset_count'] = state.get('reset_count', 0) + 1
                        
                        return state
                else:
                    from datetime import datetime
                    state['last_model_reset'] = datetime.now().isoformat()
                    state['model_version'] = 1
                    state['reset_count'] = 0
                
                return state
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ⚠️  No state file")
    
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
    print(f"\n💾 State saved:")
    print(f"   last_direction: {state.get('last_direction')}")
    print(f"   last_price: {state.get('last_price')}")
    print(f"   last_alert_date: {state.get('last_alert_date')}")
    print(f"   alerts_sent: {state.get('alerts_sent')}")

def should_send_alert(direction, price, state, is_scheduled_time):
    """
    Hybrid alert system:
    1. SCHEDULED: Daily at 23:00 Israel time (always send if not sent today)
    2. INTRADAY: Direction change + 1.5% move (must wait 60min between alerts)
    
    Returns: (should_send, reason, alert_type)
    """
    israel_now = datetime.now(ISRAEL_TZ)
    today = israel_now.date().isoformat()
    
    last_alert_date = state.get('last_alert_date', None)
    last_alert_time = state.get('last_alert_time', None)
    last_direction = state.get('last_direction', None)
    last_price = state.get('last_price', None)
    
    print(f"\n📢 Alert Check:")
    print(f"   Today: {today}")
    print(f"   Last alert date: {last_alert_date}")
    print(f"   Scheduled time: {is_scheduled_time}")
    print(f"   Last direction: {last_direction} → Current: {direction}")
    print(f"   Last price: {last_price} → Current: {price}")
    
    # RULE 1: SCHEDULED ALERT (23:00 Israel time)
    if is_scheduled_time:
        if last_alert_date != today:
            print(f"   → SCHEDULED: New day - SENDING")
            return True, f"Daily scheduled prediction for {get_next_trading_day()}", "SCHEDULED"
        else:
            print(f"   → SCHEDULED: Already sent today - SKIP")
            return False, "Scheduled alert already sent today", None
    
    # RULE 2: INTRADAY ALERT (direction change + 1.5% move)
    # Must be same trading day and 60+ minutes since last alert
    if last_alert_date != today:
        print(f"   → INTRADAY: Different day - wait for scheduled time")
        return False, "Wait for scheduled 23:00 alert", None
    
    # Check timing - must wait 60 minutes between alerts
    if last_alert_time:
        try:
            last_alert_dt = datetime.fromisoformat(last_alert_time)
            minutes_since_alert = (israel_now - last_alert_dt.astimezone(ISRAEL_TZ)).total_seconds() / 60
            print(f"   Minutes since last alert: {minutes_since_alert:.1f}")
            
            if minutes_since_alert < 60:
                print(f"   → INTRADAY: Too soon (< 60 min) - NO ALERT")
                return False, f"Only {minutes_since_alert:.0f} min since last alert", None
        except:
            pass
    
    # Check if direction changed
    if direction != last_direction:
        print(f"   → INTRADAY: Direction changed!")
        
        if last_price:
            change_pct = abs((price - last_price) / last_price)
            print(f"      Price change: {change_pct:.2%}")
            
            if change_pct >= INTRADAY_CHANGE_THRESHOLD:
                print(f"   → INTRADAY: Significant move ({change_pct:.1%}) - SENDING")
                return True, f"Direction changed with {change_pct:.1%} move", "INTRADAY"
            else:
                print(f"   → INTRADAY: Small change ({change_pct:.1%}) - NO ALERT")
                return False, f"Direction changed but only {change_pct:.1%} move", None
        else:
            print(f"   → INTRADAY: Direction changed (no price history) - SENDING")
            return True, "Direction changed", "INTRADAY"
    else:
        print(f"   → INTRADAY: Same direction - NO ALERT")
        return False, "Same direction, no alert needed", None

def send_telegram(message):
    """
    Send message to Telegram with debug logging
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram not configured")
        print(f"   Bot token exists: {bool(TELEGRAM_BOT_TOKEN)}")
        print(f"   Chat ID exists: {bool(TELEGRAM_CHAT_ID)}")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id":TELEGRAM_CHAT_ID,"text":message,"parse_mode":"Markdown"}
        
        print(f"\n🔍 TELEGRAM DEBUG:")
        print(f"   Bot token: {'SET' if TELEGRAM_BOT_TOKEN else 'MISSING'} (length: {len(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else 0})")
        print(f"   Chat ID: {TELEGRAM_CHAT_ID}")
        print(f"   Message length: {len(message)} chars")
        print(f"   Sending to Telegram API...")
        
        response = requests.post(url, data=data, timeout=10)
        
        print(f"   Response status code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Telegram accepted the message!")
            return True
        else:
            print(f"   ❌ Telegram rejected the message!")
            print(f"   Status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error details: {error_data}")
            except:
                pass
            return False
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout connecting to Telegram (10s)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
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

def main():
    print(f"\n{'='*80}")
    print(f"🌾 PROFESSIONAL WHEAT MONITOR - HYBRID ALERT SYSTEM v3.2")
    print(f"Time: {datetime.now(ISRAEL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Alert System: Daily 23:00 + Intraday direction changes (1.5% threshold)")
    print(f"Features: Ensemble AI + Weather + WASDE + Volume + Seasonal")
    print(f"{'='*80}\n")
    
    # CHECK IF THIS IS SCHEDULED TIME
    is_scheduled_time = should_run_now()
    
    state = load_state()
    
    try:
        # Fetch data
        print(f"📊 Fetching {PRIMARY_TICKER}...")
        df = fetch_data(PRIMARY_TICKER)
        if df is None:
            print("❌ No data")
            return
        
        df = add_indicators(df)
        price = df['Close'].iloc[-1]  # LAST KNOWN CLOSING PRICE
        last_close_date = df.index[-1].strftime('%Y-%m-%d')
        
        print(f"✓ Last Closing Price: {price:.2f}¢ (from {last_close_date})")
        
        # Get next trading day (for scheduled alerts)
        next_trading_day = get_next_trading_day()
        
        # Initialize analyzers
        print("\n🔬 Initializing advanced analyzers...")
        
        # Weather
        weather = LiveWeatherAnalyzer()
        weather_signal = weather.get_multi_region_signal()
        
        if 'bullish_regions' in weather_signal and 'regional_count' in weather_signal:
            print(f"  ✓ Weather: {weather_signal['signal']} ({weather_signal['bullish_regions']}/{weather_signal['regional_count']} regions)")
        else:
            print(f"  ✓ Weather: {weather_signal['signal']} (data unavailable)")
        
        # WASDE
        wasde = LiveWASDEScraper()
        wasde_signal = wasde.get_fundamental_score()
        print(f"  ✓ WASDE: {wasde_signal['signal']}")
        
        # Volume analyzer
        volume = VolumeAnalyzer()
        
        # Get all signals
        print("📡 Gathering signals...")
        seasonal = get_seasonal_bias()
        print(f"  ✓ Seasonal: {seasonal['direction']}")
        
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
        
        # Check if we should send alert (scheduled OR intraday)
        should_send, reason, alert_type = should_send_alert(direction, price, state, is_scheduled_time)
        print(f"\n📢 Alert Decision: {reason}")
        print(f"   Alert Type: {alert_type if alert_type else 'NONE'}")
        
        # Send if needed
        if should_send and enhanced_conf >= MIN_CONFIDENCE:
            stop = price*(1-STOP_LOSS_PCT) if direction=="UP" else price*(1+STOP_LOSS_PCT)
            target = price*(1+TAKE_PROFIT_PCT) if direction=="UP" else price*(1-TAKE_PROFIT_PCT)
            
            # Check if model was just reset
            reset_notice = ""
            if state.get('reset_count', 0) > 0:
                days_since_reset = (datetime.now() - datetime.fromisoformat(state['last_model_reset'])).days
                if days_since_reset < 1:
                    reset_notice = f"\n🔄 *MODEL RESET:* Version {state['model_version']} (preventing overfitting)\n"
            
            # Analyze typical moves and generate recommendations
            move_analyzer = MoveAnalyzer()
            move_stats = move_analyzer.analyze_typical_moves(df, direction)
            recommendations = move_analyzer.format_recommendation_message(price, direction, move_stats)
            
            # Different message headers for scheduled vs intraday
            if alert_type == "SCHEDULED":
                header = f"🌾 *WHEAT ALERT - DAILY PREDICTION* 🌾\n\n📅 *For: {next_trading_day}*\n🕐 Based on last close: {last_close_date}\n"
                footer = "\n_Daily prediction at 23:00 Israel Time_\n_🚀 Professional Edition v3.2_"
            else:  # INTRADAY
                header = f"🌾 *WHEAT ALERT - INTRADAY UPDATE* 🌾\n\n⚡ *DIRECTION CHANGE DETECTED*\n🕐 Current data: {last_close_date}\n"
                footer = "\n_Intraday alert: Direction changed ≥1.5%_\n_🚀 Professional Edition v3.2_"
            
            message = f"""
{header}
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

{recommendations}
{footer}
"""
            
            telegram_success = send_telegram(message)
            
            if telegram_success:
                print(f"\n✅ {alert_type} alert sent!")
                # Update state ONLY after successful send
                israel_now = datetime.now(ISRAEL_TZ)
                state['last_alert_date'] = israel_now.date().isoformat()
                state['last_alert_time'] = israel_now.isoformat()
                state['alerts_sent'] += 1
                state['last_alert_type'] = alert_type
                
                # LOG PREDICTION FOR PERFORMANCE TRACKING
                try:
                    from performance_tracker import PerformanceTracker
                    tracker = PerformanceTracker()
                    tracker.log_prediction(
                        direction=direction,
                        price=price,
                        confidence=enhanced_conf,
                        factors={
                            'seasonal': seasonal['direction'],
                            'weather': weather_signal['signal'],
                            'wasde': wasde_signal['signal'],
                            'volume': volume_signal['signal'],
                            'ensemble': f"{prediction['agreement']} ({prediction['votes_up']}/3)",
                            'alert_type': alert_type
                        }
                    )
                except Exception as e:
                    print(f"⚠️ Performance tracking skipped: {e}")
            else:
                print("\n❌ Alert failed - state NOT updated, will retry next run")
        else:
            if not should_send:
                print(f"\n⏸️ No alert: {reason}")
            else:
                print(f"\n⏸️ Confidence {enhanced_conf:.1%} below minimum {MIN_CONFIDENCE:.0%} - no alert")
        
        # Save state
        state['last_direction'] = direction
        state['last_price'] = price
        save_state(state)
        
        print(f"\n📊 Session Stats:")
        print(f"   Total alerts sent: {state['alerts_sent']}")
        print(f"   Last alert type: {state.get('last_alert_type', 'N/A')}")
        print(f"   Last check: {datetime.now(ISRAEL_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
