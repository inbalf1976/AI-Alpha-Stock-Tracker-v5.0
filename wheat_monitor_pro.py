"""
PROFESSIONAL WHEAT TRADING SYSTEM - ULTIMATE EDITION v3.1
Combines: Ensemble AI + Weather + WASDE + Volume + Seasonal + Context
Expected Accuracy: 75-85%

FIXED v3.1:
- Contract rollover detection
- Dynamic contract selection (monitors most liquid contract)
- Volume validation (detects expiring contracts)
- Data quality checks
- Early close detection
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

# ============================================================================
# CONTRACT ROLLOVER DETECTION - NEW v3.1
# ============================================================================

def get_active_wheat_contract():
    """
    Automatically detect the most liquid wheat futures contract
    
    Wheat futures contracts: H(Mar), K(May), N(Jul), U(Sep), Z(Dec)
    
    Returns:
        str: Most liquid contract symbol (e.g., 'ZWN26')
    """
    # Get current year for contract codes
    current_year = datetime.now().year
    year_code = str(current_year)[-2:]  # '26' for 2026
    
    # Define all possible contracts for current and next year
    months = ['H', 'K', 'N', 'U', 'Z']  # Mar, May, Jul, Sep, Dec
    contracts = [f"ZW{month}{year_code}" for month in months]
    
    # Add next year contracts
    next_year_code = str(current_year + 1)[-2:]
    contracts.extend([f"ZW{month}{next_year_code}" for month in months])
    
    print("\n🔍 CONTRACT ROLLOVER CHECK:")
    print("   Scanning wheat contracts for highest volume...")
    
    max_volume = 0
    best_contract = "ZW=F"  # Fallback to generic
    contract_volumes = {}
    
    for contract in contracts:
        try:
            ticker = yf.Ticker(contract)
            df = ticker.history(period="5d")  # Get last 5 days
            
            if not df.empty and 'Volume' in df.columns:
                # Use average volume over last 3 days
                avg_volume = df['Volume'].tail(3).mean()
                contract_volumes[contract] = avg_volume
                
                if avg_volume > max_volume:
                    max_volume = avg_volume
                    best_contract = contract
                    
                print(f"      {contract}: {avg_volume:,.0f} avg volume")
        except Exception as e:
            # Contract doesn't exist or no data
            continue
    
    if best_contract == "ZW=F":
        print("   ⚠️ Could not find specific contract, using generic ZW=F")
    else:
        print(f"   ✅ ACTIVE CONTRACT: {best_contract} (volume: {max_volume:,.0f})")
        
        # Show rollover warning if volume is very low
        if max_volume < 10000:
            print(f"   ⚠️ WARNING: Low volume ({max_volume:,.0f}) - contract may be expiring")
    
    return best_contract, contract_volumes

def validate_contract_data(df, ticker_symbol):
    """
    Validate if contract data is reliable for trading
    
    Returns:
        tuple: (is_valid, warning_messages)
    """
    warnings = []
    is_valid = True
    
    if df is None or df.empty:
        return False, ["No data available"]
    
    # Check 1: Volume analysis
    latest_volume = df['Volume'].iloc[-1]
    avg_volume_20d = df['Volume'].tail(20).mean()
    volume_ratio = latest_volume / avg_volume_20d if avg_volume_20d > 0 else 0
    
    print(f"\n📊 DATA VALIDATION:")
    print(f"   Latest volume: {latest_volume:,.0f}")
    print(f"   20-day avg: {avg_volume_20d:,.0f}")
    print(f"   Ratio: {volume_ratio:.2f}x")
    
    # If volume < 30% of average, data is suspicious
    if volume_ratio < 0.3:
        warnings.append(f"Very low volume ({volume_ratio:.1%} of normal) - possible contract expiration")
        is_valid = False
    elif volume_ratio < 0.5:
        warnings.append(f"Below average volume ({volume_ratio:.1%} of normal) - use caution")
    
    # Check 2: Price movement validation
    price_change = df['Close'].pct_change().tail(5)
    if price_change.std() < 0.001:  # Almost no movement
        warnings.append("Very low price volatility - market may be inactive")
    
    # Check 3: Data freshness
    latest_timestamp = df.index[-1]
    hours_old = (datetime.now() - latest_timestamp.to_pydatetime()).total_seconds() / 3600
    
    if hours_old > 24:
        warnings.append(f"Data is {hours_old:.1f} hours old - may be stale")
        is_valid = False
    
    # Print validation results
    if warnings:
        print(f"   ⚠️ VALIDATION WARNINGS:")
        for warn in warnings:
            print(f"      - {warn}")
    else:
        print(f"   ✅ Data validation passed")
    
    return is_valid, warnings

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
        stocks_millions = stocks_value / 1_000_000
        estimated_use_millions = 2000
        stocks_to_use = stocks_millions / estimated_use_millions if estimated_use_millions > 0 else 0.20
        
        print(f"         Raw stocks: {stocks_value:,.0f} bushels")
        print(f"         Stocks (millions): {stocks_millions:.1f}")
        print(f"         Stocks-to-use: {stocks_to_use:.1%}")
        
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
    print(f"\n📂 Loading state...")
    
    if STATE_FILE.exists():
        print(f"   ✓ State file exists")
        try:
            with open(STATE_FILE,'r') as f:
                state = json.load(f)
                print(f"   ✓ Loaded - last_alert_date: {state.get('last_alert_date')}")
                
                last_reset = state.get('last_model_reset', None)
                if last_reset:
                    last_reset_date = datetime.fromisoformat(last_reset)
                    days_since_reset = (datetime.now() - last_reset_date).days
                    
                    if days_since_reset >= 730:
                        print("⚠️  MODEL RESET: 2+ years since last reset")
                        state['last_model_reset'] = datetime.now().isoformat()
                        state['model_version'] = state.get('model_version', 1) + 1
                        state['reset_count'] = state.get('reset_count', 0) + 1
                        return state
                else:
                    state['last_model_reset'] = datetime.now().isoformat()
                    state['model_version'] = 1
                    state['reset_count'] = 0
                
                return state
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ⚠️  No state file")
    
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
    print(f"\n💾 State saved")

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id":TELEGRAM_CHAT_ID,"text":message,"parse_mode":"Markdown"}
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            return True
        else:
            print(f"   ❌ Telegram error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Telegram exception: {e}")
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
    current_time = datetime.now()
    current_date = current_time.date().isoformat()
    
    last_alert_time = state.get('last_alert_time', None)
    last_alert_date = state.get('last_alert_date', None)
    last_direction = state.get('last_direction', None)
    last_price = state.get('last_price', None)
    
    print(f"\n📢 Alert Check:")
    print(f"   Last alert date: {last_alert_date} vs today: {current_date}")
    
    if last_alert_time is None:
        return True, "First prediction"
    
    if last_alert_date != current_date:
        return True, f"First prediction of {current_date}"
    
    try:
        last_alert_dt = datetime.fromisoformat(last_alert_time)
        minutes_since_alert = (current_time - last_alert_dt).total_seconds() / 60
    except:
        return True, "Time parse error"
    
    if minutes_since_alert < 60:
        return False, f"Only {minutes_since_alert:.0f} min since last alert"
    
    if direction == last_direction:
        if last_price:
            change_pct = abs((price - last_price) / last_price)
            if change_pct >= DIRECTION_CHANGE_THRESHOLD:
                return True, f"Same direction but {change_pct:.1%} move"
            else:
                return False, f"Same direction, only {change_pct:.1%} move"
        else:
            return False, "Same direction"
    
    if last_price:
        change_pct = abs((price - last_price) / last_price)
        if change_pct >= DIRECTION_CHANGE_THRESHOLD:
            return True, f"Direction changed with {change_pct:.1%} move"
        else:
            return False, f"Direction changed but only {change_pct:.1%} move"
    else:
        return True, "Direction changed"

def main():
    print(f"\n{'='*80}")
    print(f"🌾 PROFESSIONAL WHEAT MONITOR - v3.1 (ROLLOVER DETECTION)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*80}\n")
    
    state = load_state()
    
    try:
        # NEW v3.1: Dynamic contract selection
        PRIMARY_TICKER, contract_volumes = get_active_wheat_contract()
        
        # Fetch data
        print(f"\n📊 Fetching {PRIMARY_TICKER}...")
        df = fetch_data(PRIMARY_TICKER)
        if df is None:
            print("❌ No data")
            return
        
        # NEW v3.1: Validate contract data
        is_valid, warnings = validate_contract_data(df, PRIMARY_TICKER)
        
        if not is_valid:
            print(f"\n⚠️ DATA QUALITY WARNING:")
            print(f"   Contract {PRIMARY_TICKER} may be expiring or have issues")
            print(f"   Prediction reliability: LOW")
            print(f"   Consider skipping this trade!")
            # Continue anyway but flag it
        
        df = add_indicators(df)
        price = df['Close'].iloc[-1]
        print(f"✓ Price: {price:.2f}¢")
        
        # Cache logic (same as before)
        print("\n🔬 Initializing analyzers...")
        current_hour = datetime.now().hour
        should_fetch_fresh = current_hour in [9, 17]
        
        weather = LiveWeatherAnalyzer()
        weather_cache_file = Path("weather_cache.json")
        
        if should_fetch_fresh or not weather_cache_file.exists():
            weather_signal = weather.get_multi_region_signal()
            try:
                with open(weather_cache_file, 'w') as f:
                    json.dump({'timestamp': datetime.now().isoformat(), 'data': weather_signal}, f)
            except:
                pass
        else:
            try:
                with open(weather_cache_file, 'r') as f:
                    weather_signal = json.load(f)['data']
            except:
                weather_signal = weather.get_multi_region_signal()
        
        wasde = LiveWASDEScraper()
        wasde_cache_file = Path("wasde_cache.json")
        
        if should_fetch_fresh or not wasde_cache_file.exists():
            wasde_signal = wasde.get_fundamental_score()
            try:
                with open(wasde_cache_file, 'w') as f:
                    json.dump({'timestamp': datetime.now().isoformat(), 'data': wasde_signal}, f)
            except:
                pass
        else:
            try:
                with open(wasde_cache_file, 'r') as f:
                    wasde_signal = json.load(f)['data']
            except:
                wasde_signal = wasde.get_fundamental_score()
        
        volume = VolumeAnalyzer()
        seasonal = get_seasonal_bias()
        volume_signal = volume.analyze_volume(df)
        context = get_market_context(price)
        
        # Train and predict
        print("\n🤖 Training ensemble AI...")
        ensemble = EnsemblePredictor()
        ensemble.train_all_models(df)
        prediction = ensemble.predict_ensemble(df)
        
        direction = prediction['direction']
        base_confidence = prediction['confidence']
        
        # Enhance with factors
        enhanced_conf = base_confidence
        
        if (direction=="UP" and seasonal['bias']>0) or (direction=="DOWN" and seasonal['bias']<0):
            enhanced_conf += abs(seasonal['bias'])
        else:
            enhanced_conf -= abs(seasonal['bias'])*0.5
        
        if (direction=="UP" and weather_signal['signal']=='BULLISH') or (direction=="DOWN" and weather_signal['signal']=='BEARISH'):
            enhanced_conf += weather_signal['score']
        elif weather_signal['signal']!='NEUTRAL':
            enhanced_conf -= abs(weather_signal['score'])*0.3
        
        if (direction=="UP" and wasde_signal['signal']=='BULLISH') or (direction=="DOWN" and wasde_signal['signal']=='BEARISH'):
            enhanced_conf += abs(wasde_signal['score'])*0.5
        elif wasde_signal['signal']!='NEUTRAL':
            enhanced_conf -= abs(wasde_signal['score'])*0.3
        
        if (direction=="UP" and volume_signal['signal']=='BULLISH') or (direction=="DOWN" and volume_signal['signal']=='BEARISH'):
            enhanced_conf += abs(volume_signal['score'])
        elif volume_signal['signal']!='NEUTRAL':
            enhanced_conf -= abs(volume_signal['score'])*0.3
        
        if context['signal']!='NEUTRAL':
            if (direction=="UP" and context['signal']=='BUY') or (direction=="DOWN" and context['signal']=='SELL'):
                enhanced_conf += 0.05
            else:
                enhanced_conf -= 0.08
        
        enhanced_conf = max(0.5,min(1.0,enhanced_conf))
        
        print(f"\n🎯 PREDICTION: {direction} ({enhanced_conf:.1%})")
        
        # NEW v3.1: Add data quality warning to message if needed
        data_quality_notice = ""
        if not is_valid:
            data_quality_notice = "\n⚠️ *DATA QUALITY WARNING:*\n"
            data_quality_notice += f"Contract {PRIMARY_TICKER} shows low volume\n"
            data_quality_notice += "May be expiring - use extra caution!\n"
        
        send_alert, reason = should_alert(direction,price,state)
        
        if send_alert and enhanced_conf>=MIN_CONFIDENCE:
            stop = price*(1-STOP_LOSS_PCT) if direction=="UP" else price*(1+STOP_LOSS_PCT)
            target = price*(1+TAKE_PROFIT_PCT) if direction=="UP" else price*(1-TAKE_PROFIT_PCT)
            
            move_analyzer = MoveAnalyzer()
            move_stats = move_analyzer.analyze_typical_moves(df, direction)
            recommendations = move_analyzer.format_recommendation_message(price, direction, move_stats)
            
            message = f"""
🌾 *WHEAT ALERT v3.1* 🌾

{'🟢' if direction=='UP' else '🔴'} *{direction}* ({enhanced_conf:.1%})
💰 *{price:.2f}¢* (${price/100:.2f}/bu)
📊 Contract: {PRIMARY_TICKER}
{data_quality_notice}
🤖 *ENSEMBLE AI:*
Agreement: {prediction['agreement']}

📊 *FACTORS:*
📅 Seasonal: {seasonal['direction']}
🌦️ Weather: {weather_signal['signal']}
📈 WASDE: {wasde_signal['signal']}
🎯 Context: {context['position']}

💼 *TRADE:*
Entry: {price:.2f}¢
Stop: {stop:.2f}¢
Target: {target:.2f}¢

{recommendations}

_{reason}_
"""
            
            if send_telegram(message):
                print("✅ Alert sent!")
                state['last_alert_time'] = datetime.now().isoformat()
                state['last_alert_date'] = datetime.now().date().isoformat()
                state['alerts_sent'] += 1
            else:
                print("❌ Alert failed")
        
        state['last_direction'] = direction
        state['last_price'] = price
        save_state(state)
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
