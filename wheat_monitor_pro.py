"""
PROFESSIONAL WHEAT TRADING SYSTEM - ULTIMATE EDITION
Combines: Ensemble AI + Weather + WASDE + Volume + Seasonal + Context

CHANGES FROM ORIGINAL:
1. Incomplete candle fix in fetch_data()
2. Israel timezone slot-based alerting (1AM and 4PM Israel)
3. FORCE_ALERT for manual triggers
4. Double stop loss in message (1.5% tight + 2.5% wide)
5. Auto stop recommendation based on volume
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

# ============================================================================
# LIVE WEATHER ANALYZER
# ============================================================================

class LiveWeatherAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("VISUAL_CROSSING_API_KEY", "W2FNC8VKT94JKH9ZRZYHUE63P")
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        self.wheat_regions = {
            'Kansas': '38.5,-98.0',
            'Oklahoma': '35.5,-98.0',
            'North Dakota': '47.5,-100.5',
            'Montana': '47.0,-110.0',
            'Ukraine': '46.5,32.0',
            'Russia': '45.0,39.0',
            'Canada': '52.0,-106.0',
            'Australia': '-32.0,148.0'
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
                temp_note = f"Freeze risk! Min {min_temp:.1f}C"
            elif min_temp < 0:
                temperature_score = 0.10
                temp_note = f"Cold temps {min_temp:.1f}C"
            else:
                temperature_score = 0.0
                temp_note = f"Mild winter {avg_temp:.1f}C"
        elif month in [5, 6, 7]:
            if max_temp > 35:
                temperature_score = 0.18
                temp_note = f"Heat stress! Max {max_temp:.1f}C"
            elif max_temp > 30:
                temperature_score = 0.08
                temp_note = f"Warm temps {max_temp:.1f}C"
            else:
                temperature_score = 0.0
                temp_note = f"Good temps {avg_temp:.1f}C"
        else:
            temperature_score = 0.0
            temp_note = f"Normal temps {avg_temp:.1f}C"

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
        print("   Fetching live weather for wheat regions...")

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
            print("   No weather data available")
            return self._get_neutral_signal()

        avg_score = sum(s['score'] for s in regional_signals) / len(regional_signals)
        bullish_count = sum(1 for s in regional_signals if s['signal'] == 'BULLISH')

        if bullish_count >= 5:
            signal = 'BULLISH'
            confidence = 0.75
        elif bullish_count >= 3:
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
# LIVE WASDE SCRAPER
# ============================================================================

class LiveWASDEScraper:
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
PRIMARY_TICKER = "ZW=F"
STOP_LOSS_PCT = 0.015
STOP_LOSS_WIDE_PCT = 0.025
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

# ============================================================================
# HELPERS
# ============================================================================

def get_israel_time():
    """Get current Israel time, auto-adjusting for DST"""
    now_utc = datetime.utcnow()
    year = now_utc.year

    # DST start: last Friday before April 2
    april2 = datetime(year, 4, 2)
    days_to_friday = (april2.weekday() - 4) % 7
    dst_start = april2 - timedelta(days=days_to_friday)
    dst_start = dst_start.replace(hour=0, minute=0, second=0)

    # DST end: last Sunday before October 10
    oct10 = datetime(year, 10, 10)
    days_to_sunday = (oct10.weekday() - 6) % 7
    dst_end = oct10 - timedelta(days=days_to_sunday)
    dst_end = dst_end.replace(hour=0, minute=0, second=0)

    if dst_start <= now_utc < dst_end:
        offset = 3
        tz_name = "IDT (UTC+3)"
    else:
        offset = 2
        tz_name = "IST (UTC+2)"

    israel_time = now_utc + timedelta(hours=offset)
    return israel_time, offset, tz_name


def get_seasonal_bias():
    month = datetime.now().month
    bias = SEASONAL_BIAS.get(month, 0.0)
    explanations = {
        1:"Neutral", 2:"Pre-spring", 3:"Spring rally", 4:"Peak planting",
        5:"Max premium", 6:"Harvest (LOW)", 7:"Post-harvest", 8:"Summer lull",
        9:"Fall recovery", 10:"Winter demand", 11:"Pre-winter", 12:"Winter (HIGH)"
    }
    direction = 'BULLISH' if bias > 0.02 else 'BEARISH' if bias < -0.02 else 'NEUTRAL'
    return {'bias': bias, 'direction': direction, 'explanation': explanations.get(month, "")}


def get_market_context(price):
    if price < NORMAL_RANGE_LOW:
        return {'position': 'BELOW_NORMAL', 'signal': 'BUY'}
    elif price > NORMAL_RANGE_HIGH:
        return {'position': 'ABOVE_NORMAL', 'signal': 'SELL'}
    else:
        return {'position': 'NORMAL', 'signal': 'NEUTRAL'}


def load_state():
    print(f"\nLoading state...")
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                print(f"   Loaded - last_alert_date: {state.get('last_alert_date')}")
                last_reset = state.get('last_model_reset', None)
                if last_reset:
                    last_reset_date = datetime.fromisoformat(last_reset)
                    days_since_reset = (datetime.now() - last_reset_date).days
                    if days_since_reset >= 730:
                        state['last_model_reset'] = datetime.now().isoformat()
                        state['model_version'] = state.get('model_version', 1) + 1
                        state['reset_count'] = state.get('reset_count', 0) + 1
                else:
                    state['last_model_reset'] = datetime.now().isoformat()
                    state['model_version'] = 1
                    state['reset_count'] = 0
                return state
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"   No state file")

    return {
        'last_direction': None,
        'last_price': None,
        'alerts_sent': 0,
        'last_model_reset': datetime.now().isoformat(),
        'model_version': 1,
        'reset_count': 0,
        'alerts_today': {}
    }


def save_state(state):
    state['last_check'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"\nState saved: direction={state.get('last_direction')} price={state.get('last_price')} alerts={state.get('alerts_sent')}")


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        print(f"\nTELEGRAM: sending {len(message)} chars...")
        response = requests.post(url, data=data, timeout=10)
        print(f"   Response: {response.status_code}")
        print(f"   Body: {response.text}")
        if response.status_code == 200:
            print("   Telegram accepted!")
            return True
        else:
            print("   Telegram rejected!")
            return False
    except Exception as e:
        print(f"   Telegram error: {e}")
        return False


def fetch_data(ticker, days=730):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            return None

        # ALWAYS drop the last candle if market is currently open
        # Wheat futures market hours: 19:00 - 13:30 CT (next day)
        # = 00:00 - 18:30 UTC
        # At 1AM Israel (22:00-23:00 UTC) market IS open
        # So we ALWAYS drop the last candle to ensure we use previous day close
        now_utc = datetime.utcnow()
        last_date = df.index[-1].date()
        today_utc = now_utc.date()
        yesterday_utc = (now_utc - timedelta(days=1)).date()

        # Drop last candle if it's today OR yesterday but market was open
        # Safe rule: always use the candle from 2+ days ago during market hours
        # OR use yesterday if we're running after market close (after 20:00 UTC)
        market_closed_for_day = now_utc.hour >= 20  # After 20:00 UTC market day is done

        if last_date >= today_utc:
            df = df.iloc[:-1]
            print(f"   Dropped today's candle - using {df.index[-1].date()}")
        elif last_date == yesterday_utc and not market_closed_for_day:
            df = df.iloc[:-1]
            print(f"   Dropped yesterday's incomplete candle (market still open) - using {df.index[-1].date()}")
        else:
            print(f"   Using candle: {last_date}")

        print(f"   Previous day CLOSE: {df['Close'].iloc[-1]:.2f}c")
        return df
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


def should_alert(direction, price, state):
    """Slot-based alerting: send at 01:00 and 16:00 Israel time only.
    Manual triggers (workflow_dispatch) always send immediately."""

    # Check for manual/force trigger
    force_alert = os.getenv('FORCE_ALERT', '').lower() in ('true', '1', 'yes')
    github_event = os.getenv('GITHUB_EVENT_NAME', '')
    is_manual = force_alert or 'workflow_dispatch' in github_event

    print(f"\nAlert Check:")
    print(f"   FORCE_ALERT={force_alert}, GITHUB_EVENT_NAME={github_event}")

    if is_manual:
        print(f"   Manual trigger - sending immediately")
        return True, "Manual alert"

    # Get Israel time
    israel_time, utc_offset, tz_name = get_israel_time()
    israel_hour = israel_time.hour
    israel_date = israel_time.date().isoformat()

    print(f"   Israel time: {israel_time.strftime('%Y-%m-%d %H:%M')} {tz_name}")
    print(f"   Israel hour: {israel_hour}:00")

    # Determine slot - MORNING ONLY (1AM Israel)
    if israel_hour in (1, 2):
        slot = 'morning'
        slot_label = f"Morning Alert (01:00 {tz_name})"
    else:
        print(f"   Not a scheduled hour ({israel_hour}:00 Israel) - NO ALERT")
        print(f"   Scheduled hour: 01:00 Israel time only")
        return False, f"Not scheduled hour ({israel_hour}:00 Israel)"

    # Check if slot already sent today
    alerts_today = state.get('alerts_today', {})
    slot_key = f"{israel_date}_{slot}"
    if alerts_today.get(slot_key, False):
        print(f"   {slot_label} already sent today - NO ALERT")
        return False, f"{slot_label} already sent"

    print(f"   {slot_label} - SENDING")
    return True, slot_label


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print(f"WHEAT MONITOR - ULTIMATE EDITION v3.0")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*80}\n")

    state = load_state()

    try:
        print(f"Fetching {PRIMARY_TICKER}...")
        df = fetch_data(PRIMARY_TICKER)
        if df is None:
            print("No data")
            return

        df = add_indicators(df)
        price = df['Close'].iloc[-1]
        print(f"Price: {price:.2f}c")

        print("\nInitializing analyzers...")
        current_hour = datetime.now().hour
        should_fetch_fresh = current_hour in [9, 17]

        if should_fetch_fresh:
            print("Fetching FRESH weather & WASDE data")
        else:
            print("Using CACHED weather & WASDE data")

        # Weather
        weather = LiveWeatherAnalyzer()
        weather_cache_file = Path("weather_cache.json")
        if should_fetch_fresh or not weather_cache_file.exists():
            weather_signal = weather.get_multi_region_signal()
            try:
                with open(weather_cache_file, 'w') as f:
                    json.dump({'timestamp': datetime.now().isoformat(), 'data': weather_signal}, f)
                print("  Weather data cached")
            except:
                pass
        else:
            try:
                with open(weather_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    weather_signal = cache_data['data']
                    cache_age = datetime.now() - datetime.fromisoformat(cache_data['timestamp'])
                    print(f"  Using cached weather (age: {cache_age.seconds//3600}h)")
            except:
                weather_signal = weather.get_multi_region_signal()

        if 'bullish_regions' in weather_signal:
            print(f"  Weather: {weather_signal['signal']} ({weather_signal['bullish_regions']}/{weather_signal['regional_count']} regions)")
        else:
            print(f"  Weather: {weather_signal['signal']}")

        # WASDE
        wasde = LiveWASDEScraper()
        wasde_cache_file = Path("wasde_cache.json")
        if should_fetch_fresh or not wasde_cache_file.exists():
            wasde_signal = wasde.get_fundamental_score()
            try:
                with open(wasde_cache_file, 'w') as f:
                    json.dump({'timestamp': datetime.now().isoformat(), 'data': wasde_signal}, f)
                print("  WASDE data cached")
            except:
                pass
        else:
            try:
                with open(wasde_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    wasde_signal = cache_data['data']
                    cache_age = datetime.now() - datetime.fromisoformat(cache_data['timestamp'])
                    print(f"  Using cached WASDE (age: {cache_age.seconds//3600}h)")
            except:
                wasde_signal = wasde.get_fundamental_score()

        print(f"  WASDE: {wasde_signal['signal']}")

        volume = VolumeAnalyzer()

        print("Gathering signals...")
        seasonal = get_seasonal_bias()
        print(f"  Seasonal: {seasonal['direction']}")
        volume_signal = volume.analyze_volume(df)
        print(f"  Volume: {volume_signal['signal']}")
        context = get_market_context(price)
        print(f"  Context: {context['position']}")

        print("\nTraining ensemble AI (LSTM + RF + XGB)...")
        ensemble = EnsemblePredictor()
        ensemble.train_all_models(df)

        print("Making ensemble prediction...")
        prediction = ensemble.predict_ensemble(df)

        direction = prediction['direction']
        base_confidence = prediction['confidence']

        print(f"\nBASE PREDICTION: {direction} ({base_confidence:.1%})")
        print(f"   Agreement: {prediction['agreement']} ({prediction['votes_up']}/3 UP)")
        print(f"   Models: LSTM={prediction['lstm_pred']:.3f}, RF={prediction['rf_pred']:.3f}, XGB={prediction['xgb_pred']:.3f}")

        # Enhance with fundamentals
        print("\nEnhancing with fundamental factors...")
        enhanced_conf = base_confidence
        boost_details = []

        if (direction == "UP" and seasonal['bias'] > 0) or (direction == "DOWN" and seasonal['bias'] < 0):
            boost = abs(seasonal['bias'])
            enhanced_conf += boost
            boost_details.append(f"Seasonal: +{boost:.2%}")
        else:
            penalty = abs(seasonal['bias']) * 0.5
            enhanced_conf -= penalty
            boost_details.append(f"Seasonal: -{penalty:.2%}")

        if (direction == "UP" and weather_signal['signal'] == 'BULLISH') or (direction == "DOWN" and weather_signal['signal'] == 'BEARISH'):
            boost = weather_signal['score']
            enhanced_conf += boost
            boost_details.append(f"Weather: +{boost:.2%}")
        elif weather_signal['signal'] != 'NEUTRAL':
            penalty = abs(weather_signal['score']) * 0.3
            enhanced_conf -= penalty
            boost_details.append(f"Weather: -{penalty:.2%}")

        if (direction == "UP" and wasde_signal['signal'] == 'BULLISH') or (direction == "DOWN" and wasde_signal['signal'] == 'BEARISH'):
            boost = abs(wasde_signal['score']) * 0.5
            enhanced_conf += boost
            boost_details.append(f"WASDE: +{boost:.2%}")
        elif wasde_signal['signal'] != 'NEUTRAL':
            penalty = abs(wasde_signal['score']) * 0.3
            enhanced_conf -= penalty
            boost_details.append(f"WASDE: -{penalty:.2%}")

        if (direction == "UP" and volume_signal['signal'] == 'BULLISH') or (direction == "DOWN" and volume_signal['signal'] == 'BEARISH'):
            boost = abs(volume_signal['score'])
            enhanced_conf += boost
            boost_details.append(f"Volume: +{boost:.2%}")
        elif volume_signal['signal'] != 'NEUTRAL':
            penalty = abs(volume_signal['score']) * 0.3
            enhanced_conf -= penalty
            boost_details.append(f"Volume: -{penalty:.2%}")

        if context['signal'] != 'NEUTRAL':
            if (direction == "UP" and context['signal'] == 'BUY') or (direction == "DOWN" and context['signal'] == 'SELL'):
                boost = 0.05
                enhanced_conf += boost
                boost_details.append(f"Context: +{boost:.2%}")
            else:
                penalty = 0.08
                enhanced_conf -= penalty
                boost_details.append(f"Context: -{penalty:.2%}")

        enhanced_conf = max(0.5, min(1.0, enhanced_conf))

        print(f"\nFINAL PREDICTION: {direction} ({enhanced_conf:.1%})")
        print(f"   Boost: {enhanced_conf-base_confidence:+.1%} ({', '.join(boost_details)})")

        send_alert, reason = should_alert(direction, price, state)
        print(f"\nAlert decision: {reason}")

        if send_alert and enhanced_conf >= MIN_CONFIDENCE:
            stop = price * (1 - STOP_LOSS_PCT) if direction == "UP" else price * (1 + STOP_LOSS_PCT)
            stop_wide = price * (1 - STOP_LOSS_WIDE_PCT) if direction == "UP" else price * (1 + STOP_LOSS_WIDE_PCT)
            target = price * (1 + TAKE_PROFIT_PCT) if direction == "UP" else price * (1 - TAKE_PROFIT_PCT)

            # Auto stop recommendation
            vol_exp = volume_signal.get('explanation', '').lower()
            if 'divergence' in vol_exp:
                stop_rec = "USE STOP 2 - volume divergence detected"
            elif 'spike' in vol_exp:
                stop_rec = "USE STOP 2 - volume spike detected"
            else:
                stop_rec = "USE STOP 1 - normal volume"

            move_analyzer = MoveAnalyzer()
            move_stats = move_analyzer.analyze_typical_moves(df, direction)
            recommendations = move_analyzer.format_recommendation_message(price, direction, move_stats)

            # Clean text for Telegram markdown safety
            def clean(text):
                return str(text).replace('_', ' ').replace('*', '').replace('`', '').replace('[', '').replace(']', '')

            message = (
                f"*WHEAT ALERT - ULTIMATE v3.0*\n"
                f"Morning Alert\n\n"
                f"{'UP' if direction == 'UP' else 'DOWN'} ({enhanced_conf:.1%})\n"
                f"Price: {price:.2f}c\n\n"
                f"*ENSEMBLE AI:*\n"
                f"LSTM: {clean(prediction['model_details']['LSTM'])}\n"
                f"RF: {clean(prediction['model_details']['RandomForest'])}\n"
                f"XGB: {clean(prediction['model_details']['XGBoost'])}\n"
                f"Agreement: {clean(prediction['agreement'])}\n\n"
                f"*FUNDAMENTAL FACTORS:*\n"
                f"Seasonal: {clean(seasonal['direction'])} - {clean(seasonal['explanation'])}\n"
                f"Weather: {clean(weather_signal['signal'])} ({clean(weather_signal['explanation'])})\n"
                f"WASDE: {clean(wasde_signal['signal'])} (Stocks: {wasde_signal['data']['stocks_to_use']:.0%})\n"
                f"Volume: {clean(volume_signal['signal'])} ({clean(volume_signal['explanation'])})\n"
                f"Context: {clean(context['position'])}\n\n"
                f"*TRADE SETUP:*\n"
                f"Entry: {price:.2f}c\n"
                f"Stop 1 (Tight): {stop:.2f}c ({STOP_LOSS_PCT:.1%}) - normal market\n"
                f"Stop 2 (Wide): {stop_wide:.2f}c ({STOP_LOSS_WIDE_PCT:.1%}) - Jane Street protection\n"
                f"{stop_rec}\n"
                f"Target: {target:.2f}c ({TAKE_PROFIT_PCT:.1%})\n"
                f"R:R = 1.67:1\n\n"
                f"{clean(recommendations)}\n\n"
                f"{clean(reason)}\n"
                f"Professional Edition"
            )

            telegram_success = send_telegram(message)

            if telegram_success:
                print("\nAlert sent!")
                state['last_alert_time'] = datetime.now().isoformat()
                state['last_alert_date'] = datetime.now().date().isoformat()
                state['alerts_sent'] = state.get('alerts_sent', 0) + 1
                state['last_confidence'] = enhanced_conf

                # Track slot
                israel_time, _, _ = get_israel_time()
                israel_hour = israel_time.hour
                israel_date = israel_time.date().isoformat()
                slot = 'morning' if israel_hour in (1, 2) else f'manual_{israel_hour}h'
                slot_key = f"{israel_date}_{slot}"
                if 'alerts_today' not in state:
                    state['alerts_today'] = {}
                state['alerts_today'] = {
                    k: v for k, v in state['alerts_today'].items()
                    if k >= (datetime.now() - timedelta(days=3)).date().isoformat()
                }
                state['alerts_today'][slot_key] = True

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
                            'ensemble': f"{prediction['agreement']} ({prediction['votes_up']}/3)"
                        }
                    )
                except Exception as e:
                    print(f"Performance tracking skipped: {e}")
            else:
                print("\nAlert failed - state NOT updated")
        else:
            print(f"No alert: {reason if not send_alert else f'Confidence {enhanced_conf:.1%} below {MIN_CONFIDENCE:.0%}'}")

        state['last_direction'] = direction
        state['last_price'] = price
        save_state(state)

        print(f"\nTotal alerts sent: {state.get('alerts_sent', 0)}")
        print(f"Last check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
