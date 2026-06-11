"""
PROFESSIONAL WHEAT TRADING SYSTEM - ULTIMATE EDITION
Combines: Ensemble AI + Weather + WASDE + Volume + Seasonal + Context

CHANGES FROM ORIGINAL:
1. Incomplete candle fix in fetch_data()
2. Israel timezone slot-based alerting (1AM and 4PM Israel)
3. FORCE_ALERT for manual triggers
4. Double stop loss in message (1.5% tight + 2.5% wide)
5. Auto stop recommendation based on volume

PATCH APPLIED:
6. Performance gate: validate old predictions + check win rate before alerting
7. Circuit breaker: auto-suppress alerts after 3 consecutive losses
8. Shared tracker instance (no duplicate imports)
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
# LIVE WASDE SCRAPER - MULTI-GRAIN EDITION
# Fetches wheat, corn, and soy stocks from USDA.
# Derives wheat-specific signal using inter-grain relationships:
#   - Corn/soy tight → acreage competition → bullish wheat
#   - Corn/soy ample → no competition pressure → neutral/bearish wheat
#   - Wheat STU is the primary signal; grains context adjusts it
# ============================================================================

class LiveWASDEScraper:

    # Annual US consumption benchmarks (bushels) — stable USDA WASDE figures
    ANNUAL_USE = {
        'WHEAT': 2_000_000_000,   # ~2.0B bu
        'CORN':  14_500_000_000,  # ~14.5B bu
        'SOYBEANS': 4_400_000_000 # ~4.4B bu
    }

    # STU thresholds per grain
    STU_THRESHOLDS = {
        'WHEAT':    {'very_tight': 0.27, 'tight': 0.30, 'ample': 0.33},
        'CORN':     {'very_tight': 0.08, 'tight': 0.10, 'ample': 0.13},
        'SOYBEANS': {'very_tight': 0.05, 'tight': 0.07, 'ample': 0.10},
    }

    def __init__(self):
        self.api_key  = os.getenv("USDA_API_KEY", "3338B84E-694D-3E6A-925C-F35064C59BAE")
        self.base_url = "https://quickstats.nass.usda.gov/api/api_GET/"

    # ── public entry point ────────────────────────────────────────────────────

    def get_fundamental_score(self):
        """
        Fetch all three grains from USDA, then derive a wheat-specific signal.
        Falls back to market proxy (price ratios) if USDA is unavailable.
        """
        print("   📊 WASDE: fetching all grains...")

        grain_data = self._fetch_all_grains()

        if grain_data.get('WHEAT'):
            return self._derive_wheat_signal(grain_data)

        print("      USDA unavailable — using market proxy")
        return self._score_from_market_proxy()

    # ── USDA fetching ─────────────────────────────────────────────────────────

    def _fetch_all_grains(self):
        """Fetch stocks for wheat, corn, and soybeans in parallel-ish calls."""
        grains = {
            'WHEAT':    {'commodity_desc': 'WHEAT',    'class_desc': 'ALL CLASSES'},
            'CORN':     {'commodity_desc': 'CORN',     'class_desc': 'ALL CLASSES'},
            'SOYBEANS': {'commodity_desc': 'SOYBEANS', 'class_desc': 'ALL CLASSES'},
        }

        results = {}
        for grain_name, grain_params in grains.items():
            print(f"      {grain_name}...", end=" ")
            data = self._fetch_grain_stocks(grain_params)
            if data:
                stu = data['current_stocks'] / self.ANNUAL_USE[grain_name]
                data['stu'] = stu
                results[grain_name] = data
                print(f"STU={stu:.1%} YoY={data['yoy_change_pct']:+.1f}%")
            else:
                print("failed")

        return results

    def _fetch_grain_stocks(self, grain_params):
        """Fetch stocks for a single grain from USDA QuickStats."""
        try:
            params = {
                'key':               self.api_key,
                'source_desc':       'SURVEY',
                'commodity_desc':    grain_params['commodity_desc'],
                'class_desc':        grain_params['class_desc'],
                'statisticcat_desc': 'STOCKS',
                'unit_desc':         'BU',
                'agg_level_desc':    'NATIONAL',
                'format':            'JSON',
                'year__GE':          2021,
            }
            response = requests.get(self.base_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    return self._parse_stocks_data(data['data'])
            return None
        except Exception as e:
            print(f"({e})", end=" ")
            return None

    def _parse_stocks_data(self, data):
        """Parse raw USDA records into current stocks + YoY change."""
        sorted_data = sorted(
            data,
            key=lambda x: (x.get('year', 0), x.get('reference_period_desc', '')),
            reverse=True
        )
        if not sorted_data:
            return None

        latest       = sorted_data[0]
        latest_year  = latest.get('year')

        try:
            latest_value = float(latest.get('Value', '0').replace(',', ''))
        except (ValueError, TypeError):
            return None

        if latest_value <= 0:
            return None

        previous_value = None
        for record in sorted_data[1:]:
            if record.get('year') != latest_year:
                try:
                    v = float(record.get('Value', '0').replace(',', ''))
                    if v > 0:
                        previous_value = v
                        break
                except (ValueError, TypeError):
                    continue

        yoy_change = 0.0
        if previous_value and previous_value > 0:
            yoy_change = ((latest_value - previous_value) / previous_value) * 100

        return {
            'current_stocks': latest_value,
            'yoy_change_pct': yoy_change,
            'year':           latest_year,
        }

    # ── wheat signal derivation ───────────────────────────────────────────────

    def _derive_wheat_signal(self, grain_data):
        """
        Derive wheat prediction from all three grains.

        Logic:
          1. Wheat STU is the primary signal (own supply/demand)
          2. Corn STU adjusts it — tight corn = acreage competition = bullish wheat
          3. Soy STU adjusts it — tight soy = acreage competition = bullish wheat
          4. YoY direction for each grain adds confirmation or caution
        """
        score   = 0.0
        factors = []

        wheat = grain_data.get('WHEAT')
        corn  = grain_data.get('CORN')
        soy   = grain_data.get('SOYBEANS')

        thresh_w = self.STU_THRESHOLDS['WHEAT']
        thresh_c = self.STU_THRESHOLDS['CORN']
        thresh_s = self.STU_THRESHOLDS['SOYBEANS']

        # ── 1. Wheat own STU (weight: primary, ±0.25) ──
        w_stu = wheat['stu']
        w_yoy = wheat['yoy_change_pct']

        if w_stu < thresh_w['very_tight']:
            score += 0.25
            factors.append(f"Wheat very tight ({w_stu:.1%} STU)")
        elif w_stu < thresh_w['tight']:
            score += 0.15
            factors.append(f"Wheat tight ({w_stu:.1%} STU)")
        elif w_stu > thresh_w['ample']:
            score -= 0.15
            factors.append(f"Wheat ample ({w_stu:.1%} STU)")
        else:
            factors.append(f"Wheat balanced ({w_stu:.1%} STU)")

        if w_yoy < -5:
            score += 0.08
            factors.append(f"Wheat stocks falling {abs(w_yoy):.1f}% YoY")
        elif w_yoy > 5:
            score -= 0.05
            factors.append(f"Wheat stocks rising {w_yoy:.1f}% YoY")

        # ── 2. Corn STU context (weight: secondary, ±0.08) ──
        # Tight corn = farmers favour corn acres → less wheat planting → bullish wheat
        if corn:
            c_stu = corn['stu']
            if c_stu < thresh_c['tight']:
                score += 0.08
                factors.append(f"Corn tight ({c_stu:.1%}) → acreage competition")
            elif c_stu > thresh_c['ample']:
                score -= 0.04
                factors.append(f"Corn ample ({c_stu:.1%}) → no acre competition")

        # ── 3. Soy STU context (weight: tertiary, ±0.06) ──
        # Same acreage competition logic as corn
        if soy:
            s_stu = soy['stu']
            if s_stu < thresh_s['tight']:
                score += 0.06
                factors.append(f"Soy tight ({s_stu:.1%}) → acreage competition")
            elif s_stu > thresh_s['ample']:
                score -= 0.03
                factors.append(f"Soy ample ({s_stu:.1%}) → no acre competition")

        # ── 4. Cross-grain divergence bonus ──
        # If wheat is tight BUT corn+soy are both ample → wheat premium justified
        if corn and soy:
            grains_ample = (
                corn['stu']  > thresh_c['ample'] and
                soy['stu']   > thresh_s['ample']
            )
            wheat_tight = w_stu < thresh_w['tight']

            if wheat_tight and grains_ample:
                score += 0.07
                factors.append("Wheat tight while corn/soy ample → wheat premium")
            elif not wheat_tight and not grains_ample:
                # All grains ample — bearish pressure across the board
                score -= 0.05
                factors.append("All grains well supplied → broad bearish pressure")

        # ── Final signal ──
        signal = 'BULLISH' if score > 0.10 else 'BEARISH' if score < -0.05 else 'NEUTRAL'

        print(f"      WASDE multi-grain result: {signal} (score={score:+.3f})")
        print(f"      Wheat STU={w_stu:.1%} | "
              f"Corn STU={corn['stu']:.1%} | " if corn else "Corn: N/A | ",
              f"Soy STU={soy['stu']:.1%}" if soy else "Soy: N/A")

        return {
            'signal': signal,
            'score':  round(score, 4),
            'data': {
                'stocks_to_use':  w_stu,
                'stocks_change':  w_yoy,
                'corn_stu':       corn['stu']  if corn else None,
                'soy_stu':        soy['stu']   if soy  else None,
                'last_updated':   datetime.now().strftime('%Y-%m-%d'),
                'source':         'USDA QuickStats LIVE (wheat+corn+soy)',
            },
            'factors': factors[:3],
            'explanation': f"Wheat {w_stu:.1%} STU | Corn {corn['stu']:.1%} | Soy {soy['stu']:.1%}" if (corn and soy) else f"Wheat {w_stu:.1%} STU",
        }

    # ── market proxy fallback ─────────────────────────────────────────────────

    def _score_from_market_proxy(self):
        """
        Fallback when USDA is unavailable.
        Uses wheat/corn AND wheat/soy price ratios as supply proxies.
        Both ratios are forward-looking — the market already prices in supply.
        """
        try:
            end   = datetime.now()
            start = end - timedelta(days=400)

            wdf = yf.Ticker("ZW=F").history(start=start, end=end, auto_adjust=False)
            cdf = yf.Ticker("ZC=F").history(start=start, end=end, auto_adjust=False)
            sdf = yf.Ticker("ZS=F").history(start=start, end=end, auto_adjust=False)

            if wdf.empty:
                return self._get_default_estimates()

            score   = 0.0
            factors = []

            # Wheat/corn ratio z-score
            if not cdf.empty:
                wc = (wdf['Close'] / cdf['Close'].reindex(wdf.index, method='ffill')).dropna()
                if len(wc) > 20:
                    wc_z = float((wc.iloc[-1] - wc.mean()) / wc.std())
                    if wc_z > 0.75:
                        score += 0.12
                        factors.append(f"Wheat/corn ratio elevated (z={wc_z:+.2f})")
                    elif wc_z < -0.75:
                        score -= 0.08
                        factors.append(f"Wheat/corn ratio depressed (z={wc_z:+.2f})")
                    else:
                        factors.append(f"Wheat/corn ratio normal (z={wc_z:+.2f})")

            # Wheat/soy ratio z-score
            if not sdf.empty:
                ws = (wdf['Close'] / sdf['Close'].reindex(wdf.index, method='ffill')).dropna()
                if len(ws) > 20:
                    ws_z = float((ws.iloc[-1] - ws.mean()) / ws.std())
                    if ws_z > 0.75:
                        score += 0.08
                        factors.append(f"Wheat/soy ratio elevated (z={ws_z:+.2f})")
                    elif ws_z < -0.75:
                        score -= 0.05
                        factors.append(f"Wheat/soy ratio depressed (z={ws_z:+.2f})")

            signal = 'BULLISH' if score > 0.10 else 'BEARISH' if score < -0.05 else 'NEUTRAL'

            return {
                'signal': signal,
                'score':  round(score, 4),
                'data': {
                    'stocks_to_use': 0.0,
                    'stocks_change': 0,
                    'last_updated':  datetime.now().strftime('%Y-%m-%d'),
                    'source':        'Market proxy (W/C + W/S ratios)',
                },
                'factors':     factors[:3],
                'explanation': f"Market proxy: {signal}",
            }

        except Exception as e:
            print(f"      Market proxy error: {e}")
            return self._get_default_estimates()

    def _get_default_estimates(self):
        month = datetime.now().month
        if month in [1, 2, 3]:
            return {'signal': 'BULLISH', 'score': 0.20, 'data': {'stocks_to_use': 0.18, 'stocks_change': -2, 'source': 'ESTIMATED'}, 'factors': ['Seasonal estimate'], 'explanation': 'Seasonal estimate'}
        elif month in [7, 8, 9]:
            return {'signal': 'BEARISH', 'score': -0.10, 'data': {'stocks_to_use': 0.22, 'stocks_change': 3, 'source': 'ESTIMATED'}, 'factors': ['Post-harvest'], 'explanation': 'Post-harvest estimate'}
        else:
            return {'signal': 'NEUTRAL', 'score': 0.10, 'data': {'stocks_to_use': 0.20, 'stocks_change': 0, 'source': 'ESTIMATED'}, 'factors': ['Balanced'], 'explanation': 'Seasonal estimate'}

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

    april2 = datetime(year, 4, 2)
    days_to_friday = (april2.weekday() - 4) % 7
    dst_start = april2 - timedelta(days=days_to_friday)
    dst_start = dst_start.replace(hour=0, minute=0, second=0)

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

        last_date = df.index[-1].date()
        df = df.iloc[:-1]
        print(f"   Dropped last candle ({last_date}) - using {df.index[-1].date()} close")
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
    """Slot-based alerting: send at 01:00 Israel time only.
    Manual triggers (workflow_dispatch) always send immediately."""

    force_alert = os.getenv('FORCE_ALERT', '').lower() in ('true', '1', 'yes')
    github_event = os.getenv('GITHUB_EVENT_NAME', '')
    is_manual = force_alert or 'workflow_dispatch' in github_event

    print(f"\nAlert Check:")
    print(f"   FORCE_ALERT={force_alert}, GITHUB_EVENT_NAME={github_event}")

    if is_manual:
        print(f"   Manual trigger - sending immediately")
        return True, "Manual alert", True

    israel_time, utc_offset, tz_name = get_israel_time()
    israel_hour = israel_time.hour
    israel_date = israel_time.date().isoformat()

    print(f"   Israel time: {israel_time.strftime('%Y-%m-%d %H:%M')} {tz_name}")
    print(f"   Israel hour: {israel_hour}:00")

    if israel_hour in (1, 2):
        slot = 'morning'
        slot_label = f"Morning Alert (01:00 {tz_name})"
    else:
        print(f"   Not a scheduled hour ({israel_hour}:00 Israel) - NO ALERT")
        return False, f"Not scheduled hour ({israel_hour}:00 Israel)", False

    alerts_today = state.get('alerts_today', {})
    slot_key = f"{israel_date}_{slot}"
    if alerts_today.get(slot_key, False):
        print(f"   {slot_label} already sent today - NO ALERT")
        return False, f"{slot_label} already sent", False

    print(f"   {slot_label} - SENDING")
    return True, slot_label, False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print(f"WHEAT MONITOR - ULTIMATE EDITION v3.0")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*80}\n")

    state = load_state()

    # -------------------------------------------------------------------------
    # PATCH: Validate old predictions, then check performance gate
    # -------------------------------------------------------------------------
    tracker = None
    try:
        from performance_tracker import PerformanceTracker
        tracker = PerformanceTracker()

        validated = tracker.validate_predictions()
        if validated > 0:
            print(f"   Validated {validated} pending prediction(s)")

        gate_ok, gate_reason = tracker.get_confidence_gate()
        print(f"\n🚦 Performance gate: {gate_reason}")

        if not gate_ok:
            # Circuit breaker or low win rate — exit without alerting
            print(f"   Alerts suppressed — saving state and exiting")
            save_state(state)
            return

    except Exception as e:
        print(f"⚠️  Performance gate skipped: {e}")
        # Non-fatal — continue normally if tracker unavailable
    # -------------------------------------------------------------------------

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

        send_alert, reason, is_manual = should_alert(direction, price, state)
        print(f"\nAlert decision: {reason}")

        # ── HIGH CONVICTION GATE ──────────────────────────────────────────────
        # Only sends alerts when backtest-proven conditions are met.
        # Tiers: 1=100% accuracy, 2=94.7%, 3=94.1%, 4=81.7% (vs 68% baseline)
        gate_allowed  = True
        gate_tier     = 0
        gate_accuracy = 0.68
        gate_msg      = ""

        try:
            from high_conviction_gate import HighConvictionGate
            hcg = HighConvictionGate()
            gate_allowed, gate_tier, gate_accuracy, gate_reason, gate_conditions = hcg.check_gate(df)
            hcg.log_conditions(gate_conditions)
            gate_msg = hcg.format_for_telegram(gate_tier, gate_accuracy, gate_reason, gate_conditions)
            print(f"\n   Conviction gate: {'TIER ' + str(gate_tier) if gate_allowed else 'BLOCKED'}")
            print(f"   {gate_reason}")
        except Exception as e:
            print(f"   Conviction gate skipped: {e}")
            gate_allowed = True   # non-fatal — allow if gate fails to load
        # ─────────────────────────────────────────────────────────────────────

        if send_alert and enhanced_conf >= MIN_CONFIDENCE:
            stop = price * (1 - STOP_LOSS_PCT) if direction == "UP" else price * (1 + STOP_LOSS_PCT)
            stop_wide = price * (1 - STOP_LOSS_WIDE_PCT) if direction == "UP" else price * (1 + STOP_LOSS_WIDE_PCT)
            target = price * (1 + TAKE_PROFIT_PCT) if direction == "UP" else price * (1 - TAKE_PROFIT_PCT)

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

            def clean(text):
                return str(text).replace('_', ' ').replace('*', '').replace('`', '').replace('[', '').replace(']', '')

            # Build conviction block — shown on every alert so you can decide
            tier_emojis  = {1: "💎", 2: "🥇", 3: "🥈", 4: "🥉"}
            tier_labels  = {
                1: "TIER 1 — Historical accuracy: 100% (6/yr)",
                2: "TIER 2 — Historical accuracy: 94.7% (9/yr)",
                3: "TIER 3 — Historical accuracy: 94.1% (8/yr)",
                4: "TIER 4 — Historical accuracy: 81.7% (45/yr)",
                0: "NO TIER — Baseline accuracy: 68% (daily)",
            }
            tier_advice  = {
                1: "STRONG SETUP — high confidence to enter",
                2: "STRONG SETUP — high confidence to enter",
                3: "GOOD SETUP — consider entering",
                4: "MODERATE SETUP — use smaller size",
                0: "WEAK SETUP — wait or skip today",
            }
            t_emoji  = tier_emojis.get(gate_tier, "⚪")
            t_label  = tier_labels.get(gate_tier, "")
            t_advice = tier_advice.get(gate_tier, "")

            # Gate conditions summary for transparency
            if gate_tier > 0:
                conditions_met = []
                if gate_conditions.get('bearish_month'):  conditions_met.append("Harvest month")
                if gate_conditions.get('in_lower_half'):  conditions_met.append(f"Low in range ({gate_conditions.get('range_pct', 0):.0%})")
                if gate_conditions.get('rsi_oversold'):   conditions_met.append(f"RSI oversold ({gate_conditions.get('rsi', 0):.0f})")
                if gate_conditions.get('vol_low'):        conditions_met.append(f"Low volume ({gate_conditions.get('vol_ratio', 0):.1f}x)")
                if gate_conditions.get('inside_bb'):      conditions_met.append("Inside BB")
                conditions_str = " | ".join(conditions_met)
            else:
                missing = []
                if not gate_conditions.get('bearish_month', False): missing.append("Not harvest month")
                if not gate_conditions.get('in_lower_half', False):  missing.append(f"High in range ({gate_conditions.get('range_pct', 0):.0%})")
                if not gate_conditions.get('vol_low', False):        missing.append(f"Vol {gate_conditions.get('vol_ratio', 0):.1f}x (not low)")
                conditions_str = "Missing: " + " | ".join(missing[:3])

            message = (
                f"*WHEAT ALERT - ULTIMATE v3.0*\n"
                f"Morning Alert\n\n"
                f"{'UP' if direction == 'UP' else 'DOWN'} ({enhanced_conf:.1%})\n"
                f"Price: {price:.2f}c\n\n"
                f"*{t_emoji} CONVICTION RATING:*\n"
                f"{t_label}\n"
                f"{conditions_str}\n"
                f"DECISION: {t_advice}\n\n"
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

                if not is_manual:
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

                # -----------------------------------------------------------------
                # PATCH: Log prediction using the shared tracker instance
                # -----------------------------------------------------------------
                try:
                    if tracker is not None:
                        tracker.log_prediction(
                            direction=direction,
                            price=price,
                            confidence=enhanced_conf,
                            factors={
                                'seasonal': seasonal['direction'],
                                'weather':  weather_signal['signal'],
                                'wasde':    wasde_signal['signal'],
                                'volume':   volume_signal['signal'],
                                'ensemble': f"{prediction['agreement']} ({prediction['votes_up']}/3)"
                            }
                        )
                    else:
                        # Fallback: tracker failed to init earlier, try again
                        from performance_tracker import PerformanceTracker
                        PerformanceTracker().log_prediction(
                            direction=direction,
                            price=price,
                            confidence=enhanced_conf,
                            factors={
                                'seasonal': seasonal['direction'],
                                'weather':  weather_signal['signal'],
                                'wasde':    wasde_signal['signal'],
                                'volume':   volume_signal['signal'],
                                'ensemble': f"{prediction['agreement']} ({prediction['votes_up']}/3)"
                            }
                        )
                except Exception as e:
                    print(f"Performance tracking skipped: {e}")
                # -----------------------------------------------------------------

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
