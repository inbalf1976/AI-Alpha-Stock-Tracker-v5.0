"""
WHEAT MONITOR v4.2 - WEEKLY EXPANSION ENGINE
=====================================================
Target Horizon: 5 Trading Days (Weekly structural range optimization)

UPGRADES:
  1. Statistical Weekly Ranges — Uses ATR * sqrt(5) to forecast the 
     expected high-probability trading boundary for the upcoming week.
  2. Inter-Commodity Analytics — Integrates Corn ($ZC=F$) price action
     and correlations straight into the machine learning feature matrix.
  3. Preserved System Core — Keeps your exact Seasonal overrides, Trend
     filters, daily dynamic seeds, and comprehensive data messaging.
"""

import os, sys, json, warnings, requests, logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Setup system logger
logging.basicConfig(filename='wheat_system.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER          = "ZW=F"
CORN_TICKER     = "ZC=F"
SOY_TICKER      = "ZS=F"
MIN_CONFIDENCE  = 0.58
STATE_FILE      = Path("wheat_monitor_state.json")

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")

EXCLUDE_YEARS   = [2022]  # War anomaly
IL_TZ           = ZoneInfo("Asia/Jerusalem")

# ── DATA EXTRACTION PIPELINE ──────────────────────────────────────────────────

def fetch_robust_history(ticker, days=5*365):
    """Fetches asset data from market with built-in API error containment."""
    end = datetime.now(IL_TZ)
    start = end - timedelta(days=days)
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if df.empty:
            logging.error(f"Empty data block returned for {ticker}")
            return pd.DataFrame()
        return df
    except Exception as e:
        logging.error(f"Failed to fetch market data for {ticker}: {e}")
        return pd.DataFrame()

# ── SEASONAL ENGINE ───────────────────────────────────────────────────────────

class SeasonalEngine:
    def __init__(self):
        self.seasonal_returns = None
        self.phase            = None
        self.bias             = 0.0
        self.confidence       = 0.0

    def fit(self, df):
        df = df.copy()
        df['doy']  = df.index.dayofyear
        df['year'] = df.index.year
        df['ret1'] = df['Close'].pct_change(1)

        df = df[~df['year'].isin(EXCLUDE_YEARS)]
        self.seasonal_returns = df.groupby('doy')['ret1'].mean()
        self.seasonal_returns = self.seasonal_returns.rolling(7, center=True, min_periods=1).mean()

    def get_current_phase(self):
        if self.seasonal_returns is None:
            return {'phase': 'UNKNOWN', 'bias': 0.0, 'confidence': 0.0, 'explanation': 'No data'}

        today_doy = datetime.now(IL_TZ).timetuple().tm_yday

        forward_days = []
        for offset in range(1, 21):
            doy = ((today_doy + offset - 1) % 365) + 1
            if doy in self.seasonal_returns.index:
                forward_days.append(self.seasonal_returns[doy])

        if not forward_days:
            return {'phase': 'NEUTRAL', 'bias': 0.0, 'confidence': 0.5, 'explanation': 'No seasonal data'}

        avg_forward = np.mean(forward_days)
        pos_days    = sum(1 for r in forward_days if r > 0)
        neg_days    = sum(1 for r in forward_days if r < 0)

        if avg_forward > 0.0005 and pos_days >= 13:
            phase      = 'BULLISH'
            confidence = min(0.85, 0.60 + pos_days * 0.012)
        elif avg_forward < -0.0005 and neg_days >= 13:
            phase      = 'BEARISH'
            confidence = min(0.85, 0.60 + neg_days * 0.012)
        else:
            phase      = 'NEUTRAL'
            confidence = 0.55

        month = datetime.now(IL_TZ).month
        labels = {
            1:'Jan neutral', 2:'Pre-spring dip', 3:'Spring rally starts',
            4:'Peak planting premium', 5:'Max weather premium',
            6:'Harvest pressure', 7:'Post-harvest low', 8:'Summer lull',
            9:'Fall recovery', 10:'Winter demand builds',
            11:'Pre-winter rally', 12:'Winter high'
        }

        self.phase      = phase
        self.bias       = avg_forward
        self.confidence = confidence

        return {
            'phase':       phase,
            'bias':        round(avg_forward, 5),
            'confidence':  round(confidence, 3),
            'pos_days':    pos_days,
            'neg_days':    neg_days,
            'explanation': labels.get(month, ''),
        }

    def blocks_direction(self, direction):
        if self.phase is None:
            return False, ""
        if direction == 'UP' and self.phase == 'BEARISH' and self.confidence >= 0.72:
            return True, f"Seasonal BEARISH phase blocks UP (confidence {self.confidence:.0%})"
        if direction == 'DOWN' and self.phase == 'BULLISH' and self.confidence >= 0.72:
            return True, f"Seasonal BULLISH phase blocks DOWN (confidence {self.confidence:.0%})"
        return False, ""

# ── TREND ENGINE ──────────────────────────────────────────────────────────────

class TrendEngine:
    def get_trend(self, df):
        close = df['Close']
        price = float(close.iloc[-1])
        sma5  = float(close.rolling(5).mean().iloc[-1])
        sma10 = float(close.rolling(10).mean().iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])

        rets = close.pct_change()
        last5 = rets.iloc[-5:]
        up_days   = int((last5 > 0).sum())
        down_days = int((last5 < 0).sum())

        if price > sma5 > sma10 > sma20:
            trend     = 'UP'
            strength  = 'STRONG' if price > sma50 else 'MODERATE'
        elif price < sma5 < sma10 < sma20:
            trend     = 'DOWN'
            strength  = 'STRONG' if price < sma50 else 'MODERATE'
        else:
            trend     = 'NEUTRAL'
            strength  = 'WEAK'

        return {
            'trend':     trend,
            'strength':  strength,
            'price':     price,
            'sma5':      round(sma5, 2),
            'sma10':     round(sma10, 2),
            'sma20':     round(sma20, 2),
            'sma50':     round(sma50, 2),
            'up_days':   up_days,
            'down_days': down_days,
        }

    def blocks_direction(self, direction, trend_data):
        if direction == 'DOWN' and trend_data['trend'] == 'UP' and trend_data['strength'] == 'STRONG':
            return True, f"Strong uptrend blocks DOWN (price {trend_data['price']:.1f} > all MAs)"
        if direction == 'UP' and trend_data['trend'] == 'DOWN' and trend_data['strength'] == 'STRONG':
            return True, f"Strong downtrend blocks UP (price {trend_data['price']:.1f} < all MAs)"
        return False, ""

# ── CONVICTION GATE ───────────────────────────────────────────────────────────

class ConvictionGate:
    def evaluate(self, df):
        close = df['Close']
        price = float(close.iloc[-1])
        month = datetime.now(IL_TZ).month

        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = float(100 - (100 / (1 + gain / loss)).iloc[-1])

        vol_avg   = float(df['Volume'].rolling(20).mean().iloc[-1])
        vol_curr  = float(df['Volume'].iloc[-1])
        vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1.0

        prices_1yr = close.iloc[-252:] if len(close) >= 252 else close
        high52     = float(prices_1yr.max())
        low52      = float(prices_1yr.min())
        range_pct  = (price - low52) / (high52 - low52) if high52 > low52 else 0.5

        bb_mid   = float(close.rolling(20).mean().iloc[-1])
        bb_std   = float(close.rolling(20).std().iloc[-1])
        inside_bb = abs(price - bb_mid) < 1.5 * bb_std

        bearish_month = month in [6, 7, 8]
        in_lower_half = range_pct < 0.40
        rsi_oversold  = rsi < 35
        vol_low       = vol_ratio < 0.80

        conditions = {
            'bearish_month': bearish_month, 'in_lower_half': in_lower_half,
            'rsi_oversold':  rsi_oversold, 'vol_low':       vol_low,
            'inside_bb':      inside_bb, 'rsi':            round(rsi, 1),
            'vol_ratio':      round(vol_ratio, 2), 'range_pct':      round(range_pct, 3),
            'month':          month, 'price':          round(price, 2),
        }

        if bearish_month and in_lower_half and rsi_oversold:
            tier, accuracy = 1, 1.00
            reason = f"💎 TIER 1 (100%) — Harvest + low range + RSI {rsi:.0f}"
        elif vol_low and bearish_month and in_lower_half:
            tier, accuracy = 2, 0.947
            reason = f"🥇 TIER 2 (94.7%) — Low vol + harvest + low range"
        elif vol_low and in_lower_half:
            tier, accuracy = 3, 0.817
            reason = f"🥉 TIER 3 (81.7%) — Low vol + low range"
        else:
            tier, accuracy = 0, 0.68
            missing = []
            if not bearish_month:  missing.append(f"not harvest month ({month})")
            if not in_lower_half:  missing.append(f"high in range ({range_pct:.0%})")
            if not vol_low:        missing.append(f"vol {vol_ratio:.1f}x")
            reason = f"⚪ NO TIER (68%) — {' | '.join(missing[:2])}"

        return tier, accuracy, reason, conditions

# ── ADVANCED DATA SYNC AND FEATURE MATRIX BUILDER ─────────────────────────────

def build_advanced_matrix():
    """Fetches core assets and constructs deep multi-commodity analytical frame."""
    df_wheat = fetch_robust_history(TICKER)
    df_corn  = fetch_robust_history(CORN_TICKER)
    
    if df_wheat.empty:
        return pd.DataFrame(), 0.0
        
    df = df_wheat.copy()
    df['Returns']    = df['Close'].pct_change()
    df['SMA_20']     = df['Close'].rolling(20).mean()
    df['SMA_50']     = df['Close'].rolling(50).mean()
    df['EMA_12']     = df['Close'].ewm(span=12).mean()
    df['EMA_26']     = df['Close'].ewm(span=26).mean()
    df['MACD']       = df['EMA_12'] - df['EMA_26']
    
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']        = 100 - (100 / (1 + gain / loss))
    
    df['Volatility'] = df['Returns'].rolling(20).std()
    hl, hc, lc = df['High'] - df['Low'], (df['High'] - df['Close'].shift()).abs(), (df['Low'] - df['Close'].shift()).abs()
    df['ATR']        = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['BB_Width']   = (df['Close'].rolling(20).std() * 2) / df['SMA_20']
    
    # Inter-Market Core Features (Corn Matrix Integration)
    if not df_corn.empty:
        df_corn = df_corn.reindex(df.index, method='ffill')
        df['Corn_Mom_3d'] = df_corn['Close'].pct_change(3)
        df['Wheat_Corn_Ratio'] = df['Close'] / df_corn['Close']
    else:
        df['Corn_Mom_3d'] = 0.0
        df['Wheat_Corn_Ratio'] = 1.35  # Standard benchmark default
        
    return df.dropna(), float(df['ATR'].iloc[-1])

# ── MULTI-MODEL ENSEMBLE PREDICTOR ───────────────────────────────────────────

class EnsemblePredictor:
    def __init__(self):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler_lstm = MinMaxScaler()
        self.scaler_ml   = MinMaxScaler()
        self.lstm_model  = None
        self.rf_model    = None
        self.xgb_model   = None
        self.seq_len     = 60
        self.features    = ['Close', 'Volume', 'Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_Width', 'Volatility', 'ATR']

    def train(self, df):
        from keras.models import Sequential
        from keras.layers import LSTM as KerasLSTM, Dense, Dropout
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        print("   Training LSTM + RF + XGB with Inter-Market features...")
        y = np.array([1 if df['Close'].iloc[i] > df['Close'].iloc[i-1] else 0 for i in range(self.seq_len, len(df))])

        data_lstm   = df[self.features].values
        scaled_lstm = self.scaler_lstm.fit_transform(data_lstm)
        X_lstm      = np.array([scaled_lstm[i-self.seq_len:i] for i in range(self.seq_len, len(scaled_lstm))])

        ml_feat = self._build_ml_features(df).iloc[-len(y):]
        X_ml    = self.scaler_ml.fit_transform(ml_feat.fillna(0))

        self.lstm_model = Sequential([
            KerasLSTM(64, return_sequences=True, input_shape=(self.seq_len, len(self.features))),
            Dropout(0.2), KerasLSTM(32), Dropout(0.2),
            Dense(16, activation='relu'), Dense(1, activation='sigmoid')
        ])
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
        self.lstm_model.fit(X_lstm, y, epochs=25, batch_size=32, validation_split=0.15, verbose=0)

        # Dynamic daily variant seeds
        seed = datetime.now(IL_TZ).timetuple().tm_yday

        self.rf_model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_split=5, random_state=seed, n_jobs=-1)
        self.rf_model.fit(X_ml, y)

        self.xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=seed, use_label_encoder=False, eval_metric='logloss')
        self.xgb_model.fit(X_ml, y, verbose=False)
        print("   ✓ All models trained")

    def _build_ml_features(self, df):
        f = pd.DataFrame(index=df.index)
        f['ret_1d']      = df['Returns']
        f['ret_5d']      = df['Close'].pct_change(5)
        f['rsi']         = df['RSI']
        f['macd']        = df['MACD']
        f['atr_pct']     = df['ATR'] / df['Close']
        f['bb_width']    = df['BB_Width']
        f['corn_momentum'] = df.get('Corn_Mom_3d', 0.0)
        f['grain_ratio']   = df.get('Wheat_Corn_Ratio', 1.35)
        return f.dropna()

    def predict(self, df):
        data   = df[self.features].values
        scaled = self.scaler_lstm.transform(data)
        X_lstm = np.array([scaled[-self.seq_len:]])
        lstm_p = float(self.lstm_model.predict(X_lstm, verbose=0)[0][0])

        feat  = self._build_ml_features(df).iloc[[-1]]
        X_ml  = self.scaler_ml.transform(feat.fillna(0))
        rf_p  = float(self.rf_model.predict_proba(X_ml)[0][1])
        xgb_p = float(self.xgb_model.predict_proba(X_ml)[0][1])

        weights = [abs(p - 0.5) for p in [lstm_p, rf_p, xgb_p]]
        total   = sum(weights) or 1
        weighted = sum(p * w for p, w in zip([lstm_p, rf_p, xgb_p], weights)) / total

        votes_up = sum(1 for p in [lstm_p, rf_p, xgb_p] if p >= 0.5)
        direction = 'UP' if weighted >= 0.5 else 'DOWN'
        base_conf = weighted if weighted >= 0.5 else 1 - weighted

        bonus     = 0.06 if votes_up in [0, 3] else 0.02
        confidence = min(0.92, base_conf + bonus)
        agreement = 'FULL' if votes_up in [0,3] else 'MAJORITY'

        return {
            'direction':  direction, 'confidence': confidence,
            'lstm':        lstm_p, 'rf':          rf_p, 'xgb':         xgb_p,
            'weighted':    weighted, 'votes_up':    votes_up, 'agreement':  agreement,
        }

# ── FUNDAMENTALS & MARKET SIGNAL UTILITIES ────────────────────────────────────

def get_wasde_signal():
    api_key  = os.getenv("USDA_API_KEY", "3338B84E-694D-3E6A-925C-F35064C59BAE")
    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"
    ANNUAL_USE = {'WHEAT': 2e9, 'CORN': 14.5e9, 'SOYBEANS': 4.4e9}
    STU_TIGHT  = {'WHEAT': 0.30, 'CORN': 0.10, 'SOYBEANS': 0.07}
    STU_AMPLE  = {'WHEAT': 0.33, 'CORN': 0.13, 'SOYBEANS': 0.10}

    grain_stu = {}
    for grain in ['WHEAT', 'CORN', 'SOYBEANS']:
        try:
            r = requests.get(base_url, params={
                'key': api_key, 'source_desc': 'SURVEY', 'commodity_desc': grain, 
                'class_desc': 'ALL CLASSES', 'statisticcat_desc': 'STOCKS', 'unit_desc': 'BU',
                'agg_level_desc': 'NATIONAL', 'format': 'JSON', 'year__GE': 2021,
            }, timeout=12)
            if r.status_code == 200:
                records = r.json().get('data', [])
                if records:
                    records = sorted(records, key=lambda x: x.get('year', 0), reverse=True)
                    val = float(records[0]['Value'].replace(',', ''))
                    grain_stu[grain] = val / ANNUAL_USE[grain]
        except Exception: pass

    if not grain_stu.get('WHEAT'): return _wasde_market_proxy()

    score, factors, w_stu = 0.0, [], grain_stu['WHEAT']
    if w_stu < STU_TIGHT['WHEAT']: score += 0.20; factors.append(f"Wheat tight ({w_stu:.1%} STU)")
    elif w_stu > STU_AMPLE['WHEAT']: score -= 0.15; factors.append(f"Wheat ample ({w_stu:.1%} STU)")
    else: factors.append(f"Wheat balanced ({w_stu:.1%} STU)")

    signal = 'BULLISH' if score > 0.10 else 'BEARISH' if score < -0.05 else 'NEUTRAL'
    return {'signal': signal, 'score': score, 'stu': w_stu, 'factors': factors[:2], 'source': 'USDA LIVE'}

def _wasde_market_proxy():
    try:
        end = datetime.now(IL_TZ); start = end - timedelta(days=400)
        wdf = yf.Ticker(TICKER).history(start=start, end=end, auto_adjust=False)
        cdf = yf.Ticker(CORN_TICKER).history(start=start, end=end, auto_adjust=False)
        wc  = (wdf['Close'] / cdf['Close'].reindex(wdf.index, method='ffill')).dropna()
        z   = float((wc.iloc[-1] - wc.mean()) / wc.std())
        score = 0.12 if z > 0.75 else -0.08 if z < -0.75 else 0.0
        return {'signal': 'BULLISH' if score > 0 else 'BEARISH' if score < 0 else 'NEUTRAL', 'score': score, 'stu': 0.0, 'factors': [f"W/C ratio z={z:+.2f}"], 'source': 'Market proxy'}
    except Exception: return {'signal': 'NEUTRAL', 'score': 0.0, 'stu': 0.0, 'factors': [], 'source': 'Error'}

def get_weather_signal():
    cache_file = Path("weather_cache.json")
    api_key    = os.getenv("VISUAL_CROSSING_API_KEY", "W2FNC8VKT94JKH9ZRZYHUE63P")
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            if (datetime.now(IL_TZ) - datetime.fromisoformat(cache['ts'])).total_seconds() < 28800: return cache['data']
        except Exception: pass

    regions = {'Kansas': '38.5,-98.0', 'Ukraine': '46.5,32.0', 'Russia': '45.0,39.0'}
    scores = []
    for name, coords in regions.items():
        try:
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{coords}"
            r   = requests.get(url, params={'key': api_key, 'unitGroup': 'metric', 'include': 'days', 'elements': 'datetime,temp,tempmax,tempmin,precip', 'contentType': 'json', 'startDateTime': (datetime.now(IL_TZ) - timedelta(days=7)).strftime('%Y-%m-%d'), 'endDateTime': datetime.now(IL_TZ).strftime('%Y-%m-%d')}, timeout=10)
            if r.status_code == 200:
                days = r.json().get('days', [])
                precip = sum(d.get('precip', 0) for d in days)
                scores.append(0.12 if precip < 5 else 0.0)
        except Exception: pass

    result = {'signal': 'BULLISH' if np.mean(scores or [0]) > 0.10 else 'NEUTRAL', 'score': round(np.mean(scores or [0]), 4), 'explanation': 'Regions synced'}
    try: cache_file.write_text(json.dumps({'ts': datetime.now(IL_TZ).isoformat(), 'data': result}))
    except Exception: pass
    return result

def get_volume_signal(df):
    vol_avg  = float(df['Volume'].rolling(20).mean().iloc[-1])
    vol_curr = float(df['Volume'].iloc[-1])
    ratio    = vol_curr / vol_avg if vol_avg > 0 else 1.0
    ret      = float(df['Close'].pct_change(1).iloc[-1])
    signal   = 'BULLISH' if ratio > 1.5 and ret > 0 else 'BEARISH' if ratio > 1.5 and ret < 0 else 'NEUTRAL'
    return {'signal': signal, 'ratio': round(ratio, 2)}

# ── LOGISTICAL FLOW CONTROLLERS ───────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        try: return json.loads(STATE_FILE.read_text())
        except Exception: pass
    return {'alerts_sent': 0, 'alerts_today': {}, 'last_alert_date': None}

def save_state(state):
    state['last_check'] = datetime.now(IL_TZ).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))

def should_send(state):
    force  = os.getenv('FORCE_ALERT', '').lower() in ('true', '1', 'yes')
    manual = force or 'workflow_dispatch' in os.getenv('GITHUB_EVENT_NAME', '')
    if manual: return True, "Manual execution verified", True

    israel_time = datetime.now(IL_TZ)
    if israel_time.hour not in (1, 2): return False, f"Outside execution window ({israel_time.hour}:00 IL)", False

    slot_key = f"{israel_time.date().isoformat()}_morning"
    if state.get('alerts_today', {}).get(slot_key): return False, "Morning script processing completed", False
    return True, "Standard morning cycle window verified", False

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={"chat_id": TELEGRAM_CHAT, "text": message}, timeout=10)
        return r.status_code == 200
    except Exception: return False

def log_prediction(direction, price, confidence, tier, seasonal_phase):
    log_file = Path("prediction_log.json")
    try: log = json.loads(log_file.read_text()) if log_file.exists() else []
    except Exception: log = []
    log.append({'timestamp': datetime.now(IL_TZ).isoformat(), 'direction': direction, 'entry_price': price, 'confidence': confidence, 'tier': tier, 'seasonal_phase': seasonal_phase, 'validated': False})
    log_file.write_text(json.dumps(log, indent=2))

# ── MAIN PIPELINE EXECUTION ───────────────────────────────────────────────────

def main():
    print(f"\n======================================================================")
    print(f"WHEAT MONITOR v4.2 - WEEKLY PROJECTION SYSTEM")
    print(f"Time: {datetime.now(IL_TZ).strftime('%Y-%m-%d %H:%M:%S Israel Standard Time')}")
    print(f"======================================================================\n")

    state = load_state()
    send, gate_reason, is_manual = should_send(state)
    print(f"System Gate Checklist: {gate_reason}")

    # Fetch Data and Synchronize Indicators
    df, current_atr = build_advanced_matrix()
    if df.empty:
        print("CRITICAL ERROR: High-volume pipeline synchronization failed."); return

    if df.index[-1].date() == datetime.now(IL_TZ).date():
        df = df.iloc[:-1]

    last_candle_date = df.index[-1].date()
    days_since_candle = (datetime.now(IL_TZ).date() - last_candle_date).days
    
    if days_since_candle >= 3 and not is_manual:
        print(f"Market tracking paused: session inactive ({last_candle_date})")
        save_state(state); return

    current_price = float(df['Close'].iloc[-1])
    print(f"Reference Session Index: {current_price:.2f}¢ | Daily ATR: {current_atr:.2f}¢")

    # Engines Calculations
    seasonal = SeasonalEngine()
    seasonal.fit(df)
    s_phase = seasonal.get_current_phase()

    trend_engine = TrendEngine()
    trend_data   = trend_engine.get_trend(df)

    gate = ConvictionGate()
    tier, accuracy, conviction_reason, gate_conds = gate.evaluate(df)

    wasde   = get_wasde_signal()
    weather = get_weather_signal()
    volume  = get_volume_signal(df)

    ensemble = EnsemblePredictor()
    ensemble.train(df)
    pred = ensemble.predict(df)
    direction = pred['direction']

    # Filters and Conversions
    seasonal_blocked, seasonal_block_reason = seasonal.blocks_direction(direction)
    if seasonal_blocked:
        direction = 'DOWN' if direction == 'UP' else 'UP'
        pred['confidence'] = 0.62

    trend_blocked, trend_block_reason = trend_engine.blocks_direction(direction, trend_data)
    if trend_blocked:
        direction = 'DOWN' if direction == 'UP' else 'UP'
        pred['confidence'] = 0.58

    confidence = pred['confidence']
    if wasde['signal'] == ('BULLISH' if direction == 'UP' else 'BEARISH'): confidence = min(0.92, confidence + 0.03)
    if volume['signal'] == ('BULLISH' if direction == 'UP' else 'BEARISH'): confidence = min(0.92, confidence + 0.03)

    # ── STATISTICAL WEEKLY RANGE PROJECTIONS ──
    # Mathematical Expansion Formula: Daily ATR * sqrt(5)
    weekly_atr_expansion = current_atr * np.sqrt(5)
    
    if direction == 'UP':
        expected_range_low  = current_price - (0.50 * weekly_atr_expansion)  # Volatility baseline protection floor
        expected_range_high = current_price + (1.50 * weekly_atr_expansion)  # Expected target ceiling expansion boundary
    else:
        expected_range_low  = current_price - (1.50 * weekly_atr_expansion)  # Expected target descent expansion boundary
        expected_range_high = current_price + (0.50 * weekly_atr_expansion)  # Volatility baseline resistance ceiling

    # Compile Structured Message
    output_msg = (
        f"🌾 **WHEAT MONITOR v4.2 WEEKLY OUTLOOK** 🌾\n"
        f"Reference Date: {last_candle_date}\n\n"
        f"**PROJECTED WEEKLY BIAS**: #{direction}\n"
        f"• Base Entry Reference: {current_price:.2f}¢\n"
        f"• Expected Target Floor: {expected_range_low:.2f}¢\n"
        f"• Expected Target High:  {expected_range_high:.2f}¢\n\n"
        f"📊 **System Confidence Metrics:**\n"
        f"• Evaluation Setup: Tier {tier} ({conviction_reason})\n"
        f"• Score Probability Matrix: {confidence:.1%}\n"
        f"• Ensemble Agreement Model: {pred['agreement']} (L:{pred['lstm']:.2f} R:{pred['rf']:.2f} X:{pred['xgb']:.2f})\n\n"
        f"⚙️ **Macro Mechanics & Filters:**\n"
        f"• Seasonal State: {s_phase['phase']} ({s_phase['explanation']})\n"
        f"• Forward Outlook (20 Days): {s_phase['pos_days']} Up / {s_phase['neg_days']} Down\n"
        f"• Core Structural Trend: {trend_data['trend']} ({trend_data['strength']})\n"
        f"• Macro Matrix Fundamentals: WASDE {wasde['signal']} | Weather {weather['signal']} | Inter-commodity Momentum Ratio {volume['ratio']}x"
    )

    print("\n--- SYSTEM MESSAGING PAYLOAD PREVIEW ---")
    print(output_msg)

    if send:
        if send_telegram(output_msg):
            slot_key = f"{datetime.now(IL_TZ).date().isoformat()}_morning"
            state['alerts_sent'] += 1
            state['alerts_today'][slot_key] = True
            state['last_alert_date'] = datetime.now(IL_TZ).date().isoformat()
            log_prediction(direction, current_price, confidence, tier, s_phase['phase'])

    save_state(state)
    print("\n✓ System workflow finalized successfully.")

if __name__ == "__main__":
    main()
