"""
WHEAT MONITOR v4.0 - CLEAN REBUILD
=====================================
Built from scratch using everything learned over the past month.

DESIGN PRINCIPLES:
  1. Seasonal truth first — 5 years of ZW=F history defines the calendar
     2022 excluded (Ukraine war = global anomaly)
  2. Real price always — uses LIVE current price (not stale daily bar)
  3. Trend respect — never fights a confirmed multi-day trend
  4. Conviction gate — only alerts on historically proven setups
  5. Honest confidence — no artificial boosting, real probabilities only
  6. One alert per day — no duplicates, no noise

SIGNAL HIERARCHY (in order of weight):
  1. Seasonal phase      — derived from 5yr history, hard override
  2. Trend direction     — 5/10/20 day MA alignment
  3. Conviction tier     — backtest-proven condition combinations
  4. Ensemble models     — LSTM + RF + XGB with daily-sensitive features
  5. Fundamental context — WASDE multi-grain, weather, volume

ACCURACY TARGET: 80%+ on Tier 1/2 setups (~6-10 alerts/month)

CHANGELOG (this version):
  - FIX: current_price now comes from a live quote (get_live_price),
    not the last daily bar. The old logic dropped "today's" candle
    to avoid using an incomplete bar, but that meant current_price
    could silently lag by days around weekends/holidays. Now the
    daily bars still drive all indicators/seasonal/trend calcs —
    only the single "current price" used for entry/stop/target is
    live. If the live fetch fails, this is now flagged explicitly
    (⚠️ STALE) instead of failing silently.
"""

import os, sys, json, warnings, requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

IL = ZoneInfo("Asia/Jerusalem")   # Israel timezone — used everywhere

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── CONFIG ────────────────────────────────────────────────────────────────────

TICKER          = "ZW=F"
CORN_TICKER     = "ZC=F"
SOY_TICKER      = "ZS=F"
STOP_PCT        = 0.015   # 1.5%
TARGET_PCT      = 0.025   # 2.5%
MIN_CONFIDENCE  = 0.58
STATE_FILE      = Path("wheat_monitor_state.json")

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT   = os.getenv("TELEGRAM_CHAT_ID")

# Years to exclude from seasonal calculation (global anomalies)
EXCLUDE_YEARS   = [2022]

# ── LIVE PRICE FETCH ──────────────────────────────────────────────────────────

def get_live_price(ticker=TICKER):
    """
    Fetches the actual live/last-traded price, separate from the
    daily historical bars used for indicators. Daily bars can lag
    by days around holidays/weekends or while today's session is
    still forming — this pulls the real current quote instead.

    Returns (price, is_live). is_live=False means the live fetch
    failed and the caller should fall back to the last daily close,
    while flagging it clearly rather than trusting it silently.
    """
    try:
        fast = yf.Ticker(ticker).fast_info
        live = fast.get('last_price') or fast.get('lastPrice')
        if live and live > 0:
            return float(live), True
    except Exception as e:
        print(f"   fast_info live price failed: {e}")

    # Fallback: try 1-minute intraday bars for today
    try:
        intraday = yf.Ticker(ticker).history(period='1d', interval='1m')
        if not intraday.empty:
            return float(intraday['Close'].iloc[-1]), True
    except Exception as e:
        print(f"   intraday fallback failed: {e}")

    return None, False


# ── SEASONAL ENGINE ───────────────────────────────────────────────────────────

class SeasonalEngine:
    """
    Derives the wheat seasonal calendar directly from 5 years of
    ZW=F price history. No hardcoded assumptions — the data speaks.
    Excludes 2022 (Ukraine war anomaly).
    """

    def __init__(self):
        self.seasonal_returns = None
        self.phase            = None
        self.bias             = 0.0
        self.confidence       = 0.0

    def fit(self, df):
        """Calculate average return by day-of-year across 5 years, excluding anomaly years."""
        df = df.copy()
        df['doy']  = df.index.dayofyear
        df['year'] = df.index.year
        df['ret1'] = df['Close'].pct_change(1)

        # Exclude anomaly years
        df = df[~df['year'].isin(EXCLUDE_YEARS)]

        # Average return by day-of-year
        self.seasonal_returns = df.groupby('doy')['ret1'].mean()

        # Smooth with 7-day rolling average
        self.seasonal_returns = self.seasonal_returns.rolling(7, center=True, min_periods=1).mean()

    def get_current_phase(self):
        """
        Returns seasonal phase for today based on historical patterns.
        Looks at next 20 trading days to determine trend direction.
        """
        if self.seasonal_returns is None:
            return {'phase': 'UNKNOWN', 'bias': 0.0, 'confidence': 0.0, 'explanation': 'No data'}

        today_doy = datetime.now(IL).timetuple().tm_yday

        # Look at next 20 days of seasonal returns
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

        # Determine phase
        if avg_forward > 0.0005 and pos_days >= 13:
            phase      = 'BULLISH'
            confidence = min(0.85, 0.60 + pos_days * 0.012)
        elif avg_forward < -0.0005 and neg_days >= 13:
            phase      = 'BEARISH'
            confidence = min(0.85, 0.60 + neg_days * 0.012)
        else:
            phase      = 'NEUTRAL'
            confidence = 0.55

        # Month labels
        month = datetime.now(IL).month
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
        """
        Hard seasonal override.
        If seasonal phase strongly disagrees with direction → block.
        This is the most important filter in the system.
        """
        if self.phase is None:
            return False, ""

        if direction == 'UP' and self.phase == 'BEARISH' and self.confidence >= 0.72:
            return True, f"Seasonal BEARISH phase blocks UP (confidence {self.confidence:.0%})"

        if direction == 'DOWN' and self.phase == 'BULLISH' and self.confidence >= 0.72:
            return True, f"Seasonal BULLISH phase blocks DOWN (confidence {self.confidence:.0%})"

        return False, ""


# ── TREND ENGINE ──────────────────────────────────────────────────────────────

class TrendEngine:
    """
    Determines the current trend from price action.
    Never fight a confirmed trend.
    """

    def get_trend(self, df):
        close = df['Close']
        price = float(close.iloc[-1])
        sma5  = float(close.rolling(5).mean().iloc[-1])
        sma10 = float(close.rolling(10).mean().iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])

        # Consecutive up/down days
        rets = close.pct_change()
        last5 = rets.iloc[-5:]
        up_days   = int((last5 > 0).sum())
        down_days = int((last5 < 0).sum())

        # Trend strength
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
        """Block signals that fight a strong confirmed trend."""
        if direction == 'DOWN' and trend_data['trend'] == 'UP' and trend_data['strength'] == 'STRONG':
            return True, f"Strong uptrend blocks DOWN (price {trend_data['price']:.1f} > all MAs)"
        if direction == 'UP' and trend_data['trend'] == 'DOWN' and trend_data['strength'] == 'STRONG':
            return True, f"Strong downtrend blocks UP (price {trend_data['price']:.1f} < all MAs)"
        return False, ""


# ── CONVICTION GATE ───────────────────────────────────────────────────────────

class ConvictionGate:
    """
    REBUILT 2026-07-09 using real train/holdout backtest validation
    (see backtest.py and backtest_results.json).

    The previous version of this class used combinations found by
    searching hundreds of condition combos against a single dataset,
    reporting the best result as "100% accuracy". That number was
    proven fake: when tested against a holdout period the combos
    were never fitted to, most either collapsed to ~55-62% (barely
    better than a coin flip) or never occurred at all in the last
    4 months of data.

    This version uses ONLY single conditions that were individually
    validated on a real holdout set AND beat the baseline UP rate
    (67.46% — wheat trended up most of this 2-year window anyway,
    so anything below that adds zero real value, even if it "looks"
    high in isolation).

    CONFIRMED CONDITIONS (holdout period 2026-03-20 to 2026-07-09):
      vol_low       : 84.8% UP (n=33 holdout) — strongest real signal
      momentum_up   : 84.0% UP (n=25 holdout) — strong, worth watching for drift
      macd_bullish  : 70.0% UP (n=30 holdout) — modest but real edge
      bearish_month : 68.0% UP (n=25 holdout) — barely above baseline, weak

    EXPLICITLY EXCLUDED (proven unreliable on holdout — DO NOT re-add):
      rsi_oversold   : collapsed to 50.0% and flipped direction on holdout
      momentum_down  : collapsed to 57.7% and flipped direction on holdout
      near_bb_lower  : collapsed to 57.1% on tiny holdout sample (n=7)
      in_lower_half  : never occurred once in the entire holdout period
                        (consistent with this year's drought-driven price
                        strength keeping wheat out of the bottom of its range)
      wc_bullish, rsi_neutral, inside_bb, vol_good : all held up on holdout
                        but scored BELOW the 67.46% baseline, meaning they
                        add no real predictive value on their own

    IMPORTANT: this backtest is a rolling 2-year window ending 2026-07-09.
    Re-run backtest.py periodically (e.g. monthly) and update the numbers
    below — do not let this drift stale the way the old hardcoded "100%"
    numbers did.
    """

    # Holdout-validated accuracies — update these when you re-run backtest.py
    HOLDOUT_ACCURACY = {
        'vol_low':       0.848,
        'momentum_up':   0.840,
        'macd_bullish':  0.700,
        'bearish_month': 0.680,
    }
    BASELINE_UP = 0.6746  # from backtest_results.json baseline_up_full

    def evaluate(self, df):
        close = df['Close']
        price = float(close.iloc[-1])

        # Volume (for vol_low)
        vol_avg   = float(df['Volume'].rolling(20).mean().iloc[-1])
        vol_curr  = float(df['Volume'].iloc[-1])
        vol_ratio = vol_curr / vol_avg if vol_avg > 0 else 1.0
        vol_low   = vol_ratio < 0.80

        # Momentum (for momentum_up) — same-direction 1d and 3d returns
        ret_1d = float(close.pct_change(1).iloc[-1])
        ret_3d = float(close.pct_change(3).iloc[-1])
        momentum_up = (ret_1d > 0) and (ret_3d > 0)

        # MACD (for macd_bullish)
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        macd_bullish = float(macd.iloc[-1]) > float(macd_signal.iloc[-1])

        # Month (for bearish_month)
        month = datetime.now(IL).month
        bearish_month = month in [6, 7, 8]

        conditions = {
            'vol_low':       vol_low,
            'momentum_up':   momentum_up,
            'macd_bullish':  macd_bullish,
            'bearish_month': bearish_month,
            'vol_ratio':     round(vol_ratio, 2),
            'ret_1d':        round(ret_1d, 4),
            'ret_3d':        round(ret_3d, 4),
            'month':         month,
            'price':         round(price, 2),
        }

        # Rank active conditions by their real holdout accuracy — highest wins
        active = [(name, acc) for name, acc in self.HOLDOUT_ACCURACY.items()
                  if conditions.get(name)]

        if not active:
            tier, accuracy = 0, self.BASELINE_UP
            reason = f"⚪ NO SIGNAL — baseline only ({self.BASELINE_UP:.1%}, no validated condition active)"
        else:
            active.sort(key=lambda x: x[1], reverse=True)
            best_name, best_acc = active[0]
            active_names = " + ".join(name for name, _ in active)

            if best_acc >= 0.80:
                tier = 2
                reason = f"🥇 TIER 2 (holdout-validated {best_acc:.1%}) — {active_names}"
            elif best_acc >= 0.68:
                tier = 1
                reason = f"🥈 TIER 1 (holdout-validated {best_acc:.1%}) — {active_names}"
            else:
                tier = 0
                reason = f"⚪ WEAK — {active_names} ({best_acc:.1%}, near baseline)"
            accuracy = best_acc

        return tier, accuracy, reason, conditions


# ── INDICATORS ────────────────────────────────────────────────────────────────

def add_indicators(df):
    df = df.copy()
    # Preserve corn column if present before any operations
    corn_close = df['Corn_Close'].copy() if 'Corn_Close' in df.columns else None

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
    bb_mid           = df['Close'].rolling(20).mean()
    bb_std           = df['Close'].rolling(20).std()
    df['BB_Upper']   = bb_mid + 2 * bb_std
    df['BB_Lower']   = bb_mid - 2 * bb_std
    df['BB_Width']   = (bb_std * 2) / bb_mid
    df['Volatility'] = df['Returns'].rolling(20).std()
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    df['ATR']        = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df = df.dropna()

    # Re-attach corn after dropna (forward-fill any gaps)
    if corn_close is not None:
        df['Corn_Close'] = corn_close.reindex(df.index, method='ffill')

    return df


# ── ENSEMBLE MODELS ───────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Three models with daily-sensitive features.
    No frozen predictions — all three retrain fresh each run.
    """

    def __init__(self):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler_lstm = MinMaxScaler()
        self.scaler_ml   = MinMaxScaler()
        self.lstm_model  = None
        self.rf_model    = None
        self.xgb_model   = None
        self.seq_len     = 60
        self.features    = [
            'Close', 'Volume', 'Returns', 'SMA_20', 'SMA_50',
            'RSI', 'MACD', 'BB_Width', 'Volatility', 'ATR'
        ]

    def train(self, df):
        from keras.models import Sequential
        from keras.layers import LSTM as KerasLSTM, Dense, Dropout
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        print("   Training LSTM + RF + XGB...")

        # Labels: did price go up next day?
        y = np.array([
            1 if df['Close'].iloc[i] > df['Close'].iloc[i-1] else 0
            for i in range(self.seq_len, len(df))
        ])

        # LSTM data
        data_lstm   = df[self.features].values
        scaled_lstm = self.scaler_lstm.fit_transform(data_lstm)
        X_lstm      = np.array([scaled_lstm[i-self.seq_len:i] for i in range(self.seq_len, len(scaled_lstm))])

        # ML features — daily-sensitive, not frozen 60-day window
        ml_feat = self._build_ml_features(df)
        n       = len(y)
        ml_feat = ml_feat.iloc[-n:]
        X_ml    = self.scaler_ml.fit_transform(ml_feat.fillna(0))

        # Train LSTM
        self.lstm_model = Sequential([
            KerasLSTM(64, return_sequences=True, input_shape=(self.seq_len, len(self.features))),
            Dropout(0.2),
            KerasLSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1,  activation='sigmoid')
        ])
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
        self.lstm_model.fit(X_lstm, y, epochs=25, batch_size=32, validation_split=0.15, verbose=0)

        # Daily seed so RF/XGB vary each day
        seed = datetime.now(IL).timetuple().tm_yday

        self.rf_model = RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=5,
            random_state=seed, n_jobs=-1
        )
        self.rf_model.fit(X_ml, y)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            random_state=seed, use_label_encoder=False, eval_metric='logloss'
        )
        self.xgb_model.fit(X_ml, y, verbose=False)

        print("   ✓ All models trained")

    def _build_ml_features(self, df):
        f = pd.DataFrame(index=df.index)
        f['ret_1d']      = df['Close'].pct_change(1)
        f['ret_3d']      = df['Close'].pct_change(3)
        f['ret_5d']      = df['Close'].pct_change(5)
        f['ret_10d']     = df['Close'].pct_change(10)
        f['ret_20d']     = df['Close'].pct_change(20)
        sma5             = df['Close'].rolling(5).mean()
        sma10            = df['Close'].rolling(10).mean()
        f['sma5_vs_20']  = sma5  / df['SMA_20'] - 1
        f['sma10_vs_50'] = sma10 / df['SMA_50'] - 1
        f['above_sma20'] = (df['Close'] > df['SMA_20']).astype(float)
        f['above_sma50'] = (df['Close'] > df['SMA_50']).astype(float)
        f['rsi']         = df['RSI']
        f['rsi_change']  = df['RSI'].diff(3)
        f['macd']        = df['MACD']
        f['macd_change'] = df['MACD'].diff(3)
        f['atr_pct']     = df['ATR'] / df['Close']
        f['bb_width']    = df['BB_Width']
        vol_avg          = df['Volume'].rolling(20).mean()
        f['vol_ratio']   = df['Volume'] / vol_avg
        high10           = df['High'].rolling(10).max()
        low10            = df['Low'].rolling(10).min()
        f['range_pos']   = (df['Close'] - low10) / (high10 - low10 + 1e-6)
        f['volatility']  = df['Volatility']

        # ── Corn inter-market features (if available) ──
        # When wheat/corn ratio is high → wheat expensive vs corn → bearish wheat
        # When corn is rising → acreage competition → bullish wheat
        if 'Corn_Close' in df.columns:
            corn_close         = df['Corn_Close']
            wc_ratio           = df['Close'] / corn_close.replace(0, np.nan)
            wc_ratio_mean      = wc_ratio.rolling(60).mean()
            wc_ratio_std       = wc_ratio.rolling(60).std().replace(0, np.nan)
            f['wc_ratio_z']    = (wc_ratio - wc_ratio_mean) / wc_ratio_std
            f['corn_mom_3d']   = corn_close.pct_change(3)
            f['corn_mom_5d']   = corn_close.pct_change(5)

        return f.dropna()

    def predict(self, df):
        """
        REBUILT 2026-07-09 — fixed a real bug in how model outputs
        were combined.

        OLD BEHAVIOR (removed): each model's "weight" was set to
        abs(prediction - 0.5) — meaning the MORE EXTREME a model's
        guess, the MORE it controlled the final answer. On a real
        alert (2026-07-09), this meant XGB's 0.001 (essentially "0%
        chance", more likely a miscalibrated/overconfident output
        than genuine certainty) got ~50% of the total decision
        weight, while LSTM's honest, moderate 0.481 (near a genuine
        coin flip) got under 2% influence. Combined with an "all
        models agree" bonus, this produced a fake 92% confidence
        built almost entirely on the single most extreme number.
        Two days earlier, the same mechanism had produced a 92%+
        confidence in the OPPOSITE direction (RF/XGB near 0.95-0.998
        UP) — proof the models are unstable day to day, and the old
        formula was amplifying that instability into false certainty
        instead of damping it.

        NEW BEHAVIOR: equal weighting (simple average) — no model's
        opinion counts more just because it's extreme. Confidence is
        reported honestly, and real disagreement between models is
        surfaced explicitly (reliability flag) instead of hidden
        behind an agreement bonus.
        """
        # LSTM
        data   = df[self.features].values
        scaled = self.scaler_lstm.transform(data)
        X_lstm = np.array([scaled[-self.seq_len:]])
        lstm_p = float(self.lstm_model.predict(X_lstm, verbose=0)[0][0])

        # RF + XGB
        feat  = self._build_ml_features(df).iloc[[-1]]
        X_ml  = self.scaler_ml.transform(feat.fillna(0))
        rf_p  = float(self.rf_model.predict_proba(X_ml)[0][1])
        xgb_p = float(self.xgb_model.predict_proba(X_ml)[0][1])

        preds = [lstm_p, rf_p, xgb_p]

        # Equal-weighted average — no model gets extra say for being extreme
        weighted = float(np.mean(preds))

        votes_up = sum(1 for p in preds if p >= 0.5)
        direction = 'UP' if weighted >= 0.5 else 'DOWN'
        confidence = weighted if weighted >= 0.5 else 1 - weighted

        # Real disagreement measure — how spread out are the 3 opinions?
        spread = float(np.std(preds))

        agreement = 'FULL' if votes_up in [0, 3] else 'MAJORITY' if votes_up in [1, 2] else 'SPLIT'

        # If models disagree substantially, that's real information —
        # cap confidence instead of letting one extreme model dominate.
        # A high spread means "the models don't actually know," which
        # should LOWER stated confidence, not get averaged away.
        reliability = 'LOW' if spread > 0.35 else 'MODERATE' if spread > 0.15 else 'HIGH'
        if reliability == 'LOW':
            confidence = min(confidence, 0.60)  # don't claim high confidence when models sharply disagree

        return {
            'direction':   direction,
            'confidence':  confidence,
            'lstm':        lstm_p,
            'rf':          rf_p,
            'xgb':         xgb_p,
            'weighted':    weighted,
            'votes_up':    votes_up,
            'agreement':   agreement,
            'spread':      round(spread, 3),
            'reliability': reliability,
        }


# ── WASDE MULTI-GRAIN ─────────────────────────────────────────────────────────

def get_wasde_signal():
    """Fetch wheat, corn, soy from USDA. Derive wheat signal from all three."""
    api_key  = os.getenv("USDA_API_KEY", "3338B84E-694D-3E6A-925C-F35064C59BAE")
    base_url = "https://quickstats.nass.usda.gov/api/api_GET/"

    ANNUAL_USE = {'WHEAT': 2e9, 'CORN': 14.5e9, 'SOYBEANS': 4.4e9}
    STU_TIGHT  = {'WHEAT': 0.30, 'CORN': 0.10, 'SOYBEANS': 0.07}
    STU_AMPLE  = {'WHEAT': 0.33, 'CORN': 0.13, 'SOYBEANS': 0.10}

    grain_stu = {}
    for grain in ['WHEAT', 'CORN', 'SOYBEANS']:
        try:
            r = requests.get(base_url, params={
                'key': api_key, 'source_desc': 'SURVEY',
                'commodity_desc': grain, 'class_desc': 'ALL CLASSES',
                'statisticcat_desc': 'STOCKS', 'unit_desc': 'BU',
                'agg_level_desc': 'NATIONAL', 'format': 'JSON', 'year__GE': 2021,
            }, timeout=15)
            if r.status_code == 200:
                records = r.json().get('data', [])
                if records:
                    records = sorted(records, key=lambda x: x.get('year', 0), reverse=True)
                    val = float(records[0]['Value'].replace(',', ''))
                    grain_stu[grain] = val / ANNUAL_USE[grain]
        except Exception:
            pass

    if not grain_stu.get('WHEAT'):
        # Fallback: use wheat/corn ratio from yfinance
        return _wasde_market_proxy()

    score   = 0.0
    factors = []
    w_stu   = grain_stu['WHEAT']

    if w_stu < STU_TIGHT['WHEAT']:
        score += 0.20; factors.append(f"Wheat tight ({w_stu:.1%} STU)")
    elif w_stu > STU_AMPLE['WHEAT']:
        score -= 0.15; factors.append(f"Wheat ample ({w_stu:.1%} STU)")
    else:
        factors.append(f"Wheat balanced ({w_stu:.1%} STU)")

    for grain in ['CORN', 'SOYBEANS']:
        if grain in grain_stu:
            stu = grain_stu[grain]
            if stu < STU_TIGHT[grain]:
                score += 0.06; factors.append(f"{grain.title()} tight → acre competition")
            elif stu > STU_AMPLE[grain]:
                score -= 0.03

    signal = 'BULLISH' if score > 0.10 else 'BEARISH' if score < -0.05 else 'NEUTRAL'
    return {'signal': signal, 'score': round(score, 4),
            'stu': w_stu, 'factors': factors[:2], 'source': 'USDA LIVE'}


def _wasde_market_proxy():
    """Fallback: wheat/corn + wheat/soy ratio z-scores."""
    try:
        end   = datetime.now(IL)
        start = end - timedelta(days=400)
        wdf   = yf.Ticker(TICKER).history(start=start, end=end, auto_adjust=False)
        cdf   = yf.Ticker(CORN_TICKER).history(start=start, end=end, auto_adjust=False)

        if wdf.empty or cdf.empty:
            return {'signal': 'NEUTRAL', 'score': 0.0, 'stu': 0.0, 'factors': ['No data'], 'source': 'Proxy'}

        wc    = (wdf['Close'] / cdf['Close'].reindex(wdf.index, method='ffill')).dropna()
        z     = float((wc.iloc[-1] - wc.mean()) / wc.std())
        score = 0.12 if z > 0.75 else -0.08 if z < -0.75 else 0.0
        sig   = 'BULLISH' if score > 0 else 'BEARISH' if score < 0 else 'NEUTRAL'
        return {'signal': sig, 'score': score, 'stu': 0.0,
                'factors': [f"W/C ratio z={z:+.2f}"], 'source': 'Market proxy'}
    except Exception:
        return {'signal': 'NEUTRAL', 'score': 0.0, 'stu': 0.0, 'factors': [], 'source': 'Error'}


# ── WEATHER ───────────────────────────────────────────────────────────────────

def get_weather_signal():
    """Fetch weather for key wheat regions. Cache for 8 hours."""
    cache_file = Path("weather_cache.json")
    api_key    = os.getenv("VISUAL_CROSSING_API_KEY", "W2FNC8VKT94JKH9ZRZYHUE63P")

    # Use cache if fresh
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
            age   = (datetime.now(IL) - datetime.fromisoformat(cache['ts'])).total_seconds()
            if age < 28800:  # 8 hours
                return cache['data']
        except Exception:
            pass

    regions = {
        'Kansas': '38.5,-98.0', 'Oklahoma': '35.5,-98.0',
        'N.Dakota': '47.5,-100.5', 'Ukraine': '46.5,32.0',
        'Russia': '45.0,39.0', 'Canada': '52.0,-106.0',
    }

    scores = []
    for name, coords in regions.items():
        try:
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{coords}"
            end = datetime.now(IL)
            r   = requests.get(url, params={
                'key': api_key, 'unitGroup': 'metric', 'include': 'days',
                'elements': 'datetime,temp,tempmax,tempmin,precip',
                'contentType': 'json',
                'startDateTime': (end - timedelta(days=7)).strftime('%Y-%m-%d'),
                'endDateTime': end.strftime('%Y-%m-%d'),
            }, timeout=12)
            if r.status_code == 200:
                days   = r.json().get('days', [])
                precip = sum(d.get('precip', 0) for d in days)
                tmax   = max(d.get('tempmax', 20) for d in days)
                tmin   = min(d.get('tempmin', 0)  for d in days)
                month  = datetime.now(IL).month
                s      = 0.0
                if precip < 5:   s += 0.12
                if month in [5,6,7] and tmax > 35: s += 0.15
                if month in [12,1,2] and tmin < -10: s += 0.18
                scores.append(s)
        except Exception:
            pass

    if not scores:
        result = {'signal': 'NEUTRAL', 'score': 0.0, 'explanation': 'No data'}
    else:
        avg    = np.mean(scores)
        signal = 'BULLISH' if avg > 0.10 else 'BEARISH' if avg < -0.05 else 'NEUTRAL'
        result = {'signal': signal, 'score': round(avg, 4),
                  'explanation': f"{len(scores)}/6 regions checked"}

    try:
        cache_file.write_text(json.dumps({'ts': datetime.now(IL).isoformat(), 'data': result}))
    except Exception:
        pass

    return result


# ── VOLUME SIGNAL ─────────────────────────────────────────────────────────────

def get_volume_signal(df):
    vol_avg   = float(df['Volume'].rolling(20).mean().iloc[-1])
    vol_curr  = float(df['Volume'].iloc[-1])
    ratio     = vol_curr / vol_avg if vol_avg > 0 else 1.0
    ret       = float(df['Close'].pct_change(1).iloc[-1])

    if ratio > 1.5 and ret > 0:
        signal = 'BULLISH'
    elif ratio > 1.5 and ret < 0:
        signal = 'BEARISH'
    elif ratio < 0.7:
        signal = 'QUIET'
    else:
        signal = 'NEUTRAL'

    return {'signal': signal, 'ratio': round(ratio, 2),
            'explanation': f"{ratio:.1f}x average volume"}


# ── STATE ─────────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {'alerts_sent': 0, 'alerts_today': {}, 'last_alert_date': None}


def save_state(state):
    state['last_check'] = datetime.now(IL).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── ALERT GATE ────────────────────────────────────────────────────────────────

def should_send(state):
    """Only send at 1AM Israel time. Manual always sends."""
    force  = os.getenv('FORCE_ALERT', '').lower() in ('true', '1', 'yes')
    event  = os.getenv('GITHUB_EVENT_NAME', '')
    manual = force or 'workflow_dispatch' in event

    if manual:
        return True, "Manual trigger", True

    israel  = datetime.now(IL)
    il_hour = israel.hour
    il_date = israel.date().isoformat()

    if il_hour not in (1, 2):
        return False, f"Not scheduled hour ({il_hour}:00 Israel)", False

    slot_key = f"{il_date}_morning"
    if state.get('alerts_today', {}).get(slot_key):
        return False, "Morning alert already sent today", False

    return True, "Scheduled morning alert (01:00 Israel)", False


# ── TELEGRAM ──────────────────────────────────────────────────────────────────

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("Telegram not configured")
        return False
    try:
        # Send as plain text — no markdown parsing, no 400 errors
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": message},
            timeout=10
        )
        success = r.status_code == 200
        print(f"   Telegram: {'✓ sent' if success else '✗ failed'} ({r.status_code})")
        if not success:
            print(f"   Error: {r.text[:200]}")
        return success
    except Exception as e:
        print(f"   Telegram error: {e}")
        return False


# ── PERFORMANCE LOG ───────────────────────────────────────────────────────────

def log_prediction(direction, price, confidence, tier, seasonal_phase):
    log_file = Path("prediction_log.json")
    try:
        log = json.loads(log_file.read_text()) if log_file.exists() else []
    except Exception:
        log = []

    log.append({
        'timestamp':      datetime.now(IL).isoformat(),
        'direction':      direction,
        'entry_price':    price,
        'confidence':     confidence,
        'tier':           tier,
        'seasonal_phase': seasonal_phase,
        'validated':      False,
        'outcome':        None,
        'exit_reason':    None,
        'pnl_cents':      None,
    })

    log_file.write_text(json.dumps(log, indent=2))
    print(f"   Prediction logged: {direction} at {price:.2f}¢ (Tier {tier})")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print(f"WHEAT MONITOR v4.0")
    print(f"Time: {datetime.now(IL).strftime('%Y-%m-%d %H:%M:%S')} Israel")
    print(f"{'='*70}\n")

    state = load_state()

    send, reason, is_manual = should_send(state)
    print(f"Alert gate: {reason}")

    # ── Fetch 5 years of data ──
    print(f"\nFetching {TICKER} (5 years)...")
    end    = datetime.now(IL)
    start  = end - timedelta(days=5 * 365)
    df_raw = yf.Ticker(TICKER).history(start=start, end=end, auto_adjust=False)

    if df_raw.empty:
        print("ERROR: No data"); return

    # ── Fetch corn for inter-market features ──
    print(f"Fetching {CORN_TICKER} (corn correlation)...")
    try:
        corn_raw = yf.Ticker(CORN_TICKER).history(start=start, end=end, auto_adjust=False)
        if not corn_raw.empty:
            # Align corn to wheat index and add as column
            corn_aligned = corn_raw['Close'].reindex(df_raw.index, method='ffill')
            df_raw['Corn_Close'] = corn_aligned
            print(f"  Corn data: {len(corn_raw)} candles merged")
        else:
            print("  Corn data unavailable — inter-market features disabled")
    except Exception as e:
        print(f"  Corn fetch skipped: {e}")

    if df_raw.index[-1].date() == datetime.now(IL).date():
        df_raw = df_raw.iloc[:-1]

    last_candle_date  = df_raw.index[-1].date()
    days_since_candle = (datetime.now(IL).date() - last_candle_date).days

    if days_since_candle >= 3 and not is_manual:
        print(f"\nMarket closed — last candle {last_candle_date} ({days_since_candle}d ago). No alert.")
        save_state(state)
        return

    # ── FIX: use LIVE price for current_price, daily bars stay for indicators ──
    live_price, is_live_price = get_live_price()
    if is_live_price:
        current_price = live_price
        print(f"Price: {current_price:.2f}c  (LIVE — daily bar was {last_candle_date})")
    else:
        current_price = float(df_raw['Close'].iloc[-1])
        print(f"Price: {current_price:.2f}c  ⚠️ (STALE — daily bar {last_candle_date}, live fetch failed)")

    df = add_indicators(df_raw)

    # ── Engines ──
    print("\nRunning engines...")
    seasonal = SeasonalEngine()
    seasonal.fit(df)
    s_phase = seasonal.get_current_phase()
    print(f"  Seasonal: {s_phase['phase']} ({s_phase['confidence']:.0%}) — {s_phase['explanation']}")

    trend_engine = TrendEngine()
    trend_data   = trend_engine.get_trend(df)
    print(f"  Trend:    {trend_data['trend']} ({trend_data['strength']})")

    gate = ConvictionGate()
    tier, accuracy, gate_reason, gate_conds = gate.evaluate(df)
    print(f"  Gate:     {gate_reason}")

    # ── Signals ──
    print("\nFetching signals...")
    wasde   = get_wasde_signal()
    weather = get_weather_signal()
    volume  = get_volume_signal(df)
    print(f"  WASDE: {wasde['signal']} | Weather: {weather['signal']} | Vol: {volume['ratio']:.1f}x")

    # ── Ensemble ──
    print("\nTraining models...")
    ensemble = EnsemblePredictor()
    ensemble.train(df)
    pred      = ensemble.predict(df)
    direction = pred['direction']
    print(f"  Ensemble: {direction} | LSTM={pred['lstm']:.3f} RF={pred['rf']:.3f} XGB={pred['xgb']:.3f}")

    # ── Filters ──
    seasonal_blocked, _ = seasonal.blocks_direction(direction)
    trend_blocked, _    = trend_engine.blocks_direction(direction, trend_data)

    if seasonal_blocked:
        direction          = 'DOWN' if direction == 'UP' else 'UP'
        pred['confidence'] = 0.60
        print(f"  Seasonal override → {direction}")

    if trend_blocked:
        direction          = 'DOWN' if direction == 'UP' else 'UP'
        pred['confidence'] = 0.58
        print(f"  Trend filter → {direction}")

    # ── Cost floor ──
    print("\nCalculating cost floor...")
    cost_signal = None
    try:
        from cost_floor_analyzer import CostFloorAnalyzer
        cost_signal = CostFloorAnalyzer().get_floor_signal(current_price)
    except Exception as e:
        print(f"  Cost floor skipped: {e}")

    cost_floor_cents = cost_signal['floor_cents'] if cost_signal else None

    # ── Weekly range prediction ──
    print("\nBuilding weekly range prediction...")
    try:
        from weekly_range_engine import WeeklyRangeEngine
        wre = WeeklyRangeEngine()
        wre.fit(df, exclude_years=[2022])
        weekly  = wre.predict_next_week(df, current_price, cost_floor_cents)
        monthly = wre.predict_monthly_range(df, current_price, cost_floor_cents)

        print(f"  Weekly range: {weekly['range_low']:.0f} - {weekly['range_high']:.0f}c")
        print(f"  Weekly bias:  {weekly['bias']} ({weekly['confidence']:.0%})")
        print(f"  Monthly bias: {monthly['bias'] if monthly else 'N/A'}")

        # Use weekly bias to override/confirm ensemble direction
        if weekly['bias'] != 'NEUTRAL' and weekly['confidence'] >= 0.65:
            if weekly['bias'] != direction:
                print(f"  Weekly range bias ({weekly['bias']}) overrides ensemble ({direction})")
                direction = weekly['bias']

        # Build weekly message
        message = wre.format_alert(
            weekly      = weekly,
            monthly     = monthly,
            tier        = tier,
            gate_conds  = gate_conds,
            wasde       = wasde,
            weather     = weather,
            seasonal    = s_phase,
            cost_signal = cost_signal,
            gate_accuracy = accuracy,
            gate_reason   = gate_reason,
        )

        # Add ensemble footnote
        message += (
            f"\nMODELS (supporting data):\n"
            f"LSTM: {pred['lstm']:.3f} | RF: {pred['rf']:.3f} | XGB: {pred['xgb']:.3f}\n"
            f"Agreement: {pred['agreement']} | Trend: {trend_data['trend']}\n"
        )

        use_weekly = True

    except Exception as e:
        print(f"  Weekly engine error: {e}")
        import traceback; traceback.print_exc()
        use_weekly = False

    # ── Fallback to daily message if weekly fails ──
    if not use_weekly:
        stop    = current_price * (1 - STOP_PCT) if direction == 'UP' else current_price * (1 + STOP_PCT)
        target  = current_price * (1 + TARGET_PCT) if direction == 'UP' else current_price * (1 - TARGET_PCT)
        message = (
            f"WHEAT MONITOR v4.0\n"
            f"------------------------------\n"
            f"{direction} ({pred['confidence']:.1%})\n"
            f"Price: {current_price:.2f}c\n\n"
            f"SEASONAL: {s_phase['phase']} ({s_phase['confidence']:.0%})\n"
            f"WASDE: {wasde['signal']} | Weather: {weather['signal']}\n"
            f"MODELS: LSTM={pred['lstm']:.3f} RF={pred['rf']:.3f} XGB={pred['xgb']:.3f}\n\n"
            f"Entry: {current_price:.2f}c | Stop: {stop:.2f}c | Target: {target:.2f}c\n"
        )

    print(f"\nFINAL: {direction} | Tier {tier}")

    # ── Send ──
    if send:
        success = send_telegram(message)
        if success:
            state['alerts_sent'] = state.get('alerts_sent', 0) + 1
            state['last_alert_date'] = datetime.now(IL).date().isoformat()
            if not is_manual:
                slot_key = f"{datetime.now(IL).date().isoformat()}_morning"
                state.setdefault('alerts_today', {})[slot_key] = True
            log_prediction(direction, current_price, pred['confidence'], tier, s_phase['phase'])
    else:
        print(f"No alert: {reason}")

    state['last_direction'] = direction
    state['last_price']     = current_price
    save_state(state)

    print(f"\nTotal alerts sent: {state.get('alerts_sent', 0)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
