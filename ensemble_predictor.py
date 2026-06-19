"""
Ensemble Predictor - FIXED VERSION
====================================
FIXES vs original:
  1. RF and XGBoost now use recent price features (5/10/20 day returns)
     instead of full 60-day flattened window — much more sensitive to
     daily market changes, no more frozen predictions
  2. Trend filter added — blocks DOWN signals when price is above 5-day
     AND 10-day MA (don't fight a clear uptrend)
  3. Agreement boost capped — FULL agreement adds 0.10 not 0.15,
     prevents runaway confidence inflation
  4. random_state uses current date — ensures RF/XGB retrain differently
     each day rather than producing identical outputs
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class EnsemblePredictor:

    def __init__(self):
        self.lstm_model  = None
        self.rf_model    = None
        self.xgb_model   = None

        self.scaler_lstm = MinMaxScaler()
        self.scaler_ml   = MinMaxScaler()

        self.seq_len  = 60
        self.features = [
            'Close', 'Volume', 'Returns',
            'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'BB_Width', 'Volatility', 'ATR'
        ]

    # ── training ──────────────────────────────────────────────────────────────

    def train_all_models(self, df):
        print("🤖 Training ensemble models...")

        X_lstm, y = self._prepare_lstm_data(df)
        X_ml      = self._prepare_ml_features(df)   # ← new feature set

        print("  - Training LSTM...")
        self.lstm_model = self._train_lstm(X_lstm, y)

        print("  - Training Random Forest...")
        self.rf_model = self._train_random_forest(X_ml, y)

        print("  - Training XGBoost...")
        self.xgb_model = self._train_xgboost(X_ml, y)

        print("✓ All models trained")

    def _prepare_lstm_data(self, df):
        """LSTM: sequential window — unchanged from original."""
        data   = df[self.features].values
        scaled = self.scaler_lstm.fit_transform(data)

        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i-self.seq_len:i])
            y.append(1 if df['Close'].iloc[i] > df['Close'].iloc[i-1] else 0)

        return np.array(X), np.array(y)

    def _prepare_ml_features(self, df):
        """
        FIX: RF and XGBoost now use interpretable daily features
        instead of the flattened 60-day window.

        These features change meaningfully every day so the models
        are sensitive to current market conditions.
        """
        feat = pd.DataFrame(index=df.index)

        # Price momentum at multiple timeframes
        feat['ret_1d']  = df['Close'].pct_change(1)
        feat['ret_3d']  = df['Close'].pct_change(3)
        feat['ret_5d']  = df['Close'].pct_change(5)
        feat['ret_10d'] = df['Close'].pct_change(10)
        feat['ret_20d'] = df['Close'].pct_change(20)

        # Trend
        feat['sma5']    = df['Close'].rolling(5).mean()
        feat['sma10']   = df['Close'].rolling(10).mean()
        feat['above_sma20'] = (df['Close'] > df['SMA_20']).astype(float)
        feat['above_sma50'] = (df['Close'] > df['SMA_50']).astype(float)
        feat['sma5_vs_20']  = feat['sma5'] / df['SMA_20'] - 1
        feat['sma10_vs_50'] = feat['sma10'] / df['SMA_50'] - 1

        # Momentum indicators
        feat['rsi']         = df['RSI']
        feat['rsi_change']  = df['RSI'].diff(3)
        feat['macd']        = df['MACD']
        feat['macd_change'] = df['MACD'].diff(3)

        # Volatility
        feat['atr_pct']     = df['ATR'] / df['Close']
        feat['bb_width']    = df['BB_Width']
        feat['volatility']  = df['Volatility']

        # Volume
        vol_avg = df['Volume'].rolling(20).mean()
        feat['vol_ratio']   = df['Volume'] / vol_avg

        # Price position in recent range
        feat['high10'] = df['High'].rolling(10).max()
        feat['low10']  = df['Low'].rolling(10).min()
        feat['range_pos'] = (df['Close'] - feat['low10']) / (feat['high10'] - feat['low10'] + 1e-6)

        feat = feat.dropna()

        # Align with LSTM y labels: y has (len(df) - seq_len) rows
        # so we take the last (len(df) - seq_len) rows of feat
        n_labels = len(df) - self.seq_len
        feat = feat.iloc[-n_labels:]

        # Scale
        scaled = self.scaler_ml.fit_transform(feat.fillna(0))
        return scaled

    def _train_lstm(self, X, y):
        if len(X) < 100:
            raise ValueError("Not enough data for LSTM")

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1,  activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
        return model

    def _train_random_forest(self, X, y):
        # FIX: random_state uses day-of-year so it varies daily
        daily_seed = datetime.now().timetuple().tm_yday

        # Align X rows with y
        X_aligned = X[-len(y):]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=daily_seed,
            n_jobs=-1
        )
        model.fit(X_aligned, y)
        return model

    def _train_xgboost(self, X, y):
        daily_seed = datetime.now().timetuple().tm_yday

        X_aligned = X[-len(y):]

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=daily_seed,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_aligned, y, verbose=False)
        return model

    # ── prediction ────────────────────────────────────────────────────────────

    def predict_ensemble(self, df):
        """
        Make prediction using all 3 models with trend filter.
        Trend filter: blocks DOWN signals when price is clearly above
        5-day and 10-day MA — don't fight an uptrend.
        """
        # LSTM prediction
        lstm_data   = df[self.features].tail(self.seq_len).values
        lstm_scaled = self.scaler_lstm.transform(lstm_data)
        X_lstm      = np.array([lstm_scaled])
        lstm_pred   = float(self.lstm_model.predict(X_lstm, verbose=0)[0][0])

        # RF + XGB prediction using new feature set
        feat_df  = self._build_prediction_features(df)
        X_ml     = self.scaler_ml.transform(feat_df)
        rf_pred  = float(self.rf_model.predict_proba(X_ml)[0][1])
        xgb_pred = float(self.xgb_model.predict_proba(X_ml)[0][1])

        # Weighted average by confidence
        lstm_conf = abs(lstm_pred - 0.5) * 2
        rf_conf   = abs(rf_pred   - 0.5) * 2
        xgb_conf  = abs(xgb_pred  - 0.5) * 2
        total     = lstm_conf + rf_conf + xgb_conf

        weighted_pred = (
            (lstm_pred * lstm_conf + rf_pred * rf_conf + xgb_pred * xgb_conf) / total
            if total > 0 else
            (lstm_pred + rf_pred + xgb_pred) / 3
        )

        # Votes
        votes_up   = sum(1 for p in [lstm_pred, rf_pred, xgb_pred] if p >= 0.5)
        votes_down = 3 - votes_up

        # Initial direction
        direction = "UP" if weighted_pred >= 0.5 else "DOWN"

        # ── TREND FILTER ──────────────────────────────────────────────────────
        # If price is clearly above 5-day AND 10-day MA → block DOWN signal
        # This prevents calling DOWN into a sustained uptrend
        price   = float(df['Close'].iloc[-1])
        sma5    = float(df['Close'].rolling(5).mean().iloc[-1])
        sma10   = float(df['Close'].rolling(10).mean().iloc[-1])
        sma20   = float(df['SMA_20'].iloc[-1])

        in_uptrend   = price > sma5 and price > sma10 and sma5 > sma20
        in_downtrend = price < sma5 and price < sma10 and sma5 < sma20

        trend_override = None
        if direction == "DOWN" and in_uptrend:
            direction      = "UP"
            trend_override = f"Trend filter: DOWN blocked (price {price:.1f} > SMA5 {sma5:.1f} > SMA20 {sma20:.1f})"
            # Reduce confidence to reflect override
            weighted_pred  = 0.55
        elif direction == "UP" and in_downtrend:
            direction      = "DOWN"
            trend_override = f"Trend filter: UP blocked (price {price:.1f} < SMA5 {sma5:.1f} < SMA20 {sma20:.1f})"
            weighted_pred  = 0.55
        # ─────────────────────────────────────────────────────────────────────

        # Confidence
        base_confidence = weighted_pred if weighted_pred >= 0.5 else (1 - weighted_pred)

        # FIX: smaller agreement boost (was 0.15, now 0.08)
        if votes_up == 3 or votes_down == 3:
            agreement_boost = 0.08
        elif votes_up == 2 or votes_down == 2:
            agreement_boost = 0.03
        else:
            agreement_boost = 0.0

        final_confidence = min(0.95, base_confidence + agreement_boost)

        agreement = (
            'FULL'     if votes_up == 3 or votes_down == 3 else
            'MAJORITY' if votes_up == 2 or votes_down == 2 else
            'SPLIT'
        )

        if trend_override:
            print(f"   ⚡ {trend_override}")

        return {
            'direction':     direction,
            'confidence':    final_confidence,
            'lstm_pred':     lstm_pred,
            'rf_pred':       rf_pred,
            'xgb_pred':      xgb_pred,
            'weighted_pred': weighted_pred,
            'votes_up':      votes_up,
            'votes_down':    votes_down,
            'agreement':     agreement,
            'trend_override': trend_override,
            'in_uptrend':    in_uptrend,
            'in_downtrend':  in_downtrend,
            'model_details': {
                'LSTM':         f"{lstm_pred:.3f}",
                'RandomForest': f"{rf_pred:.3f}",
                'XGBoost':      f"{xgb_pred:.3f}",
            }
        }

    def _build_prediction_features(self, df):
        """Build the same feature set used in training, for a single prediction row."""
        feat = {}

        feat['ret_1d']  = df['Close'].pct_change(1).iloc[-1]
        feat['ret_3d']  = df['Close'].pct_change(3).iloc[-1]
        feat['ret_5d']  = df['Close'].pct_change(5).iloc[-1]
        feat['ret_10d'] = df['Close'].pct_change(10).iloc[-1]
        feat['ret_20d'] = df['Close'].pct_change(20).iloc[-1]

        sma5  = df['Close'].rolling(5).mean().iloc[-1]
        sma10 = df['Close'].rolling(10).mean().iloc[-1]
        feat['sma5']        = sma5
        feat['sma10']       = sma10
        feat['above_sma20'] = float(df['Close'].iloc[-1] > df['SMA_20'].iloc[-1])
        feat['above_sma50'] = float(df['Close'].iloc[-1] > df['SMA_50'].iloc[-1])
        feat['sma5_vs_20']  = sma5 / df['SMA_20'].iloc[-1] - 1
        feat['sma10_vs_50'] = sma10 / df['SMA_50'].iloc[-1] - 1

        feat['rsi']         = df['RSI'].iloc[-1]
        feat['rsi_change']  = df['RSI'].diff(3).iloc[-1]
        feat['macd']        = df['MACD'].iloc[-1]
        feat['macd_change'] = df['MACD'].diff(3).iloc[-1]

        feat['atr_pct']    = df['ATR'].iloc[-1] / df['Close'].iloc[-1]
        feat['bb_width']   = df['BB_Width'].iloc[-1]
        feat['volatility'] = df['Volatility'].iloc[-1]

        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        feat['vol_ratio'] = df['Volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1.0

        high10 = df['High'].rolling(10).max().iloc[-1]
        low10  = df['Low'].rolling(10).min().iloc[-1]
        feat['high10']    = high10
        feat['low10']     = low10
        feat['range_pos'] = (df['Close'].iloc[-1] - low10) / (high10 - low10 + 1e-6)

        return pd.DataFrame([feat])


# ── standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yfinance as yf
    from datetime import timedelta
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ticker = yf.Ticker("ZW=F")
    df = ticker.history(
        start=datetime.now() - timedelta(days=730),
        end=datetime.now(),
        auto_adjust=False
    )
    df = df.iloc[:-1]  # drop incomplete candle

    df['Returns']   = df['Close'].pct_change()
    df['SMA_20']    = df['Close'].rolling(20).mean()
    df['SMA_50']    = df['Close'].rolling(50).mean()
    df['EMA_12']    = df['Close'].ewm(span=12).mean()
    df['EMA_26']    = df['Close'].ewm(span=26).mean()
    df['MACD']      = df['EMA_12'] - df['EMA_26']
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']       = 100 - (100 / (1 + gain / loss))
    bb_mid          = df['Close'].rolling(20).mean()
    bb_std          = df['Close'].rolling(20).std()
    df['BB_Width']  = (bb_std * 2) / bb_mid
    df['Volatility']= df['Returns'].rolling(20).std()
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df = df.dropna()

    ensemble = EnsemblePredictor()
    ensemble.train_all_models(df)
    pred = ensemble.predict_ensemble(df)

    print(f"\nDirection:   {pred['direction']}")
    print(f"Confidence:  {pred['confidence']:.1%}")
    print(f"Agreement:   {pred['agreement']}")
    print(f"LSTM:        {pred['lstm_pred']:.3f}")
    print(f"RF:          {pred['rf_pred']:.3f}")
    print(f"XGB:         {pred['xgb_pred']:.3f}")
    print(f"Uptrend:     {pred['in_uptrend']}")
    print(f"Downtrend:   {pred['in_downtrend']}")
    if pred['trend_override']:
        print(f"Override:    {pred['trend_override']}")
