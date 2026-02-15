"""
Ensemble Predictor - Combines 3 Machine Learning Models
LSTM (neural network) + Random Forest + XGBoost
Voting system for higher accuracy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """
    Ensemble model combining 3 algorithms:
    1. LSTM - Captures sequential patterns
    2. Random Forest - Non-linear relationships
    3. XGBoost - Gradient boosting power
    """
    
    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.xgb_model = None
        
        self.scaler_lstm = MinMaxScaler()
        self.scaler_ml = MinMaxScaler()
        
        self.seq_len = 60
        self.features = [
            'Close', 'Volume', 'Returns',
            'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'BB_Width', 'Volatility', 'ATR'
        ]
    
    def train_all_models(self, df):
        """Train all 3 models"""
        print("🤖 Training ensemble models...")
        
        # Prepare data
        X_lstm, y = self._prepare_lstm_data(df)
        X_ml = self._prepare_ml_data(df)
        
        # Train LSTM
        print("  - Training LSTM...")
        self.lstm_model = self._train_lstm(X_lstm, y)
        
        # Train Random Forest
        print("  - Training Random Forest...")
        self.rf_model = self._train_random_forest(X_ml, y)
        
        # Train XGBoost
        print("  - Training XGBoost...")
        self.xgb_model = self._train_xgboost(X_ml, y)
        
        print("✓ All models trained")
    
    def _prepare_lstm_data(self, df):
        """Prepare data for LSTM (sequential)"""
        data = df[self.features].values
        scaled = self.scaler_lstm.fit_transform(data)
        
        X, y = [], []
        for i in range(self.seq_len, len(scaled)):
            X.append(scaled[i-self.seq_len:i])
            y.append(1 if df['Close'].iloc[i] > df['Close'].iloc[i-1] else 0)
        
        return np.array(X), np.array(y)
    
    def _prepare_ml_data(self, df):
        """Prepare data for ML models (non-sequential)"""
        # Use last 60 days flattened + additional features
        data = df[self.features].values
        scaled = self.scaler_ml.fit_transform(data)
        
        X = []
        for i in range(self.seq_len, len(scaled)):
            # Take last N days and flatten
            sequence = scaled[i-self.seq_len:i].flatten()
            X.append(sequence)
        
        return np.array(X)
    
    def _train_lstm(self, X, y):
        """Train LSTM model"""
        if len(X) < 100:
            raise ValueError("Not enough data")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
        
        return model
    
    def _train_random_forest(self, X, y):
        """Train Random Forest model"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        return model
    
    def _train_xgboost(self, X, y):
        """Train XGBoost model"""
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X, y, verbose=False)
        return model
    
    def predict_ensemble(self, df):
        """
        Make prediction using all 3 models
        Returns: Combined prediction with confidence
        """
        # Prepare data for prediction
        lstm_data = df[self.features].tail(self.seq_len).values
        lstm_scaled = self.scaler_lstm.transform(lstm_data)
        X_lstm = np.array([lstm_scaled])
        
        ml_data = df[self.features].values
        ml_scaled = self.scaler_ml.transform(ml_data)
        X_ml = ml_scaled[-self.seq_len:].flatten().reshape(1, -1)
        
        # Get predictions from each model
        lstm_pred = self.lstm_model.predict(X_lstm, verbose=0)[0][0]
        rf_pred = self.rf_model.predict_proba(X_ml)[0][1]
        xgb_pred = self.xgb_model.predict_proba(X_ml)[0][1]
        
        # Calculate weights based on model confidence
        lstm_confidence = abs(lstm_pred - 0.5) * 2  # 0-1 scale
        rf_confidence = abs(rf_pred - 0.5) * 2
        xgb_confidence = abs(xgb_pred - 0.5) * 2
        
        total_confidence = lstm_confidence + rf_confidence + xgb_confidence
        
        if total_confidence > 0:
            # Weighted average
            weighted_pred = (
                lstm_pred * lstm_confidence +
                rf_pred * rf_confidence +
                xgb_pred * xgb_confidence
            ) / total_confidence
        else:
            # Simple average
            weighted_pred = (lstm_pred + rf_pred + xgb_pred) / 3
        
        # Voting system (for extra confirmation)
        votes_up = sum([1 for p in [lstm_pred, rf_pred, xgb_pred] if p >= 0.5])
        votes_down = 3 - votes_up
        
        # Determine direction
        direction = "UP" if weighted_pred >= 0.5 else "DOWN"
        
        # Calculate confidence
        base_confidence = weighted_pred if weighted_pred >= 0.5 else (1 - weighted_pred)
        
        # Boost confidence if all 3 models agree
        if votes_up == 3 or votes_down == 3:
            agreement_boost = 0.15
        elif votes_up == 2 or votes_down == 2:
            agreement_boost = 0.05
        else:
            agreement_boost = 0.0
        
        final_confidence = min(1.0, base_confidence + agreement_boost)
        
        return {
            'direction': direction,
            'confidence': final_confidence,
            'lstm_pred': float(lstm_pred),
            'rf_pred': float(rf_pred),
            'xgb_pred': float(xgb_pred),
            'weighted_pred': float(weighted_pred),
            'votes_up': votes_up,
            'votes_down': votes_down,
            'agreement': 'FULL' if votes_up == 3 or votes_down == 3 else 'MAJORITY' if votes_up == 2 or votes_down == 2 else 'SPLIT',
            'model_details': {
                'LSTM': f"{lstm_pred:.3f}",
                'RandomForest': f"{rf_pred:.3f}",
                'XGBoost': f"{xgb_pred:.3f}"
            }
        }

# Quick test
if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # Fetch test data
    ticker = yf.Ticker("ZW=F")
    df = ticker.history(start=datetime.now()-timedelta(days=730), end=datetime.now())
    
    # Add indicators
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    delta = df['Close'].diff()
    gain = (delta.where(delta>0,0)).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+gain/loss))
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Width'] = (bb_std * 2) / df['BB_Middle']
    df['Volatility'] = df['Returns'].rolling(20).std()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High']-df['Close'].shift())
    low_close = np.abs(df['Low']-df['Close'].shift())
    ranges = pd.concat([high_low,high_close,low_close],axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df = df.dropna()
    
    # Train and predict
    ensemble = EnsemblePredictor()
    ensemble.train_all_models(df)
    prediction = ensemble.predict_ensemble(df)
    
    print(f"\nEnsemble Prediction: {prediction}")
