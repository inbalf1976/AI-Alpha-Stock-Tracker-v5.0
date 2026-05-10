"""
Ensemble Predictor - Combines LSTM, Random Forest, and XGBoost
FIXED: Voting logic now correctly uses majority vote
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class EnsemblePredictor:
    """
    Ensemble of 3 models:
    1. LSTM (deep learning, sequence patterns)
    2. Random Forest (tree-based, non-linear)
    3. XGBoost (gradient boosting, trend detection)
    """
    
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.rf_model = None
        self.xgb_model = None
        
    def prepare_lstm_data(self, df):
        """Prepare data for LSTM (sequences)"""
        data = df[['Close']].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y)
    
    def prepare_ml_data(self, df):
        """Prepare data for Random Forest and XGBoost (features)"""
        features = ['Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 
                   'BB_Width', 'Volatility', 'ATR']
        
        X = df[features].values
        y = (df['Close'].shift(-1) > df['Close']).astype(int).values
        
        # Remove last row (no future data)
        X = X[:-1]
        y = y[:-1]
        
        return X, y
    
    def train_lstm(self, df):
        """Train LSTM model"""
        try:
            X, y = self.prepare_lstm_data(df)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Simple LSTM architecture
            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            self.lstm_model.compile(optimizer='adam', loss='mse')
            
            # Train with minimal epochs for speed
            self.lstm_model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            
            return True
        except Exception as e:
            print(f"   ❌ LSTM training failed: {e}")
            return False
    
    def train_random_forest(self, df):
        """Train Random Forest model"""
        try:
            X, y = self.prepare_ml_data(df)
            
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.rf_model.fit(X, y)
            return True
        except Exception as e:
            print(f"   ❌ Random Forest training failed: {e}")
            return False
    
    def train_xgboost(self, df):
        """Train XGBoost model"""
        try:
            X, y = self.prepare_ml_data(df)
            
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            self.xgb_model.fit(X, y)
            return True
        except Exception as e:
            print(f"   ❌ XGBoost training failed: {e}")
            return False
    
    def train_all_models(self, df):
        """Train all 3 models"""
        print("🤖 Training ensemble models...")
        
        print(" - Training LSTM...")
        lstm_success = self.train_lstm(df)
        
        print(" - Training Random Forest...")
        rf_success = self.train_random_forest(df)
        
        print(" - Training XGBoost...")
        xgb_success = self.train_xgboost(df)
        
        if lstm_success and rf_success and xgb_success:
            print("✓ All models trained")
            return True
        else:
            print("⚠️ Some models failed to train")
            return False
    
    def predict_lstm(self, df):
        """Get LSTM prediction"""
        if self.lstm_model is None:
            return 0.5
        
        try:
            data = df[['Close']].values
            scaled_data = self.scaler.transform(data)
            
            last_sequence = scaled_data[-self.lookback:]
            last_sequence = last_sequence.reshape((1, self.lookback, 1))
            
            prediction = self.lstm_model.predict(last_sequence, verbose=0)[0][0]
            return float(prediction)
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return 0.5
    
    def predict_rf(self, df):
        """Get Random Forest prediction"""
        if self.rf_model is None:
            return 0.5
        
        try:
            features = ['Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 
                       'BB_Width', 'Volatility', 'ATR']
            X = df[features].iloc[-1:].values
            
            prediction = self.rf_model.predict(X)[0]
            return float(prediction)
        except Exception as e:
            print(f"RF prediction error: {e}")
            return 0.5
    
    def predict_xgb(self, df):
        """Get XGBoost prediction"""
        if self.xgb_model is None:
            return 0.5
        
        try:
            features = ['Returns', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 
                       'BB_Width', 'Volatility', 'ATR']
            X = df[features].iloc[-1:].values
            
            prediction = self.xgb_model.predict(X)[0]
            return float(prediction)
        except Exception as e:
            print(f"XGBoost prediction error: {e}")
            return 0.5
    
    def predict_ensemble(self, df):
        """
        Get ensemble prediction with FIXED voting logic
        
        Returns:
        - direction: 'UP' or 'DOWN' based on MAJORITY vote
        - confidence: weighted average of model confidences
        - votes_up: number of models predicting UP
        - agreement: 'FULL' if all agree, 'MAJORITY' if 2/3, 'SPLIT' if 50/50
        """
        
        # Get predictions from all 3 models
        lstm_pred = self.predict_lstm(df)
        rf_pred = self.predict_rf(df)
        xgb_pred = self.predict_xgb(df)
        
        # Convert to binary votes (threshold = 0.5)
        lstm_vote = 'UP' if lstm_pred > 0.5 else 'DOWN'
        rf_vote = 'UP' if rf_pred > 0.5 else 'DOWN'
        xgb_vote = 'UP' if xgb_pred > 0.5 else 'DOWN'
        
        # Count UP votes
        votes_up = sum([1 for v in [lstm_vote, rf_vote, xgb_vote] if v == 'UP'])
        
        # ✅ FIX: Use MAJORITY VOTE, not weighted average
        if votes_up >= 2:
            direction = 'UP'
        else:
            direction = 'DOWN'
        
        # Determine agreement level
        if votes_up == 3 or votes_up == 0:
            agreement = 'FULL'
        elif votes_up == 2 or votes_up == 1:
            agreement = 'MAJORITY'
        else:
            agreement = 'SPLIT'
        
        # Calculate confidence based on how far from 0.5 threshold
        # Higher confidence when models are more certain
        lstm_conf = abs(lstm_pred - 0.5) * 2  # 0 to 1 scale
        rf_conf = abs(rf_pred - 0.5) * 2
        xgb_conf = abs(xgb_pred - 0.5) * 2
        
        # Average confidence, boosted by agreement
        base_confidence = (lstm_conf + rf_conf + xgb_conf) / 3
        
        # Boost confidence if models agree
        if agreement == 'FULL':
            confidence = min(1.0, base_confidence * 1.2)  # 20% boost for full agreement
        elif agreement == 'MAJORITY':
            confidence = base_confidence
        else:
            confidence = base_confidence * 0.8  # Reduce for split decision
        
        # Ensure minimum confidence
        confidence = max(0.5, confidence)
        
        # Model details for debugging
        model_details = {
            'LSTM': f"{'↑' if lstm_vote == 'UP' else '↓'} {lstm_pred:.3f}",
            'RandomForest': f"{'↑' if rf_vote == 'UP' else '↓'} {rf_pred:.3f}",
            'XGBoost': f"{'↑' if xgb_vote == 'UP' else '↓'} {xgb_pred:.3f}"
        }
        
        return {
            'direction': direction,
            'confidence': confidence,
            'votes_up': votes_up,
            'agreement': agreement,
            'lstm_pred': lstm_pred,
            'rf_pred': rf_pred,
            'xgb_pred': xgb_pred,
            'model_details': model_details
        }
