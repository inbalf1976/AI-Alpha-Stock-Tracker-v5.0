"""
Volume Analyzer - Detects unusual volume patterns
"""

import pandas as pd
import numpy as np


class VolumeAnalyzer:
    """Analyze volume patterns for conviction signals"""
    
    def __init__(self):
        self.lookback = 20
    
    def analyze_volume(self, df):
        """
        Analyze volume patterns
        
        Returns signal based on:
        - Volume vs average (high volume = conviction)
        - Volume trend (increasing = building momentum)
        """
        
        if 'Volume' not in df.columns or df['Volume'].isna().all():
            return self._get_neutral_signal()
        
        try:
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].rolling(window=self.lookback).mean().iloc[-1]
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume trend (last 5 days)
            recent_volume = df['Volume'].tail(5)
            volume_trend = 'up' if recent_volume.is_monotonic_increasing else \
                          'down' if recent_volume.is_monotonic_decreasing else 'flat'
            
            # Determine signal
            score = 0.0
            factors = []
            
            # High volume signals
            if volume_ratio > 1.5:
                score += 0.15
                factors.append(f"High volume ({volume_ratio:.1f}x average)")
            elif volume_ratio > 1.2:
                score += 0.08
                factors.append(f"Above average volume ({volume_ratio:.1f}x)")
            elif volume_ratio < 0.5:
                score -= 0.10
                factors.append(f"Low volume ({volume_ratio:.1f}x average) - weak conviction")
            else:
                factors.append(f"Normal volume ({volume_ratio:.1f}x average)")
            
            # Volume trend
            if volume_trend == 'up':
                score += 0.05
                factors.append("Volume trending up")
            elif volume_trend == 'down':
                score -= 0.05
                factors.append("Volume trending down")
            
            # Signal classification
            if score > 0.10:
                signal = 'BULLISH'
            elif score < -0.05:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'
            
            explanation = ', '.join(factors)
            
            return {
                'signal': signal,
                'score': score,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'explanation': explanation,
                'factors': factors
            }
            
        except Exception as e:
            print(f"Volume analysis error: {e}")
            return self._get_neutral_signal()
    
    def _get_neutral_signal(self):
        return {
            'signal': 'NEUTRAL',
            'score': 0.0,
            'volume_ratio': 1.0,
            'volume_trend': 'flat',
            'explanation': 'Volume data unavailable',
            'factors': ['Volume data unavailable']
        }
