"""
Volume Analyzer - Trading Volume and Open Interest Analysis
Detects institutional money flow and momentum shifts
"""

import pandas as pd
import numpy as np

class VolumeAnalyzer:
    """Analyze trading volume patterns for confirmation signals"""
    
    def __init__(self):
        self.lookback = 20  # Days for average
    
    def analyze_volume(self, df):
        """
        Analyze volume patterns and generate signal
        
        Args:
            df: DataFrame with 'Volume', 'Close', 'High', 'Low' columns
        
        Returns:
            dict with volume signal and confidence
        """
        try:
            # Calculate volume metrics
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].rolling(window=self.lookback).mean().iloc[-1]
            volume_std = df['Volume'].rolling(window=self.lookback).std().iloc[-1]
            
            # Price change
            price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Analyze patterns
            signal = self._interpret_volume(volume_ratio, price_change, df)
            
            return signal
        
        except Exception as e:
            print(f"Volume analysis error: {e}")
            return {
                'signal': 'NEUTRAL',
                'score': 0.0,
                'confidence': 0.5,
                'volume_ratio': 1.0,
                'explanation': 'Volume data unavailable'
            }
    
    def _interpret_volume(self, volume_ratio, price_change, df):
        """Interpret volume patterns"""
        
        score = 0.0
        confidence = 0.5
        factors = []
        
        # HIGH VOLUME PATTERNS
        if volume_ratio > 2.0:  # Volume spike (2x average)
            if price_change > 0.01:  # Price up + high volume = bullish
                score += 0.15
                confidence += 0.15
                factors.append('Strong buying pressure')
            elif price_change < -0.01:  # Price down + high volume = bearish
                score -= 0.15
                confidence += 0.15
                factors.append('Strong selling pressure')
            else:
                factors.append('High volume, indecisive price')
        
        elif volume_ratio > 1.5:  # Above average volume
            if price_change > 0.005:
                score += 0.10
                confidence += 0.08
                factors.append('Good buying interest')
            elif price_change < -0.005:
                score -= 0.10
                confidence += 0.08
                factors.append('Selling pressure')
        
        # LOW VOLUME PATTERNS
        elif volume_ratio < 0.5:  # Very low volume
            score -= 0.05  # Low conviction
            confidence -= 0.10
            factors.append('Low volume - weak conviction')
        
        # VOLUME-PRICE DIVERGENCE
        recent_prices = df['Close'].tail(5)
        recent_volumes = df['Volume'].tail(5)
        
        if recent_prices.is_monotonic_increasing and recent_volumes.is_monotonic_decreasing:
            score -= 0.12  # Bearish divergence
            factors.append('Bearish divergence: Price up, volume down')
        elif recent_prices.is_monotonic_decreasing and recent_volumes.is_monotonic_increasing:
            score += 0.12  # Bullish divergence (capitulation)
            factors.append('Possible capitulation: Price down, volume up')
        
        # VOLUME TREND
        volume_trend = self._calculate_volume_trend(df)
        if volume_trend > 0.2:
            score += 0.08
            factors.append('Volume trending up')
        elif volume_trend < -0.2:
            score -= 0.05
            factors.append('Volume trending down')
        
        # Determine signal
        if score > 0.10:
            signal = 'BULLISH'
        elif score < -0.10:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'signal': signal,
            'score': score,
            'confidence': max(0.5, min(1.0, confidence)),
            'volume_ratio': volume_ratio,
            'factors': factors,
            'explanation': self._generate_explanation(signal, volume_ratio, factors)
        }
    
    def _calculate_volume_trend(self, df):
        """Calculate volume trend over recent period"""
        try:
            recent_vol = df['Volume'].tail(10).mean()
            older_vol = df['Volume'].tail(30).head(10).mean()
            
            if older_vol > 0:
                return (recent_vol - older_vol) / older_vol
            return 0.0
        except:
            return 0.0
    
    def _generate_explanation(self, signal, volume_ratio, factors):
        """Generate human-readable explanation"""
        
        if volume_ratio > 2.0:
            vol_desc = f"Volume spike ({volume_ratio:.1f}x average)"
        elif volume_ratio > 1.5:
            vol_desc = f"Above average volume ({volume_ratio:.1f}x)"
        elif volume_ratio < 0.5:
            vol_desc = f"Low volume ({volume_ratio:.1f}x average)"
        else:
            vol_desc = f"Normal volume ({volume_ratio:.1f}x)"
        
        if factors:
            return f"{vol_desc} - {', '.join(factors[:2])}"
        else:
            return vol_desc
    
    def get_accumulation_distribution(self, df):
        """
        Calculate Accumulation/Distribution indicator
        Shows if smart money is accumulating or distributing
        """
        try:
            # Money Flow Multiplier
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mfm = mfm.fillna(0)
            
            # Money Flow Volume
            mfv = mfm * df['Volume']
            
            # Accumulation/Distribution Line
            ad_line = mfv.cumsum()
            
            # Check if trending up or down
            recent_ad = ad_line.tail(10).mean()
            older_ad = ad_line.tail(30).head(10).mean()
            
            if recent_ad > older_ad * 1.05:
                return {'signal': 'ACCUMULATION', 'score': 0.10}
            elif recent_ad < older_ad * 0.95:
                return {'signal': 'DISTRIBUTION', 'score': -0.10}
            else:
                return {'signal': 'NEUTRAL', 'score': 0.0}
        
        except:
            return {'signal': 'NEUTRAL', 'score': 0.0}

# Quick test
if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    from datetime import datetime, timedelta
    
    ticker = yf.Ticker("ZW=F")
    df = ticker.history(start=datetime.now()-timedelta(days=90), end=datetime.now())
    
    analyzer = VolumeAnalyzer()
    signal = analyzer.analyze_volume(df)
    print(f"Volume Signal: {signal}")
    
    ad = analyzer.get_accumulation_distribution(df)
    print(f"A/D Signal: {ad}")
