"""
Move Analyzer - Historical Price Movement Analysis
Provides realistic exit recommendations based on actual wheat behavior
"""

import pandas as pd
import numpy as np

class MoveAnalyzer:
    """Analyze historical price movements to set realistic expectations"""
    
    def __init__(self):
        self.lookback_days = 30  # Analyze last 30 trading days
    
    def analyze_typical_moves(self, df, direction):
        """
        Analyze how far wheat typically moves in given direction
        
        Args:
            df: Price DataFrame
            direction: 'UP' or 'DOWN'
        
        Returns:
            dict with move statistics and recommendations
        """
        try:
            # Calculate daily moves
            df['Daily_High_Move'] = ((df['High'] - df['Open']) / df['Open']) * 100
            df['Daily_Low_Move'] = ((df['Low'] - df['Open']) / df['Open']) * 100
            df['Daily_Close_Move'] = ((df['Close'] - df['Open']) / df['Open']) * 100
            
            # Get recent period
            recent = df.tail(self.lookback_days)
            
            if direction == 'UP':
                # Analyze upward moves
                up_days = recent[recent['Close'] > recent['Open']]
                
                if len(up_days) < 5:  # Not enough data
                    return self._get_default_stats('UP')
                
                high_moves = up_days['Daily_High_Move'].values
                close_moves = up_days['Daily_Close_Move'].values
                
                stats = {
                    'direction': 'UP',
                    'sample_size': len(up_days),
                    'avg_intraday_high': np.mean(high_moves),
                    'avg_close_move': np.mean(close_moves),
                    'percentile_50': np.percentile(close_moves, 50),  # Median
                    'percentile_75': np.percentile(close_moves, 75),  # Upper quartile
                    'percentile_90': np.percentile(close_moves, 90),  # Rare big moves
                    'max_move': np.max(close_moves),
                    'win_days': len(up_days),
                    'total_days': len(recent)
                }
            
            else:  # DOWN
                # Analyze downward moves
                down_days = recent[recent['Close'] < recent['Open']]
                
                if len(down_days) < 5:
                    return self._get_default_stats('DOWN')
                
                low_moves = abs(down_days['Daily_Low_Move'].values)
                close_moves = abs(down_days['Daily_Close_Move'].values)
                
                stats = {
                    'direction': 'DOWN',
                    'sample_size': len(down_days),
                    'avg_intraday_low': np.mean(low_moves),
                    'avg_close_move': np.mean(close_moves),
                    'percentile_50': np.percentile(close_moves, 50),
                    'percentile_75': np.percentile(close_moves, 75),
                    'percentile_90': np.percentile(close_moves, 90),
                    'max_move': np.max(close_moves),
                    'win_days': len(down_days),
                    'total_days': len(recent)
                }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(stats, direction)
            stats['recommendations'] = recommendations
            
            return stats
        
        except Exception as e:
            print(f"Move analysis error: {e}")
            return self._get_default_stats(direction)
    
    def _generate_recommendations(self, stats, direction):
        """Generate practical exit recommendations"""
        
        avg = stats['avg_close_move']
        p75 = stats['percentile_75']
        p90 = stats['percentile_90']
        
        return {
            'conservative': {
                'percent': round(avg, 2),
                'description': 'Average move - Most common',
                'probability': '~50%'
            },
            'moderate': {
                'percent': round(p75, 2),
                'description': 'Strong move - Upper quartile',
                'probability': '~25%'
            },
            'aggressive': {
                'percent': round(p90, 2),
                'description': 'Exceptional move - Rare',
                'probability': '~10%'
            }
        }
    
    def _get_default_stats(self, direction):
        """Return default stats when not enough data"""
        return {
            'direction': direction,
            'sample_size': 0,
            'avg_close_move': 1.2,
            'percentile_75': 1.8,
            'percentile_90': 2.5,
            'recommendations': {
                'conservative': {'percent': 1.2, 'description': 'Typical move', 'probability': '~50%'},
                'moderate': {'percent': 1.8, 'description': 'Good move', 'probability': '~25%'},
                'aggressive': {'percent': 2.5, 'description': 'Target move', 'probability': '~10%'}
            }
        }
    
    def format_recommendation_message(self, entry_price, direction, stats):
        """
        Format recommendation message for Telegram
        
        Args:
            entry_price: Entry price in cents
            direction: 'UP' or 'DOWN'
            stats: Move statistics
        
        Returns:
            Formatted message string
        """
        recs = stats['recommendations']
        
        # Calculate actual price levels
        if direction == 'UP':
            conservative_price = entry_price * (1 + recs['conservative']['percent'] / 100)
            moderate_price = entry_price * (1 + recs['moderate']['percent'] / 100)
            aggressive_price = entry_price * (1 + recs['aggressive']['percent'] / 100)
        else:
            conservative_price = entry_price * (1 - recs['conservative']['percent'] / 100)
            moderate_price = entry_price * (1 - recs['moderate']['percent'] / 100)
            aggressive_price = entry_price * (1 - recs['aggressive']['percent'] / 100)
        
        message = f"""
📊 *TYPICAL {direction} MOVES* (Last {self.lookback_days} days):

💡 *EXIT RECOMMENDATIONS:*
• Conservative: {conservative_price:.2f}¢ ({recs['conservative']['percent']:+.1f}%) - {recs['conservative']['probability']}
• Moderate: {moderate_price:.2f}¢ ({recs['moderate']['percent']:+.1f}%) - {recs['moderate']['probability']}
• Aggressive: {aggressive_price:.2f}¢ ({recs['aggressive']['percent']:+.1f}%) - {recs['aggressive']['probability']}

_Based on {stats['sample_size']} {direction} days_
"""
        return message.strip()

# Quick test
if __name__ == "__main__":
    import yfinance as yf
    
    # Test with real data
    ticker = yf.Ticker("ZW=F")
    df = ticker.history(period="60d")
    
    analyzer = MoveAnalyzer()
    
    # Test UP direction
    stats_up = analyzer.analyze_typical_moves(df, 'UP')
    print("UP Move Stats:", stats_up)
    
    msg = analyzer.format_recommendation_message(550, 'UP', stats_up)
    print("\nFormatted Message:")
    print(msg)
