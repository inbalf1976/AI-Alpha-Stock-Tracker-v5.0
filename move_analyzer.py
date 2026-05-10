"""
Move Analyzer - Analyzes typical price moves for exit recommendations
"""

import pandas as pd
import numpy as np


class MoveAnalyzer:
    """Analyze historical moves to set realistic profit targets"""
    
    def __init__(self):
        self.lookback = 30
    
    def analyze_typical_moves(self, df, direction):
        """
        Analyze typical moves in the given direction over last 30 days
        
        Returns:
        - percentiles: 50th, 75th, 90th percentile moves
        - count: number of moves in that direction
        - avg_move: average move size
        """
        
        try:
            # Calculate daily returns
            df_recent = df.tail(self.lookback).copy()
            df_recent['Daily_Return'] = df_recent['Close'].pct_change()
            
            # Filter by direction
            if direction == 'UP':
                moves = df_recent[df_recent['Daily_Return'] > 0]['Daily_Return']
            else:  # DOWN
                moves = df_recent[df_recent['Daily_Return'] < 0]['Daily_Return'].abs()
            
            if len(moves) < 3:
                return self._get_default_stats(direction)
            
            # Calculate percentiles
            p50 = moves.quantile(0.50)  # Median (conservative)
            p75 = moves.quantile(0.75)  # Moderate
            p90 = moves.quantile(0.90)  # Aggressive
            
            return {
                'p50': p50,
                'p75': p75,
                'p90': p90,
                'count': len(moves),
                'avg': moves.mean(),
                'max': moves.max()
            }
            
        except Exception as e:
            print(f"Move analysis error: {e}")
            return self._get_default_stats(direction)
    
    def _get_default_stats(self, direction):
        """Default statistics when not enough data"""
        return {
            'p50': 0.012,  # 1.2%
            'p75': 0.018,  # 1.8%
            'p90': 0.025,  # 2.5%
            'count': 0,
            'avg': 0.015,
            'max': 0.030
        }
    
    def format_recommendation_message(self, entry_price, direction, stats):
        """
        Format exit recommendations for Telegram message
        
        Returns formatted string with conservative/moderate/aggressive targets
        """
        
        if direction == 'UP':
            conservative_target = entry_price * (1 + stats['p50'])
            moderate_target = entry_price * (1 + stats['p75'])
            aggressive_target = entry_price * (1 + stats['p90'])
            symbol = "+"
        else:  # DOWN
            conservative_target = entry_price * (1 - stats['p50'])
            moderate_target = entry_price * (1 - stats['p75'])
            aggressive_target = entry_price * (1 - stats['p90'])
            symbol = "+"
        
        message = f"""📊 *TYPICAL {direction} MOVES* (Last {self.lookback} days):

*EXIT RECOMMENDATIONS:*
• Conservative: {conservative_target:.2f}¢ ({symbol}{stats['p50']*100:.1f}%) - ~50%
• Moderate: {moderate_target:.2f}¢ ({symbol}{stats['p75']*100:.1f}%) - ~25%
• Aggressive: {aggressive_target:.2f}¢ ({symbol}{stats['p90']*100:.1f}%) - ~10%

Based on {stats['count']} {direction} days"""
        
        return message
