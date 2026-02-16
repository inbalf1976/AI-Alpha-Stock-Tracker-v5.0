"""
Performance Tracker - Black Box for Trading System
Logs predictions and validates accuracy 24 hours later
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

class PerformanceTracker:
    """Track and validate system predictions"""
    
    def __init__(self):
        self.log_file = Path("prediction_log.json")
        self.stats_file = Path("performance_stats.json")
    
    def log_prediction(self, direction, price, confidence, factors):
        """
        Log a new prediction
        
        Args:
            direction: UP or DOWN
            price: Entry price
            confidence: Prediction confidence (0-1)
            factors: Dict with all factor signals
        """
        prediction = {
            'timestamp': datetime.now().isoformat(),
            'direction': direction,
            'entry_price': price,
            'confidence': confidence,
            'factors': {
                'seasonal': factors.get('seasonal', 'UNKNOWN'),
                'weather': factors.get('weather', 'UNKNOWN'),
                'wasde': factors.get('wasde', 'UNKNOWN'),
                'volume': factors.get('volume', 'UNKNOWN'),
                'ensemble': factors.get('ensemble', 'UNKNOWN')
            },
            'validated': False,
            'result': None,
            'actual_move': None
        }
        
        # Load existing predictions
        predictions = self._load_predictions()
        predictions.append(prediction)
        
        # Save
        with open(self.log_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"\n📝 Prediction logged: {direction} at {price:.2f}¢ ({confidence:.1%})")
    
    def validate_predictions(self):
        """
        Check predictions that are 24+ hours old and validate them
        
        Returns:
            Number of predictions validated
        """
        predictions = self._load_predictions()
        validated_count = 0
        
        for pred in predictions:
            # Skip already validated
            if pred.get('validated'):
                continue
            
            # Check if 24 hours have passed
            pred_time = datetime.fromisoformat(pred['timestamp'])
            hours_passed = (datetime.now() - pred_time).total_seconds() / 3600
            
            if hours_passed >= 24:
                result = self._validate_single_prediction(pred)
                if result:
                    pred['validated'] = True
                    pred['result'] = result['correct']
                    pred['actual_move'] = result['actual_move']
                    pred['exit_price'] = result['exit_price']
                    validated_count += 1
        
        # Save updated predictions
        if validated_count > 0:
            with open(self.log_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            # Update stats
            self._update_stats()
        
        return validated_count
    
    def _validate_single_prediction(self, pred):
        """Validate a single prediction by checking actual price movement"""
        try:
            # Get price 24 hours after prediction
            pred_time = datetime.fromisoformat(pred['timestamp'])
            check_time = pred_time + timedelta(hours=24)
            
            # Fetch data
            ticker = yf.Ticker("ZW=F")
            df = ticker.history(
                start=pred_time - timedelta(days=1),
                end=check_time + timedelta(days=1)
            )
            
            if len(df) < 2:
                return None
            
            entry_price = pred['entry_price']
            
            # Find closest price to 24h mark
            exit_price = df['Close'].iloc[-1]
            
            # Calculate actual move
            actual_move_pct = ((exit_price - entry_price) / entry_price) * 100
            
            # Determine if prediction was correct
            predicted_direction = pred['direction']
            
            if predicted_direction == 'UP':
                correct = actual_move_pct > 0
            else:  # DOWN
                correct = actual_move_pct < 0
            
            return {
                'correct': correct,
                'actual_move': actual_move_pct,
                'exit_price': exit_price
            }
        
        except Exception as e:
            print(f"Validation error: {e}")
            return None
    
    def _update_stats(self):
        """Calculate and save performance statistics"""
        predictions = self._load_predictions()
        
        # Filter validated predictions
        validated = [p for p in predictions if p.get('validated')]
        
        if not validated:
            return
        
        # Calculate stats
        total = len(validated)
        correct = sum(1 for p in validated if p.get('result'))
        win_rate = (correct / total * 100) if total > 0 else 0
        
        # Calculate by confidence levels
        high_conf = [p for p in validated if p.get('confidence', 0) >= 0.70]
        high_conf_wins = sum(1 for p in high_conf if p.get('result'))
        high_conf_rate = (high_conf_wins / len(high_conf) * 100) if high_conf else 0
        
        # Calculate by direction
        up_signals = [p for p in validated if p.get('direction') == 'UP']
        down_signals = [p for p in validated if p.get('direction') == 'DOWN']
        
        up_wins = sum(1 for p in up_signals if p.get('result'))
        down_wins = sum(1 for p in down_signals if p.get('result'))
        
        up_rate = (up_wins / len(up_signals) * 100) if up_signals else 0
        down_rate = (down_wins / len(down_signals) * 100) if down_signals else 0
        
        # Recent performance (last 10)
        recent = validated[-10:] if len(validated) >= 10 else validated
        recent_wins = sum(1 for p in recent if p.get('result'))
        recent_rate = (recent_wins / len(recent) * 100) if recent else 0
        
        stats = {
            'last_updated': datetime.now().isoformat(),
            'total_predictions': total,
            'correct_predictions': correct,
            'overall_win_rate': win_rate,
            'high_confidence_rate': high_conf_rate,
            'high_confidence_count': len(high_conf),
            'up_signal_rate': up_rate,
            'up_signal_count': len(up_signals),
            'down_signal_rate': down_rate,
            'down_signal_count': len(down_signals),
            'recent_10_rate': recent_rate,
            'recent_10_count': len(recent)
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📊 Stats updated: {correct}/{total} correct ({win_rate:.1f}%)")
    
    def get_stats(self):
        """Get current performance statistics"""
        if not self.stats_file.exists():
            return None
        
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def generate_weekly_summary(self):
        """
        Generate weekly performance summary
        
        Returns:
            Formatted summary string for Telegram
        """
        stats = self.get_stats()
        
        if not stats:
            return "📊 No performance data yet. Waiting for predictions..."
        
        total = stats['total_predictions']
        correct = stats['correct_predictions']
        win_rate = stats['overall_win_rate']
        
        # Get last week's predictions
        predictions = self._load_predictions()
        week_ago = datetime.now() - timedelta(days=7)
        
        recent_week = [
            p for p in predictions 
            if p.get('validated') and 
            datetime.fromisoformat(p['timestamp']) > week_ago
        ]
        
        if recent_week:
            week_correct = sum(1 for p in recent_week if p.get('result'))
            week_rate = (week_correct / len(recent_week) * 100)
        else:
            week_correct = 0
            week_rate = 0
        
        summary = f"""
📊 *WEEKLY PERFORMANCE REPORT* 📊

*This Week:*
• Signals: {len(recent_week)}
• Correct: {week_correct}
• Win Rate: {week_rate:.1f}%

*All Time:*
• Total Signals: {total}
• Correct: {correct}
• Win Rate: {win_rate:.1f}%

*Breakdown:*
• UP Signals: {stats['up_signal_rate']:.1f}% ({stats['up_signal_count']} total)
• DOWN Signals: {stats['down_signal_rate']:.1f}% ({stats['down_signal_count']} total)
• High Confidence (70%+): {stats['high_confidence_rate']:.1f}% ({stats['high_confidence_count']} total)

*Recent Trend (Last 10):*
• Win Rate: {stats['recent_10_rate']:.1f}%

_System is {'performing as expected' if win_rate >= 70 else 'below target' if win_rate < 65 else 'on track'}!_
"""
        return summary
    
    def _load_predictions(self):
        """Load predictions from file"""
        if not self.log_file.exists():
            return []
        
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except:
            return []

# Test
if __name__ == "__main__":
    tracker = PerformanceTracker()
    
    # Example: Log a prediction
    # tracker.log_prediction(
    #     direction='UP',
    #     price=550.0,
    #     confidence=0.75,
    #     factors={'seasonal': 'BULLISH', 'weather': 'BULLISH', 'wasde': 'NEUTRAL'}
    # )
    
    # Validate old predictions
    validated = tracker.validate_predictions()
    print(f"Validated {validated} predictions")
    
    # Get stats
    stats = tracker.get_stats()
    if stats:
        print(f"\nCurrent win rate: {stats['overall_win_rate']:.1f}%")
