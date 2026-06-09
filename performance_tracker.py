"""
Performance Tracker - FIXED VERSION
====================================
KEY FIXES vs original:
  1. Validates against actual stop/target hits using intraday High/Low
  2. Feeds win rate back to wheat_monitor via get_confidence_gate()
  3. Circuit breaker: suppresses alerts after 3 consecutive losses
  4. Tracks real P&L (cents/bushel), not just directional accuracy
  5. Separates "directional accuracy" from "trading accuracy"
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

# Must match wheat_monitor.py exactly
STOP_LOSS_PCT   = 0.015   # 1.5%
TAKE_PROFIT_PCT = 0.025   # 2.5%
TICKER          = "ZW=F"

# Circuit breaker settings
MAX_CONSECUTIVE_LOSSES = 3
PAUSE_HOURS_AFTER_BREACH = 48
MIN_PREDICTIONS_FOR_GATE = 10   # need at least this many before suppressing
CONFIDENCE_GATE_THRESHOLD = 0.52  # below this win rate → suppress alerts


class PerformanceTracker:

    def __init__(self):
        self.log_file   = Path("prediction_log.json")
        self.stats_file = Path("performance_stats.json")

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def log_prediction(self, direction, price, confidence, factors):
        """Log a new prediction. Called from wheat_monitor.py."""
        prediction = {
            'timestamp':   datetime.now().isoformat(),
            'direction':   direction,
            'entry_price': price,
            'confidence':  confidence,
            'factors':     {
                'seasonal': factors.get('seasonal', 'UNKNOWN'),
                'weather':  factors.get('weather',  'UNKNOWN'),
                'wasde':    factors.get('wasde',    'UNKNOWN'),
                'volume':   factors.get('volume',   'UNKNOWN'),
                'ensemble': factors.get('ensemble', 'UNKNOWN'),
            },
            # validation fields filled in later
            'validated':          False,
            'outcome':            None,   # 'WIN' | 'LOSS' | 'OPEN'
            'exit_reason':        None,   # 'TARGET_HIT' | 'STOP_HIT' | 'EXPIRED_UP' | 'EXPIRED_DOWN'
            'directional_correct': None,  # True/False (old metric, kept for reference)
            'actual_move_pct':    None,
            'exit_price':         None,
            'pnl_cents':          None,   # actual profit/loss in cents per bushel
        }

        predictions = self._load_predictions()
        predictions.append(prediction)
        self._save_predictions(predictions)

        print(f"\n📝 Prediction logged: {direction} at {price:.2f}¢ ({confidence:.1%})")

    def validate_predictions(self):
        """
        Validate all pending predictions that are 24+ hours old.
        Uses intraday High/Low to check if stop or target was hit.
        Returns number of newly validated predictions.
        """
        predictions = self._load_predictions()
        validated_count = 0

        for pred in predictions:
            if pred.get('validated'):
                continue

            pred_time    = datetime.fromisoformat(pred['timestamp'])
            hours_passed = (datetime.now() - pred_time).total_seconds() / 3600

            if hours_passed < 24:
                continue

            result = self._validate_with_intraday(pred)
            if result is None:
                continue

            pred['validated']           = True
            pred['outcome']             = result['outcome']
            pred['exit_reason']         = result['exit_reason']
            pred['directional_correct'] = result['directional_correct']
            pred['actual_move_pct']     = result['actual_move_pct']
            pred['exit_price']          = result['exit_price']
            pred['pnl_cents']           = result['pnl_cents']
            validated_count += 1

        if validated_count > 0:
            self._save_predictions(predictions)
            self._update_stats(predictions)
            print(f"✅ Validated {validated_count} new prediction(s)")

        return validated_count

    def get_confidence_gate(self):
        """
        Called by wheat_monitor.py BEFORE sending an alert.
        Returns (allowed: bool, reason: str)

        Blocks alerts when:
          - Circuit breaker tripped (3 consecutive losses)
          - Recent win rate too low (last 10 predictions < 52%)
        """
        stats = self.get_stats()

        if stats is None:
            return True, "No history yet — allowing"

        # --- Circuit breaker ---
        cb = stats.get('circuit_breaker', {})
        if cb.get('active'):
            tripped_at = datetime.fromisoformat(cb['tripped_at'])
            hours_paused = (datetime.now() - tripped_at).total_seconds() / 3600
            if hours_paused < PAUSE_HOURS_AFTER_BREACH:
                remaining = PAUSE_HOURS_AFTER_BREACH - hours_paused
                return False, f"🛑 Circuit breaker active — {remaining:.0f}h remaining ({MAX_CONSECUTIVE_LOSSES} consecutive losses)"
            else:
                # Auto-reset after pause period
                print("⚡ Circuit breaker auto-reset after pause period")

        # --- Win rate gate ---
        recent_count = stats.get('recent_10_count', 0)
        recent_rate  = stats.get('recent_10_win_rate', 100) / 100

        if recent_count >= MIN_PREDICTIONS_FOR_GATE and recent_rate < CONFIDENCE_GATE_THRESHOLD:
            return False, f"📉 Win rate too low ({recent_rate:.0%} on last {recent_count}) — suppressing alerts"

        return True, f"✅ Gate open (recent win rate: {recent_rate:.0%})"

    def get_stats(self):
        """Return current stats dict, or None if no data yet."""
        if not self.stats_file.exists():
            return None
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def generate_summary(self):
        """Format a Telegram-ready performance summary."""
        stats = self.get_stats()

        if not stats:
            return "📊 No performance data yet — waiting for validated predictions."

        total    = stats['total_predictions']
        wins     = stats['wins']
        win_rate = stats['trading_win_rate']

        cb_line = ""
        cb = stats.get('circuit_breaker', {})
        if cb.get('active'):
            cb_line = f"\n🛑 *CIRCUIT BREAKER ACTIVE* ({MAX_CONSECUTIVE_LOSSES} consecutive losses)\n"

        return f"""
📊 *PERFORMANCE REPORT* 📊
{cb_line}
*Trading Accuracy (real):*
• Total validated: {total}
• Wins: {wins} | Losses: {total - wins}
• Win rate: {win_rate:.1f}%

*Exit Breakdown:*
• Target hit (2.5%): {stats['target_hits']}
• Stop hit (1.5%):   {stats['stop_hits']}
• Expired UP:        {stats['expired_up']}
• Expired DOWN:      {stats['expired_down']}

*P&L:*
• Avg win:  +{stats['avg_win_cents']:.1f}¢/bu
• Avg loss: {stats['avg_loss_cents']:.1f}¢/bu
• Total P&L: {stats['total_pnl_cents']:+.1f}¢/bu

*Directional accuracy (for reference):*
• {stats['directional_accuracy']:.1f}% (up from 50% random)

*Recent trend (last 10):*
• Win rate: {stats['recent_10_win_rate']:.1f}%
• Consecutive losses: {stats['consecutive_losses']}

_Target: 58%+ to be profitable at 1.67:1 R:R_
"""

    # ------------------------------------------------------------------ #
    #  CORE VALIDATION LOGIC                                               #
    # ------------------------------------------------------------------ #

    def _validate_with_intraday(self, pred):
        """
        The key fix: use intraday High/Low to check if stop or target
        was actually hit within 24 hours of the prediction.

        Returns a result dict or None if data unavailable.
        """
        try:
            pred_time  = datetime.fromisoformat(pred['timestamp'])
            end_time   = pred_time + timedelta(hours=28)   # small buffer
            entry      = pred['entry_price']
            direction  = pred['direction']

            stop_price   = entry * (1 - STOP_LOSS_PCT)   if direction == 'UP' else entry * (1 + STOP_LOSS_PCT)
            target_price = entry * (1 + TAKE_PROFIT_PCT) if direction == 'UP' else entry * (1 - TAKE_PROFIT_PCT)

            ticker = yf.Ticker(TICKER)
            df = ticker.history(
                start=pred_time - timedelta(days=1),
                end=end_time + timedelta(days=1),
                interval='1h'   # hourly bars → proper intraday resolution
            )

            if df is None or df.empty or len(df) < 2:
                print(f"   ⚠️  Not enough intraday data to validate {pred['timestamp'][:10]}")
                return None

            # Only look at candles AFTER the prediction was made
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            future_bars = df[df.index >= pred_time]

            if future_bars.empty:
                return None

            # Walk bar by bar — first level hit wins
            outcome      = None
            exit_reason  = None
            exit_price   = None

            for _, bar in future_bars.iterrows():
                bar_high = bar['High']
                bar_low  = bar['Low']

                if direction == 'UP':
                    # Stop hit if Low touches stop_price
                    if bar_low <= stop_price:
                        outcome     = 'LOSS'
                        exit_reason = 'STOP_HIT'
                        exit_price  = stop_price
                        break
                    # Target hit if High reaches target_price
                    if bar_high >= target_price:
                        outcome     = 'WIN'
                        exit_reason = 'TARGET_HIT'
                        exit_price  = target_price
                        break
                else:  # DOWN
                    if bar_high >= stop_price:
                        outcome     = 'LOSS'
                        exit_reason = 'STOP_HIT'
                        exit_price  = stop_price
                        break
                    if bar_low <= target_price:
                        outcome     = 'WIN'
                        exit_reason = 'TARGET_HIT'
                        exit_price  = target_price
                        break

            # Neither stop nor target hit in 24h → expired
            if outcome is None:
                final_close = future_bars['Close'].iloc[-1]
                exit_price  = final_close
                actual_move = (final_close - entry) / entry * 100

                if direction == 'UP':
                    outcome     = 'WIN'  if actual_move > 0 else 'LOSS'
                    exit_reason = 'EXPIRED_UP' if actual_move > 0 else 'EXPIRED_DOWN'
                else:
                    outcome     = 'WIN'  if actual_move < 0 else 'LOSS'
                    exit_reason = 'EXPIRED_DOWN' if actual_move < 0 else 'EXPIRED_UP'

            # Directional accuracy (old metric, kept for reference)
            final_close_ref    = future_bars['Close'].iloc[-1]
            actual_move_pct    = (final_close_ref - entry) / entry * 100
            directional_correct = (actual_move_pct > 0) if direction == 'UP' else (actual_move_pct < 0)

            # Real P&L in cents per bushel
            if exit_reason == 'TARGET_HIT':
                pnl = entry * TAKE_PROFIT_PCT if direction == 'UP' else entry * TAKE_PROFIT_PCT
            elif exit_reason == 'STOP_HIT':
                pnl = -(entry * STOP_LOSS_PCT)
            else:
                pnl = abs(exit_price - entry) * (1 if outcome == 'WIN' else -1)

            print(f"   📋 Validated {pred['timestamp'][:10]}: "
                  f"{direction} → {outcome} ({exit_reason}) | "
                  f"P&L: {pnl:+.2f}¢")

            return {
                'outcome':             outcome,
                'exit_reason':         exit_reason,
                'directional_correct': directional_correct,
                'actual_move_pct':     round(actual_move_pct, 3),
                'exit_price':          round(float(exit_price), 4),
                'pnl_cents':           round(float(pnl), 4),
            }

        except Exception as e:
            print(f"   ❌ Validation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------ #
    #  STATS                                                               #
    # ------------------------------------------------------------------ #

    def _update_stats(self, predictions=None):
        if predictions is None:
            predictions = self._load_predictions()

        validated = [p for p in predictions if p.get('validated')]
        if not validated:
            return

        total  = len(validated)
        wins   = sum(1 for p in validated if p.get('outcome') == 'WIN')
        losses = total - wins

        win_rate = (wins / total * 100) if total > 0 else 0

        target_hits  = sum(1 for p in validated if p.get('exit_reason') == 'TARGET_HIT')
        stop_hits    = sum(1 for p in validated if p.get('exit_reason') == 'STOP_HIT')
        expired_up   = sum(1 for p in validated if p.get('exit_reason') == 'EXPIRED_UP')
        expired_down = sum(1 for p in validated if p.get('exit_reason') == 'EXPIRED_DOWN')

        win_pnls  = [p['pnl_cents'] for p in validated if p.get('outcome') == 'WIN'  and p.get('pnl_cents') is not None]
        loss_pnls = [p['pnl_cents'] for p in validated if p.get('outcome') == 'LOSS' and p.get('pnl_cents') is not None]
        all_pnls  = [p['pnl_cents'] for p in validated if p.get('pnl_cents') is not None]

        avg_win  =  sum(win_pnls)  / len(win_pnls)  if win_pnls  else 0
        avg_loss =  sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
        total_pnl = sum(all_pnls)

        dir_correct = sum(1 for p in validated if p.get('directional_correct'))
        dir_accuracy = (dir_correct / total * 100) if total > 0 else 0

        # Recent 10
        recent     = validated[-10:]
        recent_wins = sum(1 for p in recent if p.get('outcome') == 'WIN')
        recent_rate = (recent_wins / len(recent) * 100) if recent else 0

        # Consecutive losses (most recent streak)
        consecutive_losses = 0
        for p in reversed(validated):
            if p.get('outcome') == 'LOSS':
                consecutive_losses += 1
            else:
                break

        # Circuit breaker
        cb_state = self.get_stats().get('circuit_breaker', {}) if self.get_stats() else {}
        if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            if not cb_state.get('active'):
                cb_state = {
                    'active':     True,
                    'tripped_at': datetime.now().isoformat(),
                    'reason':     f"{consecutive_losses} consecutive losses",
                }
                print(f"\n🛑 CIRCUIT BREAKER TRIPPED: {consecutive_losses} consecutive losses — pausing {PAUSE_HOURS_AFTER_BREACH}h")
        else:
            if cb_state.get('active'):
                tripped_at = datetime.fromisoformat(cb_state['tripped_at'])
                if (datetime.now() - tripped_at).total_seconds() / 3600 >= PAUSE_HOURS_AFTER_BREACH:
                    cb_state = {'active': False}

        stats = {
            'last_updated':         datetime.now().isoformat(),
            'total_predictions':    total,
            'wins':                 wins,
            'losses':               losses,
            'trading_win_rate':     round(win_rate, 2),
            'target_hits':          target_hits,
            'stop_hits':            stop_hits,
            'expired_up':           expired_up,
            'expired_down':         expired_down,
            'avg_win_cents':        round(avg_win, 2),
            'avg_loss_cents':       round(avg_loss, 2),
            'total_pnl_cents':      round(total_pnl, 2),
            'directional_accuracy': round(dir_accuracy, 2),
            'recent_10_win_rate':   round(recent_rate, 2),
            'recent_10_count':      len(recent),
            'consecutive_losses':   consecutive_losses,
            'circuit_breaker':      cb_state,
        }

        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n📊 Stats updated: {wins}/{total} wins ({win_rate:.1f}%) | "
              f"P&L: {total_pnl:+.1f}¢ | "
              f"Consecutive losses: {consecutive_losses}")

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _load_predictions(self):
        if not self.log_file.exists():
            return []
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def _save_predictions(self, predictions):
        with open(self.log_file, 'w') as f:
            json.dump(predictions, f, indent=2)


# ------------------------------------------------------------------ #
#  STANDALONE TEST                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    tracker = PerformanceTracker()

    print("🔄 Validating pending predictions...")
    validated = tracker.validate_predictions()
    print(f"Validated {validated} predictions")

    gate_ok, gate_reason = tracker.get_confidence_gate()
    print(f"\n🚦 Confidence gate: {'OPEN' if gate_ok else 'CLOSED'} — {gate_reason}")

    stats = tracker.get_stats()
    if stats:
        print(f"\n📊 Trading win rate: {stats['trading_win_rate']:.1f}%")
        print(f"   Directional accuracy: {stats['directional_accuracy']:.1f}%")
        print(f"   Total P&L: {stats['total_pnl_cents']:+.1f}¢/bu")
        print(f"   Circuit breaker: {'ACTIVE' if stats['circuit_breaker'].get('active') else 'off'}")
    else:
        print("\n📊 No validated predictions yet")

    print("\n" + tracker.generate_summary())
