"""
SCORE NEWS SIGNALS
=====================
Companion to score_predictions.py, for the new (2026-07-19) LLM news
signal specifically. Checks every logged news_signal_log.json entry
old enough to have a real outcome, and marks it WIN/LOSS the same way
score_predictions.py already does for ConvictionGate's conditions.

This is the actual reward/punishment mechanism for the news signal:
run this for a few weeks, then check the real win rate below against
what get_news_signal() currently assumes in wheat_monitor_pro.py.
If it's not meaningfully better than a coin flip after 15-20+ scored
signals, its nudge weight should be reduced further, not increased —
same discipline used to validate (and in vol_low's case, invalidate)
every other signal in this system.

Usage:
  python3 score_news_signals.py
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf

IL = ZoneInfo("Asia/Jerusalem")
TICKER = "ZW=F"
MOVE_THRESHOLD_PCT = 0.01  # 1% move within the window counts as a real WIN/LOSS
MIN_AGE_HOURS = 24
MAX_LOOKFORWARD_HOURS = 72

LOG_FILE = Path("news_signal_log.json")


def load_log():
    if not LOG_FILE.exists():
        print(f"No {LOG_FILE} found. Nothing to score.")
        return []
    return json.loads(LOG_FILE.read_text())


def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2))


def fetch_price_history():
    end = datetime.now(IL)
    start = end - timedelta(days=10)
    df = yf.Ticker(TICKER).history(start=start, end=end, interval='1h', auto_adjust=False)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def score_one_signal(entry, price_df):
    if entry['signal'] == 'NEUTRAL' or entry.get('entry_price') is None:
        return None, None, None  # nothing directional to score

    entry_time = datetime.fromisoformat(entry['timestamp']).replace(tzinfo=None)
    age_hours = (datetime.now(IL).replace(tzinfo=None) - entry_time).total_seconds() / 3600
    if age_hours < MIN_AGE_HOURS:
        return None, None, None

    future_bars = price_df[price_df.index > entry_time]
    window_end = entry_time + timedelta(hours=MAX_LOOKFORWARD_HOURS)
    future_bars = future_bars[future_bars.index <= window_end]
    if future_bars.empty:
        return None, None, None

    entry_price = entry['entry_price']
    final_price = float(future_bars['Close'].iloc[-1])
    pct_move = (final_price - entry_price) / entry_price

    predicted_up = entry['signal'] == 'BULLISH'
    actual_up = pct_move > MOVE_THRESHOLD_PCT
    actual_down = pct_move < -MOVE_THRESHOLD_PCT

    if not actual_up and not actual_down:
        outcome = 'FLAT'
    elif (predicted_up and actual_up) or (not predicted_up and actual_down):
        outcome = 'WIN'
    else:
        outcome = 'LOSS'

    return outcome, round(pct_move * 100, 2), round(final_price, 2)


def main():
    log = load_log()
    if not log:
        return

    print(f"Loaded {len(log)} logged news signals.")
    price_df = fetch_price_history()
    print(f"Fetched {len(price_df)} hourly bars for scoring.\n")

    newly_scored = 0
    for entry in log:
        if entry.get('validated'):
            continue

        outcome, pct_move, final_price = score_one_signal(entry, price_df)
        if outcome is None:
            continue

        entry['validated'] = True
        entry['outcome'] = outcome
        entry['pnl_cents'] = pct_move
        newly_scored += 1

        print(f"  {entry['timestamp'][:16]} | {entry['signal']:<8} ({entry['confidence']}%) "
              f"-> {outcome} ({pct_move:+.2f}% move) | \"{entry['key_phrase']}\"")

    save_log(log)
    print(f"\nNewly scored this run: {newly_scored}")

    scored = [e for e in log if e.get('validated') and e['outcome'] in ('WIN', 'LOSS')]
    if not scored:
        print("\nNo directional (non-NEUTRAL, non-FLAT) signals scored yet.")
        return

    wins = sum(1 for e in scored if e['outcome'] == 'WIN')
    print("\n" + "=" * 60)
    print("LIVE NEWS SIGNAL ACCURACY")
    print("=" * 60)
    print(f"Directional signals scored: {len(scored)}")
    print(f"Win rate: {wins}/{len(scored)} = {wins/len(scored):.1%}")
    print("\nCompare this to a coin flip (50%). Do NOT increase the news")
    print("signal's nudge weight in weekly_range_engine.py until this has")
    print("15-20+ scored signals AND stays meaningfully above 50-55%.")


if __name__ == "__main__":
    main()
