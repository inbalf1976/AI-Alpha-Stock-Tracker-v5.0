"""
SCORE PREDICTIONS - CLOSE THE VALIDATION LOOP
================================================
wheat_monitor_pro.py has been logging predictions to prediction_log.json
for months via log_prediction(), but nothing ever went back and checked
whether those predictions were actually right. Every entry sits with
validated=False, outcome=None forever.

This script is the missing piece: for every prediction old enough to
have a real outcome, it fetches what ZW=F actually did afterward, and
marks the prediction WIN or LOSS using the same stop/target logic as
your live trading rules (STOP_PCT=1.5%, TARGET_PCT=2.5% from
wheat_monitor_pro.py).

This is the ONLY real test of whether the rebuilt ConvictionGate
(holdout-validated on 2 years of history) actually holds up on live,
forward predictions it makes from today onward — a backtest can prove
something wasn't fake on past data, but only live tracking proves it
still works going forward.

Usage:
  python3 score_predictions.py

Run this periodically (e.g. weekly) as more predictions age past the
minimum lookforward window and become scoreable.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf

IL = ZoneInfo("Asia/Jerusalem")
TICKER = "ZW=F"
STOP_PCT = 0.015
TARGET_PCT = 0.025
MIN_AGE_DAYS = 2          # need at least this many days of price history after entry
MAX_LOOKFORWARD_DAYS = 10  # if neither stop nor target hit within this window, score by direction at this point

LOG_FILE = Path("prediction_log.json")


def load_log():
    if not LOG_FILE.exists():
        print(f"No {LOG_FILE} found. Nothing to score.")
        return []
    return json.loads(LOG_FILE.read_text())


def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2))


def fetch_price_history():
    """Fetch enough daily history to cover the oldest unvalidated prediction through today."""
    end = datetime.now(IL)
    start = end - timedelta(days=400)  # generous buffer
    df = yf.Ticker(TICKER).history(start=start, end=end, interval='1d', auto_adjust=False)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def score_one_prediction(entry, price_df):
    """
    Walk forward from the prediction's entry date using the SAME
    stop/target logic as the live system, and determine the real
    outcome. Returns (outcome, exit_reason, pnl_cents) or
    (None, None, None) if not enough time has passed yet to score it.
    """
    entry_time = datetime.fromisoformat(entry['timestamp'])
    entry_date = entry_time.date()
    entry_price = entry['entry_price']
    direction = entry['direction']

    age_days = (datetime.now(IL).date() - entry_date).days
    if age_days < MIN_AGE_DAYS:
        return None, None, None  # too soon to score

    # Get daily bars strictly after the entry date
    future_bars = price_df[price_df.index.date > entry_date]
    if future_bars.empty:
        return None, None, None

    # Limit to the lookforward window
    window_end = entry_date + timedelta(days=MAX_LOOKFORWARD_DAYS)
    future_bars = future_bars[future_bars.index.date <= window_end]
    if future_bars.empty:
        return None, None, None

    if direction == 'UP':
        stop_price   = entry_price * (1 - STOP_PCT)
        target_price = entry_price * (1 + TARGET_PCT)
    else:  # DOWN
        stop_price   = entry_price * (1 + STOP_PCT)
        target_price = entry_price * (1 - TARGET_PCT)

    # Walk bars in order, check which was hit first using daily High/Low
    for _, bar in future_bars.iterrows():
        if direction == 'UP':
            hit_target = bar['High'] >= target_price
            hit_stop   = bar['Low']  <= stop_price
        else:
            hit_target = bar['Low']  <= target_price
            hit_stop   = bar['High'] >= stop_price

        if hit_target and hit_stop:
            # Ambiguous same-bar hit — conservative: count as loss
            pnl = -(entry_price * STOP_PCT)
            return 'LOSS', 'same_bar_ambiguous_conservative_loss', round(pnl, 2)
        if hit_target:
            pnl = entry_price * TARGET_PCT
            return 'WIN', 'target_hit', round(pnl, 2)
        if hit_stop:
            pnl = -(entry_price * STOP_PCT)
            return 'LOSS', 'stop_hit', round(pnl, 2)

    # Neither hit within the window — did we run out of available bars,
    # or run out of the time window? Either way, if window has fully
    # elapsed, score by final direction; otherwise leave unscored.
    if age_days >= MAX_LOOKFORWARD_DAYS or len(future_bars) >= MAX_LOOKFORWARD_DAYS:
        final_price = float(future_bars['Close'].iloc[-1])
        if direction == 'UP':
            won = final_price > entry_price
        else:
            won = final_price < entry_price
        pnl = (final_price - entry_price) if direction == 'UP' else (entry_price - final_price)
        return ('WIN' if won else 'LOSS'), 'window_expired_no_stop_target_hit', round(pnl, 2)

    return None, None, None  # still open, not enough time elapsed yet


def main():
    log = load_log()
    if not log:
        return

    print(f"Loaded {len(log)} logged predictions.")
    price_df = fetch_price_history()
    print(f"Fetched {len(price_df)} daily bars for scoring.\n")

    newly_scored = 0
    for entry in log:
        if entry.get('validated'):
            continue  # already scored, skip

        outcome, exit_reason, pnl = score_one_prediction(entry, price_df)
        if outcome is None:
            continue  # not enough time has passed yet

        entry['validated']   = True
        entry['outcome']     = outcome
        entry['exit_reason'] = exit_reason
        entry['pnl_cents']   = pnl
        newly_scored += 1

        print(f"  {entry['timestamp'][:10]} | {entry['direction']:<4} @ {entry['entry_price']:.2f}c "
              f"| Tier {entry['tier']} | -> {outcome} ({exit_reason}, {pnl:+.2f}c)")

    save_log(log)
    print(f"\nNewly scored this run: {newly_scored}")

    # ── Summary stats, overall and by tier ──
    scored = [e for e in log if e.get('validated')]
    if not scored:
        print("\nNo predictions scored yet — check back after a few more days.")
        return

    print("\n" + "=" * 60)
    print("LIVE PREDICTION ACCURACY (real forward tracking)")
    print("=" * 60)

    wins = sum(1 for e in scored if e['outcome'] == 'WIN')
    print(f"Overall: {wins}/{len(scored)} = {wins/len(scored):.1%}")

    tiers = sorted(set(e['tier'] for e in scored))
    for t in tiers:
        tier_entries = [e for e in scored if e['tier'] == t]
        tier_wins = sum(1 for e in tier_entries if e['outcome'] == 'WIN')
        print(f"  Tier {t}: {tier_wins}/{len(tier_entries)} = {tier_wins/len(tier_entries):.1%}  (n={len(tier_entries)})")

    print("\nCompare Tier 1/2 numbers above to ConvictionGate's claimed")
    print("HOLDOUT_ACCURACY (84.8% / 84.0% / 70.0% / 68.0%). If live results")
    print("run meaningfully below those claims once you have 15-20+ scored")
    print("predictions per tier, the gate needs re-tuning — small samples")
    print("(<10) are not yet reliable enough to conclude anything.")


if __name__ == "__main__":
    main()
