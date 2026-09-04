"""
SCORE NEWS SIGNALS
=====================
Companion to score_predictions.py, for the LLM news signal (from
news_scanner.py). Checks every news_log.json entry old enough to have
a real outcome, and marks it WIN/LOSS the same way score_predictions.py
already does for ConvictionGate's conditions.

REWRITTEN 2026-09-04, real fix: this script originally read
news_signal_log.json and expected a stored entry_price + numeric
confidence on each entry. news_scanner.py was rewritten on 2026-07-28
to a broader macro/commodity scanner writing news_log.json instead
(see wheat_monitor_pro.py's get_news_signal() docstring) — nothing has
written news_signal_log.json since, so this script had been silently
scoring nothing for over a month despite running find (it just always
printed "no entries" and exited quietly, no error). This version reads
news_log.json directly and reconstructs entry_price by fetching real
price history at each entry's timestamp (news_log.json never stored
one), using the exact same wheat_impact normalization logic as
get_news_signal() (string or dict shape, see below) so this scores
against precisely what the live signal actually is, not an assumption.

Deliberately does NOT write scoring fields back onto news_log.json
entries — that file is the shared raw scan log other things read
(bug_detector.py, immediate_risk_tracker.py indirectly). Scored
results go in their own file, news_signal_scored.json, same
separation-of-concerns pattern immediate_risk_log.json already uses
relative to news_log.json.

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

NEWS_LOG_FILE = Path("news_log.json")
SCORED_FILE = Path("news_signal_scored.json")


def normalize_signal(entry):
    """
    Exact same normalization as wheat_monitor_pro.py's get_news_signal()
    — wheat_impact can be a plain string ("BULLISH"/"BEARISH"/"NEUTRAL")
    or a dict ({"direction": ..., "reason": ...}) depending on how the
    model formatted its JSON that run. Returns 'BULLISH', 'BEARISH', or
    None (NEUTRAL/unrecognized — nothing directional to score).
    """
    analysis = entry.get('llm_analysis')
    if not analysis:
        return None
    wheat_impact = analysis.get('wheat_impact')
    if isinstance(wheat_impact, dict):
        signal = wheat_impact.get('direction', 'NEUTRAL')
    else:
        signal = wheat_impact or 'NEUTRAL'
    signal = str(signal).upper()
    return signal if signal in ('BULLISH', 'BEARISH') else None


def load_scored():
    if not SCORED_FILE.exists():
        return {}
    return json.loads(SCORED_FILE.read_text())


def save_scored(scored):
    SCORED_FILE.write_text(json.dumps(scored, indent=2))


def fetch_price_history(earliest_needed=None):
    """
    UPDATED 2026-09-04, real bug found on the first real run: this used
    a fixed 10-day window, but news_log.json's real entries go back to
    2026-07-26 — every entry older than 10 days was silently
    unscoreable (empty price lookup, treated as "not enough data yet"
    even though it never would be, since the window always slides
    forward with "now"). Now takes the earliest timestamp actually
    needing to be scored and reaches back far enough to cover it, with
    a small buffer. Falls back to the old 10-day window if nothing is
    passed in (e.g. if called standalone).
    """
    end = datetime.now(IL)
    if earliest_needed is not None:
        start = earliest_needed - timedelta(days=1)  # 1-day buffer before the earliest entry
    else:
        start = end - timedelta(days=10)
    df = yf.Ticker(TICKER).history(start=start, end=end, interval='1h', auto_adjust=False)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def score_one_signal(entry_time, signal, price_df):
    age_hours = (datetime.now(IL).replace(tzinfo=None) - entry_time).total_seconds() / 3600
    if age_hours < MIN_AGE_HOURS:
        return None, None, None

    # entry_price reconstructed from real price history — news_log.json
    # never stored one (see module docstring)
    at_or_before = price_df[price_df.index <= entry_time]
    if at_or_before.empty:
        return None, None, None
    entry_price = float(at_or_before.iloc[-1]['Close'])

    future_bars = price_df[price_df.index > entry_time]
    window_end = entry_time + timedelta(hours=MAX_LOOKFORWARD_HOURS)
    future_bars = future_bars[future_bars.index <= window_end]
    if future_bars.empty:
        return None, None, None

    final_price = float(future_bars['Close'].iloc[-1])
    pct_move = (final_price - entry_price) / entry_price

    predicted_up = signal == 'BULLISH'
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
    if not NEWS_LOG_FILE.exists():
        print(f"No {NEWS_LOG_FILE} found. Nothing to score.")
        return

    log = json.loads(NEWS_LOG_FILE.read_text())
    scored = load_scored()
    print(f"Loaded {len(log)} logged news scans.")

    # Find the earliest not-yet-scored directional entry so the price
    # fetch reaches back far enough to cover it — see
    # fetch_price_history()'s docstring for why this matters.
    candidates = [
        e['timestamp'] for e in log
        if 'timestamp' in e and 'llm_analysis' in e
        and e['timestamp'] not in scored
        and normalize_signal(e) is not None
    ]
    earliest_needed = None
    if candidates:
        earliest_needed = datetime.fromisoformat(min(candidates)).astimezone(IL)

    price_df = fetch_price_history(earliest_needed)
    print(f"Fetched {len(price_df)} hourly bars for scoring.\n")

    newly_scored = 0
    for entry in log:
        # UPDATED 2026-09-04, real bug found on first run: 5 entries
        # from 2026-07-23 to 07-26 use an even older schema
        # (scan_time/headlines, no llm_analysis at all) from before
        # the 2026-07-28 news_scanner.py rewrite — predates the schema
        # this whole file already assumes. They can't be scored (no
        # signal data exists in that format), so skip cleanly instead
        # of crashing on the missing key.
        if 'timestamp' not in entry or 'llm_analysis' not in entry:
            continue

        ts_key = entry['timestamp']
        if ts_key in scored:
            continue

        signal = normalize_signal(entry)
        if signal is None:
            continue  # NEUTRAL or unrecognized — nothing directional to score

        entry_time = datetime.fromisoformat(ts_key).replace(tzinfo=None)
        outcome, pct_move, final_price = score_one_signal(entry_time, signal, price_df)
        if outcome is None:
            continue  # too soon, or no price data available yet

        key_phrase = (entry.get('llm_analysis', {}) or {}).get('key_risk') \
            or (entry.get('llm_analysis', {}) or {}).get('summary', '')[:80]

        scored[ts_key] = {
            'signal': signal, 'outcome': outcome,
            'pct_move': pct_move, 'final_price': final_price,
            'key_phrase': key_phrase,
        }
        newly_scored += 1
        print(f"  {ts_key[:16]} | {signal:<8} -> {outcome} ({pct_move:+.2f}% move) | \"{key_phrase}\"")

    save_scored(scored)
    print(f"\nNewly scored this run: {newly_scored}")

    directional = [v for v in scored.values() if v['outcome'] in ('WIN', 'LOSS')]
    if not directional:
        print("\nNo directional (non-NEUTRAL, non-FLAT) signals scored yet.")
        return

    wins = sum(1 for v in directional if v['outcome'] == 'WIN')
    print("\n" + "=" * 60)
    print("LIVE NEWS SIGNAL ACCURACY")
    print("=" * 60)
    print(f"Directional signals scored: {len(directional)}")
    print(f"Win rate: {wins}/{len(directional)} = {wins/len(directional):.1%}")
    print("\nCompare this to a coin flip (50%). Do NOT increase the news")
    print("signal's nudge weight in weekly_range_engine.py until this has")
    print("15-20+ scored signals AND stays meaningfully above 50-55%.")


if __name__ == "__main__":
    main()
