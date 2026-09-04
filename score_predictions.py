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
    Walk forward from the prediction's entry date and determine the
    real outcome.

    UPDATED 2026-09-03, real fix: previously ALWAYS computed its own
    synthetic stop/target (STOP_PCT=1.5%/TARGET_PCT=2.5% from
    entry_price) — a definition that had nothing to do with the actual
    weekly setup shown in the Telegram alert (frozen entry/stop/target
    from weekly_range_cache.json, regenerated on break — see
    wheat_monitor_pro.py's _check_breach/log_weekly_break). Now uses
    the REAL stop_price/target_price stored on the entry (added
    2026-09-03 in log_prediction()) when present. Entries logged before
    that change don't have these fields (None) and fall back to the
    old synthetic definition exactly as before — this is deliberately
    NOT retroactive, so historical scored entries don't shift under
    anyone; only predictions logged from now on get scored against
    what was actually communicated.

    Returns (outcome, exit_reason, pnl_cents, scoring_method) or
    (None, None, None, None) if not enough time has passed yet to
    score it. scoring_method is 'real_setup' or 'legacy_synthetic' —
    stored on the entry so it's always visible which definition was
    used, rather than silently blending two different metrics.
    """
    entry_time = datetime.fromisoformat(entry['timestamp'])
    entry_date = entry_time.date()
    entry_price = entry['entry_price']
    direction = entry['direction']

    age_days = (datetime.now(IL).date() - entry_date).days
    if age_days < MIN_AGE_DAYS:
        return None, None, None, None  # too soon to score

    # Get daily bars strictly after the entry date
    future_bars = price_df[price_df.index.date > entry_date]
    if future_bars.empty:
        return None, None, None, None

    # Limit to the lookforward window
    window_end = entry_date + timedelta(days=MAX_LOOKFORWARD_DAYS)
    future_bars = future_bars[future_bars.index.date <= window_end]
    if future_bars.empty:
        return None, None, None, None

    real_stop = entry.get('stop_price')
    real_target = entry.get('target_price')
    using_real_setup = real_stop is not None and real_target is not None

    if using_real_setup:
        stop_price = real_stop
        target_price = real_target
        scoring_method = 'real_setup'
    else:
        if direction == 'UP':
            stop_price   = entry_price * (1 - STOP_PCT)
            target_price = entry_price * (1 + TARGET_PCT)
        else:  # DOWN
            stop_price   = entry_price * (1 + STOP_PCT)
            target_price = entry_price * (1 - TARGET_PCT)
        scoring_method = 'legacy_synthetic'

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
            pnl = -(entry_price - stop_price) if direction == 'UP' else -(stop_price - entry_price)
            return 'LOSS', 'same_bar_ambiguous_conservative_loss', round(pnl, 2), scoring_method
        if hit_target:
            pnl = (target_price - entry_price) if direction == 'UP' else (entry_price - target_price)
            return 'WIN', 'target_hit', round(pnl, 2), scoring_method
        if hit_stop:
            pnl = -(entry_price - stop_price) if direction == 'UP' else -(stop_price - entry_price)
            return 'LOSS', 'stop_hit', round(pnl, 2), scoring_method

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
        return ('WIN' if won else 'LOSS'), 'window_expired_no_stop_target_hit', round(pnl, 2), scoring_method

    return None, None, None, None  # still open, not enough time elapsed yet


def score_daily_direction(log, price_df):
    """
    Separate metric from score_one_prediction() above — added 2026-09-03
    after a real conversation surfaced that "win/loss accuracy" was being
    asked about without anyone being sure which of several different
    things was actually meant. There are (at least) two genuinely
    different questions this codebase can answer, and they must never be
    blended into one number:

      - score_one_prediction() / 'real_setup': did the actual weekly
        setup (entry/stop/target from weekly_range_cache.json, the same
        numbers shown in the Telegram "TRADE SETUP" line) hit its real
        target or stop? This is what the summary below calls SETUP
        win/loss.

      - score_daily_direction() (this function): was the "Daily
        direction (today)" UP/DOWN call — a fresh live ensemble read
        every run, independent of the frozen setup, see
        wheat_monitor_pro.py's `direction` variable and its
        "Daily direction (today)" print line — actually right, checked
        against the next real trading day's Close? This is DIRECTION
        accuracy, a completely different, much simpler question with
        no stop-loss involved at all.

    These can and do diverge substantially — confirmed in the real
    2026-08-13 to 2026-08-21 window, where SETUP win rate and DIRECTION
    accuracy told two different stories for the same days.

    Dedup: prediction_log.json can carry multiple log entries for the
    same real underlying call (see the still-open price-triggered
    duplicate-logging issue) — collapses to one scoring attempt per
    (calendar day, direction, ~entry_price) so that bug doesn't also
    contaminate this metric. This is a stopgap, not a fix for the
    underlying duplicate-logging bug itself.

    Known caveat, not fixed here: price_df comes from plain TICKER=ZW=F
    (see module docstring), not the front-month-correct contract
    wheat_monitor_pro.py actually uses — so results near a contract
    roll may carry a little price noise. Flagged, not silently fixed,
    since fixing it is a separate, already-tracked open item.

    Mutates log in place with 'direction_scored' (bool),
    'direction_correct' (bool or None), 'direction_check_date' (str) —
    entirely separate fields from validated/outcome/scoring_method
    above, so this never interferes with SETUP scoring.
    """
    seen_keys = set()
    newly_scored = 0

    for entry in log:
        if entry.get('direction_scored'):
            continue  # already scored in a previous run

        day = entry['timestamp'][:10]
        key = (day, entry['direction'], round(entry['entry_price']))
        if key in seen_keys:
            entry['direction_scored'] = True
            entry['direction_correct'] = None  # duplicate of a same-day call, not independently scored
            entry['direction_check_date'] = None
            continue
        seen_keys.add(key)

        entry_date = datetime.fromisoformat(entry['timestamp']).date()
        future = price_df[price_df.index.date > entry_date]
        if future.empty:
            continue  # too soon — next trading day hasn't happened yet, try again next run

        next_bar = future.iloc[0]
        next_close = float(next_bar['Close'])
        next_date = future.index[0].date().isoformat()

        correct = (next_close > entry['entry_price']) if entry['direction'] == 'UP' \
            else (next_close < entry['entry_price'])

        entry['direction_scored'] = True
        entry['direction_correct'] = correct
        entry['direction_check_date'] = next_date
        newly_scored += 1

    return newly_scored


def print_direction_accuracy_summary(log):
    scored = [e for e in log if e.get('direction_scored') and e.get('direction_correct') is not None]
    print(f"\n--- DAILY DIRECTION accuracy — separate metric, see score_daily_direction() docstring (n={len(scored)}) ---")
    if not scored:
        print("(No entries scored yet.)")
        return
    correct = sum(1 for e in scored if e['direction_correct'])
    print(f"Overall: {correct}/{len(scored)} = {correct/len(scored):.1%}")
    tiers = sorted(set(e.get('tier', 'N/A') for e in scored), key=lambda x: (x == 'N/A', x))
    for t in tiers:
        tier_entries = [e for e in scored if e.get('tier', 'N/A') == t]
        tier_correct = sum(1 for e in tier_entries if e['direction_correct'])
        print(f"  Tier {t}: {tier_correct}/{len(tier_entries)} = {tier_correct/len(tier_entries):.1%}  (n={len(tier_entries)})")
    print("(This is DIRECTION only — no stop/target involved. Compare against")
    print(" 'Scored against the REAL weekly setup' above only if you mean to")
    print(" ask two different questions; they are not the same metric.)")


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

        outcome, exit_reason, pnl, scoring_method = score_one_prediction(entry, price_df)
        if outcome is None:
            continue  # not enough time has passed yet

        entry['validated']   = True
        entry['outcome']     = outcome
        entry['exit_reason'] = exit_reason
        entry['pnl_cents']   = pnl
        entry['scoring_method'] = scoring_method  # 'real_setup' or 'legacy_synthetic' — see score_one_prediction() docstring
        newly_scored += 1

        tier_display = entry.get('tier', 'N/A')
        direction_display = entry.get('direction', '?')
        entry_price_display = entry.get('entry_price', 0)
        timestamp_display = entry.get('timestamp', 'unknown')[:10] if entry.get('timestamp') else 'unknown'

        print(f"  {timestamp_display} | {direction_display:<4} @ {entry_price_display:.2f}c "
              f"| Tier {tier_display} | -> {outcome} ({exit_reason}, {pnl:+.2f}c)")

    save_log(log)
    print(f"\nNewly scored this run: {newly_scored}")

    newly_direction_scored = score_daily_direction(log, price_df)
    save_log(log)
    print(f"Newly direction-scored this run: {newly_direction_scored}")

    # ── Summary stats, overall and by tier ──
    scored = [e for e in log if e.get('validated')]
    if not scored:
        print("\nNo predictions scored yet — check back after a few more days.")
        return

    print("\n" + "=" * 60)
    print("LIVE PREDICTION ACCURACY (real forward tracking)")
    print("=" * 60)

    # UPDATED 2026-09-03: legacy entries (scored before this fix, or
    # entries whose stop_price/target_price weren't available) used a
    # synthetic 1.5%/2.5% trade unrelated to the real weekly setup —
    # kept separate from 'real_setup' entries below so the two
    # definitions never silently blend into one misleading number.
    real_setup_scored = [e for e in scored if e.get('scoring_method') == 'real_setup']
    legacy_scored = [e for e in scored if e.get('scoring_method') != 'real_setup']

    if real_setup_scored:
        print(f"\n--- Scored against the REAL weekly setup (n={len(real_setup_scored)}) ---")
        wins = sum(1 for e in real_setup_scored if e['outcome'] == 'WIN')
        print(f"Overall: {wins}/{len(real_setup_scored)} = {wins/len(real_setup_scored):.1%}")
        tiers = sorted(set(e.get('tier', 'N/A') for e in real_setup_scored), key=lambda x: (x == 'N/A', x))
        for t in tiers:
            tier_entries = [e for e in real_setup_scored if e.get('tier', 'N/A') == t]
            tier_wins = sum(1 for e in tier_entries if e['outcome'] == 'WIN')
            print(f"  Tier {t}: {tier_wins}/{len(tier_entries)} = {tier_wins/len(tier_entries):.1%}  (n={len(tier_entries)})")
    else:
        print("\n--- No entries scored against the real weekly setup yet ---")
        print("(New predictions logged from 2026-09-03 onward carry stop_price/")
        print(" target_price; give it a few days for the first ones to age past")
        print(f" MIN_AGE_DAYS={MIN_AGE_DAYS} and become scoreable.)")

    print(f"\n--- Legacy synthetic 1.5%/2.5% definition, for reference (n={len(legacy_scored)}) ---")
    if legacy_scored:
        wins = sum(1 for e in legacy_scored if e['outcome'] == 'WIN')
        print(f"Overall: {wins}/{len(legacy_scored)} = {wins/len(legacy_scored):.1%}")
        tiers = sorted(set(e.get('tier', 'N/A') for e in legacy_scored), key=lambda x: (x == 'N/A', x))
        for t in tiers:
            tier_entries = [e for e in legacy_scored if e.get('tier', 'N/A') == t]
            tier_wins = sum(1 for e in tier_entries if e['outcome'] == 'WIN')
            print(f"  Tier {t}: {tier_wins}/{len(tier_entries)} = {tier_wins/len(tier_entries):.1%}  (n={len(tier_entries)})")
    print("(This is the OLD metric — matches backtest.py's holdout definition,")
    print(" but does not correspond to the real entry/stop/target shown in the")
    print(" Telegram alert. Kept only so nothing from before 2026-09-03 is lost.)")

    print_direction_accuracy_summary(log)

    print("\nNote: ConvictionGate's HOLDOUT_ACCURACY claims (84.8%/84.0%/70.0%/68.0%)")
    print("were computed by backtest.py using the legacy synthetic 1.5%/2.5%")
    print("definition — so they're only directly comparable to the 'legacy")
    print("synthetic' numbers above, not the 'real weekly setup' numbers.")
    print("Once enough real_setup predictions accumulate (15-20+ per tier),")
    print("that number is the one that actually matters for real trading —")
    print("but it isn't backtest-validated against anything yet.")


if __name__ == "__main__":
    main()
