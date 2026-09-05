"""
BACKFILL VERIFIED HISTORICAL EVENTS
======================================
Added 2026-09-04. Small, deliberately conservative backfill of
news_signal_scored.json — NOT the original 10 events Gemini recalled
(5 of those had wrong or backwards price claims when checked against
real data). Only the 2 events independently confirmed via real web
search (multiple credible, independent sources, not just Gemini's
say-so) are included here:

1. 2026-07-11 — Operation MoLoChKa / Kerch Strait-Don-Azov closure.
   Confirmed via Wikipedia, Moscow Times, Reuters-sourced reporting
   across multiple outlets. Real Ukrainian drone campaign against
   Russian shipping starting 2026-07-06, Russia halting the Kerch
   Strait and Don-Azov Canal ~2026-07-10/11, disrupting ~25% of
   Russia's wheat export logistics. Lines up with the real +13.01%
   5-day price move independently found in find_significant_price_
   moves.py ending 2026-07-15.

2. 2024-12-11 — India cuts wheat stock limit (traders/wholesalers
   2,000 MT -> 1,000 MT). Confirmed via India's own Ministry of
   Consumer Affairs, Food and Public Distribution press releases.
   Gemini had misdated this by 5 days (claimed Dec 16); real
   government announcement date used here instead.

Reuses score_news_signals.py's exact score_directional() logic so
these entries are computed identically to how live-scanned entries
will be — same abnormal-return-vs-corn adjustment, same thresholds.
Entries are tagged 'backfilled_verified': true so they're always
distinguishable from organic scans, and this script will never
overwrite an entry that's already there (checked by key).

Deliberately small — 2 entries, not a large synthetic dataset. Real
verification takes real time; padding this with unverified events
would defeat the entire point of the verification step.

Run once. Safe to re-run — idempotent, skips entries already present.

Usage:
  python3 backfill_historical_events.py
"""

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util

IL = ZoneInfo("Asia/Jerusalem")
SCORED_FILE = Path("news_signal_scored.json")

# Import score_news_signals.py's functions directly rather than
# duplicating the scoring logic — these entries must be computed
# exactly the same way live ones are.
spec = importlib.util.spec_from_file_location("sns", "score_news_signals.py")
sns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sns)

VERIFIED_EVENTS = [
    {
        'timestamp': '2026-07-11T00:00:00+03:00',
        'signal': 'BULLISH',
        'category': 'confirmed_physical_disruption',
        'key_phrase': 'Operation MoLoChKa — Russia halts Kerch Strait/Don-Azov Canal '
                       'shipping after Ukrainian drone strikes, ~25% of Russian wheat '
                       'export logistics disrupted (verified: Wikipedia, Moscow Times, '
                       'Reuters via multiple outlets)',
    },
    {
        'timestamp': '2024-12-11T00:00:00+03:00',
        'signal': 'BEARISH',  # domestic supply-management tightening, not an export restriction
        'category': 'other_fundamental',
        'key_phrase': 'India cuts wheat stock limit for traders/wholesalers from 2,000 MT '
                       'to 1,000 MT to curb hoarding (verified: India Ministry of Consumer '
                       'Affairs, Food and Public Distribution press release)',
    },
]


def load_scored():
    if not SCORED_FILE.exists():
        return {}
    return json.loads(SCORED_FILE.read_text())


def save_scored(scored):
    SCORED_FILE.write_text(json.dumps(scored, indent=2))


def main():
    scored = load_scored()
    needed_dates = [datetime.fromisoformat(e['timestamp']).astimezone(IL) for e in VERIFIED_EVENTS]
    earliest_needed = min(needed_dates)

    print(f"Fetching price history back to {earliest_needed.date()}...")
    wheat_df, corn_df = sns.fetch_price_history(earliest_needed)
    print(f"Fetched {len(wheat_df)} wheat bars, {len(corn_df)} corn bars.\n")

    added = 0
    for ev in VERIFIED_EVENTS:
        ts_key = ev['timestamp']
        if ts_key in scored:
            print(f"  {ts_key} already present, skipping (idempotent).")
            continue

        entry_time = datetime.fromisoformat(ts_key).replace(tzinfo=None)
        entry_date = entry_time.date()
        at_or_before = wheat_df[wheat_df.index.date <= entry_date]
        if at_or_before.empty:
            print(f"  {ts_key}: no wheat price data available before this date — skipping.")
            continue
        entry_price = float(at_or_before.iloc[-1]['Close'])

        result = sns.score_directional(entry_price, entry_date, ev['signal'], wheat_df, corn_df)
        if result is None:
            print(f"  {ts_key}: not enough trading days elapsed yet to score — skipping for now.")
            continue

        scored[ts_key] = {
            'signal': ev['signal'], 'category': ev['category'],
            'key_phrase': ev['key_phrase'], 'result': result,
            'backfilled_verified': True,
        }
        added += 1
        print(f"  ADDED {ts_key} | {ev['category']} | {ev['signal']} -> {result['outcome']}")

    save_scored(scored)
    print(f"\nBackfilled {added} verified historical event(s).")
    print("These are tagged 'backfilled_verified': true — always distinguishable")
    print("from organically-scanned live entries.")


if __name__ == "__main__":
    main()
