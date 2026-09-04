"""
SCORE RANGES - per-period range/setup accuracy
================================================
Added 2026-09-03. Different question from score_predictions.py, which
scores individual logged predictions one at a time. This script scores
whole PERIODS (an ISO week, a calendar month) with a single outcome
each, matching the exact definition the user gave:

  - If the range/setup held the whole period with no break -> 100%
  - If it broke and got corrected DURING the period, and that
    correction ultimately proved right -> still 100%
  - If it broke and was never redeemed by period's end (or the
    correction was also wrong) -> 0%

This produces the numbers for three of the alert's own sections:
  WEEKLY FINAL CALL-%, EXPECTED RANGE-%, TRADE SETUP-%  (weekly)
  MONTHLY OUTLOOK Range-%                                (monthly)

IMPORTANT, told to the user before building this: in the current
system, a weekly break event flips direction, regenerates the range,
AND resets entry/stop/target all at once (see wheat_monitor_pro.py's
_check_breach/log_weekly_break). So WEEKLY FINAL CALL-%, EXPECTED
RANGE-%, and TRADE SETUP-% will always be the SAME number here — not a
bug, just how tightly coupled those three alert lines currently are.
They're still computed as separate, clearly labeled outputs below so
the reporting matches the alert's own structure and stays correct if
that coupling ever changes.

MONTHLY is a genuinely different, independent metric — monthly_range
has no direction/entry/stop/target at all, just a range, so its
accuracy is its own real number, not derived from anything else here.

Known caveat: relies on weekly_break_log.json / monthly_break_log.json
as the source of "was there a break" — and we already know separately
that break detection can lag a real intraday breach by hours to
several days (see the 2026-09-02/03 weekly-range detection-lag
finding). This script's numbers inherit that same lag: a period marked
"held, 100%" could in rare cases have had a real breach the system
simply hasn't detected yet. Not fixed here — that's the separate,
already-tracked weekly-breach-detection-lag open item.

Usage:
  python3 score_ranges.py
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yfinance as yf

IL = ZoneInfo("Asia/Jerusalem")
TICKER = "ZW=F"  # same known caveat as score_predictions.py — plain
                 # continuous contract, not the front-month-correct
                 # ticker wheat_monitor_pro.py actually uses.


def fetch_price_history():
    end = datetime.now(IL)
    start = end - timedelta(days=400)
    df = yf.Ticker(TICKER).history(start=start, end=end, interval='1d', auto_adjust=False)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def _direction_of(stop, target):
    return 'UP' if target > stop else 'DOWN'


def _walk_outcome(stop, target, start_iso, end_iso, price_df):
    """
    Walk real daily bars from the day AFTER start_iso (exclusive of the
    break/start day itself — that day's full High/Low can include price
    action from BEFORE the break happened, which belongs to the OLD
    setup, not this one; same convention score_predictions.py already
    uses via `> entry_date`, found necessary here too during testing —
    see 2026-09-03 fix note) through end_iso (exclusive, or None for
    open-ended). Returns 'WIN' (target hit first), 'LOSS' (stop hit
    first), or None (neither hit yet within available data).
    """
    direction = _direction_of(stop, target)
    start_date = datetime.fromisoformat(start_iso).date()
    end_date = datetime.fromisoformat(end_iso).date() if end_iso else None

    bars = price_df[price_df.index.date > start_date]
    if end_date:
        bars = bars[bars.index.date < end_date]

    for _, bar in bars.iterrows():
        if direction == 'UP':
            hit_target = bar['High'] >= target
            hit_stop = bar['Low'] <= stop
        else:
            hit_target = bar['Low'] <= target
            hit_stop = bar['High'] >= stop
        if hit_target:
            return 'WIN'
        if hit_stop:
            return 'LOSS'
    return None


def _build_weekly_segments():
    """
    Chronological list of {start, end(None=ongoing), stop, target}
    reconstructed from weekly_break_log.json + the current live
    weekly_range_cache.json — same reconstruction validated by hand
    earlier in this conversation against real CBOT data.
    """
    breaks = json.loads(Path('weekly_break_log.json').read_text())
    breaks.sort(key=lambda b: b['broken_at'])
    current = json.loads(Path('weekly_range_cache.json').read_text())['weekly']

    segments = []
    prev_end = '2000-01-01T00:00:00+00:00'
    for b in breaks:
        segments.append({'start': prev_end, 'end': b['broken_at'],
                          'stop': b['old_stop'], 'target': b['old_target']})
        prev_end = b['broken_at']
    segments.append({'start': prev_end, 'end': None,
                      'stop': current['stop'], 'target': current['target']})
    return segments


def score_weekly_ranges():
    """
    Returns {iso_key: 'WIN'/'LOSS'/None} — one outcome per ISO week the
    system has ever tracked (from weekly_performance_log.json's known
    iso_keys). None means still open / not yet resolvable.

    Method: for each week, find whichever segment was ACTIVE AT THE
    WEEK'S END (the final correction that week, or the original if
    none broke) and score that segment from ITS OWN start (which may
    be before the week began, if it carried over from a prior week)
    through to real resolution — matching "if it was corrected during
    the window and was right/wrong" rather than clipping the
    resolution check to exactly 7 days.
    """
    segments = _build_weekly_segments()
    price_df = fetch_price_history()
    perf = json.loads(Path('weekly_performance_log.json').read_text())
    iso_keys = sorted(set(e['iso_key'] for e in perf))

    results = {}
    for iso_key in iso_keys:
        year, week = iso_key.split('-W')
        year, week = int(year), int(week)
        week_end = datetime.fromisocalendar(year, week, 7).replace(
            tzinfo=IL, hour=23, minute=59) + timedelta(seconds=1)
        week_end_iso = week_end.isoformat()

        # last segment that had already started by this week's end
        active = None
        for seg in segments:
            if seg['start'] <= week_end_iso:
                active = seg
            else:
                break

        if active is None:
            results[iso_key] = None
            continue

        outcome = _walk_outcome(active['stop'], active['target'],
                                 active['start'], active['end'], price_df)
        results[iso_key] = outcome
    return results


def score_monthly_ranges():
    """
    Same idea as score_weekly_ranges() but for calendar months, using
    monthly_break_log.json + monthly_range_cache.json. Monthly has no
    direction/stop/target — just a pure [low, high] range — so "held"
    means real price never left [low, high] for however long that
    range was the active one.

    Returns {month_key: 'WIN'/'LOSS'/None}.
    """
    breaks = json.loads(Path('monthly_break_log.json').read_text())
    breaks.sort(key=lambda b: b['broken_at'])
    current = json.loads(Path('monthly_range_cache.json').read_text())['monthly']

    segments = []
    prev_end = '2000-01-01T00:00:00+00:00'
    for b in breaks:
        lo, hi = b['old_range'].split('-')
        segments.append({'start': prev_end, 'end': b['broken_at'],
                          'low': float(lo), 'high': float(hi)})
        prev_end = b['broken_at']
    segments.append({'start': prev_end, 'end': None,
                      'low': current['monthly_low'], 'high': current['monthly_high']})

    price_df = fetch_price_history()

    # month keys: from every break/segment we've ever seen, plus the current one
    month_keys = sorted(set(b['month_key'] for b in breaks) | {current.get('month_key', '')})
    month_keys = [m for m in month_keys if m]

    results = {}
    for month_key in month_keys:
        year, month = map(int, month_key.split('-'))
        if month == 12:
            month_end = datetime(year + 1, 1, 1, tzinfo=IL)
        else:
            month_end = datetime(year, month + 1, 1, tzinfo=IL)
        month_end_iso = month_end.isoformat()

        active = None
        for seg in segments:
            if seg['start'] <= month_end_iso:
                active = seg
            else:
                break
        if active is None:
            results[month_key] = None
            continue

        start_date = datetime.fromisoformat(active['start']).date()
        end_date = datetime.fromisoformat(active['end']).date() if active['end'] else None
        bars = price_df[price_df.index.date > start_date]  # see _walk_outcome's docstring — same fix applies here
        if end_date:
            bars = bars[bars.index.date < end_date]

        outcome = None
        for _, bar in bars.iterrows():
            if bar['High'] > active['high'] or bar['Low'] < active['low']:
                outcome = 'LOSS'
                break
        else:
            if not bars.empty:
                outcome = 'WIN'
        results[month_key] = outcome
    return results


def _print_summary(label, results):
    print(f"\n--- {label} ---")
    resolved = {k: v for k, v in results.items() if v is not None}
    pending = [k for k, v in results.items() if v is None]
    if not resolved:
        print("(Nothing resolved yet.)")
    else:
        wins = sum(1 for v in resolved.values() if v == 'WIN')
        print(f"Overall: {wins}/{len(resolved)} = {wins/len(resolved):.1%}")
        for k in sorted(resolved.keys()):
            print(f"  {k}: {resolved[k]}")
    if pending:
        print(f"  (still open / not yet resolvable: {', '.join(sorted(pending))})")


def main():
    weekly = score_weekly_ranges()
    monthly = score_monthly_ranges()

    print("=" * 60)
    print("PER-PERIOD RANGE/SETUP ACCURACY")
    print("=" * 60)
    print("\nNOTE: in the current system, WEEKLY FINAL CALL-%, EXPECTED")
    print("RANGE-%, and TRADE SETUP-% (weekly) are driven by the exact")
    print("same break/hold event — the numbers below apply to all three")
    print("alert lines identically. See module docstring for why.")

    _print_summary("WEEKLY FINAL CALL-% / EXPECTED RANGE-% / TRADE SETUP-% (weekly)", weekly)
    _print_summary("MONTHLY OUTLOOK Range-%", monthly)


if __name__ == "__main__":
    main()
