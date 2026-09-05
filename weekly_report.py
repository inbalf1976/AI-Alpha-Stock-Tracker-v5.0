"""
WEEKLY REPORT GENERATOR
==========================
Runs once, Saturday (after the trading week is done), and reads
weekly_performance_log.json — the day-by-day record written by
wheat_monitor_pro.py's log_daily_performance() every day this week —
to produce a REAL, measured account of how the week's frozen forecast
actually held up. Not a memory-based impression — an actual, logged,
day-by-day record.

Also checks weekly_break_log.json to report whether/when the
forecast broke and got regenerated mid-week.

Sends the report via Telegram, same channel as the daily alerts.

Usage:
  python3 weekly_report.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

IL = ZoneInfo("Asia/Jerusalem")

PERFORMANCE_LOG_FILE = Path("weekly_performance_log.json")
BREAK_LOG_FILE = Path("weekly_break_log.json")

# ADDED 2026-08-23: this script previously wrote NOTHING to disk — it only
# sent to Telegram and exited, so there was no way for bug_detector.py to
# verify it actually ran on the correct day/hour (a real gap identified
# alongside the weekend-alert and backtest-day checks, same session).
STATE_FILE = Path("weekly_report_state.json")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")


def record_run():
    """Writes a small timestamp marker after every send attempt (even the
    'no data' early-exit cases) so bug_detector.py has something real to
    check against."""
    try:
        STATE_FILE.write_text(json.dumps({
            "last_run": datetime.now(IL).isoformat()
        }, indent=2))
    except Exception as e:
        print(f"Failed to record run timestamp: {e}")


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("Telegram not configured — printing report only")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": message},
            timeout=10
        )
        success = r.status_code == 200
        print(f"Telegram: {'✓ sent' if success else '✗ failed'} ({r.status_code})")
        return success
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def get_current_iso_key():
    today = datetime.now(IL)
    iso_year, iso_week, _ = today.isocalendar()
    return f"{iso_year}-W{iso_week}"


def main():
    iso_key = get_current_iso_key()

    if not PERFORMANCE_LOG_FILE.exists():
        msg = f"WEEKLY REPORT — {iso_key}\nNo performance log found. Nothing to report."
        print(msg)
        send_telegram(msg)
        record_run()
        return

    all_entries = json.loads(PERFORMANCE_LOG_FILE.read_text())
    week_entries = [e for e in all_entries if e['iso_key'] == iso_key]
    week_entries.sort(key=lambda e: e['date'])

    if not week_entries:
        msg = f"WEEKLY REPORT — {iso_key}\nNo entries logged for this week yet."
        print(msg)
        send_telegram(msg)
        record_run()
        return

    breaks_this_week = []
    if BREAK_LOG_FILE.exists():
        all_breaks = json.loads(BREAK_LOG_FILE.read_text())
        breaks_this_week = [b for b in all_breaks if b['iso_key'] == iso_key]

    day_lines = []
    for e in week_entries:
        day_short = e['day_name'][:3].lower()
        low, high = e['range_low'], e['range_high']
        pos = e['position_in_range_pct']
        pos_str = f"{pos:.0f}%" if pos is not None else "N/A"
        within = "✓ in range" if e['within_range'] else "✗ OUT OF RANGE"
        day_lines.append(
            f"{day_short}: price {e['price']:.0f}c | range {low:.0f}-{high:.0f}c "
            f"| position {pos_str} | {within}"
        )

    days_in_range = sum(1 for e in week_entries if e['within_range'])
    total_days = len(week_entries)
    in_range_rate = (days_in_range / total_days * 100) if total_days > 0 else 0

    break_lines = ""
    if breaks_this_week:
        break_lines = "\n⚠️ FORECAST BROKEN THIS WEEK:\n"
        for b in breaks_this_week:
            break_lines += f"  {b['broken_at'][:16]} — {b['reason']}\n"

    if breaks_this_week:
        # UPDATED 2026-09-05, real bug found and confirmed — the old
        # conclusion here led with "X/Y days (Z%) stayed within
        # whichever range was active that day", which is structurally
        # near-meaningless right after a break: a freshly-regenerated
        # range is calibrated AROUND wherever price already is, so it
        # will show "in range" almost every time regardless of whether
        # the week actually went well. Real case that exposed this:
        # 2026-W36 broke Thursday with a real stop-hit LOSS, then
        # showed "100% (6/6 days) stayed within range" as the headline
        # — the opposite impression of what actually happened. Fixed
        # to lead with each break's real WIN/LOSS outcome (parsed from
        # the break log's own reason text) instead, and to separately
        # report only whether the CURRENT (most recent, possibly still
        # open) range is holding as of the latest data — which is the
        # one part of "days in range" that's still a real, meaningful,
        # forward-looking fact rather than a tautology.
        outcomes = []
        for b in breaks_this_week:
            reason = b.get('reason', '')
            if '(WIN)' in reason:
                outcomes.append('WIN')
            elif '(LOSS)' in reason:
                outcomes.append('LOSS')
            else:
                outcomes.append('UNKNOWN')
        wins = outcomes.count('WIN')
        losses = outcomes.count('LOSS')
        outcome_summary = ' + '.join(
            f"{n} {label}" for label, n in (('WIN', wins), ('LOSS', losses)) if n > 0
        ) or 'outcome unclear from break log'

        last_entry = week_entries[-1]
        current_range_status = (
            f"the current range (as of {last_entry['day_name']}) is still holding "
            f"({last_entry['position_in_range_pct']:.0f}% through it)"
            if last_entry['within_range'] else
            f"the current range is ALSO broken as of {last_entry['day_name']}"
        )

        conclusion = (
            f"Forecast broke {len(breaks_this_week)}x this week — real market move(s) "
            f"exceeded the frozen plan ({outcome_summary}). {current_range_status}. "
            f"(The old 'X/Y days stayed in range' stat is not shown here after a break — "
            f"it's structurally misleading, since a freshly-regenerated range is fitted "
            f"around wherever price already is.)"
        )
    elif in_range_rate == 100:
        conclusion = f"Frozen range held for all {total_days} days — clean week, no breaks."
    elif in_range_rate >= 70:
        conclusion = (
            f"Frozen range mostly held: {days_in_range}/{total_days} days "
            f"({in_range_rate:.0f}%) within range."
        )
    else:
        conclusion = (
            f"Frozen range struggled: only {days_in_range}/{total_days} days "
            f"({in_range_rate:.0f}%) within range, though no formal break threshold was crossed."
        )

    message = (
        f"WEEKLY REPORT — {iso_key}\n"
        f"{'=' * 30}\n\n"
        + "\n".join(day_lines)
        + f"\n{break_lines}\n"
        f"CONCLUSION:\n{conclusion}\n\n"
        f"(Real logged data — cross-check anytime against weekly_performance_log.json)"
    )

    print(message)
    send_telegram(message)
    record_run()


if __name__ == "__main__":
    main()
