"""
LOSS FORENSICS — Wheat Monitor supervisor tool
=================================================
Read-only. Never fixes anything, never touches live trading files.
Its only job: for every stop-out LOSS in weekly_break_log.json, find
the nearest news_log.json scan entries before and after the break,
and report them side by side so a human can judge whether the loss
was news-driven, and whether the scanner detected it but underweighted
it, or genuinely had nothing to go on.

This does NOT classify causes, does NOT compute a "hit rate" verdict,
and does NOT touch NEWS_SIGNAL_NUDGE_SCALE or any other live trading
parameter. It only assembles the evidence. Judging what the evidence
means — real signal vs coincidence vs pure noise — is left to you.

WHY THIS EXISTS (2026-08-08 session):
A single loss (2026-08-07, W32) showed the news scanner correctly
flagging BULLISH drought news 2.5h before a DOWN setup got stopped
out — a real "detected but underweighted" case. But one match is not
a pattern. This script exists to accumulate matched cases
automatically, over time, so a real decision about news-signal
weighting can eventually be made on N cases instead of 1 — without
repeating the mistake of reacting to small-sample noise (the same
mistake that produced the original fake 100%-accuracy ConvictionGate
conditions, later corrected by real holdout backtesting).

TRIGGER MODES:
  --telegram    Always send the report to Telegram right now. Use this
                for a manual/on-demand run (e.g. GitHub Actions
                workflow_dispatch, or running it locally) — it sends
                regardless of day or whether anything changed.
  --scheduled   For an automated daily cron. Only sends to Telegram on
                the periodic report days (Sunday = weekly, 1st of the
                month = monthly); other days it just prints to the
                Actions log, same silent-unless-relevant pattern as
                bug_detector.py. Does NOT send on every run — this
                tool has no "issue found" concept like bug_detector,
                so without --telegram/--scheduled it never pings
                Telegram at all, only prints.
  (no flag)     Print the report only. Never sends to Telegram. Safe
                default for local testing.

See loss_forensics.yml (companion GitHub Actions workflow) for how
--scheduled and workflow_dispatch (--telegram) are wired up.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

IL = ZoneInfo("Asia/Jerusalem")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

WEEKLY_BREAK_LOG_FILE = Path("weekly_break_log.json")
NEWS_LOG_FILE         = Path("news_log.json")

# How far before/after a loss to still consider a news entry "relevant"
# context, purely for display — entries beyond this are shown but
# labeled as far away rather than a plausible trigger/reaction.
RELEVANT_WINDOW_HOURS = 24


def _load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"__load_error__": str(e)}


def _parse_ts(ts_str):
    """Parse an ISO timestamp, defaulting to Israel tz if naive."""
    if not ts_str:
        return None
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IL)
        return ts
    except Exception:
        return None


def _extract_wheat_impact(entry):
    """
    news_log.json's llm_analysis.wheat_impact has appeared in the wild
    as either a plain string ("BULLISH") or a dict
    ({"direction": "BULLISH", "reason": "..."}) depending on how the
    LLM formatted its JSON that run (see news_scanner.py's
    get_news_signal() docstring, same normalization logic mirrored
    here). Also tolerate the rare shape seen in some scans where
    wheat_impact_reason is a sibling key instead of nested.
    """
    analysis = entry.get("llm_analysis")
    if not analysis or not isinstance(analysis, dict):
        return None, None

    wi = analysis.get("wheat_impact")
    if isinstance(wi, dict):
        direction = wi.get("direction")
        reason = wi.get("reason")
    else:
        direction = wi
        reason = analysis.get("wheat_impact_reason")

    if not direction:
        return None, None
    return str(direction).upper(), reason


def find_nearest_news(news_log, target_dt, direction):
    """
    direction: 'before' or 'after'.
    Returns the news_log entry whose timestamp is closest to target_dt
    on the requested side, or None if no such entry exists in the log.
    news_log entries are newest-first (per news_scanner.py's
    update_news_log(), which inserts at index 0).
    """
    best = None
    best_delta = None
    for entry in news_log:
        ts = _parse_ts(entry.get("timestamp"))
        if ts is None:
            continue
        if direction == "before" and ts > target_dt:
            continue
        if direction == "after" and ts < target_dt:
            continue
        delta = abs((ts - target_dt).total_seconds())
        if best_delta is None or delta < best_delta:
            best = entry
            best_delta = delta
    return best, best_delta


def format_news_summary(entry, delta_seconds, target_dt):
    if entry is None:
        return "  (none found in news_log.json)"

    ts = _parse_ts(entry.get("timestamp"))
    hours = delta_seconds / 3600 if delta_seconds is not None else None
    direction, reason = _extract_wheat_impact(entry)

    far_flag = ""
    if hours is not None and hours > RELEVANT_WINDOW_HOURS:
        far_flag = f"  [>{RELEVANT_WINDOW_HOURS}h away — likely unrelated, shown for completeness]"

    lines = []
    ts_str = ts.strftime("%Y-%m-%d %H:%M IL") if ts else entry.get("timestamp", "?")
    hours_str = f"{hours:.1f}h" if hours is not None else "?"
    lines.append(f"  {ts_str}  ({hours_str} away){far_flag}")

    if direction:
        lines.append(f"    wheat_impact: {direction}" + (f" — {reason}" if reason else ""))
    else:
        lines.append("    wheat_impact: (no LLM analysis on this scan)")

    return "\n".join(lines)


def analyze_loss(loss_entry, news_log):
    """Returns a formatted block of text for one LOSS entry."""
    iso_key = loss_entry.get("iso_key", "?")
    broken_at_str = loss_entry.get("broken_at")
    price = loss_entry.get("price_at_break")
    reason = loss_entry.get("reason", "")

    target_dt = _parse_ts(broken_at_str)
    lines = [f"=== {iso_key}  |  broke at {broken_at_str}  |  price {price}  |  {reason} ==="]

    if target_dt is None:
        lines.append("  Could not parse broken_at timestamp — skipping news match.")
        return "\n".join(lines)

    if news_log is None:
        lines.append("  news_log.json missing — no news data to match against.")
        return "\n".join(lines)
    if isinstance(news_log, dict) and "__load_error__" in news_log:
        lines.append(f"  news_log.json failed to parse: {news_log['__load_error__']}")
        return "\n".join(lines)

    before_entry, before_delta = find_nearest_news(news_log, target_dt, "before")
    after_entry, after_delta = find_nearest_news(news_log, target_dt, "after")

    lines.append("BEFORE the break:")
    lines.append(format_news_summary(before_entry, before_delta, target_dt))
    lines.append("AFTER the break:")
    lines.append(format_news_summary(after_entry, after_delta, target_dt))

    return "\n".join(lines)


def build_report():
    break_log = _load_json(WEEKLY_BREAK_LOG_FILE)
    news_log = _load_json(NEWS_LOG_FILE)

    if break_log is None:
        return "weekly_break_log.json is missing — nothing to analyze."
    if isinstance(break_log, dict) and "__load_error__" in break_log:
        return f"weekly_break_log.json failed to parse: {break_log['__load_error__']}"
    if not isinstance(break_log, list) or not break_log:
        return "weekly_break_log.json is empty — no breaks logged yet."

    losses = [e for e in break_log if e.get("reason", "").upper().find("LOSS") != -1]

    if not losses:
        return "No LOSS entries found in weekly_break_log.json — nothing to analyze."

    now_str = datetime.now(IL).strftime("%Y-%m-%d %H:%M")
    header = (
        f"Loss Forensics Report ({now_str})\n"
        f"{len(losses)} loss(es) found in weekly_break_log.json.\n"
        f"Matching each against nearest news_log.json entries before/after.\n"
        f"NOTE: absence of matching news is common and expected, especially\n"
        f"for losses before news_log.json existed or during scanner gaps —\n"
        f"that's real information (unexplained volatility), not a bug.\n"
    )

    blocks = [analyze_loss(loss, news_log) for loss in losses]

    return header + "\n\n" + "\n\n".join(blocks)


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("Telegram not configured — printing report instead.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": message[:4000]},  # Telegram msg length limit
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


def is_weekly_report_day(now):
    return now.weekday() == 6  # Sunday, same convention as bug_detector.py


def is_monthly_report_day(now):
    return now.day == 1


def main():
    report = build_report()
    print(report)

    manual = "--telegram" in sys.argv
    scheduled = "--scheduled" in sys.argv

    if manual:
        send_telegram(report)
        return

    if scheduled:
        now = datetime.now(IL)
        if is_monthly_report_day(now):
            send_telegram("📅 MONTHLY " + report)
        elif is_weekly_report_day(now):
            send_telegram("🗓️ WEEKLY " + report)
        else:
            print("(scheduled run, not a weekly/monthly report day — printed only, no Telegram send)")


if __name__ == "__main__":
    main()
