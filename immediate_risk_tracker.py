"""
immediate_risk_tracker.py — Outcome tracking for news_scanner.py's
"immediate risk" alert (see news_scanner.py CHANGELOG 2026-08-15)
========================================================================

Purpose: turn "how many cycles has the immediate_risk alert fired,
and was it actually right?" from something tracked by hand into a
real, automatic number — the evidence base for the eventual decision
(after ~20-30 scored cycles) on whether to give the news signal real
weight in wheat_monitor_pro.py's calculations.

READ-ONLY WITH RESPECT TO THE MODEL: this file never touches
wheat_monitor_pro.py, ConvictionGate, NEWS_SIGNAL_NUDGE_SCALE, or any
gate/weight/parameter. It only reads/writes immediate_risk_log.json,
a file the model does not consume.

How it works:
  1. log_event() — called by news_scanner.py right after it sends an
     immediate_risk Telegram alert. Records the reason, the predicted
     direction (wheat_impact), and the wheat price AT that moment.
     Entry starts unscored.
  2. score_pending() — called every news_scanner.py run (piggybacks
     on the existing 3x/day schedule, no new cron needed). For any
     logged event whose SCORING_WINDOW_HOURS has elapsed and isn't
     scored yet, fetches the current price, compares the % change
     against the predicted direction, and marks it HIT / MISS / FLAT.
     FLAT = the move was too small to count as directionally
     meaningful either way (see DIRECTION_NOISE_THRESHOLD_PCT) —
     these are excluded from the hit-rate calculation entirely,
     same as how tiny/ambiguous cases are treated everywhere else in
     this project (avoids inflating or deflating the rate on noise).
  3. summary() / main() — prints (and optionally Telegram-sends) the
     running tally: total cycles, scored, pending, hits, misses,
     flats, hit rate. This is the number to check at the ~20-30 cycle
     mark.

NOTE ON PRICE FETCHING: uses a simple yfinance ZW=F fetch, NOT
wheat_monitor_pro.py's exact front-month-contract-selection logic
(e.g. ZWU26.CBT) — that logic lives inside wheat_monitor_pro.py and
this file deliberately avoids importing from it to stay fully
decoupled from the model. ZW=F is the continuous front-month proxy
and is good enough for a directional hit/miss check over a
24h window; if this ever needs the exact same contract-roll
precision the model uses, that's a deliberate follow-up, not
something assumed here.
"""

import os
import re
import json
import logging
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("immediate_risk_tracker")

IL = ZoneInfo("Asia/Jerusalem")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

LOG_FILE = Path("immediate_risk_log.json")
MAX_LOG_ENTRIES = 300

SCORING_WINDOW_HOURS = 24        # how long to wait before scoring an event
DIRECTION_NOISE_THRESHOLD_PCT = 0.3   # moves smaller than this are FLAT, not HIT/MISS


def _load_log():
    if not LOG_FILE.exists():
        return []
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception as e:
        logger.warning(f"Could not read {LOG_FILE}: {e}")
        return []


def _save_log(entries):
    entries = entries[:MAX_LOG_ENTRIES]
    try:
        LOG_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Failed to write {LOG_FILE}: {e}")


def get_price_now():
    """
    Best-effort current wheat price via yfinance ZW=F. Returns None
    (never raises) on failure — callers must treat None as "couldn't
    fetch, try again next run", same fail-soft convention as the rest
    of this project's network calls.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("ZW=F")
        price = ticker.fast_info.get("lastPrice")
        if price:
            return float(price)
        return None
    except Exception as e:
        logger.warning(f"   Price fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# DEDUPLICATION (added 2026-09-03) — real incident: news_scanner.py has no
# concept of "already alerted on this". When an ongoing situation (e.g. a
# multi-day Hormuz blockade) stays the top headline across scans, Gemini can
# re-flag immediate_risk=true every single cycle even though nothing new is
# happening. Confirmed real duplicates: alert #47/#48 and #49/#50. This also
# contaminates the HIT/MISS sample in score_pending() — repeats of the same
# event get scored as independent cycles, skewing the hit rate.
#
# First attempt used word-overlap text similarity to detect "same topic."
# Tested against the real log before shipping and rejected: non-duplicate
# consecutive alerts scored just as high (0.5-0.57 overlap coefficient) as
# the two confirmed real duplicates (0.40, 0.50) — Hormuz/tanker/Iran/
# blockade vocabulary is inherently clustered in this domain regardless of
# whether it's actually the same specific event, so text similarity alone
# can't reliably separate them.
#
# Simplified to pure time-based cooldown instead: ANY immediate_risk flag
# within DEDUP_COOLDOWN_HOURS of the last one is suppressed, regardless of
# content. Justified by immediate_risk's own definition — a "rare, high-bar
# flag" for genuinely new, actively-unfolding events — so two truly
# independent immediate-risk events landing in the same 18h window should
# itself be rare. Missing that rare case costs less than the duplicate-spam
# already observed (50+ alerts logged, the large majority about the same
# ongoing situation).
# ---------------------------------------------------------------------------
DEDUP_COOLDOWN_HOURS = 18


def is_duplicate_risk(cooldown_hours=DEDUP_COOLDOWN_HOURS):
    """
    Returns True if ANY immediate_risk alert was already logged within
    cooldown_hours, regardless of content (see module comment above for
    why content-based similarity was tried and rejected). Called by
    news_scanner.py BEFORE both send_immediate_risk_alert() and
    log_event() — a suppressed duplicate should neither spam Telegram nor
    count as a fresh, independent data point for scoring.
    """
    now = datetime.now(timezone.utc)
    entries = _load_log()

    for e in entries:
        try:
            ts = datetime.fromisoformat(e["alert_timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IL)
            ts = ts.astimezone(timezone.utc)
        except Exception:
            continue

        age_hours = (now - ts).total_seconds() / 3600
        if age_hours > cooldown_hours:
            # entries are newest-first (log_event() inserts at index 0),
            # so nothing after this can be within the window either
            break

        logger.info(
            f"   Duplicate risk suppressed — alert #{e.get('alert_number')} "
            f"was logged {age_hours:.1f}h ago, within the {cooldown_hours}h "
            f"cooldown"
        )
        return True

    return False


def get_next_alert_number():
    """
    Returns the sequential number the NEXT alert should use — total
    entries ever logged + 1. Includes both scheduled and manual
    triggers (there's no separate counter for each), so the number
    reflects the true running total of immediate_risk cycles.
    """
    return len(_load_log()) + 1


def log_event(reason, wheat_impact, alert_timestamp_iso, alert_number=None):
    """
    Records a new immediate_risk alert for later scoring. Called by
    news_scanner.py right after it sends the Telegram alert.
    wheat_impact should be "BULLISH" or "BEARISH" (a "NEUTRAL" or
    unknown value is still logged but can never score as a hit/miss —
    there's no predicted direction to check against).

    alert_number: the sequential alert count shown in the Telegram
    message (see get_next_alert_number()) — stored alongside the
    entry so a specific alert can be looked up later by its number.
    """
    price = get_price_now()
    entries = _load_log()
    entries.insert(0, {
        "alert_number": alert_number,
        "alert_timestamp": alert_timestamp_iso,
        "reason": reason,
        "predicted_direction": wheat_impact,
        "price_at_alert": price,
        "scored": False,
        "outcome": None,
        "price_at_scoring": None,
        "pct_change": None,
        "scored_at": None,
    })
    _save_log(entries)
    if price is not None:
        logger.info(f"   Logged immediate_risk event #{alert_number} for scoring (price at alert: {price})")
    else:
        logger.info(f"   Logged immediate_risk event #{alert_number}, but price fetch failed — will retry scoring later with no baseline (unscoreable).")


def score_pending():
    """
    Scores any logged event whose SCORING_WINDOW_HOURS has elapsed
    and isn't scored yet. Safe to call every run — does nothing if
    there's nothing ready to score. Returns the list of newly-scored
    entries (for optional reporting by the caller).
    """
    entries = _load_log()
    if not entries:
        return []

    now = datetime.now(timezone.utc)
    newly_scored = []
    current_price = None  # fetched lazily, once, only if needed

    for entry in entries:
        if entry.get("scored"):
            continue
        try:
            alert_dt = datetime.fromisoformat(entry["alert_timestamp"])
            if alert_dt.tzinfo is None:
                alert_dt = alert_dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue

        hours_elapsed = (now - alert_dt).total_seconds() / 3600
        if hours_elapsed < SCORING_WINDOW_HOURS:
            continue  # not ready yet

        if entry.get("price_at_alert") is None:
            # Never had a baseline price — can't score, mark as such
            # so it doesn't sit pending forever.
            entry["scored"] = True
            entry["outcome"] = "UNSCOREABLE (no baseline price)"
            entry["scored_at"] = now.isoformat()
            newly_scored.append(entry)
            continue

        if current_price is None:
            current_price = get_price_now()
        if current_price is None:
            continue  # try again next run

        price_before = entry["price_at_alert"]
        pct_change = (current_price - price_before) / price_before * 100

        # A moved price that agrees with the predicted direction is a
        # HIT, disagrees is a MISS; too small to call either way is FLAT.
        predicted = str(entry.get("predicted_direction", "")).upper()
        if abs(pct_change) < DIRECTION_NOISE_THRESHOLD_PCT:
            outcome = "FLAT"
        elif predicted == "BULLISH":
            outcome = "HIT" if pct_change > 0 else "MISS"
        elif predicted == "BEARISH":
            outcome = "HIT" if pct_change < 0 else "MISS"
        else:
            outcome = "UNSCOREABLE (no clear predicted direction)"

        entry["scored"] = True
        entry["outcome"] = outcome
        entry["price_at_scoring"] = current_price
        entry["pct_change"] = round(pct_change, 3)
        entry["scored_at"] = now.isoformat()
        newly_scored.append(entry)

    if newly_scored:
        _save_log(entries)
        logger.info(f"   Scored {len(newly_scored)} immediate_risk event(s) this run.")

    return newly_scored


def summary():
    entries = _load_log()
    scored = [e for e in entries if e.get("scored")]
    pending = [e for e in entries if not e.get("scored")]
    hits = [e for e in scored if e.get("outcome") == "HIT"]
    misses = [e for e in scored if e.get("outcome") == "MISS"]
    flats = [e for e in scored if e.get("outcome") == "FLAT"]
    unscoreable = [e for e in scored if e.get("outcome", "").startswith("UNSCOREABLE")]

    judged = len(hits) + len(misses)  # FLAT and UNSCOREABLE excluded from hit rate
    hit_rate = (len(hits) / judged * 100) if judged else None

    lines = [
        f"Immediate-Risk Alert Tracker — {len(entries)} total cycle(s) logged",
        f"  Scored: {len(scored)}  |  Pending (< {SCORING_WINDOW_HOURS}h old): {len(pending)}",
        f"  HIT: {len(hits)}  |  MISS: {len(misses)}  |  FLAT: {len(flats)}  |  Unscoreable: {len(unscoreable)}",
    ]
    if hit_rate is not None:
        lines.append(f"  Directional hit rate (HIT / (HIT+MISS)): {hit_rate:.1f}%  (n={judged})")
    else:
        lines.append("  Directional hit rate: not enough scored HIT/MISS cycles yet")

    lines.append("")
    lines.append(f"NOTE: {len(entries)}/~20-30 cycles needed before this is a meaningful "
                 f"sample for a real weighting decision. FLAT and unscoreable cycles "
                 f"don't count toward that threshold in a strict sense — judged (HIT+MISS) "
                 f"cycles are what actually matter.")

    return "\n".join(lines)


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("Telegram not configured — printing summary instead.")
        return False
    try:
        data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT, "text": message}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data=data,
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")
        return False


def main():
    import sys
    score_pending()
    report = summary()
    print(report)
    if "--telegram" in sys.argv:
        send_telegram(report)


if __name__ == "__main__":
    main()
