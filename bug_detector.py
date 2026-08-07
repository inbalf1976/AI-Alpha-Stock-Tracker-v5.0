"""
BUG DETECTOR — Wheat Monitor supervisor tool
=============================================
Read-only. Never fixes anything, never touches live files, never
sends trading alerts. Its only job: notice when something is wrong
and tell you (via Telegram) so you and Claude can decide what to do.

Run this on its own schedule (e.g. daily or weekly via GitHub
Actions) — separate workflow from wheat_monitor_pro.py.

CHECKS INCLUDED (the four "core" ones + drift):
  1. Missed-run detection      — did the monitor actually run recently?
  2. Prediction log sanity     — schema drift, bad values, stuck entries
  3. Live vs backtest accuracy — divergence between real and claimed win rate
  4. State file health         — unbounded growth, stale/unused fields
  5. Signal-source freshness   — news/weather/validated_conditions going stale

Add new checks as functions returning a list of issue strings.
Nothing here calls send_telegram except main() — easy to test checks
in isolation by just calling them and printing the list.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

IL = ZoneInfo("Asia/Jerusalem")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE       = Path("wheat_monitor_state.json")
PREDICTION_LOG   = Path("prediction_log.json")
VALIDATED_COND   = Path("validated_conditions.json")
NEWS_LOG         = Path("news_log.json")
WEATHER_CACHE    = Path("weather_cache.json")

# ── thresholds (tune these over time, keep them here in one place) ──────────
MAX_HOURS_SINCE_LAST_CHECK   = 30     # flag if monitor hasn't run in this long
ALERTS_TODAY_CLEANUP_LIMIT   = 40     # flag if old date-keys are piling up
ACCURACY_DIVERGENCE_PCT      = 15     # flag if live vs backtest win rate gap exceeds this
MIN_SAMPLE_FOR_ACCURACY_CHECK = 8     # don't judge accuracy off tiny samples
STALE_SIGNAL_HOURS           = 48     # news/weather age flag (these should update daily)
STALE_VALIDATED_COND_HOURS   = 200    # validated_conditions.json only regenerates weekly (~168h) — give it slack


def _load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        return {"__load_error__": str(e)}


# ── 1. MISSED-RUN DETECTION ──────────────────────────────────────────────────

def check_missed_run(state):
    issues = []
    if state is None:
        return ["wheat_monitor_state.json is missing entirely — monitor may never have run, or the file was deleted."]
    if isinstance(state, dict) and "__load_error__" in state:
        return [f"wheat_monitor_state.json exists but failed to parse: {state['__load_error__']}"]

    last_check = state.get("last_check")
    if not last_check:
        issues.append("state file has no 'last_check' timestamp — can't confirm the monitor is running.")
        return issues

    try:
        last_dt = datetime.fromisoformat(last_check)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=IL)
    except Exception:
        issues.append(f"'last_check' timestamp is unparseable: {last_check!r}")
        return issues

    hours_since = (datetime.now(IL) - last_dt).total_seconds() / 3600
    if hours_since > MAX_HOURS_SINCE_LAST_CHECK:
        issues.append(
            f"Monitor hasn't checked in for {hours_since:.0f}h (last: {last_check}). "
            f"Could be a GitHub Actions failure, quota limit, or disabled workflow."
        )
    return issues


# ── 2. PREDICTION LOG SANITY ─────────────────────────────────────────────────

def check_prediction_log(log):
    issues = []
    if log is None:
        return ["prediction_log.json is missing — no predictions have ever been logged, or the file was deleted."]
    if isinstance(log, dict) and "__load_error__" in log:
        return [f"prediction_log.json failed to parse: {log['__load_error__']}"]
    if not isinstance(log, list) or not log:
        return ["prediction_log.json is empty or not a list — nothing to validate."]

    # Schema drift: entries should consistently have the same shape.
    # (We know the log has old-format entries with 'factors' instead of
    # 'tier'/'seasonal_phase' — that's expected history, not a live bug.
    # Only flag if drift shows up in RECENT entries, i.e. going forward.)
    recent = [e for e in log if _is_recent(e.get("timestamp"), days=14)]
    for e in recent:
        if "tier" not in e:
            issues.append(f"Recent prediction ({e.get('timestamp')}) missing 'tier' field — schema drift.")
        elif e["tier"] not in (0, 1, 2):
            issues.append(f"Recent prediction ({e.get('timestamp')}) has out-of-range tier: {e['tier']!r}")

    # Tier-0 entries logged as real predictions after the tier-gating fix
    # shouldn't exist going forward — flag any recent tier:0 with a real
    # outcome, since that fix was specifically meant to stop this.
    for e in recent:
        if e.get("tier") == 0 and e.get("outcome") is not None:
            issues.append(
                f"Tier 0 prediction ({e.get('timestamp')}) has a logged outcome — "
                f"should not be logged as a trade under the current tier-gating rule."
            )

    # Stuck entries: validated=false for a long time (should resolve within
    # ~1-2 weeks given weekly setups break/close regularly).
    for e in log:
        if e.get("validated") is False and _is_recent(e.get("timestamp"), days=14, invert=True):
            issues.append(f"Prediction from {e.get('timestamp')} still unvalidated after 14+ days — may be stuck.")

    # Duplicate timestamps (exact same second) can indicate a double-run bug.
    timestamps = [e.get("timestamp") for e in log if e.get("timestamp")]
    dupes = {t for t in timestamps if timestamps.count(t) > 1}
    if dupes:
        issues.append(f"Duplicate prediction timestamps found ({len(dupes)}) — possible double-logging.")

    return issues


def _is_recent(ts_str, days, invert=False):
    if not ts_str:
        return False
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IL)
    except Exception:
        return False
    age_days = (datetime.now(IL) - ts).total_seconds() / 86400
    return (age_days > days) if invert else (age_days <= days)


# ── 3. LIVE VS BACKTEST ACCURACY DIVERGENCE ──────────────────────────────────

def check_accuracy_divergence(log, validated):
    issues = []
    if not isinstance(log, list) or not log:
        return issues  # already flagged by check_prediction_log

    claimed = {}
    if isinstance(validated, dict) and "validated_conditions" in validated:
        claimed = validated.get("validated_conditions", {})

    for tier in (1, 2):
        tier_entries = [e for e in log if e.get("tier") == tier and e.get("outcome") in ("WIN", "LOSS")]
        if len(tier_entries) < MIN_SAMPLE_FOR_ACCURACY_CHECK:
            continue
        wins = sum(1 for e in tier_entries if e["outcome"] == "WIN")
        live_rate = wins / len(tier_entries) * 100

        # Best available claimed accuracy for this tier's threshold band
        # (2 = >=80%, 1 = >=68%) — approximate reference point, not exact.
        band = [v for v in claimed.values() if isinstance(v, (int, float))]
        if not band:
            continue
        claimed_rate = max(band) * 100 if tier == 2 else min(band) * 100 if band else None
        if claimed_rate is None:
            continue

        gap = abs(live_rate - claimed_rate)
        if gap > ACCURACY_DIVERGENCE_PCT:
            issues.append(
                f"Tier {tier} live win rate ({live_rate:.1f}%, n={len(tier_entries)}) diverges "
                f"from backtest-claimed accuracy (~{claimed_rate:.1f}%) by {gap:.1f} points."
            )
    return issues


# ── 4. STATE FILE HEALTH ─────────────────────────────────────────────────────

def check_state_health(state):
    issues = []
    if not isinstance(state, dict) or "__load_error__" in (state or {}):
        return issues  # already flagged elsewhere

    alerts_today = state.get("alerts_today", {})
    if isinstance(alerts_today, dict) and len(alerts_today) > ALERTS_TODAY_CLEANUP_LIMIT:
        issues.append(
            f"'alerts_today' has {len(alerts_today)} entries with no apparent cleanup — "
            f"old date-keys (e.g. from May) are still present. File will grow unbounded."
        )

    # last_alert_time vs last_alert_date consistency check — current code
    # only ever sets last_alert_date, so a very old last_alert_time next to
    # a fresh last_alert_date is a stale/dead field, not a live bug, but
    # worth a heads-up since it can be misread as "last alert was in June."
    lad = state.get("last_alert_date")
    lat = state.get("last_alert_time")
    if lad and lat:
        try:
            date_part = lat.split("T")[0]
            if date_part != lad:
                issues.append(
                    f"'last_alert_time' ({lat}) doesn't match 'last_alert_date' ({lad}) — "
                    f"likely a stale/unused field left over from an older version of the code."
                )
        except Exception:
            pass

    return issues


# ── 5. SIGNAL-SOURCE FRESHNESS ───────────────────────────────────────────────

def check_signal_freshness():
    issues = []
    for label, path, ts_key, max_hours in [
        ("news_log.json", NEWS_LOG, "timestamp", STALE_SIGNAL_HOURS),
        ("weather_cache.json", WEATHER_CACHE, "ts", STALE_SIGNAL_HOURS),
        ("validated_conditions.json", VALIDATED_COND, "generated_at", STALE_VALIDATED_COND_HOURS),
    ]:
        data = _load_json(path)
        if data is None:
            issues.append(f"{label} is missing.")
            continue
        if isinstance(data, dict) and "__load_error__" in data:
            issues.append(f"{label} failed to parse: {data['__load_error__']}")
            continue

        entry = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
        ts_str = entry.get(ts_key) if isinstance(entry, dict) else None
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IL)
            age_h = (datetime.now(IL) - ts).total_seconds() / 3600
            if age_h > max_hours:
                issues.append(f"{label} is {age_h:.0f}h old — may be stale (source may have stopped updating).")
        except Exception:
            pass
    return issues


# ── REPORT + SEND ─────────────────────────────────────────────────────────────

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print("Telegram not configured — printing report instead:\n")
        print(message)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": message},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


def main():
    state      = _load_json(STATE_FILE)
    log        = _load_json(PREDICTION_LOG)
    validated  = _load_json(VALIDATED_COND)

    all_issues = []
    all_issues += check_missed_run(state)
    all_issues += check_prediction_log(log)
    all_issues += check_accuracy_divergence(log, validated)
    all_issues += check_state_health(state)
    all_issues += check_signal_freshness()

    now_str = datetime.now(IL).strftime("%Y-%m-%d %H:%M")
    if all_issues:
        lines = [f"🔎 Bug Detector Report ({now_str})", f"{len(all_issues)} issue(s) found:\n"]
        for i, issue in enumerate(all_issues, 1):
            lines.append(f"{i}. {issue}")
        message = "\n".join(lines)
    else:
        message = f"🔎 Bug Detector Report ({now_str})\nAll checks passed — no issues found."

    print(message)
    send_telegram(message)


if __name__ == "__main__":
    main()
