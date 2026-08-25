"""
BUG DETECTOR — Wheat Monitor supervisor tool
=============================================
Read-only. Never fixes anything, never touches live files, never
sends trading alerts. Its only job: notice when something is wrong
and tell you (via Telegram) so you and Claude can decide what to do.

Run this on its own schedule (e.g. daily or weekly via GitHub
Actions) — separate workflow from wheat_monitor_pro.py.

CHECKS INCLUDED (the four "core" ones + drift, plus two added 2026-08-21):
  1. Missed-run detection      — did the monitor actually run recently?
  1b. Missed weekday alert     — did the script run but silently skip sending?
  1c. Workflow schedule sync   — do the cron triggers and job if-conditions
                                  in the workflow YAML still match each other?
  1d. Undetected weekly breach — did price cross stop/target intraday and
                                  recover between runs, unrecorded?
  2. Prediction log sanity     — schema drift, bad values, stuck entries
  3. Live vs backtest accuracy — divergence between real and claimed win rate
  4. State file health         — unbounded growth, stale/unused fields
  5. Signal-source freshness   — news/weather/validated_conditions going stale

Add new checks as functions returning a list of issue strings.
Nothing here calls send_telegram except main() — easy to test checks
in isolation by just calling them and printing the list.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import numpy as np
import requests
import yfinance as yf

IL = ZoneInfo("Asia/Jerusalem")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

STATE_FILE       = Path("wheat_monitor_state.json")
PREDICTION_LOG   = Path("prediction_log.json")
VALIDATED_COND   = Path("validated_conditions.json")
NEWS_LOG         = Path("news_log.json")
WEATHER_CACHE    = Path("weather_cache.json")
WEEKLY_CACHE     = Path("weekly_range_cache.json")
MONTHLY_CACHE    = Path("monthly_range_cache.json")
WORKFLOW_FILE    = Path(".github/workflows/wheat_monitor_github.yml")

# ── thresholds (tune these over time, keep them here in one place) ──────────
MAX_HOURS_SINCE_LAST_CHECK   = 30     # flag if monitor hasn't run in this long
ALERTS_TODAY_CLEANUP_LIMIT   = 40     # flag if old date-keys are piling up
ACCURACY_DIVERGENCE_PCT      = 15     # flag if live vs backtest win rate gap exceeds this
MIN_SAMPLE_FOR_ACCURACY_CHECK = 8     # don't judge accuracy off tiny samples
STALE_SIGNAL_HOURS           = 48     # news/weather age flag (these should update daily)
STALE_VALIDATED_COND_HOURS   = 200    # validated_conditions.json only regenerates weekly (~168h) — give it slack
PRICE_DIVERGENCE_PCT_THRESHOLD = 1.5  # flag if ZW=F vs the front-month specific contract differ by more than this

# UPDATED 2026-08-25: front-month contract resolution + trading-day
# checks now live in a single shared module, trading_calendar.py,
# imported by both this file and wheat_monitor_pro.py — see that
# module's docstring for the full history/reasoning. This resolves
# the tension that used to force duplication here (bug_detector.py
# stays free of wheat_monitor_pro.py's heavy TensorFlow/XGBoost/
# sklearn dependencies, since trading_calendar.py only needs
# numpy/yfinance, which this file already imports anyway). If this
# logic ever needs to change again, change it once in
# trading_calendar.py — both files pick it up automatically.
from trading_calendar import (
    is_trading_day,
    is_weekend,
    WHEAT_MONTH_CODES,
    WHEAT_ROLL_BUFFER_DAYS,
    VOLUME_CROSSOVER_MULTIPLIER,
    get_front_month_ticker,
)



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


# ── 1b. MISSED WEEKDAY ALERT (decision-level check, 2026-08-17) ─────────────
# check_missed_run() above only confirms the SCRIPT executed — it can't
# tell if the script then made a wrong call and silently skipped
# sending. That's exactly what happened on 2026-08-17: a bug in the
# "is the market closed" logic caused a real Monday alert to be
# skipped, while last_check still updated normally and every other
# check here stayed green. This check is deliberately OUTCOME-based
# rather than cause-based — it doesn't know or care WHY an alert may
# be missing, so it will catch this same failure shape again even if
# the next bug is completely unrelated to market-closed logic.
def check_missed_weekday_alert(state):
    issues = []
    if not isinstance(state, dict) or "__load_error__" in (state or {}):
        return issues

    last_alert_date_str = state.get("last_alert_date")
    if not last_alert_date_str:
        return issues  # no alert history yet — nothing to compare

    try:
        last_alert_date = datetime.fromisoformat(last_alert_date_str).date()
    except Exception:
        return issues

    today = datetime.now(IL).date()

    # Most recent COMPLETED weekday (Mon-Fri) strictly before today —
    # by the time this check runs, that day's ~01:00 IL alert cycle
    # should already have happened.
    check_date = today - timedelta(days=1)
    while is_weekend(datetime(check_date.year, check_date.month, check_date.day, tzinfo=IL)):
        check_date -= timedelta(days=1)

    if last_alert_date < check_date:
        issues.append(
            f"No alert recorded since {last_alert_date_str}, but {check_date.isoformat()} "
            f"was a weekday that should have had one — possible silent skip (e.g. a bad "
            f"'market closed' decision) even though the monitor script itself ran on schedule. "
            f"NOTE: this can also fire on a genuine multi-day market holiday — check the "
            f"Actions log for that date before assuming it's a bug."
        )
    return issues


# ── 1c2. WEEKEND ALERT SENT + LATE HOUR (2026-08-23) ────────────────────────
# ADDED after a real incident: the weekly backtest cron shared the same
# job as the daily monitor, and its cron string ('30 22 * * 6', UTC
# Saturday) actually landed on IL SUNDAY — not the intended Saturday —
# so the shared job ran the full monitor and sent a real "scheduled"
# alert on a non-trading day. Fixed at the source (separate job,
# corrected cron) AND at the script level (should_send() now refuses to
# send on Sat/Sun regardless of trigger). This check is the equivalent
# safety net at the detection layer: rather than trying to prove cron
# correctness from the YAML alone (which doesn't state day-of-week
# INTENT, only what it literally does), it judges by the actual
# recorded outcome — did wheat_monitor_state.json's alerts_today ever
# get a real entry dated a Saturday or Sunday? That catches this whole
# incident class regardless of which future cron/job/timezone mistake
# might cause it.
#
# EXTENDED same day: alerts_today's value changed from a bare `True` to
# the actual HH:MM send time (see wheat_monitor_pro.py), specifically so
# this check can also verify the HOUR, not just the day — the daily
# target is ~00:53 IL; a genuine GitHub Actions delay of a few minutes
# is expected and fine (see the earlier cron-shift fix), but a send
# hours off target would indicate something is wrong even on a correct
# weekday. LATE_ALERT_HOUR_THRESHOLD is generous on purpose to avoid
# false alarms from ordinary platform delay.
LATE_ALERT_HOUR_THRESHOLD = 4  # flag if the scheduled morning send lands at/after this IL hour

def check_weekend_alert_sent(state):
    issues = []
    if not isinstance(state, dict) or "__load_error__" in (state or {}):
        return issues

    alerts_today = state.get("alerts_today", {})
    if not isinstance(alerts_today, dict):
        return issues

    for slot_key, send_time in alerts_today.items():
        date_str = slot_key.split("_")[0]
        try:
            d = datetime.fromisoformat(date_str).date()
        except Exception:
            continue

        # UPDATED 2026-08-25, real incident: wheat_monitor_pro.py's
        # alerts_today write is gated `if not is_manual` (since
        # 2026-08-23), so any entry written by the CURRENT code can
        # only be a genuine scheduled send — never a manual trigger.
        # But entries written BEFORE that gating fix landed still
        # exist in the wild, stored as a bare `True` rather than the
        # HH:MM string format introduced the same day. A bare-`True`
        # weekend entry is unattributable history (could predate
        # several different fixes, including the day-of-week guard
        # itself) — NOT proof the current code has a live bug. An
        # HH:MM-format weekend entry, by contrast, could only have
        # been written by the current, already-fixed code, so it IS
        # solid evidence of a real, ongoing problem.
        is_new_format = isinstance(send_time, str) and ":" in send_time

        if is_weekend(datetime(d.year, d.month, d.day, tzinfo=IL)):
            if is_new_format:
                issues.append(
                    f"⚠️ LIVE BUG: a scheduled alert landed at {send_time} IL on "
                    f"{date_str} ({d.strftime('%A')}) — not a trading day. This "
                    f"entry uses the current HH:MM format (introduced 2026-08-23 "
                    f"alongside is_manual-gated writes), meaning the CURRENT "
                    f"should_send() code did not block a real scheduled send — "
                    f"investigate the actual cron trigger for that day."
                )
            else:
                issues.append(
                    f"Historical note only: an old-format alert record exists for "
                    f"{date_str} ({d.strftime('%A')}) — not a trading day. This "
                    f"entry predates the 2026-08-23 alerts_today format/gating "
                    f"change, so it can't be attributed to the current code — "
                    f"likely old history from before an earlier fix. No action "
                    f"needed unless a NEW-format (HH:MM) weekend entry also appears."
                )
            continue  # wrong-day already flagged, skip the hour check for this entry

        # Hour check — only meaningful for entries that DO have a real
        # HH:MM time recorded (older entries before this fix may still
        # be a bare `True`, skip those rather than error on them).
        if isinstance(send_time, str) and ":" in send_time:
            try:
                send_hour = int(send_time.split(":")[0])
                if send_hour >= LATE_ALERT_HOUR_THRESHOLD:
                    issues.append(
                        f"Scheduled alert on {date_str} landed at {send_time} IL — "
                        f"well past the ~00:53 IL target (threshold: hour "
                        f"{LATE_ALERT_HOUR_THRESHOLD}+). Worth checking Actions run "
                        f"history for that day for unusual delay or a job that ran "
                        f"very late."
                    )
            except Exception:
                pass
    return issues


# ── 1c3. BACKTEST RAN ON WRONG DAY OR HOUR (2026-08-23) ──────────────────────
# ADDED alongside check_weekend_alert_sent() above, same real incident —
# now that the backtest has its own dedicated job (never sends an
# alert), it still deserves the same "did it actually land on the
# intended day AND hour" check. backtest.py writes run_date via
# datetime.now() with NO timezone — on a GitHub Actions runner that's
# naive UTC, not IL, so it must be explicitly localized before checking
# weekday/hour (the same UTC/IL confusion behind several bugs this
# session). Flags if the most recent backtest run_date, converted to
# IL, isn't a Saturday, OR lands well outside its ~01:30 IL target.
BACKTEST_RESULTS_FILE = Path("backtest_results.json")
BACKTEST_EXPECTED_HOUR_RANGE = (0, 4)  # generous buffer around the ~01:30 IL target

def check_backtest_ran_wrong_day():
    issues = []
    data = _load_json(BACKTEST_RESULTS_FILE)
    if not isinstance(data, dict) or "run_date" not in data:
        return issues

    try:
        run_date_naive = datetime.fromisoformat(data["run_date"])
        run_date_utc = run_date_naive.replace(tzinfo=ZoneInfo("UTC"))
        run_date_il = run_date_utc.astimezone(IL)
    except Exception:
        return issues

    if run_date_il.weekday() != 5:  # Saturday=5
        issues.append(
            f"Most recent backtest run_date ({data['run_date']} UTC = "
            f"{run_date_il.strftime('%Y-%m-%d %H:%M %A')} IL) was NOT a Saturday — "
            f"the weekly_backtest job's cron may be pointing at the wrong day again, "
            f"or this was a manual/off-schedule run (check before assuming a bug)."
        )
    elif not (BACKTEST_EXPECTED_HOUR_RANGE[0] <= run_date_il.hour < BACKTEST_EXPECTED_HOUR_RANGE[1]):
        issues.append(
            f"Most recent backtest ran on the correct day (Saturday) but at "
            f"{run_date_il.strftime('%H:%M')} IL — outside the expected "
            f"{BACKTEST_EXPECTED_HOUR_RANGE[0]:02d}:00-{BACKTEST_EXPECTED_HOUR_RANGE[1]:02d}:00 "
            f"window. Worth checking whether this was a manual run or a real delay."
        )
    return issues


# ── WEEKLY REPORT RAN ON WRONG DAY OR HOUR (2026-08-23) ──────────────────────
# ADDED to close a real gap identified alongside the backtest/weekend-alert
# checks: weekly_report.py previously wrote NOTHING to disk, so there was
# no way to verify it actually ran on the correct day/hour — it now
# writes weekly_report_state.json (already IL-aware, no naive-UTC issue
# like the backtest had). Target is IL Saturday ~03:00.
WEEKLY_REPORT_STATE_FILE = Path("weekly_report_state.json")
WEEKLY_REPORT_EXPECTED_HOUR_RANGE = (2, 6)  # generous buffer around the ~03:00 IL target

def check_weekly_report_ran_wrong_day():
    issues = []
    data = _load_json(WEEKLY_REPORT_STATE_FILE)
    if not isinstance(data, dict) or "last_run" not in data:
        return issues

    try:
        last_run = datetime.fromisoformat(data["last_run"])
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=IL)
        last_run_il = last_run.astimezone(IL)
    except Exception:
        return issues

    if last_run_il.weekday() != 5:  # Saturday=5
        issues.append(
            f"Most recent weekly_report.py run ({last_run_il.strftime('%Y-%m-%d %H:%M %A')} IL) "
            f"was NOT a Saturday — the saturday_report job's cron may be pointing at the "
            f"wrong day, or this was a manual/off-schedule run (check before assuming a bug)."
        )
    elif not (WEEKLY_REPORT_EXPECTED_HOUR_RANGE[0] <= last_run_il.hour < WEEKLY_REPORT_EXPECTED_HOUR_RANGE[1]):
        issues.append(
            f"Most recent weekly_report.py run was on the correct day (Saturday) but at "
            f"{last_run_il.strftime('%H:%M')} IL — outside the expected "
            f"{WEEKLY_REPORT_EXPECTED_HOUR_RANGE[0]:02d}:00-{WEEKLY_REPORT_EXPECTED_HOUR_RANGE[1]:02d}:00 "
            f"window. Worth checking whether this was a manual run or a real delay."
        )
    return issues


# just silently skips on every scheduled trigger, with the workflow
# still showing green. Nothing else in this file could have caught
# it: check_missed_run() only confirms the SCRIPT ran, but if the job
# never starts at all, there's no script execution to check.
# This is a plain text/regex check on the workflow YAML itself — no
# yaml/pyyaml dependency, kept lightweight on purpose. It checks that
# every cron string declared under `on: schedule:` is referenced by
# at least one job's `if:` condition, and vice versa — catching drift
# in EITHER direction, whatever the specific rename turns out to be.
def check_workflow_schedule_consistency():
    issues = []
    if not WORKFLOW_FILE.exists():
        return issues  # not fatal — just can't run this check here

    try:
        text = WORKFLOW_FILE.read_text()
    except Exception as e:
        return [f"Could not read {WORKFLOW_FILE}: {e}"]

    declared_crons = set(re.findall(r"- cron:\s*'([^']+)'", text))

    referenced_crons = set()
    for list_match in re.findall(r"fromJSON\('(\[[^\]]*\])'\)", text):
        try:
            referenced_crons.update(json.loads(list_match))
        except Exception:
            pass
    referenced_crons.update(re.findall(r"github\.event\.schedule\s*==\s*'([^']+)'", text))

    missing_from_if = declared_crons - referenced_crons
    missing_from_schedule = referenced_crons - declared_crons

    if missing_from_if:
        issues.append(
            f"Workflow schedule/if-condition MISMATCH: cron(s) {sorted(missing_from_if)} are "
            f"declared under 'on: schedule:' but not referenced by any job's 'if:' condition — "
            f"that job would silently never run on this trigger."
        )
    if missing_from_schedule:
        issues.append(
            f"Workflow schedule/if-condition MISMATCH: cron(s) {sorted(missing_from_schedule)} "
            f"are referenced in a job's 'if:' condition but no longer exist under 'on: schedule:' — "
            f"likely a stale string left over from a schedule change."
        )
    return issues


# ── 1e. UNDETECTED WEEKLY BREACH — INTRADAY CHECK (2026-08-21) ──────────────
# ADDED after a real (unresolved) ambiguity: get_frozen_weekly_plan() in
# wheat_monitor_pro.py only compares a single LIVE PRICE SNAPSHOT against
# stop/target each time it runs. If price crosses the stop or target
# between scheduled runs and recovers before the next one, the breach
# is never recorded — even though a real resting order at that stop/
# target would have been filled. This re-checks the CURRENTLY FROZEN
# weekly setup against real intraday high/low since it was frozen (not
# just a snapshot) and flags a likely-missed breach. Supplementary only
# — never blocks or alters the live monitor's own freeze/break logic,
# just surfaces the discrepancy for a human to look at.
def check_undetected_weekly_breach():
    issues = []
    cached = _load_json(WEEKLY_CACHE)
    if not isinstance(cached, dict) or "weekly" not in cached:
        return issues

    weekly = cached["weekly"]
    stop, target = weekly.get("stop"), weekly.get("target")
    final_call = weekly.get("final_call")
    frozen_at_str = cached.get("frozen_at")
    if stop is None or target is None or final_call not in ("UP", "DOWN") or not frozen_at_str:
        return issues

    try:
        frozen_at = datetime.fromisoformat(frozen_at_str)
        if frozen_at.tzinfo is None:
            frozen_at = frozen_at.replace(tzinfo=IL)
    except Exception:
        return issues

    try:
        ticker = get_front_month_ticker()
        hist = yf.Ticker(ticker).history(period="7d", interval="60m")
        if hist.empty:
            return issues
        hist = hist[hist.index >= frozen_at]
        if hist.empty:
            return issues
        period_high = float(hist['High'].max())
        period_low  = float(hist['Low'].min())
    except Exception:
        return issues  # data-source hiccup — don't fail loudly on a supplementary check

    if final_call == 'UP':
        if period_low <= stop:
            issues.append(
                f"Currently frozen weekly setup (UP, stop {stop:.0f}c) shows an intraday LOW of "
                f"{period_low:.0f}c since it was frozen ({frozen_at_str}) — the stop may have been "
                f"crossed and price recovered before the next scheduled run, without being recorded "
                f"as a LOSS. Worth a manual check against weekly_break_log.json."
            )
        if period_high >= target:
            issues.append(
                f"Currently frozen weekly setup (UP, target {target:.0f}c) shows an intraday HIGH of "
                f"{period_high:.0f}c since it was frozen ({frozen_at_str}) — the target may have been "
                f"reached and price pulled back before the next scheduled run, without being recorded "
                f"as a WIN."
            )
    else:  # DOWN
        if period_high >= stop:
            issues.append(
                f"Currently frozen weekly setup (DOWN, stop {stop:.0f}c) shows an intraday HIGH of "
                f"{period_high:.0f}c since it was frozen ({frozen_at_str}) — the stop may have been "
                f"crossed without being recorded as a LOSS."
            )
        if period_low <= target:
            issues.append(
                f"Currently frozen weekly setup (DOWN, target {target:.0f}c) shows an intraday LOW of "
                f"{period_low:.0f}c since it was frozen ({frozen_at_str}) — the target may have been "
                f"reached without being recorded as a WIN."
            )
    return issues



# ADDED after a real incident: wheat_monitor_pro.py's live price was
# reported as ~693-694c while the actual tradeable market (confirmed
# against Plus500 and a direct side-by-side yfinance diagnostic) was
# ~677-679c — a ~2.3% gap large enough to materially change trade
# setup math. Root cause: ZW=F (generic continuous symbol) was
# silently tracking a different contract month than the specific
# front-month contract (e.g. ZWU26.CBT). wheat_monitor_pro.py's
# get_live_price() was fixed to prefer the specific contract, but
# NOTHING previously checked on an ongoing basis whether these two
# sources still agree — this fills that gap, the same way
# check_missed_weekday_alert() fills the "script ran but made a wrong
# call" gap check_missed_run() can't see.
#
# NOTE: some divergence between ZW=F and the specific front-month
# contract is NORMAL, especially right around a contract roll
# (calendar spread / contango) — this flags anything above threshold
# for a human glance, it does not assert something is definitely
# broken.
def check_price_source_divergence():
    issues = []
    try:
        front_ticker = get_front_month_ticker()

        zwf_fast = yf.Ticker("ZW=F").fast_info
        zwf_price = zwf_fast.get("last_price") or zwf_fast.get("lastPrice")

        front_fast = yf.Ticker(front_ticker).fast_info
        front_price = front_fast.get("last_price") or front_fast.get("lastPrice")

        if zwf_price and front_price and front_price > 0:
            pct_diff = abs(zwf_price - front_price) / front_price * 100
            if pct_diff > PRICE_DIVERGENCE_PCT_THRESHOLD:
                issues.append(
                    f"ZW=F ({zwf_price}) and front-month contract {front_ticker} ({front_price}) "
                    f"diverge by {pct_diff:.1f}% — beyond the {PRICE_DIVERGENCE_PCT_THRESHOLD}% "
                    f"threshold. Can be normal near a contract roll (calendar spread/contango), "
                    f"but worth a quick check against a live broker price to confirm which "
                    f"source is currently accurate."
                )
    except Exception as e:
        # Don't fail the whole bug_detector run if Yahoo Finance can't
        # be reached right now — skip silently, same fail-soft
        # convention as every other check in this file.
        pass
    return issues


# ── 1d. CURRENTLY-BREACHED RANGE CHECK (2026-08-19) ──────────────────────────
# ADDED after confirming wheat_monitor_pro.py's weekly/monthly frozen
# ranges only get re-checked for a breach when wheat_monitor_pro.py
# itself runs — a real price spike that happens between two of its
# runs, and reverts before the next one, can never be caught by
# either range-freezing mechanism (this is an inherent limit of any
# point-in-time check, not a bug to "fix" away). What CAN be checked
# independently: is the CURRENTLY CACHED range breached RIGHT NOW,
# at bug_detector's own separate run time? Since bug_detector runs on
# its own schedule (06:00 UTC daily), this is a genuinely independent
# second check, not a duplicate of wheat_monitor_pro.py's — it can
# catch a case where the regeneration logic itself has a bug and a
# stale, currently-breached range is stuck showing, which the monitor
# script's own point-in-time check would have already reported as
# fine on its last run.
def check_range_currently_breached():
    issues = []
    try:
        front_ticker = get_front_month_ticker()
        fast = yf.Ticker(front_ticker).fast_info
        price = fast.get("last_price") or fast.get("lastPrice")
        if not price:
            return issues

        weekly_data = _load_json(WEEKLY_CACHE)
        if isinstance(weekly_data, dict) and "weekly" in weekly_data:
            w = weekly_data["weekly"]
            stop, target, final_call = w.get("stop"), w.get("target"), w.get("final_call")
            if stop is not None and target is not None and final_call in ("UP", "DOWN"):
                lo, hi = (stop, target) if final_call == "UP" else (target, stop)
                if not (lo <= price <= hi):
                    issues.append(
                        f"Live price ({price:.1f}) is currently OUTSIDE the frozen weekly "
                        f"range (stop={stop:.1f}, target={target:.1f}, {final_call}) as of "
                        f"this check — if wheat_monitor_pro.py hasn't regenerated it yet, "
                        f"the next monitor run should catch this; if it's been stale for a "
                        f"while, worth checking the regeneration logic itself."
                    )

        monthly_data = _load_json(MONTHLY_CACHE)
        if isinstance(monthly_data, dict) and "monthly" in monthly_data:
            m = monthly_data["monthly"]
            lo, hi = m.get("monthly_low"), m.get("monthly_high")
            if lo is not None and hi is not None and not (lo <= price <= hi):
                issues.append(
                    f"Live price ({price:.1f}) is currently OUTSIDE the frozen monthly "
                    f"range ({lo:.1f}-{hi:.1f}) as of this check — same note as above: "
                    f"expected to self-correct on the next monitor run, worth a look if "
                    f"it's been stale for a while."
                )
    except Exception:
        pass
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


def _entry_after(ts_str, cutoff):
    if not ts_str:
        return False
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IL)
        return ts >= cutoff
    except Exception:
        return False


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

def check_accuracy_divergence(log, validated, state):
    issues = []
    if not isinstance(log, list) or not log:
        return issues  # already flagged by check_prediction_log

    # Only count predictions from the stats cutoff forward — avoids
    # blending pre-gate-change and post-gate-change trades into one
    # misleading number. Old entries stay in the log for history; they
    # just stop feeding this comparison. Set stats_cutoff_date in
    # wheat_monitor_state.json whenever the gate/conditions change
    # meaningfully (e.g. after a ConvictionGate condition set update).
    cutoff_str = (state or {}).get("stats_cutoff_date")
    if cutoff_str:
        try:
            cutoff = datetime.fromisoformat(cutoff_str).replace(tzinfo=IL)
            log = [e for e in log if _entry_after(e.get("timestamp"), cutoff)]
        except Exception:
            pass

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


# ── NEWS SCAN MISSED WINDOW (2026-08-23) ─────────────────────────────────────
# ADDED alongside the day/hour checks above, same session — the existing
# check_signal_freshness() only catches "the MOST RECENT scan is too
# old overall"; it would miss a case where, say, scan #2 (mid-session)
# silently failed one day but scan #3 (afternoon) ran fine a few hours
# later, keeping the overall "most recent" timestamp looking fresh.
# This checks each of the 3 daily windows independently for the most
# recent day with any news_log.json activity, and flags a window with
# ZERO entries. Windows are intentionally generous (not tied to the
# exact 00:28/11:35/16:05 IL targets) since entries aren't individually
# labeled with which of the 3 daily scans wrote them — this catches a
# genuinely MISSED slot, not precise per-scan lateness (a tighter check
# would need news_scanner.py to tag which scan number wrote each
# entry — not done here, kept simple and low-risk of false positives).
# News scans run every day (no day-of-week restriction by design), so
# this checks the most recent calendar day, weekday or not.
NEWS_SCAN_WINDOWS = [
    ("scan #1 (~00:28 IL)", 0, 4),
    ("scan #2 (~11:35 IL)", 9, 13),
    ("scan #3 (~16:05 IL)", 14, 19),
]

def check_news_scan_missed_window():
    issues = []
    log = _load_json(NEWS_LOG)
    if not isinstance(log, list) or not log:
        return issues

    entries_il = []
    for e in log:
        ts_str = e.get("timestamp") if isinstance(e, dict) else None
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ZoneInfo("UTC"))
            entries_il.append(ts.astimezone(IL))
        except Exception:
            continue
    if not entries_il:
        return issues

    most_recent_date = max(e.date() for e in entries_il)
    today_entries = [e for e in entries_il if e.date() == most_recent_date]
    now_il = datetime.now(IL)

    for label, start_hour, end_hour in NEWS_SCAN_WINDOWS:
        # Only judge a window as "missed" once it has actually fully
        # elapsed — caught in testing: without this, any run of this
        # check during the current day (before later windows have even
        # happened yet) would falsely flag scan #2/#3 as missing simply
        # because the day isn't over. Only skip this guard for a date
        # strictly before today, which is always fully elapsed.
        if most_recent_date == now_il.date() and now_il.hour < end_hour:
            continue

        covered = any(start_hour <= e.hour < end_hour for e in today_entries)
        if not covered:
            issues.append(
                f"No news_log.json entry found for {label} on {most_recent_date.isoformat()} "
                f"({most_recent_date.strftime('%A')}) — that scan may have silently failed "
                f"that day, even if scans before/after it ran fine."
            )
    return issues




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
    all_issues += check_missed_weekday_alert(state)
    all_issues += check_weekend_alert_sent(state)
    all_issues += check_backtest_ran_wrong_day()
    all_issues += check_weekly_report_ran_wrong_day()
    all_issues += check_workflow_schedule_consistency()
    all_issues += check_price_source_divergence()
    all_issues += check_range_currently_breached()
    all_issues += check_undetected_weekly_breach()
    all_issues += check_prediction_log(log)
    all_issues += check_accuracy_divergence(log, validated, state)
    all_issues += check_state_health(state)
    all_issues += check_signal_freshness()
    all_issues += check_news_scan_missed_window()

    now = datetime.now(IL)
    now_str = now.strftime("%Y-%m-%d %H:%M")
    is_weekly_report_day = now.weekday() == 6  # Sunday — full report regardless of issues

    if all_issues:
        lines = [f"🔎 Bug Detector Report ({now_str})", f"{len(all_issues)} issue(s) found:\n"]
        for i, issue in enumerate(all_issues, 1):
            lines.append(f"{i}. {issue}")
        message = "\n".join(lines)
        print(message)
        send_telegram(message)  # always alert immediately when something's wrong
    elif is_weekly_report_day:
        message = f"🔎 Bug Detector — Weekly Report ({now_str})\nAll checks passed — no issues found."
        print(message)
        send_telegram(message)
    else:
        # Silent day, nothing wrong — don't ping Telegram, just log for the Actions run.
        print(f"🔎 Bug Detector ({now_str}) — all checks passed, no report sent (not weekly report day).")


if __name__ == "__main__":
    main()
