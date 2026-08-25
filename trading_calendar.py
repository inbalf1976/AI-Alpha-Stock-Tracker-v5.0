"""
trading_calendar.py — single shared source of truth for all
trading-day, trading-hour, and front-month-contract logic used
across wheat_monitor_pro.py and bug_detector.py.

WHY THIS EXISTS (2026-08-25): every "calendar/clock" bug found in
this project — the Monday market-closed false positive (2026-08-17),
the Sunday-alert bug, the Friday-close-time question, and the
September contract roll-timing issue — came from the same root
cause: duplicated date/time logic drifting out of sync between
files. get_front_month_ticker() etc. were previously intentionally
DUPLICATED between wheat_monitor_pro.py and bug_detector.py, to avoid
dragging wheat_monitor_pro.py's heavy TensorFlow/XGBoost/sklearn
dependencies into bug_detector.py (which is designed to stay
lightweight). This module resolves that tension: it only needs
datetime/zoneinfo/numpy/yfinance — all of which bug_detector.py
already imports anyway — so BOTH files import this module directly
instead of maintaining separate copies that can silently disagree.

Every file needing trading-day / trading-hour / front-month-contract
logic should import from here, not reimplement it. If this file's
logic ever changes, both callers pick it up automatically — that's
the whole point.
"""

import numpy as np
import yfinance as yf
from datetime import datetime, date
from zoneinfo import ZoneInfo

IL = ZoneInfo("Asia/Jerusalem")
UTC = ZoneInfo("UTC")

# ---------------------------------------------------------------------------
# TRADING CALENDAR
# ---------------------------------------------------------------------------
# CBOT wheat trades Monday 01:00 through Friday 23:59 Israel time —
# confirmed directly by the user (2026-08-25 correction), NOT the
# Sun-Thu window an earlier session mistakenly assumed. This module
# treats the whole of Monday-Friday as trading days; it does not
# model market holidays (a real gap, same as before this module
# existed — worth a future addition if it ever causes a real miss).

def is_trading_day(dt=None):
    """
    True if dt (default: now, IL) falls on a trading weekday
    (Mon-Fri). Naive datetimes are assumed to already be in IL.
    """
    dt = dt or datetime.now(IL)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IL)
    return dt.weekday() in (0, 1, 2, 3, 4)  # Mon=0 ... Fri=4


def is_weekend(dt=None):
    return not is_trading_day(dt)


# ---------------------------------------------------------------------------
# FRONT-MONTH CONTRACT RESOLUTION
# ---------------------------------------------------------------------------
WHEAT_MONTH_CODES = {3: 'H', 5: 'K', 7: 'N', 9: 'U', 12: 'Z'}
WHEAT_ROLL_BUFFER_DAYS = 5          # TRADING days before the ~15th cutoff
VOLUME_CROSSOVER_MULTIPLIER = 1.5   # live override threshold, see get_front_month_ticker()


def _get_contract_volume(ticker_symbol):
    """
    Best-effort fetch of today's trading volume for a contract.
    Returns None (never raises) on any failure — callers must treat
    None as "unknown", never as zero (zero would wrongly look like a
    dead/no-activity contract when it's really just an unreachable
    data source).
    """
    try:
        hist = yf.Ticker(ticker_symbol).history(period='1d')
        if not hist.empty and 'Volume' in hist.columns:
            vol = hist['Volume'].iloc[-1]
            if vol and vol > 0:
                return float(vol)
    except Exception:
        pass
    return None


def get_front_month_ticker(reference_date=None, allow_volume_override=True):
    """
    Returns the current front-month CBOT wheat contract ticker
    (e.g. 'ZWU26.CBT'). Two layers, both built from real incidents:

    1. TRADING-DAY-AWARE CALENDAR BUFFER (built 2026-08-18, refined
       2026-08-25). The old rule rolled to the NEXT contract the
       instant the calendar entered a contract's own delivery month
       (e.g. Sept 1 already pointed to December) — overly
       conservative, stopped using each contract ~2 weeks before it
       actually expired (CBOT's real last trading day is the business
       day before the 15th). Now stays on the current delivery-month
       contract until WHEAT_ROLL_BUFFER_DAYS *trading* days (Mon-Fri,
       via np.busday_count — not raw calendar days, which let a
       weekend silently eat into the buffer) before that ~15th
       cutoff.

    2. LIVE VOLUME-CROSSOVER OVERRIDE (2026-08-25, real incident):
       confirmed live that trading volume can shift decisively to the
       NEXT contract while the calendar-only rule still confidently
       points at the current one — a full contract cycle before the
       calendar rule would even consider rolling (e.g. real Sept
       volume ~918 vs. Dec volume ~2428, while still in August).
       Real market participants — and apparently brokers like
       Plus500 — can roll ahead of CBOT's official calendar based on
       where liquidity actually is; no calendar buffer, however
       tuned, can reliably predict that by date math alone. After
       picking a candidate via the calendar rule, this checks today's
       live volume for that candidate vs. the next contract in
       sequence; if the next one already has more than
       VOLUME_CROSSOVER_MULTIPLIER times the volume, uses it instead.
       This step is a LIVE NETWORK CALL — it fails soft: if the
       volume check can't complete for any reason, the calendar-based
       candidate is used unchanged.

    allow_volume_override=False skips step 2 entirely (pure offline
    date math, no network call).
    """
    ref = reference_date or datetime.now(IL)
    months = sorted(WHEAT_MONTH_CODES.keys())
    year = ref.year

    def contract_str(m, y):
        return f"ZW{WHEAT_MONTH_CODES[m]}{str(y)[-2:]}.CBT"

    def trading_days_until(target_date):
        return int(np.busday_count(ref.date(), target_date))

    candidate_m, candidate_y = None, None
    for m in months:
        if ref.month < m:
            candidate_m, candidate_y = m, year
            break
        if ref.month == m:
            fifteenth = date(year, m, 15)
            if trading_days_until(fifteenth) >= WHEAT_ROLL_BUFFER_DAYS:
                candidate_m, candidate_y = m, year
                break
    if candidate_m is None:
        candidate_m, candidate_y = months[0], year + 1

    candidate = contract_str(candidate_m, candidate_y)

    if not allow_volume_override:
        return candidate

    idx = months.index(candidate_m)
    if idx + 1 < len(months):
        next_m, next_y = months[idx + 1], candidate_y
    else:
        next_m, next_y = months[0], candidate_y + 1
    next_contract = contract_str(next_m, next_y)

    candidate_vol = _get_contract_volume(candidate)
    next_vol = _get_contract_volume(next_contract)

    if candidate_vol is not None and next_vol is not None:
        if next_vol > candidate_vol * VOLUME_CROSSOVER_MULTIPLIER:
            print(f"   ⚠️ Volume crossover: {next_contract} (vol {next_vol:.0f}) has "
                  f"overtaken {candidate} (vol {candidate_vol:.0f}) — using "
                  f"{next_contract} as front month instead.")
            return next_contract

    return candidate
