"""
anti_hunt_filter.py  (v2)
=========================
Institutional open anti-stop-hunting Short filter for Chicago SRW Wheat (ZW=F).

Runs during the safe institutional window (8:45 AM - 12:30 PM America/Chicago,
weekdays only), anchors to the session Opening Price, and calculates a protected
Short setup engineered to sit outside typical high-frequency sweep zones.

v2 additions vs v1:
  - Dual-anchor open: first 15m candle of the current session, daily bar fallback
  - Retry with backoff on yfinance fetches (Yahoo blocks datacenter IPs at times)
  - Data staleness check (15m bars older than 45 min are flagged)
  - CME holiday skip (note: update HOLIDAYS yearly; empty-data guard is the
    real safety net for partial/early-close days)
  - Tick-size rounding (ZW trades in $0.25 increments)
  - ATR(16, 15m) noise check: warns if stop distance < 2x ATR
  - Setup invalidation skip if price is already above the stop
  - Entry cutoff: no fresh setups after 11:30 CT (levels expire 12:30 CT)
  - Telegram HTML parse mode (no escaping landmines)
  - Alert failure no longer fails the run (analysis != delivery)
  - Writes setup.json (persisted as a GitHub Actions artifact)
  - Optional HEALTHCHECK_URL dead-man ping (e.g. healthchecks.io)

Dependencies: yfinance, pandas, requests
Environment:  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (required for delivery)
              HEALTHCHECK_URL (optional)
"""

import os
import sys
import json
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, time as dt_time, date
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "ZW=F"
CHICAGO_TZ = ZoneInfo("America/Chicago")
TICK_SIZE = 0.25          # ZW minimum price increment ($/bu)
ATR_PERIOD = 16           # 15m bars (~4h of session)
ATR_STOP_MIN_MULT = 2.0   # warn if stop distance < 2x ATR (noise-level)
MAX_DATA_AGE_MIN = 45     # staleness threshold for the last 15m bar
MAX_FETCH_ATTEMPTS = 3

WINDOW_OPEN = dt_time(8, 45)    # 8:45 AM Chicago  (grains open 8:30 CT)
WINDOW_CLOSE = dt_time(12, 30)  # 12:30 PM Chicago
ENTRY_CUTOFF = dt_time(11, 30)  # no fresh setups sent after this

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEALTHCHECK_URL = os.environ.get("HEALTHCHECK_URL", "").strip()

# Anti-hunt geometry (multipliers applied to the open anchor)
ENTRY_MULT = 0.994   # Sell Limit: waits for a brief pump into the entry zone
STOP_MULT = 1.012    # Stop-Loss: tucked above the institutional open/high ceiling
TARGET_MULT = 0.960  # Target: macro support zone

# CME full-day closures observed for grain futures. Update yearly.
HOLIDAYS = {
    date(2026, 1, 1),  date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),  date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1),  date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5),  date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def ping_healthcheck() -> None:
    """Dead-man switch: a monitoring service alerts us if this never fires."""
    if not HEALTHCHECK_URL:
        return
    try:
        requests.get(HEALTHCHECK_URL, timeout=10)
    except requests.RequestException as exc:
        print(f"Healthcheck ping failed (non-fatal): {exc}", file=sys.stderr)


def round_tick(price: float, tick: float = TICK_SIZE) -> float:
    """Round to the exchange tick so levels are actually fillable."""
    return round(round(price / tick) * tick, 4)


def check_time_window() -> bool:
    """
    True only if current America/Chicago time is a weekday between
    8:45 AM and 12:30 PM (the safe institutional trading window).
    """
    now_ct = datetime.now(CHICAGO_TZ)
    if now_ct.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    return WINDOW_OPEN <= now_ct.time() <= WINDOW_CLOSE


def is_cme_holiday(d: date) -> bool:
    return d in HOLIDAYS


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Newer yfinance returns MultiIndex columns; flatten to plain names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_market_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull 2 days of 15-minute bars and 5 days of daily bars for ZW=F,
    with retry/backoff (Yahoo intermittently blocks datacenter IPs).
    """
    intraday = daily = None
    for attempt in range(1, MAX_FETCH_ATTEMPTS + 1):
        try:
            intraday = _flatten(yf.download(
                TICKER, period="2d", interval="15m",
                auto_adjust=False, progress=False,
            ))
            daily = _flatten(yf.download(
                TICKER, period="5d", interval="1d",
                auto_adjust=False, progress=False,
            ))
        except Exception as exc:
            print(f"Fetch attempt {attempt} raised: {exc}", file=sys.stderr)
        if intraday is not None and daily is not None \
                and not intraday.empty and not daily.empty:
            return intraday, daily
        time.sleep(10 * attempt)  # 10s, 20s, 30s

    raise ValueError(
        "yfinance returned no usable data after "
        f"{MAX_FETCH_ATTEMPTS} attempts (holiday, blockage, or outage)."
    )


def resolve_session_open(intraday: pd.DataFrame, daily: pd.DataFrame) -> float:
    """
    Anchor: the first 15m candle of the CURRENT Chicago session (the true
    8:30 CT grain open). Falls back to the latest daily bar's open when the
    intraday series doesn't yet contain today's session.
    """
    now_ct = datetime.now(CHICAGO_TZ)
    idx = intraday.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    todays_bars = intraday[idx.tz_convert(CHICAGO_TZ).date == now_ct.date()]
    if not todays_bars.empty:
        return float(todays_bars["Open"].iloc[0]) # FIXED: Index closure added
    return float(daily["Open"].iloc[-1])


def compute_atr(intraday: pd.DataFrame) -> float | None:
    """ATR(PERIOD) on 15m bars; None if not enough history."""
    if len(intraday) < ATR_PERIOD + 1:
        return None
    high, low, close = intraday["High"], intraday["Low"], intraday["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(ATR_PERIOD).mean().iloc[-1])


def check_staleness(intraday: pd.DataFrame) -> float:
    """Age of the last 15m bar in minutes; NaN-aware."""
    idx = intraday.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    last_bar_ct = idx[-1].tz_convert(CHICAGO_TZ)
    return (datetime.now(CHICAGO_TZ) - last_bar_ct).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Telegram alerting
# ---------------------------------------------------------------------------
def send_telegram_alert(text: str) -> bool:
    """
    POST an HTML message to the Telegram Bot API.
    Returns True on success. Analysis success does NOT depend on this.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing — alert printed to stdout instead.")
        print(text)
        return False

    url = f"https://telegram.org{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            print("📲 Telegram alert delivered successfully!")
            return True
        else:
            print(f"❌ Telegram API returned error: {response.text}", file=sys.stderr)
            return False
    except Exception as exc:
        print(f"❌ Network error trying to call Telegram: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Engine core
# ---------------------------------------------------------------------------
def run_anti_hunt_logic(bypass_gates=False) -> None:
    now_ct = datetime.now(CHICAGO_TZ)
    
    # 1. Structural Environment Verification
    if not bypass_gates:
        if is_cme_holiday(now_ct.date()):
            print(f"[{now_ct.date()}] CME Holiday observed. Aborting gracefully.")
            return
        if not check_time_window():
            print(f"[{now_ct.strftime('%Y-%m-%d %H:%M %Z')}] Outside the safe institutional window (Mon-Fri, 08:45-12:30 America/Chicago). Aborting gracefully.")
            return

    print(f"🧪 RUNNING IN FORCED MANUAL TEST MODE (Bypassing Time, Day, and Staleness Gates)...")
    print(f"⚡ Institutional Core Analysis Active [{now_ct.strftime('%H:%M:%S %Z')}]")

    # 2. Ingest Data Stream
    try:
        intraday_data, daily_data = fetch_market_data()
    except Exception as e:
        print(f"🚫 Critical data error: {e}", file=sys.stderr)
        return

    # 3. Data Integrity Constraints Check
    staleness_min = check_staleness(intraday_data)
    if not bypass_gates and staleness_min > MAX_DATA_AGE_MIN:
        print(f"🚫 Pipeline Stalled: Data age is {staleness_min:.1f} minutes. Maximum allowed is {MAX_DATA_AGE_MIN}m.", file=sys.stderr)
        return
